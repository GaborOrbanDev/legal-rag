"""The Self-RAG workflow is meant to be used with Langgraph CLI."""

import yaml
from typing import Literal
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
from pydantic import BaseModel, Field


MODEL = "llama3.1:latest"
EMBEDDING_MODEL = "nomic-embed-text:latest"


vectorstore = FAISS.load_local(
    folder_path="./database", 
    index_name="us_constitution_500_0_net_spec_sep", 
    embeddings=OllamaEmbeddings(model=EMBEDDING_MODEL),
    allow_dangerous_deserialization=True
)


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    documents: list[Document] = []
    generation: str = ""


class SelfRAGState(InputState, OutputState):
    retrieve_count: int = 0
    generation_count: int = 0


class SelfRAG:
    """Self-RAG workflow for generating answers based on retrieved and self-checked documents."""

    def __init__(self, vectorstore: FAISS, max_try_count: int = 3):
        self.vectorstore = vectorstore
        self.max_try_count = max_try_count
        self.llm = ChatOllama(model=MODEL, temperature=0.6)
        self.grader_llm = ChatOllama(model=MODEL, temperature=0)

        with open("./prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

    def create_rag(self) -> CompiledStateGraph:
        """Creates Self-RAG workflow"""

        # Create the workflow
        workflow = StateGraph(SelfRAGState, input=InputState, output=OutputState)

        # Adding nodes
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_and_filter", self.grade_and_filter)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_node("generate", self.generate)
        workflow.add_node("no_answer", self.no_answer)

        # Adding edges
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_and_filter")
        workflow.add_conditional_edges(
            "grade_and_filter", 
            self.did_find_relevant_documents, 
            {
                "generate": "generate",
                "rewrite": "rewrite_question",
                "no_answer": "no_answer"
            }
        )
        workflow.add_conditional_edges(
            "generate",
            self.did_find_answer,
            {
                "end": END,
                "generate": "generate",
                "no_answer": "no_answer"
            }
        )
        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("no_answer", END)

        return workflow.compile()
    
    # --------------------------------------------------------------------------------------------------------
    # region: Node functions

    def retrieve(self, state: SelfRAGState) -> SelfRAGState:
        """Retrieves semantically similar documents from the database."""

        documents = self.vectorstore.similarity_search(query=state.question, k=5)
        return {"documents": documents, "retrieve_count": state.retrieve_count + 1}
    
    def grade_and_filter(self, state: SelfRAGState) -> SelfRAGState:
        """Grades the retrieved documents and filters out the irrelevant ones."""

        class DocumentGrade(BaseModel):
            """Grade the document based on its relevance to the question."""
            grade: int = Field(..., description="Give a score between 0 and 10 to the document based on its relevance to the question.")

        grading_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.prompts["grader_system_prompt"]),
                ("user", self.prompts["grader_instruction"]),
            ])
            | self.grader_llm.with_structured_output(DocumentGrade, method="json_schema")
        )        

        response: list[DocumentGrade] = grading_chain.batch([
            {
                "question": state.question,
                "document": document.page_content
            }
            for document in state.documents
        ])

        filtered_documents = []
        for i, grade in enumerate(response):
            if grade.grade > 5:
                filtered_documents.append(state.documents[i])
                
        return {"documents": filtered_documents}
    
    def rewrite_question(self, state: SelfRAGState) -> SelfRAGState:
        """Rewrites original question to get a better answer."""

        regeneation_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.prompts["question_rewrite_system_prompt"]),
                ("user", self.prompts["question_rewrite_instruction"]),
            ])
            | self.llm
            | StrOutputParser()
        )
        return {"question": regeneation_chain.invoke({"question": state.question})}
    
    def generate(self, state: SelfRAGState) -> SelfRAGState:
        """Generates an answer based on the retrieved documents."""

        template = HumanMessagePromptTemplate.from_template(self.prompts["generation_prompt"])
        doc_text = [doc.page_content for doc in state.documents]
        message = template.format(**{"question": state.question, "context": doc_text})
        generation = self.llm.invoke([message]).content
        return {"generation": generation, "generation_count": state.generation_count + 1}
    
    def no_answer(self, state: SelfRAGState) -> SelfRAGState:
        """Returns no answer if no relevant documents were found or the generation limit was reached."""

        return {"generation": "No relevant or factual answer was found.", "documents": []}
    
    # endregion
    # --------------------------------------------------------------------------------------------------------
    # region: Conditional edge functions
     
    def did_find_relevant_documents(self, state: SelfRAGState) -> Literal["generate", "rewrite", "no_answer"]:
        """Routes the workflow to generate an answer if found at least one relevant documents, else rewrite the question or return no answer."""

        if state.retrieve_count >= self.max_try_count and len(state.documents) == 0:
            return "no_answer"
        elif len(state.documents) > 0:
            return "generate"
        else:
            return "rewrite"
    
    def did_find_answer(self, state: SelfRAGState) -> Literal["end", "generate", "no_answer"]:
        """Checks if the generated answer is factual or not and routes the workflow accordingly."""

        class DidNotHalucinate(BaseModel):
            """Grade whether the answer is based on the retrieved documents or not."""
            grade: bool = Field(..., description="Say True if the answer is factual and based on the retrieved documents or False if the answer is not factual and not based on the retrieved documents.")

        factuality_checker_chain = (
            ChatPromptTemplate.from_messages([("user", self.prompts["halucination_detection_prompt"])])
            | self.grader_llm.with_structured_output(DidNotHalucinate, method="json_schema")
        )

        doc_text = [doc.page_content for doc in state.documents]
        did_pass_halucination_test: DidNotHalucinate = factuality_checker_chain.invoke({"answer": state.generation, "context": doc_text})
        if did_pass_halucination_test.grade:
            return "end"
        elif state.generation_count >= self.max_try_count:
            return "no_answer"
        else:
            return "generate"
        
    # endregion
    # --------------------------------------------------------------------------------------------------------
        

agent = SelfRAG(vectorstore).create_rag()