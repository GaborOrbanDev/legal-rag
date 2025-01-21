# %%
import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph.state import CompiledStateGraph, StateGraph, START, END
from pprint import pprint
from pydantic import BaseModel, Field

# %% [markdown]
# Applied local language model

# %%
MODEL = "llama3.1:latest"

# %% [markdown]
# Loading database from disk

# %%
vectorstore = FAISS.load_local(
    folder_path="./database", 
    index_name="us_constitution_1000_100", 
    embeddings=OllamaEmbeddings(model=MODEL),
    allow_dangerous_deserialization=True
)

# %% [markdown]
# Creating state schema

# %%
class InputState(BaseModel):
    question: str


class SelfRAGState(InputState):
    question: str
    documents: list[Document] = []
    generation: str = ""
    try_count: int = 0

# %%
class SelfRAG:
    def __init__(self, vectorstore: FAISS, max_try_count: int = 3):
        self.vectorstore = vectorstore
        self.max_try_count = max_try_count
        self.llm = ChatOllama(model=MODEL, temperature=0.6)
        self.grader_llm = ChatOllama(model=MODEL, temperature=0)

        with open("./prompts.yaml", "r") as file:
            self.prompts = yaml.safe_load(file)

    def create_rag(self) -> CompiledStateGraph:
        workflow = StateGraph(SelfRAGState, input=InputState)
        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_and_filter", self.grade_and_filter)
        workflow.add_node("generate", self.generate)
        workflow.add_node("rewrite_question", self.rewrite_question)
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "grade_and_filter")
        workflow.add_conditional_edges(
            "grade_and_filter", 
            self.did_find_relevant_documents, 
            {
                True: "generate",
                False: "rewrite_question"
            }
        )
        workflow.add_conditional_edges(
            "generate",
            self.did_find_answer,
            {
                True: END,
                False: "rewrite_question"
            }
        )
        workflow.add_edge("rewrite_question", "retrieve")
        return workflow.compile()

    def retrieve(self, state: SelfRAGState) -> SelfRAGState:
        documents = self.vectorstore.similarity_search(query=state.question, k=10)
        return {"documents": documents, "try_count": state.try_count + 1}
    
    def grade_and_filter(self, state: SelfRAGState) -> SelfRAGState:
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
    
    def generate(self, state: SelfRAGState) -> SelfRAGState:
        template = HumanMessagePromptTemplate.from_template(self.prompts["generation_prompt"])
        doc_text = [doc.page_content for doc in state.documents]
        message = template.format(**{"question": state.question, "context": doc_text})
        generation = self.llm.invoke([message]).content
        return {"generation": generation}
    
    def rewrite_question(self, state: SelfRAGState) -> SelfRAGState:
        regeneation_chain = (
            ChatPromptTemplate.from_messages([
                ("system", self.prompts["question_rewrite_system_prompt"]),
                ("user", self.prompts["question_rewrite_instruction"]),
            ])
            | self.llm
            | StrOutputParser()
        )
        return {"question": regeneation_chain.invoke({"question": state.question})}
    
    def did_find_relevant_documents(self, state: SelfRAGState) -> bool:
        return bool(state.documents)
    
    def did_find_answer(self, state: SelfRAGState) -> bool:
        class DidNotHalucinate(BaseModel):
            """Grade whether the answer is based on the retrieved documents or not."""
            grade: bool = Field(..., description="Say True if the answer is factual and based on the retrieved documents or False if the answer is not factual and not based on the retrieved documents.")

        class DidResolveQuestion(BaseModel):
            """Grade the answer whether it is an answer to the question or not."""
            grade: bool = Field(..., description="Say True if the answer is an answer to the question or False if the answer is not an answer to the question.")

        factuality_checker_chain = (
            ChatPromptTemplate.from_messages([("user", self.prompts["halucination_detection_prompt"])])
            | self.grader_llm.with_structured_output(DidNotHalucinate, method="json_schema")
        )

        answer_grader_chain = (
            ChatPromptTemplate.from_messages([("user", self.prompts["answer_grader_prompt"])])
            | self.grader_llm.with_structured_output(DidResolveQuestion, method="json_schema")
        )

        doc_text = [doc.page_content for doc in state.documents]
        did_pass_halucination_test: DidNotHalucinate = factuality_checker_chain.invoke({"answer": state.generation, "context": doc_text})
        if did_pass_halucination_test.grade:
            answer_grade: DidResolveQuestion = answer_grader_chain.invoke({"question": state.question, "answer": state.generation})
            if answer_grade.grade:
                return True
            elif state.try_count < self.max_try_count:
                return False
            else:
                raise Exception("Did not find an answer.")
        return False

# %%
agent = SelfRAG(vectorstore).create_rag()