# %% Importing Libraries
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings
from document_processor import pdf_loader, split_documents


# %% Loading Document(s)
documents = pdf_loader("./files/us_constitution.pdf")
documents = split_documents(documents, 1000, 100)


# %% Creating Vector Store
vector_store = FAISS.from_documents(
    documents=documents,
    embedding=OllamaEmbeddings(model="llama3.1:latest")
)

vector_store.save_local("database", index_name="us_constitution_1000_100")