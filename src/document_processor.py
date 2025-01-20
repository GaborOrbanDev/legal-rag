# %% Importing necessary libraries
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_core.documents import Document


# %% Function to download and process pages

def internet_loader(urls: str | list[str]) -> list[Document]:
    """Load documents from the given URLs.

    Args:
        urls (str | list[str]): URL or list of URLs to download and clear

    Returns:
        list[Document]: loaded documents
    """
    
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for doc in docs:
        doc.page_content = doc.page_content.strip()
        doc.page_content = re.sub(r'\n+', '\n', doc.page_content)
        doc.page_content = re.sub(r'\s+', ' ', doc.page_content)
    return docs


def pdf_loader(file_paths: str | list[str]) -> list[Document]:
    """Load documents from the given PDF files.

    Args:
        file_paths (str | list[str]): file path or list of file paths to load

    Returns:
        list[Document]: list of loaded documents
    """
    docs = []

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    for file_path in file_paths:
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
    return docs


def split_documents(documents: list[Document], chunk_size: int, overlap: int) -> list[Document]:
    """Splits the documents into chunks of the given size and overlap.

    Args:
        documents (list[Document]): list of documents to split (unit: characters)
        chunk_size (int): Size of the expected chunks (unit: characters)
        overlap (int): Size of the overlap between the chunks

    Returns:
        list[Document]: list of the split documents
    """
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=overlap, 
        length_function=len
    )
    return splitter.split_documents(documents)