from typing import List
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_openai import ChatOpenAI
from langchain.embeddings import HuggingFaceEmbeddings
import os

class FileLoadFactory:
    @staticmethod
    def get_loader(filename: str):
        ext = get_file_extension(filename)
        if ext == "pdf":
            return PyPDFLoader(filename)
        elif ext == "docx" or ext == "doc":
            return UnstructuredWordDocumentLoader(filename)
        else:
            raise NotImplementedError(f"File extension {ext} not supported.")

def get_file_extension(filename: str) -> str:
    return filename.split(".")[-1]

def load_docs(filename: str) -> List[Document]:
    file_loader = FileLoadFactory.get_loader(filename)
    pages = file_loader.load_and_split()
    return pages

def ask_docment(filename: str,query: str,) -> str:
    """根据一个PDF文档的内容，回答一个问题"""
    chatLLM = ChatOpenAI(
        api_key = os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",  
        # other params...
    )
    print(f" ### ask_docment {filename} 中 query: {query}")
    response = chatLLM.invoke(query)
    print('\n'+response.model_dump_json()+'\n')
    return response


# if __name__ == "__main__":
#     filename = "../data/供应商资格要求.pdf"
#     query = "审核流程是怎样的？"
#     response = ask_docment(filename, query)
#     print(response)