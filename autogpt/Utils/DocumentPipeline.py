import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1" 
from pathlib import Path
from typing import List, Dict, Any
from langchain_community.document_loaders import (
    UnstructuredPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from markdownify import markdownify as md2text

class DocumentPipeline:
    """
    通用文档处理管道：支持 PDF, Markdown, Excel, Word ,txt 等格式
    输出：Chroma 向量数据库
    """

    def __init__(
        self,
        data_dir: str,
        persist_dir: str = "./chroma_db",
        #embedding_model: str = "all-MiniLM-L6-v2",
        embedding_model: str = "/Users/pc/Documents/models/sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"数据目录不存在: {data_dir}")

        self.persist_dir = persist_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # 初始化嵌入模型
        self.embedding = HuggingFaceEmbeddings(
            model_name = embedding_model,
            model_kwargs={"device": "cpu"} 
        )

        # 文本分割器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,          # 中文建议 200~500 字符
            chunk_overlap=50,        # 保留上下文衔接
            length_function=len,
            is_separator_regex=False,
        )

        # 向量数据库
        self.vectorstore = None

    def load_markdown_files(self) -> List[Dict[str, Any]]:
        """加载 Markdown 文件 → 转为纯文本"""
        docs = []
        for md_path in self.data_dir.rglob("*.md"):
            try:
                with open(md_path, 'r', encoding='utf-8') as f:
                    html_content = f.read()
                    text = md2text(html_content)  # 转为纯文本
                docs.append({
                    "page_content": text,
                    "metadata": {"source": str(md_path), "type": "markdown"}
                })
            except Exception as e:
                print(f"⚠️ 跳过 Markdown 文件 {md_path}: {e}")
        return docs

    def load_excel_files(self) -> List[Dict[str, Any]]:
        """加载 Excel 文件 → 每个 sheet 转为表格文本"""
        docs = []
        for file_path in self.data_dir.rglob("*.xlsx"):
            try:
                excel_file = pd.ExcelFile(file_path)
                for sheet_name in excel_file.sheet_names:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    # 转为 Markdown 表格格式（可读性强）
                    table_text = df.to_markdown(index=False) if hasattr(df, 'to_markdown') else str(df)
                    docs.append({
                        "page_content": table_text,
                        "metadata": {
                            "source": str(file_path),
                            "sheet_name": sheet_name,
                            "type": "excel"
                        }
                    })
                excel_file.close()
            except Exception as e:
                print(f"⚠️ 跳过 Excel 文件 {file_path}: {e}")
        return docs

    def load_csv_files(self) -> List[Dict[str, Any]]:
        """加载 CSV 文件"""
        docs = []
        for csv_path in self.data_dir.rglob("*.csv"):
            try:
                loader = CSVLoader(str(csv_path))
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["type"] = "csv"
                docs.extend(raw_docs)
            except Exception as e:
                print(f"⚠️ 跳过 CSV 文件 {csv_path}: {e}")
        return docs

    def load_pdf_files(self) -> List[Dict[str, Any]]:
        """使用 pdfplumber 提取 PDF 文本"""
        from pdfplumber import open as pdf_open
        documents = []
        #pdf_dir = Path(pdf_path)
        for pdf_path in self.data_dir.glob("**/*.pdf"):
            print(f"pdf_path: {pdf_path}")
            text = ""
            with pdf_open(pdf_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
            # 模拟 LangChain Document
            documents.append({
                "page_content": text,
                "metadata": {"source": str(pdf_path), "type": "pdf"}
            })
        return documents

    def load_pdf_files1(self) -> List[Dict[str, Any]]:
        """加载 PDF 文件"""
        docs = []
        for pdf_path in self.data_dir.rglob("*.pdf"):
            try:
                loader = UnstructuredPDFLoader(str(pdf_path))
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["type"] = "pdf"
                    doc.metadata["source"] = str(pdf_path)
                docs.extend(raw_docs)
            except Exception as e:
                print(f"⚠️ 跳过 PDF 文件 {pdf_path}: {e}")
        return docs

    def load_word_files(self) -> List[Dict[str, Any]]:
        """加载 Word 文件 (.docx)"""
        docs = []
        for docx_path in self.data_dir.rglob("*.docx"):
            try:
                loader = UnstructuredWordDocumentLoader(str(docx_path)) # 下载的包与numpy冲突
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["type"] = "word"
                    doc.metadata["source"] = str(docx_path)
                docs.extend(raw_docs)
            except Exception as e:
                print(f"⚠️ 跳过 Word 文件 {docx_path}: {e}")
        return docs

    def load_text_files(self) -> List[Dict[str, Any]]:
        """加载纯文本文件 (.txt)"""
        docs = []
        for txt_path in self.data_dir.rglob("*.txt"):
            try:
                loader = TextLoader(str(txt_path), encoding='utf-8')
                raw_docs = loader.load()
                for doc in raw_docs:
                    doc.metadata["type"] = "text"
                docs.extend(raw_docs)
            except Exception as e:
                print(f"⚠️ 跳过文本文件 {txt_path}: {e}")
        return docs

    def load_all_documents(self) -> List[Dict[str, Any]]:
        """统一加载所有支持的文档类型"""
        print("🔍 开始加载文档...")
        all_docs = []

        # 按类型加载
        all_docs.extend(self.load_pdf_files())
        all_docs.extend(self.load_word_files())
        all_docs.extend(self.load_markdown_files())
        all_docs.extend(self.load_excel_files())
        all_docs.extend(self.load_csv_files())
        all_docs.extend(self.load_text_files())

        print(f"✅ 共加载 {len(all_docs)} 个文档片段")
        return all_docs

    def split_documents(self, raw_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分块处理"""
        print("✂️ 正在分块...")
        split_docs = []
        for item in raw_docs:
            content = item["page_content"]
            metadata = item["metadata"]

            # 分块
            chunks = self.text_splitter.split_text(content)
            for chunk in chunks:
                split_docs.append({
                    "page_content": chunk,
                    "metadata": metadata
                })

        print(f"✅ 分块完成，共 {len(split_docs)} 个文本块")
        return split_docs

    def build_vectorstore(self, documents: List[Dict[str, Any]]):
        """构建 Chroma 向量数据库"""
        print("📦 正在构建向量数据库...")

        texts = [doc["page_content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        # use langchain collection name as default !
        self.vectorstore = Chroma.from_texts(
            texts = texts,
            embedding = self.embedding,
            metadatas = metadatas,
            persist_directory = self.persist_dir
        )
        self.vectorstore.persist()
        print(f"✅ 向量数据库已保存至: {self.persist_dir}")

    def run(self):
        """一键运行整个管道"""
        raw_docs = self.load_all_documents()
        if not raw_docs:
            print("❌ 未找到任何文档")
            return

        split_docs = self.split_documents(raw_docs)
        self.build_vectorstore(split_docs)
        print("🎉 文档处理管道执行完成！")

    def as_retriever(self,k:int):
        """返回 retriever，用于后续问答"""
        if not self.vectorstore:
            self.vectorstore = Chroma(
                persist_directory = self.persist_dir,
                embedding_function = self.embedding
            )
        return self.vectorstore.as_retriever(search_kwargs={"k": k})