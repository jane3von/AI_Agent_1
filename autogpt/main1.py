import os
from dotenv import load_dotenv
#load_dotenv("my_api_keys.env")  # 从环境变量中加载 API keys，必须在所有 import 之前
os.environ["CHROMA_TELEMETRY_DISABLED"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # 完全离线模式 !
from AutoAgent.AutoGPT import AutoGPT
from langchain_community.chat_models import ChatTongyi
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from Tools import *
from Tools.PythonTool import ExcelAnalyser
from langchain_community.agent_toolkits.file_management.toolkit import FileManagementToolkit
import pandas as pd
from Tools.Tools import calculator_tool,calendar_tool,document_qa_tool,document_generation_tool,email_tool,excel_inspection_tool
from Utils.DocumentPipeline import DocumentPipeline

def launch_agent(agent: AutoGPT):
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"
    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() in ("quit","exit"):
            break
        reply = agent.run(task, verbose=True)
        print(f"{ai_icon}：{reply}\n")

def main():
    prompts_path = "./prompts"

    # initial 语言模型  使用 ChatTongyi（支持 bind_tools）
    _llm = ChatTongyi(
        model = "qwen-plus",
        temperature = 0.0,
        api_key = os.getenv("DASHSCOPE_API_KEY")
    )

    # 创建管道  # 存储长时记忆的向量数据库
    pipeline = DocumentPipeline(
        data_dir = "./data",           # 文档目录
        persist_dir = "./chroma_db",   # 向量库保存路径
        chunk_size = 512,
        chunk_overlap = 50
    )

    # 一键运行：加载 → 分块 → 存库 只需要运行一次，把文档都存入向量数据库，后面不用运行除非变动
    #pipeline.run()

    # 创建检索器（返回最相似的1个文档）k=5
    retriever = pipeline.as_retriever(5)

    # 自定义工具集: Tools.py
    tools = [
        calculator_tool,
        calendar_tool,
        document_qa_tool,
        document_generation_tool,
        email_tool,
        excel_inspection_tool,
    ]

    # 添加文件管理工具: copy_file, file_delete, file_search, move_file, read_file, write_file, list_directory
    tools += FileManagementToolkit(root_dir=".").get_tools()

    # 添加Excel分析工具: Tools.PythonTool.py ExcelAnalyser
    tools += [ExcelAnalyser(prompts_path = prompts_path).as_tool()]

    # 定义智能体
    agent = AutoGPT(
        llm = _llm,
        prompts_path = prompts_path,
        tools = tools,
        work_dir = "./data",
        main_prompt_file = "main.templ",
        final_prompt_file = "final.templ",
        max_thought_steps = 1,
        memery_retriever = retriever
    )

    # 运行智能体
    launch_agent(agent)


if __name__ == "__main__": 
    main()
