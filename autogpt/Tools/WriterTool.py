from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOpenAI
import os


template = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
        HumanMessagePromptTemplate.from_template("{query}"),
    ]
)

#  Did not find openai_api_key
#writer_chain = LLMChain(llm=ChatOpenAI(), prompt=template)

writer_chain = LLMChain(llm=ChatOpenAI(
    model="qwen-plus",
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    openai_api_key=os.getenv("DASHSCOPE_API_KEY"),
    max_tokens=512,
    temperature=0.1
), prompt=template)


if __name__ == "__main__":
    print(writer_chain.run("写一封邮件给张三，内容是：你好，我是李四。"))