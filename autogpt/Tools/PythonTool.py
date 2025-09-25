import re
import os
from langchain.tools import StructuredTool
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PythonExecUtil import execute_python_code
#from langchain.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatTongyi
from langchain.chains.llm import LLMChain
from .ExcelTool import get_first_n_rows, get_column_names
import datetime

class ExcelAnalyser:
    def __init__(self, prompts_path):
        self.prompt = PromptTemplateBuilder(prompts_path, "excel_analyser.templ").build()

    def analyse(self, query, filename):
        """分析一个结构化文件（例如excel文件）的内容。"""
        columns = get_column_names(filename)
        inspections = get_first_n_rows(filename, 3)
       # 调用云端强大模型，适合复杂逻辑生成
        _llm = ChatTongyi(
                model = "qwen-plus",
                temperature = 0.0, # temperature=0.0：表示输出最确定、最稳定（不随机）
                api_key = os.getenv("DASHSCOPE_API_KEY")                
            )
        # 创建一个 LangChain 的 LLMChain，它是“提示 + 模型”的组合：
        chain = LLMChain(llm = _llm, prompt = self.prompt)

        print(f" ### ExcelAnalyser -> analyse func:  {filename} 中 query: {query} , columns: {columns} , inspections: {inspections}")
        #执行链式调用：
        # 将 query, filename, columns, inspections 填入 self.prompt
        # 发送给 qwen-plus 模型
        # 模型返回一段包含 Python 代码的回答（通常包裹在 python ...  中）
        response = chain.run(
            query = query,
            filename = filename,
            columns = columns,
            inspections = inspections
        )
        code = self._extract_python_code(response)
        if code:
            # 攻击者可通过 prompt 注入恶意代码（如删除文件、窃取数据）
            # langchain自己的内置代码 可以执行python 不安全，不建议生产用。生产要自己实现！
            ans = execute_python_code(code)  
            return ans
        else:
            return "没有找到可执行的Python代码"

    def _remove_marked_lines(self, input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]
        ans = '\n'.join(lines)
        return ans

    def _extract_python_code(self, text: str) -> str:
        # 使用正则表达式找到所有的Python代码块
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)

        # 从re返回结果提取出Python代码文本
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
            python_code = self._remove_marked_lines(python_code)
            print(f"### python_code={python_code}")
            print(f"### python_code end !")
        return python_code

    def as_tool(self):
        return StructuredTool.from_function(
            func = self.analyse,
            name = "AnalyseExcel",
            description = "通过程序脚本分析一个结构化文件（例如excel文件）的内容。输人中必须包含文件的完整路径和具体分析方式和分析依据，阈值常量等。如果输入信息不完整，你可以拒绝回答。",
        )
    
#"../data/2023年8月-9月销售记录.xlsx"
if __name__ == "__main__":
    print(ExcelAnalyser(prompts_path="../prompts").analyse(query="8月销售额",filename="../data/客户问答素材.xlsx"))