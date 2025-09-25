from langchain_core.prompts import PromptTemplate
from typing import List, Optional
from langchain.tools.base import BaseTool
from langchain.schema.output_parser import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser
from .FileUtils import load_file
import os, json
from AutoAgent.Action import Action



def chinese_friendly(string) -> str:
    print(f"chinese_friendly : input string is: {string} ")
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    ret = '\n'.join(lines)
    print(f"chinese_friendly : return: {ret} ")
    return ret

class PromptTemplateBuilder:
    def __init__(self,prompt_path: str,prompt_file: str,):
        self.prompt_path = prompt_path
        self.prompt_file = prompt_file

    def _get_prompt(self, prompt_file: str) -> str:
        builder = PromptTemplateBuilder(self.prompt_path, prompt_file)
        return builder.build().format()

    def _get_tools_prompt(self, tools: List[BaseTool]) -> str:
        tools_prompt = ""
        for i, tool in enumerate(tools):
            prompt = f"{i+1}. {tool.name}: {tool.description}, \
                        args json schema: {json.dumps(tool.args, ensure_ascii=False)}\n"
            tools_prompt += prompt
        return tools_prompt

    def build(self, tools: Optional[List[BaseTool]] = None, output_parser: Optional[BaseOutputParser] = None,) -> PromptTemplate:
        main_templ_str = load_file(os.path.join(self.prompt_path, self.prompt_file))
        main_templ = PromptTemplate.from_template(main_templ_str)

        # 解析出模板main.templ中所有占位符变量 例如：variables = ['ai_name', 'ai_role', 'constraints_templ', 'format_instructions', 'instructions_templ', 'long_term_memory', 'performance_evaluation_templ', 'resources_templ', 'short_term_memory', 'task_description', 'thought_templ', 'tools', 'work_dir'] 
        variables = main_templ.input_variables 
        print(f" ### variables = {variables} ")
        partial_variables = {} # 用于存储要“预填充”的变量值

        # 模块化模板管理  遍历所有变量，如果变量以_templ结尾，读取对应的文件，将文件内容作为变量值
        for var in variables:
            if var.endswith("_templ"):
                var_file = var[:-6] + ".templ"
                var_str = self._get_prompt(var_file)
                partial_variables[var] = var_str

        if tools is not None:
            tools_prompt = self._get_tools_prompt(tools)
            partial_variables["tools"] = tools_prompt

        # 如果提供了 output_parser，生成格式说明
        if output_parser is not None:
            partial_variables["format_instructions"] = chinese_friendly(
                output_parser.get_format_instructions()
            )

        # 将有值的变量填充到模板中 partial()：返回一个 PartialTemplate，它已经预填充了部分变量（如 tools, format_instructions）
        # 剩下的变量（如 input, chat_history）将在运行时传入
        return main_templ.partial(**partial_variables) 


if __name__ == "__main__":
    builder = PromptTemplateBuilder("../prompts")
    output_parser = PydanticOutputParser(pydantic_object=Action)
    prompt_template = builder.build(tools=[], output_parser=output_parser)
    print(prompt_template.format(
        ai_name="瓜瓜",
        ai_role="智能助手机器人",
        task_description="解决问题",
        short_term_memory="",
        long_term_memory="",
        tools="",
    ))
