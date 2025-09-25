from typing import List, Optional
from langchain.chat_models.base import BaseChatModel
from langchain.llms import BaseLLM
from langchain.memory import ConversationTokenBufferMemory, VectorStoreRetrieverMemory
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.tools.base import BaseTool
from langchain.chains.llm import LLMChain
from langchain.vectorstores.base import VectorStoreRetriever
from pydantic import ValidationError
from AutoAgent.Action import Action
from Utils.PromptTemplateBuilder import PromptTemplateBuilder
from Utils.PrintUtils import *
from langchain.schema import AIMessage, BaseMessage
import json
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_models import ChatTongyi

# 格式化输出 action.name 和action.args 的字符串
def format_action(action: Action):
    ans = f"{action.name}("
    if action.args is None or len(action.args) == 0:
        ans += ")"
        return ans
    for k,v in action.args.items():
        ans += f"{k}={v},"
    ans = ans[:-1]+")"
    return ans

# 主类
class AutoGPT:
    """AutoGPT：基于Langchain实现"""
    def __init__(
            self,
            llm: BaseLLM | BaseChatModel | ChatTongyi,
            prompts_path: str,
            tools: List[BaseTool],
            work_dir: str = "./data",
            main_prompt_file: str = "main.templ",
            final_prompt_file: str = "final.templ",
            agent_name: Optional[str] = "精灵小助手",
            agent_role: Optional[str] = "强大的AI助手，可以使用工具与指令自动化解决问题",
            max_thought_steps: Optional[int] = 3,
            memery_retriever: Optional[VectorStoreRetriever] = None,
    ):
        self.llm = llm
        self.prompts_path = prompts_path
        self.tools = tools
        self.work_dir = work_dir
        self.agent_name = agent_name
        self.agent_role = agent_role
        self.max_thought_steps = max_thought_steps
        self.memery_retriever = memery_retriever
        self.output_parser = PydanticOutputParser(pydantic_object=Action)
        self.robust_parser = OutputFixingParser.from_llm(parser=self.output_parser, llm=self.llm)
        self.main_prompt_file = main_prompt_file
        self.final_prompt_file = final_prompt_file
        for tool in self.tools:
            color_print(f"初始化 self.tools 存在方法： {tool.name}",BLUE_COLOR)
        
    # 查找 tool
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        color_print(f"### 查找 tool , tool name: {tool_name}  ",BLUE_COLOR)
        for tool in self.tools:
            if tool.name == tool_name:
                color_print(f"### 查找 tool: {tool_name} 找到了！ ",BLUE_COLOR)
                return tool
        return None
    
    # 执行一步思考
    def _step(self,reason_chain,task_description,short_term_memory,long_term_memory,) -> (Action, str):
        """执行一步思考"""
        # reason_chain.run 返回大模型思考结果： response 
        response = reason_chain.run(
            short_term_memory = short_term_memory.load_memory_variables({})["history"],
            long_term_memory = long_term_memory.load_memory_variables({"prompt": task_description})["history"] if long_term_memory is not None else "",
        )
        action = self.robust_parser.parse(response) # response应该是一个json格式的字符串，action 是这个字符串转成json对象:  
        color_print(f"### 执行一步思考 _step 的 action: {format_action(action)} ",BLUE_COLOR)  # action.name = list_directory
        return (action, response)
    
    # 当解析pdf文件后的返回值会有报错，用此方法
    def _safe_output_formatting(observation, prefix="返回结果:\n"):
        """安全的输出格式化函数 参数: 1 observation: 可能是AIMessage、字符串、字典等 2 prefix: 前缀文本   返回: 格式化后的字符串 """
        try:
            # 检查是否是LangChain消息对象
            if isinstance(observation, BaseMessage):
                print(" ### 是LangChain消息对象 ")
                content = observation.content
            # 检查是否有content属性
            elif hasattr(observation, 'content'):
                content = observation.content
            # 检查是否是字典且包含content
            elif isinstance(observation, dict) and 'content' in observation:
                content = observation['content']
            # 检查是否是字典且包含output
            elif isinstance(observation, dict) and 'output' in observation:
                content = observation['output']
            # 检查是否是JSON字符串
            elif isinstance(observation, str):
                try:
                    parsed = json.loads(observation)
                    if isinstance(parsed, dict) and 'content' in parsed:
                        content = parsed['content']  
                    else:
                        content = observation
                except json.JSONDecodeError:
                    content = observation
            else:
                content = str(observation)
            return f"{prefix}{content}"
        except Exception as e:
            return f"{prefix}格式化输出时出错: {str(e)}"

    # 最后一步, 生成最终的输出
    def _final_step(self, short_term_memory, task_description) -> str:
        """ 最后一步, 生成最终的输出 """
        finish_prompt = PromptTemplateBuilder(
            self.prompts_path,
            self.final_prompt_file,
        ).build().partial(
            ai_name = self.agent_name,
            ai_role = self.agent_role,
            task_description = task_description,
            short_term_memory = short_term_memory.load_memory_variables({})["history"],
        )
        chain = LLMChain(llm = self.llm, prompt = finish_prompt)
        response = chain.run({})
        print(f" ### final step: response = {response} ")
        return response

    # 执行 action 工具
    def _exec_action(self, action: Action) -> str:
        color_print(f"### 执行动作  _exec_action() ,action.name={action.name}, action expr={format_action(action)}",BLUE_COLOR)
        # 查找工具
        tool = self._find_tool(action.name)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )
        else:
            try:
                # 执行工具
                color_print(f"执行工具, tool.run 参数是： {action.args}",BLUE_COLOR)
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"
        return observation
    
    # 格式规范化和执行这个方法
    def _format_thought_action(self,response,action_expr):
        color_print(f"_format_thought_action 方法中， response = {response}",BLUE_COLOR)
        thought_start = response.find("关键概念:")
        if thought_start == -1:
            thought_start = 0
        thought_end = response.find("执行:")
        if thought_end == -1:
            thought_end = response.find("{")
        if thought_end == -1:
            thought_end = len(response)
        thought = response[thought_start:thought_end].strip('`')  # 用切片 [thought_start:thought_end] 从响应文本中截取 “思考过程” 部分，.strip('') 去除截取内容前后的反引号（）
        return f"{thought}\n执行:{action_expr}\n"
    
    ### 函数入口方法 task_description 就是用户输入的问题， verbose=False 
    def run(self, task_description, verbose=False) -> str:
        thought_step_count = 0 # 思考步数
        # 初始化模板
        prompt_template = PromptTemplateBuilder(
            self.prompts_path,
            self.main_prompt_file,
        ).build(
            tools = self.tools,
            output_parser = self.output_parser,
        ).partial(
            work_dir = self.work_dir,
            ai_name = self.agent_name,
            ai_role = self.agent_role,
            task_description = task_description,
        )

        # short_term_memory = ConversationTokenBufferMemory(llm = self.llm,ai_prefix = "",human_prefix = "",max_token_limit = 4000)
        short_term_memory = ConversationBufferMemory()

        # 强制初始化一条数据在里面 ！
        short_term_memory.save_context(
            {"input": "\n初始化"},
            {"output": "\n开始"}
        )

        #  初始化LLM链
        chain = LLMChain(
            llm = self.llm,
            prompt = prompt_template,
        )

        # ✅ 关键：使用 create_tool_calling_agent 
        # agent = create_tool_calling_agent(
        #     llm = self.llm,
        #     tools = self.tools,
        #     prompt = prompt_template
        # )

        # chain = AgentExecutor(
        #     agent = agent,
        #     tools = self.tools,
        #     verbose = True,
        #     handle_parsing_errors = True  # 自动处理 JSON 解析错误
        # )

        # 如果有长时记忆，加载长时记忆
        if self.memery_retriever is not None:
            long_term_memory = VectorStoreRetrieverMemory(
                retriever = self.memery_retriever,
            )
        else:
            long_term_memory = None

        reply = ""
        # 循环 self.max_thought_steps 步思考
        while thought_step_count < self.max_thought_steps:
            if verbose:
                color_print(f">>>>Round: {thought_step_count}<<<<",HIGH_NOTICE_COLOR)
            # 执行一步思考
            action, response = self._step(
                chain,
                task_description = task_description,
                short_term_memory = short_term_memory,
                long_term_memory = long_term_memory,
            )
            color_print(f"### action.name = {action.name} ")
            if action.name == "FINISH":
                if verbose:
                    color_print(f"----\nFINISH",OBSERVATION_COLOR) # yellow color
                reply = self._final_step(short_term_memory, task_description)
                break

            action_expr = format_action(action)
            if verbose:
                color_print(self._format_thought_action(response,action_expr), THOUGHT_COLOR) # green color

            observation = self._exec_action(action)
            observation = self._safe_output_formatting(observation)
            color_print(f"---- after _safe_output_formatting \n{observation}")
            
            if verbose:
                color_print(f"----\n结果:\n{observation}",OBSERVATION_COLOR) # yellow color

            # 保存到短时记忆
            short_term_memory.save_context(
                {"input": self._format_thought_action(response,action_expr)},
                {"output": "返回结果:\n"+observation}
            )

            thought_step_count += 1

        if not reply:
            reply = "抱歉，我没能完成您的任务。"

        if long_term_memory is not None:
            # 保存到长时记忆
            long_term_memory.save_context(
                {"input": task_description},
                {"output": reply}
            )
            
        return reply

