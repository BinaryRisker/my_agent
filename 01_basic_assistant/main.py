"""
基础智能助手主程序
LangChain Agent 学习项目 - 阶段1
"""

import argparse
import sys
import asyncio
from typing import Optional
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

from config import get_config
from memory import AgentMemory
from tools import create_all_tools
from interface.cli import CLIInterface
from interface.web import WebInterface


class BasicAssistant:
    """基础智能助手类"""
    
    def __init__(self):
        """初始化助手"""
        self.config = get_config()
        self._setup_logging()
        self.llm = None
        self.agent = None
        self.memory = None
        self.tools = None
        
        logger.info("基础智能助手初始化开始")
        self._initialize_components()
        logger.info("基础智能助手初始化完成")
    
    def _setup_logging(self):
        """设置日志配置"""
        # 清除默认处理器
        logger.remove()
        
        # 添加控制台输出
        logger.add(
            sys.stderr,
            level=self.config.logging.level,
            format=self.config.logging.format,
            colorize=True
        )
        
        # 添加文件输出
        logger.add(
            self.config.logging.file_path,
            level=self.config.logging.level,
            format=self.config.logging.format,
            rotation=self.config.logging.rotation,
            retention=self.config.logging.retention,
            encoding="utf-8"
        )
        
        logger.info("日志系统初始化完成")
    
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化LLM
            self._initialize_llm()
            
            # 初始化工具
            self._initialize_tools()
            
            # 初始化内存
            self._initialize_memory()
            
            # 初始化Agent
            self._initialize_agent()
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def _initialize_llm(self):
        """初始化语言模型"""
        try:
            # 检查API密钥
            if not self.config.llm.api_key or self.config.llm.api_key == "your_openai_api_key_here":
                logger.warning("未配置有效的OpenAI API密钥，请在.env文件中配置OPENAI_API_KEY")
                raise ValueError("OpenAI API密钥未配置")
            
            # 创建ChatOpenAI实例
            self.llm = ChatOpenAI(
                model_name=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
                openai_api_key=self.config.llm.api_key,
                openai_api_base=self.config.llm.api_base,
                request_timeout=self.config.llm.timeout
            )
            
            logger.info(f"LLM初始化成功: {self.config.llm.model}")
            
        except Exception as e:
            logger.error(f"LLM初始化失败: {e}")
            raise
    
    def _initialize_tools(self):
        """初始化工具"""
        try:
            self.tools = create_all_tools()
            tool_names = [tool.name for tool in self.tools]
            logger.info(f"工具初始化成功: {', '.join(tool_names)}")
            
        except Exception as e:
            logger.error(f"工具初始化失败: {e}")
            raise
    
    def _initialize_memory(self):
        """初始化内存"""
        try:
            self.memory = AgentMemory(llm=self.llm)
            logger.info("内存初始化成功")
            
        except Exception as e:
            logger.error(f"内存初始化失败: {e}")
            raise
    
    def _initialize_agent(self):
        """初始化Agent"""
        try:
            # 创建Agent
            self.agent = initialize_agent(
                tools=self.tools,
                llm=self.llm,
                agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
                memory=self.memory.memory,
                verbose=self.config.debug,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )
            
            # 设置Agent的系统提示
            system_message = self._get_system_message()
            if hasattr(self.agent.agent, 'llm_chain'):
                if hasattr(self.agent.agent.llm_chain, 'prompt'):
                    # 更新提示模板
                    original_template = self.agent.agent.llm_chain.prompt.template
                    new_template = system_message + "\\n\\n" + original_template
                    self.agent.agent.llm_chain.prompt.template = new_template
            
            logger.info("Agent初始化成功")
            
        except Exception as e:
            logger.error(f"Agent初始化失败: {e}")
            raise
    
    def _get_system_message(self) -> str:
        """获取系统消息"""
        return """你是一个友好、有用的AI助手。你的名字是"基础助手"。

你有以下能力：
1. 数学计算 - 使用calculator工具进行各种数学运算
2. 天气查询 - 使用weather工具查询全球城市天气信息  
3. 文件操作 - 使用file相关工具读写文件和管理目录

使用工具时请遵循以下原则：
- 总是优先使用工具来获取准确信息
- 如果用户询问需要实时数据的问题（如天气、计算等），必须使用相应工具
- 在使用工具前，向用户说明你将要做什么
- 使用工具后，用友好的语言解释结果

请用中文与用户对话，保持友好、专业的态度。"""
    
    def chat(self, user_input: str) -> str:
        """
        与用户对话
        
        Args:
            user_input: 用户输入
            
        Returns:
            助手回复
        """
        try:
            logger.info(f"用户输入: {user_input}")
            
            # 安全检查
            if not self._is_safe_input(user_input):
                return "抱歉，我不能处理这个请求。请输入安全、合理的问题。"
            
            # 使用Agent处理输入
            response = self.agent.run(input=user_input)
            
            logger.info(f"助手回复: {response[:100]}...")
            return response
            
        except Exception as e:
            logger.error(f"对话处理失败: {e}")
            return f"抱歉，处理您的请求时出现了问题: {str(e)}"
    
    def _is_safe_input(self, user_input: str) -> bool:
        """
        检查用户输入是否安全
        
        Args:
            user_input: 用户输入
            
        Returns:
            是否安全
        """
        # 检查输入长度
        if len(user_input) > self.config.security.max_input_length:
            return False
        
        # 检查禁止的模式
        user_input_lower = user_input.lower()
        for pattern in self.config.security.forbidden_patterns:
            if pattern in user_input_lower:
                logger.warning(f"检测到禁止的模式: {pattern}")
                return False
        
        return True
    
    def get_memory_info(self) -> dict:
        """获取内存信息"""
        return self.memory.get_memory_stats()
    
    def clear_memory(self):
        """清空对话记忆"""
        self.memory.clear_memory()
        logger.info("对话记忆已清空")
    
    def save_conversation(self, file_path: str):
        """保存对话"""
        self.memory.save_conversation(file_path)
    
    def load_conversation(self, file_path: str):
        """加载对话"""
        self.memory.load_conversation(file_path)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="基础智能助手")
    parser.add_argument(
        "--mode", 
        choices=["cli", "web"], 
        default="cli",
        help="运行模式: cli(命令行) 或 web(网页界面)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=None,
        help="Web模式端口号（默认使用配置文件中的端口）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Web模式主机地址（默认使用配置文件中的地址）"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )
    
    args = parser.parse_args()
    
    try:
        # 创建助手实例
        assistant = BasicAssistant()
        
        if args.debug:
            logger.info("调试模式已启用")
        
        # 根据模式启动不同界面
        if args.mode == "cli":
            logger.info("启动CLI界面")
            cli = CLIInterface(assistant)
            cli.start()
        
        elif args.mode == "web":
            logger.info("启动Web界面")
            web = WebInterface(assistant)
            
            # 设置主机和端口
            host = args.host or assistant.config.web_ui.host
            port = args.port or assistant.config.web_ui.port
            
            web.start(host=host, port=port)
    
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"程序运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()