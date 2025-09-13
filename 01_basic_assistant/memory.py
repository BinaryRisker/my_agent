"""
内存管理模块
实现对话历史记录和上下文管理
"""

from typing import Dict, List, Any, Optional
from langchain.memory import ConversationBufferMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.schema.language_model import BaseLanguageModel
from loguru import logger

from config import get_config


class AgentMemory:
    """Agent 内存管理类"""
    
    def __init__(self, llm: Optional[BaseLanguageModel] = None):
        """
        初始化内存管理器
        
        Args:
            llm: 语言模型实例，用于生成摘要
        """
        self.config = get_config()
        self.llm = llm
        self._memory = self._create_memory()
        
    def _create_memory(self) -> ConversationBufferMemory:
        """创建内存实例"""
        memory_type = self.config.memory.type.lower()
        
        if memory_type == "buffer":
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages,
                input_key="input",
                output_key="output"
            )
        elif memory_type == "summary" and self.llm:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm,
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages,
                max_token_limit=self.config.memory.max_token_limit,
                input_key="input",
                output_key="output"
            )
        else:
            # 默认使用 buffer 内存
            memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=self.config.memory.return_messages,
                input_key="input",
                output_key="output"
            )
            
        logger.info(f"创建了 {memory_type} 类型的内存管理器")
        return memory
    
    def add_user_message(self, message: str) -> None:
        """
        添加用户消息
        
        Args:
            message: 用户消息内容
        """
        self._memory.chat_memory.add_user_message(message)
        logger.debug(f"添加用户消息: {message[:50]}...")
    
    def add_ai_message(self, message: str) -> None:
        """
        添加AI消息
        
        Args:
            message: AI消息内容
        """
        self._memory.chat_memory.add_ai_message(message)
        logger.debug(f"添加AI消息: {message[:50]}...")
    
    def get_memory_variables(self) -> Dict[str, Any]:
        """获取内存变量"""
        return self._memory.load_memory_variables({})
    
    def get_chat_history(self) -> List[BaseMessage]:
        """获取聊天历史"""
        return self._memory.chat_memory.messages
    
    def get_formatted_history(self, max_messages: Optional[int] = None) -> str:
        """
        获取格式化的聊天历史
        
        Args:
            max_messages: 最大消息数量
            
        Returns:
            格式化的聊天历史字符串
        """
        messages = self.get_chat_history()
        
        if max_messages:
            messages = messages[-max_messages:]
        
        formatted_history = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_history.append(f"用户: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_history.append(f"助手: {message.content}")
        
        return "\\n".join(formatted_history)
    
    def clear_memory(self) -> None:
        """清空内存"""
        self._memory.clear()
        logger.info("内存已清空")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """获取内存统计信息"""
        messages = self.get_chat_history()
        total_messages = len(messages)
        user_messages = sum(1 for msg in messages if isinstance(msg, HumanMessage))
        ai_messages = sum(1 for msg in messages if isinstance(msg, AIMessage))
        
        # 计算总字符数
        total_chars = sum(len(msg.content) for msg in messages)
        
        return {
            "total_messages": total_messages,
            "user_messages": user_messages,
            "ai_messages": ai_messages,
            "total_characters": total_chars,
            "memory_type": self.config.memory.type
        }
    
    def save_conversation(self, file_path: str) -> None:
        """
        保存对话到文件
        
        Args:
            file_path: 保存文件路径
        """
        try:
            history = self.get_formatted_history()
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(history)
            logger.info(f"对话已保存到: {file_path}")
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            raise
    
    def load_conversation(self, file_path: str) -> None:
        """
        从文件加载对话
        
        Args:
            file_path: 文件路径
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 解析对话内容
            lines = content.strip().split("\\n")
            self.clear_memory()
            
            for line in lines:
                if line.startswith("用户: "):
                    self.add_user_message(line[4:])
                elif line.startswith("助手: "):
                    self.add_ai_message(line[4:])
            
            logger.info(f"对话已从 {file_path} 加载")
        except Exception as e:
            logger.error(f"加载对话失败: {e}")
            raise
    
    def get_recent_context(self, max_chars: int = 1000) -> str:
        """
        获取最近的上下文
        
        Args:
            max_chars: 最大字符数
            
        Returns:
            最近的上下文字符串
        """
        messages = self.get_chat_history()
        context = []
        current_chars = 0
        
        # 从最新消息开始向前遍历
        for message in reversed(messages):
            content = message.content
            if current_chars + len(content) > max_chars:
                break
            
            if isinstance(message, HumanMessage):
                context.insert(0, f"用户: {content}")
            elif isinstance(message, AIMessage):
                context.insert(0, f"助手: {content}")
            
            current_chars += len(content)
        
        return "\\n".join(context)
    
    @property
    def memory(self) -> ConversationBufferMemory:
        """获取内存实例"""
        return self._memory


class ConversationManager:
    """对话管理器"""
    
    def __init__(self):
        self.conversations: Dict[str, AgentMemory] = {}
        self.current_conversation_id: Optional[str] = None
    
    def create_conversation(self, conversation_id: str, llm: Optional[BaseLanguageModel] = None) -> AgentMemory:
        """
        创建新对话
        
        Args:
            conversation_id: 对话ID
            llm: 语言模型实例
            
        Returns:
            内存管理器实例
        """
        memory = AgentMemory(llm=llm)
        self.conversations[conversation_id] = memory
        self.current_conversation_id = conversation_id
        
        logger.info(f"创建新对话: {conversation_id}")
        return memory
    
    def get_conversation(self, conversation_id: str) -> Optional[AgentMemory]:
        """获取指定对话"""
        return self.conversations.get(conversation_id)
    
    def get_current_conversation(self) -> Optional[AgentMemory]:
        """获取当前对话"""
        if self.current_conversation_id:
            return self.conversations.get(self.current_conversation_id)
        return None
    
    def switch_conversation(self, conversation_id: str) -> bool:
        """
        切换到指定对话
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            是否切换成功
        """
        if conversation_id in self.conversations:
            self.current_conversation_id = conversation_id
            logger.info(f"切换到对话: {conversation_id}")
            return True
        return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        删除指定对话
        
        Args:
            conversation_id: 对话ID
            
        Returns:
            是否删除成功
        """
        if conversation_id in self.conversations:
            del self.conversations[conversation_id]
            
            if self.current_conversation_id == conversation_id:
                self.current_conversation_id = None
            
            logger.info(f"删除对话: {conversation_id}")
            return True
        return False
    
    def list_conversations(self) -> List[str]:
        """获取所有对话ID列表"""
        return list(self.conversations.keys())


if __name__ == "__main__":
    # 测试内存管理
    memory = AgentMemory()
    
    # 添加测试消息
    memory.add_user_message("你好，我叫张三")
    memory.add_ai_message("你好张三，很高兴认识你！")
    memory.add_user_message("我想了解LangChain")
    memory.add_ai_message("LangChain是一个用于构建AI应用的框架...")
    
    # 获取内存统计
    stats = memory.get_memory_stats()
    print(f"内存统计: {stats}")
    
    # 获取格式化历史
    history = memory.get_formatted_history()
    print(f"对话历史:\\n{history}")
    
    # 测试对话管理器
    manager = ConversationManager()
    conv1 = manager.create_conversation("conversation_1")
    conv1.add_user_message("测试消息1")
    
    conv2 = manager.create_conversation("conversation_2")
    conv2.add_user_message("测试消息2")
    
    print(f"对话列表: {manager.list_conversations()}")
    print("内存管理测试完成！")