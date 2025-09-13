"""
LangChain集成的简单响应Agent
演示如何使用LangChain创建基础的对话Agent
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.llm_factory import LLMFactory, create_llm_with_cost_tracking
from common.model_config import ModelConfig, global_cost_tracker

logger = logging.getLogger(__name__)


class LangChainSimpleAgent:
    """基于LangChain的简单响应Agent"""
    
    def __init__(
        self, 
        provider: str = "openai", 
        model: str = "gpt-3.5-turbo",
        system_prompt: str = None,
        enable_memory: bool = True,
        enable_cost_tracking: bool = True,
        **llm_kwargs
    ):
        """
        初始化Agent
        
        Args:
            provider: LLM提供商
            model: 模型名称
            system_prompt: 系统提示词
            enable_memory: 是否启用记忆功能
            enable_cost_tracking: 是否启用成本追踪
        """
        self.provider = provider
        self.model = model
        self.enable_memory = enable_memory
        self.enable_cost_tracking = enable_cost_tracking
        
        # 默认系统提示词
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # 创建LLM实例
        if enable_cost_tracking:
            self.llm = create_llm_with_cost_tracking(provider, model, **llm_kwargs)
        else:
            self.llm = LLMFactory.create_llm(provider, model, **llm_kwargs)
        
        # 设置记忆
        if enable_memory:
            self.memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history"
            )
        else:
            self.memory = None
        
        # 创建提示模板
        self.prompt_template = self._create_prompt_template()
        
        # 对话历史
        self.conversation_history = []
        
        logger.info(f"LangChain Agent initialized with {provider}:{model}")
    
    def _get_default_system_prompt(self) -> str:
        """获取默认系统提示词"""
        return """你是一个友善、有帮助的AI助手。你能够：

1. 回答各种问题
2. 进行自然对话
3. 提供建议和解决方案
4. 协助完成各种任务

请始终：
- 保持礼貌和专业
- 提供准确、有用的信息
- 在不确定时诚实表达
- 适当使用中文回复中文问题，英文回复英文问题

当前时间：{current_time}
"""
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """创建提示模板"""
        if self.enable_memory:
            return ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])
        else:
            return ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", "{input}")
            ])
    
    def respond(self, user_input: str, **kwargs) -> str:
        """
        响应用户输入
        
        Args:
            user_input: 用户输入
            **kwargs: 额外参数
            
        Returns:
            AI响应
        """
        try:
            # 准备消息
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            if self.enable_memory and self.memory:
                # 使用记忆
                messages = self.prompt_template.format_messages(
                    input=user_input,
                    chat_history=self.memory.chat_memory.messages,
                    current_time=current_time
                )
            else:
                # 不使用记忆
                messages = self.prompt_template.format_messages(
                    input=user_input,
                    current_time=current_time
                )
            
            # 调用LLM
            response = self.llm.invoke(messages)
            response_text = response.content
            
            # 保存到记忆
            if self.enable_memory and self.memory:
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(response_text)
            
            # 保存到对话历史
            self.conversation_history.append({
                "timestamp": datetime.now(),
                "user_input": user_input,
                "ai_response": response_text
            })
            
            logger.info(f"Generated response for input: {user_input[:50]}...")
            return response_text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"抱歉，处理您的请求时出现了错误：{str(e)}"
    
    def batch_respond(self, inputs: List[str]) -> List[str]:
        """批量响应"""
        responses = []
        for user_input in inputs:
            response = self.respond(user_input)
            responses.append(response)
        return responses
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.conversation_history
    
    def clear_memory(self):
        """清除记忆"""
        if self.memory:
            self.memory.clear()
        self.conversation_history = []
        logger.info("Memory and conversation history cleared")
    
    def get_memory_summary(self) -> str:
        """获取记忆摘要"""
        if not self.enable_memory or not self.memory:
            return "记忆功能未启用"
        
        messages = self.memory.chat_memory.messages
        if not messages:
            return "暂无对话记录"
        
        return f"对话轮数：{len(messages)//2}，最近消息：{messages[-1].content[:100]}..."
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        if not self.enable_cost_tracking:
            return {"message": "成本追踪未启用"}
        
        return global_cost_tracker.get_cost_summary()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """获取Agent信息"""
        return {
            "provider": self.provider,
            "model": self.model,
            "system_prompt": self.system_prompt[:100] + "...",
            "memory_enabled": self.enable_memory,
            "cost_tracking_enabled": self.enable_cost_tracking,
            "conversation_turns": len(self.conversation_history),
            "memory_summary": self.get_memory_summary(),
        }
    
    def update_system_prompt(self, new_prompt: str):
        """更新系统提示词"""
        self.system_prompt = new_prompt
        self.prompt_template = self._create_prompt_template()
        logger.info("System prompt updated")
    
    def switch_model(self, provider: str, model: str, **kwargs):
        """切换模型"""
        try:
            if self.enable_cost_tracking:
                self.llm = create_llm_with_cost_tracking(provider, model, **kwargs)
            else:
                self.llm = LLMFactory.create_llm(provider, model, **kwargs)
            
            self.provider = provider
            self.model = model
            
            logger.info(f"Switched to model: {provider}:{model}")
            
        except Exception as e:
            logger.error(f"Failed to switch model: {e}")
            raise


class MultiModelAgent:
    """多模型Agent，可以动态选择最适合的模型"""
    
    def __init__(self):
        self.agents = {}
        self.default_agent = None
        self._initialize_available_agents()
    
    def _initialize_available_agents(self):
        """初始化可用的Agent"""
        available_providers = LLMFactory.get_available_providers()
        
        for provider, available in available_providers.items():
            if available:
                models = ModelConfig.list_available_models(provider)[provider]
                for model in models:
                    try:
                        agent_key = f"{provider}:{model}"
                        self.agents[agent_key] = LangChainSimpleAgent(
                            provider=provider,
                            model=model
                        )
                        if not self.default_agent:
                            self.default_agent = self.agents[agent_key]
                        logger.info(f"Initialized agent: {agent_key}")
                    except Exception as e:
                        logger.warning(f"Failed to initialize {agent_key}: {e}")
    
    def respond(self, user_input: str, preferred_model: str = None) -> Dict[str, Any]:
        """使用指定或最适合的模型响应"""
        if preferred_model and preferred_model in self.agents:
            agent = self.agents[preferred_model]
        else:
            # 简单的模型选择逻辑
            if "代码" in user_input or "code" in user_input.lower():
                agent = self._get_best_agent_for_task("coding")
            elif "分析" in user_input or "analysis" in user_input.lower():
                agent = self._get_best_agent_for_task("analysis")
            else:
                agent = self.default_agent
        
        if not agent:
            return {
                "response": "抱歉，没有可用的模型",
                "model_used": "none",
                "success": False
            }
        
        try:
            response = agent.respond(user_input)
            return {
                "response": response,
                "model_used": f"{agent.provider}:{agent.model}",
                "success": True,
                "cost_summary": agent.get_cost_summary()
            }
        except Exception as e:
            return {
                "response": f"错误：{str(e)}",
                "model_used": f"{agent.provider}:{agent.model}",
                "success": False
            }
    
    def _get_best_agent_for_task(self, task_type: str) -> Optional['LangChainSimpleAgent']:
        """根据任务类型获取最佳Agent"""
        provider, model = ModelConfig.get_recommended_model(task_type)
        agent_key = f"{provider}:{model}"
        return self.agents.get(agent_key, self.default_agent)
    
    def list_available_models(self) -> List[str]:
        """列出可用模型"""
        return list(self.agents.keys())
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取总成本摘要"""
        return global_cost_tracker.get_cost_summary()


# 便捷函数
def create_simple_agent(provider: str = "openai", model: str = "gpt-3.5-turbo") -> LangChainSimpleAgent:
    """快速创建简单Agent"""
    return LangChainSimpleAgent(provider=provider, model=model)


def quick_chat(message: str, provider: str = "openai", model: str = "gpt-3.5-turbo") -> str:
    """快速对话"""
    agent = create_simple_agent(provider, model)
    return agent.respond(message)


if __name__ == "__main__":
    # 简单测试
    try:
        print("🤖 LangChain Agent Test")
        print("=" * 50)
        
        # 创建Agent
        agent = LangChainSimpleAgent()
        print(f"✅ Agent created: {agent.provider}:{agent.model}")
        
        # 获取Agent信息
        info = agent.get_agent_info()
        print(f"📊 Agent Info: {info}")
        
        # 测试对话
        test_messages = [
            "你好！",
            "请介绍一下你自己",
            "今天天气怎么样？"
        ]
        
        for msg in test_messages:
            print(f"\n👤 用户: {msg}")
            response = agent.respond(msg)
            print(f"🤖 Agent: {response}")
        
        # 显示成本信息
        cost_summary = agent.get_cost_summary()
        print(f"\n💰 成本摘要: {cost_summary}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print("请检查是否已安装必要的依赖和配置API密钥")