"""
LLM工厂类
支持多种LLM提供商的统一接口
"""

import os
import logging
from typing import Any, Dict, Optional

from .model_config import ModelConfig, ModelParams, global_cost_tracker

logger = logging.getLogger(__name__)


class LLMFactory:
    """LLM工厂类，用于创建不同提供商的模型实例"""
    
    @staticmethod
    def create_llm(provider: str, model: str, **kwargs) -> Any:
        """
        创建LLM实例
        
        Args:
            provider: 提供商名称 (openai, anthropic, google, ollama)
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            LLM实例
        """
        # 获取模型参数
        params = ModelConfig.get_model_params(provider, model)
        
        # 合并默认参数和用户参数
        llm_kwargs = {
            "temperature": kwargs.get("temperature", params.temperature),
            "max_tokens": kwargs.get("max_tokens", params.max_tokens),
        }
        
        # 添加其他参数
        if hasattr(params, 'top_p'):
            llm_kwargs["top_p"] = kwargs.get("top_p", params.top_p)
        
        try:
            if provider == "openai":
                return LLMFactory._create_openai_llm(model, llm_kwargs)
            elif provider == "anthropic":
                return LLMFactory._create_anthropic_llm(model, llm_kwargs)
            elif provider == "google":
                return LLMFactory._create_google_llm(model, llm_kwargs)
            elif provider == "ollama":
                return LLMFactory._create_ollama_llm(model, llm_kwargs)
            else:
                raise ValueError(f"Unsupported provider: {provider}")
                
        except ImportError as e:
            logger.error(f"Failed to import required packages for {provider}: {e}")
            raise ImportError(
                f"Please install the required packages for {provider}. "
                f"Run: pip install langchain-{provider}"
            )
        except Exception as e:
            logger.error(f"Failed to create {provider} LLM: {e}")
            raise
    
    @staticmethod
    def _create_openai_llm(model: str, kwargs: Dict[str, Any]) -> Any:
        """创建OpenAI LLM"""
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("Please install: pip install langchain-openai")
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
        
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_anthropic_llm(model: str, kwargs: Dict[str, Any]) -> Any:
        """创建Anthropic LLM"""
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError:
            raise ImportError("Please install: pip install langchain-anthropic")
        
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logger.warning("ANTHROPIC_API_KEY not found in environment variables")
        
        return ChatAnthropic(
            model=model,
            anthropic_api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_google_llm(model: str, kwargs: Dict[str, Any]) -> Any:
        """创建Google LLM"""
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise ImportError("Please install: pip install langchain-google-genai")
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY not found in environment variables")
        
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            **kwargs
        )
    
    @staticmethod
    def _create_ollama_llm(model: str, kwargs: Dict[str, Any]) -> Any:
        """创建Ollama本地LLM"""
        try:
            from langchain_community.llms import Ollama
        except ImportError:
            raise ImportError("Please install: pip install langchain-community")
        
        # Ollama通常在本地运行，无需API密钥
        base_url = kwargs.pop("base_url", "http://localhost:11434")
        
        return Ollama(
            model=model,
            base_url=base_url,
            **kwargs
        )
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """获取可用的提供商"""
        providers = {}
        
        # 检查OpenAI
        try:
            import langchain_openai
            providers["openai"] = bool(os.getenv("OPENAI_API_KEY"))
        except ImportError:
            providers["openai"] = False
        
        # 检查Anthropic
        try:
            import langchain_anthropic
            providers["anthropic"] = bool(os.getenv("ANTHROPIC_API_KEY"))
        except ImportError:
            providers["anthropic"] = False
        
        # 检查Google
        try:
            import langchain_google_genai
            providers["google"] = bool(os.getenv("GOOGLE_API_KEY"))
        except ImportError:
            providers["google"] = False
        
        # 检查Ollama
        try:
            import langchain_community.llms
            providers["ollama"] = True  # 假设本地可用
        except ImportError:
            providers["ollama"] = False
        
        return providers
    
    @staticmethod
    def test_connection(provider: str, model: str) -> Dict[str, Any]:
        """测试连接"""
        try:
            llm = LLMFactory.create_llm(provider, model)
            
            # 简单测试
            from langchain.schema import HumanMessage
            response = llm.invoke([HumanMessage(content="Hello")])
            
            return {
                "success": True,
                "provider": provider,
                "model": model,
                "response_preview": response.content[:50] + "..." if len(response.content) > 50 else response.content
            }
            
        except Exception as e:
            return {
                "success": False,
                "provider": provider,
                "model": model,
                "error": str(e)
            }


class LLMWithCostTracking:
    """带有成本追踪功能的LLM包装器"""
    
    def __init__(self, llm: Any, model_name: str):
        self.llm = llm
        self.model_name = model_name
    
    def invoke(self, messages, **kwargs):
        """调用模型并追踪成本"""
        response = self.llm.invoke(messages, **kwargs)
        
        # 尝试获取token使用情况
        if hasattr(response, 'usage_metadata'):
            usage = response.usage_metadata
            input_tokens = getattr(usage, 'input_tokens', 0)
            output_tokens = getattr(usage, 'output_tokens', 0)
            
            # 记录使用情况
            global_cost_tracker.log_usage(
                self.model_name, 
                input_tokens, 
                output_tokens
            )
        
        return response
    
    def __getattr__(self, name):
        """代理到原始LLM对象"""
        return getattr(self.llm, name)


def create_llm_with_cost_tracking(provider: str, model: str, **kwargs) -> LLMWithCostTracking:
    """创建带有成本追踪的LLM"""
    llm = LLMFactory.create_llm(provider, model, **kwargs)
    return LLMWithCostTracking(llm, model)


# 便捷函数
def quick_llm(task_type: str = "general", **kwargs) -> Any:
    """根据任务类型快速创建推荐的LLM"""
    provider, model = ModelConfig.get_recommended_model(task_type)
    return LLMFactory.create_llm(provider, model, **kwargs)


def quick_llm_with_cost_tracking(task_type: str = "general", **kwargs) -> LLMWithCostTracking:
    """根据任务类型快速创建带成本追踪的LLM"""
    provider, model = ModelConfig.get_recommended_model(task_type)
    return create_llm_with_cost_tracking(provider, model, **kwargs)