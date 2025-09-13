"""
模型配置管理系统
支持多种LLM提供商和模型的统一配置
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelParams:
    """模型参数配置"""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


class ModelConfig:
    """模型配置管理器"""
    
    # OpenAI Models
    OPENAI_MODELS = {
        "gpt-4": ModelParams(max_tokens=8192, temperature=0.7),
        "gpt-4-turbo": ModelParams(max_tokens=128000, temperature=0.7),
        "gpt-3.5-turbo": ModelParams(max_tokens=4096, temperature=0.7),
        "gpt-3.5-turbo-16k": ModelParams(max_tokens=16384, temperature=0.7),
    }
    
    # Anthropic Models
    ANTHROPIC_MODELS = {
        "claude-3-opus": ModelParams(max_tokens=4096, temperature=0.7),
        "claude-3-sonnet": ModelParams(max_tokens=4096, temperature=0.7),
        "claude-3-haiku": ModelParams(max_tokens=4096, temperature=0.7),
        "claude-2.1": ModelParams(max_tokens=4096, temperature=0.7),
    }
    
    # Google Models
    GOOGLE_MODELS = {
        "gemini-pro": ModelParams(max_tokens=4096, temperature=0.7),
        "gemini-pro-vision": ModelParams(max_tokens=4096, temperature=0.7),
    }
    
    # Local Models (Ollama)
    LOCAL_MODELS = {
        "llama2": ModelParams(max_tokens=4096, temperature=0.7),
        "codellama": ModelParams(max_tokens=4096, temperature=0.7),
        "mistral": ModelParams(max_tokens=4096, temperature=0.7),
        "phi": ModelParams(max_tokens=2048, temperature=0.7),
    }
    
    # 成本信息 (USD per 1K tokens)
    MODEL_COSTS = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        "gemini-pro": {"input": 0.0005, "output": 0.0015},
    }
    
    @classmethod
    def get_model_params(cls, provider: str, model: str) -> ModelParams:
        """获取模型参数"""
        provider_models = {
            "openai": cls.OPENAI_MODELS,
            "anthropic": cls.ANTHROPIC_MODELS,
            "google": cls.GOOGLE_MODELS,
            "ollama": cls.LOCAL_MODELS,
        }
        
        if provider not in provider_models:
            raise ValueError(f"Unsupported provider: {provider}")
        
        models = provider_models[provider]
        if model not in models:
            # 返回默认参数
            print(f"Warning: Model {model} not in predefined config, using defaults")
            return ModelParams()
        
        return models[model]
    
    @classmethod
    def get_model_cost(cls, model: str) -> Optional[Dict[str, float]]:
        """获取模型成本信息"""
        return cls.MODEL_COSTS.get(model)
    
    @classmethod
    def list_available_models(cls, provider: Optional[str] = None) -> Dict[str, list]:
        """列出可用的模型"""
        if provider:
            provider_models = {
                "openai": list(cls.OPENAI_MODELS.keys()),
                "anthropic": list(cls.ANTHROPIC_MODELS.keys()),
                "google": list(cls.GOOGLE_MODELS.keys()),
                "ollama": list(cls.LOCAL_MODELS.keys()),
            }
            return {provider: provider_models.get(provider, [])}
        
        return {
            "openai": list(cls.OPENAI_MODELS.keys()),
            "anthropic": list(cls.ANTHROPIC_MODELS.keys()),
            "google": list(cls.GOOGLE_MODELS.keys()),
            "ollama": list(cls.LOCAL_MODELS.keys()),
        }
    
    @classmethod
    def validate_api_keys(cls) -> Dict[str, bool]:
        """验证API密钥是否已配置"""
        return {
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY")),
        }
    
    @classmethod
    def get_recommended_model(cls, task_type: str = "general") -> tuple[str, str]:
        """根据任务类型推荐模型"""
        recommendations = {
            "general": ("openai", "gpt-3.5-turbo"),
            "coding": ("openai", "gpt-4"),
            "analysis": ("anthropic", "claude-3-sonnet"),
            "creative": ("openai", "gpt-4"),
            "local": ("ollama", "llama2"),
            "cost-effective": ("openai", "gpt-3.5-turbo"),
            "high-performance": ("openai", "gpt-4-turbo"),
        }
        
        return recommendations.get(task_type, ("openai", "gpt-3.5-turbo"))


class CostTracker:
    """成本追踪器"""
    
    def __init__(self):
        self.usage_log = []
        self.total_cost = 0.0
    
    def log_usage(self, model: str, input_tokens: int, output_tokens: int):
        """记录使用情况"""
        cost_info = ModelConfig.get_model_cost(model)
        if cost_info:
            input_cost = (input_tokens / 1000) * cost_info["input"]
            output_cost = (output_tokens / 1000) * cost_info["output"]
            total_cost = input_cost + output_cost
            
            self.usage_log.append({
                "model": model,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
            })
            
            self.total_cost += total_cost
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """获取成本摘要"""
        if not self.usage_log:
            return {"total_cost": 0, "total_tokens": 0, "requests": 0}
        
        total_input_tokens = sum(log["input_tokens"] for log in self.usage_log)
        total_output_tokens = sum(log["output_tokens"] for log in self.usage_log)
        
        return {
            "total_cost": self.total_cost,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "requests": len(self.usage_log),
            "average_cost_per_request": self.total_cost / len(self.usage_log) if self.usage_log else 0,
        }


# 全局成本追踪器实例
global_cost_tracker = CostTracker()