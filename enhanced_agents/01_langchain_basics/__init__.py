"""
LangChain Basics Module
基础LangChain Agent实现
"""

from .langchain_agent import LangChainSimpleAgent, MultiModelAgent, create_simple_agent, quick_chat

__all__ = [
    "LangChainSimpleAgent",
    "MultiModelAgent", 
    "create_simple_agent",
    "quick_chat"
]