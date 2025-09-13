# Enhanced Agents with LangChain Integration
"""
This package contains upgraded versions of all agents using LangChain framework.
"""

__version__ = "2.0.0"
__author__ = "Agent Learning Project"
__description__ = "Production-ready agents built with LangChain"

from .common.llm_factory import LLMFactory
from .common.model_config import ModelConfig

__all__ = ["LLMFactory", "ModelConfig"]