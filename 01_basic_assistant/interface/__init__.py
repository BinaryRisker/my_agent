"""
界面模块
提供CLI和Web界面
"""

from .cli import CLIInterface
from .web import WebInterface

__all__ = [
    "CLIInterface",
    "WebInterface",
]