"""
配置管理模块
使用 Pydantic 进行配置验证和类型检查
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field
from dotenv import load_dotenv


# 加载环境变量
load_dotenv()


class LLMConfig(BaseSettings):
    """LLM 相关配置"""
    
    provider: str = Field(default="openai", description="LLM 提供商")
    model: str = Field(default="gpt-3.5-turbo", description="模型名称")
    api_key: str = Field(..., description="API 密钥")
    api_base: Optional[str] = Field(default=None, description="API 基础URL")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="温度参数")
    max_tokens: int = Field(default=1000, gt=0, description="最大令牌数")
    timeout: int = Field(default=30, gt=0, description="超时时间(秒)")
    
    class Config:
        env_prefix = "OPENAI_"


class ToolsConfig(BaseSettings):
    """工具相关配置"""
    
    # 天气工具配置
    weather_api_key: Optional[str] = Field(default=None, description="天气API密钥")
    weather_api_url: str = Field(
        default="http://api.openweathermap.org/data/2.5", 
        description="天气API基础URL"
    )
    weather_cache_timeout: int = Field(default=300, description="天气缓存超时(秒)")
    weather_default_units: str = Field(default="metric", description="默认单位")
    
    # 计算器工具配置
    calc_max_expression_length: int = Field(default=100, description="最大表达式长度")
    calc_allowed_functions: List[str] = Field(
        default=["sin", "cos", "tan", "log", "sqrt", "abs", "round"],
        description="允许的数学函数"
    )
    
    # 文件操作工具配置
    file_max_size: int = Field(default=1048576, description="最大文件大小(字节)")
    file_allowed_extensions: List[str] = Field(
        default=[".txt", ".md", ".json", ".csv"],
        description="允许的文件扩展名"
    )
    file_base_directory: str = Field(default="./data", description="文件基础目录")
    
    class Config:
        env_prefix = "TOOL_"


class MemoryConfig(BaseSettings):
    """内存相关配置"""
    
    type: str = Field(default="buffer", description="内存类型")
    max_token_limit: int = Field(default=2000, description="最大令牌限制")
    return_messages: bool = Field(default=True, description="是否返回消息")
    
    class Config:
        env_prefix = "MEMORY_"


class LoggingConfig(BaseSettings):
    """日志相关配置"""
    
    level: str = Field(default="INFO", description="日志级别")
    format: str = Field(
        default="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
        description="日志格式"
    )
    rotation: str = Field(default="10 MB", description="日志轮转大小")
    retention: str = Field(default="7 days", description="日志保留时间")
    file_path: str = Field(default="./logs/agent.log", description="日志文件路径")
    
    class Config:
        env_prefix = "LOG_"


class WebUIConfig(BaseSettings):
    """Web UI 相关配置"""
    
    host: str = Field(default="127.0.0.1", description="主机地址")
    port: int = Field(default=7860, description="端口号")
    theme: str = Field(default="default", description="主题")
    title: str = Field(default="LangChain Agent Assistant", description="页面标题")
    description: str = Field(default="基于 LangChain 的智能助手", description="页面描述")
    max_chat_history: int = Field(default=50, description="最大聊天历史")
    
    class Config:
        env_prefix = "WEB_"


class SecurityConfig(BaseSettings):
    """安全相关配置"""
    
    max_requests_per_minute: int = Field(default=60, description="每分钟最大请求数")
    max_input_length: int = Field(default=1000, description="最大输入长度")
    forbidden_patterns: List[str] = Field(
        default=["eval", "exec", "import os", "subprocess"],
        description="禁止的模式"
    )
    
    class Config:
        env_prefix = "SECURITY_"


class AppConfig(BaseSettings):
    """主应用配置"""
    
    # 子配置
    llm: LLMConfig = LLMConfig()
    tools: ToolsConfig = ToolsConfig()
    memory: MemoryConfig = MemoryConfig()
    logging: LoggingConfig = LoggingConfig()
    web_ui: WebUIConfig = WebUIConfig()
    security: SecurityConfig = SecurityConfig()
    
    # 全局配置
    debug: bool = Field(default=False, description="调试模式")
    environment: str = Field(default="development", description="环境")
    project_root: Path = Field(default_factory=lambda: Path(__file__).parent.parent)
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 确保必要的目录存在
        self._ensure_directories()
    
    def _ensure_directories(self):
        """确保必要的目录存在"""
        directories = [
            Path(self.logging.file_path).parent,
            Path(self.tools.file_base_directory),
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# 创建全局配置实例
config = AppConfig()


def get_config() -> AppConfig:
    """获取配置实例"""
    return config


if __name__ == "__main__":
    # 配置测试
    print("配置加载测试:")
    print(f"LLM模型: {config.llm.model}")
    print(f"调试模式: {config.debug}")
    print(f"日志级别: {config.logging.level}")
    print(f"Web端口: {config.web_ui.port}")
    print("配置加载成功!")