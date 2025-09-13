"""
工具管理器
统一管理各种外部工具和API服务
"""

import sys
import asyncio
import inspect
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable, Union, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import json
import time

# Third-party imports
from loguru import logger
import requests
import aiohttp

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class ToolType(Enum):
    """工具类型"""
    API = "api"
    LOCAL = "local"
    SHELL = "shell"
    WEB = "web"
    DATABASE = "database"
    FILE = "file"
    AI = "ai"


class ToolStatus(Enum):
    """工具状态"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class ToolConfig:
    """工具配置"""
    name: str
    type: ToolType
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    config: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 30
    retry_count: int = 3
    rate_limit: Optional[Dict[str, Any]] = None


class BaseTool(ABC):
    """工具基类"""
    
    def __init__(self, config: ToolConfig):
        """初始化工具"""
        self.config = config
        self.status = ToolStatus.LOADING
        self.last_used = None
        self.usage_count = 0
        self.error_count = 0
        
        logger.info(f"初始化工具: {self.config.name}")
    
    @abstractmethod
    async def initialize(self) -> bool:
        """初始化工具"""
        pass
    
    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """执行工具"""
        pass
    
    @abstractmethod
    async def validate(self) -> bool:
        """验证工具可用性"""
        pass
    
    async def cleanup(self):
        """清理资源"""
        logger.info(f"清理工具资源: {self.config.name}")
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "name": self.config.name,
            "type": self.config.type.value,
            "description": self.config.description,
            "version": self.config.version,
            "status": self.status.value,
            "enabled": self.config.enabled,
            "last_used": self.last_used,
            "usage_count": self.usage_count,
            "error_count": self.error_count
        }
    
    def update_usage(self):
        """更新使用统计"""
        self.usage_count += 1
        self.last_used = time.time()
    
    def log_error(self):
        """记录错误"""
        self.error_count += 1
        if self.error_count >= 5:
            self.status = ToolStatus.ERROR
            logger.error(f"工具错误次数过多，已禁用: {self.config.name}")


class APITool(BaseTool):
    """API工具类"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.base_url = config.config.get("base_url")
        self.headers = config.config.get("headers", {})
        self.api_key = config.config.get("api_key")
        self.session = None
        
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
    
    async def initialize(self) -> bool:
        """初始化API工具"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout),
                headers=self.headers
            )
            
            # 测试连接
            if await self.validate():
                self.status = ToolStatus.ACTIVE
                logger.info(f"API工具初始化成功: {self.config.name}")
                return True
            else:
                self.status = ToolStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"API工具初始化失败: {self.config.name}, 错误: {e}")
            self.status = ToolStatus.ERROR
            return False
    
    async def execute(self, method: str = "GET", endpoint: str = "", **kwargs) -> Any:
        """执行API调用"""
        if self.status != ToolStatus.ACTIVE:
            raise RuntimeError(f"工具不可用: {self.config.name}")
        
        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        
        try:
            async with self.session.request(method, url, **kwargs) as response:
                self.update_usage()
                
                if response.status == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if 'application/json' in content_type:
                        return await response.json()
                    else:
                        return await response.text()
                else:
                    error_msg = f"API调用失败: {response.status} - {await response.text()}"
                    logger.error(error_msg)
                    self.log_error()
                    raise RuntimeError(error_msg)
                    
        except Exception as e:
            logger.error(f"API执行失败: {self.config.name}, 错误: {e}")
            self.log_error()
            raise
    
    async def validate(self) -> bool:
        """验证API可用性"""
        try:
            health_endpoint = self.config.config.get("health_endpoint", "")
            if health_endpoint:
                await self.execute("GET", health_endpoint)
            return True
        except:
            return False
    
    async def cleanup(self):
        """清理API会话"""
        if self.session:
            await self.session.close()
        await super().cleanup()


class LocalTool(BaseTool):
    """本地工具类"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.function = None
        self.module_path = config.config.get("module_path")
        self.function_name = config.config.get("function_name")
    
    async def initialize(self) -> bool:
        """初始化本地工具"""
        try:
            if self.module_path and self.function_name:
                # 动态导入模块和函数
                import importlib.util
                spec = importlib.util.spec_from_file_location("tool_module", self.module_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                self.function = getattr(module, self.function_name)
                
            elif "function" in self.config.config:
                self.function = self.config.config["function"]
            
            if self.function:
                self.status = ToolStatus.ACTIVE
                logger.info(f"本地工具初始化成功: {self.config.name}")
                return True
            else:
                self.status = ToolStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"本地工具初始化失败: {self.config.name}, 错误: {e}")
            self.status = ToolStatus.ERROR
            return False
    
    async def execute(self, *args, **kwargs) -> Any:
        """执行本地函数"""
        if self.status != ToolStatus.ACTIVE:
            raise RuntimeError(f"工具不可用: {self.config.name}")
        
        try:
            self.update_usage()
            
            if inspect.iscoroutinefunction(self.function):
                return await self.function(*args, **kwargs)
            else:
                return self.function(*args, **kwargs)
                
        except Exception as e:
            logger.error(f"本地工具执行失败: {self.config.name}, 错误: {e}")
            self.log_error()
            raise
    
    async def validate(self) -> bool:
        """验证本地工具可用性"""
        return self.function is not None and callable(self.function)


class ShellTool(BaseTool):
    """Shell命令工具"""
    
    def __init__(self, config: ToolConfig):
        super().__init__(config)
        self.command_template = config.config.get("command_template", "")
        self.working_dir = config.config.get("working_dir")
        self.env_vars = config.config.get("env_vars", {})
    
    async def initialize(self) -> bool:
        """初始化Shell工具"""
        try:
            # 验证命令模板
            if self.command_template:
                self.status = ToolStatus.ACTIVE
                logger.info(f"Shell工具初始化成功: {self.config.name}")
                return True
            else:
                self.status = ToolStatus.ERROR
                return False
                
        except Exception as e:
            logger.error(f"Shell工具初始化失败: {self.config.name}, 错误: {e}")
            self.status = ToolStatus.ERROR
            return False
    
    async def execute(self, **params) -> Dict[str, Any]:
        """执行Shell命令"""
        if self.status != ToolStatus.ACTIVE:
            raise RuntimeError(f"工具不可用: {self.config.name}")
        
        try:
            # 格式化命令
            command = self.command_template.format(**params)
            
            # 执行命令
            process = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.working_dir
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=self.config.timeout
            )
            
            self.update_usage()
            
            result = {
                "command": command,
                "returncode": process.returncode,
                "stdout": stdout.decode('utf-8', errors='ignore'),
                "stderr": stderr.decode('utf-8', errors='ignore')
            }
            
            if process.returncode != 0:
                logger.warning(f"Shell命令执行警告: {self.config.name}, 返回码: {process.returncode}")
            
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Shell命令执行超时: {self.config.name}")
            self.log_error()
            raise RuntimeError("命令执行超时")
        except Exception as e:
            logger.error(f"Shell工具执行失败: {self.config.name}, 错误: {e}")
            self.log_error()
            raise
    
    async def validate(self) -> bool:
        """验证Shell工具可用性"""
        try:
            # 执行简单的测试命令
            test_result = await self.execute()
            return test_result["returncode"] == 0
        except:
            return False


class ToolManager:
    """工具管理器"""
    
    def __init__(self):
        """初始化工具管理器"""
        self.config = get_config()
        self.tools: Dict[str, BaseTool] = {}
        self.tool_types: Dict[ToolType, Type[BaseTool]] = {
            ToolType.API: APITool,
            ToolType.LOCAL: LocalTool,
            ToolType.SHELL: ShellTool
        }
        
        logger.info("工具管理器初始化完成")
    
    async def register_tool(self, tool_config: ToolConfig) -> bool:
        """注册工具"""
        try:
            if tool_config.name in self.tools:
                logger.warning(f"工具已存在，将被替换: {tool_config.name}")
            
            # 创建工具实例
            tool_class = self.tool_types.get(tool_config.type)
            if not tool_class:
                raise ValueError(f"不支持的工具类型: {tool_config.type}")
            
            tool = tool_class(tool_config)
            
            # 初始化工具
            if await tool.initialize():
                self.tools[tool_config.name] = tool
                logger.info(f"工具注册成功: {tool_config.name}")
                return True
            else:
                logger.error(f"工具初始化失败: {tool_config.name}")
                return False
                
        except Exception as e:
            logger.error(f"工具注册失败: {tool_config.name}, 错误: {e}")
            return False
    
    async def unregister_tool(self, tool_name: str) -> bool:
        """注销工具"""
        try:
            if tool_name in self.tools:
                tool = self.tools[tool_name]
                await tool.cleanup()
                del self.tools[tool_name]
                logger.info(f"工具注销成功: {tool_name}")
                return True
            else:
                logger.warning(f"工具不存在: {tool_name}")
                return False
                
        except Exception as e:
            logger.error(f"工具注销失败: {tool_name}, 错误: {e}")
            return False
    
    async def execute_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """执行工具"""
        if tool_name not in self.tools:
            raise ValueError(f"工具不存在: {tool_name}")
        
        tool = self.tools[tool_name]
        
        if not tool.config.enabled:
            raise RuntimeError(f"工具已禁用: {tool_name}")
        
        if tool.status != ToolStatus.ACTIVE:
            raise RuntimeError(f"工具不可用: {tool_name}, 状态: {tool.status.value}")
        
        return await tool.execute(*args, **kwargs)
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具实例"""
        return self.tools.get(tool_name)
    
    def get_all_tools(self) -> Dict[str, Dict[str, Any]]:
        """获取所有工具信息"""
        return {name: tool.get_info() for name, tool in self.tools.items()}
    
    def get_available_tools(self) -> List[str]:
        """获取可用工具列表"""
        return [
            name for name, tool in self.tools.items()
            if tool.config.enabled and tool.status == ToolStatus.ACTIVE
        ]
    
    async def health_check(self) -> Dict[str, Dict[str, Any]]:
        """健康检查"""
        results = {}
        
        for name, tool in self.tools.items():
            try:
                is_healthy = await tool.validate()
                results[name] = {
                    "healthy": is_healthy,
                    "status": tool.status.value,
                    "last_checked": time.time()
                }
                
                if not is_healthy and tool.status == ToolStatus.ACTIVE:
                    tool.status = ToolStatus.ERROR
                    logger.warning(f"工具健康检查失败: {name}")
                    
            except Exception as e:
                results[name] = {
                    "healthy": False,
                    "status": "error",
                    "error": str(e),
                    "last_checked": time.time()
                }
                logger.error(f"工具健康检查异常: {name}, 错误: {e}")
        
        return results
    
    async def reload_tool(self, tool_name: str) -> bool:
        """重新加载工具"""
        if tool_name in self.tools:
            tool = self.tools[tool_name]
            config = tool.config
            
            # 注销现有工具
            await self.unregister_tool(tool_name)
            
            # 重新注册
            return await self.register_tool(config)
        else:
            logger.warning(f"工具不存在，无法重新加载: {tool_name}")
            return False
    
    async def batch_execute(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量执行工具"""
        results = []
        
        tasks = []
        for i, call in enumerate(tool_calls):
            tool_name = call.get("tool_name")
            args = call.get("args", [])
            kwargs = call.get("kwargs", {})
            
            task = asyncio.create_task(
                self._safe_execute(i, tool_name, *args, **kwargs)
            )
            tasks.append(task)
        
        # 等待所有任务完成
        completed_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return completed_results
    
    async def _safe_execute(self, index: int, tool_name: str, *args, **kwargs) -> Dict[str, Any]:
        """安全执行工具（用于批量执行）"""
        try:
            result = await self.execute_tool(tool_name, *args, **kwargs)
            return {
                "index": index,
                "tool_name": tool_name,
                "success": True,
                "result": result
            }
        except Exception as e:
            return {
                "index": index,
                "tool_name": tool_name,
                "success": False,
                "error": str(e)
            }
    
    async def cleanup_all(self):
        """清理所有工具"""
        logger.info("开始清理所有工具...")
        
        for name, tool in self.tools.items():
            try:
                await tool.cleanup()
                logger.info(f"工具清理完成: {name}")
            except Exception as e:
                logger.error(f"工具清理失败: {name}, 错误: {e}")
        
        self.tools.clear()
        logger.info("所有工具清理完成")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_tools = len(self.tools)
        active_tools = len([t for t in self.tools.values() if t.status == ToolStatus.ACTIVE])
        total_usage = sum(tool.usage_count for tool in self.tools.values())
        total_errors = sum(tool.error_count for tool in self.tools.values())
        
        tool_types_count = {}
        for tool in self.tools.values():
            tool_type = tool.config.type.value
            tool_types_count[tool_type] = tool_types_count.get(tool_type, 0) + 1
        
        return {
            "total_tools": total_tools,
            "active_tools": active_tools,
            "total_usage": total_usage,
            "total_errors": total_errors,
            "tool_types": tool_types_count,
            "tool_details": {
                name: {
                    "usage_count": tool.usage_count,
                    "error_count": tool.error_count,
                    "status": tool.status.value
                }
                for name, tool in self.tools.items()
            }
        }


# 预定义的常用工具配置
PREDEFINED_TOOLS = {
    "web_search": ToolConfig(
        name="web_search",
        type=ToolType.API,
        description="网页搜索工具",
        config={
            "base_url": "https://api.example.com/search",
            "api_key": "your_api_key_here"
        }
    ),
    
    "calculator": ToolConfig(
        name="calculator",
        type=ToolType.LOCAL,
        description="本地计算器工具",
        config={
            "function": lambda expr: eval(expr)  # 注意：实际使用中需要安全的表达式求值
        }
    ),
    
    "file_processor": ToolConfig(
        name="file_processor",
        type=ToolType.SHELL,
        description="文件处理工具",
        config={
            "command_template": "wc -l {file_path}",
            "working_dir": "/tmp"
        }
    )
}


if __name__ == "__main__":
    async def test_tool_manager():
        """测试工具管理器"""
        manager = ToolManager()
        
        # 注册测试工具
        calc_config = ToolConfig(
            name="test_calc",
            type=ToolType.LOCAL,
            description="测试计算器",
            config={
                "function": lambda a, b: a + b
            }
        )
        
        await manager.register_tool(calc_config)
        
        # 测试执行
        try:
            result = await manager.execute_tool("test_calc", 3, 5)
            print(f"计算结果: {result}")
        except Exception as e:
            print(f"执行失败: {e}")
        
        # 获取工具信息
        tools_info = manager.get_all_tools()
        print(f"工具信息: {tools_info}")
        
        # 清理
        await manager.cleanup_all()
    
    # 运行测试
    asyncio.run(test_tool_manager())