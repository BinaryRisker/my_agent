"""
工具集成框架 - 统一工具接口
提供标准化的工具接口，支持各种外部工具的统一管理和调用
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import uuid
from pathlib import Path

from loguru import logger


class ToolType(Enum):
    """工具类型枚举"""
    API_TOOL = "api_tool"          # API工具
    LOCAL_FUNCTION = "local_function"   # 本地函数
    SHELL_COMMAND = "shell_command"     # Shell命令
    FILE_PROCESSOR = "file_processor"   # 文件处理器
    DATA_ANALYZER = "data_analyzer"     # 数据分析器
    ML_MODEL = "ml_model"              # 机器学习模型
    WEB_SCRAPER = "web_scraper"        # 网页爬虫
    DATABASE_CONNECTOR = "database_connector"  # 数据库连接器


class ToolStatus(Enum):
    """工具状态枚举"""
    INACTIVE = "inactive"      # 未激活
    ACTIVE = "active"         # 激活
    BUSY = "busy"            # 忙碌
    ERROR = "error"          # 错误
    MAINTENANCE = "maintenance"  # 维护中


@dataclass
class ToolMetadata:
    """工具元数据"""
    name: str
    description: str
    tool_type: ToolType
    version: str = "1.0.0"
    author: str = "Unknown"
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    config_schema: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


@dataclass
class ToolInput:
    """工具输入参数"""
    parameters: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None


@dataclass
class ToolOutput:
    """工具输出结果"""
    success: bool
    result: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    execution_time: Optional[float] = None


@dataclass
class ToolExecutionLog:
    """工具执行日志"""
    execution_id: str
    tool_name: str
    start_time: str
    end_time: Optional[str] = None
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: str = "running"
    error_message: Optional[str] = None


class BaseTool(ABC):
    """工具基础抽象类"""
    
    def __init__(self, metadata: ToolMetadata, config: Dict[str, Any] = None):
        """
        初始化工具
        
        Args:
            metadata: 工具元数据
            config: 工具配置
        """
        self.metadata = metadata
        self.config = config or {}
        self.status = ToolStatus.INACTIVE
        self.execution_logs: List[ToolExecutionLog] = []
        self._last_health_check = 0
        self._health_check_interval = 300  # 5分钟
        
        logger.info(f"工具初始化: {self.metadata.name}")
    
    @abstractmethod
    def execute(self, tool_input: ToolInput) -> ToolOutput:
        """
        执行工具
        
        Args:
            tool_input: 工具输入
            
        Returns:
            工具输出结果
        """
        pass
    
    @abstractmethod
    def validate_input(self, tool_input: ToolInput) -> bool:
        """
        验证输入参数
        
        Args:
            tool_input: 工具输入
            
        Returns:
            验证结果
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        健康检查
        
        Returns:
            健康状态
        """
        pass
    
    def activate(self) -> bool:
        """激活工具"""
        try:
            if self.health_check():
                self.status = ToolStatus.ACTIVE
                logger.info(f"工具激活成功: {self.metadata.name}")
                return True
            else:
                self.status = ToolStatus.ERROR
                logger.error(f"工具激活失败: {self.metadata.name}")
                return False
        except Exception as e:
            self.status = ToolStatus.ERROR
            logger.error(f"工具激活异常: {self.metadata.name}, {e}")
            return False
    
    def deactivate(self) -> bool:
        """停用工具"""
        try:
            self.status = ToolStatus.INACTIVE
            logger.info(f"工具已停用: {self.metadata.name}")
            return True
        except Exception as e:
            logger.error(f"工具停用失败: {self.metadata.name}, {e}")
            return False
    
    def get_status(self) -> ToolStatus:
        """获取工具状态"""
        # 定期进行健康检查
        current_time = time.time()
        if (current_time - self._last_health_check) > self._health_check_interval:
            if self.status == ToolStatus.ACTIVE:
                if not self.health_check():
                    self.status = ToolStatus.ERROR
            self._last_health_check = current_time
        
        return self.status
    
    def get_info(self) -> Dict[str, Any]:
        """获取工具信息"""
        return {
            "metadata": {
                "name": self.metadata.name,
                "description": self.metadata.description,
                "tool_type": self.metadata.tool_type.value,
                "version": self.metadata.version,
                "author": self.metadata.author,
                "tags": self.metadata.tags,
                "requirements": self.metadata.requirements,
                "created_at": self.metadata.created_at
            },
            "status": self.status.value,
            "config": self.config,
            "execution_count": len(self.execution_logs),
            "last_health_check": self._last_health_check
        }
    
    def log_execution(self, execution_log: ToolExecutionLog):
        """记录执行日志"""
        self.execution_logs.append(execution_log)
        
        # 保持最近1000条日志
        if len(self.execution_logs) > 1000:
            self.execution_logs = self.execution_logs[-1000:]
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        if not self.execution_logs:
            return {
                "total_executions": 0,
                "success_rate": 0.0,
                "average_execution_time": 0.0,
                "last_execution": None
            }
        
        total_executions = len(self.execution_logs)
        successful_executions = sum(1 for log in self.execution_logs if log.status == "success")
        success_rate = successful_executions / total_executions
        
        # 计算平均执行时间（对于有结束时间的日志）
        execution_times = []
        for log in self.execution_logs:
            if log.end_time and log.start_time:
                try:
                    start = time.mktime(time.strptime(log.start_time, "%Y-%m-%d %H:%M:%S"))
                    end = time.mktime(time.strptime(log.end_time, "%Y-%m-%d %H:%M:%S"))
                    execution_times.append(end - start)
                except:
                    pass
        
        avg_execution_time = sum(execution_times) / len(execution_times) if execution_times else 0.0
        
        return {
            "total_executions": total_executions,
            "success_rate": success_rate,
            "average_execution_time": avg_execution_time,
            "last_execution": self.execution_logs[-1].start_time if self.execution_logs else None
        }


class ToolPlugin(BaseTool):
    """工具插件类"""
    
    def __init__(self, metadata: ToolMetadata, execute_func: Callable, 
                 validate_func: Callable = None, health_check_func: Callable = None,
                 config: Dict[str, Any] = None):
        """
        初始化工具插件
        
        Args:
            metadata: 工具元数据
            execute_func: 执行函数
            validate_func: 验证函数
            health_check_func: 健康检查函数
            config: 工具配置
        """
        super().__init__(metadata, config)
        
        self._execute_func = execute_func
        self._validate_func = validate_func
        self._health_check_func = health_check_func
    
    def execute(self, tool_input: ToolInput) -> ToolOutput:
        """执行工具插件"""
        execution_id = str(uuid.uuid4())
        start_time = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # 创建执行日志
        execution_log = ToolExecutionLog(
            execution_id=execution_id,
            tool_name=self.metadata.name,
            start_time=start_time,
            input_data=tool_input.parameters
        )
        
        try:
            # 验证输入
            if not self.validate_input(tool_input):
                raise ValueError("输入参数验证失败")
            
            # 设置状态为忙碌
            self.status = ToolStatus.BUSY
            
            # 执行工具函数
            start_exec_time = time.time()
            result = self._execute_func(tool_input)
            end_exec_time = time.time()
            
            # 构建输出
            if isinstance(result, ToolOutput):
                output = result
            else:
                output = ToolOutput(
                    success=True,
                    result=result,
                    execution_time=end_exec_time - start_exec_time
                )
            
            # 更新执行日志
            execution_log.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            execution_log.output_data = {"success": output.success, "result_type": type(output.result).__name__}
            execution_log.status = "success"
            
            # 恢复状态
            self.status = ToolStatus.ACTIVE
            
            return output
            
        except Exception as e:
            # 更新执行日志
            execution_log.end_time = time.strftime("%Y-%m-%d %H:%M:%S")
            execution_log.status = "error"
            execution_log.error_message = str(e)
            
            # 恢复状态
            self.status = ToolStatus.ACTIVE
            
            return ToolOutput(
                success=False,
                error=str(e)
            )
        
        finally:
            # 记录执行日志
            self.log_execution(execution_log)
    
    def validate_input(self, tool_input: ToolInput) -> bool:
        """验证输入参数"""
        if self._validate_func:
            return self._validate_func(tool_input)
        return True
    
    def health_check(self) -> bool:
        """健康检查"""
        if self._health_check_func:
            return self._health_check_func()
        return True


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        """初始化工具注册表"""
        self._tools: Dict[str, BaseTool] = {}
        self._tool_groups: Dict[str, List[str]] = {}
        
        logger.info("工具注册表初始化完成")
    
    def register_tool(self, tool: BaseTool, group: str = "default") -> bool:
        """
        注册工具
        
        Args:
            tool: 工具实例
            group: 工具分组
            
        Returns:
            注册结果
        """
        try:
            tool_name = tool.metadata.name
            
            if tool_name in self._tools:
                logger.warning(f"工具已存在，将覆盖: {tool_name}")
            
            self._tools[tool_name] = tool
            
            # 添加到分组
            if group not in self._tool_groups:
                self._tool_groups[group] = []
            if tool_name not in self._tool_groups[group]:
                self._tool_groups[group].append(tool_name)
            
            logger.info(f"工具注册成功: {tool_name} (分组: {group})")
            return True
            
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            return False
    
    def unregister_tool(self, tool_name: str) -> bool:
        """
        注销工具
        
        Args:
            tool_name: 工具名称
            
        Returns:
            注销结果
        """
        try:
            if tool_name not in self._tools:
                logger.warning(f"工具不存在: {tool_name}")
                return False
            
            # 停用工具
            tool = self._tools[tool_name]
            tool.deactivate()
            
            # 从注册表移除
            del self._tools[tool_name]
            
            # 从分组中移除
            for group_tools in self._tool_groups.values():
                if tool_name in group_tools:
                    group_tools.remove(tool_name)
            
            logger.info(f"工具注销成功: {tool_name}")
            return True
            
        except Exception as e:
            logger.error(f"工具注销失败: {tool_name}, {e}")
            return False
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取工具"""
        return self._tools.get(tool_name)
    
    def list_tools(self, group: str = None, tool_type: ToolType = None, 
                  status: ToolStatus = None) -> List[str]:
        """
        列出工具
        
        Args:
            group: 工具分组过滤
            tool_type: 工具类型过滤
            status: 工具状态过滤
            
        Returns:
            工具名称列表
        """
        tools = list(self._tools.keys())
        
        # 按分组过滤
        if group and group in self._tool_groups:
            tools = [tool for tool in tools if tool in self._tool_groups[group]]
        
        # 按类型过滤
        if tool_type:
            tools = [tool for tool in tools 
                    if self._tools[tool].metadata.tool_type == tool_type]
        
        # 按状态过滤
        if status:
            tools = [tool for tool in tools 
                    if self._tools[tool].get_status() == status]
        
        return tools
    
    def get_registry_info(self) -> Dict[str, Any]:
        """获取注册表信息"""
        total_tools = len(self._tools)
        active_tools = len([tool for tool in self._tools.values() 
                          if tool.get_status() == ToolStatus.ACTIVE])
        
        type_counts = {}
        for tool in self._tools.values():
            tool_type = tool.metadata.tool_type.value
            type_counts[tool_type] = type_counts.get(tool_type, 0) + 1
        
        return {
            "total_tools": total_tools,
            "active_tools": active_tools,
            "tool_groups": {group: len(tools) for group, tools in self._tool_groups.items()},
            "type_distribution": type_counts,
            "registry_status": "healthy" if total_tools > 0 else "empty"
        }


# 预定义的工具创建函数
def create_function_tool(name: str, description: str, func: Callable,
                        validate_func: Callable = None, 
                        health_check_func: Callable = None,
                        config: Dict[str, Any] = None,
                        tags: List[str] = None) -> ToolPlugin:
    """
    创建函数工具
    
    Args:
        name: 工具名称
        description: 工具描述
        func: 执行函数
        validate_func: 验证函数
        health_check_func: 健康检查函数
        config: 配置
        tags: 标签
        
    Returns:
        工具插件实例
    """
    metadata = ToolMetadata(
        name=name,
        description=description,
        tool_type=ToolType.LOCAL_FUNCTION,
        tags=tags or []
    )
    
    return ToolPlugin(
        metadata=metadata,
        execute_func=func,
        validate_func=validate_func,
        health_check_func=health_check_func,
        config=config
    )


def create_api_tool(name: str, description: str, api_func: Callable,
                   validate_func: Callable = None,
                   health_check_func: Callable = None,
                   config: Dict[str, Any] = None,
                   tags: List[str] = None) -> ToolPlugin:
    """
    创建API工具
    
    Args:
        name: 工具名称
        description: 工具描述
        api_func: API调用函数
        validate_func: 验证函数
        health_check_func: 健康检查函数
        config: 配置
        tags: 标签
        
    Returns:
        工具插件实例
    """
    metadata = ToolMetadata(
        name=name,
        description=description,
        tool_type=ToolType.API_TOOL,
        tags=tags or []
    )
    
    return ToolPlugin(
        metadata=metadata,
        execute_func=api_func,
        validate_func=validate_func,
        health_check_func=health_check_func,
        config=config
    )


# 全局工具注册表实例
global_tool_registry = ToolRegistry()


if __name__ == "__main__":
    # 测试代码
    def test_function(tool_input: ToolInput) -> str:
        """测试函数"""
        name = tool_input.parameters.get("name", "World")
        return f"Hello, {name}!"
    
    def test_validate(tool_input: ToolInput) -> bool:
        """测试验证函数"""
        return "name" in tool_input.parameters
    
    # 创建测试工具
    test_tool = create_function_tool(
        name="test_greeter",
        description="测试问候工具",
        func=test_function,
        validate_func=test_validate,
        tags=["test", "greeting"]
    )
    
    # 注册工具
    success = global_tool_registry.register_tool(test_tool, "test_group")
    print(f"工具注册结果: {success}")
    
    # 激活工具
    activation_result = test_tool.activate()
    print(f"工具激活结果: {activation_result}")
    
    # 测试执行
    test_input = ToolInput(parameters={"name": "Agent"})
    result = test_tool.execute(test_input)
    print(f"执行结果: {result}")
    
    # 获取注册表信息
    registry_info = global_tool_registry.get_registry_info()
    print(f"注册表信息: {json.dumps(registry_info, indent=2, ensure_ascii=False)}")