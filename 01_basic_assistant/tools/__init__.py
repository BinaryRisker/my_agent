"""
工具模块
包含所有可用的工具实现
"""

from .calculator import create_calculator_tool, CalculatorTool
from .weather import create_weather_tool, WeatherTool
from .file_ops import create_file_tools, FileReadTool, FileWriteTool, FileListTool

# 导出所有工具创建函数和类
__all__ = [
    # 工具创建函数
    "create_calculator_tool",
    "create_weather_tool", 
    "create_file_tools",
    "create_all_tools",
    
    # 工具类
    "CalculatorTool",
    "WeatherTool",
    "FileReadTool",
    "FileWriteTool", 
    "FileListTool",
]


def create_all_tools():
    """
    创建所有可用工具的实例
    
    Returns:
        List[BaseTool]: 所有工具实例的列表
    """
    tools = []
    
    # 添加计算器工具
    tools.append(create_calculator_tool())
    
    # 添加天气工具
    tools.append(create_weather_tool())
    
    # 添加文件操作工具
    tools.extend(create_file_tools())
    
    return tools


def get_tool_descriptions():
    """
    获取所有工具的描述信息
    
    Returns:
        Dict[str, str]: 工具名称到描述的映射
    """
    tools = create_all_tools()
    return {tool.name: tool.description for tool in tools}


def get_tool_by_name(name: str):
    """
    根据名称获取工具实例
    
    Args:
        name: 工具名称
        
    Returns:
        BaseTool: 工具实例，如果未找到则返回None
    """
    tools = create_all_tools()
    for tool in tools:
        if tool.name == name:
            return tool
    return None