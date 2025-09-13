"""
文件操作工具
提供安全的文件读写功能
"""

import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from langchain.tools import BaseTool
from pydantic import Field
from loguru import logger

from config import get_config


class SafeFileHandler:
    """安全文件处理器"""
    
    def __init__(self):
        self.config = get_config()
        self.base_dir = Path(self.config.tools.file_base_directory)
        self.max_file_size = self.config.tools.file_max_size
        self.allowed_extensions = self.config.tools.file_allowed_extensions
        
        # 确保基础目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"文件处理器初始化 - 基础目录: {self.base_dir}, 最大文件大小: {self.max_file_size} bytes")
    
    def _validate_path(self, file_path: str) -> Path:
        """
        验证文件路径安全性
        
        Args:
            file_path: 文件路径
            
        Returns:
            验证后的Path对象
            
        Raises:
            ValueError: 路径不安全或不允许
        """
        # 转换为Path对象
        path = Path(file_path)
        
        # 如果是相对路径，相对于基础目录
        if not path.is_absolute():
            path = self.base_dir / path
        
        # 解析真实路径，防止目录遍历攻击
        try:
            resolved_path = path.resolve()
        except Exception as e:
            raise ValueError(f"路径解析失败: {e}")
        
        # 检查是否在允许的基础目录内
        try:
            resolved_path.relative_to(self.base_dir.resolve())
        except ValueError:
            raise ValueError(f"路径不在允许的目录内: {resolved_path}")
        
        return resolved_path
    
    def _validate_extension(self, file_path: Path) -> None:
        """
        验证文件扩展名
        
        Args:
            file_path: 文件路径
            
        Raises:
            ValueError: 扩展名不被允许
        """
        extension = file_path.suffix.lower()
        if extension not in self.allowed_extensions:
            raise ValueError(f"文件扩展名 '{extension}' 不被允许。允许的扩展名: {', '.join(self.allowed_extensions)}")
    
    def _validate_file_size(self, file_path: Path) -> None:
        """
        验证文件大小
        
        Args:
            file_path: 文件路径
            
        Raises:
            ValueError: 文件过大
        """
        if file_path.exists():
            file_size = file_path.stat().st_size
            if file_size > self.max_file_size:
                raise ValueError(f"文件过大: {file_size} bytes，最大允许: {self.max_file_size} bytes")


class FileReadTool(BaseTool):
    """文件读取工具"""
    
    name: str = "read_file"
    description: str = """
    读取文件内容的工具。支持文本文件、JSON文件、CSV文件等。
    
    输入参数: 文件路径（相对于数据目录或绝对路径）
    支持的文件格式: .txt, .md, .json, .csv
    
    示例:
    - data.txt
    - notes/readme.md
    - config.json
    - reports/sales.csv
    """
    
    def __init__(self):
        super().__init__()
        self.handler = SafeFileHandler()
    
    def _run(self, file_path: str) -> str:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件内容字符串
        """
        logger.info(f"读取文件: {file_path}")
        
        try:
            # 验证路径
            path = self.handler._validate_path(file_path)
            self.handler._validate_extension(path)
            
            # 检查文件是否存在
            if not path.exists():
                return f"错误: 文件不存在 - {path}"
            
            if not path.is_file():
                return f"错误: 路径不是文件 - {path}"
            
            # 验证文件大小
            self.handler._validate_file_size(path)
            
            # 根据文件类型读取
            extension = path.suffix.lower()
            
            if extension == '.json':
                return self._read_json_file(path)
            elif extension == '.csv':
                return self._read_csv_file(path)
            else:
                return self._read_text_file(path)
                
        except Exception as e:
            logger.error(f"读取文件失败: {e}")
            return f"读取文件失败: {str(e)}"
    
    def _read_text_file(self, path: Path) -> str:
        """读取文本文件"""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            logger.info(f"成功读取文本文件: {path} ({len(content)} 字符)")
            return f"文件内容 ({path.name}):\\n\\n{content}"
            
        except UnicodeDecodeError:
            # 尝试其他编码
            for encoding in ['gbk', 'latin-1']:
                try:
                    with open(path, 'r', encoding=encoding) as f:
                        content = f.read()
                    logger.info(f"使用 {encoding} 编码读取文件: {path}")
                    return f"文件内容 ({path.name}, 编码: {encoding}):\\n\\n{content}"
                except UnicodeDecodeError:
                    continue
            
            raise ValueError(f"无法解码文件，尝试了多种编码格式")
    
    def _read_json_file(self, path: Path) -> str:
        """读取JSON文件"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 格式化JSON输出
        formatted_json = json.dumps(data, ensure_ascii=False, indent=2)
        logger.info(f"成功读取JSON文件: {path}")
        
        return f"JSON文件内容 ({path.name}):\\n\\n{formatted_json}"
    
    def _read_csv_file(self, path: Path) -> str:
        """读取CSV文件"""
        rows = []
        with open(path, 'r', encoding='utf-8') as f:
            csv_reader = csv.reader(f)
            for i, row in enumerate(csv_reader):
                rows.append(row)
                # 限制读取行数，避免输出过长
                if i >= 50:
                    rows.append([f"... (省略剩余 {sum(1 for _ in csv.reader(open(path, 'r', encoding='utf-8'))) - 51} 行)"])
                    break
        
        # 格式化CSV输出
        formatted_rows = []
        for row in rows:
            formatted_rows.append(" | ".join(str(cell) for cell in row))
        
        logger.info(f"成功读取CSV文件: {path} ({len(rows)} 行)")
        return f"CSV文件内容 ({path.name}):\\n\\n" + "\\n".join(formatted_rows)
    
    async def _arun(self, file_path: str) -> str:
        """异步运行"""
        return self._run(file_path)


class FileWriteTool(BaseTool):
    """文件写入工具"""
    
    name: str = "write_file"
    description: str = """
    写入内容到文件的工具。支持创建新文件或覆盖现有文件。
    
    输入参数格式: file_path|content
    使用 | 分隔文件路径和内容
    
    支持的文件格式: .txt, .md, .json, .csv
    
    示例:
    - notes.txt|这是文件内容
    - data/config.json|{"key": "value"}
    - report.md|# 报告标题\\n\\n内容...
    """
    
    def __init__(self):
        super().__init__()
        self.handler = SafeFileHandler()
    
    def _run(self, input_str: str) -> str:
        """
        写入文件
        
        Args:
            input_str: "文件路径|文件内容" 格式的字符串
            
        Returns:
            操作结果字符串
        """
        logger.info("执行文件写入操作")
        
        try:
            # 解析输入参数
            if '|' not in input_str:
                return "错误: 输入格式错误，请使用 '文件路径|文件内容' 格式"
            
            file_path, content = input_str.split('|', 1)
            file_path = file_path.strip()
            
            # 验证路径
            path = self.handler._validate_path(file_path)
            self.handler._validate_extension(path)
            
            # 创建父目录
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # 检查内容长度
            if len(content.encode('utf-8')) > self.handler.max_file_size:
                return f"错误: 内容过大，最大允许 {self.handler.max_file_size} bytes"
            
            # 根据文件类型写入
            extension = path.suffix.lower()
            
            if extension == '.json':
                return self._write_json_file(path, content)
            else:
                return self._write_text_file(path, content)
                
        except Exception as e:
            logger.error(f"写入文件失败: {e}")
            return f"写入文件失败: {str(e)}"
    
    def _write_text_file(self, path: Path, content: str) -> str:
        """写入文本文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = path.stat().st_size
        logger.info(f"成功写入文本文件: {path} ({file_size} bytes)")
        
        return f"成功写入文件: {path.name} ({file_size} bytes)"
    
    def _write_json_file(self, path: Path, content: str) -> str:
        """写入JSON文件"""
        try:
            # 验证JSON格式
            json_data = json.loads(content)
            
            # 写入格式化的JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            
            file_size = path.stat().st_size
            logger.info(f"成功写入JSON文件: {path} ({file_size} bytes)")
            
            return f"成功写入JSON文件: {path.name} ({file_size} bytes)"
            
        except json.JSONDecodeError as e:
            return f"错误: JSON格式不正确 - {str(e)}"
    
    async def _arun(self, input_str: str) -> str:
        """异步运行"""
        return self._run(input_str)


class FileListTool(BaseTool):
    """文件列表工具"""
    
    name: str = "list_files"
    description: str = """
    列出目录中的文件和子目录。
    
    输入参数: 目录路径（可选，默认为数据目录）
    
    示例:
    - （空输入）：列出数据根目录
    - notes：列出notes子目录
    - reports/2023：列出reports/2023目录
    """
    
    def __init__(self):
        super().__init__()
        self.handler = SafeFileHandler()
    
    def _run(self, directory: str = "") -> str:
        """
        列出目录内容
        
        Args:
            directory: 目录路径，空字符串表示根目录
            
        Returns:
            目录内容字符串
        """
        logger.info(f"列出目录内容: {directory or '根目录'}")
        
        try:
            # 确定目录路径
            if not directory.strip():
                path = self.handler.base_dir
            else:
                path = self.handler._validate_path(directory)
            
            # 检查目录是否存在
            if not path.exists():
                return f"错误: 目录不存在 - {path}"
            
            if not path.is_dir():
                return f"错误: 路径不是目录 - {path}"
            
            # 获取目录内容
            items = []
            total_files = 0
            total_dirs = 0
            
            for item in sorted(path.iterdir()):
                relative_path = item.relative_to(self.handler.base_dir)
                
                if item.is_dir():
                    items.append(f"📁 {relative_path}/")
                    total_dirs += 1
                else:
                    file_size = item.stat().st_size
                    size_str = self._format_file_size(file_size)
                    items.append(f"📄 {relative_path} ({size_str})")
                    total_files += 1
            
            if not items:
                return f"目录为空: {path.relative_to(self.handler.base_dir.parent)}"
            
            header = f"目录内容: {path.relative_to(self.handler.base_dir.parent)} ({total_dirs} 个目录, {total_files} 个文件)"
            content = "\\n".join(items)
            
            logger.info(f"列出目录成功: {total_dirs} 个目录, {total_files} 个文件")
            return f"{header}\\n\\n{content}"
            
        except Exception as e:
            logger.error(f"列出目录失败: {e}")
            return f"列出目录失败: {str(e)}"
    
    def _format_file_size(self, size_bytes: int) -> str:
        """格式化文件大小"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
    
    async def _arun(self, directory: str = "") -> str:
        """异步运行"""
        return self._run(directory)


def create_file_tools() -> List[BaseTool]:
    """创建所有文件操作工具"""
    return [
        FileReadTool(),
        FileWriteTool(),
        FileListTool()
    ]


if __name__ == "__main__":
    # 测试文件操作工具
    tools = create_file_tools()
    read_tool, write_tool, list_tool = tools
    
    print("文件操作工具测试:")
    print("=" * 50)
    
    # 测试写入文件
    print("\\n1. 测试文件写入:")
    result = write_tool._run("test.txt|这是一个测试文件\\n包含多行内容\\n测试完成！")
    print(result)
    
    # 测试JSON写入
    print("\\n2. 测试JSON写入:")
    json_content = '{"name": "测试", "version": "1.0", "features": ["读取", "写入", "列表"]}'
    result = write_tool._run(f"config.json|{json_content}")
    print(result)
    
    # 测试目录列表
    print("\\n3. 测试目录列表:")
    result = list_tool._run("")
    print(result)
    
    # 测试文件读取
    print("\\n4. 测试文件读取:")
    result = read_tool._run("test.txt")
    print(result)
    
    print("\\n5. 测试JSON读取:")
    result = read_tool._run("config.json")
    print(result)
    
    # 测试错误情况
    print("\\n6. 测试错误情况:")
    result = read_tool._run("nonexistent.txt")
    print(result)
    
    print("\\n7. 测试不安全路径:")
    result = read_tool._run("../../../etc/passwd")
    print(result)
    
    print("\\n文件操作工具测试完成！")