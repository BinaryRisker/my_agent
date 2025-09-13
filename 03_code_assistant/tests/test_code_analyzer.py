"""
代码分析器测试用例
"""

import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目路径
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.code_analyzer import CodeAnalyzer


class TestCodeAnalyzer:
    """代码分析器测试类"""
    
    def setup_method(self):
        """测试前的准备工作"""
        self.analyzer = CodeAnalyzer()
        
        # 创建临时测试文件
        self.test_code = '''
import os
import sys
from typing import List, Dict

class Calculator:
    """简单计算器类"""
    
    def __init__(self):
        self.history = []
    
    def add(self, a: int, b: int) -> int:
        """加法运算"""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def divide(self, a: int, b: int) -> float:
        """除法运算"""
        if b == 0:
            raise ValueError("除数不能为零")
        result = a / b
        self.history.append(f"{a} / {b} = {result}")
        return result
    
    def get_history(self) -> List[str]:
        """获取计算历史"""
        return self.history.copy()

def factorial(n: int) -> int:
    """计算阶乘"""
    if n < 0:
        raise ValueError("n必须是非负整数")
    elif n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

def fibonacci(n: int) -> int:
    """计算斐波那契数列第n项"""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
'''
        
        # 创建临时文件
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        self.temp_file.write(self.test_code)
        self.temp_file.close()
        
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 在临时目录中创建多个测试文件
        self.test_files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f'test_file_{i}.py')
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f'# Test file {i}\ndef func_{i}():\n    return {i}\n')
            self.test_files.append(file_path)
    
    def teardown_method(self):
        """测试后的清理工作"""
        # 清理临时文件
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
        
        # 清理临时目录和文件
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_init(self):
        """测试初始化"""
        assert self.analyzer is not None
        assert hasattr(self.analyzer, 'supported_languages')
        assert hasattr(self.analyzer, 'config')
    
    def test_analyze_file(self):
        """测试单个文件分析"""
        result = self.analyzer.analyze_file(self.temp_file.name)
        
        # 检查基本属性
        assert 'file_path' in result
        assert 'language' in result
        assert 'line_count' in result
        assert 'file_size' in result
        assert 'analysis_type' in result
        
        # 检查分析结果
        assert result['language'] == 'python'
        assert result['analysis_type'] == 'file'
        assert result['line_count'] > 0
        
        # 检查Python特定分析结果
        assert 'functions' in result
        assert 'classes' in result
        assert 'imports' in result
        assert 'complexity' in result
        
        # 验证函数分析结果
        functions = result['functions']
        assert len(functions) >= 2  # factorial 和 fibonacci
        
        function_names = [f['name'] for f in functions]
        assert 'factorial' in function_names
        assert 'fibonacci' in function_names
        
        # 验证类分析结果
        classes = result['classes']
        assert len(classes) >= 1  # Calculator
        
        class_names = [c['name'] for c in classes]
        assert 'Calculator' in class_names
        
        # 检查Calculator类的方法
        calculator_class = next(c for c in classes if c['name'] == 'Calculator')
        method_names = [m['name'] for m in calculator_class['methods']]
        assert '__init__' in method_names
        assert 'add' in method_names
        assert 'divide' in method_names
    
    def test_analyze_directory(self):
        """测试目录分析"""
        result = self.analyzer.analyze_directory(self.temp_dir, recursive=False)
        
        # 检查基本属性
        assert 'directory_path' in result
        assert 'total_files' in result
        assert 'total_lines' in result
        assert 'language_distribution' in result
        assert 'files' in result
        assert 'analysis_type' in result
        
        # 检查分析结果
        assert result['analysis_type'] == 'directory'
        assert result['total_files'] == len(self.test_files)
        assert 'python' in result['language_distribution']
        
        # 检查文件列表
        files = result['files']
        assert len(files) == len(self.test_files)
    
    def test_analyze_nonexistent_file(self):
        """测试分析不存在的文件"""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze_file('/nonexistent/path/file.py')
    
    def test_analyze_nonexistent_directory(self):
        """测试分析不存在的目录"""
        with pytest.raises(FileNotFoundError):
            self.analyzer.analyze_directory('/nonexistent/path')
    
    def test_get_file_language(self):
        """测试文件语言检测"""
        assert self.analyzer._get_file_language(Path('test.py')) == 'python'
        assert self.analyzer._get_file_language(Path('test.js')) == 'javascript'
        assert self.analyzer._get_file_language(Path('test.java')) == 'java'
        assert self.analyzer._get_file_language(Path('test.unknown')) == 'unknown'
    
    def test_is_test_file(self):
        """测试测试文件检测"""
        assert self.analyzer._is_test_file(Path('test_example.py'))
        assert self.analyzer._is_test_file(Path('example_test.py'))
        assert self.analyzer._is_test_file(Path('tests.py'))
        assert not self.analyzer._is_test_file(Path('example.py'))
    
    def test_count_comments(self):
        """测试注释计数"""
        code_with_comments = '''
# This is a comment
def func():
    # Another comment
    return 1  # Inline comment
'''
        count = self.analyzer._count_comments(code_with_comments)
        assert count == 2  # 只计算以#开头的行
    
    def test_get_complexity_level(self):
        """测试复杂度等级判断"""
        assert self.analyzer._get_complexity_level(5) == "Low"
        assert self.analyzer._get_complexity_level(15) == "Moderate"
        assert self.analyzer._get_complexity_level(30) == "High"
        assert self.analyzer._get_complexity_level(60) == "Very High"
    
    def test_generate_report(self):
        """测试报告生成"""
        # 测试文件报告
        result = self.analyzer.analyze_file(self.temp_file.name)
        report = self.analyzer.generate_report(result)
        
        assert isinstance(report, str)
        assert '# 代码文件分析报告' in report
        assert result['file_path'] in report
        assert result['language'] in report
        
        # 测试目录报告
        result = self.analyzer.analyze_directory(self.temp_dir)
        report = self.analyzer.generate_report(result)
        
        assert isinstance(report, str)
        assert '# 代码目录分析报告' in report
        assert result['directory_path'] in report
    
    def test_analyze_syntax_error_file(self):
        """测试分析语法错误的文件"""
        # 创建包含语法错误的Python文件
        error_code = '''
def func():
    if True
        print("Missing colon")
'''
        error_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        error_file.write(error_code)
        error_file.close()
        
        try:
            result = self.analyzer.analyze_file(error_file.name)
            assert 'syntax_error' in result
            assert result['syntax_error']['line'] > 0
        finally:
            os.unlink(error_file.name)
    
    def test_analyze_empty_file(self):
        """测试分析空文件"""
        empty_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        empty_file.write('')
        empty_file.close()
        
        try:
            result = self.analyzer.analyze_file(empty_file.name)
            assert result['line_count'] == 0
            assert result['char_count'] == 0
            assert len(result.get('functions', [])) == 0
            assert len(result.get('classes', [])) == 0
        finally:
            os.unlink(empty_file.name)
    
    def test_analyze_non_python_file(self):
        """测试分析非Python文件"""
        js_code = '''
function add(a, b) {
    return a + b;
}

const result = add(1, 2);
console.log(result);
'''
        js_file = tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False, encoding='utf-8')
        js_file.write(js_code)
        js_file.close()
        
        try:
            result = self.analyzer.analyze_file(js_file.name)
            assert result['language'] == 'javascript'
            # 对于非Python文件，应该有通用分析结果
            assert 'comments' in result
            assert 'tokens' in result
        finally:
            os.unlink(js_file.name)
    
    @patch('src.code_analyzer.logger')
    def test_logging(self, mock_logger):
        """测试日志记录"""
        self.analyzer.analyze_file(self.temp_file.name)
        
        # 验证日志调用
        mock_logger.info.assert_called()
        
        # 获取日志调用的参数
        log_calls = mock_logger.info.call_args_list
        assert any('文件分析完成' in str(call) for call in log_calls)
    
    def test_large_file_handling(self):
        """测试大文件处理"""
        # 创建一个较大的文件
        large_code = '''
# Large file test
''' + '\n'.join([f'def func_{i}(): return {i}' for i in range(100)])
        
        large_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        large_file.write(large_code)
        large_file.close()
        
        try:
            result = self.analyzer.analyze_file(large_file.name)
            assert result['line_count'] > 100
            assert len(result['functions']) == 100
        finally:
            os.unlink(large_file.name)
    
    def test_unicode_file_handling(self):
        """测试Unicode文件处理"""
        unicode_code = '''
# -*- coding: utf-8 -*-
def greet():
    return "你好，世界！"

class 测试类:
    def 测试方法(self):
        return "测试"
'''
        unicode_file = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8')
        unicode_file.write(unicode_code)
        unicode_file.close()
        
        try:
            result = self.analyzer.analyze_file(unicode_file.name)
            assert result['language'] == 'python'
            assert len(result['functions']) >= 1
            assert len(result['classes']) >= 1
        finally:
            os.unlink(unicode_file.name)


if __name__ == "__main__":
    pytest.main([__file__])