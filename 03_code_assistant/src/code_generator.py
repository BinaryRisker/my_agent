"""
代码生成器
提供单元测试生成、代码补全、代码重构等功能
"""

import ast
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import re
import textwrap

# Third-party imports
from loguru import logger
import jinja2

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config
from common.llm_client import get_llm_client


class CodeGenerator:
    """代码生成器主类"""
    
    def __init__(self):
        """初始化代码生成器"""
        self.config = get_config()
        self.llm_client = get_llm_client()
        
        # 模板引擎
        self.jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # 代码模板
        self._init_templates()
        
        logger.info("代码生成器初始化完成")
    
    def _init_templates(self):
        """初始化代码模板"""
        
        # Python单元测试模板
        self.python_test_template = '''
import unittest
from unittest.mock import Mock, patch
{% if imports %}
{% for import_line in imports %}
{{ import_line }}
{% endfor %}
{% endif %}


class Test{{ class_name }}(unittest.TestCase):
    """{{ class_name }} 的测试类"""
    
    def setUp(self):
        """测试前的准备工作"""
        {% if setup_code %}
        {{ setup_code | indent(8) }}
        {% else %}
        pass
        {% endif %}
    
    {% for test_method in test_methods %}
    def {{ test_method.name }}(self):
        """{{ test_method.description }}"""
        {% if test_method.setup %}
        # 准备测试数据
        {{ test_method.setup | indent(8) }}
        {% endif %}
        
        {% if test_method.body %}
        {{ test_method.body | indent(8) }}
        {% else %}
        # TODO: 实现测试逻辑
        pass
        {% endif %}
    
    {% endfor %}
    def tearDown(self):
        """测试后的清理工作"""
        {% if teardown_code %}
        {{ teardown_code | indent(8) }}
        {% else %}
        pass
        {% endif %}


if __name__ == '__main__':
    unittest.main()
'''
        
        # Python类模板
        self.python_class_template = '''
class {{ class_name }}{% if base_classes %}({{ base_classes | join(', ') }}){% endif %}:
    """{{ class_docstring }}"""
    
    def __init__(self{% if init_params %}, {{ init_params | join(', ') }}{% endif %}):
        """初始化{{ class_name }}"""
        {% for param in init_params %}
        self.{{ param }} = {{ param }}
        {% endfor %}
    
    {% for method in methods %}
    def {{ method.name }}(self{% if method.params %}, {{ method.params | join(', ') }}{% endif %}){% if method.return_type %} -> {{ method.return_type }}{% endif %}:
        """{{ method.docstring }}"""
        {% if method.body %}
        {{ method.body | indent(8) }}
        {% else %}
        pass
        {% endif %}
    
    {% endfor %}
'''
        
        # Python函数模板
        self.python_function_template = '''
def {{ function_name }}({{ params | join(', ') }}){% if return_type %} -> {{ return_type }}{% endif %}:
    """
    {{ docstring }}
    
    {% if param_docs %}
    Args:
    {% for param, desc in param_docs.items() %}
        {{ param }}: {{ desc }}
    {% endfor %}
    {% endif %}
    
    {% if return_doc %}
    Returns:
        {{ return_doc }}
    {% endif %}
    """
    {% if body %}
    {{ body | indent(4) }}
    {% else %}
    pass
    {% endif %}
'''

    def generate_unit_tests(self, 
                           source_file: Union[str, Path],
                           target_class: Optional[str] = None,
                           coverage_target: float = 0.8) -> str:
        """
        为Python代码生成单元测试
        
        Args:
            source_file: 源代码文件路径
            target_class: 目标类名（可选）
            coverage_target: 目标覆盖率
            
        Returns:
            生成的测试代码
        """
        source_file = Path(source_file)
        
        if not source_file.exists():
            raise FileNotFoundError(f"源文件不存在: {source_file}")
        
        # 读取源代码
        with open(source_file, 'r', encoding='utf-8') as f:
            source_code = f.read()
        
        try:
            # 解析AST
            tree = ast.parse(source_code)
            
            # 分析代码结构
            code_analysis = self._analyze_code_structure(tree)
            
            # 生成测试用例
            test_cases = self._generate_test_cases(code_analysis, target_class, coverage_target)
            
            # 渲染测试代码
            test_code = self._render_test_code(test_cases, source_file)
            
            logger.info(f"单元测试生成完成: {len(test_cases['test_methods'])} 个测试方法")
            return test_code
            
        except Exception as e:
            logger.error(f"单元测试生成失败: {e}")
            raise
    
    def generate_docstring(self, 
                          function_code: str,
                          style: str = "google") -> str:
        """
        为函数生成文档字符串
        
        Args:
            function_code: 函数源代码
            style: 文档字符串风格 (google, numpy, sphinx)
            
        Returns:
            生成的文档字符串
        """
        try:
            # 解析函数
            tree = ast.parse(function_code)
            
            # 找到函数定义
            func_def = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_def = node
                    break
            
            if not func_def:
                raise ValueError("代码中未找到函数定义")
            
            # 分析函数签名
            func_info = self._analyze_function_signature(func_def)
            
            # 使用LLM生成文档字符串
            docstring = self._generate_docstring_with_llm(func_info, style)
            
            logger.info(f"文档字符串生成完成: {func_def.name}")
            return docstring
            
        except Exception as e:
            logger.error(f"文档字符串生成失败: {e}")
            raise
    
    def complete_code(self, 
                     incomplete_code: str,
                     context: Optional[str] = None,
                     language: str = "python") -> str:
        """
        代码自动补全
        
        Args:
            incomplete_code: 不完整的代码
            context: 上下文信息
            language: 编程语言
            
        Returns:
            补全后的代码
        """
        try:
            # 使用LLM进行代码补全
            completion_prompt = self._build_completion_prompt(
                incomplete_code, context, language
            )
            
            completed_code = self.llm_client.generate(
                prompt=completion_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            logger.info("代码补全完成")
            return completed_code.strip()
            
        except Exception as e:
            logger.error(f"代码补全失败: {e}")
            raise
    
    def refactor_code(self, 
                     source_code: str,
                     refactor_type: str = "extract_method",
                     parameters: Optional[Dict[str, Any]] = None) -> str:
        """
        代码重构
        
        Args:
            source_code: 源代码
            refactor_type: 重构类型 (extract_method, rename, inline)
            parameters: 重构参数
            
        Returns:
            重构后的代码
        """
        try:
            if refactor_type == "extract_method":
                return self._extract_method(source_code, parameters or {})
            elif refactor_type == "rename":
                return self._rename_symbol(source_code, parameters or {})
            elif refactor_type == "inline":
                return self._inline_variable(source_code, parameters or {})
            else:
                raise ValueError(f"不支持的重构类型: {refactor_type}")
                
        except Exception as e:
            logger.error(f"代码重构失败: {e}")
            raise
    
    def generate_class(self, 
                      class_spec: Dict[str, Any]) -> str:
        """
        生成Python类
        
        Args:
            class_spec: 类规格说明
            
        Returns:
            生成的类代码
        """
        try:
            # 验证类规格
            self._validate_class_spec(class_spec)
            
            # 渲染类模板
            template = self.jinja_env.from_string(self.python_class_template)
            class_code = template.render(**class_spec)
            
            logger.info(f"类生成完成: {class_spec['class_name']}")
            return class_code
            
        except Exception as e:
            logger.error(f"类生成失败: {e}")
            raise
    
    def generate_function(self, 
                         function_spec: Dict[str, Any]) -> str:
        """
        生成Python函数
        
        Args:
            function_spec: 函数规格说明
            
        Returns:
            生成的函数代码
        """
        try:
            # 验证函数规格
            self._validate_function_spec(function_spec)
            
            # 渲染函数模板
            template = self.jinja_env.from_string(self.python_function_template)
            function_code = template.render(**function_spec)
            
            logger.info(f"函数生成完成: {function_spec['function_name']}")
            return function_code
            
        except Exception as e:
            logger.error(f"函数生成失败: {e}")
            raise
    
    def _analyze_code_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """分析代码结构"""
        structure = {
            'classes': [],
            'functions': [],
            'imports': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'methods': [],
                    'line': node.lineno
                }
                
                # 分析类方法
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            'name': item.name,
                            'args': [arg.arg for arg in item.args.args],
                            'line': item.lineno,
                            'is_private': item.name.startswith('_')
                        }
                        class_info['methods'].append(method_info)
                
                structure['classes'].append(class_info)
                
            elif isinstance(node, ast.FunctionDef):
                # 只记录不在类中的函数
                if not any(isinstance(parent, ast.ClassDef) for parent in ast.walk(tree)):
                    function_info = {
                        'name': node.name,
                        'args': [arg.arg for arg in node.args.args],
                        'line': node.lineno
                    }
                    structure['functions'].append(function_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                # 记录导入信息
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        structure['imports'].append(f"import {alias.name}")
                else:
                    module = node.module or ""
                    for alias in node.names:
                        structure['imports'].append(f"from {module} import {alias.name}")
        
        return structure
    
    def _generate_test_cases(self, 
                           code_analysis: Dict[str, Any],
                           target_class: Optional[str],
                           coverage_target: float) -> Dict[str, Any]:
        """生成测试用例"""
        test_methods = []
        
        # 为类生成测试
        for class_info in code_analysis['classes']:
            if target_class and class_info['name'] != target_class:
                continue
                
            class_name = class_info['name']
            
            # 为每个公共方法生成测试
            for method in class_info['methods']:
                if not method['is_private']:
                    test_method = {
                        'name': f"test_{method['name']}",
                        'description': f"测试 {class_name}.{method['name']} 方法",
                        'setup': f"self.instance = {class_name}()",
                        'body': self._generate_test_method_body(method)
                    }
                    test_methods.append(test_method)
        
        # 为独立函数生成测试
        for func_info in code_analysis['functions']:
            test_method = {
                'name': f"test_{func_info['name']}",
                'description': f"测试 {func_info['name']} 函数",
                'setup': None,
                'body': self._generate_test_function_body(func_info)
            }
            test_methods.append(test_method)
        
        return {
            'class_name': target_class or 'GeneratedTests',
            'imports': code_analysis['imports'],
            'test_methods': test_methods,
            'setup_code': None,
            'teardown_code': None
        }
    
    def _generate_test_method_body(self, method_info: Dict[str, Any]) -> str:
        """生成测试方法主体"""
        method_name = method_info['name']
        args = method_info['args']
        
        if method_name == '__init__':
            return "# 构造函数在 setUp 中测试\npass"
        
        # 生成基本的测试框架
        test_body = f"""
# 测试正常情况
result = self.instance.{method_name}({', '.join(['test_value'] * (len(args) - 1))})
self.assertIsNotNone(result)

# 测试边界情况
# TODO: 添加边界条件测试

# 测试异常情况  
# TODO: 添加异常情况测试
"""
        return test_body.strip()
    
    def _generate_test_function_body(self, func_info: Dict[str, Any]) -> str:
        """生成测试函数主体"""
        func_name = func_info['name']
        args = func_info['args']
        
        test_body = f"""
# 测试正常情况
result = {func_name}({', '.join(['test_value'] * len(args))})
self.assertIsNotNone(result)

# 测试边界情况
# TODO: 添加边界条件测试

# 测试异常情况
# TODO: 添加异常情况测试
"""
        return test_body.strip()
    
    def _render_test_code(self, test_cases: Dict[str, Any], source_file: Path) -> str:
        """渲染测试代码"""
        template = self.jinja_env.from_string(self.python_test_template)
        
        # 添加源文件导入
        if 'imports' not in test_cases:
            test_cases['imports'] = []
        
        # 添加被测试模块的导入
        module_name = source_file.stem
        test_cases['imports'].insert(0, f"from {module_name} import *")
        
        return template.render(**test_cases)
    
    def _analyze_function_signature(self, func_def: ast.FunctionDef) -> Dict[str, Any]:
        """分析函数签名"""
        return {
            'name': func_def.name,
            'args': [arg.arg for arg in func_def.args.args],
            'defaults': [ast.unparse(default) if hasattr(ast, 'unparse') else str(default) 
                        for default in func_def.args.defaults],
            'returns': ast.unparse(func_def.returns) if func_def.returns else None,
            'docstring': ast.get_docstring(func_def),
            'is_async': isinstance(func_def, ast.AsyncFunctionDef)
        }
    
    def _generate_docstring_with_llm(self, func_info: Dict[str, Any], style: str) -> str:
        """使用LLM生成文档字符串"""
        prompt = f"""
为以下Python函数生成{style}风格的文档字符串：

函数名: {func_info['name']}
参数: {', '.join(func_info['args'])}
返回类型: {func_info.get('returns', 'None')}
是否异步: {func_info['is_async']}

请生成详细的文档字符串，包括：
1. 函数功能描述
2. 参数说明
3. 返回值说明
4. 可能的异常
5. 使用示例

只返回文档字符串内容，不要包含函数定义。
"""
        
        try:
            docstring = self.llm_client.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3
            )
            return docstring.strip()
        except Exception:
            # 如果LLM调用失败，返回基本的文档字符串
            return f"{func_info['name']} 函数"
    
    def _build_completion_prompt(self, 
                               incomplete_code: str,
                               context: Optional[str],
                               language: str) -> str:
        """构建代码补全提示"""
        prompt = f"请补全以下{language}代码：\n\n"
        
        if context:
            prompt += f"上下文信息：\n{context}\n\n"
        
        prompt += f"不完整的代码：\n{incomplete_code}\n\n"
        prompt += "请提供补全后的完整代码，只返回代码，不要解释："
        
        return prompt
    
    def _extract_method(self, source_code: str, parameters: Dict[str, Any]) -> str:
        """提取方法重构"""
        # 简化实现
        logger.info("执行提取方法重构")
        return source_code  # 实际项目中需要实现具体的重构逻辑
    
    def _rename_symbol(self, source_code: str, parameters: Dict[str, Any]) -> str:
        """重命名符号重构"""
        old_name = parameters.get('old_name', '')
        new_name = parameters.get('new_name', '')
        
        if not old_name or not new_name:
            raise ValueError("重命名参数不完整")
        
        # 使用正则表达式进行简单替换
        pattern = r'\b' + re.escape(old_name) + r'\b'
        refactored_code = re.sub(pattern, new_name, source_code)
        
        logger.info(f"重命名完成: {old_name} -> {new_name}")
        return refactored_code
    
    def _inline_variable(self, source_code: str, parameters: Dict[str, Any]) -> str:
        """内联变量重构"""
        logger.info("执行内联变量重构")
        return source_code  # 实际项目中需要实现具体的重构逻辑
    
    def _validate_class_spec(self, class_spec: Dict[str, Any]):
        """验证类规格"""
        required_fields = ['class_name', 'class_docstring']
        for field in required_fields:
            if field not in class_spec:
                raise ValueError(f"类规格缺少必需字段: {field}")
    
    def _validate_function_spec(self, function_spec: Dict[str, Any]):
        """验证函数规格"""
        required_fields = ['function_name', 'docstring']
        for field in required_fields:
            if field not in function_spec:
                raise ValueError(f"函数规格缺少必需字段: {field}")


if __name__ == "__main__":
    # 测试代码生成器
    generator = CodeGenerator()
    
    # 测试函数生成
    function_spec = {
        'function_name': 'calculate_area',
        'params': ['length', 'width'],
        'return_type': 'float',
        'docstring': '计算矩形面积',
        'param_docs': {
            'length': '矩形长度',
            'width': '矩形宽度'
        },
        'return_doc': '矩形面积',
        'body': 'return length * width'
    }
    
    try:
        function_code = generator.generate_function(function_spec)
        print("生成的函数代码：")
        print(function_code)
        
        # 测试类生成
        class_spec = {
            'class_name': 'Rectangle',
            'class_docstring': '矩形类',
            'init_params': ['length', 'width'],
            'methods': [
                {
                    'name': 'area',
                    'params': [],
                    'return_type': 'float',
                    'docstring': '计算面积',
                    'body': 'return self.length * self.width'
                }
            ]
        }
        
        class_code = generator.generate_class(class_spec)
        print("\n生成的类代码：")
        print(class_code)
        
        print("\n代码生成器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")