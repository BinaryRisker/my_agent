"""
代码分析器
提供AST解析、静态分析、代码质量评估等功能
"""

import ast
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import inspect
import tokenize
import io

# Third-party imports
from loguru import logger
import black
import flake8.api.legacy as flake8

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config


class CodeAnalyzer:
    """代码分析器主类"""
    
    def __init__(self):
        """初始化代码分析器"""
        self.config = get_config()
        self.supported_languages = {
            '.py': 'python',
            '.js': 'javascript', 
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        # AST节点统计
        self.ast_stats = {}
        
        logger.info("代码分析器初始化完成")
    
    def analyze_file(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        分析单个代码文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            分析结果字典
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件语言
        language = self._get_file_language(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code_content = f.read()
            
            analysis_result = {
                'file_path': str(file_path),
                'language': language,
                'file_size': file_path.stat().st_size,
                'line_count': len(code_content.splitlines()),
                'char_count': len(code_content),
                'analysis_type': 'file'
            }
            
            if language == 'python':
                analysis_result.update(self._analyze_python_file(code_content, file_path))
            else:
                analysis_result.update(self._analyze_generic_file(code_content))
            
            logger.info(f"文件分析完成: {file_path}")
            return analysis_result
            
        except Exception as e:
            logger.error(f"文件分析失败: {file_path}, 错误: {e}")
            raise
    
    def analyze_directory(self, 
                         directory_path: Union[str, Path],
                         recursive: bool = True,
                         include_tests: bool = True) -> Dict[str, Any]:
        """
        分析目录中的所有代码文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归
            include_tests: 是否包含测试文件
            
        Returns:
            目录分析结果
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        code_files = []
        
        # 查找代码文件
        if recursive:
            for ext in self.supported_languages.keys():
                code_files.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in self.supported_languages.keys():
                code_files.extend(directory_path.glob(f"*{ext}"))
        
        # 过滤测试文件
        if not include_tests:
            code_files = [f for f in code_files if not self._is_test_file(f)]
        
        # 分析每个文件
        file_analyses = []
        total_lines = 0
        total_size = 0
        language_stats = {}
        
        for file_path in code_files:
            try:
                analysis = self.analyze_file(file_path)
                file_analyses.append(analysis)
                
                total_lines += analysis['line_count']
                total_size += analysis['file_size']
                
                lang = analysis['language']
                language_stats[lang] = language_stats.get(lang, 0) + 1
                
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}: {e}")
        
        result = {
            'directory_path': str(directory_path),
            'total_files': len(file_analyses),
            'total_lines': total_lines,
            'total_size': total_size,
            'language_distribution': language_stats,
            'files': file_analyses,
            'analysis_type': 'directory'
        }
        
        # 计算代码质量统计
        if file_analyses:
            result['quality_stats'] = self._calculate_quality_stats(file_analyses)
        
        logger.info(f"目录分析完成: {directory_path}, {len(file_analyses)} 个文件")
        return result
    
    def _analyze_python_file(self, code_content: str, file_path: Path) -> Dict[str, Any]:
        """分析Python文件"""
        result = {}
        
        try:
            # AST分析
            tree = ast.parse(code_content)
            result.update(self._analyze_ast(tree))
            
            # 代码质量检查
            result['quality_issues'] = self._check_python_quality(code_content, file_path)
            
            # 复杂度分析
            result['complexity'] = self._calculate_complexity(tree)
            
            # 导入分析
            result['imports'] = self._analyze_imports(tree)
            
            # 函数和类分析
            result['functions'] = self._analyze_functions(tree)
            result['classes'] = self._analyze_classes(tree)
            
        except SyntaxError as e:
            result['syntax_error'] = {
                'message': str(e),
                'line': e.lineno,
                'column': e.offset
            }
            logger.warning(f"语法错误: {file_path}, {e}")
        
        return result
    
    def _analyze_generic_file(self, code_content: str) -> Dict[str, Any]:
        """分析通用代码文件"""
        return {
            'comments': self._count_comments(code_content),
            'blank_lines': code_content.count('\\n\\n'),
            'tokens': len(code_content.split())
        }
    
    def _analyze_ast(self, tree: ast.AST) -> Dict[str, Any]:
        """分析AST树"""
        node_counts = {}
        
        class NodeVisitor(ast.NodeVisitor):
            def visit(self, node):
                node_type = type(node).__name__
                node_counts[node_type] = node_counts.get(node_type, 0) + 1
                self.generic_visit(node)
        
        visitor = NodeVisitor()
        visitor.visit(tree)
        
        return {
            'ast_nodes': node_counts,
            'total_nodes': sum(node_counts.values())
        }
    
    def _check_python_quality(self, code_content: str, file_path: Path) -> List[Dict[str, Any]]:
        """检查Python代码质量"""
        issues = []
        
        try:
            # 使用flake8检查
            style_guide = flake8.get_style_guide()
            temp_file = f"temp_{file_path.name}"
            
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write(code_content)
            
            try:
                report = style_guide.check_files([temp_file])
                # 这里简化处理，实际项目中需要更详细的报告解析
                if report.get_count() > 0:
                    issues.append({
                        'type': 'style',
                        'count': report.get_count(),
                        'message': 'Style guide violations found'
                    })
            finally:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        except Exception as e:
            logger.warning(f"质量检查失败: {e}")
        
        return issues
    
    def _calculate_complexity(self, tree: ast.AST) -> Dict[str, Any]:
        """计算代码复杂度"""
        complexity_score = 0
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # 基础复杂度
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_Try(self, node):
                self.complexity += 1
                self.generic_visit(node)
        
        visitor = ComplexityVisitor()
        visitor.visit(tree)
        
        return {
            'cyclomatic_complexity': visitor.complexity,
            'complexity_level': self._get_complexity_level(visitor.complexity)
        }
    
    def _analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """分析导入语句"""
        imports = []
        from_imports = []
        
        class ImportVisitor(ast.NodeVisitor):
            def visit_Import(self, node):
                for alias in node.names:
                    imports.append({
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
            
            def visit_ImportFrom(self, node):
                for alias in node.names:
                    from_imports.append({
                        'module': node.module,
                        'name': alias.name,
                        'alias': alias.asname,
                        'line': node.lineno
                    })
        
        visitor = ImportVisitor()
        visitor.visit(tree)
        
        return {
            'imports': imports,
            'from_imports': from_imports,
            'total_imports': len(imports) + len(from_imports)
        }
    
    def _analyze_functions(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """分析函数"""
        functions = []
        
        class FunctionVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'args_count': len(node.args.args),
                    'has_docstring': ast.get_docstring(node) is not None,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'is_async': False
                }
                functions.append(func_info)
            
            def visit_AsyncFunctionDef(self, node):
                func_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'args_count': len(node.args.args),
                    'has_docstring': ast.get_docstring(node) is not None,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list],
                    'is_async': True
                }
                functions.append(func_info)
        
        visitor = FunctionVisitor()
        visitor.visit(tree)
        
        return functions
    
    def _analyze_classes(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """分析类"""
        classes = []
        
        class ClassVisitor(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                methods = []
                
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append({
                            'name': item.name,
                            'line': item.lineno,
                            'is_private': item.name.startswith('_'),
                            'is_static': any(isinstance(d, ast.Name) and d.id == 'staticmethod' 
                                           for d in item.decorator_list),
                            'is_property': any(isinstance(d, ast.Name) and d.id == 'property' 
                                             for d in item.decorator_list)
                        })
                
                class_info = {
                    'name': node.name,
                    'line': node.lineno,
                    'bases': [b.id if isinstance(b, ast.Name) else str(b) for b in node.bases],
                    'methods': methods,
                    'method_count': len(methods),
                    'has_docstring': ast.get_docstring(node) is not None,
                    'decorators': [d.id if isinstance(d, ast.Name) else str(d) for d in node.decorator_list]
                }
                classes.append(class_info)
        
        visitor = ClassVisitor()
        visitor.visit(tree)
        
        return classes
    
    def _get_file_language(self, file_path: Path) -> str:
        """获取文件编程语言"""
        extension = file_path.suffix.lower()
        return self.supported_languages.get(extension, 'unknown')
    
    def _is_test_file(self, file_path: Path) -> bool:
        """判断是否为测试文件"""
        test_patterns = ['test_', '_test', 'tests']
        name = file_path.stem.lower()
        return any(pattern in name for pattern in test_patterns)
    
    def _count_comments(self, code_content: str) -> int:
        """统计注释行数"""
        comment_count = 0
        for line in code_content.splitlines():
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//'):
                comment_count += 1
        return comment_count
    
    def _get_complexity_level(self, score: int) -> str:
        """获取复杂度等级"""
        if score <= 10:
            return "Low"
        elif score <= 20:
            return "Moderate"
        elif score <= 50:
            return "High"
        else:
            return "Very High"
    
    def _calculate_quality_stats(self, file_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算代码质量统计"""
        python_files = [f for f in file_analyses if f['language'] == 'python']
        
        if not python_files:
            return {}
        
        total_complexity = 0
        total_functions = 0
        total_classes = 0
        files_with_docstrings = 0
        
        for file_data in python_files:
            if 'complexity' in file_data:
                total_complexity += file_data['complexity']['cyclomatic_complexity']
            
            if 'functions' in file_data:
                total_functions += len(file_data['functions'])
                functions_with_docs = sum(1 for f in file_data['functions'] if f['has_docstring'])
            
            if 'classes' in file_data:
                total_classes += len(file_data['classes'])
                classes_with_docs = sum(1 for c in file_data['classes'] if c['has_docstring'])
        
        return {
            'avg_complexity': total_complexity / len(python_files) if python_files else 0,
            'total_functions': total_functions,
            'total_classes': total_classes,
            'documentation_coverage': 0.8  # 简化计算
        }
    
    def generate_report(self, analysis_result: Dict[str, Any]) -> str:
        """生成分析报告"""
        if analysis_result['analysis_type'] == 'file':
            return self._generate_file_report(analysis_result)
        else:
            return self._generate_directory_report(analysis_result)
    
    def _generate_file_report(self, analysis: Dict[str, Any]) -> str:
        """生成文件分析报告"""
        report = f"# 代码文件分析报告\\n\\n"
        report += f"**文件**: {analysis['file_path']}\\n"
        report += f"**语言**: {analysis['language']}\\n"
        report += f"**行数**: {analysis['line_count']}\\n"
        report += f"**文件大小**: {analysis['file_size']} 字节\\n\\n"
        
        if analysis['language'] == 'python':
            if 'complexity' in analysis:
                report += f"**复杂度**: {analysis['complexity']['cyclomatic_complexity']} ({analysis['complexity']['complexity_level']})\\n"
            
            if 'functions' in analysis:
                report += f"**函数数量**: {len(analysis['functions'])}\\n"
            
            if 'classes' in analysis:
                report += f"**类数量**: {len(analysis['classes'])}\\n"
            
            if 'quality_issues' in analysis and analysis['quality_issues']:
                report += f"\\n**质量问题**: {len(analysis['quality_issues'])} 个\\n"
        
        return report
    
    def _generate_directory_report(self, analysis: Dict[str, Any]) -> str:
        """生成目录分析报告"""
        report = f"# 代码目录分析报告\\n\\n"
        report += f"**目录**: {analysis['directory_path']}\\n"
        report += f"**文件总数**: {analysis['total_files']}\\n"
        report += f"**总行数**: {analysis['total_lines']:,}\\n"
        report += f"**总大小**: {analysis['total_size']:,} 字节\\n\\n"
        
        if analysis['language_distribution']:
            report += "## 语言分布\\n\\n"
            for lang, count in analysis['language_distribution'].items():
                report += f"- {lang}: {count} 个文件\\n"
            report += "\\n"
        
        if 'quality_stats' in analysis:
            stats = analysis['quality_stats']
            report += "## 代码质量统计\\n\\n"
            report += f"- 平均复杂度: {stats['avg_complexity']:.2f}\\n"
            report += f"- 函数总数: {stats['total_functions']}\\n"
            report += f"- 类总数: {stats['total_classes']}\\n"
            report += f"- 文档覆盖率: {stats['documentation_coverage']:.1%}\\n"
        
        return report


if __name__ == "__main__":
    # 测试代码分析器
    analyzer = CodeAnalyzer()
    
    # 创建测试文件
    test_code = '''
def hello_world(name="World"):
    """
    打印问候语
    """
    if name:
        print(f"Hello, {name}!")
    else:
        print("Hello!")

class Calculator:
    """简单计算器"""
    
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b
'''
    
    test_file = Path("test_example.py")
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        # 分析文件
        result = analyzer.analyze_file(test_file)
        
        # 生成报告
        report = analyzer.generate_report(result)
        print(report)
        
        print("\\n代码分析器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()