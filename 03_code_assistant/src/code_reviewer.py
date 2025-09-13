"""
代码审查器
提供代码质量检查、安全审计、性能分析等功能
"""

import ast
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Set
from dataclasses import dataclass
from enum import Enum

# Third-party imports
from loguru import logger

# Project imports  
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config
from common.llm_client import get_llm_client


class Severity(Enum):
    """问题严重程度"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class IssueCategory(Enum):
    """问题类别"""
    STYLE = "style"
    BUG = "bug"
    PERFORMANCE = "performance"
    SECURITY = "security"
    MAINTAINABILITY = "maintainability"
    DESIGN = "design"


@dataclass
class CodeIssue:
    """代码问题"""
    file_path: str
    line_number: int
    category: IssueCategory
    severity: Severity
    message: str
    description: str
    suggestion: Optional[str] = None
    rule_id: Optional[str] = None
    context: Optional[str] = None


class CodeReviewer:
    """代码审查器主类"""
    
    def __init__(self):
        """初始化代码审查器"""
        self.config = get_config()
        self.llm_client = get_llm_client()
        
        # 安全问题模式
        self.security_patterns = {
            'sql_injection': [
                r'execute\s*\(\s*["\'].*\+.*["\']',
                r'cursor\.execute\s*\(\s*f["\']',
                r'query\s*=\s*["\'].*\%s.*["\']'
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']'
            ]
        }
        
        # 性能问题模式
        self.performance_patterns = {
            'inefficient_loop': [
                r'for.*in.*\.keys\(\)',
                r'while.*len\(',
            ],
            'string_concatenation': [
                r'\+\s*["\'].*["\'].*\+',
            ]
        }
        
        # 代码风格规则
        self.style_rules = {
            'naming': {
                'class': r'^[A-Z][a-zA-Z0-9]*$',
                'function': r'^[a-z_][a-z0-9_]*$', 
                'variable': r'^[a-z_][a-z0-9_]*$',
                'constant': r'^[A-Z_][A-Z0-9_]*$'
            },
            'line_length': 88,
            'max_complexity': 10
        }
        
        logger.info("代码审查器初始化完成")
    
    def review_file(self, file_path: Union[str, Path]) -> List[CodeIssue]:
        """
        审查单个代码文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            发现的问题列表
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 检查不同类型的问题
            issues.extend(self._check_security_issues(content, file_path))
            issues.extend(self._check_performance_issues(content, file_path))
            issues.extend(self._check_style_issues(content, file_path))
            
            # 如果是Python文件，进行AST分析
            if file_path.suffix == '.py':
                try:
                    tree = ast.parse(content)
                    issues.extend(self._check_python_specific_issues(tree, file_path))
                except SyntaxError as e:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=e.lineno or 0,
                        category=IssueCategory.BUG,
                        severity=Severity.HIGH,
                        message="语法错误",
                        description=str(e),
                        rule_id="syntax_error"
                    ))
            
            logger.info(f"文件审查完成: {file_path}, 发现 {len(issues)} 个问题")
            return issues
            
        except Exception as e:
            logger.error(f"文件审查失败: {file_path}, 错误: {e}")
            raise
    
    def review_directory(self, 
                        directory_path: Union[str, Path],
                        recursive: bool = True,
                        file_patterns: List[str] = None) -> Dict[str, List[CodeIssue]]:
        """
        审查目录中的所有代码文件
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归
            file_patterns: 文件模式列表
            
        Returns:
            每个文件的问题字典
        """
        directory_path = Path(directory_path)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"目录不存在: {directory_path}")
        
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c']
        
        results = {}
        total_issues = 0
        
        # 查找代码文件
        code_files = []
        for pattern in file_patterns:
            if recursive:
                code_files.extend(directory_path.rglob(pattern))
            else:
                code_files.extend(directory_path.glob(pattern))
        
        # 审查每个文件
        for file_path in code_files:
            try:
                issues = self.review_file(file_path)
                results[str(file_path)] = issues
                total_issues += len(issues)
            except Exception as e:
                logger.warning(f"跳过文件 {file_path}: {e}")
        
        logger.info(f"目录审查完成: {directory_path}, {len(results)} 个文件, {total_issues} 个问题")
        return results
    
    def _check_security_issues(self, content: str, file_path: Path) -> List[CodeIssue]:
        """检查安全问题"""
        issues = []
        lines = content.splitlines()
        
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            category=IssueCategory.SECURITY,
                            severity=Severity.HIGH,
                            message=f"潜在的{category}风险",
                            description=f"在第{line_num}行发现可能的{category}漏洞",
                            suggestion=self._get_security_suggestion(category),
                            rule_id=f"security_{category}",
                            context=line.strip()
                        ))
        
        return issues
    
    def _check_performance_issues(self, content: str, file_path: Path) -> List[CodeIssue]:
        """检查性能问题"""
        issues = []
        lines = content.splitlines()
        
        for category, patterns in self.performance_patterns.items():
            for pattern in patterns:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        issues.append(CodeIssue(
                            file_path=str(file_path),
                            line_number=line_num,
                            category=IssueCategory.PERFORMANCE,
                            severity=Severity.MEDIUM,
                            message=f"性能问题: {category}",
                            description=f"在第{line_num}行发现性能问题",
                            suggestion=self._get_performance_suggestion(category),
                            rule_id=f"performance_{category}",
                            context=line.strip()
                        ))
        
        return issues
    
    def _check_style_issues(self, content: str, file_path: Path) -> List[CodeIssue]:
        """检查代码风格问题"""
        issues = []
        lines = content.splitlines()
        
        # 检查行长度
        max_length = self.style_rules['line_length']
        for line_num, line in enumerate(lines, 1):
            if len(line) > max_length:
                issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    category=IssueCategory.STYLE,
                    severity=Severity.LOW,
                    message="行过长",
                    description=f"第{line_num}行长度{len(line)}超过限制{max_length}",
                    suggestion="将长行拆分为多行",
                    rule_id="line_length"
                ))
        
        # 检查空白行
        for line_num, line in enumerate(lines, 1):
            if line.strip() == '' and line_num < len(lines):
                next_line = lines[line_num] if line_num < len(lines) else ''
                if next_line.strip() == '':
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=line_num,
                        category=IssueCategory.STYLE,
                        severity=Severity.LOW,
                        message="多余的空白行",
                        description=f"第{line_num}行有多余的空白行",
                        suggestion="删除多余的空白行",
                        rule_id="blank_lines"
                    ))
        
        return issues
    
    def _check_python_specific_issues(self, tree: ast.AST, file_path: Path) -> List[CodeIssue]:
        """检查Python特定问题"""
        issues = []
        
        # 检查命名规范
        issues.extend(self._check_naming_conventions(tree, file_path))
        
        # 检查复杂度
        issues.extend(self._check_complexity(tree, file_path))
        
        # 检查重复代码
        issues.extend(self._check_code_duplication(tree, file_path))
        
        # 检查未使用的变量和导入
        issues.extend(self._check_unused_items(tree, file_path))
        
        return issues
    
    def _check_naming_conventions(self, tree: ast.AST, file_path: Path) -> List[CodeIssue]:
        """检查命名规范"""
        issues = []
        naming_rules = self.style_rules['naming']
        
        class NamingVisitor(ast.NodeVisitor):
            def __init__(self, issues_list, file_path):
                self.issues = issues_list
                self.file_path = file_path
            
            def visit_ClassDef(self, node):
                if not re.match(naming_rules['class'], node.name):
                    self.issues.append(CodeIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        category=IssueCategory.STYLE,
                        severity=Severity.LOW,
                        message="类名不符合规范",
                        description=f"类名 '{node.name}' 应该使用驼峰命名法",
                        suggestion="使用驼峰命名法，如 MyClass",
                        rule_id="naming_class"
                    ))
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if not re.match(naming_rules['function'], node.name):
                    self.issues.append(CodeIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        category=IssueCategory.STYLE,
                        severity=Severity.LOW,
                        message="函数名不符合规范",
                        description=f"函数名 '{node.name}' 应该使用下划线分隔的小写字母",
                        suggestion="使用下划线分隔的小写字母，如 my_function",
                        rule_id="naming_function"
                    ))
                self.generic_visit(node)
        
        visitor = NamingVisitor(issues, file_path)
        visitor.visit(tree)
        
        return issues
    
    def _check_complexity(self, tree: ast.AST, file_path: Path) -> List[CodeIssue]:
        """检查复杂度"""
        issues = []
        max_complexity = self.style_rules['max_complexity']
        
        class ComplexityVisitor(ast.NodeVisitor):
            def __init__(self, issues_list, file_path, max_complexity):
                self.issues = issues_list
                self.file_path = file_path
                self.max_complexity = max_complexity
            
            def visit_FunctionDef(self, node):
                complexity = self._calculate_complexity(node)
                if complexity > self.max_complexity:
                    self.issues.append(CodeIssue(
                        file_path=str(self.file_path),
                        line_number=node.lineno,
                        category=IssueCategory.MAINTAINABILITY,
                        severity=Severity.MEDIUM,
                        message="函数复杂度过高",
                        description=f"函数 '{node.name}' 的复杂度为 {complexity}, 超过限制 {self.max_complexity}",
                        suggestion="考虑拆分函数或简化逻辑",
                        rule_id="complexity"
                    ))
                self.generic_visit(node)
            
            def _calculate_complexity(self, node):
                complexity = 1
                for child in ast.walk(node):
                    if isinstance(child, (ast.If, ast.For, ast.While, ast.Try, 
                                        ast.ExceptHandler, ast.With)):
                        complexity += 1
                return complexity
        
        visitor = ComplexityVisitor(issues, file_path, max_complexity)
        visitor.visit(tree)
        
        return issues
    
    def _check_code_duplication(self, tree: ast.AST, file_path: Path) -> List[CodeIssue]:
        """检查代码重复"""
        issues = []
        
        # 简化实现：检查相同的函数名
        function_names = []
        
        class DuplicationVisitor(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                if node.name in function_names:
                    issues.append(CodeIssue(
                        file_path=str(file_path),
                        line_number=node.lineno,
                        category=IssueCategory.DESIGN,
                        severity=Severity.MEDIUM,
                        message="重复的函数名",
                        description=f"函数名 '{node.name}' 重复",
                        suggestion="使用不同的函数名或合并重复的代码",
                        rule_id="duplication"
                    ))
                else:
                    function_names.append(node.name)
                self.generic_visit(node)
        
        visitor = DuplicationVisitor()
        visitor.visit(tree)
        
        return issues
    
    def _check_unused_items(self, tree: ast.AST, file_path: Path) -> List[CodeIssue]:
        """检查未使用的变量和导入"""
        issues = []
        
        # 收集所有名称的定义和使用
        defined_names = set()
        used_names = set()
        
        class NameVisitor(ast.NodeVisitor):
            def visit_Name(self, node):
                if isinstance(node.ctx, ast.Store):
                    defined_names.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    used_names.add(node.id)
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                defined_names.add(node.name)
                self.generic_visit(node)
            
            def visit_ClassDef(self, node):
                defined_names.add(node.name)
                self.generic_visit(node)
        
        visitor = NameVisitor()
        visitor.visit(tree)
        
        # 找出未使用的名称
        unused_names = defined_names - used_names
        for name in unused_names:
            # 跳过特殊名称
            if name.startswith('_') or name in ['main']:
                continue
                
            issues.append(CodeIssue(
                file_path=str(file_path),
                line_number=0,  # 简化实现，不确定具体行号
                category=IssueCategory.MAINTAINABILITY,
                severity=Severity.LOW,
                message="未使用的名称",
                description=f"名称 '{name}' 定义但未使用",
                suggestion="删除未使用的名称或添加使用该名称的代码",
                rule_id="unused_name"
            ))
        
        return issues
    
    def _get_security_suggestion(self, category: str) -> str:
        """获取安全问题建议"""
        suggestions = {
            'sql_injection': "使用参数化查询或ORM框架",
            'command_injection': "避免直接执行用户输入，使用安全的替代方案",
            'hardcoded_secrets': "将敏感信息存储在环境变量或配置文件中"
        }
        return suggestions.get(category, "请检查安全风险")
    
    def _get_performance_suggestion(self, category: str) -> str:
        """获取性能问题建议"""
        suggestions = {
            'inefficient_loop': "考虑使用更高效的迭代方式",
            'string_concatenation': "使用 join() 方法或 f-string 格式化"
        }
        return suggestions.get(category, "考虑优化性能")
    
    def generate_report(self, 
                       review_results: Union[List[CodeIssue], Dict[str, List[CodeIssue]]],
                       format_type: str = "markdown") -> str:
        """
        生成审查报告
        
        Args:
            review_results: 审查结果
            format_type: 报告格式 (markdown, json, html)
            
        Returns:
            生成的报告
        """
        if isinstance(review_results, list):
            # 单文件结果
            return self._generate_single_file_report(review_results, format_type)
        else:
            # 多文件结果
            return self._generate_multi_file_report(review_results, format_type)
    
    def _generate_single_file_report(self, issues: List[CodeIssue], format_type: str) -> str:
        """生成单文件报告"""
        if format_type == "markdown":
            report = "# 代码审查报告\\n\\n"
            
            # 统计信息
            total = len(issues)
            by_severity = {}
            by_category = {}
            
            for issue in issues:
                by_severity[issue.severity.value] = by_severity.get(issue.severity.value, 0) + 1
                by_category[issue.category.value] = by_category.get(issue.category.value, 0) + 1
            
            report += f"**总问题数**: {total}\\n\\n"
            
            if by_severity:
                report += "## 按严重程度分类\\n\\n"
                for severity, count in sorted(by_severity.items()):
                    report += f"- {severity.title()}: {count}\\n"
                report += "\\n"
            
            if by_category:
                report += "## 按类别分类\\n\\n"
                for category, count in sorted(by_category.items()):
                    report += f"- {category.title()}: {count}\\n"
                report += "\\n"
            
            # 详细问题列表
            if issues:
                report += "## 详细问题\\n\\n"
                for i, issue in enumerate(issues, 1):
                    report += f"### {i}. {issue.message}\\n\\n"
                    report += f"**文件**: {issue.file_path}\\n"
                    report += f"**行号**: {issue.line_number}\\n"
                    report += f"**类别**: {issue.category.value}\\n"
                    report += f"**严重程度**: {issue.severity.value}\\n"
                    report += f"**描述**: {issue.description}\\n"
                    
                    if issue.suggestion:
                        report += f"**建议**: {issue.suggestion}\\n"
                    
                    if issue.context:
                        report += f"**代码上下文**:\\n```\\n{issue.context}\\n```\\n"
                    
                    report += "\\n---\\n\\n"
            
            return report
        
        else:
            return "不支持的报告格式"
    
    def _generate_multi_file_report(self, results: Dict[str, List[CodeIssue]], format_type: str) -> str:
        """生成多文件报告"""
        if format_type == "markdown":
            report = "# 代码审查报告\\n\\n"
            
            # 总体统计
            total_issues = sum(len(issues) for issues in results.values())
            total_files = len(results)
            
            report += f"**审查文件数**: {total_files}\\n"
            report += f"**总问题数**: {total_issues}\\n\\n"
            
            # 按文件统计
            if results:
                report += "## 文件统计\\n\\n"
                for file_path, issues in results.items():
                    report += f"- {file_path}: {len(issues)} 个问题\\n"
                report += "\\n"
            
            # 严重程度统计
            severity_stats = {}
            category_stats = {}
            
            for issues in results.values():
                for issue in issues:
                    severity_stats[issue.severity.value] = severity_stats.get(issue.severity.value, 0) + 1
                    category_stats[issue.category.value] = category_stats.get(issue.category.value, 0) + 1
            
            if severity_stats:
                report += "## 严重程度分布\\n\\n"
                for severity, count in sorted(severity_stats.items()):
                    report += f"- {severity.title()}: {count}\\n"
                report += "\\n"
            
            if category_stats:
                report += "## 问题类别分布\\n\\n"
                for category, count in sorted(category_stats.items()):
                    report += f"- {category.title()}: {count}\\n"
                report += "\\n"
            
            # 每个文件的详细问题
            for file_path, issues in results.items():
                if issues:
                    report += f"## {file_path}\\n\\n"
                    for issue in issues:
                        report += f"- **第{issue.line_number}行**: {issue.message} ({issue.severity.value})\\n"
                        if issue.description:
                            report += f"  - {issue.description}\\n"
                        if issue.suggestion:
                            report += f"  - 建议: {issue.suggestion}\\n"
                    report += "\\n"
            
            return report
        
        else:
            return "不支持的报告格式"


if __name__ == "__main__":
    # 测试代码审查器
    reviewer = CodeReviewer()
    
    # 创建测试代码文件
    test_code = '''
import os
import sys

class testClass:  # 类名不符合规范
    def __init__(self):
        self.password = "123456"  # 硬编码密码
        
    def LongMethodName(self):  # 方法名不符合规范
        # 过长的行示例 - 这是一个非常长的行，超过了推荐的行长度限制，应该被拆分为多行来提高代码的可读性和维护性
        if True:
            if True:
                if True:  # 复杂度过高
                    pass
        
        query = "SELECT * FROM users WHERE id = " + str(user_id)  # SQL注入风险
        os.system("rm -rf /")  # 命令注入风险
        
        unused_variable = 42  # 未使用的变量
'''
    
    test_file = Path("test_review.py")
    
    try:
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write(test_code)
        
        # 审查文件
        issues = reviewer.review_file(test_file)
        
        # 生成报告
        report = reviewer.generate_report(issues)
        print(report)
        
        print("\\n代码审查器测试完成！")
        
    except Exception as e:
        print(f"测试失败: {e}")
    
    finally:
        # 清理测试文件
        if test_file.exists():
            test_file.unlink()