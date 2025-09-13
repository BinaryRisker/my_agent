"""
代码助手Agent主程序
集成代码分析、生成、审查等功能，提供CLI和Web界面
"""

import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

# Third-party imports
from loguru import logger
import gradio as gr

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config

from code_analyzer import CodeAnalyzer
from code_generator import CodeGenerator  
from code_reviewer import CodeReviewer


class CodeAssistantAgent:
    """代码助手Agent主类"""
    
    def __init__(self):
        """初始化代码助手Agent"""
        self.config = get_config()
        
        # 初始化各个组件
        self.analyzer = CodeAnalyzer()
        self.generator = CodeGenerator()
        self.reviewer = CodeReviewer()
        
        # 会话状态
        self.session_state = {
            'current_project': None,
            'analysis_results': {},
            'review_results': {},
            'generated_code': {}
        }
        
        logger.info("代码助手Agent初始化完成")
    
    def analyze_code(self, 
                    path: str,
                    recursive: bool = True,
                    include_tests: bool = True) -> Dict[str, Any]:
        """
        分析代码
        
        Args:
            path: 文件或目录路径
            recursive: 是否递归分析
            include_tests: 是否包含测试文件
            
        Returns:
            分析结果
        """
        try:
            path_obj = Path(path)
            
            if path_obj.is_file():
                result = self.analyzer.analyze_file(path)
            else:
                result = self.analyzer.analyze_directory(
                    path, recursive, include_tests
                )
            
            # 保存分析结果
            self.session_state['analysis_results'][path] = result
            
            logger.info(f"代码分析完成: {path}")
            return result
            
        except Exception as e:
            logger.error(f"代码分析失败: {e}")
            raise
    
    def review_code(self, 
                   path: str,
                   recursive: bool = True,
                   file_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        审查代码
        
        Args:
            path: 文件或目录路径
            recursive: 是否递归审查
            file_patterns: 文件模式列表
            
        Returns:
            审查结果
        """
        try:
            path_obj = Path(path)
            
            if path_obj.is_file():
                issues = self.reviewer.review_file(path)
                result = {path: issues}
            else:
                result = self.reviewer.review_directory(
                    path, recursive, file_patterns
                )
            
            # 保存审查结果
            self.session_state['review_results'][path] = result
            
            logger.info(f"代码审查完成: {path}")
            return result
            
        except Exception as e:
            logger.error(f"代码审查失败: {e}")
            raise
    
    def generate_unit_tests(self, 
                           source_file: str,
                           target_class: Optional[str] = None,
                           coverage_target: float = 0.8) -> str:
        """
        生成单元测试
        
        Args:
            source_file: 源代码文件
            target_class: 目标类名
            coverage_target: 目标覆盖率
            
        Returns:
            生成的测试代码
        """
        try:
            test_code = self.generator.generate_unit_tests(
                source_file, target_class, coverage_target
            )
            
            # 保存生成的代码
            key = f"test_{Path(source_file).stem}"
            self.session_state['generated_code'][key] = test_code
            
            logger.info(f"单元测试生成完成: {source_file}")
            return test_code
            
        except Exception as e:
            logger.error(f"单元测试生成失败: {e}")
            raise
    
    def generate_docstring(self, 
                          function_code: str,
                          style: str = "google") -> str:
        """
        生成文档字符串
        
        Args:
            function_code: 函数代码
            style: 文档风格
            
        Returns:
            生成的文档字符串
        """
        try:
            docstring = self.generator.generate_docstring(function_code, style)
            logger.info("文档字符串生成完成")
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
            completed_code = self.generator.complete_code(
                incomplete_code, context, language
            )
            logger.info("代码补全完成")
            return completed_code
            
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
            refactor_type: 重构类型
            parameters: 重构参数
            
        Returns:
            重构后的代码
        """
        try:
            refactored_code = self.generator.refactor_code(
                source_code, refactor_type, parameters
            )
            logger.info(f"代码重构完成: {refactor_type}")
            return refactored_code
            
        except Exception as e:
            logger.error(f"代码重构失败: {e}")
            raise
    
    def generate_analysis_report(self, path: str) -> str:
        """生成分析报告"""
        if path in self.session_state['analysis_results']:
            result = self.session_state['analysis_results'][path]
            return self.analyzer.generate_report(result)
        else:
            return "未找到分析结果，请先进行代码分析"
    
    def generate_review_report(self, path: str) -> str:
        """生成审查报告"""
        if path in self.session_state['review_results']:
            result = self.session_state['review_results'][path]
            return self.reviewer.generate_report(result)
        else:
            return "未找到审查结果，请先进行代码审查"
    
    def save_generated_code(self, key: str, file_path: str) -> bool:
        """保存生成的代码"""
        try:
            if key in self.session_state['generated_code']:
                code = self.session_state['generated_code'][key]
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(code)
                logger.info(f"代码已保存: {file_path}")
                return True
            else:
                logger.warning(f"未找到生成的代码: {key}")
                return False
        except Exception as e:
            logger.error(f"保存代码失败: {e}")
            return False
    
    def get_session_summary(self) -> Dict[str, Any]:
        """获取会话摘要"""
        return {
            'analyzed_items': len(self.session_state['analysis_results']),
            'reviewed_items': len(self.session_state['review_results']),
            'generated_code_items': len(self.session_state['generated_code']),
            'current_project': self.session_state['current_project']
        }


def create_gradio_interface(agent: CodeAssistantAgent):
    """创建Gradio Web界面"""
    
    def analyze_interface(path: str, recursive: bool, include_tests: bool):
        """分析界面"""
        try:
            result = agent.analyze_code(path, recursive, include_tests)
            report = agent.generate_analysis_report(path)
            return report, json.dumps(result, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"分析失败: {e}", ""
    
    def review_interface(path: str, recursive: bool):
        """审查界面"""
        try:
            result = agent.review_code(path, recursive)
            report = agent.generate_review_report(path)
            return report, json.dumps({k: len(v) for k, v in result.items()}, 
                                    indent=2, ensure_ascii=False)
        except Exception as e:
            return f"审查失败: {e}", ""
    
    def generate_tests_interface(source_file: str, target_class: str, coverage: float):
        """测试生成界面"""
        try:
            target_class = target_class if target_class.strip() else None
            test_code = agent.generate_unit_tests(source_file, target_class, coverage)
            return test_code
        except Exception as e:
            return f"测试生成失败: {e}"
    
    def generate_docstring_interface(function_code: str, style: str):
        """文档字符串生成界面"""
        try:
            docstring = agent.generate_docstring(function_code, style)
            return docstring
        except Exception as e:
            return f"文档字符串生成失败: {e}"
    
    def complete_code_interface(incomplete_code: str, context: str, language: str):
        """代码补全界面"""
        try:
            context = context if context.strip() else None
            completed_code = agent.complete_code(incomplete_code, context, language)
            return completed_code
        except Exception as e:
            return f"代码补全失败: {e}"
    
    # 创建Gradio界面
    with gr.Blocks(title="代码助手Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 代码助手Agent")
        gr.Markdown("提供代码分析、审查、生成等智能化功能")
        
        with gr.Tabs():
            # 代码分析标签页
            with gr.TabItem("📊 代码分析"):
                with gr.Row():
                    with gr.Column(scale=2):
                        analyze_path = gr.Textbox(
                            label="代码路径",
                            placeholder="输入文件或目录路径",
                            lines=1
                        )
                        with gr.Row():
                            analyze_recursive = gr.Checkbox(label="递归分析", value=True)
                            analyze_include_tests = gr.Checkbox(label="包含测试文件", value=True)
                        analyze_btn = gr.Button("🔍 开始分析", variant="primary")
                    
                    with gr.Column(scale=3):
                        analyze_report = gr.Textbox(
                            label="分析报告",
                            lines=20,
                            max_lines=20
                        )
                
                with gr.Row():
                    analyze_details = gr.Textbox(
                        label="详细结果 (JSON)",
                        lines=10,
                        max_lines=10
                    )
                
                analyze_btn.click(
                    analyze_interface,
                    inputs=[analyze_path, analyze_recursive, analyze_include_tests],
                    outputs=[analyze_report, analyze_details]
                )
            
            # 代码审查标签页
            with gr.TabItem("🔎 代码审查"):
                with gr.Row():
                    with gr.Column(scale=2):
                        review_path = gr.Textbox(
                            label="代码路径",
                            placeholder="输入文件或目录路径",
                            lines=1
                        )
                        review_recursive = gr.Checkbox(label="递归审查", value=True)
                        review_btn = gr.Button("🔍 开始审查", variant="primary")
                    
                    with gr.Column(scale=3):
                        review_report = gr.Textbox(
                            label="审查报告",
                            lines=20,
                            max_lines=20
                        )
                
                with gr.Row():
                    review_summary = gr.Textbox(
                        label="问题统计",
                        lines=5,
                        max_lines=5
                    )
                
                review_btn.click(
                    review_interface,
                    inputs=[review_path, review_recursive],
                    outputs=[review_report, review_summary]
                )
            
            # 单元测试生成标签页
            with gr.TabItem("🧪 测试生成"):
                with gr.Row():
                    with gr.Column():
                        test_source_file = gr.Textbox(
                            label="源代码文件",
                            placeholder="输入Python文件路径",
                            lines=1
                        )
                        test_target_class = gr.Textbox(
                            label="目标类名 (可选)",
                            placeholder="留空则为所有类生成测试",
                            lines=1
                        )
                        test_coverage = gr.Slider(
                            label="目标覆盖率",
                            minimum=0.5,
                            maximum=1.0,
                            value=0.8,
                            step=0.1
                        )
                        generate_test_btn = gr.Button("🧪 生成测试", variant="primary")
                    
                    with gr.Column():
                        generated_test_code = gr.Textbox(
                            label="生成的测试代码",
                            lines=25,
                            max_lines=25
                        )
                
                generate_test_btn.click(
                    generate_tests_interface,
                    inputs=[test_source_file, test_target_class, test_coverage],
                    outputs=[generated_test_code]
                )
            
            # 文档生成标签页
            with gr.TabItem("📝 文档生成"):
                with gr.Row():
                    with gr.Column():
                        doc_function_code = gr.Textbox(
                            label="函数代码",
                            placeholder="粘贴函数定义",
                            lines=10
                        )
                        doc_style = gr.Dropdown(
                            label="文档风格",
                            choices=["google", "numpy", "sphinx"],
                            value="google"
                        )
                        generate_doc_btn = gr.Button("📝 生成文档", variant="primary")
                    
                    with gr.Column():
                        generated_docstring = gr.Textbox(
                            label="生成的文档字符串",
                            lines=15,
                            max_lines=15
                        )
                
                generate_doc_btn.click(
                    generate_docstring_interface,
                    inputs=[doc_function_code, doc_style],
                    outputs=[generated_docstring]
                )
            
            # 代码补全标签页
            with gr.TabItem("⚡ 代码补全"):
                with gr.Row():
                    with gr.Column():
                        incomplete_code = gr.Textbox(
                            label="不完整代码",
                            placeholder="输入需要补全的代码",
                            lines=10
                        )
                        completion_context = gr.Textbox(
                            label="上下文 (可选)",
                            placeholder="提供相关上下文信息",
                            lines=3
                        )
                        completion_language = gr.Dropdown(
                            label="编程语言",
                            choices=["python", "javascript", "java", "cpp", "go"],
                            value="python"
                        )
                        complete_btn = gr.Button("⚡ 补全代码", variant="primary")
                    
                    with gr.Column():
                        completed_code = gr.Textbox(
                            label="补全后的代码",
                            lines=15,
                            max_lines=15
                        )
                
                complete_btn.click(
                    complete_code_interface,
                    inputs=[incomplete_code, completion_context, completion_language],
                    outputs=[completed_code]
                )
            
            # 会话状态标签页
            with gr.TabItem("📊 会话状态"):
                with gr.Row():
                    session_info = gr.JSON(
                        label="会话信息",
                        value=agent.get_session_summary()
                    )
                
                refresh_btn = gr.Button("🔄 刷新状态")
                refresh_btn.click(
                    lambda: agent.get_session_summary(),
                    outputs=[session_info]
                )
    
    return interface


def run_cli():
    """运行CLI模式"""
    parser = argparse.ArgumentParser(description="代码助手Agent CLI")
    parser.add_argument("command", choices=['analyze', 'review', 'generate-test', 'complete'],
                       help="执行的命令")
    parser.add_argument("--path", required=True, help="文件或目录路径")
    parser.add_argument("--recursive", action="store_true", help="递归处理")
    parser.add_argument("--output", help="输出文件路径")
    parser.add_argument("--format", choices=['markdown', 'json'], default='markdown',
                       help="输出格式")
    
    args = parser.parse_args()
    
    # 初始化Agent
    agent = CodeAssistantAgent()
    
    try:
        if args.command == 'analyze':
            result = agent.analyze_code(args.path, args.recursive)
            report = agent.generate_analysis_report(args.path)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"分析报告已保存到: {args.output}")
            else:
                print(report)
        
        elif args.command == 'review':
            result = agent.review_code(args.path, args.recursive)
            report = agent.generate_review_report(args.path)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(report)
                print(f"审查报告已保存到: {args.output}")
            else:
                print(report)
        
        elif args.command == 'generate-test':
            test_code = agent.generate_unit_tests(args.path)
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(test_code)
                print(f"测试代码已保存到: {args.output}")
            else:
                print(test_code)
        
        else:
            print(f"未实现的命令: {args.command}")
    
    except Exception as e:
        logger.error(f"命令执行失败: {e}")
        print(f"错误: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="代码助手Agent")
    parser.add_argument("--mode", choices=['cli', 'web'], default='web',
                       help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="Web服务器主机")
    parser.add_argument("--port", type=int, default=7864, help="Web服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        run_cli()
    else:
        # 初始化Agent
        agent = CodeAssistantAgent()
        
        # 创建Web界面
        interface = create_gradio_interface(agent)
        
        # 启动服务器
        logger.info(f"启动Web服务器: http://{args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )


if __name__ == "__main__":
    main()