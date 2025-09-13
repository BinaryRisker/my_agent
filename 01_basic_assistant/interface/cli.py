"""
命令行界面模块
提供交互式命令行体验
"""

import os
import sys
from typing import TYPE_CHECKING
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.syntax import Syntax
from loguru import logger

if TYPE_CHECKING:
    from main import BasicAssistant


class CLIInterface:
    """命令行界面"""
    
    def __init__(self, assistant: 'BasicAssistant'):
        """
        初始化CLI界面
        
        Args:
            assistant: 基础助手实例
        """
        self.assistant = assistant
        self.console = Console()
        self.running = False
        
        # 支持的命令
        self.commands = {
            '/help': '显示帮助信息',
            '/clear': '清空对话记忆',
            '/memory': '显示内存使用情况',
            '/save': '保存对话到文件',
            '/load': '从文件加载对话',
            '/tools': '显示可用工具',
            '/config': '显示当前配置',
            '/exit': '退出程序',
            '/quit': '退出程序',
        }
    
    def start(self):
        """启动CLI界面"""
        self.running = True
        self._show_welcome()
        
        try:
            while self.running:
                self._handle_input()
        except KeyboardInterrupt:
            self._handle_exit()
        except Exception as e:
            logger.error(f"CLI运行出错: {e}")
            self.console.print(f"[red]程序出现错误: {e}[/red]")
    
    def _show_welcome(self):
        """显示欢迎信息"""
        welcome_text = Text()
        welcome_text.append("🤖 基础智能助手 ", style="bold blue")
        welcome_text.append("- LangChain Agent 学习项目\\n", style="bold")
        welcome_text.append("输入消息与我对话，或使用命令（如 /help）获取帮助", style="dim")
        
        welcome_panel = Panel(
            welcome_text,
            title="[bold green]欢迎使用[/bold green]",
            border_style="green"
        )
        
        self.console.print(welcome_panel)
        self.console.print()
    
    def _handle_input(self):
        """处理用户输入"""
        try:
            user_input = Prompt.ask(
                "[bold blue]你[/bold blue]",
                console=self.console
            ).strip()
            
            if not user_input:
                return
            
            # 检查是否是命令
            if user_input.startswith('/'):
                self._handle_command(user_input)
            else:
                # 普通对话
                self._handle_chat(user_input)
                
        except (EOFError, KeyboardInterrupt):
            self._handle_exit()
    
    def _handle_command(self, command: str):
        """处理命令"""
        command = command.lower().strip()
        
        if command in ['/exit', '/quit']:
            self._handle_exit()
        
        elif command == '/help':
            self._show_help()
        
        elif command == '/clear':
            self._handle_clear()
        
        elif command == '/memory':
            self._show_memory_info()
        
        elif command == '/save':
            self._handle_save()
        
        elif command == '/load':
            self._handle_load()
        
        elif command == '/tools':
            self._show_tools()
        
        elif command == '/config':
            self._show_config()
        
        else:
            self.console.print(f"[red]未知命令: {command}[/red]")
            self.console.print("输入 [bold]/help[/bold] 查看可用命令")
    
    def _handle_chat(self, user_input: str):
        """处理聊天输入"""
        try:
            # 显示用户输入
            self.console.print()
            
            # 显示"正在思考"提示
            with self.console.status("[bold green]助手正在思考...[/bold green]"):
                response = self.assistant.chat(user_input)
            
            # 显示助手回复
            self._show_assistant_response(response)
            
        except Exception as e:
            logger.error(f"聊天处理出错: {e}")
            self.console.print(f"[red]抱歉，处理您的消息时出现了错误: {e}[/red]")
    
    def _show_assistant_response(self, response: str):
        """显示助手回复"""
        # 创建回复面板
        response_panel = Panel(
            response,
            title="[bold green]🤖 助手[/bold green]",
            border_style="green",
            padding=(1, 2)
        )
        
        self.console.print(response_panel)
        self.console.print()
    
    def _show_help(self):
        """显示帮助信息"""
        help_table = Table(title="可用命令", show_header=True, header_style="bold blue")
        help_table.add_column("命令", style="cyan", min_width=10)
        help_table.add_column("说明", style="white")
        
        for cmd, desc in self.commands.items():
            help_table.add_row(cmd, desc)
        
        self.console.print(help_table)
        self.console.print()
        
        # 显示工具使用提示
        tools_info = Text()
        tools_info.append("💡 工具使用提示:\\n", style="bold yellow")
        tools_info.append("• 数学计算: '计算 2+3*4'\\n")
        tools_info.append("• 天气查询: '北京天气如何？'\\n")
        tools_info.append("• 文件操作: '读取文件 data.txt' 或 '写入文件 notes.txt|内容'\\n")
        
        self.console.print(Panel(tools_info, border_style="yellow"))
        self.console.print()
    
    def _handle_clear(self):
        """处理清空记忆命令"""
        if Confirm.ask("确定要清空对话记忆吗？", default=False):
            self.assistant.clear_memory()
            self.console.print("[green]✅ 对话记忆已清空[/green]")
        else:
            self.console.print("[yellow]取消操作[/yellow]")
        self.console.print()
    
    def _show_memory_info(self):
        """显示内存信息"""
        memory_info = self.assistant.get_memory_info()
        
        info_table = Table(title="内存使用情况", show_header=True, header_style="bold blue")
        info_table.add_column("项目", style="cyan")
        info_table.add_column("值", style="white")
        
        info_table.add_row("总消息数", str(memory_info.get('total_messages', 0)))
        info_table.add_row("用户消息", str(memory_info.get('user_messages', 0)))
        info_table.add_row("助手回复", str(memory_info.get('ai_messages', 0)))
        info_table.add_row("总字符数", str(memory_info.get('total_characters', 0)))
        info_table.add_row("内存类型", memory_info.get('memory_type', 'unknown'))
        
        self.console.print(info_table)
        self.console.print()
    
    def _handle_save(self):
        """处理保存对话命令"""
        try:
            file_path = Prompt.ask(
                "[cyan]请输入保存文件路径[/cyan]",
                default="conversation.txt"
            )
            
            self.assistant.save_conversation(file_path)
            self.console.print(f"[green]✅ 对话已保存到: {file_path}[/green]")
            
        except Exception as e:
            self.console.print(f"[red]保存失败: {e}[/red]")
        
        self.console.print()
    
    def _handle_load(self):
        """处理加载对话命令"""
        try:
            file_path = Prompt.ask("[cyan]请输入文件路径[/cyan]")
            
            if not os.path.exists(file_path):
                self.console.print(f"[red]文件不存在: {file_path}[/red]")
                return
            
            self.assistant.load_conversation(file_path)
            self.console.print(f"[green]✅ 对话已从 {file_path} 加载[/green]")
            
        except Exception as e:
            self.console.print(f"[red]加载失败: {e}[/red]")
        
        self.console.print()
    
    def _show_tools(self):
        """显示可用工具"""
        if not self.assistant.tools:
            self.console.print("[yellow]没有可用的工具[/yellow]")
            return
        
        tools_table = Table(title="可用工具", show_header=True, header_style="bold blue")
        tools_table.add_column("工具名称", style="cyan", min_width=15)
        tools_table.add_column("功能描述", style="white")
        
        for tool in self.assistant.tools:
            # 简化描述，只显示第一行
            desc = tool.description.strip().split('\\n')[0]
            tools_table.add_row(tool.name, desc)
        
        self.console.print(tools_table)
        self.console.print()
    
    def _show_config(self):
        """显示当前配置"""
        config = self.assistant.config
        
        config_info = f\"\"\"🔧 当前配置信息：

[bold cyan]LLM 配置:[/bold cyan]
  模型: {config.llm.model}
  温度: {config.llm.temperature}
  最大令牌: {config.llm.max_tokens}
  超时: {config.llm.timeout}秒

[bold cyan]内存配置:[/bold cyan]
  类型: {config.memory.type}
  最大令牌限制: {config.memory.max_token_limit}
  返回消息: {config.memory.return_messages}

[bold cyan]安全配置:[/bold cyan]
  最大输入长度: {config.security.max_input_length}
  速率限制: {config.security.max_requests_per_minute}/分钟

[bold cyan]日志配置:[/bold cyan]
  级别: {config.logging.level}
  文件: {config.logging.file_path}\"\"\"
        
        self.console.print(Panel(config_info, border_style="blue"))
        self.console.print()
    
    def _handle_exit(self):
        """处理退出"""
        if Confirm.ask("确定要退出吗？", default=True):
            self.console.print("[yellow]👋 再见！感谢使用基础智能助手[/yellow]")
            self.running = False
            sys.exit(0)
        else:
            self.console.print("[green]继续对话...[/green]")
            self.console.print()


if __name__ == "__main__":
    # CLI测试代码
    from unittest.mock import Mock
    
    # 创建模拟的助手实例
    mock_assistant = Mock()
    mock_assistant.chat.return_value = "这是一个测试回复"
    mock_assistant.get_memory_info.return_value = {
        'total_messages': 4,
        'user_messages': 2,
        'ai_messages': 2,
        'total_characters': 100,
        'memory_type': 'buffer'
    }
    mock_assistant.tools = []
    mock_assistant.config = Mock()
    mock_assistant.config.llm.model = "gpt-3.5-turbo"
    mock_assistant.config.llm.temperature = 0.7
    
    # 测试CLI界面
    cli = CLIInterface(mock_assistant)
    print("CLI界面测试 - 输入 '/exit' 退出")
    
    try:
        cli.start()
    except KeyboardInterrupt:
        print("\\n测试结束")