"""
Web界面模块
使用Gradio提供Web界面
"""

import gradio as gr
import json
from datetime import datetime
from typing import TYPE_CHECKING, List, Tuple, Any
from loguru import logger

if TYPE_CHECKING:
    from main import BasicAssistant


class WebInterface:
    """Web界面"""
    
    def __init__(self, assistant: 'BasicAssistant'):
        """
        初始化Web界面
        
        Args:
            assistant: 基础助手实例
        """
        self.assistant = assistant
        self.config = assistant.config
        
        # 聊天历史（用于显示）
        self.chat_history: List[Tuple[str, str]] = []
        
        # 创建Gradio界面
        self.interface = self._create_interface()
    
    def _create_interface(self) -> gr.Interface:
        """创建Gradio界面"""
        # 自定义CSS样式
        custom_css = """
        .gradio-container {
            max-width: 1200px !important;
            margin: auto !important;
        }
        .chat-message {
            padding: 10px;
            margin: 5px 0;
            border-radius: 10px;
        }
        .user-message {
            background-color: #e3f2fd;
            text-align: right;
        }
        .assistant-message {
            background-color: #f1f8e9;
            text-align: left;
        }
        """
        
        # 使用Blocks API创建更复杂的界面
        with gr.Blocks(
            css=custom_css,
            title=self.config.web_ui.title,
            theme=gr.themes.Soft()
        ) as interface:
            # 页面标题
            gr.Markdown(f"# {self.config.web_ui.title}")
            gr.Markdown(f"*{self.config.web_ui.description}*")
            
            # 主要聊天区域
            with gr.Row():
                with gr.Column(scale=4):
                    # 聊天历史显示
                    chatbot = gr.Chatbot(
                        label="对话记录",
                        height=500,
                        show_label=True
                    )
                    
                    # 用户输入区域
                    with gr.Row():
                        user_input = gr.Textbox(
                            label="",
                            placeholder="在此输入您的消息...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("发送", variant="primary", scale=1)
                
                # 侧边栏
                with gr.Column(scale=1):
                    gr.Markdown("### 🛠️ 控制面板")
                    
                    # 内存控制
                    with gr.Group():
                        gr.Markdown("**内存管理**")
                        memory_info = gr.JSON(label="内存状态", value={})
                        clear_btn = gr.Button("清空记忆", variant="secondary")
                        refresh_memory_btn = gr.Button("刷新状态", variant="secondary")
                    
                    # 对话管理
                    with gr.Group():
                        gr.Markdown("**对话管理**")
                        save_path = gr.Textbox(
                            label="保存路径",
                            value="conversation.txt",
                            lines=1
                        )
                        save_btn = gr.Button("保存对话", variant="secondary")
                        load_path = gr.Textbox(
                            label="加载路径",
                            value="conversation.txt",
                            lines=1
                        )
                        load_btn = gr.Button("加载对话", variant="secondary")
                    
                    # 工具信息
                    with gr.Group():
                        gr.Markdown("**可用工具**")
                        tools_info = gr.JSON(
                            label="工具列表",
                            value=self._get_tools_info()
                        )
            
            # 状态消息
            status_msg = gr.Textbox(label="状态", visible=False)
            
            # 设置事件处理
            self._setup_events(
                interface=interface,
                chatbot=chatbot,
                user_input=user_input,
                send_btn=send_btn,
                clear_btn=clear_btn,
                refresh_memory_btn=refresh_memory_btn,
                save_btn=save_btn,
                load_btn=load_btn,
                save_path=save_path,
                load_path=load_path,
                memory_info=memory_info,
                status_msg=status_msg
            )
            
            # 初始化内存状态显示
            interface.load(
                fn=self._get_memory_info,
                outputs=[memory_info]
            )
        
        return interface
    
    def _setup_events(self, **components):
        """设置事件处理器"""
        interface = components['interface']
        chatbot = components['chatbot']
        user_input = components['user_input']
        send_btn = components['send_btn']
        clear_btn = components['clear_btn']
        refresh_memory_btn = components['refresh_memory_btn']
        save_btn = components['save_btn']
        load_btn = components['load_btn']
        save_path = components['save_path']
        load_path = components['load_path']
        memory_info = components['memory_info']
        status_msg = components['status_msg']
        
        # 发送消息
        def handle_message(message, history):
            return self._handle_chat(message, history)
        
        # 绑定发送事件
        send_btn.click(
            fn=handle_message,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        )
        
        # 支持回车发送
        user_input.submit(
            fn=handle_message,
            inputs=[user_input, chatbot],
            outputs=[chatbot, user_input]
        )
        
        # 清空记忆
        clear_btn.click(
            fn=self._clear_memory,
            outputs=[status_msg, memory_info]
        )
        
        # 刷新内存状态
        refresh_memory_btn.click(
            fn=self._get_memory_info,
            outputs=[memory_info]
        )
        
        # 保存对话
        save_btn.click(
            fn=self._save_conversation,
            inputs=[save_path],
            outputs=[status_msg]
        )
        
        # 加载对话
        load_btn.click(
            fn=self._load_conversation,
            inputs=[load_path],
            outputs=[status_msg, chatbot, memory_info]
        )
    
    def _handle_chat(self, message: str, history: List[List[str]]) -> Tuple[List[List[str]], str]:
        """
        处理聊天消息
        
        Args:
            message: 用户消息
            history: 聊天历史
            
        Returns:
            更新后的历史记录和清空的输入框
        """
        if not message.strip():
            return history, ""
        
        try:
            logger.info(f"Web界面收到用户消息: {message}")
            
            # 获取助手回复
            response = self.assistant.chat(message)
            
            # 更新历史记录
            if history is None:
                history = []
            
            history.append([message, response])
            
            # 限制历史记录长度
            max_history = self.config.web_ui.max_chat_history
            if len(history) > max_history:
                history = history[-max_history:]
            
            logger.info(f"Web界面助手回复长度: {len(response)}")
            
            return history, ""
            
        except Exception as e:
            logger.error(f"Web界面聊天处理出错: {e}")
            error_msg = f"抱歉，处理您的消息时出现了错误: {str(e)}"
            
            if history is None:
                history = []
            history.append([message, error_msg])
            
            return history, ""
    
    def _get_memory_info(self) -> dict:
        """获取内存信息"""
        try:
            memory_info = self.assistant.get_memory_info()
            memory_info['last_updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return memory_info
        except Exception as e:
            logger.error(f"获取内存信息失败: {e}")
            return {"error": str(e)}
    
    def _clear_memory(self) -> Tuple[str, dict]:
        """清空记忆"""
        try:
            self.assistant.clear_memory()
            status = "✅ 对话记忆已清空"
            memory_info = self._get_memory_info()
            logger.info("Web界面: 内存已清空")
            return status, memory_info
        except Exception as e:
            logger.error(f"清空内存失败: {e}")
            return f"❌ 清空失败: {str(e)}", {}
    
    def _save_conversation(self, file_path: str) -> str:
        """保存对话"""
        try:
            if not file_path.strip():
                return "❌ 请提供有效的文件路径"
            
            self.assistant.save_conversation(file_path)
            logger.info(f"Web界面: 对话已保存到 {file_path}")
            return f"✅ 对话已保存到: {file_path}"
        
        except Exception as e:
            logger.error(f"保存对话失败: {e}")
            return f"❌ 保存失败: {str(e)}"
    
    def _load_conversation(self, file_path: str) -> Tuple[str, List[List[str]], dict]:
        """加载对话"""
        try:
            if not file_path.strip():
                return "❌ 请提供有效的文件路径", [], {}
            
            self.assistant.load_conversation(file_path)
            
            # 重建聊天历史显示
            history = self._rebuild_chat_history()
            memory_info = self._get_memory_info()
            
            logger.info(f"Web界面: 对话已从 {file_path} 加载")
            return f"✅ 对话已从 {file_path} 加载", history, memory_info
        
        except Exception as e:
            logger.error(f"加载对话失败: {e}")
            return f"❌ 加载失败: {str(e)}", [], {}
    
    def _rebuild_chat_history(self) -> List[List[str]]:
        """重建聊天历史显示"""
        try:
            # 获取内存中的消息
            messages = self.assistant.memory.get_chat_history()
            history = []
            
            # 将消息转换为聊天历史格式
            current_pair = [None, None]
            for message in messages:
                if hasattr(message, 'content'):
                    if message.__class__.__name__ == 'HumanMessage':
                        if current_pair[0] is not None:
                            # 保存上一对对话
                            if current_pair[1] is not None:
                                history.append([current_pair[0], current_pair[1]])
                        current_pair = [message.content, None]
                    elif message.__class__.__name__ == 'AIMessage':
                        if current_pair[0] is not None:
                            current_pair[1] = message.content
                            history.append([current_pair[0], current_pair[1]])
                            current_pair = [None, None]
            
            return history
        
        except Exception as e:
            logger.error(f"重建聊天历史失败: {e}")
            return []
    
    def _get_tools_info(self) -> dict:
        """获取工具信息"""
        try:
            if not self.assistant.tools:
                return {"message": "没有可用的工具"}
            
            tools_info = {}
            for tool in self.assistant.tools:
                # 获取工具描述的第一行
                desc_lines = tool.description.strip().split('\\n')
                short_desc = desc_lines[0] if desc_lines else "无描述"
                
                tools_info[tool.name] = {
                    "description": short_desc,
                    "full_description": tool.description
                }
            
            return tools_info
        
        except Exception as e:
            logger.error(f"获取工具信息失败: {e}")
            return {"error": str(e)}
    
    def start(self, host: str = "127.0.0.1", port: int = 7860, share: bool = False):
        """
        启动Web服务器
        
        Args:
            host: 主机地址
            port: 端口号
            share: 是否创建公共链接
        """
        logger.info(f"启动Web界面 - {host}:{port}")
        
        try:
            self.interface.launch(
                server_name=host,
                server_port=port,
                share=share,
                show_error=True,
                quiet=False
            )
        except Exception as e:
            logger.error(f"启动Web服务器失败: {e}")
            raise


if __name__ == "__main__":
    # Web界面测试代码
    from unittest.mock import Mock
    import time
    
    # 创建模拟的助手实例
    mock_assistant = Mock()
    mock_assistant.chat.return_value = f"这是一个测试回复 - {datetime.now()}"
    mock_assistant.get_memory_info.return_value = {
        'total_messages': 4,
        'user_messages': 2,
        'ai_messages': 2,
        'total_characters': 100,
        'memory_type': 'buffer'
    }
    mock_assistant.tools = []
    
    # 模拟配置
    mock_config = Mock()
    mock_config.web_ui.title = "基础智能助手测试"
    mock_config.web_ui.description = "LangChain Agent 测试界面"
    mock_config.web_ui.max_chat_history = 50
    mock_assistant.config = mock_config
    
    # 模拟内存
    mock_memory = Mock()
    mock_memory.get_chat_history.return_value = []
    mock_assistant.memory = mock_memory
    
    # 创建并启动Web界面
    web = WebInterface(mock_assistant)
    print("启动Web界面测试...")
    print("打开浏览器访问: http://127.0.0.1:7860")
    
    try:
        web.start(host="127.0.0.1", port=7860)
    except KeyboardInterrupt:
        print("\\n测试结束")