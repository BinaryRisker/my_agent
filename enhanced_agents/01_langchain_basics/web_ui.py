"""
LangChain Agent Web UI
使用Gradio创建交互式Web界面
"""

import gradio as gr
import logging
from typing import List, Tuple, Dict, Any
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_agent import LangChainSimpleAgent, MultiModelAgent
from common.model_config import ModelConfig
from common.llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class LangChainAgentUI:
    """LangChain Agent Web UI"""
    
    def __init__(self):
        self.agent = None
        self.multi_agent = None
        self.chat_history = []
        
        # 初始化多模型代理
        try:
            self.multi_agent = MultiModelAgent()
        except Exception as e:
            logger.warning(f"Failed to initialize MultiModelAgent: {e}")
    
    def initialize_agent(
        self, 
        provider: str, 
        model: str, 
        system_prompt: str,
        enable_memory: bool,
        enable_cost_tracking: bool,
        temperature: float
    ) -> str:
        """初始化Agent"""
        try:
            self.agent = LangChainSimpleAgent(
                provider=provider,
                model=model,
                system_prompt=system_prompt if system_prompt.strip() else None,
                enable_memory=enable_memory,
                enable_cost_tracking=enable_cost_tracking,
                temperature=temperature
            )
            self.chat_history = []
            return f"✅ Agent initialized successfully with {provider}:{model}"
        except Exception as e:
            logger.error(f"Failed to initialize agent: {e}")
            return f"❌ Failed to initialize agent: {str(e)}"
    
    def chat(self, message: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """聊天功能"""
        if not self.agent:
            return "Please initialize an agent first.", history
        
        if not message.strip():
            return "Please enter a message.", history
        
        try:
            response = self.agent.respond(message)
            history.append((message, response))
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def multi_model_chat(self, message: str, preferred_model: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """多模型聊天"""
        if not self.multi_agent:
            return "Multi-model agent not available.", history
        
        if not message.strip():
            return "Please enter a message.", history
        
        try:
            result = self.multi_agent.respond(
                message, 
                preferred_model if preferred_model != "Auto" else None
            )
            
            response_text = result["response"]
            model_used = result["model_used"]
            
            # 添加模型信息到响应
            full_response = f"{response_text}\n\n*Used model: {model_used}*"
            
            history.append((message, full_response))
            return "", history
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append((message, error_msg))
            return "", history
    
    def get_agent_info(self) -> str:
        """获取Agent信息"""
        if not self.agent:
            return "No agent initialized"
        
        try:
            info = self.agent.get_agent_info()
            return json.dumps(info, indent=2, ensure_ascii=False)
        except Exception as e:
            return f"Error getting agent info: {str(e)}"
    
    def get_cost_summary(self) -> str:
        """获取成本摘要"""
        if not self.agent:
            if self.multi_agent:
                try:
                    summary = self.multi_agent.get_cost_summary()
                    return json.dumps(summary, indent=2)
                except Exception as e:
                    return f"Error getting cost summary: {str(e)}"
            return "No agent initialized"
        
        try:
            summary = self.agent.get_cost_summary()
            return json.dumps(summary, indent=2)
        except Exception as e:
            return f"Error getting cost summary: {str(e)}"
    
    def clear_chat_history(self) -> List[Tuple[str, str]]:
        """清除聊天历史"""
        if self.agent:
            self.agent.clear_memory()
        self.chat_history = []
        return []
    
    def test_connection(self, provider: str, model: str) -> str:
        """测试连接"""
        try:
            result = LLMFactory.test_connection(provider, model)
            if result["success"]:
                return f"✅ Connection successful!\nResponse: {result['response_preview']}"
            else:
                return f"❌ Connection failed: {result['error']}"
        except Exception as e:
            return f"❌ Test failed: {str(e)}"
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """获取可用模型"""
        try:
            return ModelConfig.list_available_models()
        except Exception as e:
            logger.error(f"Error getting available models: {e}")
            return {}
    
    def get_available_providers(self) -> Dict[str, bool]:
        """获取可用提供商"""
        try:
            return LLMFactory.get_available_providers()
        except Exception as e:
            logger.error(f"Error getting available providers: {e}")
            return {}


def create_gradio_interface():
    """创建Gradio界面"""
    ui = LangChainAgentUI()
    
    # 获取可用模型和提供商
    available_models = ui.get_available_models()
    available_providers = ui.get_available_providers()
    
    # 准备选择项
    provider_choices = list(available_providers.keys())
    model_choices = []
    for provider, models in available_models.items():
        for model in models:
            model_choices.append(f"{provider}:{model}")
    
    # 默认系统提示词
    default_system_prompt = """You are a helpful AI assistant. Please:
- Be friendly and professional
- Provide accurate and helpful information
- Ask for clarification when needed
- Respond in the same language as the user's question"""
    
    with gr.Blocks(title="🤖 LangChain Agent Interface", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 LangChain Agent Interface")
        gr.Markdown("Experience the power of LangChain with multiple LLM providers!")
        
        with gr.Tab("🚀 Single Agent Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Agent Configuration")
                    
                    provider_dropdown = gr.Dropdown(
                        choices=provider_choices,
                        value=provider_choices[0] if provider_choices else "openai",
                        label="LLM Provider"
                    )
                    
                    model_dropdown = gr.Dropdown(
                        choices=available_models.get(provider_choices[0] if provider_choices else "openai", []),
                        value=available_models.get(provider_choices[0] if provider_choices else "openai", [""])[0] if available_models else "gpt-3.5-turbo",
                        label="Model"
                    )
                    
                    system_prompt = gr.Textbox(
                        value=default_system_prompt,
                        label="System Prompt",
                        lines=4,
                        placeholder="Enter system prompt..."
                    )
                    
                    with gr.Row():
                        enable_memory = gr.Checkbox(value=True, label="Enable Memory")
                        enable_cost_tracking = gr.Checkbox(value=True, label="Enable Cost Tracking")
                    
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    
                    with gr.Row():
                        init_btn = gr.Button("🚀 Initialize Agent", variant="primary")
                        test_btn = gr.Button("🔍 Test Connection")
                    
                    status_output = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Chat Interface")
                    
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=400,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Type your message here...",
                            label="Message",
                            scale=4
                        )
                        send_btn = gr.Button("📤 Send", scale=1, variant="primary")
                    
                    clear_btn = gr.Button("🗑️ Clear Chat")
        
        with gr.Tab("🎭 Multi-Model Chat"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Model Selection")
                    
                    multi_model_dropdown = gr.Dropdown(
                        choices=["Auto"] + model_choices,
                        value="Auto",
                        label="Preferred Model (Auto for smart selection)"
                    )
                    
                    available_models_display = gr.JSON(
                        value=model_choices,
                        label="Available Models"
                    )
                
                with gr.Column(scale=2):
                    gr.Markdown("### Multi-Model Chat")
                    
                    multi_chatbot = gr.Chatbot(
                        label="Multi-Model Conversation",
                        height=400,
                        show_copy_button=True
                    )
                    
                    with gr.Row():
                        multi_msg_input = gr.Textbox(
                            placeholder="Ask anything and I'll choose the best model...",
                            label="Message",
                            scale=4
                        )
                        multi_send_btn = gr.Button("📤 Send", scale=1, variant="primary")
                    
                    multi_clear_btn = gr.Button("🗑️ Clear Chat")
        
        with gr.Tab("📊 Agent Info & Stats"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Agent Information")
                    agent_info_output = gr.JSON(label="Agent Info")
                    
                    refresh_info_btn = gr.Button("🔄 Refresh Info")
                
                with gr.Column():
                    gr.Markdown("### Cost Summary")
                    cost_summary_output = gr.JSON(label="Cost Summary")
                    
                    refresh_cost_btn = gr.Button("💰 Refresh Cost")
            
            with gr.Row():
                gr.Markdown("### System Status")
                providers_status = gr.JSON(
                    value=available_providers,
                    label="Available Providers"
                )
        
        # 事件处理
        def update_models(provider):
            """更新模型选择"""
            models = available_models.get(provider, [])
            return gr.Dropdown(choices=models, value=models[0] if models else "")
        
        provider_dropdown.change(
            update_models,
            inputs=[provider_dropdown],
            outputs=[model_dropdown]
        )
        
        init_btn.click(
            ui.initialize_agent,
            inputs=[provider_dropdown, model_dropdown, system_prompt, enable_memory, enable_cost_tracking, temperature],
            outputs=[status_output]
        )
        
        test_btn.click(
            ui.test_connection,
            inputs=[provider_dropdown, model_dropdown],
            outputs=[status_output]
        )
        
        msg_input.submit(
            ui.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        send_btn.click(
            ui.chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot]
        )
        
        clear_btn.click(
            ui.clear_chat_history,
            outputs=[chatbot]
        )
        
        # Multi-model chat events
        multi_msg_input.submit(
            ui.multi_model_chat,
            inputs=[multi_msg_input, multi_model_dropdown, multi_chatbot],
            outputs=[multi_msg_input, multi_chatbot]
        )
        
        multi_send_btn.click(
            ui.multi_model_chat,
            inputs=[multi_msg_input, multi_model_dropdown, multi_chatbot],
            outputs=[multi_msg_input, multi_chatbot]
        )
        
        multi_clear_btn.click(
            lambda: [],
            outputs=[multi_chatbot]
        )
        
        refresh_info_btn.click(
            ui.get_agent_info,
            outputs=[agent_info_output]
        )
        
        refresh_cost_btn.click(
            ui.get_cost_summary,
            outputs=[cost_summary_output]
        )
        
        gr.Markdown("""
        ---
        ### 📖 Usage Instructions
        
        1. **Single Agent**: Configure and initialize a single agent with specific settings
        2. **Multi-Model**: Let the system automatically choose the best model for each task
        3. **Monitor**: Check agent info and cost tracking in the stats tab
        
        #### 🔑 API Keys Required
        Set environment variables for your chosen providers:
        - `OPENAI_API_KEY` for OpenAI
        - `ANTHROPIC_API_KEY` for Anthropic/Claude
        - `GOOGLE_API_KEY` for Google/Gemini
        
        #### 🛠️ Installation
        ```bash
        pip install langchain langchain-openai langchain-anthropic langchain-google-genai
        ```
        """)
    
    return interface


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 创建和启动界面
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_api=True,
        share=False
    )
