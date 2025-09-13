"""
Multi-Agent System - 主程序入口
整合所有阶段的Agent，提供统一的管理和调度界面
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import threading

# Third-party imports
from loguru import logger
import gradio as gr

# 添加项目路径
sys.path.append(str(Path(__file__).parent))
from common.config import get_config

# 动态导入各阶段Agent
def import_agent_modules():
    """动态导入Agent模块"""
    agents = {}
    
    try:
        # 阶段1: 简单响应Agent
        sys.path.append(str(Path(__file__).parent / "01_simple_response_agent" / "src"))
        from main import SimpleResponseAgent
        agents['simple_response'] = SimpleResponseAgent
        logger.info("阶段1 简单响应Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段1 简单响应Agent 加载失败: {e}")
    
    try:
        # 阶段2: 记忆增强Agent
        sys.path.append(str(Path(__file__).parent / "02_memory_enhanced_agent" / "src"))
        from main import MemoryEnhancedAgent
        agents['memory_enhanced'] = MemoryEnhancedAgent
        logger.info("阶段2 记忆增强Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段2 记忆增强Agent 加载失败: {e}")
    
    try:
        # 阶段3: 工具使用Agent
        sys.path.append(str(Path(__file__).parent / "03_tool_using_agent" / "src"))
        from main import ToolUsingAgent
        agents['tool_using'] = ToolUsingAgent
        logger.info("阶段3 工具使用Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段3 工具使用Agent 加载失败: {e}")
    
    try:
        # 阶段4: 多工具集成Agent
        sys.path.append(str(Path(__file__).parent / "04_multi_tool_agent" / "src"))
        from main import MultiToolAgent
        agents['multi_tool'] = MultiToolAgent
        logger.info("阶段4 多工具集成Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段4 多工具集成Agent 加载失败: {e}")
    
    try:
        # 阶段5: 数据分析Agent
        sys.path.append(str(Path(__file__).parent / "05_data_analysis_agent" / "src"))
        from main import DataAnalysisAgent
        agents['data_analysis'] = DataAnalysisAgent
        logger.info("阶段5 数据分析Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段5 数据分析Agent 加载失败: {e}")
    
    try:
        # 阶段6: 自学习Agent
        sys.path.append(str(Path(__file__).parent / "06_self_learning_agent" / "src"))
        from main import SelfLearningAgent
        agents['self_learning'] = SelfLearningAgent
        logger.info("阶段6 自学习Agent 加载成功")
    except Exception as e:
        logger.warning(f"阶段6 自学习Agent 加载失败: {e}")
    
    return agents


class MultiAgentSystem:
    """多Agent系统管理器"""
    
    def __init__(self):
        """初始化多Agent系统"""
        self.config = get_config()
        self.available_agents = import_agent_modules()
        self.active_agents: Dict[str, Any] = {}
        self.system_status = {
            'total_agents': len(self.available_agents),
            'active_agents': 0,
            'system_health': 'healthy',
            'startup_time': time.strftime("%Y-%m-%d %H:%M:%S"),
            'last_activity': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        logger.info(f"多Agent系统初始化完成，可用Agent数量: {len(self.available_agents)}")
    
    def start_agent(self, agent_type: str, agent_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        启动指定类型的Agent
        
        Args:
            agent_type: Agent类型
            agent_config: Agent配置参数
            
        Returns:
            启动结果
        """
        try:
            if agent_type not in self.available_agents:
                return {
                    'success': False,
                    'message': f'未知的Agent类型: {agent_type}',
                    'available_types': list(self.available_agents.keys())
                }
            
            if agent_type in self.active_agents:
                return {
                    'success': False,
                    'message': f'Agent {agent_type} 已经在运行中'
                }
            
            # 根据Agent类型使用不同的初始化参数
            agent_class = self.available_agents[agent_type]
            
            if agent_type == 'simple_response':
                agent = agent_class(
                    name=agent_config.get('name', 'SimpleAgent'),
                    description=agent_config.get('description', 'A simple response agent')
                )
            elif agent_type == 'memory_enhanced':
                agent = agent_class(
                    memory_size=agent_config.get('memory_size', 100)
                )
            elif agent_type == 'tool_using':
                agent = agent_class()
            elif agent_type == 'multi_tool':
                agent = agent_class()
            elif agent_type == 'data_analysis':
                agent = agent_class()
            elif agent_type == 'self_learning':
                agent = agent_class(
                    knowledge_db_path=agent_config.get('db_path', 'knowledge.db'),
                    memory_size=agent_config.get('memory_size', 10000)
                )
            else:
                agent = agent_class()
            
            self.active_agents[agent_type] = agent
            self.system_status['active_agents'] = len(self.active_agents)
            self.system_status['last_activity'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Agent {agent_type} 启动成功")
            
            return {
                'success': True,
                'agent_type': agent_type,
                'message': f'Agent {agent_type} 启动成功',
                'agent_info': self._get_agent_info(agent_type)
            }
            
        except Exception as e:
            logger.error(f"启动Agent {agent_type} 失败: {e}")
            return {
                'success': False,
                'agent_type': agent_type,
                'error': str(e),
                'message': f'启动Agent失败: {e}'
            }
    
    def stop_agent(self, agent_type: str) -> Dict[str, Any]:
        """
        停止指定Agent
        
        Args:
            agent_type: Agent类型
            
        Returns:
            停止结果
        """
        try:
            if agent_type not in self.active_agents:
                return {
                    'success': False,
                    'message': f'Agent {agent_type} 未在运行中'
                }
            
            # 尝试优雅地关闭Agent
            agent = self.active_agents[agent_type]
            if hasattr(agent, 'shutdown'):
                agent.shutdown()
            
            del self.active_agents[agent_type]
            self.system_status['active_agents'] = len(self.active_agents)
            self.system_status['last_activity'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            logger.info(f"Agent {agent_type} 已停止")
            
            return {
                'success': True,
                'agent_type': agent_type,
                'message': f'Agent {agent_type} 已停止'
            }
            
        except Exception as e:
            logger.error(f"停止Agent {agent_type} 失败: {e}")
            return {
                'success': False,
                'agent_type': agent_type,
                'error': str(e),
                'message': f'停止Agent失败: {e}'
            }
    
    def _get_agent_info(self, agent_type: str) -> Dict[str, Any]:
        """获取Agent信息"""
        if agent_type not in self.active_agents:
            return {}
        
        agent = self.active_agents[agent_type]
        
        # 尝试获取Agent状态信息
        info = {
            'type': agent_type,
            'class': agent.__class__.__name__,
            'status': 'active'
        }
        
        try:
            if hasattr(agent, 'get_status'):
                status = agent.get_status()
                info.update(status)
            elif hasattr(agent, 'get_agent_status'):
                status = agent.get_agent_status()
                info.update(status)
        except Exception as e:
            logger.warning(f"获取Agent {agent_type} 状态失败: {e}")
        
        return info
    
    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        agent_details = {}
        for agent_type in self.active_agents:
            agent_details[agent_type] = self._get_agent_info(agent_type)
        
        return {
            'system_info': self.system_status,
            'available_agents': list(self.available_agents.keys()),
            'active_agents': list(self.active_agents.keys()),
            'agent_details': agent_details,
            'system_health': self._check_system_health()
        }
    
    def _check_system_health(self) -> str:
        """检查系统健康状态"""
        try:
            if len(self.active_agents) == 0:
                return 'idle'
            
            # 检查各Agent是否响应正常
            healthy_count = 0
            for agent_type, agent in self.active_agents.items():
                try:
                    # 简单的健康检查
                    if hasattr(agent, 'health_check'):
                        if agent.health_check():
                            healthy_count += 1
                    else:
                        healthy_count += 1  # 假设正常
                except:
                    pass
            
            if healthy_count == len(self.active_agents):
                return 'healthy'
            elif healthy_count > len(self.active_agents) / 2:
                return 'partial'
            else:
                return 'unhealthy'
                
        except Exception as e:
            logger.error(f"系统健康检查失败: {e}")
            return 'error'
    
    def execute_agent_task(self, agent_type: str, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        在指定Agent上执行任务
        
        Args:
            agent_type: Agent类型
            task: 任务描述
            
        Returns:
            执行结果
        """
        try:
            if agent_type not in self.active_agents:
                return {
                    'success': False,
                    'message': f'Agent {agent_type} 未在运行中，请先启动'
                }
            
            agent = self.active_agents[agent_type]
            task_type = task.get('type', 'unknown')
            task_data = task.get('data', {})
            
            # 根据Agent类型和任务类型执行不同的操作
            if agent_type == 'simple_response' and task_type == 'respond':
                if hasattr(agent, 'respond'):
                    result = agent.respond(task_data.get('input', ''))
                else:
                    result = "简单响应Agent已启动"
                    
            elif agent_type == 'memory_enhanced' and task_type == 'chat':
                if hasattr(agent, 'chat'):
                    result = agent.chat(
                        task_data.get('message', ''),
                        task_data.get('session_id', 'default')
                    )
                else:
                    result = "记忆增强Agent已启动"
                    
            elif agent_type == 'data_analysis' and task_type == 'analyze':
                # 数据分析任务示例
                result = "数据分析Agent任务执行完成"
                
            elif agent_type == 'self_learning' and task_type == 'learn':
                if hasattr(agent, 'perform_learning_session'):
                    result = agent.perform_learning_session()
                else:
                    result = "自学习Agent学习会话启动"
                    
            else:
                result = f"Agent {agent_type} 收到任务: {task_type}"
            
            self.system_status['last_activity'] = time.strftime("%Y-%m-%d %H:%M:%S")
            
            return {
                'success': True,
                'agent_type': agent_type,
                'task_type': task_type,
                'result': result,
                'message': '任务执行完成'
            }
            
        except Exception as e:
            logger.error(f"执行Agent任务失败: {e}")
            return {
                'success': False,
                'agent_type': agent_type,
                'task_type': task.get('type', 'unknown'),
                'error': str(e),
                'message': f'任务执行失败: {e}'
            }


def create_gradio_interface(system: MultiAgentSystem):
    """创建Gradio Web界面"""
    
    def start_agent_interface(agent_type: str, config_json: str):
        """启动Agent界面"""
        try:
            config = {}
            if config_json.strip():
                config = json.loads(config_json)
            
            result = system.start_agent(agent_type, config)
            
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ {result['message']}"
                
        except json.JSONDecodeError as e:
            return f"❌ 配置JSON格式错误: {e}"
        except Exception as e:
            return f"❌ 启动失败: {e}"
    
    def stop_agent_interface(agent_type: str):
        """停止Agent界面"""
        try:
            result = system.stop_agent(agent_type)
            
            if result['success']:
                return f"✅ {result['message']}"
            else:
                return f"❌ {result['message']}"
                
        except Exception as e:
            return f"❌ 停止失败: {e}"
    
    def execute_task_interface(agent_type: str, task_type: str, task_data_json: str):
        """执行任务界面"""
        try:
            task_data = {}
            if task_data_json.strip():
                task_data = json.loads(task_data_json)
            
            task = {
                'type': task_type,
                'data': task_data
            }
            
            result = system.execute_agent_task(agent_type, task)
            
            if result['success']:
                return f"✅ {result['message']}\n\n📊 执行结果:\n{json.dumps(result['result'], indent=2, ensure_ascii=False, default=str)}"
            else:
                return f"❌ {result['message']}"
                
        except json.JSONDecodeError as e:
            return f"❌ 任务数据JSON格式错误: {e}"
        except Exception as e:
            return f"❌ 任务执行失败: {e}"
    
    def get_system_status_interface():
        """获取系统状态界面"""
        status = system.get_system_status()
        return json.dumps(status, indent=2, ensure_ascii=False, default=str)
    
    # 创建Gradio界面
    with gr.Blocks(title="多Agent系统", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🤖 多Agent系统管理平台")
        gr.Markdown("统一管理和调度多个智能Agent，支持6个不同发展阶段的Agent")
        
        with gr.Tabs():
            # Agent管理标签页
            with gr.TabItem("🎛️ Agent管理"):
                gr.Markdown("### 启动Agent")
                with gr.Row():
                    with gr.Column():
                        start_agent_type = gr.Dropdown(
                            label="选择Agent类型",
                            choices=list(system.available_agents.keys()),
                            value=list(system.available_agents.keys())[0] if system.available_agents else None
                        )
                        start_config_json = gr.Textbox(
                            label="Agent配置（JSON格式，可选）",
                            placeholder='{"name": "MyAgent", "memory_size": 100}',
                            lines=3
                        )
                        start_agent_btn = gr.Button("🚀 启动Agent", variant="primary")
                    
                    with gr.Column():
                        start_agent_result = gr.Textbox(
                            label="启动结果",
                            lines=5,
                            interactive=False
                        )
                
                start_agent_btn.click(
                    start_agent_interface,
                    inputs=[start_agent_type, start_config_json],
                    outputs=[start_agent_result]
                )
                
                gr.Markdown("### 停止Agent")
                with gr.Row():
                    with gr.Column():
                        stop_agent_type = gr.Dropdown(
                            label="选择要停止的Agent",
                            choices=[],
                            interactive=True
                        )
                        stop_agent_btn = gr.Button("⏹️ 停止Agent", variant="secondary")
                    
                    with gr.Column():
                        stop_agent_result = gr.Textbox(
                            label="停止结果",
                            lines=3,
                            interactive=False
                        )
                
                stop_agent_btn.click(
                    stop_agent_interface,
                    inputs=[stop_agent_type],
                    outputs=[stop_agent_result]
                )
            
            # 任务执行标签页
            with gr.TabItem("⚡ 任务执行"):
                with gr.Row():
                    with gr.Column():
                        execute_agent_type = gr.Dropdown(
                            label="选择Agent",
                            choices=[],
                            interactive=True
                        )
                        execute_task_type = gr.Dropdown(
                            label="任务类型",
                            choices=["respond", "chat", "analyze", "learn", "optimize", "predict"],
                            value="respond"
                        )
                        execute_task_data = gr.Textbox(
                            label="任务数据（JSON格式）",
                            placeholder='{"input": "Hello", "session_id": "user1"}',
                            lines=5
                        )
                        execute_task_btn = gr.Button("▶️ 执行任务", variant="primary")
                    
                    with gr.Column():
                        execute_task_result = gr.Textbox(
                            label="执行结果",
                            lines=15,
                            interactive=False
                        )
                
                execute_task_btn.click(
                    execute_task_interface,
                    inputs=[execute_agent_type, execute_task_type, execute_task_data],
                    outputs=[execute_task_result]
                )
            
            # 系统状态标签页
            with gr.TabItem("📊 系统状态"):
                with gr.Row():
                    with gr.Column():
                        refresh_status_btn = gr.Button("🔄 刷新状态", variant="primary")
                        
                        gr.Markdown("### 系统信息")
                        gr.Markdown(f"""
                        - **可用Agent类型**: {len(system.available_agents)}个
                        - **已加载模块**: {', '.join(system.available_agents.keys())}
                        - **系统启动时间**: {system.system_status['startup_time']}
                        """)
                    
                    with gr.Column():
                        system_status_display = gr.Textbox(
                            label="系统详细状态",
                            lines=25,
                            interactive=False
                        )
                
                refresh_status_btn.click(
                    get_system_status_interface,
                    outputs=[system_status_display]
                )
            
            # Agent介绍标签页
            with gr.TabItem("📖 Agent介绍"):
                gr.Markdown("""
                ## 🎯 Agent发展阶段介绍
                
                ### 阶段1: 简单响应Agent
                - **功能**: 基础的输入输出处理
                - **特点**: 简单直接的响应机制
                - **适用场景**: 简单问答、基础交互
                
                ### 阶段2: 记忆增强Agent  
                - **功能**: 具备短期和长期记忆能力
                - **特点**: 上下文感知、会话连续性
                - **适用场景**: 多轮对话、个性化交互
                
                ### 阶段3: 工具使用Agent
                - **功能**: 能够调用外部工具和API
                - **特点**: 扩展能力、实用功能
                - **适用场景**: 实际任务执行、信息检索
                
                ### 阶段4: 多工具集成Agent
                - **功能**: 统一管理和调度多种工具
                - **特点**: 复杂任务分解、并行处理
                - **适用场景**: 复合任务、工作流自动化
                
                ### 阶段5: 数据分析Agent
                - **功能**: 数据处理、分析和可视化
                - **特点**: 机器学习、统计分析
                - **适用场景**: 数据科学、业务分析
                
                ### 阶段6: 自学习Agent
                - **功能**: 知识管理、经验学习、自我优化
                - **特点**: 持续改进、适应性学习
                - **适用场景**: 智能决策、自主优化
                
                ## 🔧 使用说明
                
                1. **启动Agent**: 在"Agent管理"页面选择需要的Agent类型并启动
                2. **执行任务**: 在"任务执行"页面向指定Agent发送任务
                3. **监控状态**: 在"系统状态"页面查看所有Agent的运行状态
                4. **配置优化**: 根据需要调整Agent的配置参数
                """)
        
        # 定期更新下拉框选项
        def update_agent_choices():
            active_agents = list(system.active_agents.keys())
            return (
                gr.Dropdown(choices=active_agents),
                gr.Dropdown(choices=active_agents)
            )
        
        # 初始化界面
        interface.load(
            get_system_status_interface,
            outputs=[system_status_display]
        )
        
        # 定期更新（每10秒）
        def periodic_update():
            while True:
                time.sleep(10)
                try:
                    # 这里可以添加定期更新逻辑
                    pass
                except:
                    pass
        
        # 启动后台更新线程
        update_thread = threading.Thread(target=periodic_update, daemon=True)
        update_thread.start()
    
    return interface


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多Agent系统管理平台")
    parser.add_argument("--mode", choices=['web', 'cli'], default='web',
                       help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机")
    parser.add_argument("--port", type=int, default=7860, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    # 初始化多Agent系统
    system = MultiAgentSystem()
    
    if args.mode == 'cli':
        # CLI模式
        print("多Agent系统 CLI模式")
        print("可用命令: start <agent_type>, stop <agent_type>, status, list, help, quit")
        print(f"可用Agent类型: {', '.join(system.available_agents.keys())}")
        
        while True:
            try:
                command = input(">>> ").strip().split()
                if not command:
                    continue
                    
                cmd = command[0].lower()
                
                if cmd in ['quit', 'exit']:
                    break
                elif cmd == 'list':
                    print("可用Agent类型:")
                    for agent_type in system.available_agents.keys():
                        status = "运行中" if agent_type in system.active_agents else "未启动"
                        print(f"  - {agent_type}: {status}")
                elif cmd == 'status':
                    status = system.get_system_status()
                    print(json.dumps(status, indent=2, ensure_ascii=False, default=str))
                elif cmd == 'start' and len(command) > 1:
                    agent_type = command[1]
                    result = system.start_agent(agent_type)
                    print(result['message'])
                elif cmd == 'stop' and len(command) > 1:
                    agent_type = command[1]
                    result = system.stop_agent(agent_type)
                    print(result['message'])
                elif cmd == 'help':
                    print("""
可用命令:
  start <agent_type>  - 启动指定类型的Agent
  stop <agent_type>   - 停止指定Agent
  list                - 列出所有Agent类型和状态
  status              - 显示系统详细状态
  help                - 显示帮助信息
  quit/exit           - 退出系统
                    """)
                else:
                    print("未知命令，输入 help 查看帮助")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"错误: {e}")
        
        # 清理资源
        for agent_type in list(system.active_agents.keys()):
            system.stop_agent(agent_type)
        
    else:
        # Web界面模式
        interface = create_gradio_interface(system)
        
        print(f"""
🤖 多Agent系统管理平台启动完成！

🌐 Web界面地址: http://{args.host}:{args.port}

📋 系统信息:
- 可用Agent类型: {len(system.available_agents)}个
- 已加载模块: {', '.join(system.available_agents.keys())}

🔧 支持的Agent类型:
{chr(10).join([f"  • {agent_type}" for agent_type in system.available_agents.keys()])}

💡 使用提示:
1. 在Web界面中启动需要的Agent
2. 通过任务执行页面与Agent交互
3. 在系统状态页面监控Agent运行情况
        """)
        
        # 启动服务器
        logger.info(f"启动多Agent系统Web界面: http://{args.host}:{args.port}")
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True
        )


if __name__ == "__main__":
    main()