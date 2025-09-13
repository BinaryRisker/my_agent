"""
多工具集成Agent主程序
整合工具管理和任务执行，提供统一的多工具协作平台
"""

import sys
import argparse
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import uvicorn
from datetime import datetime

# Third-party imports
from loguru import logger
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config

from tool_manager import ToolManager, ToolConfig, ToolType
from task_executor import TaskExecutor, TaskStatus, TaskPriority, ExecutionMode


class MultiToolAgent:
    """多工具集成Agent主类"""
    
    def __init__(self):
        """初始化多工具Agent"""
        self.config = get_config()
        
        # 初始化核心组件
        self.tool_manager = ToolManager()
        self.task_executor = TaskExecutor(self.tool_manager, max_concurrent_tasks=20)
        
        # Web API应用
        self.app = FastAPI(title="多工具集成Agent API", version="1.0.0")
        self._setup_api_routes()
        
        # 会话状态
        self.session_state = {
            'registered_tools': [],
            'submitted_tasks': [],
            'execution_history': []
        }
        
        logger.info("多工具集成Agent初始化完成")
    
    async def start(self):
        """启动Agent"""
        await self.task_executor.start()
        await self._register_default_tools()
        logger.info("多工具集成Agent已启动")
    
    async def stop(self):
        """停止Agent"""
        await self.task_executor.stop()
        await self.tool_manager.cleanup_all()
        logger.info("多工具集成Agent已停止")
    
    async def _register_default_tools(self):
        """注册默认工具"""
        default_tools = [
            # 计算器工具
            ToolConfig(
                name="calculator",
                type=ToolType.LOCAL,
                description="基础计算器，支持加减乘除运算",
                config={
                    "function": self._calculator_function
                }
            ),
            
            # 文本处理工具
            ToolConfig(
                name="text_processor",
                type=ToolType.LOCAL,
                description="文本处理工具，支持大小写转换、替换等",
                config={
                    "function": self._text_processor_function
                }
            ),
            
            # 文件操作工具
            ToolConfig(
                name="file_handler",
                type=ToolType.LOCAL,
                description="文件操作工具，支持读写文件",
                config={
                    "function": self._file_handler_function
                }
            ),
            
            # Web请求工具
            ToolConfig(
                name="web_request",
                type=ToolType.API,
                description="Web请求工具",
                config={
                    "base_url": "https://httpbin.org",
                    "headers": {"User-Agent": "MultiToolAgent/1.0"}
                }
            )
        ]
        
        for tool_config in default_tools:
            success = await self.tool_manager.register_tool(tool_config)
            if success:
                self.session_state['registered_tools'].append(tool_config.name)
                logger.info(f"默认工具注册成功: {tool_config.name}")
    
    def _calculator_function(self, a: float, b: float, operation: str = "add") -> float:
        """计算器函数"""
        if operation == "add":
            return a + b
        elif operation == "subtract":
            return a - b
        elif operation == "multiply":
            return a * b
        elif operation == "divide":
            if b == 0:
                raise ValueError("除数不能为零")
            return a / b
        else:
            raise ValueError(f"不支持的运算: {operation}")
    
    def _text_processor_function(self, text: str, action: str = "upper", **kwargs) -> str:
        """文本处理函数"""
        if action == "upper":
            return text.upper()
        elif action == "lower":
            return text.lower()
        elif action == "replace":
            old_text = kwargs.get("old", "")
            new_text = kwargs.get("new", "")
            return text.replace(old_text, new_text)
        elif action == "reverse":
            return text[::-1]
        elif action == "length":
            return str(len(text))
        else:
            raise ValueError(f"不支持的文本操作: {action}")
    
    def _file_handler_function(self, action: str = "read", file_path: str = "", content: str = "") -> str:
        """文件处理函数"""
        if action == "read":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                raise ValueError(f"读取文件失败: {e}")
        elif action == "write":
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return f"文件写入成功: {file_path}"
            except Exception as e:
                raise ValueError(f"写入文件失败: {e}")
        else:
            raise ValueError(f"不支持的文件操作: {action}")
    
    async def register_tool(self, tool_config: Dict[str, Any]) -> bool:
        """注册工具"""
        try:
            config = ToolConfig(
                name=tool_config["name"],
                type=ToolType(tool_config["type"]),
                description=tool_config["description"],
                config=tool_config.get("config", {})
            )
            
            success = await self.tool_manager.register_tool(config)
            if success:
                self.session_state['registered_tools'].append(config.name)
            
            return success
            
        except Exception as e:
            logger.error(f"工具注册失败: {e}")
            return False
    
    async def execute_single_tool(self, tool_name: str, *args, **kwargs) -> Any:
        """执行单个工具"""
        try:
            result = await self.tool_manager.execute_tool(tool_name, *args, **kwargs)
            
            # 记录执行历史
            self.session_state['execution_history'].append({
                'timestamp': datetime.now().isoformat(),
                'type': 'single_tool',
                'tool_name': tool_name,
                'args': args,
                'kwargs': kwargs,
                'result': str(result)[:200]  # 限制结果长度
            })
            
            return result
            
        except Exception as e:
            logger.error(f"单工具执行失败: {tool_name}, 错误: {e}")
            raise
    
    async def submit_task(self, task_definition: Dict[str, Any]) -> str:
        """提交复合任务"""
        try:
            # 使用任务构建器创建任务
            builder = self.task_executor.create_task_builder(
                task_definition["name"],
                task_definition.get("description", "")
            )
            
            # 添加步骤
            for step in task_definition.get("steps", []):
                builder.add_step(
                    step["tool_name"],
                    *step.get("args", []),
                    description=step.get("description", ""),
                    **step.get("kwargs", {})
                )
            
            # 设置执行模式
            if "execution_mode" in task_definition:
                mode = ExecutionMode(task_definition["execution_mode"])
                builder.set_execution_mode(mode)
            
            # 设置优先级
            if "priority" in task_definition:
                priority = TaskPriority(task_definition["priority"])
                builder.set_priority(priority)
            
            # 构建并提交任务
            task = builder.build()
            task_id = await self.task_executor.submit_task(task)
            
            # 记录提交的任务
            self.session_state['submitted_tasks'].append({
                'task_id': task_id,
                'name': task.name,
                'submitted_at': datetime.now().isoformat(),
                'status': task.status.value
            })
            
            logger.info(f"任务提交成功: {task_id}")
            return task_id
            
        except Exception as e:
            logger.error(f"任务提交失败: {e}")
            raise
    
    async def create_task_from_natural_language(self, description: str) -> str:
        """从自然语言描述创建任务"""
        try:
            task = await self.task_executor.create_task_from_description(description)
            task_id = await self.task_executor.submit_task(task)
            
            # 记录任务
            self.session_state['submitted_tasks'].append({
                'task_id': task_id,
                'name': task.name,
                'submitted_at': datetime.now().isoformat(),
                'status': task.status.value
            })
            
            return task_id
            
        except Exception as e:
            logger.error(f"自然语言任务创建失败: {e}")
            raise
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """获取工具列表"""
        return list(self.tool_manager.get_all_tools().values())
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取任务状态"""
        task = self.task_executor.get_task(task_id)
        if task:
            return {
                'id': task.id,
                'name': task.name,
                'status': task.status.value,
                'created_at': task.created_at,
                'started_at': task.started_at,
                'completed_at': task.completed_at,
                'error': task.error,
                'result': task.result,
                'step_results': task.step_results
            }
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        tool_stats = self.tool_manager.get_statistics()
        executor_stats = self.task_executor.get_statistics()
        
        return {
            'tools': tool_stats,
            'tasks': executor_stats,
            'session': {
                'registered_tools_count': len(self.session_state['registered_tools']),
                'submitted_tasks_count': len(self.session_state['submitted_tasks']),
                'execution_history_count': len(self.session_state['execution_history'])
            }
        }
    
    def _setup_api_routes(self):
        """设置API路由"""
        
        # 添加CORS中间件
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @self.app.get("/")
        async def root():
            return {"message": "多工具集成Agent API", "version": "1.0.0"}
        
        @self.app.get("/tools")
        async def list_tools():
            """获取工具列表"""
            return self.get_tool_list()
        
        @self.app.post("/tools/register")
        async def register_tool_api(tool_config: dict):
            """注册工具"""
            success = await self.register_tool(tool_config)
            if success:
                return {"message": "工具注册成功", "tool_name": tool_config["name"]}
            else:
                raise HTTPException(status_code=400, detail="工具注册失败")
        
        @self.app.post("/tools/{tool_name}/execute")
        async def execute_tool_api(tool_name: str, params: dict = None):
            """执行单个工具"""
            if params is None:
                params = {}
            
            try:
                result = await self.execute_single_tool(
                    tool_name,
                    *params.get("args", []),
                    **params.get("kwargs", {})
                )
                return {"success": True, "result": result}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/tasks/submit")
        async def submit_task_api(task_definition: dict):
            """提交任务"""
            try:
                task_id = await self.submit_task(task_definition)
                return {"task_id": task_id, "message": "任务提交成功"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/tasks/create")
        async def create_task_from_description_api(request: dict):
            """从描述创建任务"""
            description = request.get("description", "")
            if not description:
                raise HTTPException(status_code=400, detail="任务描述不能为空")
            
            try:
                task_id = await self.create_task_from_natural_language(description)
                return {"task_id": task_id, "message": "任务创建成功"}
            except Exception as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.get("/tasks/{task_id}")
        async def get_task_status_api(task_id: str):
            """获取任务状态"""
            task_info = self.get_task_status(task_id)
            if task_info:
                return task_info
            else:
                raise HTTPException(status_code=404, detail="任务不存在")
        
        @self.app.get("/tasks")
        async def list_tasks_api():
            """列出所有任务"""
            tasks = self.task_executor.list_tasks()
            return [
                {
                    'id': task.id,
                    'name': task.name,
                    'status': task.status.value,
                    'created_at': task.created_at
                }
                for task in tasks
            ]
        
        @self.app.delete("/tasks/{task_id}")
        async def cancel_task_api(task_id: str):
            """取消任务"""
            success = await self.task_executor.cancel_task(task_id)
            if success:
                return {"message": "任务取消成功"}
            else:
                raise HTTPException(status_code=400, detail="任务取消失败")
        
        @self.app.get("/statistics")
        async def get_statistics_api():
            """获取统计信息"""
            return self.get_statistics()
        
        @self.app.get("/health")
        async def health_check_api():
            """健康检查"""
            tool_health = await self.tool_manager.health_check()
            return {
                "status": "healthy",
                "tools": tool_health,
                "timestamp": datetime.now().isoformat()
            }


def create_gradio_interface(agent: MultiToolAgent):
    """创建Gradio Web界面"""
    
    def register_tool_interface(name: str, tool_type: str, description: str, config_json: str):
        """工具注册界面"""
        try:
            config = json.loads(config_json) if config_json.strip() else {}
            tool_config = {
                "name": name,
                "type": tool_type,
                "description": description,
                "config": config
            }
            
            # 异步调用需要特殊处理
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(agent.register_tool(tool_config))
            loop.close()
            
            if success:
                return f"工具 '{name}' 注册成功！", json.dumps(agent.get_tool_list(), indent=2, ensure_ascii=False)
            else:
                return f"工具 '{name}' 注册失败！", ""
        except Exception as e:
            return f"注册失败: {e}", ""
    
    def execute_tool_interface(tool_name: str, args_json: str, kwargs_json: str):
        """单工具执行界面"""
        try:
            args = json.loads(args_json) if args_json.strip() else []
            kwargs = json.loads(kwargs_json) if kwargs_json.strip() else {}
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(agent.execute_single_tool(tool_name, *args, **kwargs))
            loop.close()
            
            return f"执行成功！\n结果: {result}"
        except Exception as e:
            return f"执行失败: {e}"
    
    def submit_task_interface(task_name: str, task_description: str, steps_json: str, execution_mode: str):
        """任务提交界面"""
        try:
            steps = json.loads(steps_json) if steps_json.strip() else []
            task_definition = {
                "name": task_name,
                "description": task_description,
                "steps": steps,
                "execution_mode": execution_mode.lower()
            }
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_id = loop.run_until_complete(agent.submit_task(task_definition))
            loop.close()
            
            return f"任务提交成功！\n任务ID: {task_id}", task_id
        except Exception as e:
            return f"任务提交失败: {e}", ""
    
    def natural_language_task_interface(description: str):
        """自然语言任务创建界面"""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            task_id = loop.run_until_complete(agent.create_task_from_natural_language(description))
            loop.close()
            
            return f"任务创建成功！\n任务ID: {task_id}", task_id
        except Exception as e:
            return f"任务创建失败: {e}", ""
    
    def get_task_status_interface(task_id: str):
        """任务状态查询界面"""
        if not task_id.strip():
            return "请输入任务ID"
        
        task_info = agent.get_task_status(task_id)
        if task_info:
            return json.dumps(task_info, indent=2, ensure_ascii=False)
        else:
            return "任务不存在"
    
    def get_statistics_interface():
        """统计信息界面"""
        stats = agent.get_statistics()
        return json.dumps(stats, indent=2, ensure_ascii=False)
    
    # 创建Gradio界面
    with gr.Blocks(title="多工具集成Agent", theme=gr.themes.Soft()) as interface:
        gr.Markdown("# 🔧 多工具集成Agent")
        gr.Markdown("统一管理多种工具，支持复杂任务的分解和并行执行")
        
        with gr.Tabs():
            # 工具管理标签页
            with gr.TabItem("🛠️ 工具管理"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 注册新工具")
                        tool_name_input = gr.Textbox(label="工具名称", placeholder="输入工具名称")
                        tool_type_dropdown = gr.Dropdown(
                            label="工具类型",
                            choices=["api", "local", "shell"],
                            value="local"
                        )
                        tool_desc_input = gr.Textbox(
                            label="工具描述",
                            placeholder="输入工具功能描述"
                        )
                        tool_config_input = gr.Textbox(
                            label="配置 (JSON格式)",
                            placeholder='{"key": "value"}',
                            lines=3
                        )
                        register_btn = gr.Button("📝 注册工具", variant="primary")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 当前工具列表")
                        tools_display = gr.Textbox(
                            label="已注册工具",
                            lines=15,
                            interactive=False,
                            value=json.dumps(agent.get_tool_list(), indent=2, ensure_ascii=False)
                        )
                
                register_result = gr.Textbox(label="注册结果", lines=3)
                
                register_btn.click(
                    register_tool_interface,
                    inputs=[tool_name_input, tool_type_dropdown, tool_desc_input, tool_config_input],
                    outputs=[register_result, tools_display]
                )
            
            # 单工具执行标签页
            with gr.TabItem("⚡ 单工具执行"):
                with gr.Row():
                    with gr.Column():
                        exec_tool_name = gr.Textbox(label="工具名称", placeholder="输入要执行的工具名称")
                        exec_args = gr.Textbox(
                            label="位置参数 (JSON数组格式)",
                            placeholder='[1, 2, "hello"]',
                            lines=2
                        )
                        exec_kwargs = gr.Textbox(
                            label="关键字参数 (JSON对象格式)",
                            placeholder='{"operation": "add"}',
                            lines=3
                        )
                        execute_btn = gr.Button("▶️ 执行工具", variant="primary")
                    
                    with gr.Column():
                        execution_result = gr.Textbox(
                            label="执行结果",
                            lines=10,
                            interactive=False
                        )
                
                execute_btn.click(
                    execute_tool_interface,
                    inputs=[exec_tool_name, exec_args, exec_kwargs],
                    outputs=[execution_result]
                )
            
            # 复合任务标签页
            with gr.TabItem("🔄 复合任务"):
                with gr.Row():
                    with gr.Column():
                        task_name_input = gr.Textbox(label="任务名称", placeholder="输入任务名称")
                        task_desc_input = gr.Textbox(
                            label="任务描述",
                            placeholder="输入任务描述",
                            lines=2
                        )
                        task_steps_input = gr.Textbox(
                            label="任务步骤 (JSON格式)",
                            placeholder='''[
  {
    "tool_name": "calculator",
    "args": [3, 5],
    "kwargs": {"operation": "add"},
    "description": "计算3+5"
  }
]''',
                            lines=8
                        )
                        exec_mode_dropdown = gr.Dropdown(
                            label="执行模式",
                            choices=["sequential", "parallel", "pipeline", "conditional"],
                            value="sequential"
                        )
                        submit_task_btn = gr.Button("📤 提交任务", variant="primary")
                    
                    with gr.Column():
                        task_submit_result = gr.Textbox(
                            label="提交结果",
                            lines=5,
                            interactive=False
                        )
                        current_task_id = gr.Textbox(
                            label="当前任务ID",
                            interactive=False
                        )
                
                submit_task_btn.click(
                    submit_task_interface,
                    inputs=[task_name_input, task_desc_input, task_steps_input, exec_mode_dropdown],
                    outputs=[task_submit_result, current_task_id]
                )
            
            # 自然语言任务标签页
            with gr.TabItem("💬 智能任务创建"):
                with gr.Row():
                    with gr.Column():
                        nl_description = gr.Textbox(
                            label="任务描述",
                            placeholder="用自然语言描述你想要执行的任务...",
                            lines=4
                        )
                        create_nl_task_btn = gr.Button("🤖 创建任务", variant="primary")
                    
                    with gr.Column():
                        nl_task_result = gr.Textbox(
                            label="创建结果",
                            lines=5,
                            interactive=False
                        )
                        nl_task_id = gr.Textbox(
                            label="任务ID",
                            interactive=False
                        )
                
                create_nl_task_btn.click(
                    natural_language_task_interface,
                    inputs=[nl_description],
                    outputs=[nl_task_result, nl_task_id]
                )
            
            # 任务监控标签页
            with gr.TabItem("📊 任务监控"):
                with gr.Row():
                    with gr.Column(scale=1):
                        monitor_task_id = gr.Textbox(
                            label="任务ID",
                            placeholder="输入要查询的任务ID"
                        )
                        query_status_btn = gr.Button("🔍 查询状态")
                        
                        refresh_stats_btn = gr.Button("📈 刷新统计")
                    
                    with gr.Column(scale=2):
                        task_status_display = gr.Textbox(
                            label="任务状态",
                            lines=15,
                            interactive=False
                        )
                
                statistics_display = gr.Textbox(
                    label="系统统计信息",
                    lines=10,
                    interactive=False,
                    value=json.dumps(agent.get_statistics(), indent=2, ensure_ascii=False)
                )
                
                query_status_btn.click(
                    get_task_status_interface,
                    inputs=[monitor_task_id],
                    outputs=[task_status_display]
                )
                
                refresh_stats_btn.click(
                    get_statistics_interface,
                    outputs=[statistics_display]
                )
    
    return interface


async def run_cli():
    """运行CLI模式"""
    parser = argparse.ArgumentParser(description="多工具集成Agent CLI")
    parser.add_argument("command", choices=['register', 'execute', 'submit', 'status', 'list'],
                       help="执行的命令")
    parser.add_argument("--tool-name", help="工具名称")
    parser.add_argument("--tool-type", help="工具类型")
    parser.add_argument("--description", help="描述信息")
    parser.add_argument("--config", help="配置JSON")
    parser.add_argument("--args", help="参数JSON")
    parser.add_argument("--task-id", help="任务ID")
    parser.add_argument("--task-def", help="任务定义JSON文件")
    
    args = parser.parse_args()
    
    # 创建Agent
    agent = MultiToolAgent()
    await agent.start()
    
    try:
        if args.command == 'register':
            if not all([args.tool_name, args.tool_type, args.description]):
                print("注册工具需要提供: --tool-name, --tool-type, --description")
                return
            
            config = json.loads(args.config) if args.config else {}
            tool_config = {
                "name": args.tool_name,
                "type": args.tool_type,
                "description": args.description,
                "config": config
            }
            
            success = await agent.register_tool(tool_config)
            print(f"工具注册{'成功' if success else '失败'}: {args.tool_name}")
        
        elif args.command == 'execute':
            if not args.tool_name:
                print("执行工具需要提供: --tool-name")
                return
            
            tool_args = json.loads(args.args) if args.args else {}
            result = await agent.execute_single_tool(
                args.tool_name,
                *tool_args.get("args", []),
                **tool_args.get("kwargs", {})
            )
            print(f"执行结果: {result}")
        
        elif args.command == 'submit':
            if not args.task_def:
                print("提交任务需要提供: --task-def")
                return
            
            with open(args.task_def, 'r', encoding='utf-8') as f:
                task_definition = json.load(f)
            
            task_id = await agent.submit_task(task_definition)
            print(f"任务提交成功，ID: {task_id}")
        
        elif args.command == 'status':
            if not args.task_id:
                print("查询状态需要提供: --task-id")
                return
            
            status = agent.get_task_status(args.task_id)
            if status:
                print(json.dumps(status, indent=2, ensure_ascii=False))
            else:
                print("任务不存在")
        
        elif args.command == 'list':
            tools = agent.get_tool_list()
            print("已注册工具:")
            for tool in tools:
                print(f"- {tool['name']}: {tool['description']}")
    
    except Exception as e:
        print(f"命令执行失败: {e}")
    
    finally:
        await agent.stop()


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多工具集成Agent")
    parser.add_argument("--mode", choices=['web', 'api', 'cli'], default='web',
                       help="运行模式")
    parser.add_argument("--host", default="127.0.0.1", help="服务器主机")
    parser.add_argument("--port", type=int, default=7865, help="服务器端口")
    parser.add_argument("--api-port", type=int, default=8000, help="API服务器端口")
    parser.add_argument("--share", action="store_true", help="创建公共链接")
    
    args = parser.parse_args()
    
    if args.mode == 'cli':
        await run_cli()
    elif args.mode == 'api':
        # 纯API模式
        agent = MultiToolAgent()
        await agent.start()
        
        try:
            config = uvicorn.Config(
                agent.app,
                host=args.host,
                port=args.api_port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            logger.info(f"启动API服务器: http://{args.host}:{args.api_port}")
            await server.serve()
        finally:
            await agent.stop()
    
    else:
        # Web界面模式
        agent = MultiToolAgent()
        await agent.start()
        
        try:
            # 创建Web界面
            interface = create_gradio_interface(agent)
            
            # 同时启动API服务器
            if args.mode == 'web':
                # 在后台启动API服务器
                import threading
                api_thread = threading.Thread(
                    target=lambda: uvicorn.run(
                        agent.app,
                        host=args.host,
                        port=args.api_port,
                        log_level="warning"
                    )
                )
                api_thread.daemon = True
                api_thread.start()
                logger.info(f"API服务器已启动: http://{args.host}:{args.api_port}")
            
            # 启动Web界面
            logger.info(f"启动Web界面: http://{args.host}:{args.port}")
            interface.launch(
                server_name=args.host,
                server_port=args.port,
                share=args.share,
                show_error=True
            )
        
        finally:
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main())