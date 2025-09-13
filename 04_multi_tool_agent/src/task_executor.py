"""
任务执行引擎
支持复杂任务的分解、调度和并行执行
"""

import sys
import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor
import threading

# Third-party imports
from loguru import logger

# Project imports
sys.path.append(str(Path(__file__).parent.parent.parent))
from common.config import get_config
from common.llm_client import get_llm_client


class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING = "waiting"


class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4


class ExecutionMode(Enum):
    """执行模式"""
    SEQUENTIAL = "sequential"    # 顺序执行
    PARALLEL = "parallel"        # 并行执行
    PIPELINE = "pipeline"        # 流水线执行
    CONDITIONAL = "conditional"  # 条件执行


@dataclass
class TaskStep:
    """任务步骤"""
    id: str
    tool_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    description: str = ""
    timeout: Optional[int] = None
    retry_count: int = 0
    dependencies: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # 执行条件


@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    steps: List[TaskStep] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    timeout: Optional[int] = None
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 运行时状态
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Any = None
    step_results: Dict[str, Any] = field(default_factory=dict)


class TaskContext:
    """任务上下文"""
    
    def __init__(self, task: Task):
        self.task = task
        self.variables: Dict[str, Any] = {}
        self.step_results: Dict[str, Any] = {}
        self.shared_state: Dict[str, Any] = {}
        
        # 从任务元数据初始化变量
        self.variables.update(task.metadata)
    
    def set_variable(self, name: str, value: Any):
        """设置变量"""
        self.variables[name] = value
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """获取变量"""
        return self.variables.get(name, default)
    
    def set_step_result(self, step_id: str, result: Any):
        """设置步骤结果"""
        self.step_results[step_id] = result
    
    def get_step_result(self, step_id: str, default: Any = None) -> Any:
        """获取步骤结果"""
        return self.step_results.get(step_id, default)
    
    def evaluate_condition(self, condition: str) -> bool:
        """评估条件表达式"""
        if not condition:
            return True
        
        try:
            # 简单的条件评估（实际项目中应该使用更安全的方式）
            # 这里只是示例，实际应该使用专门的表达式引擎
            context = {
                'vars': self.variables,
                'results': self.step_results,
                'state': self.shared_state
            }
            return eval(condition, {"__builtins__": {}}, context)
        except Exception as e:
            logger.error(f"条件评估失败: {condition}, 错误: {e}")
            return False


class TaskExecutor:
    """任务执行器"""
    
    def __init__(self, tool_manager, max_concurrent_tasks: int = 10):
        """初始化任务执行器"""
        self.config = get_config()
        self.tool_manager = tool_manager
        self.llm_client = get_llm_client()
        
        # 任务管理
        self.tasks: Dict[str, Task] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        
        # 执行配置
        self.max_concurrent_tasks = max_concurrent_tasks
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_tasks)
        
        # 状态管理
        self.is_running = False
        self.task_lock = asyncio.Lock()
        
        logger.info(f"任务执行器初始化完成，最大并发任务数: {max_concurrent_tasks}")
    
    async def start(self):
        """启动任务执行器"""
        self.is_running = True
        logger.info("任务执行器已启动")
        
        # 启动任务调度器
        asyncio.create_task(self._task_scheduler())
    
    async def stop(self):
        """停止任务执行器"""
        self.is_running = False
        
        # 取消所有运行中的任务
        for task_id, running_task in self.running_tasks.items():
            if not running_task.done():
                running_task.cancel()
                logger.info(f"取消任务: {task_id}")
        
        # 等待所有任务完成
        if self.running_tasks:
            await asyncio.gather(*self.running_tasks.values(), return_exceptions=True)
        
        self.executor.shutdown(wait=True)
        logger.info("任务执行器已停止")
    
    async def submit_task(self, task: Task) -> str:
        """提交任务"""
        async with self.task_lock:
            task.status = TaskStatus.PENDING
            self.tasks[task.id] = task
            await self.task_queue.put(task.id)
            
            logger.info(f"任务已提交: {task.id} - {task.name}")
            return task.id
    
    async def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        async with self.task_lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                
                if task.status == TaskStatus.PENDING:
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"任务已取消: {task_id}")
                    return True
                
                elif task.status == TaskStatus.RUNNING and task_id in self.running_tasks:
                    running_task = self.running_tasks[task_id]
                    running_task.cancel()
                    task.status = TaskStatus.CANCELLED
                    logger.info(f"运行中任务已取消: {task_id}")
                    return True
                
                else:
                    logger.warning(f"任务无法取消，当前状态: {task.status}, 任务ID: {task_id}")
                    return False
            else:
                logger.warning(f"任务不存在: {task_id}")
                return False
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """获取任务"""
        return self.tasks.get(task_id)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """获取任务状态"""
        task = self.tasks.get(task_id)
        return task.status if task else None
    
    def list_tasks(self, status: Optional[TaskStatus] = None) -> List[Task]:
        """列出任务"""
        if status:
            return [task for task in self.tasks.values() if task.status == status]
        return list(self.tasks.values())
    
    async def _task_scheduler(self):
        """任务调度器"""
        logger.info("任务调度器已启动")
        
        while self.is_running:
            try:
                # 检查并发任务数量
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # 获取待执行任务
                try:
                    task_id = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                task = self.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # 启动任务执行
                running_task = asyncio.create_task(self._execute_task(task))
                self.running_tasks[task_id] = running_task
                
                logger.info(f"开始执行任务: {task_id}")
                
            except Exception as e:
                logger.error(f"任务调度器错误: {e}")
                await asyncio.sleep(1)
        
        logger.info("任务调度器已停止")
    
    async def _execute_task(self, task: Task):
        """执行任务"""
        task_context = TaskContext(task)
        
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = time.time()
            
            logger.info(f"执行任务: {task.id} - {task.name}")
            
            # 根据执行模式执行任务
            if task.execution_mode == ExecutionMode.SEQUENTIAL:
                result = await self._execute_sequential(task, task_context)
            elif task.execution_mode == ExecutionMode.PARALLEL:
                result = await self._execute_parallel(task, task_context)
            elif task.execution_mode == ExecutionMode.PIPELINE:
                result = await self._execute_pipeline(task, task_context)
            elif task.execution_mode == ExecutionMode.CONDITIONAL:
                result = await self._execute_conditional(task, task_context)
            else:
                raise ValueError(f"不支持的执行模式: {task.execution_mode}")
            
            # 任务完成
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            task.result = result
            task.step_results = task_context.step_results
            
            logger.info(f"任务执行完成: {task.id}")
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            logger.info(f"任务被取消: {task.id}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = time.time()
            
            logger.error(f"任务执行失败: {task.id}, 错误: {e}")
            
        finally:
            # 清理运行中任务记录
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _execute_sequential(self, task: Task, context: TaskContext) -> Any:
        """顺序执行任务步骤"""
        results = []
        
        for step in task.steps:
            if not context.evaluate_condition(step.condition):
                logger.info(f"跳过步骤（条件不满足）: {step.id}")
                continue
            
            # 检查依赖
            if not self._check_dependencies(step, context):
                logger.warning(f"步骤依赖不满足: {step.id}")
                continue
            
            # 执行步骤
            step_result = await self._execute_step(step, context)
            context.set_step_result(step.id, step_result)
            results.append(step_result)
        
        return results
    
    async def _execute_parallel(self, task: Task, context: TaskContext) -> List[Any]:
        """并行执行任务步骤"""
        # 创建并行执行任务
        parallel_tasks = []
        
        for step in task.steps:
            if not context.evaluate_condition(step.condition):
                logger.info(f"跳过步骤（条件不满足）: {step.id}")
                continue
            
            if not self._check_dependencies(step, context):
                logger.warning(f"步骤依赖不满足: {step.id}")
                continue
            
            step_task = asyncio.create_task(
                self._execute_step_with_context(step, context)
            )
            parallel_tasks.append((step.id, step_task))
        
        # 等待所有任务完成
        results = []
        for step_id, step_task in parallel_tasks:
            try:
                step_result = await step_task
                context.set_step_result(step_id, step_result)
                results.append(step_result)
            except Exception as e:
                logger.error(f"并行步骤执行失败: {step_id}, 错误: {e}")
                results.append(None)
        
        return results
    
    async def _execute_pipeline(self, task: Task, context: TaskContext) -> Any:
        """流水线执行任务步骤"""
        result = None
        
        for step in task.steps:
            if not context.evaluate_condition(step.condition):
                logger.info(f"跳过步骤（条件不满足）: {step.id}")
                continue
            
            # 将上一步的结果作为下一步的输入
            if result is not None:
                step.args = [result] + step.args
            
            step_result = await self._execute_step(step, context)
            context.set_step_result(step.id, step_result)
            result = step_result
        
        return result
    
    async def _execute_conditional(self, task: Task, context: TaskContext) -> List[Any]:
        """条件执行任务步骤"""
        results = []
        
        for step in task.steps:
            if context.evaluate_condition(step.condition):
                step_result = await self._execute_step(step, context)
                context.set_step_result(step.id, step_result)
                results.append(step_result)
            else:
                logger.info(f"跳过步骤（条件不满足）: {step.id}")
                results.append(None)
        
        return results
    
    async def _execute_step_with_context(self, step: TaskStep, context: TaskContext) -> Any:
        """带上下文执行步骤（用于并行执行）"""
        return await self._execute_step(step, context)
    
    async def _execute_step(self, step: TaskStep, context: TaskContext) -> Any:
        """执行单个任务步骤"""
        logger.debug(f"执行步骤: {step.id} - {step.tool_name}")
        
        # 处理参数中的变量替换
        processed_args = self._process_arguments(step.args, context)
        processed_kwargs = self._process_arguments(step.kwargs, context)
        
        # 执行工具
        try:
            result = await self.tool_manager.execute_tool(
                step.tool_name,
                *processed_args,
                **processed_kwargs
            )
            
            logger.debug(f"步骤执行成功: {step.id}")
            return result
            
        except Exception as e:
            logger.error(f"步骤执行失败: {step.id}, 错误: {e}")
            
            # 根据重试配置进行重试
            if step.retry_count > 0:
                for attempt in range(step.retry_count):
                    logger.info(f"重试步骤: {step.id}, 第 {attempt + 1} 次")
                    try:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                        result = await self.tool_manager.execute_tool(
                            step.tool_name,
                            *processed_args,
                            **processed_kwargs
                        )
                        logger.info(f"步骤重试成功: {step.id}")
                        return result
                    except Exception as retry_error:
                        logger.warning(f"步骤重试失败: {step.id}, 错误: {retry_error}")
                        continue
            
            # 重试失败或无重试配置
            raise
    
    def _process_arguments(self, args: Any, context: TaskContext) -> Any:
        """处理参数中的变量替换"""
        if isinstance(args, str):
            # 简单的变量替换（实际项目中应该使用模板引擎）
            if args.startswith("${") and args.endswith("}"):
                var_name = args[2:-1]
                if var_name.startswith("vars."):
                    return context.get_variable(var_name[5:])
                elif var_name.startswith("results."):
                    return context.get_step_result(var_name[8:])
                elif var_name.startswith("state."):
                    return context.shared_state.get(var_name[6:])
            return args
        elif isinstance(args, list):
            return [self._process_arguments(arg, context) for arg in args]
        elif isinstance(args, dict):
            return {k: self._process_arguments(v, context) for k, v in args.items()}
        else:
            return args
    
    def _check_dependencies(self, step: TaskStep, context: TaskContext) -> bool:
        """检查步骤依赖"""
        for dep_step_id in step.dependencies:
            if dep_step_id not in context.step_results:
                return False
        return True
    
    async def create_task_from_description(self, description: str, **kwargs) -> Task:
        """从描述创建任务（使用LLM）"""
        try:
            prompt = f"""
请将以下任务描述转换为具体的执行步骤：

任务描述: {description}

可用工具: {self.tool_manager.get_available_tools()}

请以JSON格式返回任务定义，包含：
1. 任务名称
2. 任务描述
3. 执行步骤列表，每个步骤包含：
   - 步骤ID
   - 工具名称
   - 参数
   - 描述

示例格式：
{{
  "name": "任务名称",
  "description": "任务描述", 
  "steps": [
    {{
      "id": "step_1",
      "tool_name": "工具名",
      "args": [],
      "kwargs": {{}},
      "description": "步骤描述"
    }}
  ]
}}
"""
            
            response = await self.llm_client.generate_async(
                prompt=prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # 解析LLM响应
            task_def = json.loads(response.strip())
            
            # 创建任务对象
            task = Task(
                id=str(uuid.uuid4()),
                name=task_def["name"],
                description=task_def["description"],
                steps=[
                    TaskStep(
                        id=step["id"],
                        tool_name=step["tool_name"],
                        args=step.get("args", []),
                        kwargs=step.get("kwargs", {}),
                        description=step.get("description", "")
                    )
                    for step in task_def["steps"]
                ],
                metadata=kwargs
            )
            
            logger.info(f"从描述创建任务成功: {task.name}")
            return task
            
        except Exception as e:
            logger.error(f"从描述创建任务失败: {e}")
            raise ValueError(f"无法解析任务描述: {e}")
    
    def create_task_builder(self, name: str, description: str = "") -> 'TaskBuilder':
        """创建任务构建器"""
        return TaskBuilder(name, description, self.tool_manager.get_available_tools())
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取执行统计"""
        status_counts = {}
        for task in self.tasks.values():
            status = task.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        return {
            "total_tasks": len(self.tasks),
            "running_tasks": len(self.running_tasks),
            "status_distribution": status_counts,
            "is_running": self.is_running,
            "max_concurrent_tasks": self.max_concurrent_tasks
        }


class TaskBuilder:
    """任务构建器"""
    
    def __init__(self, name: str, description: str = "", available_tools: List[str] = None):
        self.task = Task(
            id=str(uuid.uuid4()),
            name=name,
            description=description
        )
        self.available_tools = available_tools or []
    
    def add_step(self, tool_name: str, *args, description: str = "", **kwargs) -> 'TaskBuilder':
        """添加步骤"""
        if tool_name not in self.available_tools:
            logger.warning(f"工具可能不可用: {tool_name}")
        
        step = TaskStep(
            id=f"step_{len(self.task.steps) + 1}",
            tool_name=tool_name,
            args=list(args),
            kwargs=kwargs,
            description=description
        )
        self.task.steps.append(step)
        return self
    
    def set_priority(self, priority: TaskPriority) -> 'TaskBuilder':
        """设置优先级"""
        self.task.priority = priority
        return self
    
    def set_execution_mode(self, mode: ExecutionMode) -> 'TaskBuilder':
        """设置执行模式"""
        self.task.execution_mode = mode
        return self
    
    def set_timeout(self, timeout: int) -> 'TaskBuilder':
        """设置超时时间"""
        self.task.timeout = timeout
        return self
    
    def add_metadata(self, **metadata) -> 'TaskBuilder':
        """添加元数据"""
        self.task.metadata.update(metadata)
        return self
    
    def build(self) -> Task:
        """构建任务"""
        if not self.task.steps:
            raise ValueError("任务至少需要一个步骤")
        
        return self.task


if __name__ == "__main__":
    async def test_task_executor():
        """测试任务执行器"""
        from tool_manager import ToolManager, ToolConfig, ToolType
        
        # 创建工具管理器
        tool_manager = ToolManager()
        
        # 注册测试工具
        calc_config = ToolConfig(
            name="calculator",
            type=ToolType.LOCAL,
            description="计算器",
            config={
                "function": lambda a, b, op="add": a + b if op == "add" else a * b
            }
        )
        await tool_manager.register_tool(calc_config)
        
        # 创建任务执行器
        executor = TaskExecutor(tool_manager)
        await executor.start()
        
        # 创建测试任务
        task = (executor.create_task_builder("测试任务", "简单的计算任务")
                .add_step("calculator", 3, 5, op="add", description="计算 3 + 5")
                .add_step("calculator", "${results.step_1}", 2, op="multiply", description="结果乘以 2")
                .set_execution_mode(ExecutionMode.SEQUENTIAL)
                .build())
        
        # 提交任务
        task_id = await executor.submit_task(task)
        
        # 等待任务完成
        while True:
            task_status = executor.get_task_status(task_id)
            if task_status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
            await asyncio.sleep(0.5)
        
        # 获取结果
        completed_task = executor.get_task(task_id)
        print(f"任务状态: {completed_task.status}")
        print(f"任务结果: {completed_task.result}")
        print(f"步骤结果: {completed_task.step_results}")
        
        # 清理
        await executor.stop()
        await tool_manager.cleanup_all()
    
    # 运行测试
    asyncio.run(test_task_executor())