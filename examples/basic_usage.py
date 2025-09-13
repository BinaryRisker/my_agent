"""
多Agent系统基础使用示例
演示如何使用多Agent系统的核心功能
"""

import sys
import json
import time
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MultiAgentSystem


def demo_system_overview():
    """演示系统概览功能"""
    print("🔍 多Agent系统概览演示")
    print("=" * 50)
    
    # 初始化系统
    system = MultiAgentSystem()
    
    # 显示可用的Agent类型
    print(f"📋 可用Agent类型: {len(system.available_agents)}个")
    for agent_type in system.available_agents.keys():
        print(f"  • {agent_type}")
    
    # 获取系统状态
    status = system.get_system_status()
    print(f"\n🏥 系统健康状态: {status['system_health']}")
    print(f"🔢 活跃Agent数量: {len(status['active_agents'])}")
    
    return system


def demo_agent_lifecycle(system):
    """演示Agent生命周期管理"""
    print("\n🔄 Agent生命周期管理演示")
    print("=" * 50)
    
    if not system.available_agents:
        print("❌ 没有可用的Agent类型")
        return None
    
    # 选择第一个可用的Agent类型
    agent_type = list(system.available_agents.keys())[0]
    print(f"📝 选择Agent类型: {agent_type}")
    
    # 启动Agent
    print(f"\n🚀 启动Agent: {agent_type}")
    start_result = system.start_agent(agent_type)
    
    if start_result['success']:
        print(f"✅ 启动成功: {start_result['message']}")
        
        # 查看Agent信息
        agent_info = system._get_agent_info(agent_type)
        print(f"📊 Agent信息: {json.dumps(agent_info, indent=2, ensure_ascii=False)}")
        
        # 等待一下
        time.sleep(2)
        
        # 停止Agent
        print(f"\n⏹️ 停止Agent: {agent_type}")
        stop_result = system.stop_agent(agent_type)
        
        if stop_result['success']:
            print(f"✅ 停止成功: {stop_result['message']}")
        else:
            print(f"❌ 停止失败: {stop_result['message']}")
        
        return agent_type
    else:
        print(f"❌ 启动失败: {start_result['message']}")
        return None


def demo_task_execution(system, agent_type):
    """演示任务执行功能"""
    print("\n⚡ 任务执行演示")
    print("=" * 50)
    
    if not agent_type:
        print("❌ 没有可用的Agent")
        return
    
    # 启动Agent
    start_result = system.start_agent(agent_type)
    if not start_result['success']:
        print(f"❌ 无法启动Agent: {start_result['message']}")
        return
    
    try:
        # 定义测试任务
        test_tasks = [
            {
                'type': 'respond',
                'data': {'input': 'Hello, how are you?'}
            },
            {
                'type': 'chat', 
                'data': {'message': 'What can you do?', 'session_id': 'demo_session'}
            },
            {
                'type': 'analyze',
                'data': {'dataset': 'sample_data'}
            },
            {
                'type': 'learn',
                'data': {'min_experiences': 5}
            }
        ]
        
        for i, task in enumerate(test_tasks, 1):
            print(f"\n📋 执行任务 {i}: {task['type']}")
            
            result = system.execute_agent_task(agent_type, task)
            
            if result['success']:
                print(f"✅ 任务完成: {result['message']}")
                # 只显示结果的一部分，避免输出过长
                result_preview = str(result['result'])[:200]
                if len(str(result['result'])) > 200:
                    result_preview += "..."
                print(f"📊 执行结果: {result_preview}")
            else:
                print(f"❌ 任务失败: {result['message']}")
    
    finally:
        # 清理：停止Agent
        system.stop_agent(agent_type)


def demo_multiple_agents(system):
    """演示多Agent同时工作"""
    print("\n🤖 多Agent协作演示")
    print("=" * 50)
    
    available_agents = list(system.available_agents.keys())
    
    if len(available_agents) < 2:
        print("❌ 需要至少2个可用的Agent类型")
        return
    
    # 选择前两个Agent类型
    agents_to_demo = available_agents[:2]
    active_agents = []
    
    try:
        # 启动多个Agent
        for agent_type in agents_to_demo:
            print(f"\n🚀 启动Agent: {agent_type}")
            result = system.start_agent(agent_type)
            
            if result['success']:
                print(f"✅ {agent_type} 启动成功")
                active_agents.append(agent_type)
            else:
                print(f"❌ {agent_type} 启动失败: {result['message']}")
        
        if len(active_agents) < 2:
            print("❌ 无法启动足够的Agent进行演示")
            return
        
        # 显示系统状态
        print(f"\n📊 当前活跃Agent: {active_agents}")
        status = system.get_system_status()
        print(f"🏥 系统健康状态: {status['system_health']}")
        
        # 为每个Agent分配不同的任务
        tasks = [
            {'type': 'respond', 'data': {'input': 'Hello from Agent 1'}},
            {'type': 'chat', 'data': {'message': 'Hello from Agent 2'}}
        ]
        
        # 并发执行任务
        for agent_type, task in zip(active_agents, tasks):
            print(f"\n⚡ {agent_type} 执行任务: {task['type']}")
            result = system.execute_agent_task(agent_type, task)
            
            if result['success']:
                print(f"✅ {agent_type} 任务完成")
            else:
                print(f"❌ {agent_type} 任务失败: {result['message']}")
        
        # 等待一下观察效果
        time.sleep(2)
        
        # 最终状态检查
        final_status = system.get_system_status()
        print(f"\n📈 最终系统状态:")
        print(f"  活跃Agent数量: {len(final_status['active_agents'])}")
        print(f"  系统健康状态: {final_status['system_health']}")
        
    finally:
        # 清理：停止所有启动的Agent
        print(f"\n🧹 清理资源...")
        for agent_type in active_agents:
            result = system.stop_agent(agent_type)
            print(f"⏹️ {agent_type}: {result['message']}")


def demo_error_handling(system):
    """演示错误处理"""
    print("\n🚨 错误处理演示")
    print("=" * 50)
    
    # 测试启动不存在的Agent
    print("📋 测试1: 启动不存在的Agent")
    result = system.start_agent('non_existent_agent')
    print(f"🔍 结果: {result['message']}")
    
    # 测试停止未运行的Agent
    print("\n📋 测试2: 停止未运行的Agent")
    result = system.stop_agent('inactive_agent')
    print(f"🔍 结果: {result['message']}")
    
    # 测试在未启动Agent上执行任务
    print("\n📋 测试3: 在未启动Agent上执行任务")
    task = {'type': 'test', 'data': {}}
    result = system.execute_agent_task('inactive_agent', task)
    print(f"🔍 结果: {result['message']}")
    
    print("✅ 错误处理演示完成")


def demo_system_monitoring(system):
    """演示系统监控功能"""
    print("\n📊 系统监控演示")
    print("=" * 50)
    
    # 获取详细状态
    status = system.get_system_status()
    
    print("📈 系统概况:")
    print(f"  总Agent类型: {len(status['available_agents'])}")
    print(f"  当前活跃: {len(status['active_agents'])}")
    print(f"  系统健康: {status['system_health']}")
    
    print(f"\n📋 可用Agent列表:")
    for agent_type in status['available_agents']:
        print(f"  • {agent_type}")
    
    if status['active_agents']:
        print(f"\n🔄 活跃Agent详情:")
        for agent_type in status['active_agents']:
            agent_details = status['agent_details'].get(agent_type, {})
            print(f"  • {agent_type}:")
            print(f"    - 状态: {agent_details.get('status', 'unknown')}")
            print(f"    - 类名: {agent_details.get('class', 'unknown')}")
    else:
        print("\n💤 当前没有活跃的Agent")
    
    # 系统健康检查
    health = system._check_system_health()
    health_indicators = {
        'idle': '💤 空闲',
        'healthy': '✅ 健康',
        'partial': '⚠️ 部分正常',
        'unhealthy': '❌ 不健康',
        'error': '🚨 错误'
    }
    
    print(f"\n🏥 健康状态: {health_indicators.get(health, health)}")


def main():
    """主演示函数"""
    print("🤖 多Agent系统基础使用演示")
    print("🎯 本演示将展示多Agent系统的核心功能")
    print("=" * 80)
    
    try:
        # 1. 系统概览
        system = demo_system_overview()
        
        # 2. Agent生命周期
        agent_type = demo_agent_lifecycle(system)
        
        # 3. 任务执行
        if agent_type:
            demo_task_execution(system, agent_type)
        
        # 4. 多Agent协作
        demo_multiple_agents(system)
        
        # 5. 错误处理
        demo_error_handling(system)
        
        # 6. 系统监控
        demo_system_monitoring(system)
        
        print("\n🎉 演示完成！")
        print("💡 提示: 您可以运行 'python main.py' 启动Web界面进行交互式体验")
        
    except KeyboardInterrupt:
        print("\n⛔ 演示被用户中断")
    except Exception as e:
        print(f"\n🚨 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n👋 谢谢使用多Agent系统！")


if __name__ == "__main__":
    main()