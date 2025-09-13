#!/usr/bin/env python3
"""
LangChain Enhanced Agents 演示脚本
展示基本功能和使用方法
"""

import os
import sys
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from common.llm_factory import LLMFactory, quick_chat
from common.model_config import ModelConfig, global_cost_tracker
try:
    from ol_langchain_basics.langchain_agent import LangChainSimpleAgent, MultiModelAgent
except ImportError:
    # 如果直接导入失败，尝试相对导入
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '01_langchain_basics'))
    from langchain_agent import LangChainSimpleAgent, MultiModelAgent


def print_header(title: str):
    """打印标题"""
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)


def demo_model_info():
    """演示模型信息功能"""
    print_header("模型配置信息")
    
    # 显示可用模型
    models = ModelConfig.list_available_models()
    print("📋 支持的模型:")
    for provider, model_list in models.items():
        print(f"  {provider}: {', '.join(model_list)}")
    
    # 显示提供商状态
    providers = LLMFactory.get_available_providers()
    print("\n🔌 提供商状态:")
    for provider, available in providers.items():
        status = "✅ 可用" if available else "❌ 不可用"
        print(f"  {provider}: {status}")
    
    # 显示推荐模型
    tasks = ["general", "coding", "analysis", "creative", "cost-effective"]
    print("\n🎯 任务推荐模型:")
    for task in tasks:
        provider, model = ModelConfig.get_recommended_model(task)
        print(f"  {task}: {provider}:{model}")


def demo_simple_agent():
    """演示简单Agent"""
    print_header("简单Agent演示")
    
    try:
        # 创建Agent
        print("🤖 创建LangChain Agent...")
        agent = LangChainSimpleAgent(
            provider="openai",
            model="gpt-3.5-turbo",
            enable_memory=True,
            enable_cost_tracking=True
        )
        
        print("✅ Agent创建成功!")
        
        # 显示Agent信息
        info = agent.get_agent_info()
        print(f"\n📊 Agent信息:")
        print(f"  提供商: {info['provider']}")
        print(f"  模型: {info['model']}")
        print(f"  记忆: {'启用' if info['memory_enabled'] else '禁用'}")
        print(f"  成本追踪: {'启用' if info['cost_tracking_enabled'] else '禁用'}")
        
        # 测试对话
        test_messages = [
            "你好！请介绍一下你自己。",
            "你能帮我做什么？",
            "请记住我的名字是张三，我是一名软件工程师。",
            "你还记得我的职业吗？"
        ]
        
        print("\n💬 对话测试:")
        for i, message in enumerate(test_messages, 1):
            print(f"\n👤 用户 #{i}: {message}")
            response = agent.respond(message)
            print(f"🤖 助手: {response[:100]}..." if len(response) > 100 else f"🤖 助手: {response}")
        
        # 显示成本信息
        cost_summary = agent.get_cost_summary()
        print(f"\n💰 成本摘要: {cost_summary}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        print("请检查API密钥配置!")


def demo_multi_model_agent():
    """演示多模型Agent"""
    print_header("多模型Agent演示")
    
    try:
        # 创建多模型Agent
        print("🎭 创建多模型Agent...")
        multi_agent = MultiModelAgent()
        
        if not multi_agent.agents:
            print("❌ 没有可用的模型，请检查API密钥配置")
            return
            
        print("✅ 多模型Agent创建成功!")
        print(f"📋 可用模型: {multi_agent.list_available_models()}")
        
        # 测试不同类型的任务
        test_tasks = [
            ("写一个Python函数来计算斐波那契数列", "代码任务"),
            ("分析一下人工智能的发展趋势", "分析任务"),
            ("Hello! How are you today?", "通用对话"),
        ]
        
        print("\n🎯 任务测试:")
        for task, task_type in test_tasks:
            print(f"\n📝 {task_type}: {task}")
            result = multi_agent.respond(task)
            
            if result["success"]:
                response = result["response"]
                model = result["model_used"]
                print(f"🤖 模型 [{model}]: {response[:150]}..." if len(response) > 150 else f"🤖 模型 [{model}]: {response}")
            else:
                print(f"❌ 错误: {result['response']}")
        
        # 显示总成本
        total_cost = multi_agent.get_cost_summary()
        print(f"\n💰 总成本: {total_cost}")
        
    except Exception as e:
        print(f"❌ 错误: {e}")


def demo_quick_functions():
    """演示快速功能"""
    print_header("快速功能演示")
    
    try:
        # 快速对话
        print("💬 快速对话测试:")
        questions = [
            "什么是LangChain？",
            "解释一下什么是Agent？",
        ]
        
        for question in questions:
            print(f"\n❓ 问题: {question}")
            try:
                answer = quick_chat(question)
                print(f"💡 回答: {answer[:100]}..." if len(answer) > 100 else f"💡 回答: {answer}")
            except Exception as e:
                print(f"❌ 快速对话失败: {e}")
        
    except Exception as e:
        print(f"❌ 快速功能演示失败: {e}")


def demo_cost_tracking():
    """演示成本追踪"""
    print_header("成本追踪演示")
    
    # 显示全局成本追踪
    summary = global_cost_tracker.get_cost_summary()
    print("📊 全局成本统计:")
    print(f"  总成本: ${summary.get('total_cost', 0):.4f}")
    print(f"  总请求数: {summary.get('requests', 0)}")
    print(f"  总Token数: {summary.get('total_tokens', 0)}")
    
    if summary.get('requests', 0) > 0:
        print(f"  平均每次请求成本: ${summary.get('average_cost_per_request', 0):.4f}")


def main():
    """主函数"""
    print("🚀 欢迎使用 LangChain Enhanced Agents 演示!")
    print("本演示将展示项目的主要功能和使用方法")
    
    # 检查API密钥
    api_keys = {
        "OpenAI": os.getenv("OPENAI_API_KEY"),
        "Anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "Google": os.getenv("GOOGLE_API_KEY"),
    }
    
    print("\n🔑 API密钥状态:")
    has_key = False
    for provider, key in api_keys.items():
        status = "✅ 已配置" if key else "❌ 未配置"
        print(f"  {provider}: {status}")
        if key:
            has_key = True
    
    if not has_key:
        print("\n⚠️  警告: 未检测到任何API密钥!")
        print("请在 .env 文件中配置至少一个API密钥以运行完整演示")
        print("例如: OPENAI_API_KEY=your_key_here")
    
    # 运行演示
    try:
        demo_model_info()
        
        if has_key:
            demo_simple_agent()
            demo_multi_model_agent()
            demo_quick_functions()
            demo_cost_tracking()
        else:
            print("\n⏭️  跳过需要API密钥的演示...")
        
        print_header("演示完成")
        print("🎉 恭喜！您已完成 LangChain Enhanced Agents 的基本演示")
        print("\n📚 下一步:")
        print("  1. 配置API密钥以体验完整功能")
        print("  2. 运行 Web UI: python 01_langchain_basics/web_ui.py")
        print("  3. 查看 ENHANCEMENT_PLAN.md 了解更多高级功能")
        print("  4. 查看 QUICK_START.md 获取详细使用指南")
        
    except KeyboardInterrupt:
        print("\n\n👋 演示被用户中断，再见!")
    except Exception as e:
        print(f"\n❌ 演示过程中出现错误: {e}")
        print("请检查您的配置并重试")


if __name__ == "__main__":
    main()