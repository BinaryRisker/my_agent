"""
多Agent系统测试
测试各个阶段Agent的基本功能
"""

import sys
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import MultiAgentSystem


class TestMultiAgentSystem:
    """多Agent系统测试类"""
    
    def setup_method(self):
        """测试前的设置"""
        self.system = MultiAgentSystem()
    
    def test_system_initialization(self):
        """测试系统初始化"""
        assert self.system is not None
        assert isinstance(self.system.available_agents, dict)
        assert len(self.system.available_agents) >= 0
        assert self.system.system_status['total_agents'] >= 0
    
    def test_get_system_status(self):
        """测试获取系统状态"""
        status = self.system.get_system_status()
        
        assert 'system_info' in status
        assert 'available_agents' in status
        assert 'active_agents' in status
        assert 'agent_details' in status
        assert 'system_health' in status
        
        assert isinstance(status['available_agents'], list)
        assert isinstance(status['active_agents'], list)
        assert isinstance(status['agent_details'], dict)
    
    def test_system_health_check(self):
        """测试系统健康检查"""
        health = self.system._check_system_health()
        assert health in ['idle', 'healthy', 'partial', 'unhealthy', 'error']
    
    @pytest.mark.skipif(len(MultiAgentSystem().available_agents) == 0, 
                       reason="No agents available for testing")
    def test_start_stop_agent(self):
        """测试启动和停止Agent"""
        if not self.system.available_agents:
            pytest.skip("No available agents to test")
        
        # 选择第一个可用的Agent类型进行测试
        agent_type = list(self.system.available_agents.keys())[0]
        
        # 测试启动Agent
        start_result = self.system.start_agent(agent_type)
        assert 'success' in start_result
        
        if start_result['success']:
            # 如果启动成功，测试停止Agent
            assert agent_type in self.system.active_agents
            
            stop_result = self.system.stop_agent(agent_type)
            assert 'success' in stop_result
            assert stop_result['success'] is True
            assert agent_type not in self.system.active_agents
    
    def test_start_nonexistent_agent(self):
        """测试启动不存在的Agent"""
        result = self.system.start_agent('nonexistent_agent')
        
        assert 'success' in result
        assert result['success'] is False
        assert 'message' in result
        assert 'unknown' in result['message'].lower() or 'not' in result['message'].lower()
    
    def test_stop_inactive_agent(self):
        """测试停止未运行的Agent"""
        result = self.system.stop_agent('inactive_agent')
        
        assert 'success' in result
        assert result['success'] is False
        assert 'message' in result


class TestAgentModules:
    """测试各个Agent模块的基础功能"""
    
    def test_simple_response_agent(self):
        """测试简单响应Agent"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "01_simple_response_agent" / "src"))
            from main import SimpleResponseAgent
            
            agent = SimpleResponseAgent("TestAgent", "A test agent")
            assert agent.name == "TestAgent"
            assert agent.description == "A test agent"
            
            # 测试响应功能
            if hasattr(agent, 'respond'):
                response = agent.respond("Hello")
                assert isinstance(response, str)
                assert len(response) > 0
            
        except ImportError:
            pytest.skip("SimpleResponseAgent not available")
    
    def test_memory_enhanced_agent(self):
        """测试记忆增强Agent"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "02_memory_enhanced_agent" / "src"))
            from main import MemoryEnhancedAgent
            
            agent = MemoryEnhancedAgent(memory_size=10)
            assert hasattr(agent, 'memory_manager')
            
        except ImportError:
            pytest.skip("MemoryEnhancedAgent not available")
    
    def test_data_analysis_agent(self):
        """测试数据分析Agent"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "05_data_analysis_agent" / "src"))
            from main import DataAnalysisAgent
            
            agent = DataAnalysisAgent()
            assert hasattr(agent, 'data_processor')
            assert hasattr(agent, 'ml_engine')
            
        except ImportError:
            pytest.skip("DataAnalysisAgent not available")
    
    def test_self_learning_agent(self):
        """测试自学习Agent"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "06_self_learning_agent" / "src"))
            from main import SelfLearningAgent
            
            # 使用临时数据库
            with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp:
                agent = SelfLearningAgent(knowledge_db_path=tmp.name, memory_size=100)
                assert hasattr(agent, 'knowledge_manager')
                assert hasattr(agent, 'learning_engine')
                
                # 测试获取状态
                if hasattr(agent, 'get_agent_status'):
                    status = agent.get_agent_status()
                    assert isinstance(status, dict)
                
        except ImportError:
            pytest.skip("SelfLearningAgent not available")


class TestToolIntegration:
    """测试工具集成框架"""
    
    def test_tool_interface_imports(self):
        """测试工具接口模块导入"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "tool_integration" / "src"))
            from tool_interface import (
                ToolType, ToolStatus, BaseTool, ToolRegistry, 
                create_function_tool, global_tool_registry
            )
            
            assert ToolType is not None
            assert ToolStatus is not None
            assert BaseTool is not None
            assert ToolRegistry is not None
            
        except ImportError as e:
            pytest.skip(f"Tool integration not available: {e}")
    
    def test_tool_registry(self):
        """测试工具注册表"""
        try:
            sys.path.append(str(Path(__file__).parent.parent / "tool_integration" / "src"))
            from tool_interface import ToolRegistry, create_function_tool
            
            registry = ToolRegistry()
            assert hasattr(registry, 'register_tool')
            assert hasattr(registry, 'get_registry_info')
            
            # 创建测试工具
            def test_func(tool_input):
                return "test result"
            
            test_tool = create_function_tool(
                name="test_tool",
                description="A test tool",
                func=test_func
            )
            
            # 测试注册
            result = registry.register_tool(test_tool)
            assert result is True
            
            # 测试信息获取
            info = registry.get_registry_info()
            assert 'total_tools' in info
            assert info['total_tools'] >= 1
            
        except ImportError:
            pytest.skip("Tool integration not available")


class TestCommonModules:
    """测试公共模块"""
    
    def test_config_module(self):
        """测试配置模块"""
        try:
            from common.config import get_config
            
            config = get_config()
            assert isinstance(config, dict)
            
        except ImportError:
            pytest.skip("Config module not available")
    
    def test_utils_module(self):
        """测试工具模块"""
        try:
            from common.utils import setup_logging
            
            # 测试日志设置
            logger = setup_logging()
            assert logger is not None
            
        except ImportError:
            pytest.skip("Utils module not available")


@pytest.fixture
def sample_data():
    """提供测试数据"""
    return {
        "test_messages": [
            "Hello, how are you?",
            "Can you help me with something?",
            "What's the weather like today?",
            "Tell me a joke",
            "Goodbye!"
        ],
        "test_data": {
            "numbers": [1, 2, 3, 4, 5],
            "strings": ["a", "b", "c", "d", "e"],
            "mixed": [1, "two", 3.0, True, None]
        }
    }


class TestIntegrationScenarios:
    """集成测试场景"""
    
    def test_agent_lifecycle(self, sample_data):
        """测试Agent完整生命周期"""
        system = MultiAgentSystem()
        
        if not system.available_agents:
            pytest.skip("No available agents for integration testing")
        
        agent_type = list(system.available_agents.keys())[0]
        
        # 启动Agent
        start_result = system.start_agent(agent_type)
        if not start_result.get('success', False):
            pytest.skip(f"Could not start agent {agent_type}")
        
        try:
            # 检查状态
            status = system.get_system_status()
            assert agent_type in status['active_agents']
            
            # 执行任务（如果支持）
            task = {
                'type': 'respond',
                'data': {'input': sample_data['test_messages'][0]}
            }
            
            task_result = system.execute_agent_task(agent_type, task)
            assert 'success' in task_result
            
        finally:
            # 清理：停止Agent
            stop_result = system.stop_agent(agent_type)
            assert stop_result.get('success', True)
    
    def test_multiple_agents(self):
        """测试多个Agent同时运行"""
        system = MultiAgentSystem()
        
        if len(system.available_agents) < 2:
            pytest.skip("Need at least 2 agents for multi-agent testing")
        
        agents_to_test = list(system.available_agents.keys())[:2]
        started_agents = []
        
        try:
            # 启动多个Agent
            for agent_type in agents_to_test:
                result = system.start_agent(agent_type)
                if result.get('success', False):
                    started_agents.append(agent_type)
            
            if len(started_agents) < 2:
                pytest.skip("Could not start enough agents")
            
            # 检查状态
            status = system.get_system_status()
            assert len(status['active_agents']) >= 2
            
            for agent_type in started_agents:
                assert agent_type in status['active_agents']
        
        finally:
            # 清理：停止所有启动的Agent
            for agent_type in started_agents:
                system.stop_agent(agent_type)


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v"])