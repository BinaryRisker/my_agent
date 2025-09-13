# 🚀 LangChain Agent 快速开始指南

## 📋 前置条件

### 1. 安装 Python（3.8+）
确保您已安装Python 3.8或更高版本。

### 2. 安装依赖
```bash
cd enhanced_agents
pip install -r requirements.txt
```

### 3. 设置API密钥
创建 `.env` 文件并设置您的API密钥：

```bash
# .env 文件内容

# OpenAI
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic (Claude)
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Google (Gemini)
GOOGLE_API_KEY=your_google_api_key_here
```

## 🎯 快速测试

### 1. 命令行测试
```bash
cd 01_langchain_basics
python langchain_agent.py
```

### 2. Web界面测试
```bash
cd 01_langchain_basics
python web_ui.py
```

然后在浏览器中访问 `http://localhost:7860`

## 📖 使用示例

### 基本Agent创建
```python
from enhanced_agents.common.llm_factory import LLMFactory
from enhanced_agents.01_langchain_basics.langchain_agent import LangChainSimpleAgent

# 创建Agent
agent = LangChainSimpleAgent(
    provider="openai",
    model="gpt-3.5-turbo",
    enable_memory=True,
    enable_cost_tracking=True
)

# 开始对话
response = agent.respond("Hello! Tell me about yourself.")
print(response)

# 获取成本信息
cost_summary = agent.get_cost_summary()
print(cost_summary)
```

### 多模型Agent
```python
from enhanced_agents.01_langchain_basics.langchain_agent import MultiModelAgent

# 创建多模型Agent
multi_agent = MultiModelAgent()

# 自动选择最适合的模型
result = multi_agent.respond("Write some Python code to calculate fibonacci numbers")
print(f"Response: {result['response']}")
print(f"Model used: {result['model_used']}")
```

### 快速工厂函数
```python
from enhanced_agents.common.llm_factory import quick_llm, quick_chat

# 快速创建LLM
llm = quick_llm("coding")  # 自动选择适合编程的模型

# 快速对话
response = quick_chat("What is machine learning?")
print(response)
```

## 🔧 配置选项

### 模型选择
支持的提供商和模型：

| 提供商 | 推荐模型 | 用途 |
|--------|----------|------|
| OpenAI | gpt-4 | 高质量任务 |
| OpenAI | gpt-3.5-turbo | 通用对话 |
| Anthropic | claude-3-sonnet | 分析任务 |
| Google | gemini-pro | 多模态任务 |
| Ollama | llama2 | 本地部署 |

### 参数调优
```python
agent = LangChainSimpleAgent(
    provider="openai",
    model="gpt-4",
    temperature=0.7,        # 创造性 0.0-2.0
    enable_memory=True,     # 启用对话记忆
    enable_cost_tracking=True,  # 启用成本追踪
    system_prompt="Custom prompt..."  # 自定义系统提示词
)
```

## 🌐 Web界面功能

### Single Agent Chat
- 配置单个Agent
- 自定义系统提示词
- 调整Temperature参数
- 实时对话

### Multi-Model Chat
- 自动模型选择
- 手动指定模型
- 显示使用的模型

### Agent Info & Stats
- Agent配置信息
- 成本统计
- 系统状态监控

## 🔍 故障排除

### 常见问题

**Q: ImportError: No module named 'langchain'**
```bash
pip install langchain langchain-openai langchain-community
```

**Q: API密钥错误**
- 检查 `.env` 文件是否正确配置
- 确认API密钥有效且有足够余额

**Q: 连接测试失败**
- 检查网络连接
- 确认防火墙设置
- 尝试不同的模型

**Q: 本地模型不可用**
```bash
# 安装Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# 拉取模型
ollama pull llama2
```

### 调试模式
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 这将显示详细的调试信息
```

## 📚 下一步

1. **探索更多功能**：查看 `ENHANCEMENT_PLAN.md` 了解完整功能路线图

2. **自定义工具**：学习如何创建自定义工具和Agent

3. **生产部署**：了解如何部署到生产环境

4. **高级特性**：探索RAG、Function Calling等高级功能

## 🤝 获取帮助

- 查看项目文档
- 提交Issue到GitHub仓库
- 加入社区讨论

---

**祝您使用愉快！开始您的LangChain Agent之旅吧！** 🎉