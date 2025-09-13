# 🚀 阶段1: 基础智能助手

欢迎来到LangChain Agent学习项目的第一阶段！本阶段将带您构建一个功能完整的基础智能助手，掌握LangChain和AI Agent开发的核心概念。

## 🎯 学习目标

通过本阶段的学习，您将：

1. **理解LangChain基础架构** - 掌握Agent、Tools、Memory等核心概念
2. **掌握工具创建和使用** - 学习如何创建自定义工具并集成到Agent中
3. **学习内存管理** - 了解对话历史记录和上下文管理
4. **熟悉提示工程基础** - 学习如何设计有效的系统提示
5. **实现用户界面** - 创建CLI和Web界面，提供良好的用户体验

## 📚 核心概念

### 1. LangChain Agent架构
- **Agent**: 决策引擎，决定使用哪些工具以及如何使用
- **Tools**: 执行具体任务的工具（计算器、天气查询、文件操作）
- **Memory**: 管理对话历史和上下文
- **LLM**: 大语言模型，提供推理能力

### 2. 工具系统
- **Tool接口**: 基于LangChain BaseTool的标准化工具接口
- **安全设计**: 输入验证、错误处理、权限控制
- **可扩展性**: 模块化设计，易于添加新工具

### 3. 内存管理
- **ConversationBufferMemory**: 缓存对话历史
- **持久化**: 对话保存和加载功能
- **上下文管理**: 智能的上下文窗口管理

## 🏗️ 项目架构

```
01_basic_assistant/
├── main.py                 # 主程序入口
├── config.py              # 配置管理
├── memory.py              # 内存管理模块
├── tools/                 # 工具模块
│   ├── __init__.py
│   ├── calculator.py      # 数学计算工具
│   ├── weather.py         # 天气查询工具
│   └── file_ops.py        # 文件操作工具
├── interface/             # 用户界面
│   ├── __init__.py
│   ├── cli.py            # 命令行界面
│   └── web.py            # Web界面
├── tests/                # 测试文件
├── logs/                 # 日志目录
└── README.md            # 本文档
```

## 🛠️ 功能特性

### ✅ 已实现功能

#### 🧮 数学计算工具
- 安全的表达式评估（基于AST解析）
- 支持基本运算：+、-、*、/、**、%
- 支持数学函数：sin、cos、tan、log、sqrt、abs等
- 支持数学常量：π、e、τ
- 错误处理和输入验证

#### 🌤️ 天气查询工具
- 支持全球城市天气查询
- 实时天气信息（或模拟数据）
- 缓存机制提升响应速度
- 详细的天气信息展示

#### 📁 文件操作工具
- 安全的文件读写操作
- 支持多种文件格式：txt、md、json、csv
- 目录浏览和文件列表
- 路径安全验证和权限控制

#### 🧠 智能对话
- 基于OpenAI GPT的对话能力
- 上下文理解和记忆功能
- 工具使用决策和执行
- 友好的中文交互体验

#### 🖥️ 多种界面
- **CLI界面**: 丰富的命令行交互体验
- **Web界面**: 基于Gradio的现代Web界面
- 实时状态显示和内存管理

## 🚀 快速开始

### 1. 环境准备

```powershell
# 创建虚拟环境
python -m venv agent_env
agent_env\\Scripts\\Activate.ps1

# 安装依赖（从项目根目录运行）
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并填入您的API密钥：

```bash
# 复制环境变量文件
cp ../.env.example ../.env
```

编辑 `.env` 文件：
```bash
OPENAI_API_KEY=your_openai_api_key_here
WEATHER_API_KEY=your_weather_api_key_here  # 可选
```

### 3. 运行助手

#### CLI模式（推荐新手）
```powershell
python main.py --mode cli
```

#### Web模式
```powershell
python main.py --mode web
```

Web界面将在 http://127.0.0.1:7860 启动

## 💡 使用示例

### CLI界面使用

```
🤖 基础智能助手 - LangChain Agent 学习项目
输入消息与我对话，或使用命令（如 /help）获取帮助

你: 计算 2 + 3 * 4
助手: 我来帮您计算 2 + 3 * 4。

[使用计算器工具...]

结果是 14。

你: 北京今天天气如何？
助手: 我来查询北京的天气信息。

[使用天气查询工具...]

📍 北京, CN
🌡️ 温度: 22.5°C (体感 24.1°C)  
☁️ 天气: 晴朗 (Clear)
💧 湿度: 45%
...
```

### 可用命令

- `/help` - 显示帮助信息
- `/clear` - 清空对话记忆  
- `/memory` - 显示内存使用情况
- `/tools` - 显示可用工具
- `/save` - 保存对话到文件
- `/load` - 从文件加载对话
- `/exit` - 退出程序

## 🧪 测试和验证

### 运行测试（计划中）
```powershell
pytest tests/ -v
```

### 手动测试清单

#### ✅ 基础功能测试
- [ ] 助手启动和初始化
- [ ] 简单对话交互
- [ ] 工具调用功能

#### ✅ 计算器工具测试
- [ ] 基本运算：`2 + 3 * 4`
- [ ] 数学函数：`sin(pi/2)`、`sqrt(16)`
- [ ] 复杂表达式：`sqrt(sin(pi/4)**2 + cos(pi/4)**2)`
- [ ] 错误处理：`1/0`、无效表达式

#### ✅ 天气工具测试
- [ ] 中文城市查询：`北京天气如何？`
- [ ] 英文城市查询：`New York weather`
- [ ] 不存在的城市处理

#### ✅ 文件工具测试
- [ ] 写入文件：`写入 test.txt|这是测试内容`
- [ ] 读取文件：`读取文件 test.txt`
- [ ] 列出目录：`列出文件`
- [ ] 错误处理：读取不存在的文件

#### ✅ 内存管理测试
- [ ] 对话记忆保持
- [ ] 内存状态查看
- [ ] 内存清空功能
- [ ] 对话保存和加载

## 🔧 配置选项

### LLM配置
```python
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.7
  max_tokens: 1000
  timeout: 30
```

### 工具配置
```python
tools:
  weather:
    cache_timeout: 300  # 缓存时间（秒）
  calculator:
    max_expression_length: 100
  file_ops:
    max_file_size: 1048576  # 1MB
    allowed_extensions: [".txt", ".md", ".json", ".csv"]
```

## 📖 扩展学习

### 添加新工具

1. 在 `tools/` 目录创建新的工具文件
2. 继承 `BaseTool` 类
3. 实现 `_run` 方法
4. 在 `tools/__init__.py` 中注册工具

示例：
```python
from langchain.tools import BaseTool

class MyTool(BaseTool):
    name = "my_tool"
    description = "我的自定义工具"
    
    def _run(self, input_str: str) -> str:
        # 工具逻辑
        return "工具输出"
```

### 更换语言模型

修改 `config.py` 中的LLM配置：
```python
# 使用不同的OpenAI模型
llm.model = "gpt-4"

# 或使用其他提供商
llm.provider = "anthropic"
```

### 定制系统提示

修改 `main.py` 中的 `_get_system_message` 方法：
```python
def _get_system_message(self) -> str:
    return """你的自定义系统提示..."""
```

## 🐛 常见问题

### Q: 助手无法启动？
A: 检查以下项目：
1. OpenAI API密钥是否正确配置
2. 依赖包是否正确安装
3. Python版本是否兼容（建议3.8+）

### Q: 工具调用失败？
A: 检查：
1. 输入格式是否正确
2. 权限设置是否合适
3. 查看日志文件了解详细错误信息

### Q: Web界面无法访问？
A: 尝试：
1. 检查端口是否被占用
2. 防火墙设置
3. 使用 `--host 0.0.0.0` 参数

## 📚 学习资源

### 推荐阅读
- [LangChain官方文档](https://python.langchain.com/)
- [OpenAI API文档](https://platform.openai.com/docs)
- [Gradio文档](https://gradio.app/docs/)

### 下一步学习
完成本阶段后，您可以继续学习：
- **阶段2**: 文档分析Agent（RAG系统）
- **阶段3**: 代码助手Agent（代码分析和生成）
- **阶段4**: 多工具协作Agent（复杂任务分解）

## 🤝 贡献

发现问题或有改进建议？欢迎：
- 提交Issue报告问题
- 创建Pull Request贡献代码
- 分享您的学习心得和扩展实现

## 📄 许可证

本项目遵循MIT许可证，详见根目录LICENSE文件。

---

**🎉 恭喜您完成基础智能助手的构建！**

这是您AI Agent开发之旅的第一步。通过本阶段的学习，您已经掌握了：

- ✅ LangChain Agent的基础架构
- ✅ 工具系统的设计和实现
- ✅ 内存管理和对话流程
- ✅ 用户界面的创建
- ✅ 安全和错误处理

现在您已经准备好进入下一个阶段的学习了！🚀