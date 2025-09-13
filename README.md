# 🤖 Multi-Agent System - 多Agent系统

一个完整的多Agent系统，展示了从简单响应到自学习的Agent发展历程，包含6个不同发展阶段的智能Agent。

## 🎯 项目概述

本项目是一个渐进式的多Agent系统开发框架，通过6个阶段展示了智能Agent的发展演进：

1. **阶段1**: 简单响应Agent - 基础的输入输出处理
2. **阶段2**: 记忆增强Agent - 具备记忆和上下文感知能力
3. **阶段3**: 工具使用Agent - 能够调用外部工具和API
4. **阶段4**: 多工具集成Agent - 统一管理和调度多种工具
5. **阶段5**: 数据分析Agent - 数据处理、分析和机器学习
6. **阶段6**: 自学习Agent - 知识管理和自我优化

## 🏗️ 项目结构

```
my_agent/
├── main.py                          # 多Agent系统主入口
├── README.md                        # 项目说明文档
├── requirements.txt                 # 项目依赖
├── common/                          # 公共模块
│   ├── __init__.py
│   ├── config.py                    # 配置管理
│   └── utils.py                     # 工具函数
├── tool_integration/                # 工具集成框架
│   └── src/
│       └── tool_interface.py       # 统一工具接口
├── 01_simple_response_agent/        # 阶段1：简单响应Agent
│   ├── src/
│   │   └── main.py
│   └── README.md
├── 02_memory_enhanced_agent/        # 阶段2：记忆增强Agent
│   ├── src/
│   │   ├── main.py
│   │   └── memory_manager.py
│   └── README.md
├── 03_tool_using_agent/            # 阶段3：工具使用Agent
│   ├── src/
│   │   ├── main.py
│   │   └── tool_manager.py
│   └── README.md
├── 04_multi_tool_agent/            # 阶段4：多工具集成Agent
│   ├── src/
│   │   ├── main.py
│   │   ├── tool_manager.py
│   │   └── task_executor.py
│   └── README.md
├── 05_data_analysis_agent/         # 阶段5：数据分析Agent
│   ├── src/
│   │   ├── main.py
│   │   ├── data_processor.py
│   │   └── ml_engine.py
│   └── README.md
├── 06_self_learning_agent/         # 阶段6：自学习Agent
│   ├── src/
│   │   ├── main.py
│   │   ├── knowledge_base.py
│   │   └── learning_engine.py
│   └── README.md
├── tests/                          # 测试文件
│   ├── test_agents.py
│   └── test_integration.py
└── examples/                       # 示例和演示
    ├── basic_usage.py
    ├── advanced_scenarios.py
    └── data/
        └── sample_data.csv
```

## 🚀 快速开始

### 环境要求

- Python 3.8+
- 建议使用虚拟环境

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动多Agent系统

#### Web界面模式（推荐）

```bash
python main.py
```

然后打开浏览器访问 http://127.0.0.1:7860

#### CLI模式

```bash
python main.py --mode cli
```

### 单独运行各阶段Agent

每个阶段的Agent都可以独立运行：

```bash
# 阶段1：简单响应Agent
cd 01_simple_response_agent/src
python main.py

# 阶段2：记忆增强Agent
cd 02_memory_enhanced_agent/src  
python main.py

# 阶段3：工具使用Agent
cd 03_tool_using_agent/src
python main.py

# 阶段4：多工具集成Agent
cd 04_multi_tool_agent/src
python main.py

# 阶段5：数据分析Agent
cd 05_data_analysis_agent/src
python main.py

# 阶段6：自学习Agent
cd 06_self_learning_agent/src
python main.py
```

## 📚 各阶段Agent详细介绍

### 🔸 阶段1：简单响应Agent

**核心功能**：
- 基础的输入输出处理
- 简单的响应生成机制
- 基本的配置管理

**核心概念**:
- LangChain Agent 架构
- Tool 定义和使用
- ConversationBufferMemory
- OpenAI API 集成

**实践内容**:
- 创建计算器、天气查询工具
- 实现CLI和Web界面
- 添加对话记忆功能
- 错误处理和日志记录

**技能收获**:
- Agent开发基础
- 工具创建方法
- 用户界面设计
- 系统架构思维

### 阶段 2: 文档分析Agent (02_document_analyzer)
**状态**: 📋 规划中
**学习目标**:
- 掌握文档处理和向量化
- 理解RAG（检索增强生成）原理
- 学习向量数据库操作
- 实现智能问答系统

**核心概念**:
- Document Loaders
- Text Splitters
- Vector Stores (Chroma, FAISS)
- Retrieval Chains

**实践内容**:
- PDF/Word文档解析
- 文本向量化和存储
- 相似性搜索实现
- 基于文档的问答

### 阶段 3: 代码助手Agent (03_code_assistant)
**状态**: 📋 规划中
**学习目标**:
- 理解代码分析和生成
- 学习静态分析工具集成
- 掌握代码质量评估
- 实现自动化代码优化

**核心概念**:
- Code Parsing
- AST (抽象语法树) 分析
- Code Generation
- Static Analysis Integration

**实践内容**:
- 代码审查和建议
- 单元测试自动生成
- 代码重构建议
- 多语言代码支持

### 阶段 4: 多工具协作Agent (04_multi_tool_agent)
**状态**: 📋 规划中
**学习目标**:
- 理解复杂任务分解
- 掌握工具链编排
- 学习错误恢复机制
- 实现自动化工作流

**核心概念**:
- Tool Chains
- Sequential Chains
- Router Chains
- Error Handling Strategies

**实践内容**:
- 网络搜索+数据分析
- 自动报告生成
- 邮件和日程管理
- 多步骤任务执行

### 阶段 5: 数据分析Agent (05_data_analysis_agent)
**状态**: 📋 规划中
**学习目标**:
- 掌握自然语言到SQL转换
- 学习数据可视化自动化
- 理解统计分析自动化
- 实现智能数据洞察

**核心概念**:
- Natural Language to SQL
- Data Visualization
- Statistical Analysis
- Business Intelligence

**实践内容**:
- 数据库查询生成
- 图表自动生成
- 数据趋势分析
- 商业洞察报告

### 阶段 6: 自主学习Agent (06_self_learning_agent)
**状态**: 📋 规划中
**学习目标**:
- 理解强化学习在Agent中的应用
- 掌握知识图谱构建
- 学习自适应行为调整
- 实现持续学习机制

**核心概念**:
- Reinforcement Learning
- Knowledge Graphs
- Adaptive Behavior
- Continuous Learning

**实践内容**:
- 用户反馈学习
- 行为策略优化
- 知识图谱更新
- 性能持续改进

## 🛠️ 环境设置

### Windows + PowerShell 设置

1. **克隆项目**
```powershell
git clone https://github.com/yourusername/my_agent.git
cd my_agent
```

2. **创建虚拟环境**
```powershell
python -m venv agent_env
agent_env\\Scripts\\Activate.ps1
```

3. **安装依赖**
```powershell
pip install -r requirements.txt
```

4. **配置环境变量**
```powershell
cp .env.example .env
# 编辑 .env 文件，填入您的API密钥
```

### API 密钥获取

- **OpenAI API**: https://platform.openai.com/api-keys
- **天气API**: https://openweathermap.org/api (免费)

## 🚀 快速开始

### 运行第一个项目
```powershell
cd 01_basic_assistant
python main.py --mode cli
```

### 启动Web界面
```powershell
cd 01_basic_assistant
python main.py --mode web
```

### 运行测试
```powershell
pytest 01_basic_assistant/tests/
```

## 📖 学习建议

1. **循序渐进**: 严格按照阶段顺序学习，每个阶段都为后续奠定基础
2. **动手实践**: 运行每个项目，修改代码，观察效果
3. **深入理解**: 阅读代码注释，理解设计理念
4. **扩展练习**: 完成每个阶段的练习任务
5. **记录总结**: 建议建立学习笔记，记录问题和心得

## 📁 项目结构

```
my_agent/
├── 01_basic_assistant/          # 阶段1: 基础智能助手
│   ├── main.py                  # 入口文件
│   ├── tools/                   # 工具模块
│   ├── interface/               # 用户界面
│   ├── tests/                   # 测试文件
│   └── README.md                # 阶段说明
├── 02_document_analyzer/        # 阶段2: 文档分析
├── 03_code_assistant/           # 阶段3: 代码助手
├── 04_multi_tool_agent/         # 阶段4: 多工具协作
├── 05_data_analysis_agent/      # 阶段5: 数据分析
├── 06_self_learning_agent/      # 阶段6: 自主学习
├── common/                      # 共享组件
├── docs/                        # 文档
├── tests/                       # 集成测试
└── requirements.txt             # 依赖清单
```

## 🔧 开发工具

- **代码格式化**: `black .`
- **代码检查**: `flake8 .`
- **测试运行**: `pytest`
- **依赖检查**: `pip-audit`

## 🤝 贡献指南

1. Fork 本项目
2. 创建特性分支
3. 提交变更
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 🆘 获取帮助

- 📧 Email: your.email@example.com
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/my_agent/discussions)
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/my_agent/issues)

---

**开始您的AI Agent学习之旅吧！** 🎉

从 `01_basic_assistant` 开始，逐步掌握LangChain和Agent开发的精髓！