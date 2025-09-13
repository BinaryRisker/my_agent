# 🚀 LangChain Agent 学习项目

一个循序渐进的LangChain智能体学习项目，通过6个阶段从基础到高级，全面掌握AI Agent开发技能。

## 🎯 项目概述

本项目设计为一个完整的学习路径，让您通过实践项目深入理解LangChain和AI Agent的各个方面：

- **渐进式学习**: 从简单到复杂，每个阶段都建立在前一阶段的基础上
- **实际应用**: 每个项目都是可运行的完整应用
- **最佳实践**: 代码遵循工业级标准和最佳实践
- **全面覆盖**: 涵盖工具使用、记忆管理、文档处理、多模态等

## 📚 学习路径

### 阶段 1: 基础智能助手 (01_basic_assistant) 
**状态**: ✅ 实现中
**学习目标**:
- 理解LangChain基础架构
- 掌握Agent和工具的基本概念
- 学习内存管理和对话流程
- 熟悉提示工程基础

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