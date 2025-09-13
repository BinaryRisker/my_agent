# 🤝 贡献指南

感谢您对LangChain Agent学习项目的关注！我们欢迎各种形式的贡献，包括但不限于代码改进、文档完善、问题反馈和功能建议。

## 📋 贡献类型

### 🐛 问题反馈
- 发现bug或错误
- 文档错误或不清楚的地方
- 环境兼容性问题
- 性能问题

### 💡 功能建议
- 新工具的建议
- 界面改进建议
- 学习路径优化
- 代码架构改进

### 📖 文档改进
- 修正拼写和语法错误
- 增加使用示例
- 完善安装指南
- 添加常见问题解答

### 💻 代码贡献
- Bug修复
- 新功能实现
- 性能优化
- 测试用例添加

## 🚀 快速开始

### 1. Fork项目
点击GitHub页面右上角的"Fork"按钮，将项目复制到您的账户下。

### 2. 克隆项目
```bash
git clone https://github.com/yourusername/my_agent.git
cd my_agent
```

### 3. 设置开发环境
```bash
# 运行环境设置脚本
.\\setup.ps1

# 或手动设置
python -m venv agent_env
agent_env\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

### 4. 创建分支
```bash
git checkout -b feature/your-feature-name
# 或
git checkout -b bugfix/issue-description
```

## 📝 开发规范

### 代码风格
- 使用Python PEP 8代码风格
- 使用类型提示（Type Hints）
- 编写清晰的文档字符串
- 保持函数和类的单一职责

```python
def calculate_result(expression: str) -> Union[float, int, str]:
    \"\"\"
    计算数学表达式
    
    Args:
        expression: 数学表达式字符串
        
    Returns:
        计算结果或错误信息
    \"\"\"
    # 实现代码...
```

### 提交规范
使用清晰的提交信息：
```bash
# 功能添加
git commit -m "feat: 添加新的计算器函数支持"

# Bug修复
git commit -m "fix: 修复天气API缓存问题"

# 文档更新
git commit -m "docs: 更新安装指南"

# 测试添加
git commit -m "test: 添加文件操作工具测试"
```

### 文件结构
添加新功能时，请遵循现有的文件结构：
```
阶段目录/
├── src/           # 源代码
├── tests/         # 测试文件
├── docs/          # 文档
└── README.md      # 说明文档
```

## 🧪 测试要求

### 单元测试
- 为新功能编写测试用例
- 确保测试覆盖率不低于80%
- 使用pytest框架

```python
def test_calculator_basic_operations():
    calc = CalculatorTool()
    assert calc._run("2 + 3") == "5"
    assert calc._run("10 / 2") == "5.0"
```

### 集成测试
- 测试完整的用户流程
- 验证多个组件的协作
- 模拟真实使用场景

### 运行测试
```bash
# 运行所有测试
pytest

# 运行特定测试文件
pytest tests/test_calculator.py

# 查看覆盖率报告
pytest --cov=src tests/
```

## 📚 文档规范

### README更新
- 保持文档与代码同步
- 添加新功能的使用示例
- 更新安装和配置说明

### 代码注释
- 为复杂逻辑添加注释
- 解释设计决策的原因
- 提供使用示例

### 类型提示
```python
from typing import List, Dict, Optional, Union

class WeatherTool(BaseTool):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
    
    def get_weather(self, city: str) -> Dict[str, Any]:
        # 实现代码...
```

## 🔍 Pull Request流程

### 1. 提交前检查
- [ ] 代码通过所有测试
- [ ] 代码风格符合规范
- [ ] 文档已更新
- [ ] 提交信息清晰明确

### 2. 创建Pull Request
- 提供详细的PR描述
- 说明变更的原因和影响
- 引用相关的Issue
- 添加适当的标签

### 3. PR模板
```markdown
## 变更描述
简要描述这次变更的内容和目的

## 变更类型
- [ ] Bug修复
- [ ] 新功能
- [ ] 文档更新
- [ ] 性能优化
- [ ] 其他

## 测试
- [ ] 已添加新的测试用例
- [ ] 所有测试通过
- [ ] 已测试在不同环境下的兼容性

## 影响范围
说明这次变更可能影响的组件或功能

## 截图（如果适用）
添加相关截图帮助说明变更

## 检查清单
- [ ] 代码风格符合规范
- [ ] 文档已更新
- [ ] 测试覆盖充分
- [ ] 没有引入新的安全风险
```

### 4. Code Review
- 耐心等待维护者审查
- 积极响应审查意见
- 根据反馈进行修改
- 保持讨论的专业性

## 🏷️ 标签系统

### Issue标签
- `bug` - 程序错误
- `enhancement` - 功能增强
- `documentation` - 文档相关
- `good-first-issue` - 适合新手
- `help-wanted` - 需要帮助
- `priority-high` - 高优先级
- `priority-low` - 低优先级

### PR标签
- `ready-for-review` - 准备审查
- `work-in-progress` - 进行中
- `needs-changes` - 需要修改
- `approved` - 已批准

## 🎯 特殊贡献指南

### 添加新工具
1. 在适当的阶段目录下创建工具文件
2. 继承`BaseTool`类
3. 实现`_run`方法
4. 添加详细的文档字符串
5. 编写单元测试
6. 更新工具注册列表

### 添加新阶段
1. 创建新的阶段目录
2. 提供完整的学习计划
3. 实现核心功能
4. 编写详细的README
5. 添加测试用例
6. 更新主README

### 改进现有功能
1. 保持向后兼容性
2. 添加弃用警告（如需要）
3. 更新相关文档
4. 通知可能受影响的用户

## 🌍 国际化

我们欢迎多语言支持的贡献：
- 翻译文档为其他语言
- 添加多语言界面支持
- 改进中文显示效果

## 📧 联系方式

如有任何问题，请通过以下方式联系：

- **GitHub Issues**: 提交问题和建议
- **GitHub Discussions**: 参与讨论和交流
- **Email**: your.email@example.com

## 🙏 致谢

感谢所有贡献者的辛勤工作！您的贡献使这个项目变得更加完善。

特别感谢：
- 核心贡献者们
- 问题反馈者们
- 文档改进者们
- 测试人员们

---

**再次感谢您的贡献！** 🎉

让我们一起构建更好的LangChain Agent学习资源！