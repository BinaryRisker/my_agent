# 阶段3：代码助手Agent 🤖

智能化代码分析、生成、审查工具，提供全面的代码开发辅助功能。

## ✨ 核心特性

### 🔍 代码分析
- **多语言支持**: Python, JavaScript, TypeScript, Java, C++, Go, Rust
- **AST解析**: 深度分析代码结构和语法树
- **静态分析**: 函数、类、导入依赖关系分析
- **复杂度评估**: 圈复杂度计算和代码质量评分
- **统计信息**: 行数、文件大小、语言分布等

### 🔎 代码审查
- **安全审计**: SQL注入、命令注入、硬编码密钥检测
- **性能分析**: 低效循环、字符串拼接等性能问题
- **代码风格**: 命名规范、行长度、空白行检查
- **质量检查**: 未使用变量、重复代码、复杂度过高
- **详细报告**: Markdown格式的审查报告生成

### ⚡ 代码生成
- **单元测试生成**: 自动生成unittest格式的测试代码
- **文档字符串**: Google、Numpy、Sphinx风格的文档生成
- **代码补全**: 基于LLM的智能代码补全
- **代码重构**: 方法提取、重命名、内联等重构功能
- **模板驱动**: Jinja2模板引擎支持自定义代码模板

### 🖥️ 双重界面
- **Web界面**: 基于Gradio的现代化Web UI
- **CLI工具**: 命令行界面支持批处理和CI/CD集成
- **实时交互**: 支持即时代码分析和生成
- **会话状态**: 保持分析结果和生成历史

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### Web界面模式

```bash
cd 03_code_assistant/src
python main.py --mode web --port 7864
```

访问: http://localhost:7864

### CLI模式

#### 代码分析
```bash
python main.py --mode cli analyze --path /path/to/code --recursive --output analysis_report.md
```

#### 代码审查
```bash
python main.py --mode cli review --path /path/to/code --recursive --output review_report.md
```

#### 生成测试
```bash
python main.py --mode cli generate-test --path source_file.py --output test_file.py
```

## 📋 功能详解

### 代码分析器 (CodeAnalyzer)

```python
from code_analyzer import CodeAnalyzer

analyzer = CodeAnalyzer()

# 分析单个文件
result = analyzer.analyze_file("example.py")

# 分析目录
result = analyzer.analyze_directory("src/", recursive=True)

# 生成报告
report = analyzer.generate_report(result)
```

**支持的分析指标:**
- 文件统计: 行数、大小、字符数
- 函数分析: 参数数量、文档字符串、装饰器
- 类分析: 方法数量、继承关系、属性
- 复杂度: 圈复杂度计算
- 导入分析: 依赖关系和模块使用

### 代码生成器 (CodeGenerator)

```python
from code_generator import CodeGenerator

generator = CodeGenerator()

# 生成单元测试
test_code = generator.generate_unit_tests("source.py", target_class="MyClass")

# 生成文档字符串
docstring = generator.generate_docstring(function_code, style="google")

# 代码补全
completed = generator.complete_code(incomplete_code, context="web scraping")

# 代码重构
refactored = generator.refactor_code(source_code, "rename", {
    "old_name": "oldFunction",
    "new_name": "newFunction"
})
```

**代码模板:**
- Python单元测试模板
- Python类/函数模板
- 自定义Jinja2模板支持

### 代码审查器 (CodeReviewer)

```python
from code_reviewer import CodeReviewer

reviewer = CodeReviewer()

# 审查文件
issues = reviewer.review_file("example.py")

# 审查目录
results = reviewer.review_directory("src/", recursive=True)

# 生成报告
report = reviewer.generate_report(results)
```

**问题检测类型:**
- **安全问题**: SQL注入、命令注入、硬编码密钥
- **性能问题**: 低效循环、字符串拼接
- **代码风格**: 命名规范、行长度
- **可维护性**: 复杂度过高、未使用变量
- **设计问题**: 代码重复、架构缺陷

## 🎯 Web界面功能

### 代码分析页面
- 路径输入和选项配置
- 实时分析结果显示
- JSON格式详细数据展示
- 支持文件和目录分析

### 代码审查页面
- 递归审查配置
- 问题严重程度分类
- 详细问题报告
- 修复建议提供

### 测试生成页面
- 源文件选择
- 目标类指定
- 覆盖率目标设置
- 测试代码实时生成

### 文档生成页面
- 函数代码输入
- 多种文档风格选择
- 实时文档生成预览

### 代码补全页面
- 多语言代码补全
- 上下文信息支持
- 智能补全建议

## 🔧 配置选项

### 代码风格规则
```python
style_rules = {
    'naming': {
        'class': r'^[A-Z][a-zA-Z0-9]*$',
        'function': r'^[a-z_][a-z0-9_]*$',
        'variable': r'^[a-z_][a-z0-9_]*$',
        'constant': r'^[A-Z_][A-Z0-9_]*$'
    },
    'line_length': 88,
    'max_complexity': 10
}
```

### 安全检查模式
```python
security_patterns = {
    'sql_injection': [
        r'execute\s*\(\s*["\'].*\+.*["\']',
        r'cursor\.execute\s*\(\s*f["\']'
    ],
    'command_injection': [
        r'os\.system\s*\(',
        r'subprocess\.call\s*\('
    ]
}
```

## 📊 输出示例

### 分析报告
```markdown
# 代码文件分析报告

**文件**: example.py
**语言**: python
**行数**: 150
**文件大小**: 4096 字节

**复杂度**: 8 (Moderate)
**函数数量**: 12
**类数量**: 3
```

### 审查报告
```markdown
# 代码审查报告

**审查文件数**: 25
**总问题数**: 18

## 严重程度分布
- Critical: 2
- High: 5
- Medium: 7
- Low: 4
```

## 🧪 测试用例

```python
# 运行所有测试
python -m pytest tests/ -v

# 测试特定模块
python -m pytest tests/test_analyzer.py
python -m pytest tests/test_generator.py
python -m pytest tests/test_reviewer.py
```

## 🛠️ 开发指南

### 添加新的检查规则
```python
# 在 CodeReviewer 中添加新的模式
new_pattern = {
    'custom_issue': [
        r'your_regex_pattern_here'
    ]
}
```

### 扩展代码模板
```python
# 在 CodeGenerator 中添加新模板
custom_template = '''
{{ custom_content }}
'''
```

### 支持新编程语言
```python
# 在 CodeAnalyzer 中添加语言支持
supported_languages['.ext'] = 'language_name'
```

## 🔌 API集成

### REST API (可选)
```python
from fastapi import FastAPI
from main import CodeAssistantAgent

app = FastAPI()
agent = CodeAssistantAgent()

@app.post("/analyze")
async def analyze_code(request: AnalyzeRequest):
    return agent.analyze_code(request.path)
```

### 插件接口
```python
class CustomAnalyzer:
    def analyze(self, code: str) -> Dict[str, Any]:
        # 自定义分析逻辑
        return analysis_result
```

## 📈 性能优化

- **并行处理**: 多文件分析支持并行处理
- **缓存机制**: 分析结果智能缓存
- **增量分析**: 仅分析变更的文件
- **内存优化**: 大文件流式处理

## 🔄 持续集成

### GitHub Actions
```yaml
- name: Code Analysis
  run: |
    python 03_code_assistant/src/main.py --mode cli analyze \
      --path . --recursive --output analysis.md
```

### Jenkins Pipeline
```groovy
stage('Code Review') {
    steps {
        sh 'python main.py --mode cli review --path src/ --output review.md'
    }
}
```

## 🌟 最佳实践

1. **定期审查**: 在每次提交前运行代码审查
2. **测试覆盖**: 为新功能自动生成单元测试
3. **文档维护**: 使用文档生成器保持文档更新
4. **性能监控**: 定期检查代码复杂度和性能问题
5. **安全扫描**: 在生产部署前进行安全审计

## 🤝 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交变更
4. 创建 Pull Request

## 📝 更新日志

### v1.0.0
- 初始版本发布
- 支持Python代码分析、审查、生成
- Web界面和CLI工具
- 多种代码质量检查规则
- 模板化代码生成

### v1.1.0 (计划中)
- 支持更多编程语言
- 增强的性能分析
- 自定义规则配置
- API接口完善

## 📄 许可证

MIT License - 详见 LICENSE 文件

## 🆘 支持

如有问题或建议，请创建 Issue 或联系开发团队。

---

**代码助手Agent** - 让代码开发更智能、更高效！ 🚀