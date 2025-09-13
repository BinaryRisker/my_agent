# LangChain Agent 学习项目环境设置脚本
# 适用于 Windows + PowerShell

param(
    [switch]$Force,
    [switch]$SkipVenv,
    [string]$Python = "python"
)

# 设置错误处理
$ErrorActionPreference = "Stop"

Write-Host "🚀 LangChain Agent 学习项目环境设置" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# 检查Python版本
function Test-PythonVersion {
    try {
        $pythonVersion = & $Python --version 2>&1
        Write-Host "✅ 发现Python: $pythonVersion" -ForegroundColor Green
        
        # 提取版本号
        $version = $pythonVersion -replace "Python ", ""
        $versionParts = $version.Split(".")
        $major = [int]$versionParts[0]
        $minor = [int]$versionParts[1]
        
        if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 8)) {
            throw "Python版本过低，需要Python 3.8或更高版本"
        }
        
        return $true
    }
    catch {
        Write-Host "❌ Python检查失败: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "请确保已安装Python 3.8+并添加到PATH环境变量" -ForegroundColor Yellow
        return $false
    }
}

# 创建虚拟环境
function New-VirtualEnvironment {
    param([string]$VenvPath)
    
    if (Test-Path $VenvPath) {
        if ($Force) {
            Write-Host "🔄 删除现有虚拟环境..." -ForegroundColor Yellow
            Remove-Item -Path $VenvPath -Recurse -Force
        } else {
            Write-Host "⚠️  虚拟环境已存在: $VenvPath" -ForegroundColor Yellow
            $continue = Read-Host "是否继续使用现有环境? (y/N)"
            if ($continue -ne "y" -and $continue -ne "Y") {
                Write-Host "❌ 用户取消操作" -ForegroundColor Red
                exit 1
            }
            return $VenvPath
        }
    }
    
    Write-Host "📦 创建Python虚拟环境..." -ForegroundColor Blue
    & $Python -m venv $VenvPath
    
    if ($LASTEXITCODE -ne 0) {
        throw "虚拟环境创建失败"
    }
    
    Write-Host "✅ 虚拟环境创建成功: $VenvPath" -ForegroundColor Green
    return $VenvPath
}

# 激活虚拟环境
function Enable-VirtualEnvironment {
    param([string]$VenvPath)
    
    $activateScript = Join-Path $VenvPath "Scripts\\Activate.ps1"
    
    if (-not (Test-Path $activateScript)) {
        throw "找不到虚拟环境激活脚本: $activateScript"
    }
    
    Write-Host "🔄 激活虚拟环境..." -ForegroundColor Blue
    & $activateScript
    
    # 验证激活
    $currentPython = Get-Command python -ErrorAction SilentlyContinue
    if ($currentPython -and $currentPython.Source.Contains($VenvPath)) {
        Write-Host "✅ 虚拟环境已激活" -ForegroundColor Green
        return $true
    } else {
        Write-Host "⚠️  虚拟环境激活可能失败" -ForegroundColor Yellow
        return $false
    }
}

# 安装Python依赖
function Install-Dependencies {
    Write-Host "📦 安装Python依赖包..." -ForegroundColor Blue
    
    # 升级pip
    Write-Host "升级pip..." -ForegroundColor Gray
    python -m pip install --upgrade pip
    
    # 安装依赖
    if (Test-Path "requirements.txt") {
        Write-Host "从 requirements.txt 安装依赖..." -ForegroundColor Gray
        pip install -r requirements.txt
        
        if ($LASTEXITCODE -eq 0) {
            Write-Host "✅ 依赖安装完成" -ForegroundColor Green
        } else {
            throw "依赖安装失败"
        }
    } else {
        Write-Host "⚠️  未找到 requirements.txt，跳过依赖安装" -ForegroundColor Yellow
    }
}

# 设置环境变量文件
function Set-EnvironmentFile {
    $envExample = ".env.example"
    $envFile = ".env"
    
    if (Test-Path $envExample) {
        if (-not (Test-Path $envFile)) {
            Write-Host "📝 创建环境变量文件..." -ForegroundColor Blue
            Copy-Item $envExample $envFile
            Write-Host "✅ 已创建 .env 文件，请编辑并填入您的API密钥" -ForegroundColor Green
            Write-Host "   主要需要配置:" -ForegroundColor Yellow
            Write-Host "   - OPENAI_API_KEY: OpenAI API密钥" -ForegroundColor Yellow
            Write-Host "   - WEATHER_API_KEY: 天气API密钥(可选)" -ForegroundColor Yellow
        } else {
            Write-Host "✅ .env 文件已存在" -ForegroundColor Green
        }
    } else {
        Write-Host "⚠️  未找到 .env.example 文件" -ForegroundColor Yellow
    }
}

# 创建必要目录
function New-RequiredDirectories {
    $directories = @("logs", "data", "temp")
    
    foreach ($dir in $directories) {
        if (-not (Test-Path $dir)) {
            Write-Host "📁 创建目录: $dir" -ForegroundColor Blue
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
        }
    }
    
    Write-Host "✅ 必要目录检查完成" -ForegroundColor Green
}

# 运行基础测试
function Test-Installation {
    Write-Host "🧪 运行基础测试..." -ForegroundColor Blue
    
    # 测试Python导入
    $testScript = @"
import sys
print(f"Python版本: {sys.version}")

try:
    import langchain
    print(f"✅ LangChain版本: {langchain.__version__}")
except ImportError as e:
    print(f"❌ LangChain导入失败: {e}")
    sys.exit(1)

try:
    import openai
    print("✅ OpenAI库导入成功")
except ImportError as e:
    print(f"❌ OpenAI库导入失败: {e}")
    sys.exit(1)

try:
    import gradio
    print(f"✅ Gradio版本: {gradio.__version__}")
except ImportError as e:
    print(f"❌ Gradio导入失败: {e}")
    sys.exit(1)

print("✅ 基础依赖检查通过")
"@

    $testScript | python
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✅ 安装测试通过" -ForegroundColor Green
        return $true
    } else {
        Write-Host "❌ 安装测试失败" -ForegroundColor Red
        return $false
    }
}

# 显示使用说明
function Show-Usage {
    Write-Host "`n🎉 环境设置完成！" -ForegroundColor Green
    Write-Host "==================" -ForegroundColor Green
    
    Write-Host "`n📖 使用说明:" -ForegroundColor Blue
    Write-Host "1. 激活虚拟环境 (如果尚未激活):" -ForegroundColor White
    Write-Host "   agent_env\\Scripts\\Activate.ps1" -ForegroundColor Gray
    
    Write-Host "`n2. 配置API密钥:" -ForegroundColor White
    Write-Host "   编辑 .env 文件，填入您的 OPENAI_API_KEY" -ForegroundColor Gray
    
    Write-Host "`n3. 运行第一个项目:" -ForegroundColor White
    Write-Host "   cd 01_basic_assistant" -ForegroundColor Gray
    Write-Host "   python main.py --mode cli" -ForegroundColor Gray
    
    Write-Host "`n4. 或启动Web界面:" -ForegroundColor White
    Write-Host "   python main.py --mode web" -ForegroundColor Gray
    
    Write-Host "`n📚 学习路径:" -ForegroundColor Blue
    Write-Host "01_basic_assistant     -> 基础智能助手" -ForegroundColor White
    Write-Host "02_document_analyzer   -> 文档分析Agent" -ForegroundColor White
    Write-Host "03_code_assistant      -> 代码助手Agent" -ForegroundColor White
    Write-Host "04_multi_tool_agent    -> 多工具协作Agent" -ForegroundColor White
    Write-Host "05_data_analysis_agent -> 数据分析Agent" -ForegroundColor White
    Write-Host "06_self_learning_agent -> 自主学习Agent" -ForegroundColor White
    
    Write-Host "`n🆘 获取帮助:" -ForegroundColor Blue
    Write-Host "- 查看项目README: README.md" -ForegroundColor White
    Write-Host "- 查看阶段文档: 01_basic_assistant/README.md" -ForegroundColor White
    Write-Host "- GitHub Issues: https://github.com/yourusername/my_agent/issues" -ForegroundColor White
}

# 主执行逻辑
function Main {
    try {
        # 显示开始信息
        Write-Host "开始环境设置..." -ForegroundColor Blue
        Write-Host "当前目录: $(Get-Location)" -ForegroundColor Gray
        
        # 检查Python
        if (-not (Test-PythonVersion)) {
            exit 1
        }
        
        # 创建虚拟环境
        if (-not $SkipVenv) {
            $venvPath = "agent_env"
            New-VirtualEnvironment -VenvPath $venvPath
            
            # 在当前进程中激活虚拟环境（用于后续安装）
            $env:PATH = "$((Resolve-Path $venvPath).Path)\\Scripts;$env:PATH"
        }
        
        # 安装依赖
        Install-Dependencies
        
        # 设置环境文件
        Set-EnvironmentFile
        
        # 创建必要目录
        New-RequiredDirectories
        
        # 运行测试
        if (Test-Installation) {
            Show-Usage
        } else {
            Write-Host "❌ 环境设置过程中出现问题，请检查错误信息" -ForegroundColor Red
            exit 1
        }
        
    }
    catch {
        Write-Host "❌ 环境设置失败: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "请检查错误信息并重试，或查看文档获取帮助" -ForegroundColor Yellow
        exit 1
    }
}

# 显示帮助信息
function Show-Help {
    Write-Host "LangChain Agent 学习项目环境设置脚本" -ForegroundColor Green
    Write-Host "======================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "用法: .\\setup.ps1 [选项]" -ForegroundColor Blue
    Write-Host ""
    Write-Host "选项:" -ForegroundColor Blue
    Write-Host "  -Force        强制重建虚拟环境" -ForegroundColor White
    Write-Host "  -SkipVenv     跳过虚拟环境创建" -ForegroundColor White
    Write-Host "  -Python <cmd> 指定Python命令 (默认: python)" -ForegroundColor White
    Write-Host "  -Help         显示此帮助信息" -ForegroundColor White
    Write-Host ""
    Write-Host "示例:" -ForegroundColor Blue
    Write-Host "  .\\setup.ps1                    # 标准安装" -ForegroundColor Gray
    Write-Host "  .\\setup.ps1 -Force             # 强制重建环境" -ForegroundColor Gray
    Write-Host "  .\\setup.ps1 -Python python3    # 使用python3命令" -ForegroundColor Gray
}

# 检查是否请求帮助
if ($args -contains "-Help" -or $args -contains "--help" -or $args -contains "/?") {
    Show-Help
    exit 0
}

# 执行主逻辑
Main