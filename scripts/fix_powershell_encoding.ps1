# PowerShell UTF-8 编码配置脚本
# 这个脚本会帮你配置PowerShell以正确处理UTF-8编码

Write-Host "正在配置 PowerShell UTF-8 编码支持..." -ForegroundColor Green

# 1. 检查并创建 PowerShell Profile
$profilePath = $PROFILE
$profileDir = Split-Path -Parent $profilePath

if (-not (Test-Path $profileDir)) {
    New-Item -ItemType Directory -Path $profileDir -Force | Out-Null
    Write-Host "✓ 创建了 PowerShell Profile 目录" -ForegroundColor Green
}

# 2. 添加 UTF-8 配置到 Profile
$encodingConfig = @"

# ============================================
# UTF-8 Encoding Configuration
# 自动配置 PowerShell 使用 UTF-8 编码，解决中文乱码问题
# ============================================
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

# 设置 Python 编码环境变量（如果使用 Python）
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONLEGACYWINDOWSSTDIO = "1"

Write-Host "✓ UTF-8 编码已启用" -ForegroundColor Green
"@

# 检查是否已经存在配置
if (Test-Path $profilePath) {
    $content = Get-Content $profilePath -Raw -ErrorAction SilentlyContinue
    if ($content -notmatch "UTF-8 Encoding Configuration") {
        Add-Content -Path $profilePath -Value $encodingConfig
        Write-Host "✓ 已将 UTF-8 配置添加到 PowerShell Profile" -ForegroundColor Green
    } else {
        Write-Host "✓ UTF-8 配置已存在于 PowerShell Profile" -ForegroundColor Yellow
    }
} else {
    Set-Content -Path $profilePath -Value $encodingConfig
    Write-Host "✓ 已创建 PowerShell Profile 并添加 UTF-8 配置" -ForegroundColor Green
}

# 3. 设置系统环境变量（可选，永久生效）
Write-Host "`n是否要设置系统级环境变量？(Y/N): " -ForegroundColor Yellow -NoNewline
$response = Read-Host

if ($response -eq "Y" -or $response -eq "y") {
    # 设置用户级环境变量
    [Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
    [Environment]::SetEnvironmentVariable("PYTHONLEGACYWINDOWSSTDIO", "1", "User")
    
    Write-Host "✓ 已设置系统环境变量（需要重启终端才能生效）" -ForegroundColor Green
    Write-Host "  环境变量已设置，但当前会话不会立即生效" -ForegroundColor Yellow
    Write-Host "  请关闭并重新打开 PowerShell 窗口" -ForegroundColor Yellow
}

Write-Host "`n配置完成！" -ForegroundColor Green
Write-Host "当前会话已启用 UTF-8，请测试查看日志文件：" -ForegroundColor Cyan
Write-Host "  Get-Content `"C:\Users\hpc\.mini-agent\log\*.log`" -Encoding UTF8 | Select-Object -Last 20" -ForegroundColor Gray
Write-Host "`n要永久生效，请：`n1. 重新打开 PowerShell 窗口，或`n2. 运行: . `$PROFILE" -ForegroundColor Yellow
