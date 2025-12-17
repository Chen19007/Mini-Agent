# 查看最新日志文件（UTF-8编码，解决中文乱码）
# 使用方法：在PowerShell中运行 .\view_log_utf8.ps1 或直接复制命令执行

# 设置 PowerShell 输出编码为 UTF-8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# 查看最新的日志文件（显示最后30行）
Get-ChildItem "C:\Users\hpc\.mini-agent\log" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    Get-Content -Encoding UTF8 | 
    Select-Object -Last 30
