# PowerShell UTF-8 编码乱码解决方案

## 问题说明

在 PowerShell 中查看 UTF-8 编码的日志文件时出现乱码，这是因为 PowerShell 默认使用 GBK 编码，而日志文件使用 UTF-8 编码保存。

## 解决方案

### 方案一：自动化脚本（推荐）

运行提供的脚本：

```powershell
# 运行配置脚本
.\fix_powershell_encoding.ps1
```

脚本会自动：
1. 创建/修改 PowerShell Profile 配置文件
2. 添加 UTF-8 编码设置
3. 可选：设置系统环境变量

### 方案二：手动配置 PowerShell Profile

1. **打开 PowerShell Profile 文件：**
   ```powershell
   notepad $PROFILE
   ```
   如果文件不存在，系统会提示创建。

2. **添加以下内容：**
   ```powershell
   # UTF-8 Encoding Configuration
   [Console]::OutputEncoding = [System.Text.Encoding]::UTF8
   [Console]::InputEncoding = [System.Text.Encoding]::UTF8
   $OutputEncoding = [System.Text.Encoding]::UTF8
   ```

3. **保存并重新加载：**
   ```powershell
   . $PROFILE
   ```

### 方案三：设置系统环境变量（永久解决）

#### 方法 A：通过 PowerShell（用户级）

```powershell
# 设置用户级环境变量
[Environment]::SetEnvironmentVariable("PYTHONIOENCODING", "utf-8", "User")
[Environment]::SetEnvironmentVariable("PYTHONLEGACYWINDOWSSTDIO", "1", "User")
```

#### 方法 B：通过系统设置（系统级）

1. 右键"此电脑" → "属性" → "高级系统设置"
2. 点击"环境变量"
3. 在"用户变量"中添加：
   - 变量名：`PYTHONIOENCODING`
   - 变量值：`utf-8`
   - 变量名：`PYTHONLEGACYWINDOWSSTDIO`
   - 变量值：`1`

### 方案四：临时解决方案（每次使用时）

在查看日志文件时使用：

```powershell
# 方式1：使用 -Encoding UTF8 参数
Get-Content "C:\Users\hpc\.mini-agent\log\agent_run_*.log" -Encoding UTF8

# 方式2：临时设置编码
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-Content "C:\Users\hpc\.mini-agent\log\agent_run_*.log" -Encoding UTF8

# 方式3：查看最新日志
Get-ChildItem "C:\Users\hpc\.mini-agent\log" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    Get-Content -Encoding UTF8
```

## 推荐配置组合

为了彻底解决问题，建议同时配置：

1. **PowerShell Profile**（用于 PowerShell 输出）
2. **系统环境变量**（用于 Python 程序）

这样无论在哪里使用都能正确显示中文。

## 验证配置

配置完成后，运行以下命令验证：

```powershell
# 测试编码设置
[Console]::OutputEncoding
$OutputEncoding
$env:PYTHONIOENCODING

# 查看日志（应该能正常显示中文）
Get-ChildItem "C:\Users\hpc\.mini-agent\log" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    Get-Content -Encoding UTF8 | 
    Select-Object -Last 10
```

## 注意事项

1. **环境变量设置后需要重启终端**才能生效
2. **PowerShell Profile 修改后**需要运行 `. $PROFILE` 或重新打开 PowerShell
3. **系统级环境变量**需要管理员权限，通常只需要用户级即可

## 快速查看日志的命令别名（可选）

在 PowerShell Profile 中添加：

```powershell
# 查看最新日志的快捷命令
function View-LatestLog {
    param([int]$Lines = 50)
    Get-ChildItem "C:\Users\hpc\.mini-agent\log" | 
        Sort-Object LastWriteTime -Descending | 
        Select-Object -First 1 | 
        Get-Content -Encoding UTF8 | 
        Select-Object -Last $Lines
}

# 使用：View-LatestLog 或 View-LatestLog -Lines 100
Set-Alias -Name vlog -Value View-LatestLog
```
