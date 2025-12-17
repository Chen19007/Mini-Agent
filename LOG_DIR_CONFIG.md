# 日志目录配置说明

## 配置日志目录

现在可以通过配置文件自定义日志目录位置。

### 配置方法

在配置文件 `config.yaml` 中添加 `log_dir` 配置项：

```yaml
# ===== Agent Configuration =====
max_steps: 100
workspace_dir: "./workspace"
system_prompt_path: "system_prompt.md"
log_dir: "E:/m2_work/log"  # 自定义日志目录
```

### 配置说明

- **`log_dir`**: 日志目录路径
  - 如果设置为 `null` 或不设置，使用默认目录 `~/.mini-agent/log/`
  - 如果设置为路径字符串，将使用该路径作为日志目录
  - 支持相对路径和绝对路径
  - 如果目录不存在，会自动创建

### 配置示例

#### 示例1：使用默认目录（不配置或设置为 null）

```yaml
log_dir: null
# 或者直接不写这行
# 日志将保存在：~/.mini-agent/log/ (Windows: C:\Users\<用户名>\.mini-agent\log\)
```

#### 示例2：使用绝对路径

```yaml
log_dir: "E:/m2_work/log"
# Windows 也可以使用反斜杠
log_dir: "E:\\m2_work\\log"
# 日志将保存在：E:\m2_work\log\
```

#### 示例3：使用相对路径（相对于当前工作目录）

```yaml
log_dir: "./logs"
# 日志将保存在当前目录下的 logs 文件夹
```

### 配置文件位置

配置文件按以下优先级查找：

1. `mini_agent/config/config.yaml` - 开发模式（当前目录）
2. `~/.mini-agent/config/config.yaml` - 用户配置目录
3. `<package>/mini_agent/config/config.yaml` - 包安装目录

### 修改配置后

修改配置后需要：

1. **重新安装工具**（如果使用 `uv tool install`）：
   ```powershell
   uv tool install -e . --force
   ```

2. **重启 ACP 服务器**（如果使用 ACP 模式）

3. **重新运行 Agent**（如果使用 CLI 模式）

### 验证配置

配置生效后，日志文件会保存在你指定的目录中。日志文件名格式为：
```
agent_run_YYYYMMDD_HHMMSS.log
```

例如：`agent_run_20251217_194530.log`

### 查看日志

使用以下命令查看最新日志（记得修改路径）：

```powershell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Get-ChildItem "E:\m2_work\log" | 
    Sort-Object LastWriteTime -Descending | 
    Select-Object -First 1 | 
    Get-Content -Encoding UTF8 | 
    Select-Object -Last 30
```
