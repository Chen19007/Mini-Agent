# 自定义配置文件路径使用说明

现在可以通过命令行参数或环境变量指定配置文件路径，而不必使用默认的搜索路径。

## CLI 模式（mini-agent 命令）

### 使用 --config 参数

```bash
# 使用指定的配置文件
mini-agent --config /path/to/config.yaml

# 同时指定工作空间和配置文件
mini-agent --workspace /path/to/workspace --config /path/to/config.yaml

# 使用简写形式
mini-agent -w /path/to/workspace -c /path/to/config.yaml
```

### 示例

```bash
# 使用自定义配置文件
mini-agent --config E:\m2_work\config.yaml

# 使用相对路径
mini-agent --config ./my-config.yaml

# 结合工作空间使用
mini-agent --workspace E:\m2_work --config E:\m2_work\config.yaml
```

## ACP 模式（mini-agent-acp 命令）

由于 ACP 模式使用 stdio 通信，无法直接传递命令行参数，因此使用环境变量来指定配置文件路径。

### 使用环境变量

```powershell
# Windows PowerShell
$env:MINI_AGENT_CONFIG_PATH = "E:\m2_work\config.yaml"
mini-agent-acp

# 或者一行命令
$env:MINI_AGENT_CONFIG_PATH = "E:\m2_work\config.yaml"; mini-agent-acp
```

```bash
# Linux/macOS
export MINI_AGENT_CONFIG_PATH=/path/to/config.yaml
mini-agent-acp

# 或者一行命令
MINI_AGENT_CONFIG_PATH=/path/to/config.yaml mini-agent-acp
```

### 在 Zed 编辑器中配置

如果要在 Zed 编辑器中配置自定义配置文件路径，需要：

1. **设置环境变量**（推荐）

   在启动 Zed 之前设置环境变量，或者在系统环境变量中设置：

   **Windows:**
   ```powershell
   # 临时设置（当前会话）
   $env:MINI_AGENT_CONFIG_PATH = "E:\m2_work\config.yaml"
   
   # 永久设置（用户级）
   [Environment]::SetEnvironmentVariable("MINI_AGENT_CONFIG_PATH", "E:\m2_work\config.yaml", "User")
   ```

   **Linux/macOS:**
   ```bash
   # 临时设置（当前会话）
   export MINI_AGENT_CONFIG_PATH=/path/to/config.yaml
   
   # 永久设置（添加到 ~/.bashrc 或 ~/.zshrc）
   echo 'export MINI_AGENT_CONFIG_PATH=/path/to/config.yaml' >> ~/.bashrc
   ```

2. **修改 Zed 配置**

   在 Zed 的 `settings.json` 中，Agent Server 配置可以包含环境变量（如果 Zed 支持）：

   ```json
   {
     "agent_servers": {
       "mini-agent": {
         "command": "mini-agent-acp",
         "env": {
           "MINI_AGENT_CONFIG_PATH": "E:\\m2_work\\config.yaml"
         }
       }
     }
   }
   ```

   注意：具体的配置方式取决于 Zed 的版本和对环境变量的支持情况。

## 配置优先级

1. **CLI 模式**：
   - `--config` 参数指定的路径（最高优先级）
   - 默认搜索路径（如果未指定）

2. **ACP 模式**：
   - `MINI_AGENT_CONFIG_PATH` 环境变量（如果设置）
   - 默认搜索路径（如果未设置环境变量）

## 重要说明：配置文件目录

**当指定了配置文件路径后，其他配置文件（system_prompt.md, mcp.json 等）会从同一目录加载。**

例如，如果你指定：
```bash
mini-agent --config E:\m2_work\config.yaml
```

那么系统会自动从 `E:\m2_work\` 目录查找：
- `system_prompt.md`
- `mcp.json`
- 以及其他配置文件中引用的相对路径文件

这确保了所有配置文件都在同一个目录下，便于管理和迁移。

## 默认搜索路径

如果不指定配置文件路径，系统会按以下顺序搜索：

1. `mini_agent/config/config.yaml` - 开发模式（当前目录）
2. `~/.mini-agent/config/config.yaml` - 用户配置目录
3. `<package>/mini_agent/config/config.yaml` - 包安装目录

## 配置文件目录结构

当你指定了配置文件路径后，建议将相关配置文件放在同一目录下：

```
E:\m2_work\
├── config.yaml          # 主配置文件
├── system_prompt.md     # 系统提示词文件
└── mcp.json             # MCP 工具配置文件
```

这样系统会从同一目录自动查找这些文件。

## 注意事项

1. **配置文件路径**：
   - 支持绝对路径和相对路径
   - 相对路径相对于当前工作目录
   - 路径中的 `~` 会被展开为用户主目录

2. **配置文件格式**：
   - 必须是有效的 YAML 文件
   - 必须包含必需的字段（如 `api_key`）

3. **配置文件查找顺序**：
   - 如果指定了配置文件路径，相关文件（system_prompt.md, mcp.json）优先从同一目录查找
   - 如果同一目录找不到，才会使用默认搜索路径

4. **错误处理**：
   - 如果指定的配置文件不存在，会显示错误信息
   - 如果配置文件格式错误或缺少必需字段，会显示相应的错误提示

## 完整示例

### 示例 1：CLI 模式使用自定义配置

```bash
# 创建自定义配置文件
cat > E:\m2_work\my-config.yaml << EOF
api_key: "your-api-key"
api_base: "https://api.minimax.io"
model: "MiniMax-M2"
max_steps: 100
workspace_dir: "E:/m2_work"
log_dir: "E:/m2_work/log"
EOF

# 使用自定义配置运行
mini-agent --config E:\m2_work\my-config.yaml --workspace E:\m2_work
```

### 示例 2：ACP 模式使用自定义配置

```powershell
# Windows PowerShell
# 设置环境变量
$env:MINI_AGENT_CONFIG_PATH = "E:\m2_work\my-config.yaml"

# 启动 ACP 服务器（Zed 会自动调用）
# 或者在命令行直接运行测试
mini-agent-acp
```

### 示例 3：项目特定的配置

如果你有多个项目需要不同的配置：

```bash
# 项目 A
cd project-a
mini-agent --config ./config-a.yaml --workspace ./workspace-a

# 项目 B  
cd project-b
mini-agent --config ./config-b.yaml --workspace ./workspace-b
```

## 查看帮助

使用 `--help` 参数查看所有可用选项：

```bash
mini-agent --help
```

