# 配置文件目录使用示例

## 概念说明

当使用 `--config` 参数或 `MINI_AGENT_CONFIG_PATH` 环境变量指定配置文件路径后，系统会：

1. 从指定的配置文件路径加载 `config.yaml`
2. **自动从同一目录查找其他配置文件**（如 `system_prompt.md`, `mcp.json`）

这意味着你可以将整个配置目录作为一个单元来管理。

## 推荐的目录结构

```
E:\m2_work\
├── config.yaml          # 主配置文件
├── system_prompt.md     # 系统提示词（可选）
└── mcp.json             # MCP 工具配置（可选）
```

或者使用专门的配置目录：

```
E:\projects\
├── project-a\
│   └── config\
│       ├── config.yaml
│       ├── system_prompt.md
│       └── mcp.json
└── project-b\
    └── config\
        ├── config.yaml
        ├── system_prompt.md
        └── mcp.json
```

## 使用示例

### 示例 1：项目级别的配置

```bash
# 项目 A
cd E:\projects\project-a
mini-agent --config config\config.yaml --workspace .

# 项目 B
cd E:\projects\project-b
mini-agent --config config\config.yaml --workspace .
```

### 示例 2：集中管理的配置

```bash
# 使用集中的配置目录
mini-agent --config E:\m2_work\config.yaml --workspace E:\m2_work
```

### 示例 3：ACP 模式使用配置目录

```powershell
# 设置环境变量指向配置目录中的配置文件
$env:MINI_AGENT_CONFIG_PATH = "E:\m2_work\config.yaml"
mini-agent-acp

# 系统会自动从 E:\m2_work\ 查找 system_prompt.md 和 mcp.json
```

## 配置文件查找逻辑

当指定了配置文件路径 `E:\m2_work\config.yaml` 后，查找其他配置文件时的优先级：

1. **优先从同一目录查找** (`E:\m2_work\`)
   - `E:\m2_work\system_prompt.md`
   - `E:\m2_work\mcp.json`

2. **如果同一目录找不到，使用默认搜索路径**：
   - `mini_agent/config/` (开发模式)
   - `~/.mini-agent/config/` (用户配置)
   - `<package>/mini_agent/config/` (包安装目录)

## 迁移配置文件

如果你想将配置迁移到新目录：

```powershell
# 1. 创建新配置目录
mkdir E:\m2_work\config

# 2. 复制所有配置文件
Copy-Item ~\.mini-agent\config\* E:\m2_work\config\

# 3. 使用新配置
mini-agent --config E:\m2_work\config\config.yaml
```

这样所有配置文件都在同一个目录下，便于备份和管理。
