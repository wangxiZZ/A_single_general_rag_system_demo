# MCP 接入说明

本文把 RAG 知识库服务接到 Claude Code 或 Claude 桌面端。MCP server 是 **stdio 进程 +
HTTP 转发**：它不连接数据库，也不实现检索，只把三个工具调用转发给已经跑起来的 REST 服务。
因此顺序永远是「先起服务，再起 MCP」。

## 1. 前置条件

1. 服务已启动且就绪：

   ```bash
   docker compose up -d
   curl -s localhost:8000/readyz | python -m json.tool   # ready=true
   ```

2. 安装 MCP 可选依赖（只在接入端需要，主服务镜像不装）：

   ```bash
   uv sync --extra mcp
   ```

3. 自检入口可执行（该命令会阻塞等待 stdio 输入，看到进程不再报错即可 `Ctrl+C`）：

   ```bash
   uv run --extra mcp rag-kb-mcp
   ```

MCP server 读取两个环境变量：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `RAG_KB_BASE_URL` | `http://localhost:8000` | REST 服务地址；MCP 可部署在另一台机器，只连远端 |
| `RAG_KB_TIMEOUT` | `30` | 单次请求超时（秒）；问答含 LLM 生成，网络慢时调大 |

> 不要把模型 API Key 写进 MCP 配置。MCP 只与本项目 REST 通信，Key 属于**服务端** `.env`。

## 2. Claude Code

在仓库根目录执行（`-s user` 表示对所有项目生效；只想本项目生效时改成 `-s project`，
会在仓库生成 `.mcp.json`）：

```bash
claude mcp add -s user --transport stdio \
  rag-kb -e RAG_KB_BASE_URL=http://localhost:8000 \
  -- uv run --extra mcp rag-kb-mcp
```

检查注册结果：

```bash
claude mcp list          # rag-kb: ... - ✓ Connected
```

在会话里用 `/mcp` 查看工具清单，应出现：

- `mcp__rag-kb__list_knowledge_bases`
- `mcp__rag-kb__search_knowledge_base`
- `mcp__rag-kb__ask_knowledge_base`

可以先直接说一句「列出所有知识库」，确认 `list_knowledge_bases` 被调用并返回 JSON。

**Windows 注意**：如果 Claude Code 从桌面快捷方式启动，可能找不到 `uv`。先执行
`where uv` 取得完整路径，再把它写进命令：

```powershell
claude mcp add -s user --transport stdio `
  rag-kb -e RAG_KB_BASE_URL=http://localhost:8000 `
  -- C:\Users\<you>\.local\bin\uv.exe run --extra mcp rag-kb-mcp
```

若 `uv run` 的工作目录不是仓库根，项目无法被解析；必要时先 `cd` 到仓库再注册，
或把 `--directory E:\path\to\rag_system_demo_test` 放在 `uv` 与 `run` 之间。

## 3. Claude 桌面端

编辑配置文件（不存在则新建）：

- macOS：`~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows：`%APPDATA%\Claude\claude_desktop_config.json`

```json
{
  "mcpServers": {
    "rag-kb": {
      "command": "uv",
      "args": [
        "--directory",
        "E:\\WorkSpace\\Format_Project01\\rag_system_demo_test",
        "run",
        "--extra",
        "mcp",
        "rag-kb-mcp"
      ],
      "env": {
        "RAG_KB_BASE_URL": "http://localhost:8000",
        "RAG_KB_TIMEOUT": "60"
      }
    }
  }
}
```

macOS / Linux 把 `command` 换成 `uv` 的绝对路径（`which uv`），`--directory` 换成仓库路径，
并把反斜杠改成正斜杠。保存后**完全退出并重启**桌面端（不是关窗口）。

## 4. 安装 Skill

Skill 让模型知道「什么时候该查库、先列库不猜 id、引用怎么标注」。源文件在
`skills/rag-kb/SKILL.md`，安装目标是项目级 `.claude/skills/rag-kb/`：

```bash
# macOS / Linux / Git Bash
mkdir -p .claude/skills/rag-kb
cp skills/rag-kb/SKILL.md .claude/skills/rag-kb/SKILL.md
```

```powershell
# Windows PowerShell
New-Item -ItemType Directory -Force .claude\skills\rag-kb | Out-Null
Copy-Item skills\rag-kb\SKILL.md .claude\skills\rag-kb\SKILL.md -Force
```

安装后重新打开会话，在 `/skills`（或 `/help` 的 Skill 列表）里应能看到 `rag-kb`。
源文件与安装目标内容必须一致；修改 Skill 后重新复制一次。

## 5. 工具行为

| 工具 | 参数 | 返回 |
|---|---|---|
| `list_knowledge_bases` | 无 | `{"knowledge_bases": [{id, name, description, created_at}]}` |
| `search_knowledge_base` | `kb_id`、`query`、`mode=hybrid`、`top_k=10` | `{"hits": [...], "count": n}`，命中含 `chunk_id` / `doc_name` / `page_no` / `heading_path` / `snippet` / `score` |
| `ask_knowledge_base` | `kb_id`、`question`、`mode=hybrid`、`top_k=8` | `{"answer", "citations", "no_evidence", "attempted_channels"}` |

- `kb_id` 必须先由 `list_knowledge_bases` 取得，不要猜；
- `ask` 返回 `no_evidence=true` 表示服务已拒答，应换关键词或换库重试，不要把拒答文本当答案转述；
- 工具错误以文本返回，包含 HTTP 状态码与服务端 `detail`。

## 6. 常见问题

| 现象 | 原因与处理 |
|---|---|
| `无法连接 RAG 知识库服务` | 服务没起或 `RAG_KB_BASE_URL` 不对；先 `curl localhost:8000/readyz` |
| MCP 显示 Connected 但工具为空 | `uv sync` 没带 `--extra mcp`，或启动目录不是仓库根；用 `--directory` 指到仓库 |
| 桌面端看不到 rag-kb | 配置 JSON 语法错误，或没有完全退出重启；查看 `~/Library/Logs/Claude/mcp*.log`（Windows 在 `%APPDATA%\Claude\logs`） |
| 问答超时 | 首次问答含 LLM 生成，调大 `RAG_KB_TIMEOUT`（如 60–120） |
| 中文显示为乱码 | 终端代码页问题，不是数据问题；`chcp 65001` 或改用 UTF-8 终端 |
| 修改 Skill 后不生效 | 只改了源文件；重新复制到 `.claude/skills/rag-kb/` 并重开会话 |

## 7. 安全边界

- MCP 配置里只有服务地址与超时，**没有模型 Key**；
- MCP 只转发 REST，不新增删除/上传等写操作入口；
- 本项目无鉴权，定位为本机/内网 DEMO；不要把 8000 端口直接暴露到公网。
