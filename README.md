# YuwenAQ · 语文答疑机器人

基于《义务教育语文课程标准（2022 年版）》的本地 RAG 答疑助手。后端由本机 Ollama 托管的大模型驱动，无需联网即可问答。

## 功能特性
- **知识库 RAG**：对《语文课程标准》做多信号混合检索（向量 + 关键词）+ 重排（向量相似度 / 词频 IDF / 路径 / 覆盖度 / 稀有词命中）。
- **意图路由**：自动在 `rag`（知识库）/ `tool`（本地工具）/ `chat`（闲聊）三条通道间分流。
- **多轮对话**：追问改写（把"那四年级呢？"补全为独立问题）+ 跨启动对话持久化（`/new` 可清空）。
- **本地工具**：NL→SQL（课程数据库）、古诗检索、文本统计等，全部本地确定性实现。
- **评测**：内置 LLM-as-Judge 自动化测评（`/eval`）。

## 技术栈
- 语言：Python 3.11+
- 向量检索：自研轻量向量库（`qa_bot/rag/`，无需 FAISS/chromadb）
- 大模型：Ollama 本地部署

## 环境要求
1. 安装 [Ollama](https://ollama.com)。
2. 拉取模型：
   ```bash
   ollama pull deepseek-r1:7b   # 生成模型
   ollama pull bge-m3           # 嵌入模型
   ```

## 知识库
课程标准语料已内置在仓库的 `语文课程标准RAG知识库/` 目录（8 个 Markdown 分册）。
如需替换为自己的语料，可设置环境变量 `QA_KB_DIR` 指向任意目录。

## 快速开始
```bash
# 1) 创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 2) 安装依赖
pip install -r requirements.txt

# 3) 启动 Ollama（见下方"已知限制"——上下文上限 4096）
ollama serve                    # 或双击桌面快捷方式 / 系统托盘图标

# 4) 构建向量索引（首次或改动知识库后）
python -c "from qa_bot.rag.index import build_index; build_index()"

# 5) 启动对话
python -m qa_bot.cli            # 交互式；输入 /build 重建索引、/new 开新对话、/exit 退出
```

## 主要配置项（环境变量，均有默认值）
| 变量 | 说明 | 默认 |
|---|---|---|
| `QA_KB_DIR` | 知识库目录 | 内置 `语文课程标准RAG知识库/` |
| `OLLAMA_BASE_URL` | Ollama 地址 | `http://127.0.0.1:11434` |
| `OLLAMA_CONTEXT_LENGTH` | 上下文上限（**见限制**） | `4096` |
| `QA_EMBED_MODEL` | 嵌入模型 | `bge-m3` |
| `QA_GEN_PRESET` | 生成预设 | `factual` |
| `QA_TOP_K` | 召回条数 | `6` |

## 已知限制
- **上下文硬上限 4096 token**：本项目运行的 AMD Radeon 780M 集显（Vulkan 后端）对 `deepseek-r1:7b` 在 `n_ctx > 4096` 时会因驱动/分配问题直接崩溃。多轮与长回答通过"token 预算裁剪"（仅压缩历史、不裁系统提示与知识）实现，不可调大该值。请保持 `OLLAMA_CONTEXT_LENGTH=4096`。
- **单模型常驻**：Ollama 默认只常驻一个模型，bge-m3 嵌入与 deepseek 生成会互相驱逐，单次提问存在数秒模型重载开销（预期行为）。
- **无 NVIDIA 支持**：本机无独显，走 Vulkan iGPU 加速（`OLLAMA_IGPU_ENABLE=1`）；如需 CUDA 请使用相应版本的 Ollama。

## 目录结构
```
qa_bot/
  bot/       对话门面、多轮、追问改写
  rag/       检索、嵌入、索引、切块
  routing/   意图路由
  agents/    工具代理
  tools/     本地工具（SQL/古诗/统计）
  llm/       Ollama 客户端封装
  evaluation/ 评测
  prompts/   提示词模板
语文课程标准RAG知识库/   课程语料（内置）
index/      向量索引（构建生成，不入库）
data/       课程数据库
```

## 许可证
请遵守《义务教育语文课程标准（2022 年版）》的版权与使用规定。
