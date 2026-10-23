"""项目统一配置。

所有可调参数集中在这里；既可以直接改，也可以用环境变量覆盖
（项目根目录的 .env 会被自动读取）。对应各参考文档的能力开关也在此收口：
  - 2.1 基础问答 / 2.2 向量知识库 / 2.3 提示词 / 2.4 评测 / 2.5 RAG 调优 / 2.6 插件路由
命名统一：模型用 *_MODEL，开关用 *_ENABLED，路径用 *_DIR/*_PATH。
"""
import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def _load_env() -> None:
    """极简 .env 读取，避免引入额外依赖。仅在不曾设置过时写入。"""
    p = PROJECT_DIR / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_env()


def _bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------- Ollama 本地服务
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "ollama")

# ---------------------------------------------------------------- 模型
GEN_MODEL = os.getenv("QA_GEN_MODEL", "deepseek-r1:7b")
# 嵌入模型：2026-10-21 从 nomic-embed-text（768 维，中文区分度弱）切换到 bge-m3
# （1024 维，中文检索明显更强）。切换后必须重建索引（向量维度变了，旧 .npy 不兼容）。
EMBED_MODEL = os.getenv("QA_EMBED_MODEL", "bge-m3")
# bge-m3 官方建议：检索任务中 query 侧要加指令前缀（文档侧不加），
# 对短查询提升明显。换回其他模型时默认自动置空，也可用环境变量强制指定。
EMBED_QUERY_PREFIX = os.getenv(
    "QA_EMBED_QUERY_PREFIX",
    "为这个句子生成表示以用于检索相关文章：" if EMBED_MODEL.startswith("bge-m3") else "",
)
# 2.4 自动化测评的"裁判"模型：默认复用生成模型；可设 QA_JUDGE_MODEL 换成
# 非推理模型（如 qwen2.5:7b）以提速并避免思考标签干扰。
JUDGE_MODEL = os.getenv("QA_JUDGE_MODEL", GEN_MODEL)

# ---------------------------------------------------------------- 生成参数（2.5：按场景预设）
GEN_PRESETS = {
    # 2.8 修复：factual 的 max_tokens 从 800 提到 1200——deepseek-r1:7b 偏啰嗦，
    # 学业质量这类问题原 800 上限会把回答从半截（"家长反馈："）硬截断，用户感知为"回答不完整"。
    "factual": {"temperature": 0.1, "top_p": 0.8, "max_tokens": 1200, "presence_penalty": 0.0, "seed": 42, "think": False},
    "creative": {"temperature": 0.7, "top_p": 0.9, "max_tokens": 1500, "presence_penalty": 0.6, "seed": None, "think": True},
}
GEN_PRESET = os.getenv("QA_GEN_PRESET", "factual")
DEFAULT_TEMPERATURE = GEN_PRESETS[GEN_PRESET]["temperature"]
DEFAULT_TOP_P = GEN_PRESETS[GEN_PRESET]["top_p"]

# 上下文窗口（token）。**必须保持 4096**：
# 本机 AMD Radeon 780M 的 Vulkan 后端（驱动偏旧，serve 日志 "AMD driver is too old"）
# 对 deepseek-r1:7b 只要 n_ctx > 4096 就在预热阶段崩溃（实测 5120/6144/8192/12288 全崩，
# llama-server exit status 0xe06d7363 / 0xc0000409，业务侧表现为 HTTP 500 "llama-server
# process has terminated"）。4096 是这台机器 iGPU 的安全上限。
# 因此不再靠"加大窗口"解决多轮，改为把历史与知识注入压进 4096 预算
# （见 HISTORY_MSG_MAX_CHARS / CONTEXT_MAX_CHARS，以及 llm/client.py 的 _fit_budget）。
# 注意：Ollama 0.40.5 的 OpenAI 接口会忽略请求体里的 num_ctx，真正生效靠
# 环境变量 OLLAMA_CONTEXT_LENGTH（已设 4096）；这里仍随请求下发一份 best-effort 值。
NUM_CTX = int(os.getenv("QA_NUM_CTX", "4096"))

# ---------------------------------------------------------------- 检索 / 切分（2.5）
CHUNK_SIZE = int(os.getenv("QA_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("QA_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("QA_TOP_K", "6"))
CANDIDATE_K = int(os.getenv("QA_CANDIDATE_K", "24"))
VECTOR_WEIGHT = float(os.getenv("QA_VECTOR_WEIGHT", "0.7"))
LEXICAL_WEIGHT = float(os.getenv("QA_LEXICAL_WEIGHT", "0.3"))
# 2.5 重排（Rerank）阶段各信号的融合权重：向量余弦 + 关键词IDF + 标题路径命中 + query词覆盖率
RERANK_ENABLED = _bool("QA_RERANK_ENABLED", "1")
RERANK_WEIGHTS = {
    "vector": float(os.getenv("QA_RERANK_VECTOR", "0.45")),
    "lexical": float(os.getenv("QA_RERANK_LEXICAL", "0.30")),
    "path": float(os.getenv("QA_RERANK_PATH", "0.20")),
    "coverage": float(os.getenv("QA_RERANK_COVERAGE", "0.05")),
    # 稀有词命中：query 里 IDF 高的实词（如"写字""260万"）是否真的出现在块里。
    # 关键词总分会稀释掉个别高区分度词的贡献（长块靠一堆常用词也能刷分），
    # 单列这一信号把"含生僻实词的正文块"顶上来（本质上是一个软性 must-have 词约束）。
    "rare": float(os.getenv("QA_RERANK_RARE", "0.15")),
}
# 稀有词的判定阈值：query 内 IDF 前 25% 分位、且 IDF 不低于该下限。
RARE_IDF_PERCENTILE = int(os.getenv("QA_RARE_IDF_PERCENTILE", "75"))
RARE_IDF_FLOOR = float(os.getenv("QA_RARE_IDF_FLOOR", "2.0"))
SIMILARITY_CUTOFF = float(os.getenv("QA_SIMILARITY_CUTOFF", "0.2"))
# 2.5+ 检索噪声治理（本轮定向优化）：
#   00_总览与检索指南.md 是"目录树 + 检索指南"这类导航/元信息文件，几乎把全课标的术语
#   都塞在少数几块里，导致向量/关键词/覆盖率三路信号都虚高，常年霸占 top_k。
#   它不是课标正文，不该当答案来源，故给它一个降权系数（仍可被检索到，只是不再压制正文）。
NAV_SOURCES = tuple(os.getenv("QA_NAV_SOURCES", "00_").split(","))
NAV_PRIOR = float(os.getenv("QA_NAV_PRIOR", "0.55"))
# 同一章节最多取几块：防止某个长章节（如"教学研究与教师培训"8 条）用相似分占满 top_k，
# 把别的章节（如"教学建议"）挤出局。0 表示不限制。
SECTION_CAP = int(os.getenv("QA_SECTION_CAP", "2"))
# 2.8 年级/学段语义桥接：用户口语说"五年级"，课标原文只有"第三学段（5~6年级）"，
# 字面零重叠，纯关键词信号全部失效（实测学段描述块 lex=0，被父章节字面含
# "学业质量"的块压制）。这里做三件事：
#   ① GRADE_XUEDUAN：年级→学段名映射（含"第X学段"直述）；
#   ② QUERY_SYNONYMS：口语词→课标用词（"检测"→"评价/测评"），只用于关键词类信号；
#   ③ GRADE_MATCH_BOOST / GRADE_MISS_PENALTY：路径含目标学段的块升权、含其他学段的块降权
#      （乘在最终分上；课标内容是学段专用的，给五年级看第一学段要求就是事实错误）。
GRADE_XUEDUAN = {
    "一年级": "第一学段", "二年级": "第一学段",
    "三年级": "第二学段", "四年级": "第二学段",
    "五年级": "第三学段", "六年级": "第三学段",
    "七年级": "第四学段", "八年级": "第四学段", "九年级": "第四学段",
}
QUERY_SYNONYMS = {
    "检测": ["评价", "测评"],
    "测评": ["评价"],
    "考查": ["评价", "测评"],
    "考核": ["评价"],
    "考试成绩": ["学业水平考试"],
    "考试": ["学业水平考试", "评价"],
}
GRADE_MATCH_BOOST = float(os.getenv("QA_GRADE_MATCH_BOOST", "2.0"))
GRADE_MISS_PENALTY = float(os.getenv("QA_GRADE_MISS_PENALTY", "0.35"))
# 2.5 HyDE（假设性文档嵌入）：先用生成模型编一篇假想答案，再取其向量检索。默认关（慢）。
HYDE_ENABLED = _bool("QA_HYDE_ENABLED", "0")
HYDE_MODEL = os.getenv("QA_HYDE_MODEL", "modelscope.cn/unsloth/DeepSeek-R1-Distill-Qwen-1.5B-GGUF:latest")

# ---------------------------------------------------------------- 路径
DATA_DIR = PROJECT_DIR / "data"
INDEX_DIR = PROJECT_DIR / "index"
# 知识库默认位置：优先用项目同名目录 ../../语文课程标准RAG知识库（本地开发沿用原布局），
# 若克隆后该目录不存在（即不在仓库外的同级目录），回退到仓库内置副本 ./语文课程标准RAG知识库。
# 仍可用环境变量 QA_KB_DIR 强制指定。
_KB_LOCAL = PROJECT_DIR / "语文课程标准RAG知识库"
_KB_PARENT = PROJECT_DIR.parent / "语文课程标准RAG知识库"
KB_DIR = Path(os.getenv("QA_KB_DIR", str(_KB_PARENT if _KB_PARENT.exists() else _KB_LOCAL)))
DEFAULT_SOURCE = os.getenv("QA_SOURCE", str(KB_DIR))
INDEX_PATH = INDEX_DIR / "vectorstore"

# ---------------------------------------------------------------- 2.6 插件 / 工具调用（Function Calling 本地适配）
# 注意：本地 deepseek-r1:7b 的原生 tool_calls 不稳定（实测不触发），
# 故采用「路由分诊 + 提示词式工具调用」等价方案：模型负责决策调哪个工具、提取参数，
# 代码负责执行并回灌——即文档「模型只决策、代码执行」的核心思想。
ROUTER_ENABLED = _bool("QA_ROUTER_ENABLED", "1")   # 是否启用意图路由（rag/tool/chat 分流）
# 多轮追问改写（2.8）：有历史且判定为追问时，先用 LLM 把追问补全成独立问题再检索/路由
REWRITE_ENABLED = _bool("QA_REWRITE_ENABLED", "1")
# 对话历史上限（轮）：本机 iGPU 的 num_ctx 硬顶 4096 token，几轮长回答就爆，
# 只保留最近 N 轮（一问一答算一轮）送进上下文。
HISTORY_MAX_TURNS = int(os.getenv("QA_HISTORY_MAX_TURNS", "4"))
# 单条历史消息的字符上限：助手回答动辄 600~900 字，全量进历史 3~4 轮就吃掉
# 2000+ token。截断保留开头（足够让追问改写抓住话题），大幅省上下文。
HISTORY_MSG_MAX_CHARS = int(os.getenv("QA_HISTORY_MSG_MAX_CHARS", "400"))
# 注入提示词的知识块总字符上限（实测 qwen2 分词约 0.6 token/字，2400 字≈1440 token）。
# 与 4096 窗口的配平（实测值）：
#   system = 人设+规范(~500 字) + 知识(2400 字) ≈ 2900 字 ≈ 1900 tok
#   2 轮历史(单条约 400 字) ≈ 550 tok　→　prompt 合计 ≈ 2450 tok
#   输出预留 1200 tok → 峰值 ≈ 3650 tok，仍在 4096 内，不会触发 context shift。
# 注：不要超过 3000 字——3~4 轮后 system+历史会顶到 4096 上限。
CONTEXT_MAX_CHARS = int(os.getenv("QA_CONTEXT_MAX_CHARS", "2400"))
TOOL_ENABLED = _bool("QA_TOOL_ENABLED", "1")       # 是否启用本地工具（NL2SQL/古诗/文本统计等）
AGENT_MODEL = os.getenv("QA_AGENT_MODEL", GEN_MODEL)  # 路由/工具决策所用模型，默认复用生成模型
TOOL_MAX_TURNS = int(os.getenv("QA_TOOL_MAX_TURNS", "3"))  # ReAct 式工具循环最大轮数

# ---------------------------------------------------------------- 人设（语文小蜜，2.1/2.3）
SYSTEM_PROMPT = (
    "你是「语文小蜜」，一名熟悉《义务教育语文课程标准（2022年版）》的语文课程答疑助手，"
    "服务对象是中小学生、家长和一线语文老师。语气简洁、准确、友好。"
    "若用户的问题依赖对话历史（如追问、代词指代“它/那/这个/同样”），请结合上下文理解其真实含义后再作答。"
)