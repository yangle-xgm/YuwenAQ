"""多轮对话的追问改写（2.8）：把"那四年级呢？"还原成独立完整的问题。

为什么需要它：检索只认当前这句话。"五年级呢？"单独拿去检索，
字面上既没有"学业质量"也没有"评价"，必然召不回正确内容——
用户感觉"机器人没有多轮对话"，一半原因是历史没进检索。

两条路：
  1. LLM 改写（默认）：拿最近几轮历史，让模型把追问补全成独立问题，
     检索和路由都用改写后的问题；失败（超时/空输出）自动降级；
  2. 拼接兜底：把上一轮用户问题 + 当前追问拼起来送进检索，零成本兜底。
"""
from ..config import AGENT_MODEL, REWRITE_ENABLED
from ..llm import OllamaClient

# 出现这些词/特征，大概率是依赖上下文的追问
# （刻意不收"再/也"这类单字——"再见"是闲聊不是追问，改写它反而会误导路由）
_FOLLOWUP_HINTS = ("呢", "那样", "那这", "它", "其上", "上述", "前面", "继续", "还有", "同样")
# 常见闲聊/寒暄：短但不是追问，改写给路由添乱，直接放过
_CHAT_PHRASES = ("你好", "您好", "再见", "谢谢", "感谢", "嗨", "哈喽", "在吗", "早上好", "下午好", "晚上好")
_REWRITE_SYSTEM = (
    "你是对话改写器。下面给出一段对话历史和用户的最新追问。"
    "请把最新追问改写成一个独立完整的问题：补全指代（它/那/这个）和省略的主语/话题"
    "（追问常常只换了主语，如'那四年级呢？'实际是问四年级同学的同一个话题），"
    "保留原意，不要回答它。改写结果必须自包含：不看对话历史也能读懂在问什么。"
    "只输出改写后的问题本身，不要解释，不要加引号。"
)


def needs_rewrite(query: str, history) -> bool:
    """判断是否需要改写：有历史 + 看起来是追问。"""
    if not REWRITE_ENABLED or not history:
        return False
    q = (query or "").strip()
    if not q or q in _CHAT_PHRASES:
        return False
    if len(q) >= 20:
        return False  # 长句多半自带主语，是完整问题
    if any(h in q for h in _FOLLOWUP_HINTS):
        return True
    return len(q) <= 6  # 很短的问句也按追问处理（如"第二学段""识字量呢"）


def _fallback(query: str, history) -> str:
    """LLM 改写失败时的兜底：拼上上一轮用户问题。"""
    last_user = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
    )
    return f"{last_user} {query}".strip() if last_user else query


def rewrite_query(query: str, history, client: OllamaClient = None) -> str:
    """把追问改写成独立问题；任何失败都安全降级为拼接或原句。"""
    if not needs_rewrite(query, history):
        return query
    try:
        client = client or OllamaClient(model=AGENT_MODEL)
        conv = "\n".join(
            f"{'用户' if m.get('role') == 'user' else '助手'}:{str(m.get('content', ''))[:200]}"
            for m in history[-6:]
        )
        raw = client.chat(
            [
                {"role": "system", "content": _REWRITE_SYSTEM},
                {"role": "user", "content": f"{conv}\n用户最新追问：{query}"},
            ],
            temperature=0.1,
            think=False,
        ).strip()
        # 只取第一行、去掉引号/书名号，防止模型啰嗦
        line = raw.splitlines()[0].strip().strip("\"'“”『』").strip()
        if line and len(line) <= 120 and line != query:
            return line
    except Exception:  # noqa: BLE001 - 改写失败不应影响主流程
        pass
    return _fallback(query, history)
