"""意图路由（2.6 第五~七段：RAG 不是万能的，先分诊再分流）。

对应文档的 get_question_type / ask_llm_route：先用一次轻量 LLM 调用把请求分类，
再分流到不同处理链路。本地适配：候选类别定为三类 + 工具选择合一：

  - rag  ：知识库问答（课标内容、教学建议）——走 2.2/2.5 的向量 RAG。
  - tool ：需要"办事情/查结构化数据"——调用本地工具（NL2SQL/古诗/文本统计/算数）。
  - chat ：闲聊、问候、与课标无关且无工具可答——直接对话，不浪费 RAG。

模型只输出一段 JSON（{"channel","tool","arg"}），代码据此决策。
容错：解析失败或工具名无效时回落到 rag（RAG 是核心能力，最安全）。

设计要点（呼应文档补充知识点）：
  - 给固定候选类别 + 只输出 JSON，把开放题变封闭题，输出可控；
  - 用 Few-shot 示例教模型区分"含课标词但意图不同"的请求；
  - 分类可用更便宜的小模型（本项目复用生成模型，生产可换 1.5B 提速）。
"""
import json
import re

from ..config import AGENT_MODEL, TOOL_ENABLED
from ..llm import OllamaClient
from ..tools import get_tool_specs, get_tool


class RouteDecision:
    """路由决策结果。"""
    def __init__(self, channel: str, tool: str = None, arg: str = None):
        self.channel = channel   # "rag" | "tool" | "chat"
        self.tool = tool         # 工具名或 None
        self.arg = arg or ""     # 传给工具的参数

    def __repr__(self):
        return f"RouteDecision(channel={self.channel!r}, tool={self.tool!r}, arg={self.arg[:20]!r})"


_ROUTER_SYSTEM = """你是「语文小蜜」的意图路由器。请判断用户请求的类型，并只输出一段 JSON。

可选 channel（三选一）：
- "rag"：概念解释、教学建议、课标原文理解、写作/阅读方法类问题，需要查知识库作答。
- "tool"：明确指向"具体量化数字指标"（识字量、阅读量等学段统计）、查古诗/诗人、
         统计文本、算数计算等"动作型/结构化数据"请求。
- "chat"：闲聊、问候、或与语文课标无关且无工具可答。

【判断优先级（重要）】
只有当问题在问"具体数字指标"（多少字、多少篇、多少万字、几个）时才选 tool
（用 query_curriculum_db 查结构化数字，比 RAG 更准）。
注意：提到"学段/年级/课标"不等于量化指标——问学习内容、要求、描述、
怎么检测/如何评价、教学建议的，一律选 rag（知识库里有大段原文，工具只有干巴巴的数字）。

当 channel="tool" 时，必须从下方"可用工具"里选一个 tool 名称，并在 arg 里填写调用参数。
可用工具：
{tool_specs}

输出格式（只输出 JSON，不要解释，不要 markdown 代码块）：
{{"channel": "rag|tool|chat", "tool": "工具名或null", "arg": "传给工具的参数或空串"}}

【Few-shot 示例】
用户：语文课程的核心素养有哪些？
{{"channel": "rag", "tool": null, "arg": ""}}
用户：第一学段要求课外阅读多少万字？
{{"channel": "tool", "tool": "query_curriculum_db", "arg": "第一学段 课外阅读量"}}
用户：各学段识字量要求分别是多少？
{{"channel": "tool", "tool": "query_curriculum_db", "arg": "各学段 识字量 对比"}}
用户：四个学段课外阅读量加起来一共多少？
{{"channel": "tool", "tool": "query_curriculum_db", "arg": "各学段 课外阅读量 求和"}}
用户：李白写过哪些有名的诗？
{{"channel": "tool", "tool": "lookup_poem", "arg": "李白"}}
用户：帮我数一下这段话有多少字：床前明月光，疑是地上霜。
{{"channel": "tool", "tool": "analyze_text", "arg": "床前明月光，疑是地上霜。"}}
用户：《静夜思》的作者是谁？
{{"channel": "tool", "tool": "lookup_poem", "arg": "静夜思"}}
用户：怎么提高小学生的写作兴趣？
{{"channel": "rag", "tool": null, "arg": ""}}
用户：三年级的学习内容主要有哪些？
{{"channel": "rag", "tool": null, "arg": ""}}
用户：如何检测五年级的学业质量？
{{"channel": "rag", "tool": null, "arg": ""}}
用户：学业质量描述是什么样的？
{{"channel": "rag", "tool": null, "arg": ""}}
用户：三年级的识字量要求是多少？
{{"channel": "tool", "tool": "query_curriculum_db", "arg": "第二学段 识字量"}}
用户：你好
{{"channel": "chat", "tool": null, "arg": ""}}
"""


def _parse_json(text: str) -> dict:
    """从模型输出里尽力抠出第一段合法 JSON。"""
    text = text.strip()
    # 去掉可能的 ```json ``` 围栏
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return {}


def route(query: str) -> RouteDecision:
    """对一条用户请求做意图分类，返回 RouteDecision。

    不启用工具时，tool 类请求回落为 rag（让知识库尽量兜底）。
    """
    system = _ROUTER_SYSTEM.format(tool_specs=get_tool_specs() if TOOL_ENABLED else "（工具未启用）")
    client = OllamaClient(model=AGENT_MODEL)
    raw = client.chat(
        [{"role": "system", "content": system},
         {"role": "user", "content": f"用户：{query}"}],
        think=False,
    )
    data = _parse_json(raw)
    channel = str(data.get("channel", "")).strip().lower()
    tool = data.get("tool")
    arg = data.get("arg", "")

    # 工具未启用 / 工具名无效 -> 不能走 tool
    if not TOOL_ENABLED or not (tool and get_tool(tool)):
        if channel == "tool":
            channel = "rag"
            tool = None
    if channel not in ("rag", "tool", "chat"):
        channel = "rag"  # 默认回落 RAG（最安全）
    return RouteDecision(channel=channel, tool=tool if channel == "tool" else None, arg=arg)
