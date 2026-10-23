"""工具增强智能体（2.6 总装线：分诊 -> 分流 -> 诊治）。

把路由（routing）与工具调度（tools）焊成端到端入口，对外仍是"一个机器人"，
对内是一条自动分流的流水线——这正是文档第七段 ask_llm_route 的本地落地：

  1. 分诊：route(query) 用结构化分类器判断意图（rag / tool / chat），决策透明可见；
  2. 分流：tool -> 调本地工具执行；rag -> 走 2.2/2.5 向量检索问答；chat -> 直接对话；
  3. 诊治：工具/模型产出最终回答。

与文档的差异（本地适配）：
  - 文档用百炼 Assistant API 的"模型自主多轮 tool_calls"；本地 deepseek-r1:7b 原生
    tool_calls 不稳定，故把"决策调哪个工具"收敛到一次路由分类调用（更可靠、更省），
    仍严格遵循"模型只决策、代码执行"的核心思想。
  - 工具实现全部本地确定性（见 tools 包），离线可跑。
"""
from ..config import SYSTEM_PROMPT
from ..llm import OllamaClient
from ..bot.core import CourseQA, _remember
from ..bot.scope import refusal_for
from ..bot.writing import detect_writing_request
from ..routing import route, RouteDecision
from ..tools import dispatch


class ToolAgent:
    """带工具能力的答疑智能体。"""

    def __init__(self, use_kb=True, qa=None):
        # RAG 通道复用已有的 CourseQA（含检索 + 历史）。
        # 上层（CourseQA.ask_smart）会把自己传进来，从而全机器人只有一份历史、
        # 一个检索器——否则 ToolAgent 自建的 CourseQA 会变成"第二本账"，
        # 外层 /new 清不到它，检索器也会被加载两遍。
        self.qa = qa or CourseQA(use_kb=use_kb)
        # 三通道共享同一份历史：直接别名 CourseQA 的列表，
        # 这样工具/闲聊/rag 之间互相看得见上文（否则跨通道即失忆）。
        self.history = self.qa.history
        # 闲聊通道复用自己的客户端（与 qa.client 等价，独立一份便于替换）
        self.client = OllamaClient()
        # 最近一次追问改写（None = 未发生），供上层展示
        self.last_rewrite = None

    # ----------------------------------------------------------- 分诊
    def decide(self, query: str):
        """返回 RouteDecision（路由分诊结果）。"""
        return route(query)

    # ----------------------------------------------------------- 分流 + 诊治
    def answer(self, query: str, decision=None, retrieve_query=None) -> str:
        """按决策执行对应通道，返回最终回答字符串。

        retrieve_query：多轮追问改写后的问题，透传给 RAG 通道做检索；
        工具/闲聊通道用不到（工具参数由路由决定，闲聊直接带历史对话）。
        """
        if decision is None:
            decision = self.decide(query)

        # 2.9：学段越界先拦截——不论路由成 rag/chat/tool，越界问题一律标准化拒答。
        # 放在分流之前，避免"闲聊通道"把越界问题用参数知识自由发挥。
        refusal = refusal_for(retrieve_query or query)
        if refusal:
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", refusal)
            return refusal

        # 2.10：范文请求（覆盖被路由到 chat/tool 的情况）
        wr = detect_writing_request(retrieve_query or query)
        if wr:
            return self.qa._write_sample(query, wr)

        if decision.channel == "tool":
            result = dispatch(decision.tool, decision.arg)
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", result)
            return result

        if decision.channel == "chat":
            answer = self.client.ask(
                query, system_prompt=SYSTEM_PROMPT, history=self.history
            )
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", answer)
            return answer

        # 默认 rag：走知识库问答（带检索与历史）
        return self.qa.ask(query, retrieve_query=retrieve_query)

    def ask(self, query: str):
        """便捷入口：返回 (决策, 回答)。

        多轮追问先改写成独立问题，再拿它去路由和检索——
        否则"那四年级呢？"既路由不准也检索不到。
        """
        self.last_rewrite = None
        # 2.9：越界学段在"追问改写 + 意图路由"之前就短路——这两步各要一次 LLM 调用
        # （路由约 20s），越界题没必要付这个延迟，直接给标准化拒答。
        refusal = refusal_for(query)
        if refusal:
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", refusal)
            return RouteDecision(channel="refuse"), refusal

        # 2.10：索要范文 → 走写作专用通道（跳过路由，省一次 LLM 分类）
        wr = detect_writing_request(query)
        if wr:
            return RouteDecision(channel="write"), self.qa._write_sample(query, wr)

        q_eff = query
        if self.history:
            q_eff = self.qa.effective_query(query)
            if q_eff != query:
                self.last_rewrite = q_eff
        decision = self.decide(q_eff)
        return decision, self.answer(query, decision, retrieve_query=q_eff)

    def reset(self):
        """清空对话历史（与 CourseQA 共享同一列表，清一次即可）。"""
        self.history.clear()
        self.qa.reset()
