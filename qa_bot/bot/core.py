"""答疑机器人门面（2.1 主线 + 2.2 向量检索 + 2.3 提示词模板化）。

职责：把"人设 + 结构化提示词 + 检索知识 + 多轮历史"拼成一次对话。
系统提示词不再写死在本文件，而是统一从 prompts 包构建（2.3 优化提示词）。
"""
from ..config import HISTORY_MAX_TURNS, HISTORY_MSG_MAX_CHARS, CONTEXT_MAX_CHARS
from ..llm import OllamaClient
from ..prompts import build_system_prompt
from ..rag.index import get_retriever
# from ..rag.loader import load_documents
# from ..rag.retriever import NaiveRetriever


def _clip(text, limit):
    """把文本裁到 limit 字符；超长时保留开头并加省略标记（省上下文，不破坏话题）。"""
    s = (text or "").strip()
    if limit and len(s) > limit:
        return s[:limit].rstrip() + "…"
    return s


def _remember(history, role, content):
    """往对话历史里记一笔，并做轮数封顶 + 单条长度封顶。

    一问一答算一轮；超过 HISTORY_MAX_TURNS 轮就丢最旧的。
    单条按 HISTORY_MSG_MAX_CHARS 截断：助手回答常有 600~900 字，
    全量留在历史里几轮就把 4096 的上下文窗口吃光（本机 iGPU 上限就是 4096）。
    工具通道也复用它，保证全机器人一份账、同一个截断规则。
    """
    history.append({"role": role, "content": _clip(content, HISTORY_MSG_MAX_CHARS)})
    limit = HISTORY_MAX_TURNS * 2
    if len(history) > limit:
        del history[: len(history) - limit]



class CourseQA:
    """
    课程问答类（CourseQA）。
    职责：把"知识库检索"和"大模型问答"组合起来，
    形成一个带上下文记忆、可调用《语文课程标准》知识库的问答机器人。
    """

    def __init__(self, use_kb=True):
        """
        初始化方法（构造器）。
        在创建 CourseQA 实例时自动执行。
        参数:
            use_kb (bool): 是否启用知识库检索，默认 True。
                           False 时退化为"纯模型问答"，不加载/不使用检索器。
        """
        # 创建 Ollama 客户端，封装了对本地大模型（如 Ollama 部署的模型）的调用
        self.client = OllamaClient()

        # 对话历史列表，用于多轮对话的上下文记忆（存储 user / assistant 消息）
        self.history = []

        # 是否启用知识库，保存配置供其他方法判断
        self.use_kb = use_kb

        # 最近一次多轮追问的改写结果（None = 本轮未发生改写），供上层展示
        self.last_rewrite = None

        # 若启用知识库，则加载文档并初始化一个朴素检索器（NaiveRetriever）
        # load_documents() 负责把语料（如《语文课程标准》）读进来
        # NaiveRetriever 负责对 query 做简单匹配召回相关片段
        #self.retriever = NaiveRetriever(load_documents()) if use_kb else None
        self.retriever = get_retriever() if use_kb else None

    def retrieve_knowledge(self, query, effective=None):
        """对外暴露：根据 query 检索知识库，返回拼好的知识块字符串。

        供 _system() 注入提示词，也供 2.4 评测器取回"机器人实际参考了哪些片段"
        （对应 RAGAS 的 context / 阿里文档 response.source_nodes 透视思路）。
        无知识库或未命中时返回空串。

        effective：多轮追问时由上层传入的"改写后独立问题"（见 effective_query）；
        不传则内部自行判断——无历史时就是原句，行为与从前一致。
        """
        if not (self.use_kb and self.retriever):
            return ""
        rq = effective if effective is not None else self.effective_query(query)
        hits = self.retriever.retrieve(rq)
        if not hits:
            return ""
        # 逐块拼接，直到触达 CONTEXT_MAX_CHARS：知识块是"上下文大户"，
        # 本机 iGPU 的 num_ctx 硬顶 4096 token，必须给历史与输出留出余量。
        # 注意每块（含第一块）都要先裁到上限：原实现的首块无条件加入，
        # 一旦换语料或调大 CHUNK_SIZE 出现超长块，system 会被顶出窗口直接 500（D-7）。
        blocks, used = [], 0
        for _, src, txt in hits:
            # _clip 会在末尾补一个"…"，故留 1 字符余量，保证单块不超上限。
            piece = _clip(f"[出处：{src}]\n{txt}", CONTEXT_MAX_CHARS - 1)
            if blocks and used + len(piece) > CONTEXT_MAX_CHARS:
                break
            blocks.append(piece)
            used += len(piece) + 2
        return "\n\n".join(blocks)

    def effective_query(self, query):
        """检索/路由实际使用的 query（2.8）。

        多轮且判定为追问时，先用 LLM 把"那四年级呢？"改写成独立完整的问题，
        再拿去检索/路由；单轮或完整问句则原样返回。
        """
        from .rewrite import rewrite_query

        return rewrite_query(query, self.history, self.client)

    def _scope_refusal(self, query):
        """学段越界检测（2.9）：越界则返回标准拒答文案，否则 None。

        知识库只覆盖义务教育 1~9 年级；高中/学前/大学等越界问题若放行，
        检索会硬返回一堆无关段落，模型随之自由发挥（幻觉）。
        """
        from .scope import refusal_for

        return refusal_for(query)

    def _write_sample(self, query, wr):
        """范文请求的确定性处理（2.10）。

        说明文字由代码生成（`writing.disclaimer`），保证一定出现"课标未收录范文"，
        不依赖 7B 自觉；正文用写作专用提示词，并把该学段写作要求检索出来当约束。
        """
        from .writing import build_writing_system, disclaimer

        stage = (wr or {}).get("stage")
        requirements = ""
        if self.use_kb and self.retriever:
            hits = self.retriever.retrieve(f"{stage or '第四学段'} 写作 表达与交流 要求")
            blocks, used = [], 0
            for _, src, txt in hits[:3]:
                piece = f"[出处：{src}]\n{txt}"
                if blocks and used + len(piece) > CONTEXT_MAX_CHARS:
                    break
                blocks.append(piece)
                used += len(piece) + 2
            requirements = "\n\n".join(blocks)

        system = build_writing_system(query, stage, requirements)
        body = self.client.ask(
            query, system_prompt=system, history=self.history, max_tokens=1500
        )
        answer = disclaimer(stage) + (body or "").strip()
        _remember(self.history, "user", query)
        _remember(self.history, "assistant", answer)
        return answer

    def _writing_request(self, query):
        """范文/写一篇请求检测（2.10）；不是这类请求返回 None。"""
        from .writing import detect_writing_request

        return detect_writing_request(query)

    def _system(self, query, effective=None):
        """
        私有（内部）方法：构建 system prompt（系统提示词）。
        2.3 优化：不再写死字符串，而是调用 prompts.build_system_prompt 套用结构化模板
        （人设 + 回答规范 + {context} 知识占位 + {query} 问题占位）。
        返回值: 拼装好知识库内容的完整 system prompt 字符串。
        """
        # 复用 retrieve_knowledge：有知识就注入，无知识则退化为纯人设问答
        knowledge = self.retrieve_knowledge(query, effective=effective)
        return build_system_prompt(query, knowledge)

    def ask(self, query, retrieve_query=None):
        """
        非流式问答方法：一次性返回完整回答。
        参数: query (str) 用户提问；retrieve_query 可指定检索用的问题
        （多轮场景由 ToolAgent 传入改写后的问题，避免重复改写）。
        返回值: 模型生成的完整回答字符串。
        """
        # 调用客户端的 ask：传入问题、动态构建的 system prompt、以及历史上下文
        # 2.9：先做学段越界拦截，越界直接拒答，不检索、不生成（避免无关段落引发幻觉）
        refusal = self._scope_refusal(retrieve_query or query)
        if refusal:
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", refusal)
            return refusal

        # 2.10：索要范文 → 走写作专用通道（确定性声明 + 课标写作要求约束）
        wr = self._writing_request(retrieve_query or query)
        if wr:
            return self._write_sample(query, wr)

        answer = self.client.ask(query, system_prompt=self._system(query, retrieve_query),
                                 history=self.history)

        # 把这一轮的用户提问和模型回答追加进历史，供后续多轮对话使用
        _remember(self.history, "user", query)
        _remember(self.history, "assistant", answer)

        return answer

    def ask_stream(self, query, retrieve_query=None, think=None):
        """
        流式问答方法：逐字/逐段返回回答（生成器）。
        适用于前端需要"打字机"式实时显示回答的场景。
        使用 yield from 把客户端产生的流式片段逐个向上透传。

        think：推理模型的思考开关（None=跟随 GEN_PRESETS 预设）。CLI 的快速模式
        会传 think=False 关闭思考，此前该参数未在签名里，直接 TypeError（D-2）。
        """
        parts = []
        # 2.9：学段越界直接拒答（流式通道同样短路，不消耗一次生成）
        refusal = self._scope_refusal(retrieve_query or query)
        if refusal:
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", refusal)
            yield refusal
            return
        # 2.10：范文请求不走流式生成（正文需整体拼装说明前缀），一次性产出
        wr = self._writing_request(retrieve_query or query)
        if wr:
            yield self._write_sample(query, wr)
            return
        for piece in self.client.ask_stream(
            query, system_prompt=self._system(query, retrieve_query),
            history=self.history, think=think,
        ):
            parts.append(piece)
            yield piece
        # 流式结束后补记历史（老代码在这里漏记，导致流式通道实际没有多轮）
        _remember(self.history, "user", query)
        _remember(self.history, "assistant", "".join(parts))

    def ask_stream_reasoning(self, query, think=True, retrieve_query=None):
        """
        流式 + 推理问答方法：相比 ask_stream，通常用于"带思考过程"的模型，
        会一并流式返回模型的推理链路（reasoning）与最终回答。
        """
        parts = []
        # 2.9：学段越界直接拒答（带思考的流式通道同样短路）
        refusal = self._scope_refusal(retrieve_query or query)
        if refusal:
            _remember(self.history, "user", query)
            _remember(self.history, "assistant", refusal)
            yield {"type": "answer_start"}
            yield {"type": "content", "text": refusal}
            return
        # 2.10：范文请求一次性产出（带确定性说明前缀）
        wr = self._writing_request(retrieve_query or query)
        if wr:
            text = self._write_sample(query, wr)
            yield {"type": "answer_start"}
            yield {"type": "content", "text": text}
            return
        for ev in self.client.ask_stream_reasoning(
            query, system_prompt=self._system(query, retrieve_query),
            history=self.history, think=think,
        ):
            if ev.get("type") == "content":
                parts.append(ev.get("text", ""))
            yield ev
        # 同上：流式通道也要记历史（只记正式回答，不记思考过程）
        _remember(self.history, "user", query)
        _remember(self.history, "assistant", "".join(parts))

    def reset(self):
        """
        重置方法：清空对话历史。
        用于在切换话题、开始新会话时，丢弃之前的上下文记忆。
        注意用 clear() 而非重新赋值——ToolAgent 与本对象共享同一个列表。
        """
        self.history.clear()

    # ---------------------------------------------------------- 2.8 对话持久化（跨启动记忆）
    def save_history(self, path):
        """把当前对话历史落盘（JSON），供下次启动恢复（解决"两次对话独立存在"）。"""
        try:
            import json
            from pathlib import Path
            Path(path).write_text(
                json.dumps(self.history, ensure_ascii=False), encoding="utf-8"
            )
        except Exception:  # noqa: BLE001 - 持久化失败不应影响主流程
            pass

    def load_history(self, path):
        """从磁盘恢复对话历史；超长则按 HISTORY_MAX_TURNS 截断，避免撑爆上下文。"""
        try:
            import json
            from pathlib import Path
            p = Path(path)
            if not p.exists():
                return False
            data = json.loads(p.read_text(encoding="utf-8")) or []
            limit = HISTORY_MAX_TURNS * 2
            self.history = data[-limit:] if len(data) > limit else data
            return bool(self.history)
        except Exception:  # noqa: BLE001
            return False

    # ---------------------------------------------------------- 2.6 工具增强通道
    def ask_smart(self, query):
        """
        智能路由问答（2.6）：先由意图路由分诊，再分流到
        工具调用 / RAG 知识库 / 纯闲聊 三种通道之一。

        懒导入 agents 包，避免与 agents.tool_agent 形成循环依赖。
        返回 (RouteDecision, 回答字符串)，便于上层展示"路由决策"这一透明信息。
        """
        from ..agents import ToolAgent
        if not hasattr(self, "_tool_agent"):
            # 把 self 传进去：智能体与本对象共享同一份历史与检索器
            self._tool_agent = ToolAgent(use_kb=self.use_kb, qa=self)
        result = self._tool_agent.ask(query)
        # 透出追问改写信息，供上层（CLI）展示"这个问题实际是按什么检索的"
        self.last_rewrite = self._tool_agent.last_rewrite
        return result

