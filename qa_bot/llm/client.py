"""本地大模型客户端（对 Ollama OpenAI 兼容接口的封装）。

覆盖参考文档能力：
  - 2.1 第二/三/十段：首次调用、流式 yield、思考过程分离
  - 2.6 第三段：Function Calling（tools）——本地 Ollama 支持 tools，可纯本地闭环

相比百炼示例的差异：base_url 指向本地、api_key 占位、思考字段三兼容
（reasoning_content / reasoning / thinking）。所有 localhost 请求走 trust_env=False，
不被本机代理（Clash）拦截。
"""
import httpx
from openai import OpenAI

from ..config import (
    OLLAMA_BASE_URL,
    OLLAMA_API_KEY,
    GEN_MODEL,
    GEN_PRESETS,
    GEN_PRESET,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    NUM_CTX,
)
from .think import ThinkParser


def _estimate_tokens(messages):
    """粗估 messages 的 token 数（中文按 ~0.65 token/字，ASCII 按 4 字符/token）。

    系数按 qwen2 分词器实测校准：2259 字的中文 system 实际约 1374 token（≈0.6）。
    取 0.65 略偏保守，既不误判超窗，也不会过度裁掉知识/历史。
    """
    total = 0.0
    for m in messages:
        text = m.get("content") or ""
        if not isinstance(text, str):
            text = str(text)
        ascii_chars = sum(1 for ch in text if ord(ch) < 128)
        total += ascii_chars / 4.0 + (len(text) - ascii_chars) * 0.65 + 4
    return int(total)


def _clip_text(text, limit):
    s = (text or "").strip()
    return s if len(s) <= limit else s[:limit].rstrip() + "…"


def _fit_budget(messages, budget):
    """把 messages 压进 budget token，避免超出 Ollama 上下文窗口。

    只动"中间的对话历史"：system（含人设+回答规范）与最后一条
    （当前问题 + 注入的知识块）**永不裁剪**——裁了就等于没知识。
    顺序：先按"一问一答"成对丢最旧的，仍超再把剩余历史逐条截短。
    """
    if not messages or _estimate_tokens(messages) <= budget:
        return messages
    msgs = list(messages)
    head = [msgs[0]] if msgs[0].get("role") == "system" else []
    tail = [msgs[-1]] if len(msgs) - len(head) >= 1 else []
    mid = msgs[len(head): len(msgs) - len(tail)]
    while mid and _estimate_tokens(head + mid + tail) > budget:
        mid = mid[2:] if len(mid) >= 2 else []
    limit = 160
    while mid and _estimate_tokens(head + mid + tail) > budget and limit >= 40:
        mid = [{**m, "content": _clip_text(m.get("content") or "", limit)} for m in mid]
        limit //= 2
    return head + mid + tail


class OllamaClient:
    """封装与本地 Ollama 的对话；方法都吃 messages 列表，方便注入知识/历史/工具。"""

    def __init__(self, base_url=OLLAMA_BASE_URL, api_key=OLLAMA_API_KEY, model=GEN_MODEL):
        self.model = model
        self.client = OpenAI(
            base_url=base_url.rstrip("/") + "/v1",
            api_key=api_key,
            http_client=httpx.Client(
                trust_env=False,  # localhost 直连，不走代理
                timeout=httpx.Timeout(300.0),
            ),
        )

    # -------------------------------------------------------------- 基础调用
    def create(self, messages, tools=None, temperature=DEFAULT_TEMPERATURE,
               top_p=DEFAULT_TOP_P, stream=False, think=None, **extra):
        """最底层：直接调 /v1/chat/completions，返回原始响应（供 Function Calling 用）。

        think: 仅对推理模型（如 deepseek-r1）有效，Ollama 请求级开关。
               None=跟随预设；True/False 显式开/关思考。非标准字段经 extra_body 透传。
        """
        params = dict(
            model=self.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            stream=stream,
        )
        params.update(GEN_PRESETS.get(GEN_PRESET, {}))  # 应用 max_tokens/seed/think 等预设
        params.update(extra)
        # 'think' 非 OpenAI 标准字段，提取后走 extra_body（Ollama 专用）
        if think is None:
            think = params.pop("think", None)
        else:
            params.pop("think", None)
        if tools:
            params["tools"] = tools
        # 把 Ollama 专用字段塞进 extra_body：think（推理开关）+ num_ctx（上下文窗口）。
        # 注意 num_ctx 必须随每次请求下发，否则会回落到 Ollama 默认的 4096，多轮长回答会爆窗。
        extra_body = dict(params.get("extra_body") or {})
        if think is not None:
            extra_body["think"] = think
        extra_body["num_ctx"] = NUM_CTX
        params["extra_body"] = extra_body
        # 关键兜底：本机 iGPU 的 num_ctx 硬顶 4096，超了 Ollama 直接崩进程。
        # 发送前把 messages 压进"窗口 - 输出预留 - 安全余量"的预算内。
        max_out = int(params.get("max_tokens") or 0)
        budget = max(NUM_CTX - max_out - 128, 800)
        params["messages"] = _fit_budget(messages, budget)
        return self.client.chat.completions.create(**params)

    def chat(self, messages, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
             include_reasoning=False, think=None, **extra) -> str:
        """非流式：返回完整回答（默认只取正式回答，过滤思考）。

        **extra 透传给 create()，例如 max_tokens=1500（写作通道需要更长输出）。
        """
        resp = self.create(messages, temperature=temperature, top_p=top_p,
                           stream=False, think=think, **extra)
        msg = resp.choices[0].message
        content = msg.content or ""
        if include_reasoning:
            reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
            if reasoning:
                content += "\n\n【思考过程】\n" + reasoning
        return content

    def stream(self, messages, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
               think=None):
        """流式：逐字产出正式回答（过滤 None，规避原文档提到的 TypeError 坑）。"""
        stream = self.create(messages, temperature=temperature, top_p=top_p,
                             stream=True, think=think)
        for chunk in stream:
            if not chunk.choices:
                continue
            content = getattr(chunk.choices[0].delta, "content", None)
            if content:
                yield content

    def stream_reasoning(self, messages, temperature=DEFAULT_TEMPERATURE, top_p=DEFAULT_TOP_P,
                         think=None):
        """流式 + 思考过程分离（2.1 第十段）。

        产出事件字典：
          {"type": "reasoning",  "text": ...}
          {"type": "answer_start"}
          {"type": "content",   "text": ...}
          {"type": "usage",     "usage": ...}
        """
        stream = self.create(
            messages, temperature=temperature, top_p=top_p,
            stream=True, stream_options={"include_usage": True}, think=think,
        )
        parser = ThinkParser()
        answering = False
        for chunk in stream:
            if not chunk.choices:
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    yield {"type": "usage", "usage": usage}
                continue
            delta = chunk.choices[0].delta
            reasoning = (getattr(delta, "reasoning_content", None)
                         or getattr(delta, "reasoning", None)
                         or getattr(delta, "thinking", None))
            content = getattr(delta, "content", None) or ""
            for kind, text in parser.feed(content):
                if kind == "reasoning":
                    yield {"type": "reasoning", "text": text}
                else:
                    if not answering:
                        answering = True
                        yield {"type": "answer_start"}
                    yield {"type": "content", "text": text}
            if reasoning:
                yield {"type": "reasoning", "text": reasoning}
        for kind, text in parser.flush():
            if kind == "reasoning":
                yield {"type": "reasoning", "text": text}
            else:
                if not answering:
                    answering = True
                    yield {"type": "answer_start"}
                yield {"type": "content", "text": text}

    # -------------------------------------------------------------- 便捷封装（吃 user/system）
    @staticmethod
    def _wrap(user_prompt, system_prompt=None, history=None):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def ask(self, user_prompt, system_prompt=None, history=None, **kw) -> str:
        return self.chat(self._wrap(user_prompt, system_prompt, history), **kw)

    def ask_stream(self, user_prompt, system_prompt=None, history=None, **kw):
        return self.stream(self._wrap(user_prompt, system_prompt, history), **kw)
    def ask_stream_reasoning(self, user_prompt, system_prompt=None, history=None, **kw):
        """流式 + 思考分离的便捷封装（吃 user/system/history）。"""
        return self.stream_reasoning(
            self._wrap(user_prompt, system_prompt, history), **kw
        )
