"""思考/回答分拣器。

不同后端返回"思考过程"的方式不同：
  - 百炼思考型模型：delta.reasoning_content
  - 本地 Ollama（DeepSeek-R1 蒸馏）：delta.reasoning
  - 部分版本：把思考写进 content 的内联 <think>...</think> 标签

ThinkParser 是一个"有状态的分拣员"：来一块文本，判断属于思考还是回答，
交给对应出口（对应 2.1 第十段 is_answering 开关的思想，并兼容内联标签）。
"""


class ThinkParser:
    """
    思考内容解析器（ThinkParser）。
    用途：在"流式"接收大模型输出时，把文本按 <think>...</think> 标签
    切分成两类事件——
        'reasoning'：标签内部的"推理/思考"内容
        'content'  ：标签外部的"正式回答"内容
    因为输入是"一小段一小段"流式到达的，标签可能被截断在两批之间，
    所以这个类用缓冲区（buf）+ 状态机（in_think）来优雅处理半截标签。
    """

    # 标签常量：思考块的开始与结束标记
    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self):
        """
        初始化：建立解析状态。
        in_think：当前是否正处于 <think> 内部（状态位）
        buf     ：尚未"凑齐可吐出"的缓冲文本
        """
        self.in_think = False   # 初始在标签外
        self.buf = ""           # 缓冲为空

    def feed(self, text: str):
        """
        核心方法：喂入流式文本的一小段，立即返回可以吐出的事件列表。
        参数: text (str) 这一批到达的文本片段
        返回: [(kind, text), ...]  kind ∈ {'reasoning', 'content'}
        关键逻辑：边收边解析，能确定的部分立即输出，半截标签留在 buf 里等下一批。
        """
        events = []                 # 本批要返回的事件
        self.buf += text or ""     # 把新文本追加进缓冲区（text 为 None 时按空串处理）

        while True:
            if self.in_think:
                # ---- 当前在 <think> 内部：找结束标签 </think> ----
                i = self.buf.find(self.CLOSE)
                if i == -1:
                    # 缓冲里还没出现结束标签 → 整段都是"推理"，但还没完，
                    # 先全量吐出，并清空 buf（下一批继续累积推理内容）
                    events.append(("reasoning", self.buf))
                    self.buf = ""
                    break
                if i:
                    # 结束标签之前有推理正文 → 先吐出这部分推理
                    events.append(("reasoning", self.buf[:i]))
                # 截掉已处理的推理部分（连同 CLOSE 标签），退出"思考中"状态
                self.buf = self.buf[i + len(self.CLOSE):]
                self.in_think = False

            else:
                # ---- 当前在 <think> 外部：找开始标签 <think> ----
                i = self.buf.find(self.OPEN)
                if i == -1:
                    # 没找到开始标签，但要小心：缓冲区末尾可能是 OPEN 的"半截前缀"
                    # 例如 buf 结尾是 "<thi"，下一批可能补成 "<think>"
                    # 不能急着把 "<thi" 当作 content 吐出去，否则会被错误显示。
                    hold = 0
                    # 从"最多可能的前缀长度"向下检查 buf 尾部
                    for n in range(min(len(self.buf), len(self.OPEN) - 1), 0, -1):
                        if self.OPEN.startswith(self.buf[-n:]):
                            hold = n   # 发现 buf 末尾有 n 个字符是 OPEN 的前缀
                            break
                    # body = 除掉"疑似半截标签前缀"之外的主体内容
                    body = self.buf[: len(self.buf) - hold] if hold else self.buf
                    if body:
                        events.append(("content", body))
                    # 把疑似半截前缀留在 buf 里，等下一批拼接后再判断
                    self.buf = self.buf[len(self.buf) - hold:] if hold else ""
                    break
                if i:
                    # 开始标签之前是正式回答内容 → 先吐出
                    events.append(("content", self.buf[:i]))
                # 截掉 content 部分（连同 OPEN 标签），进入"思考中"状态
                self.buf = self.buf[i + len(self.OPEN):]
                self.in_think = True

        return events

    def flush(self):
        """
        流结束时的收尾方法：把缓冲区里最后剩下的文本一次性吐出。
        返回值: [(kind, text)]  根据当前状态决定归类。
        注意：调用完 flush 后实例应丢弃或重置，否则可能重复吐出。
        """
        if not self.buf:
            return []
        # 若结束时仍在思考区内，剩余全算 reasoning；否则算 content
        kind = "reasoning" if self.in_think else "content"
        out = self.buf
        self.buf = ""           # 清空，防止重复输出
        return [(kind, out)]

