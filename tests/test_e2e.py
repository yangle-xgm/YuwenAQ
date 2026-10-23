"""端到端测试：从一句话提问走到最终回答（含路由 / 检索 / 生成）。

分两档：
  - 越界拒答：确定性短路，不调 LLM，秒回 → 默认执行；
  - 需要生成的（RAG / 范文 / 多轮改写）：标记 slow，用 pytest -m slow 运行。
"""
import time

import pytest

pytestmark = pytest.mark.e2e


# ------------------------------------------------------------------ 越界短路（不调 LLM）
@pytest.mark.retrieval
@pytest.mark.parametrize(
    "query,label",
    [
        ("高一年级语文学习有哪些要求？", "高中"),
        ("幼儿园阶段如何进行语言启蒙？", "学前"),
        ("考研语文该怎么复习？", "研究生"),
    ],
)
def test_scope_refusal_short_circuits_everything(query, label):
    """越界问题必须在「改写 + 路由」之前就被拦掉：不进 LLM、不检索、秒回。"""
    from qa_bot.agents import ToolAgent

    agent = ToolAgent()
    t0 = time.time()
    decision, answer = agent.ask(query)
    cost = time.time() - t0

    assert decision.channel == "refuse"
    assert label in answer
    assert "超出本知识库" in answer
    assert cost < 3.0, f"越界拒答耗时 {cost:.1f}s，说明没有短路成功"


@pytest.mark.retrieval
def test_refusal_recorded_in_history():
    """拒答也要记进历史，否则下一轮追问会失去上下文。"""
    from qa_bot.agents import ToolAgent

    agent = ToolAgent()
    agent.ask("高一年级语文学习有哪些要求？")
    assert len(agent.history) == 2
    assert agent.history[0]["role"] == "user"


# ------------------------------------------------------------------ RAG 通道
@pytest.mark.slow
def test_rag_channel_answers_with_citation():
    """正常课标问题：走 rag 通道，回答非空且带出处/学段信息。"""
    from qa_bot.bot.core import CourseQA

    qa = CourseQA()
    t0 = time.time()
    answer = qa.ask("第三学段（5~6年级）的课外阅读总量要求是多少？")
    cost = time.time() - t0

    assert answer.strip()
    assert "100" in answer, f"未答出关键事实 100 万字：{answer[:200]}"
    assert len(answer) > 30
    print(f"\n[RAG] {cost:.1f}s, {len(answer)} 字")


@pytest.mark.slow
def test_writing_channel_declares_not_textbook():
    """范文通道：必须带代码生成的声明，且正文达到第四学段字数下限。"""
    from qa_bot.agents import ToolAgent

    agent = ToolAgent()
    decision, answer = agent.ask("写一篇初一的作文，题目自拟")
    assert decision.channel == "write"
    assert "未收录范文" in answer, "缺少确定性声明"
    body = answer.split("\n\n", 1)[-1]
    assert len(body) >= 500, f"正文仅 {len(body)} 字，未达 500 字下限"
    print(f"\n[写作] {len(answer)} 字")


@pytest.mark.slow
def test_multi_turn_rewrite_bridges_context():
    """多轮：第二轮「那四年级呢？」必须被改写成独立问题，而不是原句去检索。"""
    from qa_bot.agents import ToolAgent

    agent = ToolAgent()
    agent.ask("如何检测五年级的学业质量？")
    decision, answer = agent.ask("那四年级呢？")

    assert agent.last_rewrite, "未发生追问改写"
    assert "四年级" in agent.last_rewrite
    assert decision.channel == "rag"
    print(f"\n[改写] {agent.last_rewrite}")


@pytest.mark.slow
def test_tool_channel_dispatches_curriculum_db():
    """量化指标问题应被路由到工具通道，直接给出结构化答案。"""
    from qa_bot.agents import ToolAgent

    agent = ToolAgent()
    decision, answer = agent.ask("各学段识字量要求分别是多少？")
    assert decision.channel == "tool", f"未走工具通道：{decision}"
    assert "1600" in answer and "3500" in answer
