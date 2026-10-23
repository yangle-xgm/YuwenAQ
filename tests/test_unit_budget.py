"""单元层：token 预算裁剪（llm/client.py）测试。

4096 是本机的硬天花板，所有"记不住 / 回答被截断"问题最终都归到这套预算逻辑。
它必须满足三条不变量：
  1. 不超预算时不改动任何消息（避免无谓的信息损失）；
  2. 超预算时 system 与"当前问题"永不裁剪（裁了等于没知识、没提问）；
  3. 无论输入多极端，输出都必须落回预算内（否则 Ollama 直接崩）。
"""
import pytest

pytestmark = pytest.mark.unit


def test_estimate_tokens_ascii():
    """ASCII 按 4 字符/token 估算（每条消息另加 4 的结构开销）。"""
    from qa_bot.llm.client import _estimate_tokens

    n = _estimate_tokens([{"role": "user", "content": "a" * 400}])
    assert 95 <= n <= 105


def test_estimate_tokens_chinese():
    """中文按 0.65 token/字估算。"""
    from qa_bot.llm.client import _estimate_tokens

    n = _estimate_tokens([{"role": "user", "content": "语" * 1000}])
    assert 640 <= n <= 660


def test_estimate_tokens_empty():
    """空输入估 0。"""
    from qa_bot.llm.client import _estimate_tokens

    assert _estimate_tokens([]) == 0
    assert _estimate_tokens([{"role": "user", "content": ""}]) == 4


def test_estimate_tokens_non_str_content():
    """content 非字符串时不应崩溃（多模态消息的兜底）。"""
    from qa_bot.llm.client import _estimate_tokens

    assert _estimate_tokens([{"role": "user", "content": 12345}]) > 0


def test_fit_budget_noop_when_within_budget():
    """未超预算：必须原样返回（不做任何截断）。"""
    from qa_bot.llm.client import _fit_budget

    msgs = [
        {"role": "system", "content": "人设"},
        {"role": "user", "content": "问题"},
    ]
    assert _fit_budget(msgs, 10000) == msgs


def test_fit_budget_empty_list():
    """空列表安全返回。"""
    from qa_bot.llm.client import _fit_budget

    assert _fit_budget([], 100) == []


def test_fit_budget_preserves_system_and_current_question():
    """超预算时，system 与最后一条（当前问题+知识）必须逐字保留。"""
    from qa_bot.llm.client import _fit_budget

    system = {"role": "system", "content": "人设" * 100}
    q = {"role": "user", "content": "当前问题" * 50}
    mid = []
    for i in range(6):  # 6 轮长历史
        mid.append({"role": "user", "content": f"历史问{i}" + "问" * 600})
        mid.append({"role": "assistant", "content": f"历史答{i}" + "答" * 900})
    msgs = [system] + mid + [q]

    out = _fit_budget(msgs, 800)
    assert out[0] == system, "system 被裁了"
    assert out[-1] == q, "当前问题被裁了"
    assert len(out) < len(msgs), "历史没有被裁剪"


def test_fit_budget_extreme_history_lands_in_budget():
    """极端 6 轮、约 1.5 万字的输入必须压回预算内。"""
    from qa_bot.llm.client import _estimate_tokens, _fit_budget

    system = {"role": "system", "content": "人设" * 200}
    q = {"role": "user", "content": "当前问题" * 60}
    mid = []
    for i in range(6):
        mid.append({"role": "user", "content": "问" * 1200})
        mid.append({"role": "assistant", "content": "答" * 1500})
    msgs = [system] + mid + [q]

    before = _estimate_tokens(msgs)
    out = _fit_budget(msgs, 2000)
    after = _estimate_tokens(out)
    assert before > 8000, f"构造的历史不够长：{before}"
    assert after <= 2000, f"裁剪后仍超预算：{after}"


def test_fit_budget_odd_middle_dropped_cleanly():
    """历史条数为奇数时，成对丢弃后应把落单的那条也丢掉，不留半轮。"""
    from qa_bot.llm.client import _fit_budget

    msgs = [
        {"role": "system", "content": "人设"},
        {"role": "user", "content": "长" * 2000},
        {"role": "assistant", "content": "长" * 2000},
        {"role": "user", "content": "落单" + "长" * 2000},
        {"role": "user", "content": "当前问题"},
    ]
    out = _fit_budget(msgs, 600)
    assert out[0]["role"] == "system"
    assert out[-1]["content"] == "当前问题"
    assert all("落单" not in m["content"] for m in out), "落单的历史没被清掉"


def test_fit_budget_single_huge_system_is_kept():
    """预算极小且只有 system 时无法裁剪——此时应原样返回而非丢消息。"""
    from qa_bot.llm.client import _fit_budget

    msgs = [{"role": "system", "content": "人设" * 5000}]
    assert _fit_budget(msgs, 100) == msgs


def test_clip_text_marks_truncation():
    """截断必须留下省略号，便于人工识别"这里被裁过"。"""
    from qa_bot.llm.client import _clip_text

    out = _clip_text("字" * 500, 100)
    assert len(out) <= 102
    assert out.endswith("…")
    assert _clip_text("短", 100) == "短"
