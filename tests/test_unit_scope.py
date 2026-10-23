"""单元层：学段越界检测（bot/scope.py）测试。

核心风险是双向的：
  - 漏拦 → 越界幻觉（问高一却编通用高中建议）；
  - 误拦 → 正常问题被拒答（"高一级学校""提高考试分数"《大学》）。
所以正常与异常用例必须成对出现。
"""
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query,label",
    [
        ("高一年级语文学习有哪些要求？", "高中"),
        ("高一语文怎么学", "高中"),
        ("高二的古诗文要求", "高中"),
        ("高三语文复习应该怎么安排？", "高中"),
        ("高考作文怎么写", "高中"),
        ("普通高中语文课程标准有什么要求", "高中"),
        ("幼儿园阶段如何进行语言启蒙？", "学前"),
        ("学前教育该教什么", "学前"),
        ("学前儿童的语言发展", "学前"),
        ("大学语文课程的教学目标是什么？", "大学"),
        ("大学阶段的语文学习", "大学"),
        ("考研语文该怎么复习？", "研究生"),
        ("研究生阶段的语文研究", "研究生"),
    ],
)
def test_out_of_scope_detected(query, label):
    """越界学段必须被识别并打上正确标签。"""
    from qa_bot.bot.scope import detect_out_of_scope

    assert detect_out_of_scope(query) == label


@pytest.mark.parametrize(
    "query",
    [
        "高一级学校有哪些选择",          # 升学去向，不是高中
        "如何提高考试分数",              # 「提高」含「高」、「考试」含「考」
        "《大学》的教学价值",            # 四书篇名，属第四学段教学内容
        "五年级学业质量如何检测？",       # 正常义务教育
        "初一写作范本",                  # 正常义务教育
        "第一学段识字量要求是多少",
        "语文课程核心素养包括哪些方面",
    ],
)
def test_normal_queries_not_blocked(query):
    """易误伤表述必须放行——误拦比漏拦更伤体验。"""
    from qa_bot.bot.scope import detect_out_of_scope

    assert detect_out_of_scope(query) is None


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_query_safe(query):
    """空输入必须安全返回 None，不能抛异常。"""
    from qa_bot.bot.scope import detect_out_of_scope

    assert detect_out_of_scope(query) is None


def test_refusal_text_mentions_course_and_guidance():
    """拒答文案要说明覆盖范围 + 指路，不能只说「不知道」。"""
    from qa_bot.bot.scope import refusal_for

    txt = refusal_for("高一年级语文学习有哪些要求？")
    assert txt
    assert "高中" in txt
    assert "义务教育" in txt
    assert "普通高中语文课程标准" in txt


def test_refusal_none_for_normal_query():
    """未越界时必须返回 None（调用方据此决定继续生成）。"""
    from qa_bot.bot.scope import refusal_for

    assert refusal_for("五年级学业质量如何检测？") is None


def test_switch_can_disable_guard(monkeypatch):
    """QA_SCOPE_GUARD=0 时应可整体关闭（便于对照实验）。"""
    import qa_bot.bot.scope as scope

    monkeypatch.setattr(scope, "SCOPE_GUARD_ENABLED", False)
    assert scope.detect_out_of_scope("高一年级语文学习有哪些要求？") is None
