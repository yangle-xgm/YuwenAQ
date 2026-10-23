"""单元层：范文请求识别（bot/writing.py）测试。

这里要区分三类极易混淆的输入：
  ① 索要范文     「写一篇初一的作文」        → 应走 write 通道
  ② 问写作要求   「第一学段写话要求是什么」  → 应走 RAG
  ③ 问写作方法   「如何写一篇写人作文」      → 应走 RAG（不是直接给一篇作文）
②③ 误判会直接抢占知识库回答，是本模块最大的风险点。
"""
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "query,stage",
    [
        ("写一篇初一的作文", "第四学段"),
        ("给我一篇初二范文", "第四学段"),
        ("初三作文示例文", "第四学段"),
        ("五年级写话范文", "第三学段"),
        ("六年级的范文", "第三学段"),
        ("三年级作文范文", "第二学段"),
        ("一年级写话示例", "第一学段"),
        ("七年级的范文", "第四学段"),
    ],
)
def test_sample_request_with_stage(query, stage):
    """索要范文：应识别为 write 请求，并解析出正确学段（含初中口语别名）。"""
    from qa_bot.bot.writing import detect_writing_request

    got = detect_writing_request(query)
    assert got is not None, f"未识别为范文请求：{query}"
    assert got["stage"] == stage


@pytest.mark.parametrize(
    "query",
    [
        "第一学段写话要求是什么",
        "各学段的写作要求有哪些",
        "如何提高学生的习作水平",
        "初一作文怎么写",
        "作文评语怎么写",
        "第四学段写作字数要求是多少",
        "写作教学建议有哪些",
    ],
)
def test_requirement_questions_not_treated_as_sample(query):
    """问「要求/字数/怎么教」的，一律不是索要范文。"""
    from qa_bot.bot.writing import detect_writing_request

    assert detect_writing_request(query) is None, f"被误判为范文请求：{query}"


@pytest.mark.parametrize(
    "query",
    [
        "如何写一篇写人作文",   # 问方法
        "怎样写一篇好作文",     # 问方法
    ],
)
def test_method_questions_not_treated_as_sample(query):
    """【期望行为】问「如何/怎样写一篇…」是求方法，不该直接给一篇作文。

    当前实现里 _SAMPLE_HINTS 分支没有过 _ASK_ONLY，会把这类问题判成范文请求，
    属于已知缺陷（见日志第十七章）。
    """
    from qa_bot.bot.writing import detect_writing_request

    assert detect_writing_request(query) is None, f"问方法被误判为索要范文：{query}"


@pytest.mark.parametrize("query", ["", "   ", None])
def test_empty_input_safe(query):
    """空输入安全返回 None。"""
    from qa_bot.bot.writing import detect_writing_request

    assert detect_writing_request(query) is None


def test_stage_of_parses_arabic_numeral():
    """阿拉伯数字年级（如 5年级）也应能解析成学段。"""
    from qa_bot.bot.writing import _stage_of

    assert _stage_of("5年级的范文") == "第三学段"
    assert _stage_of("9年级作文") == "第四学段"


def test_stage_of_none_when_unspecified():
    """没提学段时返回 None，由提示词按第四学段兜底。"""
    from qa_bot.bot.writing import _stage_of

    assert _stage_of("写一篇关于秋天的作文") is None


def test_disclaimer_states_not_course_standard():
    """说明文字必须由代码生成并明确「课标未收录范文」，不能依赖模型自觉。"""
    from qa_bot.bot.writing import disclaimer

    txt = disclaimer("第四学段")
    assert "未收录范文" in txt
    assert "参考示例" in txt
    # 第四学段要带上课标的字数硬指标
    assert "500" in txt


def test_disclaimer_without_stage_still_safe():
    """学段未知时说明文字仍应完整生成（不能出现「None」字样）。"""
    from qa_bot.bot.writing import disclaimer

    txt = disclaimer(None)
    assert "未收录范文" in txt
    assert "None" not in txt


def test_writing_system_prompt_constraints():
    """写作 system prompt 必须含学段、字数下限与「只输出正文」约束。"""
    from qa_bot.bot.writing import build_writing_system

    sys_prompt = build_writing_system("写一篇初一的作文", "第四学段", "（课标写作要求）")
    assert "第四学段" in sys_prompt
    assert "500" in sys_prompt
    assert "只输出作文正文" in sys_prompt
    assert "写一篇初一的作文" in sys_prompt


def test_writing_system_prompt_no_stage_fallback():
    """学段缺失时提示词应给出兜底口径，而不是空着。"""
    from qa_bot.bot.writing import build_writing_system

    sys_prompt = build_writing_system("写一篇作文", None, "")
    assert "第四学段" in sys_prompt  # 兜底按第四学段
