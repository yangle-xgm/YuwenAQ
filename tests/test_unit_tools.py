"""单元层：本地工具（tools/）测试。

四个工具全部是确定性实现，是"离线可复现"的基石，必须覆盖：
正常取值、参数抽取、注入防护、空输入、以及运算符/过滤组合的边界。
"""
import pytest

pytestmark = pytest.mark.unit


# ------------------------------------------------------------------ 注册表
def test_registry_has_four_tools():
    """四个工具都应注册成功。"""
    from qa_bot.tools import list_tools

    names = {t.name for t in list_tools()}
    assert names == {"query_curriculum_db", "lookup_poem", "analyze_text", "calc_expression"}


def test_tool_specs_render_for_router():
    """工具清单要能渲染成给路由提示词的文本（名称+描述+参数齐全）。"""
    from qa_bot.tools import get_tool_specs

    specs = get_tool_specs()
    for name in ("query_curriculum_db", "lookup_poem", "analyze_text", "calc_expression"):
        assert name in specs
    assert "参数：" in specs


def test_dispatch_unknown_tool_is_friendly():
    """未知工具名不能抛异常，要返回可读错误。"""
    from qa_bot.tools import dispatch

    out = dispatch("no_such_tool", "x")
    assert "未找到" in out


def test_dispatch_wraps_tool_exception(monkeypatch):
    """工具内部抛异常时，dispatch 应兜住并返回错误文本（不拖垮对话）。"""
    import qa_bot.tools.registry as reg

    def boom(arg):
        raise RuntimeError("炸了")

    monkeypatch.setitem(reg.TOOL_REGISTRY, "boom", reg.Tool("boom", "d", {}, boom))
    assert "炸了" in reg.dispatch("boom", "")


# ------------------------------------------------------------------ 古诗
@pytest.mark.parametrize(
    "arg,expect_min",
    [("李白", 3), ("唐", 10), ("宋", 5), ("静夜思", 1), ("苏轼", 2)],
)
def test_lookup_poem_hits(arg, expect_min):
    """按作者/朝代/篇名都应能命中，且数量符合内置库。"""
    from qa_bot.tools import dispatch

    out = dispatch("lookup_poem", arg)
    assert out.count("《") >= expect_min, out


def test_lookup_poem_not_found():
    """查不到时应给出友好提示，而不是空字符串。"""
    from qa_bot.tools import dispatch

    out = dispatch("lookup_poem", "莎士比亚")
    assert "未在古诗库中匹配" in out


def test_lookup_poem_empty_arg():
    """空参数应提示用法。"""
    from qa_bot.tools import dispatch

    assert "请提供" in dispatch("lookup_poem", "")


# ------------------------------------------------------------------ 文本统计
def test_analyze_text_counts():
    """字数/汉字数/标点数/段落数/句数五项统计口径要一致。"""
    from qa_bot.tools import dispatch

    out = dispatch("analyze_text", "床前明月光，疑是地上霜。")
    assert "总字符数（含标点/空格）：12" in out
    assert "汉字数：10" in out
    assert "标点数：2" in out
    assert "段落数：1" in out
    assert "句数（按句末标点切分）：1" in out


def test_analyze_text_extracts_after_colon():
    """用户原话带「这段话：…」时应只统计冒号后的正文。"""
    from qa_bot.tools import dispatch

    out = dispatch("analyze_text", "帮我数一下这段话有多少字：床前明月光，疑是地上霜。")
    assert "汉字数：10" in out


def test_analyze_text_multiline_paragraphs():
    """多段文本的段落数应按非空行计。"""
    from qa_bot.tools import dispatch

    out = dispatch("analyze_text", "第一段内容。\n\n第二段内容。")
    assert "段落数：2" in out


def test_analyze_text_empty():
    """空输入提示用法。"""
    from qa_bot.tools import dispatch

    out = dispatch("analyze_text", "")
    assert "请提供" in out


# ------------------------------------------------------------------ 计算
def test_calc_expression_basic():
    """基础四则运算。"""
    from qa_bot.tools import dispatch

    assert "4100" in dispatch("calc_expression", "1600 + 2500")


def test_calc_expression_parentheses():
    """括号优先级正确。"""
    from qa_bot.tools import dispatch

    assert "405" in dispatch("calc_expression", "(5 + 40 + 100 + 260)")


def test_calc_expression_rejects_injection():
    """注入防护：只允许数字与 + - * / ()，其余一律拒绝。"""
    from qa_bot.tools import dispatch

    out = dispatch("calc_expression", "__import__('os').system('dir')")
    assert "已拒绝不安全输入" in out


def test_calc_expression_rejects_names():
    """变量名/函数名同样拒绝。"""
    from qa_bot.tools import dispatch

    assert "已拒绝" in dispatch("calc_expression", "abs(-1)")


def test_calc_expression_div_zero():
    """除零属于可预期异常，应返回失败提示而不是崩溃。"""
    from qa_bot.tools import dispatch

    assert "计算失败" in dispatch("calc_expression", "1/0")


def test_calc_expression_empty():
    """空表达式提示用法。"""
    from qa_bot.tools import dispatch

    assert "请提供" in dispatch("calc_expression", "")


# ------------------------------------------------------------------ 结构化指标
def test_curriculum_point_query(tmp_db):
    """单点查询：第一学段识字量 1600。"""
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "第一学段识字量")
    assert "1600" in out


def test_curriculum_list_query(tmp_db):
    """对比查询：应返回四个学段四行。"""
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "各学段阅读量对比")
    lines = [ln for ln in out.splitlines() if "学段" in ln]
    assert len(lines) == 4
    assert "260" in out


def test_curriculum_sum_query(tmp_db):
    """全学段求和：5+40+100+260 = 405。"""
    from qa_bot.tools import dispatch

    assert "405" in dispatch("query_curriculum_db", "四个学段课外阅读量加起来一共多少")


def test_curriculum_max_query(tmp_db):
    """取最大值：第四学段 260。"""
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "课外阅读量最多的是哪个学段")
    assert "第四学段" in out and "260" in out


def test_curriculum_written_chars(tmp_db):
    """「会写」与「识字」（认识）必须是两列，不能混。"""
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "第二学段会写多少字")
    assert "要求会写的字：1600" in out


def test_curriculum_sum_respects_stage_filter(tmp_db):
    """【期望行为】带学段的求和只应统计该学段。

    当前 _build_sql 在 op=sum 时重建 SQL、直接丢掉 WHERE，导致
    「第一学段课外阅读量求和」返回全表合计 405（正确值应为 5）。
    """
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "第一学段课外阅读量求和")
    assert "405" not in out, f"学段过滤被忽略，返回了全表合计：{out}"
    assert "5" in out


def test_curriculum_avg_query(tmp_db):
    """【期望行为】含「平均」的查询应走聚合，而不是回退成明细列表。

    当前 _detect_op 先判「各学段」，导致「各学段课外阅读量平均」被判成 list，
    avg 分支永远进不去。
    """
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "各学段课外阅读量平均")
    assert "平均的课外阅读量" in out, f"未走平均值聚合：{out}"


def test_curriculum_primary_school_excludes_junior(tmp_db):
    """【期望行为】「小学」只应覆盖前三学段，不应包含第四学段（初中）。"""
    from qa_bot.tools import dispatch

    out = dispatch("query_curriculum_db", "小学课外阅读量要求")
    assert "第四学段" not in out, f"「小学」查询混入了初中：{out}"
