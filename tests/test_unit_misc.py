"""单元层：提示词、追问判定、解析器、历史记账与接口契约。

这一批覆盖的是"胶水层"——单看每个函数都很小，但它们错一处，
表现就是"多轮失忆""路由乱跳""流式显示乱码"这类难查的用户可见问题。
"""
import ast
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------ 提示词模板
def test_system_prompt_renders_without_placeholders():
    """模板必须被完整填充，不能残留 {context}/{query}。"""
    from qa_bot.prompts import build_system_prompt

    out = build_system_prompt("五年级学业质量如何检测", "知识块内容")
    assert "知识块内容" in out
    assert "五年级学业质量如何检测" in out
    assert "{context}" not in out and "{query}" not in out


def test_system_prompt_empty_knowledge_fallback():
    """无知识时应填入兜底说明，而不是让模型看到空白。"""
    from qa_bot.prompts import build_system_prompt

    out = build_system_prompt("问题", "")
    assert "（暂无参考知识）" in out


def test_system_prompt_contains_boundary_rules():
    """第 7/8 条（学段边界、范文声明）必须在提示词里。"""
    from qa_bot.prompts import build_system_prompt

    out = build_system_prompt("q", "k")
    assert "超出本知识库" in out
    assert "不包含范文" in out
    assert "不得臆造" in out or "请勿臆造" in out


def test_task_book_template_three_sections():
    """结构化任务书三段式：任务要求 / 输出要求 / 用户输入。"""
    from qa_bot.prompts import build_task_book

    out = build_task_book("审查错别字", "只输出 JSON", "待审文本")
    for seg in ("【任务要求】", "【输出要求】", "【用户输入】"):
        assert seg in out


# ------------------------------------------------------------------ 追问改写判定
@pytest.mark.parametrize(
    "query,expected",
    [
        ("那四年级呢？", True),
        ("它的要求是什么", True),
        ("上述内容还有哪些", True),
        ("第二学段", True),                    # 极短句按追问处理
        ("你好", False),                        # 寒暄不是追问
        ("谢谢", False),
        ("这是一句很长很长的完整问题，长度已经超过二十个字符，不需要改写。", False),
        ("", False),
    ],
)
def test_needs_rewrite(query, expected):
    """追问判定：指代词/极短句→改写；寒暄与完整长句→不改写。"""
    from qa_bot.bot.rewrite import needs_rewrite

    assert needs_rewrite(query, [{"role": "user", "content": "上一轮问题"}]) is expected


def test_needs_rewrite_without_history():
    """没有历史时永远不需要改写。"""
    from qa_bot.bot.rewrite import needs_rewrite

    assert needs_rewrite("那四年级呢？", []) is False


def test_needs_rewrite_can_be_disabled(monkeypatch):
    """REWRITE_ENABLED=0 时应整体关闭。"""
    import qa_bot.bot.rewrite as rw

    monkeypatch.setattr(rw, "REWRITE_ENABLED", False)
    assert rw.needs_rewrite("那四年级呢？", [{"role": "user", "content": "x"}]) is False


def test_fallback_concatenates_last_question():
    """LLM 改写失败时的兜底：拼上上一轮用户问题，保证检索仍有话题词。"""
    from qa_bot.bot.rewrite import _fallback

    hist = [{"role": "user", "content": "如何检测五年级学业质量"},
            {"role": "assistant", "content": "……"}]
    out = _fallback("那四年级呢？", hist)
    assert "如何检测五年级学业质量" in out and "那四年级呢" in out


def test_fallback_without_history_returns_query():
    """无历史时兜底应原样返回。"""
    from qa_bot.bot.rewrite import _fallback

    assert _fallback("随便问问", []) == "随便问问"


# ------------------------------------------------------------------ 路由 JSON 解析
@pytest.mark.parametrize(
    "raw,channel",
    [
        ('{"channel": "rag", "tool": null, "arg": ""}', "rag"),
        ('```json\n{"channel":"tool","tool":"lookup_poem","arg":"李白"}\n```', "tool"),
        ('好的，结果如下：{"channel":"chat"} 以上。', "chat"),
    ],
)
def test_parse_json_extracts_channel(raw, channel):
    """模型输出常带前缀/代码围栏，解析器要能抠出 JSON。"""
    from qa_bot.routing.router import _parse_json

    assert _parse_json(raw).get("channel") == channel


@pytest.mark.parametrize("raw", ["我不想输出 JSON", '{"channel": "rag"', "", "{}"])
def test_parse_json_garbage_returns_dict(raw):
    """非法/空输出统一返回空 dict，由上层回落 rag。"""
    from qa_bot.routing.router import _parse_json

    data = _parse_json(raw)
    assert isinstance(data, dict)
    assert data == {} or "channel" not in data


# ------------------------------------------------------------------ 裁判解析
def test_judge_parse_labels():
    """裁判只应吐出两个固定标签之一。"""
    from qa_bot.evaluation.judge import Judge

    labels = ("有效回答", "无效回答")
    assert Judge._parse("有效回答", labels, "无效回答") == "有效回答"
    assert Judge._parse("无效回答", labels, "无效回答") == "无效回答"
    assert Judge._parse("我认为是有效的", labels, "无效回答") == "有效回答"
    assert Judge._parse("随便说点什么", labels, "无效回答") == "无效回答"


def test_judge_strips_think_block():
    """必须剥离 <think> 思考块，否则思考里的「无效」会污染判定。"""
    from qa_bot.evaluation.judge import _strip_think

    assert _strip_think("<think>这题无效</think>有效回答") == "有效回答"
    assert _strip_think("<think>未闭合的思考") == ""


# ------------------------------------------------------------------ 思考分离器
def test_think_parser_plain_content():
    """纯正文直接归 content。"""
    from qa_bot.llm.think import ThinkParser

    assert ThinkParser().feed("你好") == [("content", "你好")]


def test_think_parser_splits_tag():
    """一行内同时含正文/思考/正文时，应正确切成三段。"""
    from qa_bot.llm.think import ThinkParser

    out = ThinkParser().feed("正文<think>想</think>结尾")
    assert out == [("content", "正文"), ("reasoning", "想"), ("content", "结尾")]


def test_think_parser_holds_partial_tag():
    """标签被流式切断时，半截前缀必须留在缓冲区，不能当正文吐出去。"""
    from qa_bot.llm.think import ThinkParser

    p = ThinkParser()
    assert p.feed("<thi") == []          # 半截标签不输出
    assert p.feed("nk>内部") == [("reasoning", "内部")]


def test_think_parser_flush_remaining():
    """流结束时残留在缓冲区的文本要吐出来，且清空防重复。"""
    from qa_bot.llm.think import ThinkParser

    p = ThinkParser()
    p.feed("<thi")
    assert p.flush() == [("content", "<thi")]
    assert p.flush() == []


# ------------------------------------------------------------------ 历史记账
def test_clip_short_untouched():
    """未超长时不加省略号。"""
    from qa_bot.bot.core import _clip

    assert _clip("abc", 10) == "abc"
    assert _clip(None, 10) == ""


def test_clip_long_truncated():
    """超长裁剪保留开头并加省略号。"""
    from qa_bot.bot.core import _clip

    out = _clip("字" * 500, 100)
    assert len(out) == 101 and out.endswith("…")


def test_remember_caps_turns():
    """历史轮数封顶：超过 HISTORY_MAX_TURNS 轮时丢最旧的。"""
    from qa_bot.bot.core import _remember
    from qa_bot.config import HISTORY_MAX_TURNS

    hist = []
    for i in range(10):
        _remember(hist, "user", f"问{i}")
        _remember(hist, "assistant", f"答{i}")
    assert len(hist) == HISTORY_MAX_TURNS * 2
    assert "问0" not in [m["content"] for m in hist]


def test_remember_clips_single_message():
    """单条消息按 HISTORY_MSG_MAX_CHARS 截断。"""
    from qa_bot.bot.core import _remember
    from qa_bot.config import HISTORY_MSG_MAX_CHARS

    hist = []
    _remember(hist, "assistant", "答" * 2000)
    assert len(hist[0]["content"]) <= HISTORY_MSG_MAX_CHARS + 1


# ------------------------------------------------------------------ 持久化
def _bare_qa():
    """构造一个不加载检索器的 CourseQA（只测历史读写）。"""
    from qa_bot.bot.core import CourseQA

    qa = CourseQA.__new__(CourseQA)
    qa.history = []
    return qa


def test_history_roundtrip(history_file):
    """保存→加载必须逐字还原（跨启动记忆的基础）。"""
    qa = _bare_qa()
    qa.history = [{"role": "user", "content": "五年级学业质量"}, {"role": "assistant", "content": "…"}]
    qa.save_history(history_file)

    qa2 = _bare_qa()
    assert qa2.load_history(history_file) is True
    assert qa2.history == qa.history


def test_load_history_missing_file(history_file):
    """文件不存在时返回 False（首次启动的正常路径）。"""
    assert _bare_qa().load_history(history_file) is False


def test_load_history_corrupt_json(history_file):
    """坏 JSON 不能让程序崩溃，返回 False。"""
    history_file.write_text("{不是合法 json", encoding="utf-8")
    assert _bare_qa().load_history(history_file) is False


def test_load_history_truncates_overlong(history_file):
    """超长历史应被截到 HISTORY_MAX_TURNS 轮，避免启动即超预算。"""
    import json

    from qa_bot.config import HISTORY_MAX_TURNS

    data = [{"role": "user", "content": f"问{i}"} for i in range(20)]
    history_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    qa = _bare_qa()
    qa.load_history(history_file)
    assert len(qa.history) == HISTORY_MAX_TURNS * 2


# ------------------------------------------------------------------ 知识注入上限
class _StubRetriever:
    def __init__(self, pieces):
        self.pieces = pieces

    def retrieve(self, query, top_k=None):
        return [(1.0, f"s{i}.md", t) for i, t in enumerate(self.pieces)]


def _qa_with(pieces):
    from qa_bot.bot.core import CourseQA

    qa = CourseQA.__new__(CourseQA)
    qa.use_kb = True
    qa.retriever = _StubRetriever(pieces)
    qa.history = []
    qa.client = None
    return qa


def test_retrieve_knowledge_respects_context_cap():
    """知识注入必须逐块拼到上限就停（否则撑爆 4096）。"""
    from qa_bot.config import CONTEXT_MAX_CHARS

    qa = _qa_with(["字" * 800 for _ in range(6)])
    out = qa.retrieve_knowledge("q", effective="q")
    assert len(out) <= CONTEXT_MAX_CHARS


def test_retrieve_knowledge_disabled_kb():
    """关闭知识库时返回空串（降级为纯模型问答）。"""
    from qa_bot.bot.core import CourseQA

    qa = CourseQA.__new__(CourseQA)
    qa.use_kb = False
    qa.retriever = None
    qa.history = []
    assert qa.retrieve_knowledge("q", effective="q") == ""


def test_retrieve_knowledge_single_oversized_block():
    """回归（D-7）：单个知识块本身超过上限时也应被裁剪。

    原实现「首块无条件加入」，若某块长于 CONTEXT_MAX_CHARS，
    会把 system 顶到 4096 窗口外（现有语料单块约 500 字，属潜在触发）。
    """
    from qa_bot.config import CONTEXT_MAX_CHARS

    qa = _qa_with(["字" * 5000])
    out = qa.retrieve_knowledge("q", effective="q")
    assert len(out) <= CONTEXT_MAX_CHARS


# ------------------------------------------------------------------ 接口契约（CLI ↔ 核心）
def test_cli_call_sites_match_method_signatures():
    """静态契约检查：cli.py 里对 bot 的所有调用，参数名必须被方法签名接受。

    这类错误 pytest 抓不到（只有真的走到那条分支才炸），所以用 AST 静态扫一遍。
    """
    from qa_bot.bot.core import CourseQA

    tree = ast.parse((ROOT / "qa_bot" / "cli.py").read_text(encoding="utf-8"))
    problems = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)):
            continue
        if f.value.id != "bot" or not hasattr(CourseQA, f.attr):
            continue
        sig = inspect.signature(getattr(CourseQA, f.attr))
        kwargs = {k.arg: None for k in node.keywords if k.arg}
        try:
            sig.bind(None, *([None] * len(node.args)), **kwargs)
        except TypeError as e:
            problems.append(f"cli.py:{node.lineno} bot.{f.attr}(...) -> {e}")
    assert not problems, "CLI 调用与方法签名不一致：\n" + "\n".join(problems)


def test_ask_stream_forwards_think_flag():
    """回归（D-2）：ask_stream 的 think 参数必须真的透传到 LLM 客户端。

    快速模式（CLI 的 /fast）靠 think=False 关闭思考；此前该参数不在签名里，
    一提问就 TypeError。静态契约只能保证"能调用"，这里验证"能生效"。
    """
    from qa_bot.bot.core import CourseQA

    qa = CourseQA(use_kb=False)
    seen = {}

    def fake_ask_stream(user_prompt, system_prompt=None, history=None, **kw):
        seen.update(kw)
        return iter(["回答"])

    qa.client.ask_stream = fake_ask_stream
    assert "".join(qa.ask_stream("你好", think=False)) == "回答"
    assert seen.get("think") is False, f"think 未透传：{seen}"


def test_ask_stream_default_think_follows_preset():
    """不传 think 时应保持 None（由 GEN_PRESETS 决定），不能悄悄改成 True/False。"""
    from qa_bot.bot.core import CourseQA

    qa = CourseQA(use_kb=False)
    seen = {}

    def fake_ask_stream(user_prompt, system_prompt=None, history=None, **kw):
        seen.update(kw)
        return iter(["回答"])

    qa.client.ask_stream = fake_ask_stream
    list(qa.ask_stream("你好"))
    assert seen.get("think") is None, f"默认 think 应为 None：{seen}"
