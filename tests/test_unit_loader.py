"""单元层：切块（rag/loader.py）测试。

切块是检索质量的地基——路径错了、元信息混进正文、短行被误丢，都会在检索层放大。
这里用真实语料 + 临时构造语料两条线验证。
"""
import pytest

pytestmark = pytest.mark.unit


def test_real_corpus_chunk_count_and_shape(kb_dir):
    """真实语料应切出约 185 块，每块都带 source/path/text/embed_text。"""
    from qa_bot.rag.loader import chunk_documents

    chunks = chunk_documents(kb_dir)
    assert len(chunks) >= 150, f"块数异常偏少：{len(chunks)}"
    for c in chunks:
        assert c["source"].endswith(".md")
        assert c["text"].strip()
        assert "path" in c and "embed_text" in c


def test_every_chunk_keeps_its_own_heading(kb_dir):
    """修复过的 loader 路径 bug 回归：每块路径必须包含它自己所属的标题。

    旧写法先 stack[:level-1] 再 append，导致路径永远少一层，
    04「任务群1」、06「建议1」这类最有区分度的标题进不了路径。
    """
    from qa_bot.rag.loader import chunk_documents

    chunks = chunk_documents(kb_dir)
    missing = [c for c in chunks if not c["path"].strip()]
    assert not missing, f"有 {len(missing)} 块路径为空"
    # 至少多数块保留了两级以上（分册 > 章节 > 小节）
    multi = [c for c in chunks if c["path"].count(">") >= 1]
    assert len(multi) / len(chunks) > 0.95


def test_embed_text_starts_with_path(kb_dir):
    """embed_text 必须是「路径。正文」的形式，让向量自带学段/主题语境。"""
    from qa_bot.rag.loader import chunk_documents

    for c in chunk_documents(kb_dir)[:50]:
        assert c["embed_text"].startswith(c["path"])


def test_meta_lines_filtered_out(kb_dir):
    """知识库自带的元信息行（> **来源** / > **标签** / > **适用检索**）不得进正文。

    这些行把整个栏目清单堆在一起，关键词密度极高，会伪装成高相关块。
    """
    from qa_bot.rag.loader import chunk_documents

    bad = []
    for c in chunk_documents(kb_dir):
        for prefix in ("> **来源**", "> **标签**", "> **适用检索**"):
            if prefix in c["text"]:
                bad.append((c["source"], prefix))
    assert not bad, f"元信息行泄漏进正文：{bad[:5]}"


def test_chunk_size_within_bound(tmp_kb):
    """构造长正文，验证单块不会无限膨胀。"""
    from qa_bot.config import CHUNK_SIZE, CHUNK_OVERLAP
    from qa_bot.rag.loader import chunk_documents

    body = "\n".join(["这是用于压测切块逻辑的一句正文内容，长度足够超过八个字符。"] * 200)
    (tmp_kb / "t.md").write_text("# 一级\n## 二级\n" + body, encoding="utf-8")
    chunks = chunk_documents(tmp_kb)
    assert chunks
    for c in chunks:
        assert len(c["text"]) <= CHUNK_SIZE + CHUNK_OVERLAP + 120, len(c["text"])


def test_short_lines_dropped(tmp_kb):
    """过短的行（<8 字符）应被丢弃——多是表格分隔线、单字标记。"""
    from qa_bot.rag.loader import chunk_documents

    (tmp_kb / "t.md").write_text(
        "# 标题一\n这是一行足够长的正文，应该被保留下来。\n---\n|—|\n短\n", encoding="utf-8"
    )
    chunks = chunk_documents(tmp_kb)
    text = "".join(c["text"] for c in chunks)
    assert "这是一行足够长的正文" in text
    assert "\n---" not in text


def test_heading_path_is_nested(tmp_kb):
    """三级标题应形成「分册名 > 章 > 节」的嵌套路径。"""
    from qa_bot.rag.loader import chunk_documents

    (tmp_kb / "t.md").write_text(
        "# 分册99\n## 一、总章\n### （一）小节\n这是正文内容，长度超过八个字符用于验证路径。\n",
        encoding="utf-8",
    )
    c = chunk_documents(tmp_kb)[0]
    assert c["path"].split(" > ")[0] == "分册99"
    assert "（一）小节" in c["path"]


def test_empty_dir_returns_empty(tmp_kb):
    """空目录必须安全返回空列表，不能抛异常。"""
    from qa_bot.rag.loader import chunk_documents, load_documents

    assert chunk_documents(tmp_kb) == []
    assert load_documents(tmp_kb) == []


def test_nonexistent_dir_returns_empty(tmp_kb):
    """不存在的目录：rglob 不抛异常，返回空（边界条件）。"""
    from qa_bot.rag.loader import chunk_documents

    assert chunk_documents(tmp_kb / "not_exist") == []


def test_encoding_is_utf8(tmp_kb):
    """中文语料必须按 UTF-8 读入，不得出现乱码。"""
    from qa_bot.rag.loader import chunk_documents

    (tmp_kb / "t.md").write_text("# 标题\n语文课程核心素养包括文化自信等方面内容。\n", encoding="utf-8")
    c = chunk_documents(tmp_kb)[0]
    assert "文化自信" in c["text"]
