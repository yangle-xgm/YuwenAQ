"""检索层测试（需要 Ollama bge-m3 嵌入，秒级）。

断言分成两类：
  - 回归类：曾经出过 bug、修好后必须一直成立（学段桥接、噪声治理、章节限流）；
  - 质量类：当前实现的真实短板，用断言把差距量化出来（无关查询、写字教学建议）。
"""
import numpy as np
import pytest

pytestmark = pytest.mark.retrieval


# ------------------------------------------------------------------ 索引完整性
def test_index_shape_matches_embed_model(retriever):
    """索引维度必须与 bge-m3 一致（1024），块数约 185。"""
    assert len(retriever.chunks) >= 150
    assert retriever.matrix.shape == (len(retriever.chunks), 1024)


def test_index_vectors_are_normalized(retriever):
    """向量必须已归一化，否则点积不能当余弦用。"""
    norms = np.linalg.norm(retriever.matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_idf_built_for_all_chunks(retriever):
    """IDF 表应覆盖语料里的 bigram（非空）。"""
    assert len(retriever.idf) > 1000


# ------------------------------------------------------------------ 事实召回（回归）
@pytest.mark.parametrize(
    "query,must_have",
    [
        ("第一学段（1~2年级）的识字量要求是多少？", "1600"),
        ("第三学段（5~6年级）的课外阅读总量要求是多少？", "100万"),
        ("第四学段（7~9年级）的课外阅读总量要求是多少？", "260万"),
        ("义务教育语文课程一共设置了几个学习任务群？", "六个"),
    ],
)
def test_fact_recall_in_topk(retriever, query, must_have):
    """关键事实必须出现在 top6 的正文里（检索层的事实命中率基线）。"""
    hits = retriever.retrieve(query)
    blob = "\n".join(t for _, _, t in hits)
    assert must_have in blob, f"未召回关键事实 {must_have}"


def test_core_literacy_all_four_in_topk(retriever):
    """核心素养四要素必须一次召回齐全。"""
    hits = retriever.retrieve("语文课程核心素养包括哪四个方面？")
    blob = "\n".join(t for _, _, t in hits)
    for k in ("文化自信", "语言运用", "思维能力", "审美创造"):
        assert k in blob, f"缺核心素养要素 {k}"


# ------------------------------------------------------------------ 学段桥接（回归）
def test_grade_bridge_junior_one_maps_to_stage_four(retriever):
    """回归 2.10 bug：「初一年级」曾被子串匹配成第一学段，导致召回小学内容。"""
    hits = retriever.retrieve("初一年级写作范本")
    assert hits
    assert "第四学段" in hits[0][2], f"top1 不是第四学段：{hits[0][2][:60]}"


def test_grade_bridge_grade_five_maps_to_stage_three(retriever):
    """「五年级」应映射到第三学段，top1/top2 必须落在第三学段。"""
    hits = retriever.retrieve("如何检测五年级的学业质量？")
    for _, _, text in hits[:2]:
        assert "第三学段" in text, f"学段错配：{text[:60]}"


def test_grade_targets_parser():
    """学段解析器本身要能同时认「初一」「五年级」「第X学段」「5年级」。"""
    from qa_bot.rag.retriever_vector import _grade_targets

    assert _grade_targets("初一写作") == {"第四学段"}
    assert _grade_targets("五年级学业质量") == {"第三学段"}
    assert _grade_targets("第三学段要求") == {"第三学段"}
    assert _grade_targets("5年级识字量") == {"第三学段"}
    assert _grade_targets("核心素养是什么") == set()


# ------------------------------------------------------------------ 噪声治理（回归）
def test_nav_source_never_dominates(retriever):
    """00_总览（目录树/导航）不得出现在写字类查询的 top3。"""
    hits = retriever.retrieve("关于写字教学，语文课程标准提出了哪些教学建议？")
    for _, src, _ in hits[:3]:
        assert not src.startswith("00_"), "导航文件仍占据前三位"


def test_section_cap_limits_same_section(retriever):
    """同一（文件, 二级章节）最多取 SECTION_CAP 块，保证覆盖面。"""
    from collections import Counter

    from qa_bot.config import SECTION_CAP

    for q in ("学业质量描述", "教学建议有哪些", "学习任务群设计"):
        keys = []
        for _, src, text in retriever.retrieve(q):
            path = text[1: text.index("]")] if text.startswith("[") else ""
            depths = [x.strip() for x in path.split(">")]
            keys.append((src, depths[1] if len(depths) > 1 else ""))
        worst = Counter(keys).most_common(1)[0][1]
        assert worst <= SECTION_CAP, f"{q} 同一章节取了 {worst} 块"


# ------------------------------------------------------------------ 边界与参数
def test_empty_query_returns_empty(retriever):
    """空查询直接返回空列表，不打嵌入接口。"""
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []


def test_top_k_parameter_respected(retriever):
    """top_k 参数应能改变返回条数。"""
    assert len(retriever.retrieve("核心素养", top_k=2)) <= 2
    assert len(retriever.retrieve("核心素养", top_k=4)) <= 4


def test_hits_are_sorted_desc(retriever):
    """返回必须按分数降序。"""
    scores = [s for s, _, _ in retriever.retrieve("学业质量评价建议")]
    assert scores == sorted(scores, reverse=True)


def test_hits_contain_path_for_attribution(retriever):
    """每块都要带 [标题路径] 前缀，否则回答没法标出处。"""
    for _, _, text in retriever.retrieve("识字量要求"):
        assert text.startswith("[") and "]" in text


# ------------------------------------------------------------------ 质量差距（已知短板）
def test_irrelevant_query_should_not_return_noise(retriever):
    """【期望行为】与语文课标无关的问题应返回空（或极低分），让上游如实说"未覆盖"。

    当前 SIMILARITY_CUTOFF=0.2 对中文长句几乎不设防（bge-m3 余弦普遍偏高），
    「今天天气怎么样」照样返回 6 条，最高分 0.33——这些噪声会直接进提示词。
    """
    hits = retriever.retrieve("今天天气怎么样")
    assert hits == [], f"无关问题仍返回 {len(hits)} 条噪声，top1={hits[0][0]}"


def test_writing_teaching_advice_reaches_course_impl(retriever):
    """【期望行为】问「写字教学建议」，至少应召回 06_课程实施 的教学建议正文。

    实测 top6 全被 07 附录（字表/FAQ）与 04 任务群占满，真正的教学建议一条没进，
    说明稀有词信号被「附录4 识字、写字教学基本字表」抢走。
    """
    hits = retriever.retrieve("关于写字教学，语文课程标准提出了哪些教学建议？")
    assert any(src.startswith("06_") for _, src, _ in hits), \
        f"教学建议正文未入榜：{[s for _, s, _ in hits]}"
