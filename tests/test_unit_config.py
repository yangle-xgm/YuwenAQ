"""单元层：配置参数回归测试（config.py）。

目的：把「曾经踩过坑、改过又被改回去」的关键参数钉住。
这些值一旦被误改，会引发整机崩溃或回答截断，属于最高优先级回归项。
"""
import pytest

pytestmark = pytest.mark.unit


def test_num_ctx_must_stay_4096():
    """上下文窗口必须锁死 4096：本机 780M 的 Vulkan 后端 >4096 直接崩进程。"""
    from qa_bot.config import NUM_CTX

    assert NUM_CTX == 4096


def test_factual_max_tokens_not_too_small():
    """factual 预设的输出上限不得低于 1200，否则学业质量类长回答会被半截切断。"""
    from qa_bot.config import GEN_PRESETS

    assert GEN_PRESETS["factual"]["max_tokens"] >= 1200


def test_creative_preset_has_thinking():
    """创作预设应开启思考，且温度高于事实类（文风需要发散）。"""
    from qa_bot.config import GEN_PRESETS

    assert GEN_PRESETS["creative"]["think"] is True
    assert GEN_PRESETS["creative"]["temperature"] > GEN_PRESETS["factual"]["temperature"]


@pytest.mark.parametrize(
    "grade,expected",
    [
        ("一年级", "第一学段"), ("二年级", "第一学段"),
        ("三年级", "第二学段"), ("四年级", "第二学段"),
        ("五年级", "第三学段"), ("六年级", "第三学段"),
        ("七年级", "第四学段"), ("八年级", "第四学段"), ("九年级", "第四学段"),
        # 口语别名（2.10 修复过：缺这几项会让「初一年级」被误判成第一学段）
        ("初一", "第四学段"), ("初二", "第四学段"), ("初三", "第四学段"), ("初中", "第四学段"),
    ],
)
def test_grade_mapping_complete(grade, expected):
    """年级→学段映射必须覆盖 1~9 年级 + 初中口语别名。"""
    from qa_bot.config import GRADE_XUEDUAN

    assert GRADE_XUEDUAN[grade] == expected


def test_grade_mapping_values_are_legal():
    """映射值只能是四个合法学段名，防止手写错字导致加权静默失效。"""
    from qa_bot.config import GRADE_XUEDUAN

    legal = {"第一学段", "第二学段", "第三学段", "第四学段"}
    assert set(GRADE_XUEDUAN.values()) <= legal


def test_history_limits_are_tight_enough():
    """历史预算必须收紧：轮数与单条长度超了就会撞 4096 上限。"""
    from qa_bot.config import HISTORY_MAX_TURNS, HISTORY_MSG_MAX_CHARS, CONTEXT_MAX_CHARS

    assert HISTORY_MAX_TURNS <= 4
    assert HISTORY_MSG_MAX_CHARS <= 400
    # 知识注入上限不能超过 3000：再多会把 system+历史顶到 4096
    assert CONTEXT_MAX_CHARS <= 3000


def test_kb_dir_exists_and_has_corpus():
    """知识库目录必须存在且含 8 个分册 md（仓库自包含的关键）。"""
    from qa_bot.config import KB_DIR

    assert KB_DIR.exists(), f"知识库目录不存在：{KB_DIR}"
    mds = list(KB_DIR.rglob("*.md"))
    assert len(mds) >= 8, f"语料分册不足：{[p.name for p in mds]}"


def test_refusal_template_has_course_placeholder():
    """拒答文案必须保留 {course} 占位符，否则无法按学段替换。"""
    from qa_bot.config import SCOPE_REFUSAL_TEXT

    assert "{course}" in SCOPE_REFUSAL_TEXT
    assert "义务教育" in SCOPE_REFUSAL_TEXT


def test_system_prompt_declares_scope_and_sample_limits():
    """人设必须声明学段边界与「不收录范文」，这是 2.9/2.10 的提示词层防线。"""
    from qa_bot.config import SYSTEM_PROMPT

    assert "义务教育" in SYSTEM_PROMPT
    assert "范文" in SYSTEM_PROMPT


def test_rerank_weights_are_sane():
    """重排各信号权重必须为正、总和在合理区间。

    注：当前和为 1.15（向量0.45/关键词0.30/路径0.20/覆盖0.05/稀有0.15），
    因此融合分理论上可达 1.15——与 retriever 文档里"0~1 量级"的说法略有出入，
    但不影响排序（未做绝对分阈值），仅在日志中记为低危文档不一致。
    """
    from qa_bot.config import RERANK_WEIGHTS

    assert all(v > 0 for v in RERANK_WEIGHTS.values())
    assert 1.0 <= sum(RERANK_WEIGHTS.values()) <= 1.2
