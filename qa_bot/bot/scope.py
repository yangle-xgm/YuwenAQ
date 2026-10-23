"""学段越界（out-of-scope）检测 —— 2.9 拒答能力加固。

背景：知识库只收录《义务教育语文课程标准（2022 年版）》，法定覆盖小学一年级至
初中九年级（第一~第四学段）。高中、学前、大学等学段由**其他文件**规定
（高中对应《普通高中语文课程标准》），不在本库内。

为什么必须硬拦截：检索器没有"相关性下限"，问越界问题也会硬返回 top-k。
实测「高一年级的学习方法有什么」召回的 6 条全是**第一学段（1~2 年级）**的段落
（相似度 0.535~0.638，看起来"正常"），7B 模型便拿这些无关材料 + 自身参数知识
自由发挥，编出一段通用高中学习建议——既无出处也不声明不覆盖，是典型的越界幻觉。

因此这里做**确定性短路**：命中越界学段就不进 LLM，直接给出标准化拒答。
关键词都带上下文限定，避免误伤「高一级学校」「提高考试分数」「《大学》」等正常表述。
"""
import re

from ..config import SCOPE_GUARD_ENABLED, SCOPE_REFUSAL_TEXT

# (学段标签, 匹配规则)。按顺序匹配，先中者为准。
_PATTERNS = [
    # 高中：高中/普通高中/高一(非"高一级")/高二/高三/高考(非"提高考试")
    ("高中", re.compile(r"(?<!提)普通高中|(?<!提)高中|高一(?!级)|高二|高三|(?<!提)高考")),
    # 学前教育
    ("学前", re.compile(r"幼儿园|学前教育|学前阶段|学前儿童")),
    # 高等教育（限定搭配，排除《大学》篇名）
    ("大学", re.compile(r"大学(?:语文|阶段|课程|课堂|学习|教材)")),
    ("研究生", re.compile(r"考研|研究生")),
]


def detect_out_of_scope(query):
    """判断问题是否越出知识库学段范围。

    返回学段标签（如 "高中"/"学前"），未越界返回 None。
    """
    if not SCOPE_GUARD_ENABLED:
        return None
    q = (query or "").strip()
    if not q:
        return None
    for label, pattern in _PATTERNS:
        if pattern.search(q):
            return label
    return None


def refusal_for(query):
    """越界时返回标准拒答文案；未越界返回 None（调用方据此决定是否继续生成）。"""
    label = detect_out_of_scope(query)
    if not label:
        return None
    return SCOPE_REFUSAL_TEXT.format(course=label)
