"""写作示范（范文）请求的确定性处理 —— 2.10。

课标**不收录任何范文/例文**（全文 0 命中，只有写作要求）。但用户常直接索要"范文"，
7B 模型会把它当成知识答出来、还自称"课标范本"——实测即使提示词明确要求声明，
它也无视（生成 387 字、零声明）。故改为**代码兜底**：

  1. 说明文字由代码生成（确定性前缀），明确"课标未收录范文，以下是参考示例"，
     不依赖模型自觉；
  2. 正文用专门的写作提示词（不是通用 RAG 模板），把该学段写作要求 + 字数下限作为硬约束；
  3. 正文质量仍受 7B 能力限制（可能写不满字数、真情实感不足），但至少不会冒充课标原文。
"""
import re

from ..config import GRADE_XUEDUAN

# 明确"要一篇范文"
_SAMPLE_HINTS = re.compile(r"范文|范本|例文|示例文|写一篇|写个|写一段|给我写|帮忙写")
# 可能是"要写作任务"，也可能是"问写作要求"，需再排除
_TASK_HINTS = re.compile(r"作文|习作|写话")
_ASK_ONLY = re.compile(
    r"要求|规定|标准|次数|多少字|字数|怎么教|教学建议|评价"
    r"|如何|怎么|怎样|提高|方法|策略|水平|建议|指导|技巧"
)

_STAGE_RE = re.compile(r"第[一二三四]学段")
_GRADE_RE = re.compile(r"初[一二三]|初中|(?<![初中高])[一二三四五六七八九十]年级|[1-9]年级")

# 课标明确的字数额度（只有第四学段写了；其他学段课标未规定统一字数）
STAGE_MIN_WORDS = {"第四学段": 500}


def _stage_of(query):
    """从问句里解析目标学段（无则 None）。"""
    m = _STAGE_RE.search(query)
    if m:
        return m.group(0)
    for g in _GRADE_RE.findall(query):
        if g in GRADE_XUEDUAN:
            return GRADE_XUEDUAN[g]
        if g.endswith("年级") and g[0].isdigit():
            cn = "一二三四五六七八九"[int(g[0]) - 1] + "年级"
            return GRADE_XUEDUAN.get(cn)
    return None


def detect_writing_request(query):
    """判断是否为"索要范文/写一篇"的请求。返回 {"stage": 学段或 None}，否则 None。

    _SAMPLE_HINTS 与 _TASK_HINTS 两个分支**都必须过 _ASK_ONLY**：
    「写一篇」既可能是"替我写一篇"，也可能是"如何写一篇"（求方法）。
    原实现只给 _TASK_HINTS 分支加了排除，导致"如何写一篇写人作文"被直接产出作文（D-6）。
    """
    q = (query or "").strip()
    if not q:
        return None
    # 先判"问方法/问要求"：命中即为知识问答，不代写
    if _ASK_ONLY.search(q):
        return None
    if _SAMPLE_HINTS.search(q):
        return {"stage": _stage_of(q)}
    # "作文/习作/写话"没说方法也没说要一篇 → 按索要范文处理
    if _TASK_HINTS.search(q):
        return {"stage": _stage_of(q)}
    return None


def disclaimer(stage):
    """代码生成的固定说明（不经过模型，保证一定会出现）。"""
    s = stage or "对应"
    floor = STAGE_MIN_WORDS.get(stage or "")
    tail = "（该学段课标要求：45 分钟能完成不少于 500 字的习作）" if floor else ""
    return (
        f"【说明】本知识库只收录《义务教育语文课程标准（2022 年版）》条文，**未收录范文**。"
        f"以下不是课标原文，而是依据{s}写作要求编写的**参考示例**，仅供对照学习{tail}。\n\n"
    )


# ---------------------------------------------------------------- 点评剥离（2.10 遗留项）
# 提示词已写"只输出作文正文，不要输出任何解释、点评或说明"，但 7B 仍会在正文后
# 追加一段"这篇作文符合第四学段学生的表达水平，结构完整，语言朴实……"。
# 与 2.10 的判断一致：模型不听话的地方用代码兜底。
#
# 判定用三个特征相乘（而不是只看一个词），避免误删正文最后一段：
#   ① 段落以"这篇/本文/整篇/以上…作文|文章|习作"开头，或以【点评】之类的标记开头；
#   ② 命中两个以上评价词（结构完整/语言朴实/中心明确…）；
#   ③ 段落长度不超过 150 字（点评通常很短，正文段落往往更长）。
_COMMENTARY_HEAD = re.compile(
    r"^[【\[]?\s*(这篇|本文|整篇|以上|该作文|此作文|这篇习作|总评|点评|评析)"
)
_COMMENTARY_MARK_HEAD = re.compile(r"^[【\[]\s*(点评|评析|总评|说明|教师点评)")
_COMMENTARY_WORDS = (
    "结构完整", "语言朴实", "中心明确", "详略得当", "条理清晰", "符合第",
    "表达流畅", "情感真挚", "首尾呼应", "切合题意", "符合题意", "选材真实",
    "以上就是", "希望能帮", "仅供参考",
)
_COMMENTARY_MAXLEN = 150


def _is_commentary(para: str) -> bool:
    s = para.strip()
    if not s or len(s) > _COMMENTARY_MAXLEN:
        return False
    if _COMMENTARY_MARK_HEAD.match(s):
        return True
    word_hits = sum(1 for w in _COMMENTARY_WORDS if w in s)
    if word_hits >= 2:
        return True
    # 以"这篇作文…"开头 + 至少一个评价词，也判为点评
    return bool(_COMMENTARY_HEAD.match(s)) and word_hits >= 1


def strip_commentary(text):
    """剥掉正文之后追加的点评段落，最多检查末尾两段。

    只从末尾往前剥，且要求"剥完还剩足够长的正文"（≥200 字）——
    宁可不剥，也不能把整篇作文误删成一段评语。
    """
    s = (text or "").strip()
    if not s:
        return s
    paras = [p for p in s.split("\n\n") if p.strip()]
    if len(paras) < 2:
        return s
    removed = 0
    while len(paras) > 1 and removed < 2 and _is_commentary(paras[-1]):
        # 剥完剩下的正文必须还够长，否则说明这是"没有正文只有评语"，不能剥
        rest = "\n\n".join(paras[:-1])
        if len(rest) < 200:
            break
        paras.pop()
        removed += 1
    return "\n\n".join(paras).strip()


def build_writing_system(query, stage, requirements):
    """写作专用 system prompt：把学段要求 + 字数下限写成硬约束。"""
    floor = STAGE_MIN_WORDS.get(stage or "")
    floor_line = (
        f"【字数下限】不少于 {floor} 字（课标：45 分钟能完成不少于 500 字的习作）。\n"
        if floor else ""
    )
    return (
        "你是语文写作辅导老师。请依据下面的课标写作要求，为学生写一篇**参考示例**（不是课标原文）。\n"
        f"【目标学段】{stage or '按用户问题判断（如未说明，按第四学段处理）'}\n"
        f"{floor_line}"
        "【硬性要求】\n"
        "1. 写学生自己的生活经历与真实感受，不要空泛写景、不要堆砌华丽辞藻。\n"
        "2. 围绕一个明确的中心，结构完整（开头—主体—结尾），详略得当。\n"
        "3. 语言朴实通顺，符合该学段学生的表达水平。\n"
        "4. 只输出作文正文，不要输出任何解释、点评或说明。\n\n"
        "【课标写作要求（参考）】\n"
        f"{requirements or '（未检索到具体条文，按第四学段通用要求）'}\n\n"
        f"【写作任务】{query}\n"
        "作文："
    )
