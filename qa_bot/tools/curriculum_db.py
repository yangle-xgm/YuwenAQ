"""工具：query_curriculum_db —— 用「约束式 NL2SQL」查询语文课标的结构化指标。

对应阿里文档 2.6 第二段 NL2SQL：RAG 管"非结构化文档"，SQL 管"结构化表格"，
问"第一学段课外阅读要求多少万字"这类问题，正解是查数据库，而不是向量检索。

本地适配说明：
  - 文档用大模型直接生成原始 SQL 字符串再 if== 匹配（教学简化）。
  - 我们改为「自然语言 -> 结构化查询计划 -> SQL」的约束式 NL2SQL：
    先确定性地把用户问题解析成 {学段, 指标, 运算} 三要素，再据此拼出合法 SQL 执行。
    好处是 100% 离线可跑、不会因为模型生成 SQL 格式抖动而失败；
    生产环境若要更泛化的自然语言，可把"解析"这一步换成让模型生成 SQL（本项目已预留思路）。
  - 数据库是随工具懒加载创建的本地 SQLite 文件（data/curriculum.db），无需联网。
"""
import os
import sqlite3
from ..config import DATA_DIR
from .registry import tool

DB_PATH = DATA_DIR / "curriculum.db"

# 《义务教育语文课程标准（2022年版）》义务教育阶段的典型量化指标（公开常识口径，用于演示）
_ROWS = [
    # 学段, 识字量_认识, 识字量_会写, 课外阅读量_万字
    ("第一学段（1~2年级）", 1600, 800, 5),
    ("第二学段（3~4年级）", 2500, 1600, 40),
    ("第三学段（5~6年级）", 3000, 2500, 100),
    ("第四学段（7~9年级）", 3500, 3000, 260),
]

# 学段关键词 -> 标准学段名（确定性匹配，避免模型自由发挥）
# 值为 str = 单个学段；值为 tuple = 一组学段（SQL 用 IN）；值为 None = 全部学段。
_PRIMARY_STAGES = (
    "第一学段（1~2年级）", "第二学段（3~4年级）", "第三学段（5~6年级）",
)
_STAGE_MAP = [
    ("第四学段", "第四学段（7~9年级）"), ("七八九年级", "第四学段（7~9年级）"), ("7~9年级", "第四学段（7~9年级）"),
    ("初中", "第四学段（7~9年级）"),
    ("第三学段", "第三学段（5~6年级）"), ("五六年级", "第三学段（5~6年级）"), ("5~6年级", "第三学段（5~6年级）"),
    ("第二学段", "第二学段（3~4年级）"), ("三四年级", "第二学段（3~4年级）"), ("3~4年级", "第二学段（3~4年级）"),
    ("第一学段", "第一学段（1~2年级）"), ("一二年级", "第一学段（1~2年级）"), ("1~2年级", "第一学段（1~2年级）"),
    # 「小学」只覆盖前三个学段（义务教育的 1~6 年级），不能映射成"全部学段"，
    # 否则会把第四学段（初中 7~9 年级）的指标混进来（D-5）。
    ("小学", _PRIMARY_STAGES),
    ("全部学段", None), ("各学段", None), ("所有学段", None),
]

_METRIC_MAP = [
    ("会写", "识字量_会写"),
    ("写字", "识字量_会写"),
    ("认识", "识字量_认识"),
    ("识字", "识字量_认识"),
    ("认字", "识字量_认识"),
    ("阅读量", "课外阅读量_万字"),
    ("课外阅读", "课外阅读量_万字"),
    ("读书量", "课外阅读量_万字"),
    ("课外阅读量", "课外阅读量_万字"),
]


def ensure_db() -> None:
    """懒加载：若数据库不存在则建表并灌入演示数据。"""
    if DB_PATH.exists():
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(DB_PATH))
    con.execute(
        "CREATE TABLE curriculum_metrics ("
        "学段 TEXT, 识字量_认识 INTEGER, 识字量_会写 INTEGER, 课外阅读量_万字 INTEGER)"
    )
    con.executemany(
        "INSERT INTO curriculum_metrics VALUES (?,?,?,?)", _ROWS
    )
    con.commit()
    con.close()


def _detect_stage(text: str):
    """返回 (stage_filter, is_all)。

    stage_filter：具体学段名（str）／一组学段（tuple）／None（未提学段）。
    is_all：用户明确说了"各学段/全部学段"，即统计口径为全量。
    """
    for kw, stage in _STAGE_MAP:
        if kw in text:
            return stage, (stage is None and kw in ("全部学段", "各学段", "所有学段"))
    return None, False


def _detect_metric(text: str) -> str:
    """返回要查询的列；命中'字'+'阅读'等组合时优先更具体的指标。"""
    # 阅读量优先级高（含'阅读'即视为阅读量）
    if any(k in text for k in ("阅读", "读书")):
        return "课外阅读量_万字"
    if "会写" in text or "写字" in text:
        return "识字量_会写"
    if any(k in text for k in ("认识", "识字", "认字")):
        return "识字量_认识"
    return None  # None 表示所有量化列都返回


def _detect_op(text: str) -> str:
    """判定运算类型。**聚合运算优先于 list**：

    原实现先判"各学段/分别/比较"→ list，导致"各学段课外阅读量平均是多少"
    永远走不到 avg 分支（avg 不可达，D-4）。聚合语义比"列出来看"更强，
    故先判 sum/avg/max/min，都不命中才落到 list。
    """
    if any(k in text for k in ("求和", "总共", "累计", "一共", "合计", "加起来")):
        return "sum"
    if any(k in text for k in ("平均", "均值")):
        return "avg"
    if any(k in text for k in ("最多", "最高", "最大")):
        return "max"
    if any(k in text for k in ("最少", "最低", "最小")):
        return "min"
    if any(k in text for k in ("对比", "比较", "各学段", "分别", "各个")):
        return "list"
    return "point"


def _where_clause(stage) -> str:
    """把学段解析结果转成 WHERE 子句（str 精确匹配 / tuple 用 IN / None 不加条件）。"""
    if isinstance(stage, (list, tuple)):
        quoted = ", ".join(f"'{s}'" for s in stage)
        return f" WHERE 学段 IN ({quoted})"
    if stage:
        return f" WHERE 学段 = '{stage}'"
    return ""


def _build_sql(stage, is_all_stages, metric, op) -> str:
    """拼 SQL。**聚合分支也必须带上 WHERE**——

    原实现在 sum/avg/max/min 时整体重写 SQL 字符串，把前面的 WHERE 丢了，
    于是"第一学段课外阅读量求和"会返回四个学段的合计 405（正确值是 5）（D-3）。
    """
    cols = ["学段", metric] if metric else ["学段", "识字量_认识", "识字量_会写", "课外阅读量_万字"]
    where = _where_clause(stage)
    if op in ("sum", "avg", "max", "min"):
        col = metric or "课外阅读量_万字"
        if op == "sum":
            return f"SELECT '合计', SUM({col}) FROM curriculum_metrics{where}"
        if op == "avg":
            return f"SELECT '平均', AVG({col}) FROM curriculum_metrics{where}"
        direction, label = ("DESC", "最高") if op == "max" else ("ASC", "最低")
        return (
            f"SELECT 学段, {col} FROM curriculum_metrics{where} "
            f"ORDER BY {col} {direction} LIMIT 1"
        )
    return f"SELECT {', '.join(cols)} FROM curriculum_metrics{where}"


# 数据库字段名 -> 给用户看的中文可读名
_DISPLAY_NAMES = {
    "识字量_认识": "要求认识的字",
    "识字量_会写": "要求会写的字",
    "课外阅读量_万字": "课外阅读量（万字）",
}


def _disp(col: str) -> str:
    return _DISPLAY_NAMES.get(col, col)


def _format(rows, cols, op) -> str:
    if op in ("sum", "avg"):
        label, val = rows[0]
        # cols[1] 在聚合时为 'SUM(课外阅读量_万字)' 这类，提取内层列名再映射
        inner = cols[1]
        if "(" in inner and ")" in inner:
            inner = inner[inner.index("(") + 1: inner.index(")")]
        return f"{label}的{_disp(inner)}：{val}"
    if op in ("max", "min"):
        stage, val = rows[0]
        return f"{'最高' if op == 'max' else '最低'}的是「{stage}」，{_disp(cols[1])}：{val}"
    # point / list
    lines = []
    for r in rows:
        parts = [f"{r[0]}"]
        for i, c in enumerate(cols[1:], start=1):
            parts.append(f"{_disp(c)}：{r[i]}")
        lines.append("，".join(parts))
    return "\n".join(lines)


@tool(
    "query_curriculum_db",
    "查询《语文课程标准》的结构化量化指标，例如各学段识字量（认识/会写）要求、课外阅读量要求。当用户问到具体数字指标、学段对比、累计/平均时用。",
    {"query": "用户的自然语言问题，如『第一学段课外阅读要求多少万字』"},
)
def query_curriculum_db(arg: str) -> str:
    ensure_db()
    stage, is_all = _detect_stage(arg)
    metric = _detect_metric(arg)
    op = _detect_op(arg)
    sql = _build_sql(stage, is_all, metric, op)
    con = sqlite3.connect(str(DB_PATH))
    cur = con.execute(sql)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    con.close()
    if not rows:
        return "（未在课标指标库中匹配到对应数据，可换种问法，如『第二学段识字量多少』『各学段阅读量对比』）"
    return _format(rows, cols, op)
