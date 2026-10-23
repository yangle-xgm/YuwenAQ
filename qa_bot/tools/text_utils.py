"""工具：analyze_text / calc_expression —— 两类"动作型"能力。

演示 2.6 文档的核心论点：只会查资料的 RAG 机器人不会"办事情"，
而插件（工具）能让它真正去"算"、去"数"。这两类工具都是纯本地确定性逻辑。

- analyze_text：统计一段文字的字数、标点数、段落数、句数（语文老师批改/作业常用）。
- calc_expression：安全计算算术表达式（如把几个学段阅读量加起来做对比）。
"""
import re
from .registry import tool

# 常见中文/全角标点集合，用于统计"标点数"
_PUNCT = set("，。！？、；：""''（）《》【】…—·,.!?;:\"'()[]{}<>-—")


def _extract_text(arg: str) -> str:
    """从用户指令里抽取'真正要分析的文本'：取最后一个冒号/冒号后的内容。"""
    if "：" in arg:
        return arg.rsplit("：", 1)[1].strip()
    if ":" in arg:
        return arg.rsplit(":", 1)[1].strip()
    return arg.strip()


@tool(
    "analyze_text",
    "统计一段文字的字数、标点数、段落数、句数。当用户要求'数一下这段话有多少字/几个标点/几段'时用。",
    {"text": "要统计的文本内容（若用户原话带『这段话：…』，取冒号后的正文）"},
)
def analyze_text(arg: str) -> str:
    text = _extract_text(arg)
    if not text:
        return "（请提供要统计的文本内容，例如：帮我数一下这段话有多少字：床前明月光，疑是地上霜。）"
    total = len(text)
    cn = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")  # 汉字数
    punct = sum(1 for ch in text if ch in _PUNCT)             # 标点数
    paragraphs = [p for p in text.split("\n") if p.strip()]   # 段落（非空行）
    # 句数：按中英文句末标点切分
    sentences = [s for s in re.split(r"[。！？.!?；;]", text) if s.strip()]
    return (
        f"文本统计结果：\n"
        f"- 总字符数（含标点/空格）：{total}\n"
        f"- 汉字数：{cn}\n"
        f"- 标点数：{punct}\n"
        f"- 段落数：{len(paragraphs)}\n"
        f"- 句数（按句末标点切分）：{len(sentences)}"
    )


@tool(
    "calc_expression",
    "计算一个算术表达式的结果（仅支持 + - * / 和括号）。当用户要相加/相乘/求平均值等数字计算时用。",
    {"expression": "算术表达式，如 '1600 + 2500' 或 '(5 + 40 + 100 + 260)'"},
)
def calc_expression(arg: str) -> str:
    expr = (arg or "").strip()
    if not expr:
        return "（请提供算术表达式，例如：1600 + 2500）"
    # 安全校验：只允许数字与基本运算符，杜绝代码注入
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return "（只支持数字与 + - * / () 的算术表达式，已拒绝不安全输入）"
    try:
        result = eval(expr, {"__builtins__": {}}, {})  # 受限命名空间，无内置函数
    except Exception as e:
        return f"（计算失败：{e}）"
    return f"{expr} = {result}"
