"""2.4 自动化测评 —— 用本地 Ollama 模型当"裁判"（LLM-as-Judge）。

对应阿里《2.4 自动化测评》文档"手搓裁判"思路（第四、五段）：
  - judge_answer(question, answer)      -> 判定"有效回答" / "无效回答"
  - judge_contexts(question, contexts)  -> 判定"参考信息有用" / "参考信息无用"

文档原版用云端 qwen-max 当裁判；本项目适配本地 Ollama（零成本离线）：
  - 裁判模型默认复用本地生成模型（deepseek-r1:7b），可用 QA_JUDGE_MODEL 换成
    非推理模型（如 qwen2.5:7b）以提速并避免思考标签干扰；
  - 无论哪种模型，解析时都会先剥离 <think> 标签、再按关键词判定，保证能被程序读取。
"""
import re

from ..config import JUDGE_MODEL
from ..llm import OllamaClient


def _strip_think(text: str) -> str:
    """去掉 deepseek-r1 类模型的 <think>...</think> 思考块，并裁掉残破标签。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)  # 防未闭合
    return text.strip()


class Judge:
    """用本地大模型当评委，对答疑机器人的"回答"和"检索资料"打分。"""

    def __init__(self, model: str = JUDGE_MODEL):
        self.client = OllamaClient(model=model)

    # ---------------------------------------------------------- 评"生成质量"
    def judge_answer(self, question: str, answer: str) -> str:
        """判定回答是否有效回答了问题 -> '有效回答' / '无效回答'。"""
        prompt = (
            "你是一个严格的测试人员。\n"
            "你需要判断下面的【回答】是否有效地回答了用户的【问题】。\n"
            "判定标准：只有正面回应了用户问题核心、且未编造的，才算有效；"
            "含糊拒答、答非所问、或凭空编造的都算无效。\n"
            "你只能回复四个字：有效回答 或者 无效回答。不要给出任何其他内容。\n"
            "------\n"
            f"问题是：{question}\n"
            "------\n"
            f"回答是：{answer}\n"
            "------\n"
            "判定："
        )
        raw = self.client.ask(prompt)
        return self._parse(raw, ("有效回答", "无效回答"), default="无效回答")

    # ---------------------------------------------------------- 评"检索质量"
    def judge_contexts(self, question: str, contexts: str) -> str:
        """判定检索到的参考资料是否有用 -> '参考信息有用' / '参考信息无用'。"""
        prompt = (
            "你是一个测试人员。\n"
            "你需要判断下面这些【参考资料】是否有助于回答用户的【问题】。\n"
            "只要资料中包含回答问题所需的关键信息，就判定为有用；"
            "若资料与问题无关、或不含答案信息，则判定为无用。\n"
            "你只能回复六个字：参考信息有用 或者 参考信息无用。不要给出任何其他内容。\n"
            "------\n"
            f"问题是：{question}\n"
            "------\n"
            f"参考资料是：{contexts}\n"
            "------\n"
            "判定："
        )
        raw = self.client.ask(prompt)
        return self._parse(raw, ("参考信息有用", "参考信息无用"), default="参考信息无用")

    # ---------------------------------------------------------- 解析工具
    @staticmethod
    def _parse(raw: str, labels: tuple, default: str) -> str:
        """剥离思考块后按关键词判定；退化匹配保证可程序化读取。"""
        text = _strip_think(raw)
        for lab in labels:
            if lab in text:
                return lab
        # 退化匹配：只找核心词
        if "无效" in text or "无用" in text:
            return labels[1]
        if "有效" in text or "有用" in text:
            return labels[0]
        return default
