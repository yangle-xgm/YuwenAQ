"""2.4 评测运行器：跑一遍评测集，产出可读报告 + JSON。

流程（对应 2.4 文档"被测对象 → 裁判 → 打分"闭环）：
  1. 对每个样本，让 CourseQA 真实作答，并取回它实际检索到的知识片段；
  2. 用本地 LLM 裁判判定 ① 回答是否有效 ② 检索资料是否有用；
  3. 用关键事实做确定性硬校验（事实命中率）——这是比 LLM 裁判更可靠的事实验证；
  4. 汇总：回答有效率 / 资料有用率 / 事实平均命中率 / 拒答正确率，打印表格并存盘。
"""
import json
import time
from pathlib import Path

from ..bot.core import CourseQA
from .judge import Judge
from .dataset import EVAL_DATASET

# 拒答信号词：回答里出现这些，说明模型诚实承认"知识库未覆盖 / 无法回答"。
# 用于判定"知识库未覆盖题"是否被正确拒答（不能用裁判的"有效/无效回答"标签，
# 因为陷阱题诚实说不知道时裁判会判"有效回答"——它确实有效回应了用户）。
REFUSE_MARKERS = ("无法回答", "知识库", "未覆盖", "不知道", "不在我的",
                  "没有相关信息", "暂未", "不属于", "超出")


def run_evaluation(bot=None, judge=None, dataset=None):
    """返回 (rows, summary)。rows 为每条明细，summary 为汇总字典。"""
    bot = bot or CourseQA()
    judge = judge or Judge()
    dataset = dataset or EVAL_DATASET

    rows = []
    for item in dataset:
        q = item["question"]
        bot.reset()  # 每题独立，避免多轮历史污染后续评测
        answer = bot.ask(q)
        contexts = bot.retrieve_knowledge(q)

        # 裁判判定
        ans_label = judge.judge_answer(q, answer)
        ctx_label = judge.judge_contexts(q, contexts) if contexts.strip() else "参考信息无用"

        # 确定性硬校验：关键事实命中率
        facts = item.get("key_facts", [])
        if facts:
            hit = sum(1 for f in facts if f in answer)
            fact_rate = hit / len(facts)
        else:
            hit, fact_rate = None, None

        # should_know：知识库是否覆盖该问题（来自评测集，用于分层统计/日志）
        should_know = item.get("should_know", True)

        # 拒答能力：仅当题目显式要求拒答（expect_refuse=True，即知识库确实未覆盖、
        # 按 2.3 规范应"如实说明暂未覆盖"）时，才检查回答是否含拒答信号。
        # 注意：LLM 裁判的"有效/无效回答"衡量的是"对用户是否有帮助"，
        # 陷阱题诚实说"不知道"会被判为"有效回答"，因此拒答判定必须用语义关键词，
        # 不能依赖 ans_label == "无效回答"。
        expect_refuse = item.get("expect_refuse", False)
        refuse_ok = any(m in answer for m in REFUSE_MARKERS) if expect_refuse else None

        rows.append({
            "question": q,
            "answer": answer.strip(),
            "contexts_preview": contexts[:200],
            "ans_label": ans_label,
            "ctx_label": ctx_label,
            "key_facts": facts,
            "fact_hit": hit,
            "fact_rate": fact_rate,
            "should_know": should_know,
            "refuse_ok": refuse_ok,
        })
        print(f"  [已评] {q[:26]:<26} -> {ans_label} | {ctx_label}")

    # ---------------- 汇总 ----------------
    n = len(rows)
    valid = sum(1 for r in rows if r["ans_label"] == "有效回答")
    useful = sum(1 for r in rows if r["ctx_label"] == "参考信息有用")
    fact_rows = [r for r in rows if r["fact_rate"] is not None]
    avg_fact = (sum(r["fact_rate"] for r in fact_rows) / len(fact_rows)) if fact_rows else None
    refuse_rows = [r for r in rows if r["refuse_ok"] is not None]
    refuse_rate = (sum(1 for r in refuse_rows if r["refuse_ok"]) / len(refuse_rows)) if refuse_rows else None

    summary = {
        "样本数": n,
        "回答有效率": f"{valid}/{n} = {valid / n * 100:.1f}%",
        "资料有用率": f"{useful}/{n} = {useful / n * 100:.1f}%",
        "事实平均命中率": f"{avg_fact * 100:.1f}%" if avg_fact is not None else "N/A",
        "拒答正确率(知识库未覆盖题)": f"{refuse_rate * 100:.1f}%" if refuse_rate is not None else "N/A",
    }
    return rows, summary


def main():
    t = time.time()
    print("=== 2.4 自动化测评开始（本地 Ollama 裁判）===")
    rows, summary = run_evaluation()
    print("\n=== 汇总 ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out = Path(__file__).resolve().parent.parent.parent / "eval_results.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n明细已保存: {out}  (耗时 {time.time() - t:.1f}s)")


if __name__ == "__main__":
    main()
