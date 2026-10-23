"""检索质量回归门槛：用 qa_bot/evaluation/retrieval_eval.py 的指标守住调优成果。

为什么需要这层：
  单元测试守的是"函数行为对不对"（如 sum 带不带 WHERE），检索质量守的是
  "排序好不好"——后者没有对错，只有好坏，必须用数值门槛卡住。
  本轮调优把 Hit@1 从 0.611 提到 0.778，如果没有这条门槛，
  以后随便调一次权重就可能悄悄退回去，而所有单元测试依然全绿。

门槛按当前实测值留出合理余量（不卡死在实测值上，否则微小的合理波动会误报）：
  Hit@1 ≥ 0.72（实测 0.778）   MRR ≥ 0.80（实测 0.846）
  Meta@6 ≤ 0.12（实测 0.074）  Hit@6 = 1.0（不允许漏召回）
"""
import pytest


@pytest.fixture(scope="module")
def report():
    """跑一次检索评测，整个模块共用（评测需 11 秒，每条用例各跑一次太浪费）。"""
    from qa_bot.evaluation.retrieval_eval import evaluate

    return evaluate()


@pytest.mark.retrieval
def test_retrieval_hit_at_1(report):
    """金标块排在首位的比例：调优后 0.778（14/18），门槛 0.72。"""
    _, s = report
    assert s["Hit@1"] >= 0.72, f"Hit@1 退化：{s['Hit@1']:.3f}"


@pytest.mark.retrieval
def test_retrieval_hit_at_k_full(report):
    """金标块必须全部能进前 6，一条都不许漏——漏召回等于答不出。"""
    _, s = report
    assert s["Hit@k"] == 1.0, f"有题目召不回金标块：Hit@6={s['Hit@k']:.3f}"


@pytest.mark.retrieval
def test_retrieval_mrr(report):
    """平均倒数排名：调优后 0.846，门槛 0.80。"""
    _, s = report
    assert s["MRR"] >= 0.80, f"MRR 退化：{s['MRR']:.3f}"


@pytest.mark.retrieval
def test_meta_chunks_not_dominant(report):
    """编者生成块（00_ 导航 / FAQ / 数据说明）在前 6 中的占比不得超过 12%。

    这些不是课标原文，占比高意味着模型在拿二手解读当课标条文引用。
    调优前 0.250（每 6 条 1.5 条），调优后 0.074。
    """
    _, s = report
    assert s["Meta@6"] <= 0.12, f"编者块占比过高：Meta@6={s['Meta@6']:.3f}"


@pytest.mark.retrieval
def test_retrieval_not_overfit_to_single_question(report):
    """防止“只调好一题”的过拟合：18 题里至少 12 题排在首位。

    单题修补（比如给某题加特判）会让 Hit@1 虚高，但换一批问题就失效。
    用绝对题数卡一道，配合 Hit@k=1.0 的全覆盖要求，能挡住这类改动。
    """
    rows, _ = report
    hit1 = sum(1 for r in rows if r["hit1"])
    assert hit1 >= 12, f"仅 {hit1}/18 题排在首位，疑似针对个别题过拟合"
