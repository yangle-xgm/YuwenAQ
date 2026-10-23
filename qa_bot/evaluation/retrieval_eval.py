"""检索层评测（调优用）：只测「召回与排序」，不调用生成模型。

为什么要单独一层：
  端到端评测每题要一次 LLM 生成（本机 45 秒级），18 题就是 15 分钟，
  无法支撑"改一个权重跑一次"的调优循环。检索层每题仅一次嵌入（亚秒级），
  可以把 A/B 循环压到 20 秒内，先在这里把排序调对，再去跑生成层。

指标：
  Hit@1 / Hit@6 : 金标块是否出现在第 1 位 / 前 6 位
  MRR           : 金标块排名倒数均值（1.0 最好）
  Meta@6        : 前 6 位中"编者生成块"（00_ 导航、FAQ 问答、数据说明）的占比
                  —— 越低越好，这些块不是课标原文，不该当答案来源
  Div@6         : 前 6 位覆盖的不同二级章节数 —— 越高说明越不集中于单章

金标写法：gold 是路径子串列表，命中任意一个即算召回。用子串而非精确路径，
是因为切块后同一节的块数不固定（长章节会被切成多块），精确路径会误判。
"""
import statistics
import time

from ..config import META_PATH_KEYWORDS
from ..rag.index import get_retriever

# 每题的 gold：路径子串（任一命中即算召回）；bad：明确不该出现的路径子串
RETRIEVAL_EVAL = [
    {
        "q": "语文课程核心素养包括哪四个方面？",
        "gold": ["核心素养内涵", "核心素养四方面", "核心素养四个方面"],
        "bad": ["教学建议", "课程资源开发"],
    },
    {
        "q": "第一学段（1~2年级）的识字量要求是多少？",
        "gold": ["（一）第一学段"],
        "bad": [],
    },
    {
        "q": "义务教育语文课程一共设置了几个学习任务群？",
        "gold": ["总体设计", "三个层次", "六个学习任务群"],
        "bad": [],
    },
    {
        "q": "第三学段（5~6年级）的课外阅读总量要求是多少？",
        "gold": ["（三）第三学段"],
        "bad": [],
    },
    {
        "q": "第四学段（7~9年级）的课外阅读总量要求是多少？",
        "gold": ["（四）第四学段"],
        "bad": [],
    },
    {
        "q": "关于写字教学，语文课程标准提出了哪些教学建议？",
        "gold": ["（一）教学建议", "识字与写字"],
        "bad": [],
    },
    {
        "q": "如何检测五年级的学业质量？",
        "gold": ["学业质量描述"],
        "bad": [],
    },
    {
        "q": "初一年级写作范本",
        "gold": ["（四）第四学段"],
        "bad": [],
    },
    {
        "q": "教材编写建议有哪些？",
        "gold": ["教材编写建议"],
        "bad": [],
    },
    {
        "q": "教师培训有哪些要求？",
        "gold": ["教学研究与教师培训"],
        "bad": ["常见问答", "FAQ"],
    },
    {
        "q": "课程内容主要包括哪些主题？",
        "gold": ["总体框架"],
        "bad": ["常见问答", "FAQ"],
    },
    {
        "q": "学业质量的内涵是什么？",
        "gold": ["学业质量内涵"],
        "bad": ["常见问答", "FAQ"],
    },
    {
        "q": "过程性评价有哪些原则？",
        "gold": ["过程性评价原则", "过程性评价"],
        "bad": [],
    },
    {
        "q": "六个学习任务群分别是什么？",
        "gold": ["学习任务群1", "任务群1", "总体设计"],
        "bad": [],
    },
    {
        "q": "背诵古诗文推荐多少篇？",
        "gold": ["附录1", "背诵", "常见问答"],
        "bad": [],
    },
    {
        "q": "语法修辞知识教学有哪些要求？",
        "gold": ["附录3", "语法修辞"],
        "bad": [],
    },
    {
        "q": "如何理解语文课程工具性与人文性的统一？",
        "gold": ["课程性质", "工具性与人文性"],
        "bad": [],
    },
    {
        "q": "课程标准对课外读物有什么建议？",
        "gold": ["附录2", "课内外读物", "课外读物"],
        "bad": [],
    },
]


def _path_of(text):
    """从 retrieve() 拼好的 [path] 正文里取回路径。"""
    if text.startswith("["):
        end = text.find("]")
        if end > 0:
            return text[1:end]
    return ""


def _is_meta(path, source):
    """判断是否为编者生成的元信息块（导航/FAQ/数据说明）。"""
    if source.startswith("00_"):
        return True
    return any(k in path for k in META_PATH_KEYWORDS)


def evaluate(retriever=None, dataset=None, top_k=6, verbose=False):
    """跑一遍检索评测，返回 (rows, summary)。"""
    r = retriever or get_retriever()
    dataset = dataset or RETRIEVAL_EVAL
    rows = []
    t0 = time.time()
    for item in dataset:
        hits = r.retrieve(item["q"], top_k=top_k)
        paths = [_path_of(h[2]) for h in hits]
        rank = None
        for i, p in enumerate(paths, start=1):
            if any(g in p for g in item["gold"]):
                rank = i
                break
        bad_hit = any(
            any(b in p for b in item.get("bad", [])) for p in paths
        )
        meta = sum(1 for p, h in zip(paths, hits) if _is_meta(p, h[1]))
        sections = {
            " > ".join(p.split(">")[:3]).strip() for p in paths
        }
        rows.append({
            "q": item["q"],
            "rank": rank,
            "n": len(hits),
            "hit1": rank == 1,
            "hitk": rank is not None,
            "mrr": (1.0 / rank) if rank else 0.0,
            "meta": meta,
            "div": len(sections),
            "bad": bad_hit,
            "top1": paths[0] if paths else "",
        })
        if verbose:
            flag = "OK " if rank == 1 else ("~  " if rank else "MISS")
            print(f"  {flag} rank={rank} meta={meta} | {item['q'][:30]}")

    n = len(rows)
    summary = {
        "题数": n,
        "Hit@1": sum(1 for x in rows if x["hit1"]) / n,
        "Hit@k": sum(1 for x in rows if x["hitk"]) / n,
        "MRR": statistics.mean(x["mrr"] for x in rows),
        "Meta@6": sum(x["meta"] for x in rows) / (n * top_k),
        "Div@6": statistics.mean(x["div"] for x in rows),
        "Bad命中": sum(1 for x in rows if x["bad"]),
        "耗时": round(time.time() - t0, 2),
    }
    return rows, summary


def main():
    rows, s = evaluate(verbose=True)
    print("\n=== 检索层汇总 ===")
    for k, v in s.items():
        if isinstance(v, float) and k not in ("耗时",):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
