"""朴素检索（2.1 第八段 RAG 雏形）。

三步走：
  1) 按行切块，并记住"标题路径"（如 三、学段要求 > （四）第四学段），避免断章取义；
  2) 用相邻两字（bigram）重合度打分，并按每个 bigram 在全库的稀有度加权（IDF）——
     "核心""素养"比"语文""课程"更有区分度；
  3) 手写一小张"答案加分表"：问题命中某意图、且片段里同时出现若干标准答案词时加分。
     这是 2.1 雏形的补丁；等 2.5 上真·向量检索，整张表删掉即可。

没有 numpy、没有向量库 —— 换 2.5 时整个替换本文件。
"""
import math
import re

# (问题关键词, 答案标准词, 至少要命中几个答案词)
BOOSTS = [
    (["识字量", "认识多少汉字", "累计认识", "认多少字"], ["累计认识常用汉字"], 1),
    (["阅读量", "课外阅读", "读多少字"], ["课外阅读总量", "万字"], 1),
    (["学习任务群", "任务群", "几个任务群"], ["六个学习任务群"], 1),
    (["核心素养", "哪四个", "四个方面"], ["文化自信", "语言运用", "思维能力", "审美创造"], 2),
]


def _bigrams(text):
    t = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", text)
    return {t[i:i + 2] for i in range(len(t) - 1)}


class NaiveRetriever:
    def __init__(self, docs, top_k=3, max_chars=300):
        self.top_k = top_k
        self.max_chars = max_chars
        self.chunks = []
        for d in docs:
            self.chunks.extend(self._split(d["source"], d["text"]))
        # 统计每个 bigram 出现在多少个块里（算 IDF 用）
        self.df = {}
        for c in self.chunks:
            for b in _bigrams(c["path"] + " " + c["text"]):
                self.df[b] = self.df.get(b, 0) + 1
        self.n = len(self.chunks)

    @staticmethod
    def _split(source, text):
        chunks, stack = [], []
        for raw in text.splitlines():
            line = raw.strip()
            if not line:
                continue
            m = re.match(r"^(#+)\s*(.+)$", line)
            if m:                     # 标题：按层级维护一条"标题路径"
                level = len(m.group(1))
                stack = stack[: level - 1]
                stack.append(m.group(2).strip())
                continue
            if len(line) < 8:         # 太短的行（表格分隔等）丢掉
                continue
            chunks.append({
                "source": source,
                "path": " > ".join(stack[-2:]),
                "text": line,
            })
        return chunks

    def _idf(self, b):
        return math.log(1 + self.n / (1 + self.df.get(b, 1)))

    def _score(self, query, chunk):
        qb = _bigrams(query)
        cb = _bigrams(chunk["text"])
        pb = _bigrams(chunk["path"])
        s = 0.0
        for b in qb:
            if b in cb:
                s += self._idf(b)
            if b in pb:               # 标题就命中，说明这一节正是讲这个
                s += 0.6 * self._idf(b)
        for keys, phrases, need in BOOSTS:
            if any(k in query for k in keys):
                hit = sum(1 for p in phrases if p in chunk["path"] + chunk["text"])
                if hit >= need:
                    s += 8 + 2 * hit
        return s

    def retrieve(self, query):
        if not query.strip():
            return []
        scored = [(self._score(query, c), c) for c in self.chunks]
        scored = [x for x in scored if x[0] > 0]
        scored.sort(key=lambda x: -x[0])
        out = []
        for s, c in scored[: self.top_k]:
            text = f"[{c['path']}] {c['text']}" if c["path"] else c["text"]
            out.append((round(s, 2), c["source"], text[: self.max_chars]))
        return out
