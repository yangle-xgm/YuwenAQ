"""向量检索（2.2）+ 关键词混合（2.2 预热）+ 两阶段 Rerank 重排（2.5）+ HyDE（2.5）。

2.5 核心思想（对应阿里文档第十二段"Rerank 重排序——先多召回、再精排筛选"）：
    第一阶段（粗排/召回）：用混合检索先撒大网，召回 CANDIDATE_K（默认24）条候选，
        保证相关片段大概率入网；同时用 SIMILARITY_CUTOFF 过滤掉明显无关的噪声。
    第二阶段（精排/重排）：对这 CANDIDATE_K 条候选重新打分排序，只留 TOP_K（默认6）条。
        重排分 = 向量余弦 + 关键词IDF + 标题路径命中 + query词覆盖率 四信号融合。
        ——"标题路径命中"是本地替代 cross-encoder 的关键补丁：nomic-embed-text 中文
        弱，纯向量会让"06 课程实施"的目录块胜出；但正文块的标题路径（如"写字教学建议"）
        里含 query 关键词，命中率远高于目录块，从而把真正的答案块顶到前面。

注意：本地 Ollama 0.40.5 不支持 /api/rerank 端点、也无 rerank 模型，所以这里用手搓的
多信号融合重排替代云端 cross-encoder（qwen3-rerank），零新依赖、离线可用。

2.5+ 检索噪声治理（去掉"命中目录树"这类假相关）：
    - 导航/元信息源降权（NAV_PRIOR）：00_总览这类"目录树+检索指南"文件关键词密度极高，
      四路信号全优，会长期霸占 top_k；它不是课标正文，故整体乘一个降权系数。
    - 粗排 lex 归一化：IDF 加权和没有上界（实测可达 28），不归一化会直接压死向量信号。
    - 关键词/覆盖率只用正文：标题路径是全分册共享的样板文字，算进去等于给每个块白送分；
      路径信息改由 _path_boost 精确计分，且跳过首段"分册名"。
    - 章节限流（SECTION_CAP）：同一二级章节最多取 2 块，避免长章节用相似分占满 top_k。
    - 稀有词命中（rare）：query 内 IDF 前 25% 的实词是否真出现，纠正"长泛块靠常用词刷分"。

HyDE（2.5 第十段，默认关）：先让生成模型"脑补"一篇假想答案，再取其向量与 query 向量
融合后检索。解决"短问题 vs 长文档画风不对等"问题；代价是多一次模型调用，故默认关闭。
"""
import math
import re

import numpy as np

from ..config import (
    TOP_K,
    CANDIDATE_K,
    VECTOR_WEIGHT,
    LEXICAL_WEIGHT,
    RERANK_ENABLED,
    RERANK_WEIGHTS,
    SIMILARITY_CUTOFF,
    GRAM_HIT_FLOOR,
    HYDE_ENABLED,
    HYDE_MODEL,
    NAV_SOURCES,
    NAV_PRIOR,
    META_PATH_KEYWORDS,
    META_PRIOR,
    SECTION_CAP,
    RARE_IDF_PERCENTILE,
    RARE_IDF_FLOOR,
    GRADE_XUEDUAN,
    QUERY_SYNONYMS,
    GRADE_MATCH_BOOST,
    GRADE_MISS_PENALTY,
)
from .embeddings import embed_query

_TOKEN = re.compile(r"[\s\W]+")  # 空白与标点都当分隔，只留中英文数字字符

_ALL_XUEDUAN = ("第一学段", "第二学段", "第三学段", "第四学段")
_GRADE_CN = re.compile(r"初[一二三]|初中|(?<![初中高])[一二三四五六七八九十]年级")
_GRADE_NUM = re.compile(r"[1-9]年级")
_XUEDUAN_RE = re.compile(r"第[一二三四]学段")


def _grade_targets(query: str):
    """从 query 里解析用户关心的学段（2.8）。

    用户说"五年级"，课标原文只有"第三学段（5~6年级）"——字面零重叠。
    这里把年级（含阿拉伯数字"5年级"）和直述的"第X学段"都映射成学段名集合，
    供检索端做匹配升权/错配降权。没有学段意图时返回空集（不做任何干预）。
    """
    targets = set()
    for g in _GRADE_CN.findall(query):
        xu = GRADE_XUEDUAN.get(g)
        if xu:
            targets.add(xu)
    for g in _GRADE_NUM.findall(query):
        cn = "一二三四五六七八九"["123456789".index(g[0])] + "年级"
        xu = GRADE_XUEDUAN.get(cn)
        if xu:
            targets.add(xu)
    for x in _XUEDUAN_RE.findall(query):
        targets.add(x)
    return targets


def _bigrams(s):
    """字符级 2-gram 集合：中文按相邻字，英文/数字也按相邻字符，兼顾中英文。"""
    s = _TOKEN.sub("", s)
    if len(s) < 2:
        return set(s) if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


def _text_of(c):
    """取分块用于"关键词/覆盖"计算的文本：只用正文。

    为什么不用 embed_text（带标题路径）？因为同分册的所有块共享同一段路径
    （如"分册06 · 课程实施（教学建议 · 评价建议 · …）"），路径里的"教学/建议/课程"
    会让每个块白拿一份关键词分，把"教学研究与教师培训"这类无关章节也顶上来。
    路径信息改由 _path_boost 单独、精确地计分。
    """
    return c["text"]


def _depths(path):
    """把标题路径拆成层级列表（去掉首段"分册名/文件名"这种全分册共享的样板）。

    path 形如 "分册06 · 课程实施（教学建议 · …） > 一、（一）教学建议（4条） > 建议1：…"，
    首段含全分册的栏目名，对区分"哪一节"没有信息量，反而制造噪声，故剥离。
    """
    segs = [s.strip() for s in (path or "").split(">") if s.strip()]
    return segs[1:] if len(segs) >= 2 else []


class VectorRetriever:
    def __init__(self, matrix, chunks, top_k=TOP_K):
        self.matrix = matrix          # (N, D) 已归一化，D=嵌入模型维度（bge-m3 为 1024）
        self.chunks = chunks
        self.top_k = top_k
        # 预计算 IDF：每个 bigram 的文档频率 → 平滑逆文档频率
        df = {}
        for c in chunks:
            for g in _bigrams(_text_of(c)):
                df[g] = df.get(g, 0) + 1
        n = max(len(chunks), 1)
        self.idf = {g: math.log((n + 1) / (v + 1)) + 1.0 for g, v in df.items()}
        self._query_vec_cache = {}

    # ---- 各打分信号 ----------------------------------------------------
    def _nav_factor(self, c):
        """导航/编者元信息类块的降权系数。

        分两档，因为性质不同：
          · 00_ 总览：纯导航（目录树 + 检索指南），几乎不当答案来源 → NAV_PRIOR（0.55）
          · 编制时补的 FAQ/数据说明：二手解读，术语密度高会压过正文，但内容仍可能
            相关（有些问题课标没有直接条文） → META_PRIOR（0.75，更温和）
        """
        src = c.get("source") or ""
        if any(src.startswith(p) for p in NAV_SOURCES if p):
            return NAV_PRIOR
        path = c.get("path") or ""
        if any(k in path for k in META_PATH_KEYWORDS):
            return META_PRIOR
        return 1.0

    def _lexical(self, qgrams, c):
        """查询与某个分块正文的 IDF 加权 bigram 重合度（关键词命中强度）。"""
        grams = _bigrams(_text_of(c))
        return sum(self.idf[g] for g in qgrams if g in grams)

    def _path_boost(self, qgrams, c):
        """query 字符 bigram 命中分块“章节标题”（跳过首段分册名）的比例——2.5 重排的关键补丁。"""
        if not qgrams:
            return 0.0
        pgrams = _bigrams(" ".join(_depths(c.get("path") or "")))
        if not pgrams:
            return 0.0
        hit = sum(1 for g in qgrams if g in pgrams)
        return hit / len(qgrams)

    def _coverage(self, qgrams, c):
        """query 字符 bigram 在分块正文中的覆盖率。"""
        tgrams = _bigrams(_text_of(c))
        if not qgrams:
            return 0.0
        hit = sum(1 for g in qgrams if g in tgrams)
        return hit / len(qgrams)

    def _rare_terms(self, qgrams):
        """挑出 query 里最能"定性"的稀有 bigram（IDF 前 25% 且不低于下限）。

        为什么需要：关键词总分是"一堆 bigram 相加"，长而泛的块（如教研章节的
        "语文教师要养成读书习惯……"）靠大量常用词也能刷出高分，反而压过真正含
        "写字"这种高区分度词的正文块。单列一个"稀有词是否命中"的信号可纠偏。
        """
        if not qgrams:
            return set()
        pairs = [(g, self.idf.get(g, 1.0)) for g in qgrams]
        vals = [v for _, v in pairs]
        th = float(np.percentile(vals, RARE_IDF_PERCENTILE)) if vals else 0.0
        th = max(th, RARE_IDF_FLOOR)
        return {g for g, v in pairs if v >= th}

    def _rare_cover(self, rare, c):
        """稀有词在分块正文中的命中比例。"""
        if not rare:
            return 0.0
        tgrams = _bigrams(_text_of(c))
        return sum(1 for g in rare if g in tgrams) / len(rare)

    @staticmethod
    def _section_key(c):
        """章节去重用的键：同一文件 + 同一**二级章节**（如"五、（五）教学研究与教师培训"）。

        不用最深标题：像"教研"这种大章节下面还有 8 条 h3 子节，若按最深标题去重，
        每个子节都算"不同节"，一样会占满 top_k；按二级章节去重才能真正保证覆盖面。
        """
        depths = _depths(c.get("path") or "")
        return (c.get("source") or "", depths[0] if depths else "")

    # ---- 2.8 年级/学段语义桥接 ------------------------------------------
    def _query_grams(self, query):
        """关键词信号用的 bigram 集合 = query 自身 + 同义词扩展。

        只扩展关键词类信号（lexical/coverage/rare/path_boost），向量余弦仍用
        原始 query——扩展词是给"字面对不上"的关键词信号补桥，不污染语义向量。
        """
        grams = _bigrams(query)
        for term, syns in QUERY_SYNONYMS.items():
            if term in query:
                for s in syns:
                    grams |= _bigrams(s)
        return grams

    def _grade_factor(self, c, targets):
        """学段匹配乘性因子（2.8）。

        课标内容按学段专用：问"五年级"却召回"第一学段（1~2年级）"的描述，
        对用户就是事实性错误。故路径命中目标学段 → 升权；路径含其他学段 → 降权；
        路径不涉及学段（如总览、FAQ、任务群）→ 不动，保持中性。
        """
        if not targets:
            return 1.0
        path = c.get("path") or ""
        if any(t in path for t in targets):
            return GRADE_MATCH_BOOST
        if any(x in path for x in _ALL_XUEDUAN):
            return GRADE_MISS_PENALTY
        return 1.0

    # ---- HyDE（可选） ---------------------------------------------------
    def _query_vector(self, query):
        """取检索向量；开启 HyDE 时融合“假想答案”向量，否则就是 query 本身的向量。"""
        qv = embed_query(query)
        if not HYDE_ENABLED:
            return qv
        try:
            from ..llm import OllamaClient

            cli = OllamaClient(model=HYDE_MODEL)
            sys_msg = (
                "你是一名语文课程专家。请直接写一段可能出现在《义务教育语文课程标准》中、"
                "回答该问题的标准内容片段。只写内容本身，不要解释、不要加引号、不要使用列表符号。"
            )
            hyp = cli.chat(
                [
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": query},
                ],
                think=False,
                temperature=0.8,
            ).strip()
            if hyp:
                hv = embed_query(hyp[:400])
                qv = (qv + hv) / 2.0
        except Exception:
            # HyDE 失败（如模型不可用）则安全降级为纯 query 向量
            pass
        return qv

    # ---- 主入口 --------------------------------------------------------
    def retrieve(self, query, top_k=None):
        """返回 [(score, source, text), ...]，按分数降序，最多 top_k 条。

        score 含义随模式而变：Rerank 开启时为四信号融合分（0~1 量级），
        关闭时为混合粗排分（VECTOR_WEIGHT*cos + LEXICAL_WEIGHT*lex）。
        """
        if not query.strip() or self.matrix.size == 0:
            return []
        qv = self._query_vector(query)
        qgrams = self._query_grams(query)
        targets = _grade_targets(query)

        # 相关性下限（D-1）：先看"问题里的词有多少真的出现在课标语料里"。
        # 无关问题（天气、电影、菜谱）词面命中率实测 0.00~0.29，课标问题 0.43 以上，
        # 阈值 0.35 落在干净的间隔里。命中率不足就直接返回空——让系统提示词第 2 条
        # 引导模型说明"知识库暂未覆盖"，好过强塞 6 条噪声去买一次幻觉。
        if qgrams:
            hit = sum(1 for g in qgrams if g in self.idf)
            if hit / len(qgrams) < GRAM_HIT_FLOOR:
                return []

        cos = self.matrix @ qv                       # 向量余弦（归一化后=点积），(N,)

        # 第一阶段：相似度截断 + 粗排，取 CANDIDATE_K 候选
        # 注意 lex 必须归一化：IDF 加权和是"无上界"的（实测可到 28），而余弦只在 0.5~0.8；
        # 若直接 0.7*cos + 0.3*lex，粗排会退化成"关键词计数排名"，向量信号形同虚设。
        cand = []
        for i, c in enumerate(self.chunks):
            ci = float(cos[i])
            if ci < SIMILARITY_CUTOFF:               # 相似度截断：过滤明显无关噪声
                continue
            cand.append([i, ci, self._lexical(qgrams, c)])
        if not cand:
            return []
        lex_max = max(x[2] for x in cand) or 1.0
        for x in cand:
            x[2] = x[2] / lex_max                    # 归一到 [0,1]
        cand.sort(
            key=lambda x: (VECTOR_WEIGHT * x[1] + LEXICAL_WEIGHT * x[2])
            * self._nav_factor(self.chunks[x[0]])
            * self._grade_factor(self.chunks[x[0]], targets),
            reverse=True,
        )
        cand = cand[:CANDIDATE_K]

        # 第二阶段：Rerank 融合重排
        if RERANK_ENABLED:
            lex_pool_max = max(x[2] for x in cand) or 1.0
            rare = self._rare_terms(qgrams)
            scored = []
            for (i, ci, lex) in cand:
                c = self.chunks[i]
                cos_n = max(0.0, ci)                 # 余弦归一到 [0,1]
                lex_n = lex / lex_pool_max           # 关键词归一到 [0,1]
                path_n = self._path_boost(qgrams, c)  # 章节标题命中率（不含分册名）
                cov_n = self._coverage(qgrams, c)     # query 词覆盖率
                rare_n = self._rare_cover(rare, c)    # 稀有实词命中率
                rs = (
                    RERANK_WEIGHTS["vector"] * cos_n
                    + RERANK_WEIGHTS["lexical"] * lex_n
                    + RERANK_WEIGHTS["path"] * path_n
                    + RERANK_WEIGHTS["coverage"] * cov_n
                    + RERANK_WEIGHTS.get("rare", 0.0) * rare_n
                ) * self._nav_factor(c) * self._grade_factor(c, targets)
                scored.append((rs, i))
            scored.sort(reverse=True)
            top = scored
        else:
            # 不重排：直接取粗排顺序
            top = [(VECTOR_WEIGHT * x[1] + LEXICAL_WEIGHT * x[2], x[0]) for x in cand]
            top.sort(reverse=True)

        # 章节去重：同一节最多取 SECTION_CAP 块，避免长章节用相似分占满 top_k
        out, used = [], {}
        for rs, i in top:
            c = self.chunks[i]
            if SECTION_CAP > 0:
                key = self._section_key(c)
                if used.get(key, 0) >= SECTION_CAP:
                    continue
                used[key] = used.get(key, 0) + 1
            text = f"[{c['path']}] {c['text']}" if c.get("path") else c["text"]
            out.append((round(float(rs), 4), c["source"], text))
            if len(out) >= (top_k or self.top_k):
                break
        return out
