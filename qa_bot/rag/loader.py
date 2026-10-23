"""知识库加载（2.1 第八段：把知识读进来，准备塞进提示词）。

朴素版：直接读 markdown 全文，不切块、不向量化；切块和向量化留到 2.5 真·RAG。
"""
from pathlib import Path

from ..config import KB_DIR, CHUNK_SIZE, CHUNK_OVERLAP

import re


def load_documents(kb_dir=KB_DIR):
    """
    加载知识库文档。
    参数:
        kb_dir: 知识库目录路径，默认取全局常量 KB_DIR
                 （如《语文课程标准》等 .md 文件所在文件夹）
    返回:
        list[dict]，每个元素形如 {"source": 文件名, "text": 文件全文}，
        即"一个 md 文件 = 一段文档"。供后续检索器（NaiveRetriever）建立索引使用。
    """
    docs = []  # 结果容器：累积所有文档

    # 递归遍历 kb_dir 下所有 .md 文件；sorted 保证加载顺序稳定（按路径排序）
    for p in sorted(Path(kb_dir).rglob("*.md")):
        # 读取该 md 文件的完整文本内容（以 utf-8 解码，避免中文乱码）
        text = p.read_text(encoding="utf-8")

        # 以"文件名"作为出处(source)、"全文"作为文本(text)存入列表
        docs.append({"source": p.name, "text": text})

    return docs

def chunk_documents(kb_dir=KB_DIR):
    """把知识库所有 .md 切成小块：每块 = 一行正文 + 它所属的标题路径。"""
    chunks = []
    for p in sorted(Path(kb_dir).rglob("*.md")):
        text = p.read_text(encoding="utf-8")
        chunks.extend(_split(p.name, text))
    return chunks


def _split(source, text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """把单个 md 切成小块。

    策略（2.2 落地 + 2.5 预热的折中版）：
    - 同一小节（标题路径）下的正文行**累积合并**到约 chunk_size 字，避免"一行一块"上下文太碎；
    - 每个块记录它所属的标题路径（保留最近三级），并生成 embed_text = 路径 + 正文，
      **嵌入时连标题一起喂**，让向量自带"第几学段/哪个主题"的语境；
    - 块与块之间保留 overlap 字的尾部重叠，防止句子被拦腰切断。
    - 过滤知识库自带的元信息块（`> **来源**`/`> **标签**`/`> **适用检索**`）：
      这类行把"教学建议 / 评价建议 / 教材编写 …"整个栏目清单堆在一起，关键词密度极高，
      检索时会被当成高相关块挤进 top_k，但它们并不是课标正文（2.5+ 噪声治理）。
    """
    chunks, stack = [], []
    buf, buf_path = [], []        # buf：当前累积的正文行；buf_path：这些行的标题路径
    META_PREFIXES = ("> **来源**", "> **标签**", "> **适用检索**")

    def flush():
        """把缓冲区收成一个 chunk 落袋，并留下尾部 overlap 作为下一块的开头。"""
        nonlocal buf
        if not buf:
            return
        body = "\n".join(buf)
        # 保留最近三级标题（分册名 + 两级章节），让块自带"哪个任务群/哪条建议"的语境。
        path = " > ".join(buf_path[-3:])
        chunks.append({
            "source": source,
            "path": path,
            "text": body,
            "embed_text": f"{path}。{body}" if path else body,
        })
        keep, acc = [], 0
        for ln in reversed(buf):          # 从后往前攒够 overlap 为止
            keep.insert(0, ln)
            acc += len(ln)
            if acc >= overlap:
                break
        buf = keep

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(META_PREFIXES):  # 知识库文件的元信息块，不是正文，丢弃
            continue
        m = re.match(r"^(#+)\s*(.+)$", line)
        if m:                              # 标题：小节切换，先把上一小节落袋
            level = len(m.group(1))
            flush()
            # 修复：buf_path 必须包含"当前标题"本身，否则每个块都会丢掉自己所属的那一节
            # （旧写法先 stack[:level-1] 再 append，buf_path 里永远少一层，
            #  导致 04「任务群1」、06「建议1」这类最有区分度的标题进不了路径）。
            stack = stack[: level - 1]
            stack.append(m.group(2).strip())
            buf_path = stack[:]
            continue
        if len(line) < 8:                  # 太短的行丢掉（多是表格分隔线、空标记）
            continue
        buf.append(line)
        if sum(len(x) for x in buf) >= chunk_size:
            flush()
    flush()
    return chunks

