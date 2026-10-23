"""知识库索引的"建"与"取"（2.2 第五段封装思路）。
对外只暴露：build_index() 建一次；get_retriever() 取来即用（有则加载，无则重建）。"""
from .embeddings import embed_texts
from .loader import chunk_documents
from .vectorstore import save_index, load_index
from .retriever_vector import VectorRetriever
from ..config import KB_DIR, INDEX_PATH


def build_index(kb_dir=KB_DIR, index_path=INDEX_PATH):
    """读文档 → 切块 → 向量化 → 存盘（对应 2.2 第一段+第二段）。"""
    chunks = chunk_documents(kb_dir)
    # 嵌入用 embed_text（带标题路径的正文），展示/入库用 text（纯正文）
    matrix = embed_texts([c.get("embed_text", c["text"]) for c in chunks])
    save_index(matrix, chunks, index_path)
    return VectorRetriever(matrix, chunks)


def get_retriever(kb_dir=KB_DIR, index_path=INDEX_PATH):
    """有索引就加载（几秒），没有就当场建（几分钟+花本地算力）。"""
    matrix, chunks = load_index(index_path)
    if matrix is None:
        return build_index(kb_dir, index_path)
    return VectorRetriever(matrix, chunks)
