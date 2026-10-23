"""向量库的落盘与读取（2.2 第二、三段：persist / load）。
两件套 = vector_store.npy（向量矩阵）+ docstore.json（分块原文），合起来就是整套知识库。"""
import json
from pathlib import Path

import numpy as np

from ..config import INDEX_PATH


def save_index(matrix, chunks, index_path=INDEX_PATH):
    index_path.mkdir(parents=True, exist_ok=True)
    np.save(index_path / "vector_store.npy", matrix)
    (index_path / "docstore.json").write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(index_path=INDEX_PATH):
    npy, js = index_path / "vector_store.npy", index_path / "docstore.json"
    if not (npy.exists() and js.exists()):
        return None, None
    return np.load(npy), json.loads(js.read_text(encoding="utf-8"))


def index_exists(index_path=INDEX_PATH):
    return (index_path / "vector_store.npy").exists()
