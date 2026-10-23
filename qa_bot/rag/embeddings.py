"""本地向量化把文字变成语义指纹
用 Ollama 的 OpenAI 兼容 /v1/embeddings 接口 + config.EMBED_MODEL
（当前 bge-m3，1024 维；曾用 nomic-embed-text，768 维——维度由模型决定，不写死）。
localhost 请求必须 trust_env=False，否则被本机代理拦掉——和聊天客户端同理。
"""

import time

import numpy as np
import httpx
from openai import OpenAI

from ..config import OLLAMA_BASE_URL, OLLAMA_API_KEY, EMBED_MODEL, EMBED_QUERY_PREFIX

BATCH_SIZE = 32      # 一次送多少条给 Ollama：太大会被它内部拒掉（tokenize 端口连不上）
MAX_RETRY = 3        # 每条批次失败后的重试次数（模型冷启动/偶发 400 时很有用）


def _client():
    return OpenAI(
        base_url=OLLAMA_BASE_URL.rstrip("/") + "/v1",
        api_key=OLLAMA_API_KEY,
        http_client=httpx.Client(trust_env=False, timeout=httpx.Timeout(120.0)),
    )


def _embed_batch(client, batch):
    """送一批（<=BATCH_SIZE）给 Ollama，失败重试；仍失败则抛异常。"""
    last_err = None
    for attempt in range(1, MAX_RETRY + 1):
        try:
            resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
            return [d.embedding for d in resp.data]
        except Exception as e:          # noqa: BLE001 - 本地服务偶发 400/连接问题，统一重试
            last_err = e
            time.sleep(1.5 * attempt)   # 退避：1.5s / 3s
    raise RuntimeError(f"嵌入批次失败（{len(batch)} 条，重试 {MAX_RETRY} 次）：{last_err}")


def embed_texts(texts):
    """批量把字符串列表变成向量矩阵（N, D，D=模型维度）；顺手归一化，方便后面直接点积当余弦。"""
    texts = [t for t in texts if t.strip()]
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)   # 维度由首批真实向量决定
    client = _client()
    vecs = []
    for i in range(0, len(texts), BATCH_SIZE):
        vecs.extend(_embed_batch(client, texts[i : i + BATCH_SIZE]))
    vecs = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms

def embed_query(text):
    """单条查询向量（D，）。提问时调用它给问题做“指纹”。

    bge-m3 检索场景官方建议 query 加指令前缀（见 config.EMBED_QUERY_PREFIX），
    文档侧不加——所以前缀只在这里拼，build_index 走的 embed_texts 不受影响。
    """
    return embed_texts([EMBED_QUERY_PREFIX + text if EMBED_QUERY_PREFIX else text])[0]