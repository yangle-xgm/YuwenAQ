"""RAG 层：知识加载 + 检索。"""
from .loader import load_documents
from .retriever import NaiveRetriever

__all__ = ["load_documents", "NaiveRetriever"]
