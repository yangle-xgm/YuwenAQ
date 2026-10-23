"""pytest 公共夹具（fixtures）。

分层原则：
  - unit      纯函数/纯本地，不碰网络，毫秒级；
  - retrieval 需要 Ollama 的 bge-m3 嵌入（秒级），但不需要生成；
  - slow      需要 LLM 生成（分钟级），默认不跑。

运行方式：
    pytest                     # 只跑 unit + retrieval（自动跳过 slow）
    pytest -m unit             # 只跑单元层
    pytest -m slow             # 只跑端到端（很慢）
"""
import shutil
import socket
import sys
import tempfile
from pathlib import Path

import pytest

# 让 tests/ 能 import qa_bot（pytest 的 rootdir 是项目根，这里再兜一层）
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_collection_modifyitems(config, items):
    """没有显式 -m slow 时，自动跳过 slow 用例（避免误跑几分钟）。"""
    if config.option.markexpr and "slow" in config.option.markexpr:
        return
    skip_slow = pytest.mark.skip(reason="slow 用例需要 LLM 生成，用 -m slow 显式运行")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


def _ollama_alive() -> bool:
    try:
        with socket.create_connection(("127.0.0.1", 11434), timeout=3):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def ollama_up():
    """Ollama 是否在线；离线时 retrieval/slow 用例应跳过而不是失败。"""
    if not _ollama_alive():
        pytest.skip("Ollama 未启动（127.0.0.1:11434），跳过需要嵌入/生成的用例")
    return True


@pytest.fixture(scope="session")
def retriever(ollama_up):
    """进程内只加载一次索引：session 级复用，避免每个用例重复读 1.2MB + 重算 IDF。"""
    from qa_bot.rag.index import get_retriever

    return get_retriever()


@pytest.fixture(scope="session")
def kb_dir():
    from qa_bot.config import KB_DIR

    return Path(KB_DIR)


@pytest.fixture
def tmp_kb():
    """临时知识库目录：用于构造切块/异常输入的受控语料。"""
    d = Path(tempfile.mkdtemp(prefix="qa_kb_"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def tmp_db(tmp_path, monkeypatch):
    """把 curriculum_db 的 DB_PATH 指向临时文件，避免污染项目 data/curriculum.db。"""
    from qa_bot.tools import curriculum_db

    p = tmp_path / "curriculum.db"
    monkeypatch.setattr(curriculum_db, "DB_PATH", p)
    curriculum_db.ensure_db()
    return p


@pytest.fixture
def history_file(tmp_path):
    return tmp_path / ".chat_history.json"
