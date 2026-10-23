"""命令行交互（演示 2.1：多轮 + 流式 + 思考分离；2.6：工具增强路由）。

命令：/new 清历史   /reason 开/关思考过程   /kb 开/关知识注入
      /smart 开/关 2.6 智能路由（工具调用+RAG+闲聊自动分流）
      /tools 列出已注册工具   /route 仅看路由决策（不作答）
      /build 重建索引   /eval 跑 2.4 自动化测评   /exit 退出
"""
from .bot.core import CourseQA
from .rag.index import build_index
from .evaluation.runner import run_evaluation
from .tools import list_tools, get_tool_specs
from .routing import route
from .config import PROJECT_DIR


# 对话历史持久化文件：让 CLI 重启后仍能接着上次的对话（2.8 修复"两次对话独立存在"）。
HISTORY_PATH = PROJECT_DIR / ".chat_history.json"


def main():
    bot = CourseQA()
    show_reasoning = True
    smart_mode = True   # 默认开启 2.6 智能路由
    restored = bot.load_history(HISTORY_PATH)
    print("语文小蜜已上线（输入 /exit 退出，/new 开新对话）")
    print("默认已开启 2.6 智能路由：RAG / 工具调用 / 闲聊 自动分流。")
    if restored:
        print("（已自动恢复上次的对话，可继续追问；输入 /new 可开新对话）\n")
    else:
        print()
    while True:
        try:
            q = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q == "/build":
            build_index()
            print("（已重建知识库索引）")
            continue
        if q == "/eval":
            _, summary = run_evaluation()
            print("\n=== 2.4 评测汇总 ===")
            for k, v in summary.items():
                print(f"  {k}: {v}")
            continue
        if q == "/exit":
            break
        if q == "/new":
            bot.reset()
            bot.save_history(HISTORY_PATH)
            print("（已开始新的对话）")
            continue
        if q == "/reason":
            show_reasoning = not show_reasoning
            print(f"（思考过程：{'开' if show_reasoning else '关'}）")
            continue
        if q == "/kb":
            bot.use_kb = not bot.use_kb
            print(f"（知识注入：{'开' if bot.use_kb else '关'}）")
            continue
        if q == "/smart":
            smart_mode = not smart_mode
            print(f"（2.6 智能路由：{'开' if smart_mode else '关'}）")
            continue
        if q == "/tools":
            print("=== 已注册工具 ===")
            for t in list_tools():
                params = "、".join(f"{k}（{v}）" for k, v in t.parameters.items())
                print(f"- {t.name}：{t.description}\n    参数：{params}")
            continue
        if q.startswith("/route "):
            decision = route(q[len("/route "):].strip())
            print(f"〔路由决策〕{decision}")
            continue
        # ----- 2.6 智能路由通道 -----
        if smart_mode:
            decision, answer = bot.ask_smart(q)
            print(f"〔路由〕{decision.channel}"
                  + (f" -> 工具 {decision.tool}" if decision.tool else ""))
            if bot.last_rewrite:
                print(f"〔追问改写〕{bot.last_rewrite}")
            print("语文小蜜：", end="", flush=True)
            print(answer)
            bot.save_history(HISTORY_PATH)
            continue
        # ----- 传统 RAG 通道（/smart 关闭时）-----
        print("语文小蜜：", end="", flush=True)
        if show_reasoning:
            in_reasoning = False
            for ev in bot.ask_stream_reasoning(q, think=True):
                if ev["type"] == "reasoning":
                    if not in_reasoning:
                        print("\n〔思考〕", end="", flush=True)
                        in_reasoning = True
                    print(ev["text"], end="", flush=True)
                elif ev["type"] == "answer_start":
                    print("\n\n〔回答〕", end="", flush=True)
                    in_reasoning = False
                elif ev["type"] == "content":
                    print(ev["text"], end="", flush=True)
            print()
            bot.save_history(HISTORY_PATH)
        else:
            # 快速模式：关闭思考（think=False），配合 factual 预设的确定性参数，更适合事实类快问
            for piece in bot.ask_stream(q, think=False):
                print(piece, end="", flush=True)
            print()
            bot.save_history(HISTORY_PATH)


if __name__ == "__main__":
    main()

