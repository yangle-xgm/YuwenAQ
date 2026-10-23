"""工具注册表（2.6 插件机制的核心容器）。

对应阿里文档「Function Calling」里 tools 列表 + function_mapper 的解耦设计：
  - 模型（路由）只认「工具名 + 描述 + 参数说明」这些字符串元数据；
  - 真正的 Python 函数通过注册表与工具名一一映射，代码负责执行。

本地适配要点：
  - 用 @tool 装饰器把「工具元数据」和「实现函数」绑定到 TOOL_REGISTRY；
  - dispatch(name, arg) 就是文档里 function_mapper[name](**param) 的本地等价物；
  - 全部工具的「实现」都是确定性本地逻辑（查本地库 / 关键词匹配 / 纯计算），
    不依赖联网大模型，保证离线 100% 可跑、可复现。
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List


@dataclass
class Tool:
    """一个本地工具的描述与实现。"""
    name: str                       # 工具名（模型/路由据此决策，建议简短英文）
    description: str                # 给模型看的能力说明，写得越准调用越准
    parameters: Dict[str, str]      # 参数名 -> 中文说明（用于路由提示词）
    func: Callable[[str], str]      # 实现：吃一个字符串参数 arg，返回人类可读结果


# name -> Tool。这是文档 function_mapper 的本地等价物（协议名与实现体解耦）。
TOOL_REGISTRY: Dict[str, Tool] = {}


def tool(name: str, description: str, parameters: Dict[str, str]):
    """装饰器：把一个函数注册成名为 name 的工具。

    用法：
        @tool("query_curriculum_db", "查询课标结构化指标", {"query": "自然语言问题"})
        def _impl(arg: str) -> str:
            ...
    """
    def _wrap(func: Callable[[str], str]) -> Callable[[str], str]:
        TOOL_REGISTRY[name] = Tool(
            name=name, description=description,
            parameters=parameters, func=func,
        )
        return func
    return _wrap


def get_tool(name: str):
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[Tool]:
    return list(TOOL_REGISTRY.values())


def get_tool_specs() -> str:
    """把工具清单渲染成给路由提示词用的文本（名称 + 描述 + 参数）。"""
    lines = []
    for t in list_tools():
        params = "、".join(f"{k}（{v}）" for k, v in t.parameters.items())
        desc = t.description.rstrip("。")  # 避免与拼接的"。"重复
        lines.append(f"- {t.name}：{desc}。参数：{params}。")
    return "\n".join(lines)


def dispatch(name: str, arg: str) -> str:
    """按工具名执行对应函数，返回结果字符串。找不到工具时给出友好提示。

    等价于文档 get_agent_response 里的 function_mapper[func_name](**param)。
    """
    t = TOOL_REGISTRY.get(name)
    if t is None:
        return f"（错误：未找到名为「{name}」的工具，可用工具：{', '.join(TOOL_REGISTRY.keys())}）"
    try:
        return t.func(arg or "")
    except Exception as e:  # 工具执行异常不应拖垮整个对话
        return f"（工具「{name}」执行出错：{e}）"
