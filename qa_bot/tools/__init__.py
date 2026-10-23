"""tools 包：本地插件（Function Calling 工具）集合。

导入本包即触发各工具模块注册到 TOOL_REGISTRY（等价文档的 function_mapper 装配）。
要新增一个工具：在任一模块里用 @tool(...) 装饰函数即可，无需改动其他代码——
这正是文档第八段「给智能体加新工具」强调的"加工具只需注册"的本地落地。
"""
from .registry import (  # 注册表与调度（先导入，确保 TOOL_REGISTRY 存在）
    TOOL_REGISTRY, Tool, tool, get_tool,
    list_tools, get_tool_specs, dispatch,
)
# 导入具体工具模块（副作用：把它们 @tool 注册进 TOOL_REGISTRY）
from . import curriculum_db   # query_curriculum_db（约束式 NL2SQL）
from . import poem            # lookup_poem（古诗查询）
from . import text_utils      # analyze_text / calc_expression（动作型）

__all__ = [
    "TOOL_REGISTRY", "Tool", "tool", "get_tool",
    "list_tools", "get_tool_specs", "dispatch",
    "curriculum_db", "poem", "text_utils",
]
