"""2.4 评测数据集（语文课程标准领域）。

每个样本字段：
  question   : 用户问题
  key_facts  : 标准答案里必须出现的"关键事实"列表（用于确定性硬校验，
               对应 RAGAS 的 ground_truth 思想；比 LLM 裁判更可靠地验证事实正确性）
  should_know: True=知识库覆盖该问题；False=知识库未覆盖，期望机器人按 2.3 规范
               第 2 条"如实说明暂未覆盖"（用于测拒答能力）

设计思路（对齐 2.4 文档"同一问题 + 多种回答 + 统一标准"的好评测集原则）：
  - 覆盖"事实类"（识字量/任务群/阅读量/核心素养）+ "开放类"（写字教学建议）；
  - 故意放 2 道知识库未覆盖的陷阱题，验证"不知道就直说"的规范是否被遵守。
"""
EVAL_DATASET = [
    {
        "question": "语文课程核心素养包括哪四个方面？",
        "key_facts": ["文化自信", "语言运用", "思维能力", "审美创造"],
        "should_know": True,
    },
    {
        "question": "第一学段（1~2 年级）的识字量要求是多少？",
        "key_facts": ["1600", "左右"],
        "should_know": True,
    },
    {
        "question": "义务教育语文课程一共设置了几个学习任务群？",
        "key_facts": ["六", "六个"],
        "should_know": True,
    },
    {
        "question": "第三学段（5~6 年级）的课外阅读总量要求是多少？",
        "key_facts": ["100万"],
        "should_know": True,
    },
    {
        "question": "第四学段（7~9 年级）的课外阅读总量要求是多少？",
        "key_facts": ["260万"],
        "should_know": True,
    },
    {
        "question": "关于写字教学，语文课程标准提出了哪些教学建议？",
        "key_facts": [],
        "should_know": True,
    },
    # ---- 知识库未覆盖 / 超出语文课程范围的题 ----
    # expect_refuse=True  : 知识库确实未覆盖，按 2.3 规范应"如实说明暂未覆盖"（测拒答能力）
    # expect_refuse=False : 虽不依赖课程标准，但属合理请求（如写诗），不应强制拒答
    {
        "question": "2024 年诺贝尔文学奖得主是谁？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    # ---- 学段越界题（2.9）：知识库只覆盖义务教育 1~9 年级 ----
    # 这类题的正确行为是"声明超出覆盖范围 + 指路"，而不是拿无关段落自由发挥。
    # 它们专测 out-of-scope 拒答能力；修复前模型会编通用建议（越界幻觉）。
    {
        "question": "高一年级语文学习有哪些要求？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    {
        "question": "高三语文复习应该怎么安排？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    {
        "question": "幼儿园阶段如何进行语言启蒙？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    {
        "question": "大学语文课程的教学目标是什么？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    {
        "question": "考研语文该怎么复习？",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": True,
    },
    {
        "question": "请用英文写一首关于春天的小诗。",
        "key_facts": [],
        "should_know": False,
        "expect_refuse": False,
    },
]
