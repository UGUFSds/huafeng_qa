from langchain_core.prompts import ChatPromptTemplate


def build_routing_planner_prompt() -> ChatPromptTemplate:
    """构建路由规划提示：选择数据源并返回严格 JSON。

    期望占位符：{lang}、{question}、{available}、{example_json}、{now}
    """
    system_text = (
        "你是路由规划器，请为工业问答选择最合适的数据源并给出执行顺序。"
        "仅用提供的来源名称，且至少包含一个来源。"
        "以严格 JSON 返回：ordered_sources（按执行顺序的来源名称列表）和 strategy（简要策略）。"
        "如问题涉及时间但未指定年份，规划时默认按今年处理。"
    )
    human_text = (
        "语言：{lang}\n"
        "当前时间：{now}\n"
        "问题：{question}\n"
        "可用数据源：\n{available}\n"
        "返回 JSON：{{\"ordered_sources\": [\"name\"...], \"strategy\": \"...\"}}\n"
        "示例：{example_json}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_text),
    ])