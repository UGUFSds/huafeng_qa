from langchain_core.prompts import ChatPromptTemplate


def build_csv_tools_prompt() -> ChatPromptTemplate:
    """构建 CSV 工具代理的系统提示，生成结构化候选。

    期望占位符：{q}（用户问题文本）、{now}（当前时间）。
    """
    system = (
        "你是内存 DataFrame 'csv_lookup' 的工具调用助手。"
        "当前时间：{now}。"
        "使用提供的工具检索结构化候选（桥接列如 point_id/tag/name/desc）。"
        "如问题涉及时间但未指定年份，默认按今年处理。"
        "禁止访问本地文件或执行任意代码。请用简体中文简洁回答。"
    )
    human = "{q}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])