from langchain_core.prompts import ChatPromptTemplate


def build_evidence_summarizer_prompt() -> ChatPromptTemplate:
    """构建证据汇总提示：合并多源结果并附引用。

    期望占位符：{lang}、{strategy}、{ordered}、{evidence}、{now}
    """
    system = (
        "你是汇总器，请将多个来源的信息合并为简洁一致的答案。"
        "结尾追加来源引用（使用来源名称）。用简体中文回答。"
    )
    human = (
        "语言：{lang}\n"
        "当前时间：{now}\n"
        "路由策略：{strategy}\n"
        "执行顺序：{ordered}\n"
        "收集的证据：\n{evidence}\n"
        "先输出合并后的答案，再输出引用列表。"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])