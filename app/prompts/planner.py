from langchain_core.prompts import ChatPromptTemplate


def build_routing_planner_prompt() -> ChatPromptTemplate:
    """Builds the routing planner prompt used to select data sources.

    The prompt expects formatting variables: {lang}, {question}, {available}, {example_json}, {now}.
    """
    system_text = (
        "You are a routing planner. Choose the best data sources to answer industrial QA queries. "
        "Respond in strict JSON with keys: ordered_sources (list of source names in execution order) "
        "and strategy (short reasoning). Use only source names provided. Always include at least one source. "
        "If time context is implied but the year is not specified, plan queries to default to the current year."
    )
    human_text = (
        "User language: {lang}\n"
        "Current datetime: {now}\n"
        "Question: {question}\n"
        "Available sources:\n{available}\n"
        "Return JSON: {{\"ordered_sources\": [\"name\"...], \"strategy\": \"...\"}}\n"
        "Example: {example_json}"
    )
    return ChatPromptTemplate.from_messages([
        ("system", system_text),
        ("human", human_text),
    ])