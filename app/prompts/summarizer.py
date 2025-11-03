from langchain_core.prompts import ChatPromptTemplate


def build_evidence_summarizer_prompt() -> ChatPromptTemplate:
    """Builds the summarizer prompt that merges multi-source outputs.

    The prompt expects formatting variables: {lang}, {strategy}, {ordered}, {evidence}, {now}.
    """
    system = (
        "You are a summarizer. Merge the information from multiple sources into a single, concise answer. "
        "At the end, append a short citations section listing which sources contributed (by source name). Respond in the user's language."
    )
    human = (
        "Language: {lang}\n"
        "Current datetime: {now}\n"
        "Routing strategy: {strategy}\n"
        "Ordered sources: {ordered}\n"
        "Collected evidence:\n{evidence}\n"
        "Return the merged answer first, then a citations list."
    )
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])