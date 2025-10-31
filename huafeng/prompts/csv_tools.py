from langchain_core.prompts import ChatPromptTemplate


def build_csv_tools_prompt() -> ChatPromptTemplate:
    """Builds the CSV tools agent prompt to produce structured candidates.

    Expects formatting variable: {q} (user query text).
    """
    system = (
        "You are a CSV tool-calling assistant for the in-memory DataFrame 'opcae_lookup'. "
        "Use the provided tools to retrieve structured candidates (bridge keys like point_id/tag/name/desc). "
        "Do not access any local files or run arbitrary code. Respond concisely in the user's language."
    )
    human = "{q}"
    return ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", human),
    ])