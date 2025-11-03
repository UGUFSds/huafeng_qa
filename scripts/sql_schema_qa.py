import os
from dotenv import load_dotenv
from app.config.settings import (
    BASE_URL,
    API_KEY,
    MODEL,
    DB_URI,
    PG_PASSWORD,
)

print(
    f"[env] BASE_URL={BASE_URL}, MODEL={MODEL}, DB_URI={DB_URI.replace(PG_PASSWORD, '***')}"
)

if not API_KEY:
    raise RuntimeError("LLM_API_KEY 未设置，无法调用 LLM 提供方 API")

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent


def build_llm(base_url: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=base_url,
        temperature=0.0,
    )


def run_schema_qa(base_url: str, question: str):
    # Connect DB
    db = SQLDatabase.from_uri(DB_URI)
    # Build SQL agent that can list tables and fetch schemas
    llm = build_llm(base_url)
    agent = create_sql_agent(
        llm,
        db=db,
        agent_type="tool-calling",  # use OpenAI tool-calling style
        verbose=True,
        max_iterations=10,
    )
    result = agent.invoke({"input": question})
    return result["output"]


if __name__ == "__main__":
    # Default question can be overridden by env SCHEMA_QUESTION
    question = os.getenv(
        "SCHEMA_QUESTION",
        "请详细说明我的数据库结构：列出所有表及主要字段，并说明主键/外键关系（如有）。",
    )
    try:
        answer = run_schema_qa(BASE_URL, question)
        print("\n[answer]", answer)
    except Exception as e1:
        print("[warn] 首次调用失败：", e1)
        # Fallback: some deployments require /v1 path suffix
        fallback_url = BASE_URL + "/v1"
        print("[info] 尝试使用 fallback base_url:", fallback_url)
        try:
            answer = run_schema_qa(fallback_url, question)
            print("\n[answer]", answer)
        except Exception as e2:
            print("[error] fallback 调用仍失败：", e2)
            raise