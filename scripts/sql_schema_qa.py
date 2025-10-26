import os
from dotenv import load_dotenv

# Load environment
load_dotenv()

BASE_URL = os.getenv("HUAFENG_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("HUAFENG_DEEPSEEK_API_KEY")
MODEL = (
    os.getenv("HUAFENG_TEXT2SQL_MODEL")
    or os.getenv("HUAFENG_ANALYSIS_MODEL")
    or "deepseek-chat"
)

# Postgres connection from .env
PG_HOST = os.getenv("HUAFENG_LOCAL_POSTGRES_HOST", "127.0.0.1")
PG_PORT = os.getenv("HUAFENG_LOCAL_POSTGRES_PORT", "5433")
PG_DB = os.getenv("HUAFENG_LOCAL_POSTGRES_DB", "huafeng_db")
PG_USER = os.getenv("HUAFENG_LOCAL_POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("HUAFENG_LOCAL_POSTGRES_PASSWORD", "postgres")
PG_SSLMODE = os.getenv("HUAFENG_POSTGRES_SSLMODE", "disable")

DB_URI = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"
)

print(
    f"[env] BASE_URL={BASE_URL}, MODEL={MODEL}, DB_URI={DB_URI.replace(PG_PASSWORD, '***')}"
)

if not API_KEY:
    raise RuntimeError("HUAFENG_DEEPSEEK_API_KEY 未设置，无法调用 DeepSeek API")

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
    # Default question can be overridden by env HUAFENG_SCHEMA_QUESTION
    question = os.getenv(
        "HUAFENG_SCHEMA_QUESTION",
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