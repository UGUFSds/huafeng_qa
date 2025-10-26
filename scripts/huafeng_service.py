import os
import sys
from dotenv import load_dotenv
import argparse

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


def build_agent(base_url: str):
    db = SQLDatabase.from_uri(DB_URI)
    llm = build_llm(base_url)
    agent = create_sql_agent(
        llm,
        db=db,
        agent_type="tool-calling",
        verbose=True,
        max_iterations=10,
    )
    return agent

# 新增：命令行参数解析

def parse_args():
    parser = argparse.ArgumentParser(description="Huafeng 交互服务")
    parser.add_argument("--question", "-q", help="以非交互方式提交一次性问题")
    return parser.parse_args()


def main():
    args = parse_args()
    print("[env] BASE_URL=", BASE_URL)
    print("[env] MODEL=", MODEL)
    safe_uri = DB_URI.replace(PG_PASSWORD, "***")
    print("[env] DB_URI=", safe_uri)

    # Try primary base_url first, fallback to /v1 if needed
    try:
        agent = build_agent(BASE_URL)
    except Exception as e1:
        print("[warn] 构建Agent失败，尝试使用 fallback base_url:", e1)
        fallback_url = BASE_URL + "/v1"
        try:
            agent = build_agent(fallback_url)
            print("[info] 使用 fallback base_url:", fallback_url)
        except Exception as e2:
            print("[error] fallback仍失败：", e2)
            sys.exit(1)

    # 如果传入了 --question，非交互执行一次
    if getattr(args, "question", None):
        try:
            result = agent.invoke({"input": args.question})
            print("\n[LLM] ", result.get("output") or result)
        except Exception as e:
            print("[error] 调用失败：", e)
            sys.exit(1)
        return

    # 非交互环境检测
    if not sys.stdin.isatty():
        print("[warn] 检测到非交互环境：请在终端运行，或使用 --question 提交一次性问题。")
        sys.exit(2)

    print("\n交互式模式已启动。输入你的问题并回车提交。")
    print("输入 :quit 或 :exit 退出。")

    while True:
        try:
            question = input("[you] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] 已退出。")
            break
        if not question:
            continue
        if question in {":quit", ":exit"}:
            print("[info] 已退出。")
            break
        try:
            result = agent.invoke({"input": question})
            print("\n[LLM] ", result.get("output") or result)
            print()
        except Exception as e:
            print("[error] 调用失败：", e)


if __name__ == "__main__":
    main()