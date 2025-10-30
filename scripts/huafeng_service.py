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

# API Key 提示：本地运行使用 deepseek-api，如缺少 Key 可能无法调用
if not API_KEY:
    print("[warn] HUAFENG_DEEPSEEK_API_KEY 未设置；可能无法调用 DeepSeek API")

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_core.callbacks.base import BaseCallbackHandler
import time


def build_llm(base_url: str) -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=base_url,
        temperature=0.0,
    )


def build_agent(base_url: str, max_iterations: int = 20):
    db = SQLDatabase.from_uri(DB_URI)
    llm = build_llm(base_url)
    agent = create_sql_agent(
        llm,
        db=db,
        agent_type="tool-calling",
        verbose=False,  # 关闭默认英文日志
        max_iterations=max_iterations,
    )
    return agent

# 新增：命令行参数解析

def parse_args():
    parser = argparse.ArgumentParser(description="Huafeng 交互服务")
    parser.add_argument("--question", "-q", help="以非交互方式提交一次性问题")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="输出语言（默认中文）")
    parser.add_argument("--max-steps", type=int, default=20, help="Agent 最大迭代步数（默认20）")
    return parser.parse_args()


class TokenUsageHandler(BaseCallbackHandler):
    """Collects token usage and LLM-only runtime per single run."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.llm_runtime_sec = 0.0
        self._llm_timer_stack = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        self._llm_timer_stack.append(time.perf_counter())

    def on_llm_end(self, response, **kwargs):
        # accumulate LLM generation time
        try:
            start_ts = self._llm_timer_stack.pop() if self._llm_timer_stack else None
            if start_ts is not None:
                self.llm_runtime_sec += (time.perf_counter() - start_ts)
        except Exception:
            pass
        # accumulate tokens
        try:
            usage = response.llm_output.get("token_usage")
        except Exception:
            usage = None
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0))
        else:
            # Fallback: approximate tokens from output text
            approx_tokens = 0
            try:
                import tiktoken
                enc = tiktoken.get_encoding("cl100k_base")
                for gen_list in getattr(response, "generations", []):
                    for gen in gen_list:
                        text = getattr(gen, "text", "")
                        approx_tokens += len(enc.encode(text))
            except Exception:
                for gen_list in getattr(response, "generations", []):
                    for gen in gen_list:
                        text = getattr(gen, "text", "")
                        approx_tokens += max(1, len(text) // 4)
            self.completion_tokens += approx_tokens
            self.total_tokens += approx_tokens

def localize_question(text: str, lang: str = "zh") -> str:
    policy_zh = (
        "【数据库先验与使用政策】\n"
        "- alarm_event 为报警总表；opcae_all 为点位总表；其他为按点位拆分的历史表。\n"
        "- 禁止无差别枚举所有表；默认优先使用两个总表满足问题。\n"
        "- 仅当问题明确给出 point_id 且需要历史曲线/时间段数据时，再访问对应历史表（按命名模式匹配）。\n"
        "- 执行前尽量进行语句校验；时间窗口与返回行数需受限（如 LIMIT/聚合）。"
    )
    policy_en = (
        "[DB prior and policy]\n"
        "- alarm_event is the alarm master; opcae_all is the point master; others are per-point history tables.\n"
        "- Do not enumerate all tables; default to two masters first.\n"
        "- Only access history tables when a specific point_id and time window are required (match by naming pattern).\n"
        "- Validate SQL before execution; limit time window and rows (e.g., LIMIT/aggregation)."
    )
    if lang == "zh":
        return f"请用简体中文回答：\n{policy_zh}\n现在的问题：{text}"
    return f"Please answer in English.\n{policy_en}\nQuestion: {text}"


class ChineseConsoleCallback(BaseCallbackHandler):
    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self._top_run_id = None

    def on_chain_start(self, serialized, inputs, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        if parent_run_id is None:
            self._top_run_id = run_id

    def on_agent_action(self, action, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        log = getattr(action, "log", "") or ""
        s = log.strip()
        if s:
            lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
            out_lines = []
            for ln in lines:
                if ln.startswith("Invoking:"):
                    # 跳过英文 Invoking 行，工具调用由 on_tool_start 输出
                    continue
                if ln.startswith("responded:"):
                    msg = ln[len("responded:"):].strip()
                    if len(msg) > 600:
                        msg = msg[:600] + " …（截断）"
                    out_lines.append(f"【计划说明】{msg}")
                else:
                    if len(ln) > 600:
                        ln = ln[:600] + " …（截断）"
                    out_lines.append(f"【思考/计划】{ln}")
            if out_lines:
                print("\n".join(out_lines))

    def on_tool_start(self, tool, input_str, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        name = getattr(tool, "name", None)
        if not name:
            name = tool.get("name") if isinstance(tool, dict) else str(tool)
        inp = input_str if isinstance(input_str, str) else str(input_str)
        if len(inp) > 400:
            inp = inp[:400] + " …（截断）"
        print(f"【调用工具】{name} 入参：{inp}")

    def on_tool_end(self, output, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        out = str(output)
        if len(out) > 400:
            out = out[:400] + " …（截断）"
        print(f"【工具返回】{out}")

    def on_chain_end(self, outputs, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        if parent_run_id is None or (self._top_run_id is not None and run_id == self._top_run_id):
            print("> 链已完成。")


def main():
    args = parse_args()
    print("[env] BASE_URL=", BASE_URL)
    print("[env] MODEL=", MODEL)
    safe_uri = DB_URI.replace(PG_PASSWORD, "***")
    print("[env] DB_URI=", safe_uri)

    # Try primary base_url first, fallback to /v1 if needed
    selected_base = BASE_URL
    try:
        agent = build_agent(selected_base, max_iterations=getattr(args, "max_steps", 20))
    except Exception as e1:
        print("[warn] 构建Agent失败，尝试使用 fallback base_url:", e1)
        # deepseek-api 可能需要 /v1 后缀
        fallback_url = selected_base + "/v1"
        try:
            agent = build_agent(fallback_url, max_iterations=getattr(args, "max_steps", 20))
            print("[info] 使用 fallback base_url:", fallback_url)
        except Exception as e2:
            print("[error] fallback仍失败：", e2)
            sys.exit(1)

    # 如果传入了 --question，非交互执行一次
    if getattr(args, "question", None):
        try:
            handler = TokenUsageHandler()
            cn_cb = ChineseConsoleCallback(lang=args.lang)
            chain_start = time.perf_counter()
            q_text = localize_question(args.question, args.lang)
            result = agent.invoke({"input": q_text}, config={"callbacks": [handler, cn_cb]})
            chain_elapsed = time.perf_counter() - chain_start
            print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
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

    # 会话统计
    session_prompt_tokens = 0
    session_completion_tokens = 0
    session_total_tokens = 0
    session_llm_calls = 0
    session_questions = 0
    session_llm_runtime_sec = 0.0
    session_chain_runtime_sec = 0.0
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
            handler = TokenUsageHandler()
            cn_cb = ChineseConsoleCallback(lang=args.lang)
            chain_start = time.perf_counter()
            q_text = localize_question(question, args.lang)
            result = agent.invoke({"input": q_text}, config={"callbacks": [handler, cn_cb]})
            chain_elapsed = time.perf_counter() - chain_start
            print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
            print("\n[LLM] ", result.get("output") or result)
            print()
            session_questions += 1
            session_llm_calls += handler.llm_calls
            session_prompt_tokens += handler.prompt_tokens
            session_completion_tokens += handler.completion_tokens
            session_total_tokens += handler.total_tokens
            session_llm_runtime_sec += handler.llm_runtime_sec
            session_chain_runtime_sec += chain_elapsed
        except Exception as e:
            print("[error] 调用失败：", e)

    if session_questions > 0:
        avg_llm = session_llm_runtime_sec / session_questions
        avg_chain = session_chain_runtime_sec / session_questions
        print(f"[metrics] 会话统计：问题数 {session_questions}，LLM调用数 {session_llm_calls}，总tokens {session_total_tokens}（prompt {session_prompt_tokens}, completion {session_completion_tokens}），链总用时 {session_chain_runtime_sec:.3f}s，LLM总用时 {session_llm_runtime_sec:.3f}s，平均每问：链 {avg_chain:.3f}s / LLM {avg_llm:.3f}s")

if __name__ == "__main__":
    main()