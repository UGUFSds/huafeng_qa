import os
import sys
import asyncio
import json
import argparse
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List

from app.config.settings import (
    BASE_URL,
    API_KEY,
    PROVIDER,
    INTERNAL_BASE_URL,
    MODEL,
    PG_HOST,
    PG_PORT,
    PG_DB,
    PG_USER,
    PG_PASSWORD,
    PG_SSLMODE,
    DB_URI,
    OPCAE_CSV_PATH,
)
from app.llm.factory import build_llm
from app.callbacks import TokenUsageHandler, ChineseConsoleCallback

# API Key 提示：本地运行使用提供方 API，如缺少 Key 可能无法调用
if not API_KEY:
    print("[warn] LLM_API_KEY 未设置；可能无法调用 LLM 提供方 API")

import time

# 增强输入：优先使用 prompt_toolkit 支持左右键与历史，失败则回退到内置 input()
_USE_PTK = False
try:
    from prompt_toolkit import prompt as pt_prompt
    from prompt_toolkit.history import InMemoryHistory
    _ptk_history = InMemoryHistory()
    _USE_PTK = True
except Exception:
    _USE_PTK = False

def read_line(prompt_text: str) -> str:
    if _USE_PTK:
        try:
            return pt_prompt(prompt_text, history=_ptk_history)
        except Exception:
            pass
    return input(prompt_text)

from app.sources.base import DataSource
from app.sources.sql import build_sql_source
from app.sources.csv import build_csv_source
from app.router import (
    RoutingOrchestrator,
    AVAILABLE_SOURCES,
    register_data_sources,
    format_available_sources,
    format_schema_notes,
    extract_agent_output,
    localize_question,
)


# 数据源接口及实现位于 app.sources


# LLM 构建函数位于 app.llm.factory


"""数据源构建函数见 app.sources.csv 与 app.sources.sql。"""

# 新增：命令行参数解析

def parse_args():
    parser = argparse.ArgumentParser(description="交互式问答服务")
    parser.add_argument("--question", "-q", help="以非交互方式提交一次性问题")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="输出语言（默认中文）")
    parser.add_argument("--max-steps", type=int, default=20, help="Agent 最大迭代步数（默认20）")
    return parser.parse_args()



# 路由相关工具与注册位于 app.router

# 提示本地化见 app.router.localize_question






# 路由器实现见 app.router.RoutingOrchestrator


def main():
    args = parse_args()
    print("[env] BASE_URL=", BASE_URL)
    print("[env] PROVIDER=", PROVIDER)
    if INTERNAL_BASE_URL != BASE_URL:
        print("[env] INTERNAL_BASE_URL=", INTERNAL_BASE_URL)
    print("[env] MODEL=", MODEL)
    safe_uri = DB_URI.replace(PG_PASSWORD, "***")
    print("[env] DB_URI=", safe_uri)
    print("[env] OPCAE_CSV_PATH=", OPCAE_CSV_PATH)

    # 根据 provider 选择 base_url；必要时回退到 /v1
    selected_base = INTERNAL_BASE_URL if PROVIDER in {"internal", "gateway"} else BASE_URL
    csv_source = None
    active_base_url = selected_base
    try:
        sql_source = build_sql_source(selected_base, max_iterations=getattr(args, "max_steps", 20))
    except Exception as e1:
        print("[warn] 构建Agent失败，尝试使用 fallback base_url:", e1)
        # deepseek-api 可能需要 /v1 后缀
        fallback_url = selected_base + "/v1"
        try:
            sql_source = build_sql_source(fallback_url, max_iterations=getattr(args, "max_steps", 20))
            print("[info] 使用 fallback base_url:", fallback_url)
            active_base_url = fallback_url
        except Exception as e2:
            print("[error] fallback仍失败：", e2)
            sys.exit(1)
        csv_source = build_csv_source(fallback_url, max_iterations=getattr(args, "max_steps", 20))
    else:
        csv_source = build_csv_source(selected_base, max_iterations=getattr(args, "max_steps", 20))

    sources = [sql_source]
    if csv_source:
        sources.append(csv_source)
    register_data_sources(sources)
    print("[env] AVAILABLE_SOURCES=")
    for src in sources:
        print(f"  - {src.short_info()}")
    planner_llm = build_llm(active_base_url)
    router = RoutingOrchestrator(planner_llm, AVAILABLE_SOURCES)
    print("[env] ACTIVE_BASE_URL=", active_base_url)

    # 如果传入了 --question，非交互执行一次
    if getattr(args, "question", None):
        try:
            handler = TokenUsageHandler()
            cn_cb = ChineseConsoleCallback(lang=args.lang)
            chain_start = time.perf_counter()
            q_text = localize_question(args.question, args.lang)
            route_result = router.execute(q_text, lang=args.lang, callbacks=[handler, cn_cb])
            chain_elapsed = time.perf_counter() - chain_start
            plan_info = route_result.get("plan", {})
            ordered = plan_info.get("ordered_sources", [])
            if ordered:
                print("[route] 顺序:", " -> ".join(ordered))
            strategy = plan_info.get("strategy")
            if strategy:
                print("[route] 策略:", strategy)
            final_text = route_result.get("final_text") or extract_agent_output(route_result.get("outputs"))
            print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
            print("\n[LLM] ", final_text)
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
    if _USE_PTK:
        print("[env] 已启用增强输入（支持方向键、历史与编辑）")
    else:
        term = os.environ.get("TERM", "")
        print("[warn] 当前为基础输入模式。若方向键输出 ^[[C/^[[D，请安装 prompt_toolkit：pip install prompt_toolkit")
        if term.lower() in {"dumb", ""}:
            print("[hint] 检测到 TERM='dumb'，建议在系统终端运行或设置 TERM=xterm-256color")

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
            question = read_line("[you] > ").strip()
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
            route_result = router.execute(q_text, lang=args.lang, callbacks=[handler, cn_cb])
            chain_elapsed = time.perf_counter() - chain_start
            plan_info = route_result.get("plan", {})
            ordered = plan_info.get("ordered_sources", [])
            if ordered:
                print("[route] 顺序:", " -> ".join(ordered))
            strategy = plan_info.get("strategy")
            if strategy:
                print("[route] 策略:", strategy)
            final_text = route_result.get("final_text") or extract_agent_output(route_result.get("outputs"))
            print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
            print("\n[LLM] ", final_text)
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
