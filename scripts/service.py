import os
import sys
import asyncio
import json
import argparse
# 保证项目根目录加入 sys.path（脚本位于 scripts/ 下时需要）
try:
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
except Exception:
    pass
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
    EVALS_REPORT_DIR,
)
from app.llm.factory import build_llm
from app.callbacks import TokenUsageHandler, ChineseConsoleCallback
from app.monitor import maybe_run_evals

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
    parser.add_argument("--clarify", help="澄清选择的索引列表（逗号或空格分隔），例如: 0,2")
    parser.add_argument("--report-dir", default=EVALS_REPORT_DIR or "", help="将本次查询报告写入该目录（JSON），为空则不写")
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
    if getattr(args, "report_dir", ""):
        print("[env] REPORT_DIR=", args.report_dir)

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

    def _write_per_query_report(report_dir: str, payload: Dict[str, Any]):
        try:
            if not report_dir:
                return
            os.makedirs(report_dir, exist_ok=True)
            # timestamped filename
            ts = time.strftime("%Y%m%d_%H%M%S")
            fname = f"query_report_{ts}.json"
            path = os.path.join(report_dir, fname)
            with open(path, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
            print(f"[report] 已写入: {path}")
        except Exception as e:
            print("[warn] 写报告失败:", e)

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
            needs = route_result.get("needs_clarification")
            if needs:
                clar = route_result.get("clarification") or {}
                msg = clar.get("message") or "需要澄清以继续查询。"
                options = clar.get("options") or []
                print("\n[clarify] ", msg)
                if options:
                    print("[options]")
                    for o in options:
                        print(f"  {o.get('index')}. {o.get('label')}")
                # 若提供 --clarify，则按选择继续执行一次
                if getattr(args, "clarify", None):
                    try:
                        raw = args.clarify.strip()
                        parts = [p for p in raw.replace(",", " ").split() if p]
                        indices = [int(p) for p in parts]
                        print("[clarify] 选择索引:", indices)
                        handler2 = TokenUsageHandler()
                        cn_cb2 = ChineseConsoleCallback(lang=args.lang)
                        chain_start2 = time.perf_counter()
                        route_result2 = router.execute(q_text, lang=args.lang, callbacks=[handler2, cn_cb2], clarify_choice={"indices": indices})
                        chain_elapsed2 = time.perf_counter() - chain_start2
                        final_text2 = route_result2.get("final_text") or extract_agent_output(route_result2.get("outputs"))
                        print(f"[metrics] 本轮链用时 {chain_elapsed2:.3f}s，LLM用时 {handler2.llm_runtime_sec:.3f}s，LLM调用数 {handler2.llm_calls}，tokens {handler2.total_tokens}（prompt {handler2.prompt_tokens}, completion {handler2.completion_tokens}）")
                        print("\n[LLM] ", final_text2)
                    except Exception as e:
                        print("[error] 澄清执行失败：", e)
                else:
                    print("\n[hint] 使用 --clarify 传入索引列表以继续，如: --clarify '0,2'")
            else:
                final_text = route_result.get("final_text") or extract_agent_output(route_result.get("outputs"))
                print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
                print("\n[LLM] ", final_text)
                # Optional evals (sampled and guarded)
                eval_res = None
                try:
                    eval_res = maybe_run_evals(args.question, final_text, route_result.get("outputs") or [], {"mode": "non_interactive", "clarified": False})
                except Exception:
                    eval_res = None
                # Per-query report (JSON)
                try:
                    rep = {
                        "question": args.question,
                        "final_text": final_text,
                        "metrics": {
                            "chain_elapsed_sec": chain_elapsed,
                            "llm_elapsed_sec": handler.llm_runtime_sec,
                            "llm_calls": handler.llm_calls,
                            "prompt_tokens": handler.prompt_tokens,
                            "completion_tokens": handler.completion_tokens,
                            "total_tokens": handler.total_tokens,
                        },
                        "plan": plan_info,
                        "sources": [o.get("source") for o in (route_result.get("outputs") or []) if isinstance(o, dict)],
                        "evals": eval_res,
                        "base_url": active_base_url,
                        "lang": args.lang,
                        "clarified": False,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    _write_per_query_report(getattr(args, "report_dir", ""), rep)
                except Exception:
                    pass
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
            needs = route_result.get("needs_clarification")
            if needs:
                clar = route_result.get("clarification") or {}
                msg = clar.get("message") or "需要澄清以继续查询。"
                options = clar.get("options") or []
                print("\n[clarify] ", msg)
                if options:
                    print("[options]")
                    for o in options:
                        print(f"  {o.get('index')}. {o.get('label')}")
                # 交互式读取索引列表
                raw = read_line("[clarify] 选择索引（逗号或空格分隔，回车跳过）: ").strip()
                if raw:
                    try:
                        parts = [p for p in raw.replace(",", " ").split() if p]
                        indices = [int(p) for p in parts]
                        handler2 = TokenUsageHandler()
                        cn_cb2 = ChineseConsoleCallback(lang=args.lang)
                        chain_start2 = time.perf_counter()
                        route_result2 = router.execute(q_text, lang=args.lang, callbacks=[handler2, cn_cb2], clarify_choice={"indices": indices})
                        chain_elapsed2 = time.perf_counter() - chain_start2
                        final_text2 = route_result2.get("final_text") or extract_agent_output(route_result2.get("outputs"))
                        print(f"[metrics] 本轮链用时 {chain_elapsed2:.3f}s，LLM用时 {handler2.llm_runtime_sec:.3f}s，LLM调用数 {handler2.llm_calls}，tokens {handler2.total_tokens}（prompt {handler2.prompt_tokens}, completion {handler2.completion_tokens}）")
                        print("\n[LLM] ", final_text2)
                        # Optional evals
                        eval_res2 = None
                        try:
                            eval_res2 = maybe_run_evals(question, final_text2, route_result2.get("outputs") or [], {"mode": "interactive", "clarified": True})
                        except Exception:
                            eval_res2 = None
                        # Per-query report
                        try:
                            rep2 = {
                                "question": question,
                                "final_text": final_text2,
                                "metrics": {
                                    "chain_elapsed_sec": chain_elapsed2,
                                    "llm_elapsed_sec": handler2.llm_runtime_sec,
                                    "llm_calls": handler2.llm_calls,
                                    "prompt_tokens": handler2.prompt_tokens,
                                    "completion_tokens": handler2.completion_tokens,
                                    "total_tokens": handler2.total_tokens,
                                },
                                "plan": route_result2.get("plan", {}),
                                "sources": [o.get("source") for o in (route_result2.get("outputs") or []) if isinstance(o, dict)],
                                "evals": eval_res2,
                                "base_url": active_base_url,
                                "lang": args.lang,
                                "clarified": True,
                                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                            }
                            _write_per_query_report(getattr(args, "report_dir", ""), rep2)
                        except Exception:
                            pass
                    except Exception as e:
                        print("[error] 澄清执行失败：", e)
                else:
                    print("[info] 跳过澄清，本次不继续 SQL 查询。")
            else:
                final_text = route_result.get("final_text") or extract_agent_output(route_result.get("outputs"))
                print(f"[metrics] 本轮链用时 {chain_elapsed:.3f}s，LLM用时 {handler.llm_runtime_sec:.3f}s，LLM调用数 {handler.llm_calls}，tokens {handler.total_tokens}（prompt {handler.prompt_tokens}, completion {handler.completion_tokens}）")
                print("\n[LLM] ", final_text)
                # Optional evals
                eval_res3 = None
                try:
                    eval_res3 = maybe_run_evals(question, final_text, route_result.get("outputs") or [], {"mode": "interactive", "clarified": False})
                except Exception:
                    eval_res3 = None
                # Per-query report
                try:
                    rep3 = {
                        "question": question,
                        "final_text": final_text,
                        "metrics": {
                            "chain_elapsed_sec": chain_elapsed,
                            "llm_elapsed_sec": handler.llm_runtime_sec,
                            "llm_calls": handler.llm_calls,
                            "prompt_tokens": handler.prompt_tokens,
                            "completion_tokens": handler.completion_tokens,
                            "total_tokens": handler.total_tokens,
                        },
                        "plan": plan_info,
                        "sources": [o.get("source") for o in (route_result.get("outputs") or []) if isinstance(o, dict)],
                        "evals": eval_res3,
                        "base_url": active_base_url,
                        "lang": args.lang,
                        "clarified": False,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    _write_per_query_report(getattr(args, "report_dir", ""), rep3)
                except Exception:
                    pass
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
