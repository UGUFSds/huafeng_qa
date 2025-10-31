import os
import sys
import asyncio
import json
import argparse
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List

from huafeng.config.settings import (
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
from huafeng.llm.factory import build_llm
from huafeng.callbacks import TokenUsageHandler, ChineseConsoleCallback

# API Key 提示：本地运行使用 deepseek-api，如缺少 Key 可能无法调用
if not API_KEY:
    print("[warn] HUAFENG_DEEPSEEK_API_KEY 未设置；可能无法调用 DeepSeek API")

from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
import time

from huafeng.sources.base import DataSource
from huafeng.sources.sql import build_sql_source
from huafeng.sources.csv import build_csv_source


# DataSource 及具体实现已迁移至 huafeng.sources


# build_llm 已迁移至 huafeng.llm.factory


"""数据源构建函数已迁移至 huafeng.sources.csv 与 huafeng.sources.sql。"""

# 新增：命令行参数解析

def parse_args():
    parser = argparse.ArgumentParser(description="Huafeng 交互服务")
    parser.add_argument("--question", "-q", help="以非交互方式提交一次性问题")
    parser.add_argument("--lang", default="zh", choices=["zh", "en"], help="输出语言（默认中文）")
    parser.add_argument("--max-steps", type=int, default=20, help="Agent 最大迭代步数（默认20）")
    return parser.parse_args()



AVAILABLE_SOURCES: Dict[str, DataSource] = {}


def register_data_sources(sources: Iterable[DataSource]):
    AVAILABLE_SOURCES.clear()
    for src in sources:
        if isinstance(src, DataSource):
            AVAILABLE_SOURCES[src.name] = src


def format_available_sources() -> Dict[str, str]:
    if not AVAILABLE_SOURCES:
        return {
            "zh": "- 工业数据库（PostgreSQL）",
            "en": "- Industrial SQL database (PostgreSQL)",
        }
    lines = []
    for src in AVAILABLE_SOURCES.values():
        lines.append(f"- {src.short_info()}")
    joined = "\n".join(lines)
    return {"zh": joined, "en": joined}

def format_schema_notes() -> Dict[str, str]:
    """生成统一的 Schema 提示，包括 CSV 列与 SQL 可用表，以及跨源桥接键集合说明。"""
    csv_cols = []
    try:
        csv_src = AVAILABLE_SOURCES.get("opcae_lookup")
        if csv_src and hasattr(csv_src, "dataframe") and csv_src.dataframe is not None:
            csv_cols = [str(c) for c in list(csv_src.dataframe.columns)]
    except Exception:
        csv_cols = []
    db_tables = []
    try:
        sql_src = AVAILABLE_SOURCES.get("sql_database")
        if sql_src and hasattr(sql_src, "_db") and sql_src._db is not None:
            db_tables = sorted(list(sql_src._db.get_usable_table_names()))
    except Exception:
        db_tables = []
    zh = (
        "【Schema 提示】\n"
        f"- CSV(opcae_lookup) 列: {', '.join(csv_cols) if csv_cols else '未知'}\n"
        f"- SQL 可用表: {', '.join(db_tables[:8]) + (' 等' if len(db_tables) > 8 else '') if db_tables else '未知'}\n"
        "- 跨源桥接采用“相关列集合”，常见候选包含：point_id、tag/tag_name、name/desc、device_id、line_id。"
    )
    en = (
        "[Schema hints]\n"
        f"- CSV(opcae_lookup) columns: {', '.join(csv_cols) if csv_cols else 'unknown'}\n"
        f"- SQL usable tables: {', '.join(db_tables[:8]) + (' etc.' if len(db_tables) > 8 else '') if db_tables else 'unknown'}\n"
        "- Cross-source bridging uses a set of relevant columns: point_id, tag/tag_name, name/desc, device_id, line_id."
    )
    return {"zh": zh, "en": en}


def extract_agent_output(payload) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        return payload
    if isinstance(payload, (int, float)):
        return str(payload)
    if isinstance(payload, dict):
        for key in ("output", "result", "answer", "content", "data"):
            if key in payload and payload[key]:
                value = payload[key]
                if isinstance(value, (str, int, float)):
                    return str(value)
                try:
                    return json.dumps(value, ensure_ascii=False)
                except Exception:
                    return str(value)
        try:
            return json.dumps(payload, ensure_ascii=False)
        except Exception:
            return str(payload)
    if isinstance(payload, (list, tuple, set)):
        try:
            return json.dumps(list(payload), ensure_ascii=False)
        except Exception:
            return ", ".join(str(item) for item in payload)
    return str(payload)

def localize_question(text: str, lang: str = "zh") -> str:
    source_notes = format_available_sources()
    schema_notes = format_schema_notes()
    policy_zh = (
        "【数据源使用政策】\n"
        "- 点位主数据通过 opcae_lookup（CSV）获取。\n"
        "- 报警相关数据使用 alarm_event；历史曲线/时间段数据使用按点位拆分的历史表。\n"
        "- 禁止无差别枚举所有表；仅在明确给出检索键与时间窗口时访问历史表，并限制返回行数。\n"
        "- 执行前尽量进行语句校验；结果应简洁并附来源。"
    )
    policy_en = (
        "[Source usage policy]\n"
        "- Point master data is obtained from opcae_lookup (CSV).\n"
        "- Use alarm_event for alarms; use per-point history tables for time-series data.\n"
        "- Do not enumerate tables; only access history when specific keys and time windows are provided, with row/time limits.\n"
        "- Validate queries; keep answers concise with citations."
    )
    steps_zh = (
        "【任务流程提醒】\n"
        "1. 先基于可用数据源制定计划，至少选择一个来源。\n"
        "2. 如需点位主数据，调用 opcae_lookup（CSV）；如需历史/报警数据，调用 sql_database。\n"
        "3. 多源场景下按计划串行执行，并在最终回答中整合来源信息。"
    )
    steps_en = (
        "[Workflow reminder]\n"
        "1. Form a plan against the available sources and select at least one.\n"
        "2. Use opcae_lookup (CSV) for point master data; use sql_database for history/alarms.\n"
        "3. When multiple sources are involved, execute sequentially and consolidate the answer."
    )
    if lang == "zh":
        return (
            f"请用简体中文回答：\n{policy_zh}\n{steps_zh}\n【可用数据源】\n"
            f"{source_notes['zh']}\n{schema_notes['zh']}\n现在的问题：{text}"
        )
    return (
        f"Please answer in English.\n{policy_en}\n{steps_en}\n[Available sources]\n"
        f"{source_notes['en']}\n{schema_notes['en']}\nQuestion: {text}"
    )






class RoutingOrchestrator:
    """Top-level controller that routes queries across multiple data sources."""

    def __init__(self, planner_llm: ChatOpenAI, sources: Dict[str, DataSource]):
        self.planner_llm = planner_llm
        self.sources = dict(sources)

    # --- Static heuristics: pre-filter and ordering ---
    @staticmethod
    def _detect_intent(question: str, lang: str) -> Dict[str, Any]:
        text = (question or "").lower()
        zh = lang == "zh" or any("\u4e00" <= ch <= "\u9fff" for ch in question)
        # keywords
        kw_point_master_zh = ["点位", "主数据", "描述", "单位", "属性", "阈值", "类型", "说明"]
        kw_point_master_en = ["point", "master", "description", "unit", "attribute", "threshold", "type"]
        kw_history_zh = ["历史", "曲线", "趋势", "过去", "近", "时间段", "区间", "小时", "天", "周", "月"]
        kw_history_en = ["history", "trend", "curve", "past", "recent", "time window", "range", "hour", "day", "week", "month"]
        kw_alarm_zh = ["报警", "告警", "超限", "事件", "次数"]
        kw_alarm_en = ["alarm", "alert", "event", "count", "overlimit"]
        tokens_master = kw_point_master_zh if zh else kw_point_master_en
        tokens_history = kw_history_zh if zh else kw_history_en
        tokens_alarm = kw_alarm_zh if zh else kw_alarm_en
        def contains_any(tokens):
            return any(tok in text for tok in tokens)
        # naive point_id detection: alphanum with 3+ characters or explicit "point_id"
        import re
        point_ids = re.findall(r"\b([A-Za-z]{1,2}\d{3,}|point[_\- ]?id\s*[:=]?\s*([A-Za-z0-9_\-]+))\b", question)
        has_point_id = bool(point_ids)
        return {
            "is_point_master": contains_any(tokens_master),
            "is_history": contains_any(tokens_history),
            "is_alarm": contains_any(tokens_alarm),
            "has_point_id": has_point_id,
            "lang": "zh" if zh else "en",
        }

    def _apply_static_rules(self, question: str, lang: str) -> Tuple[List[str], str]:
        intent = self._detect_intent(question, lang)
        names = list(self.sources.keys())
        reason = ""
        # default order preference
        def order_by_pref(cands: List[str]) -> List[str]:
            pref = ["opcae_lookup", "sql_database"]
            ordered = [n for n in pref if n in cands]
            for n in cands:
                if n not in ordered:
                    ordered.append(n)
            return ordered
        if intent["is_point_master"] and "opcae_lookup" in names:
            cands = ["opcae_lookup", "sql_database"] if "sql_database" in names else ["opcae_lookup"]
            reason = "point master data -> CSV first, SQL optional"
            return order_by_pref(cands), reason
        if (intent["is_history"] or intent["is_alarm"]) and "sql_database" in names:
            cands = ["sql_database", "opcae_lookup"] if "opcae_lookup" in names else ["sql_database"]
            reason = "history/alarms -> SQL primary, CSV optional"
            return order_by_pref(cands), reason
        # fallback: use all sources
        return order_by_pref(names), "no strong signal -> use all"

    # --- Per-source query rewriting ---
    @staticmethod
    def _candidate_bridge_keys() -> List[str]:
        """返回跨源桥接的候选列集合（动态 + 常见）。"""
        keys: List[str] = []
        try:
            csv_src = AVAILABLE_SOURCES.get("opcae_lookup")
            if csv_src and hasattr(csv_src, "dataframe") and csv_src.dataframe is not None:
                cols = [str(c).lower() for c in list(csv_src.dataframe.columns)]
                keys.extend([c for c in cols if c in {"point_id", "tag", "tag_name", "name", "desc", "device_id", "line_id"}])
        except Exception:
            pass
        for k in ["point_id", "tag", "tag_name", "name", "desc", "device_id", "line_id"]:
            if k not in keys:
                keys.append(k)
        return keys
    @staticmethod
    def _extract_csv_candidates(raw: Any) -> Optional[str]:
        """从 CSV Agent 的原始输出中提取结构化候选（仅桥接列），并序列化为 JSON 字符串。"""
        try:
            data = raw.get("data") if isinstance(raw, dict) else None
            if not data:
                return None
            keys = RoutingOrchestrator._candidate_bridge_keys()
            rows = None
            if isinstance(data, dict):
                if isinstance(data.get("rows"), list):
                    rows = data["rows"]
                elif isinstance(data.get("groups"), list):
                    rows = data["groups"]
            candidates: List[Dict[str, str]] = []
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    cand: Dict[str, str] = {}
                    for k in keys:
                        v = item.get(k)
                        if isinstance(v, str) and v.strip():
                            cand[k] = v.strip()
                    if cand:
                        candidates.append(cand)
            if not candidates:
                return None
            # 去重并限制大小
            uniq: List[Dict[str, str]] = []
            seen: set = set()
            for c in candidates:
                key = json.dumps(c, ensure_ascii=False, sort_keys=True)
                if key not in seen:
                    uniq.append(c)
                    seen.add(key)
            out = uniq[:10]
            return json.dumps(out, ensure_ascii=False)
        except Exception:
            return None
    @staticmethod
    def _rewrite_for_csv(question: str, lang: str, intent: Dict[str, Any]) -> str:
        keys = RoutingOrchestrator._candidate_bridge_keys()
        keys_text = ", ".join(keys)
        if lang == "zh":
            return (
                "请使用点位主数据 CSV(opcae_lookup) 检索与问题相关的信息，并返回结构化候选。"
                "优先包含字段：point_id、desc(描述)、unit(单位)、type(类型)、threshold(阈值)，以及可能的桥接列集合："
                f"{keys_text}。"
                "如无法确定精确点位，请基于名称/描述进行合理匹配，但避免仅返回自由文本；"
                "尽量给出候选对象（含可作为后续 SQL 过滤的列值）。"
                "注意：禁止访问其他本地文件，仅使用已加载的 DataFrame。"
                "原始问题：" + question
            )
        return (
            "Use the point master CSV (opcae_lookup) to retrieve relevant info and return structured candidates. "
            "Prefer fields: point_id, desc(description), unit, type, threshold, and include likely bridge keys: "
            f"{keys_text}. "
            "If exact point is unclear, do reasonable match by name/desc; avoid returning only free text—"
            "provide candidate objects with columns usable for later SQL filtering. "
            "Do not access other local files; use only the in-memory DataFrame. "
            "Original question: " + question
        )

    @staticmethod
    def _rewrite_for_sql(question: str, lang: str, intent: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        keys = RoutingOrchestrator._candidate_bridge_keys()
        keys_text = ", ".join(keys)
        csv_hint = ""
        if context and isinstance(context.get("csv_text"), str):
            # 适度截断，避免提示过长
            t = context["csv_text"].strip()
            csv_hint = t[:500] + ("…" if len(t) > 500 else "")
        json_hint = ""
        if context and isinstance(context.get("csv_candidates_json"), str) and context.get("csv_candidates_json"):
            j = context["csv_candidates_json"].strip()
            json_hint = j[:800] + ("…" if len(j) > 800 else "")
        if lang == "zh":
            base = (
                "请使用工业数据库(sql_database)回答。点位主数据由 CSV(opcae_lookup) 提供。"
                "如需报警或历史数据：使用 alarm_event（报警总表）与按点位拆分的历史表。"
                "访问历史表应基于明确的检索键与时间窗口，限制返回条数。"
            )
            if csv_hint and json_hint:
                bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 CSV 候选：{csv_hint} 与 JSON 候选：{json_hint}。"
            elif csv_hint:
                bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 CSV 候选：{csv_hint}。"
            elif json_hint:
                bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 JSON 候选：{json_hint}。"
            else:
                bridge = f"优先使用桥接列进行过滤（例如 {keys_text}）。"
            return base + " " + bridge + " 原始问题：" + question
        base = (
            "Use the industrial SQL database. Point master data comes from the CSV (opcae_lookup). "
            "For alarms or history, use alarm_event and per-point history tables. "
            "Access history tables based on specific keys and time windows; limit returned rows. "
        )
        if csv_hint and json_hint:
            bridge = f"Prefer filtering via bridge keys (e.g., {keys_text}); consider CSV candidates: {csv_hint} and JSON candidates: {json_hint}. "
        elif csv_hint:
            bridge = f"Prefer filtering via bridge keys (e.g., {keys_text}); consider CSV candidates: {csv_hint}. "
        elif json_hint:
            bridge = f"Prefer filtering via bridge keys (e.g., {keys_text}); consider JSON candidates: {json_hint}. "
        else:
            bridge = f"Prefer filtering via bridge keys (e.g., {keys_text}). "
        return base + bridge + " Original question: " + question

    def _rewrite_queries_for_sources(self, question: str, ordered_sources: List[str], lang: str) -> Dict[str, str]:
        intent = self._detect_intent(question, lang)
        rewritten = {}
        for name in ordered_sources:
            if name == "opcae_lookup":
                rewritten[name] = self._rewrite_for_csv(question, lang, intent)
            elif name == "sql_database":
                rewritten[name] = self._rewrite_for_sql(question, lang, intent)
            else:
                rewritten[name] = question
        return rewritten

    # --- Summarize outputs with citations ---
    def _summarize_outputs(self, outputs: List[Dict[str, Any]], plan_data: Dict[str, Any], lang: str, callbacks: Optional[Sequence[BaseCallbackHandler]] = None) -> str:
        if not outputs:
            return ""
        try:
            cite_lines = []
            for item in outputs:
                src = item.get("source")
                text = extract_agent_output(item.get("raw")) or item.get("text") or ""
                snippet = text if len(text) <= 400 else text[:400] + "…"
                cite_lines.append(f"Source[{src}]: {snippet}")
            system = (
                "You are a summarizer. Merge the information from multiple sources into a single, concise answer. "
                "At the end, append a short citations section listing which sources contributed (by source name). Respond in the user's language."
            )
            human = (
                "Language: {lang}\n"
                "Routing strategy: {strategy}\n"
                "Ordered sources: {ordered}\n"
                "Collected evidence:\n{evidence}\n"
                "Return the merged answer first, then a citations list."
            )
            strategy = plan_data.get("strategy", "")
            ordered = " -> ".join(plan_data.get("ordered_sources", []))
            evidence = "\n".join(cite_lines)
            prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])
            messages = prompt.format_messages(lang=lang, strategy=strategy, ordered=ordered, evidence=evidence)
            cfg = {"callbacks": list(callbacks)} if callbacks else None
            resp = self.planner_llm.invoke(messages, config=cfg) if cfg else self.planner_llm.invoke(messages)
            return getattr(resp, "content", str(resp))
        except Exception:
            # Fallback: naive concatenation
            merged = []
            for item in outputs:
                merged.append(f"[{item.get('source')}] {item.get('text') or ''}")
            cites = ", ".join(s.get("source") for s in outputs)
            return ("\n".join(merged)) + f"\n\nSources: {cites}"

    @staticmethod
    def _notify(callbacks: Optional[Sequence[BaseCallbackHandler]], method: str, *args, **kwargs):
        if not callbacks:
            return
        for cb in callbacks:
            handler = getattr(cb, method, None)
            if callable(handler):
                try:
                    handler(*args, **kwargs)
                except Exception:
                    continue

    def _format_sources_for_prompt(self) -> str:
        lines = []
        for src in self.sources.values():
            lines.append(f"- {src.short_info()}")
        return "\n".join(lines)

    def _parse_plan(self, raw: str) -> Dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    return json.loads(raw[start : end + 1])
                except Exception:
                    return {}
        return {}

    def plan_sources(self, question: str, lang: str, callbacks: Optional[Sequence[BaseCallbackHandler]] = None):
        # Static pre-filter
        candidate_sources, reason = self._apply_static_rules(question, lang)
        # If planner LLM is unavailable (e.g., missing API key) or only one candidate, skip LLM planning.
        use_llm = bool(API_KEY) and len(candidate_sources) > 1
        available = self._format_sources_for_prompt()
        # Reduce available to candidates for prompting
        available_lines = []
        for name in candidate_sources:
            src = self.sources.get(name)
            if src:
                available_lines.append(f"- {src.short_info()}")
        available_text = "\n".join(available_lines) if available_lines else available
        system_text = (
            "You are a routing planner. Choose the best data sources to answer industrial QA queries. "
            "Respond in strict JSON with keys: ordered_sources (list of source names in execution order) "
            "and strategy (short reasoning). Use only source names provided. Always include at least one source."
        )
        human_text = (
            "User language: {lang}\n"
            "Question: {question}\n"
            "Available sources:\n{available}\n"
            "Return JSON: {{\"ordered_sources\": [\"name\"...], \"strategy\": \"...\"}}\n"
            "Example: {example_json}"
        )
        example_json = '{"ordered_sources": ["opcae_lookup", "sql_database"], "strategy": "先查CSV点位，再查数据库历史"}'
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_text),
                ("human", human_text),
            ]
        )
        if use_llm:
            messages = prompt.format_messages(lang=lang, question=question, available=available_text, example_json=example_json)
            config = {"callbacks": list(callbacks)} if callbacks else None
            response = self.planner_llm.invoke(messages, config=config) if config else self.planner_llm.invoke(messages)
            raw_content = getattr(response, "content", str(response))
            plan_data = self._parse_plan(raw_content)
        else:
            raw_content = json.dumps({"ordered_sources": candidate_sources, "strategy": f"heuristics: {reason}"}, ensure_ascii=False)
            plan_data = {"ordered_sources": list(candidate_sources), "strategy": f"heuristics: {reason}"}
        ordered = plan_data.get("ordered_sources") or []
        ordered_sources = []
        for name in ordered:
            if name in self.sources and name not in ordered_sources:
                ordered_sources.append(name)
        if not ordered_sources:
            # use candidate sources as default
            ordered_sources = list(candidate_sources) if candidate_sources else list(self.sources.keys())
        plan_data["ordered_sources"] = ordered_sources
        if not plan_data.get("strategy"):
            fallback_strategy = plan_data.get("reason") or plan_data.get("rationale") or ""
            plan_data["strategy"] = fallback_strategy
        return plan_data, raw_content

    async def _probe_async(self, source_names):
        async def probe_single(name: str):
            try:
                result = await asyncio.to_thread(self.sources[name].probe)
            except Exception as exc:
                result = f"[probe-error] {exc}"
            return name, result

        tasks = [asyncio.create_task(probe_single(name)) for name in source_names]
        results = await asyncio.gather(*tasks, return_exceptions=False)
        return {name: probe for name, probe in results}

    def probe_sources(self, source_names):
        if not source_names:
            return {}
        try:
            return asyncio.run(self._probe_async(source_names))
        except RuntimeError:
            # already inside event loop; fallback to sequential
            probe_data = {}
            for name in source_names:
                try:
                    probe_data[name] = self.sources[name].probe()
                except Exception as exc:
                    probe_data[name] = f"[probe-error] {exc}"
            return probe_data

    def execute(
        self,
        question: str,
        *,
        lang: str = "zh",
        callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    ):
        plan_data, raw_plan = self.plan_sources(question, lang, callbacks=callbacks)
        ordered_sources = plan_data.get("ordered_sources", [])
        probe_info = self.probe_sources(ordered_sources)
        self._notify(callbacks, "on_routing_plan", plan_data, raw_plan)
        self._notify(callbacks, "on_routing_probe", probe_info)
        outputs = []
        # Per-source query rewriting
        rewritten = self._rewrite_queries_for_sources(question, ordered_sources, lang)
        for name in ordered_sources:
            src = self.sources.get(name)
            if not src:
                continue
            try:
                if name == "sql_database":
                    # 将 CSV 阶段的输出作为上下文传给 SQL 重写，减少盲查
                    last_csv = None
                    for _out in reversed(outputs):
                        if _out.get("source") == "opcae_lookup":
                            last_csv = _out
                            break
                    # 结构化候选（JSON）提取：从 CSV Agent 的 data.rows / data.groups 中提取桥接列
                    csv_json = None
                    try:
                        if last_csv and isinstance(last_csv.get("raw"), dict):
                            csv_json = RoutingOrchestrator._extract_csv_candidates(last_csv.get("raw"))
                    except Exception:
                        csv_json = None
                    ctx = {
                        "csv_text": (last_csv.get("text") if last_csv else None),
                        "csv_candidates_json": csv_json,
                    }
                    intent = self._detect_intent(question, lang)
                    q = self._rewrite_for_sql(question, lang, intent, context=ctx)
                else:
                    q = rewritten.get(name, question)
                result = src.run(q, callbacks=callbacks)
                text = extract_agent_output(result)
            except Exception as exc:
                result = None
                text = f"[error] {exc}"
            outputs.append({"source": name, "text": text, "raw": result})
            self._notify(callbacks, "on_routing_step", name, text, raw=result)
        # Merge final answer with citations
        final_text = self._summarize_outputs(outputs, plan_data, lang, callbacks=callbacks)
        return {
            "plan": plan_data,
            "plan_raw": raw_plan,
            "probe": probe_info,
            "outputs": outputs,
            "final_text": final_text,
        }


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
