import asyncio
import json
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, List

from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from app.prompts import (
    build_routing_planner_prompt,
    build_evidence_summarizer_prompt,
)

from app.config.settings import API_KEY, USE_LLM_PLANNER, SUMMARIZE_SINGLE_SOURCE, PROBE_CACHE_SECONDS, USE_LLM_SUMMARIZER, ROUTER_ENABLE_PROBE, PLAN_CACHE_SECONDS, REWRITE_CACHE_SECONDS, ROUTER_PARALLEL_EXECUTE, RESULT_CACHE_SECONDS, CLARIFY_ENABLED, CLARIFY_CANDIDATE_THRESHOLD, CLARIFY_MAX_OPTIONS, SQL_YEAR_COLUMN_NAME, ROUTER_STRICT_AFTER_CLARIFICATION
from app.sources.base import DataSource


# 统一维护可用数据源注册（供提示与路由使用）
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
        csv_src = AVAILABLE_SOURCES.get("csv_lookup")
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
        f"- CSV(csv_lookup) 列: {', '.join(csv_cols) if csv_cols else '未知'}\n"
        f"- SQL 可用表: {', '.join(db_tables[:8]) + (' 等' if len(db_tables) > 8 else '') if db_tables else '未知'}\n"
        "- 跨源桥接采用“相关列集合”，优先候选：point_name、code、table_name；其次：point_id、tag/tag_name、name/desc、device_id、line_id。"
    )
    en = (
        "[Schema hints]\n"
        f"- CSV(csv_lookup) columns: {', '.join(csv_cols) if csv_cols else 'unknown'}\n"
        f"- SQL usable tables: {', '.join(db_tables[:8]) + (' etc.' if len(db_tables) > 8 else '') if db_tables else 'unknown'}\n"
        "- Cross-source bridging uses relevant columns; prefer: point_name, code, table_name; then: point_id, tag/tag_name, name/desc, device_id, line_id."
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
    policy = (
        "【数据源使用政策】\n"
        "- 点位主数据通过 csv_lookup（CSV）获取。\n"
        "- 报警相关数据使用 alarm_event；历史曲线/时间段数据使用按点位拆分的历史表。\n"
        "- 禁止无差别枚举所有表；仅在明确给出检索键与时间窗口时访问历史表，并限制返回行数。\n"
        "- 如未提及年份，默认查询今年的数据。\n"
        "- 执行前尽量进行语句校验；结果应简洁并附来源。"
    )
    steps = (
        "【任务流程提醒】\n"
        "1. 基于可用数据源制定计划，至少选择一个来源。\n"
        "2. 点位主数据用 csv_lookup（CSV）；历史/报警数据用 sql_database。\n"
        "3. 多源场景按计划执行，并在最终回答中整合来源信息。"
    )
    return (
        f"请用简体中文回答：\n{policy}\n{steps}\n【可用数据源】\n"
        f"{source_notes['zh']}\n{schema_notes['zh']}\n现在的问题：{text}"
    )


class RoutingOrchestrator:
    """Top-level controller that routes queries across multiple data sources."""

    def __init__(self, planner_llm: ChatOpenAI, sources: Dict[str, DataSource]):
        self.planner_llm = planner_llm
        self.sources = dict(sources)
        # short-lived cache to avoid repeated probe introspection
        self._probe_cache: Dict[str, Tuple[float, Dict[str, str]]] = {}
        # short-lived plan cache
        self._plan_cache: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
        # short-lived rewrite cache
        self._rewrite_cache: Dict[str, Tuple[float, str]] = {}
        # short-lived per-source result cache (keyed by lang|source|rewritten_query)
        self._result_cache: Dict[str, Tuple[float, Any, str]] = {}

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
            pref = ["csv_lookup", "sql_database"]
            ordered = [n for n in pref if n in cands]
            for n in cands:
                if n not in ordered:
                    ordered.append(n)
            return ordered
        if intent["is_point_master"] and "csv_lookup" in names:
            cands = ["csv_lookup", "sql_database"] if "sql_database" in names else ["csv_lookup"]
            reason = "point master data -> CSV first, SQL optional"
            return order_by_pref(cands), reason
        if (intent["is_history"] or intent["is_alarm"]) and "sql_database" in names:
            cands = ["sql_database", "csv_lookup"] if "csv_lookup" in names else ["sql_database"]
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
            csv_src = AVAILABLE_SOURCES.get("csv_lookup")
            if csv_src and hasattr(csv_src, "dataframe") and csv_src.dataframe is not None:
                cols = [str(c).lower() for c in list(csv_src.dataframe.columns)]
                keys.extend(
                    [
                        c
                        for c in cols
                        if c in {
                            "point_name",
                            "code",
                            "table_name",
                            "point_id",
                            "tag",
                            "tag_name",
                            "name",
                            "desc",
                            "device_id",
                            "line_id",
                        }
                    ]
                )
        except Exception:
            pass
        # 优先顺序：point_name、code、table_name；其次 point_id 与其他常见桥接键
        for k in ["point_name", "code", "table_name", "point_id", "tag", "tag_name", "name", "desc", "device_id", "line_id"]:
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

    def _format_candidate_label(self, c: Dict[str, str]) -> str:
        # 统一构造选项标签：优先 point_name(code) | table_name | desc
        name = c.get("point_name") or c.get("name") or ""
        code = c.get("code") or ""
        tbl = c.get("table_name") or ""
        desc = c.get("desc") or c.get("tag_name") or ""
        parts = []
        if name or code:
            if name and code:
                parts.append(f"{name}({code})")
            elif name:
                parts.append(name)
            else:
                parts.append(code)
        if tbl:
            parts.append(tbl)
        if desc:
            parts.append(desc)
        return " | ".join([p for p in parts if p])

    def _build_clarification_payload(self, candidates: List[Dict[str, str]], question: str, max_options: int) -> Dict[str, Any]:
        # 限制选项数量，生成友好中文提示和离线选项列表
        sliced = candidates[:max_options]
        options = []
        for idx, c in enumerate(sliced):
            options.append({
                "index": idx,
                "label": self._format_candidate_label(c) or f"候选 {idx+1}",
                "data": c,
            })
        msg = (
            "检测到多个可能的点位/对象。为避免盲目枚举查询，请先从下列候选中选择，以便精确检索：\n"
            + "\n".join([f"{o['index']}. {o['label']}" for o in options])
            + "\n\n请返回所选索引列表，如：clarify_choice={'indices':[0]}。"
        )
        return {"message": msg, "options": options, "question": question}

    @staticmethod
    def _rewrite_for_csv(question: str, lang: str, intent: Dict[str, Any]) -> str:
        keys = RoutingOrchestrator._candidate_bridge_keys()
        keys_text = ", ".join(keys)
        return (
            "请使用点位主数据 CSV(csv_lookup) 检索与问题相关的信息，并返回结构化候选。"
            "优先包含字段：point_name、code、desc(描述)、unit(单位)、type(类型)、threshold(阈值)，以及可能的桥接列集合："
            f"{keys_text}。"
            "如无法确定精确点位，请基于名称/描述进行合理匹配，但避免仅返回自由文本；"
            "尽量给出候选对象（含可作为后续 SQL 过滤的列值）。"
            "注意：禁止访问其他本地文件，仅使用已加载的 DataFrame。"
            "如问题涉及时间但未提及年份，默认查询今年的数据。"
            "原始问题：" + question
        )

    @staticmethod
    def _rewrite_for_sql(question: str, lang: str, intent: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> str:
        keys = RoutingOrchestrator._candidate_bridge_keys()
        keys_text = ", ".join(keys)
        import datetime
        now_year = datetime.datetime.now().astimezone().year
        csv_hint = ""
        if context and isinstance(context.get("csv_text"), str):
            t = context["csv_text"].strip()
            csv_hint = t[:500] + ("…" if len(t) > 500 else "")
        json_hint = ""
        if context and isinstance(context.get("csv_candidates_json"), str) and context.get("csv_candidates_json"):
            j = context["csv_candidates_json"].strip()
            json_hint = j[:800] + ("…" if len(j) > 800 else "")
        strict_tables = []
        try:
            if context and isinstance(context.get("strict_tables"), list):
                strict_tables = [str(x) for x in context.get("strict_tables") if x]
        except Exception:
            strict_tables = []
        base = (
            "请使用工业数据库(sql_database)回答。点位主数据由 CSV(csv_lookup) 提供。"
            "如需报警或历史数据：使用 alarm_event（报警总表）与按点位拆分的历史表。"
            "访问历史表应基于明确的检索键与时间窗口，限制返回条数。"
            "若 point_data.table_name 可用，优先据此定位目标历史表；过滤时优先使用 point_name 与 code。"
            f"如未提及年份，默认查询今年的数据（当前年份：{now_year}）。"
            f"在 SQL 中显式加入年份过滤，例如 EXTRACT(YEAR FROM {SQL_YEAR_COLUMN_NAME or 'ts'}) = {now_year}。"
        )
        if csv_hint and json_hint:
            bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 CSV 候选：{csv_hint} 与 JSON 候选：{json_hint}。"
        elif csv_hint:
            bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 CSV 候选：{csv_hint}。"
        elif json_hint:
            bridge = f"优先使用桥接列进行过滤（例如 {keys_text}），并参考 JSON 候选：{json_hint}。"
        else:
            bridge = f"优先使用桥接列进行过滤（例如 {keys_text}）。"
        strict_clause = ""
        if strict_tables:
            joined = ", ".join([f'"{t}"' for t in strict_tables])
            strict_clause = f" 仅限查询以下目标表：{joined}；不要扩展至相近或未列出的表。"
        return base + " " + bridge + strict_clause + " 原始问题：" + question

    def _rewrite_queries_for_sources(self, question: str, ordered_sources: List[str], lang: str) -> Dict[str, str]:
        intent = self._detect_intent(question, lang)
        rewritten = {}
        import time
        now = time.time()
        for name in ordered_sources:
            # 读取缓存
            ck = f"{lang}|{name}|{question}"
            entry = self._rewrite_cache.get(ck)
            if entry and (now - entry[0]) <= REWRITE_CACHE_SECONDS:
                rewritten[name] = entry[1]
                continue
            # 生成并写入缓存（SQL 的上下文重写在 execute 阶段另行处理）
            if name == "csv_lookup":
                q = self._rewrite_for_csv(question, lang, intent)
            elif name == "sql_database":
                q = self._rewrite_for_sql(question, lang, intent)
            else:
                q = question
            rewritten[name] = q
            self._rewrite_cache[ck] = (now, q)
        return rewritten

    # --- Summarize outputs with citations ---
    def _summarize_outputs(self, outputs: List[Dict[str, Any]], plan_data: Dict[str, Any], lang: str, callbacks: Optional[Sequence[BaseCallbackHandler]] = None) -> str:
        if not outputs:
            return ""
        # Avoid extra LLM call when only one source contributed, unless explicitly enabled
        if len(outputs) == 1 and not SUMMARIZE_SINGLE_SOURCE:
            src = outputs[0].get("source")
            text = outputs[0].get("text") or extract_agent_output(outputs[0].get("raw")) or ""
            return f"{text}\n\nSources: {src}"
        # 全局关闭汇总 LLM：朴素合并
        if not USE_LLM_SUMMARIZER:
            merged = []
            for item in outputs:
                merged.append(f"[{item.get('source')}] {item.get('text') or ''}")
            cites = ", ".join(s.get("source") for s in outputs)
            return ("\n".join(merged)) + f"\n\nSources: {cites}"
        try:
            cite_lines = []
            for item in outputs:
                src = item.get("source")
                text = extract_agent_output(item.get("raw")) or item.get("text") or ""
                snippet = text if len(text) <= 400 else text[:400] + "…"
                cite_lines.append(f"Source[{src}]: {snippet}")
            strategy = plan_data.get("strategy", "")
            ordered = " -> ".join(plan_data.get("ordered_sources", []))
            evidence = "\n".join(cite_lines)
            prompt = build_evidence_summarizer_prompt()
            import datetime
            now = datetime.datetime.now().astimezone().isoformat()
            messages = prompt.format_messages(lang=lang, strategy=strategy, ordered=ordered, evidence=evidence, now=now)
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
        use_llm = bool(API_KEY) and USE_LLM_PLANNER and len(candidate_sources) > 1
        # 规划短期缓存：按语言+问题+候选集合键
        import time
        cache_key = f"{lang}|{question}|{','.join(candidate_sources)}"
        now_ts = time.time()
        cached = self._plan_cache.get(cache_key)
        if cached and (now_ts - cached[0]) <= PLAN_CACHE_SECONDS:
            return dict(cached[1]), cached[2]
        available = self._format_sources_for_prompt()
        # Reduce available to candidates for prompting
        available_lines = []
        for name in candidate_sources:
            src = self.sources.get(name)
            if src:
                available_lines.append(f"- {src.short_info()}")
        available_text = "\n".join(available_lines) if available_lines else available
        example_json = '{"ordered_sources": ["csv_lookup", "sql_database"], "strategy": "先查CSV点位，再查数据库历史"}'
        prompt = build_routing_planner_prompt()
        if use_llm:
            import datetime
            now = datetime.datetime.now().astimezone().isoformat()
            messages = prompt.format_messages(lang=lang, question=question, available=available_text, example_json=example_json, now=now)
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
        # 写入缓存
        self._plan_cache[cache_key] = (now_ts, dict(plan_data), raw_content)
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
        # 全局跳过探测以加速（可配置）
        if not ROUTER_ENABLE_PROBE:
            return {name: "[probe-skipped]" for name in source_names}
        # Cache probe results briefly to avoid repeated DB introspection
        key = "|".join(source_names)
        import time
        now = time.time()
        cached = self._probe_cache.get(key)
        if cached and (now - cached[0]) <= PROBE_CACHE_SECONDS:
            return dict(cached[1])
        try:
            result = asyncio.run(self._probe_async(source_names))
        except RuntimeError:
            # already inside event loop; fallback to sequential
            result = {}
            for name in source_names:
                try:
                    result[name] = self.sources[name].probe()
                except Exception as exc:
                    result[name] = f"[probe-error] {exc}"
        # store cache
        self._probe_cache[key] = (now, dict(result))
        return result

    def execute(
        self,
        question: str,
        *,
        lang: str = "zh",
        callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
        clarify_choice: Optional[Dict[str, Any]] = None,
    ):
        plan_data, raw_plan = self.plan_sources(question, lang, callbacks=callbacks)
        ordered_sources = plan_data.get("ordered_sources", [])
        probe_info = self.probe_sources(ordered_sources)
        self._notify(callbacks, "on_routing_plan", plan_data, raw_plan)
        self._notify(callbacks, "on_routing_probe", probe_info)
        outputs = []
        csv_backfill_done = False
        # Per-source query rewriting
        rewritten = self._rewrite_queries_for_sources(question, ordered_sources, lang)
        # 并发执行：当不存在 CSV→SQL 的顺序依赖时尝试并行
        def _has_csv_sql_dependency(names: List[str]) -> bool:
            try:
                i_csv = names.index("csv_lookup")
                i_sql = names.index("sql_database")
                return i_csv < i_sql
            except ValueError:
                return False

        if ROUTER_PARALLEL_EXECUTE and len(ordered_sources) > 1 and not _has_csv_sql_dependency(ordered_sources):
            async def _run_single(name: str):
                src = self.sources.get(name)
                if not src:
                    return {"source": name, "text": "[error] source missing", "raw": None}
                try:
                    # 对 SQL 无上下文的情形使用预重写
                    q = rewritten.get(name, question)
                    # 结果缓存命中
                    import time
                    start_perf = time.perf_counter()
                    now_ts = time.time()
                    ck = f"{lang}|{name}|{q}"
                    cached = self._result_cache.get(ck)
                    if cached and (now_ts - cached[0]) <= RESULT_CACHE_SECONDS:
                        result, text = cached[1], cached[2]
                    else:
                        result = await asyncio.to_thread(src.run, q, callbacks=callbacks)
                        text = extract_agent_output(result)
                        self._result_cache[ck] = (now_ts, result, text)
                except Exception as exc:
                    result = None
                    text = f"[error] {exc}"
                end_perf = time.perf_counter()
                meta = {"duration_sec": float(max(0.0, end_perf - start_perf))}
                try:
                    if isinstance(result, dict):
                        d = result.get("data")
                        if isinstance(d, dict):
                            c = d.get("count")
                            if isinstance(c, int):
                                meta["rows_count"] = c
                            else:
                                rows = d.get("rows") or []
                                groups = d.get("groups") or []
                                if isinstance(rows, list):
                                    meta["rows_count"] = len(rows)
                                elif isinstance(groups, list):
                                    meta["rows_count"] = len(groups)
                except Exception:
                    pass
                self._notify(callbacks, "on_routing_step", name, text, raw=result)
                return {"source": name, "text": text, "raw": result, "meta": meta}

            async def _run_all():
                tasks = [asyncio.create_task(_run_single(n)) for n in ordered_sources]
                res = await asyncio.gather(*tasks, return_exceptions=False)
                return res

            try:
                outputs = asyncio.run(_run_all())
            except RuntimeError:
                from concurrent.futures import ThreadPoolExecutor, as_completed
                futures = []
                with ThreadPoolExecutor(max_workers=len(ordered_sources)) as ex:
                    for name in ordered_sources:
                        src = self.sources.get(name)
                        if not src:
                            continue
                        q = rewritten.get(name, question)
                        futures.append(ex.submit(lambda n, s, qq: (n, s.run(qq, callbacks=callbacks), qq), name, src, q))
                    for fut in as_completed(futures):
                        try:
                            name, result, qq = fut.result()
                            import time
                            text = extract_agent_output(result)
                            meta = {}
                            try:
                                if isinstance(result, dict):
                                    d = result.get("data")
                                    if isinstance(d, dict):
                                        c = d.get("count")
                                        if isinstance(c, int):
                                            meta["rows_count"] = c
                                        else:
                                            rows = d.get("rows") or []
                                            groups = d.get("groups") or []
                                            if isinstance(rows, list):
                                                meta["rows_count"] = len(rows)
                                            elif isinstance(groups, list):
                                                meta["rows_count"] = len(groups)
                            except Exception:
                                pass
                            outputs.append({"source": name, "text": text, "raw": result, "meta": meta})
                            self._notify(callbacks, "on_routing_step", name, text, raw=result)
                        except Exception as exc:
                            self._notify(callbacks, "on_routing_step", name, f"[error] {exc}")
        else:
            # 原有串行逻辑（保留 CSV→SQL 上下文传递）
            for name in ordered_sources:
                src = self.sources.get(name)
                if not src:
                    continue
                try:
                    if name == "sql_database":
                        # 将 CSV 阶段的输出作为上下文传给 SQL 重写，减少盲查
                        last_csv = None
                        for _out in reversed(outputs):
                            if _out.get("source") == "csv_lookup":
                                last_csv = _out
                                break
                        # 结构化候选（JSON）提取：从 CSV Agent 的 data.rows / data.groups 中提取桥接列
                        csv_json = None
                        try:
                            if last_csv and isinstance(last_csv.get("raw"), dict):
                                csv_json = RoutingOrchestrator._extract_csv_candidates(last_csv.get("raw"))
                        except Exception:
                            csv_json = None

                        # —— 澄清分支：候选过多时先返回选项，避免枚举查询 ——
                        csv_candidates_list: List[Dict[str, str]] = []
                        try:
                            if csv_json:
                                csv_candidates_list = json.loads(csv_json)
                        except Exception:
                            csv_candidates_list = []

                        # 若候选过多且未提供用户选择，则返回澄清载荷并终止后续 SQL 查询
                        if CLARIFY_ENABLED and len(csv_candidates_list) >= CLARIFY_CANDIDATE_THRESHOLD and not clarify_choice:
                            payload = self._build_clarification_payload(csv_candidates_list, question, CLARIFY_MAX_OPTIONS)
                            final_text = payload["message"]
                            return {
                                "plan": plan_data,
                                "plan_raw": raw_plan,
                                "probe": probe_info,
                                "outputs": outputs,  # 已包含 CSV 阶段结果
                                "final_text": final_text,
                                "needs_clarification": True,
                                "clarification": payload,
                            }

                        # 若提供了选择，则按选择缩小候选范围后再进行 SQL 重写
                        if clarify_choice and csv_candidates_list:
                            indices = clarify_choice.get("indices") or []
                            if isinstance(indices, list) and indices:
                                filtered = [csv_candidates_list[i] for i in indices if 0 <= i < len(csv_candidates_list)]
                                if filtered:
                                    try:
                                        csv_json = json.dumps(filtered, ensure_ascii=False)
                                    except Exception:
                                        pass
                                    self._notify(callbacks, "on_selected_candidates", filtered)
                                if ROUTER_STRICT_AFTER_CLARIFICATION:
                                    strict_tables = []
                                    try:
                                        for c in filtered:
                                            tname = c.get("table_name") or ""
                                            if tname:
                                                strict_tables.append(tname)
                                    except Exception:
                                        strict_tables = []
                        ctx = {
                            "csv_text": (last_csv.get("text") if last_csv else None),
                            "csv_candidates_json": csv_json,
                        }
                        if ROUTER_STRICT_AFTER_CLARIFICATION:
                            try:
                                ctx["strict_tables"] = strict_tables if 'strict_tables' in locals() else []
                            except Exception:
                                ctx["strict_tables"] = []
                        intent = self._detect_intent(question, lang)
                        # SQL 上下文重写缓存
                        import time
                        now = time.time()
                        # 将上下文摘要化以生成稳定的键
                        t_sig = (ctx.get("csv_text") or "" )
                        t_sig = t_sig[:200]
                        j_sig = (ctx.get("csv_candidates_json") or "" )
                        j_sig = j_sig[:200]
                        ck = f"{lang}|sql_database|{question}|t:{t_sig}|j:{j_sig}"
                        entry = self._rewrite_cache.get(ck)
                        if entry and (now - entry[0]) <= REWRITE_CACHE_SECONDS:
                            q = entry[1]
                        else:
                            q = self._rewrite_for_sql(question, lang, intent, context=ctx)
                            self._rewrite_cache[ck] = (now, q)
                    else:
                        # 非 SQL 使用预重写；SQL 在此考虑 CSV 上下文并带缓存
                        q = rewritten.get(name, question)
                    # 结果缓存命中
                    import time
                    now_ts = time.time()
                    rck = f"{lang}|{name}|{q}"
                    rcached = self._result_cache.get(rck)
                    if rcached and (now_ts - rcached[0]) <= RESULT_CACHE_SECONDS:
                        result, text = rcached[1], rcached[2]
                    else:
                        try:
                            from app.sources.sql import set_allowed_tables
                            set_allowed_tables(ctx.get("strict_tables") or [])
                        except Exception:
                            pass
                        result = src.run(q, callbacks=callbacks)
                        text = extract_agent_output(result)
                        # 动态跨源：若在 SQL 阶段出现对 csv_lookup 的无效调用意图，则主动执行 CSV，再基于候选重写并重跑一次 SQL
                        try:
                            if (not csv_backfill_done) and name == "sql_database" and (("invalid_tool" in text and "csv_lookup" in text) or ("csv_lookup is not a valid tool" in text)):
                                csv_src = self.sources.get("csv_lookup")
                                if csv_src:
                                    csv_res = csv_src.run(question, callbacks=callbacks)
                                    csv_text2 = extract_agent_output(csv_res)
                                    outputs.append({"source": "csv_lookup", "text": csv_text2, "raw": csv_res})
                                    self._notify(callbacks, "on_routing_step", "csv_lookup", csv_text2, raw=csv_res)
                                    # 提取候选并重写 SQL
                                    csv_json2 = None
                                    try:
                                        if isinstance(csv_res, dict):
                                            csv_json2 = RoutingOrchestrator._extract_csv_candidates(csv_res)
                                    except Exception:
                                        csv_json2 = None
                                    ctx["csv_text"] = csv_text2
                                    ctx["csv_candidates_json"] = csv_json2
                                    q = self._rewrite_for_sql(question, lang, intent, context=ctx)
                                    result = src.run(q, callbacks=callbacks)
                                    text = extract_agent_output(result)
                                    csv_backfill_done = True
                        except Exception:
                            pass
                        self._result_cache[rck] = (now_ts, result, text)
                except Exception as exc:
                    result = None
                    text = f"[error] {exc}"
                outputs.append({"source": name, "text": text, "raw": result})
                self._notify(callbacks, "on_routing_step", name, text, raw=result)
        # Merge final answer with citations
        final_text = self._summarize_outputs(outputs, plan_data, lang, callbacks=callbacks)
        result_payload = {
            "plan": plan_data,
            "plan_raw": raw_plan,
            "probe": probe_info,
            "outputs": outputs,
            "final_text": final_text,
        }
        try:
            if clarify_choice and ROUTER_STRICT_AFTER_CLARIFICATION:
                result_payload["selected_candidates"] = json.loads(csv_json) if csv_json else []
        except Exception:
            result_payload["selected_candidates"] = []
        return result_payload
