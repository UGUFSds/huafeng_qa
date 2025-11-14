from langchain_core.callbacks.base import BaseCallbackHandler


class ChineseConsoleCallback(BaseCallbackHandler):
    def __init__(self, lang: str = "zh"):
        self.lang = lang
        self._top_run_id = None
        self._selected_label = None

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
                    if self._selected_label:
                        out_lines.append(f"【计划说明】[所选 {self._selected_label}] {msg}")
                    else:
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
        if name == "invalid_tool":
            try:
                import re
                m = re.search(r"requested_tool_name':\s*'([^']+)" , inp)
                req = m.group(1) if m else "(未知)"
                print(f"【路由提示】跨源请求: {req} -> 已转接至相应数据源")
                return
            except Exception:
                pass
        if name == "sql_db_query":
            try:
                import re
                m = re.search(r'FROM\s+"([^"]+)"', inp)
                t = m.group(1) if m else "(未知表)"
                m1 = re.search(r"ts\s*>?=\s*'([^']+)'", inp)
                m2 = re.search(r"ts\s*<\s*'([^']+)'", inp)
                win = f"{m1.group(1)} ~ {m2.group(1)}" if (m1 and m2) else "(未指定时间窗)"
                print(f"【调用工具】{name} 入参：{inp}")
                print(f"【SQL目标】表：{t}；时间窗：{win}")
                return
            except Exception:
                pass
        print(f"【调用工具】{name} 入参：{inp}")

    def on_tool_end(self, output, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        out = str(output)
        name = kwargs.get("name") if isinstance(kwargs, dict) else None
        if name == "invalid_tool":
            return
        if len(out) > 400:
            out = out[:400] + " …（截断）"
        print(f"【工具返回】{out}")

    def on_chain_end(self, outputs, run_id=None, parent_run_id=None, **kwargs):
        if self.lang != "zh":
            return
        if parent_run_id is None or (self._top_run_id is not None and run_id == self._top_run_id):
            print("> 链已完成。")

    # Custom hooks used by RoutingOrchestrator
    def on_routing_plan(self, plan_data, raw_plan, **kwargs):
        if self.lang != "zh":
            return
        ordered = plan_data.get("ordered_sources", [])
        strategy = plan_data.get("strategy")
        if ordered:
            print(f"【路由计划】顺序：{' -> '.join(ordered)}")
        if strategy:
            snippet = strategy if len(strategy) <= 300 else strategy[:300] + "…"
            print(f"【路由计划】策略：{snippet}")

    def on_routing_probe(self, probe_info, **kwargs):
        if self.lang != "zh" or not probe_info:
            return
        for name, summary in probe_info.items():
            text = summary if len(summary) <= 300 else summary[:300] + "…"
            print(f"【路由探测】{name} -> {text}")

    def on_routing_step(self, source_name, output_text, **kwargs):
        if self.lang != "zh":
            return
        text = output_text if output_text else "(无输出)"
        if len(text) > 400:
            text = text[:400] + "…"
        print(f"【路由结果】{source_name} => {text}")

    def on_selected_candidates(self, candidates, **kwargs):
        if self.lang != "zh":
            return
        try:
            labels = []
            for c in candidates or []:
                lab = c.get("table_name") or c.get("code") or c.get("point_name")
                if lab:
                    labels.append(str(lab))
            if labels:
                self._selected_label = ", ".join(labels)
                print(f"【澄清选择】{self._selected_label}")
        except Exception:
            pass
