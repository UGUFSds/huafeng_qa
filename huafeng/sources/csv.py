import json
import os
from typing import Any, Dict, List, Optional, Sequence

from langchain_openai import ChatOpenAI
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal

try:
    import pandas as pd
except ImportError:
    pd = None

from huafeng.sources.base import DataSource
from huafeng.llm.factory import build_llm
from huafeng.config.settings import OPCAE_CSV_PATH


def _default_dataframe_description(df) -> str:
    try:
        rows, cols = df.shape
    except Exception:
        return "Dataframe"
    return f"{rows} rows × {cols} columns"


def load_opcae_dataframe(csv_path: str):
    if pd is None:
        print("[warn] pandas 未安装，无法加载 CSV 数据源。")
        return None
    if not os.path.isfile(csv_path):
        print(f"[warn] CSV 数据源文件未找到：{csv_path}")
        return None
    encodings = ("utf-8", "utf-8-sig", "gbk", "gb2312")
    last_error = None
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, encoding=enc)
            df.columns = [str(col).strip() for col in df.columns]
            df = df.fillna("")
            return df
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
    print(f"[warn] 无法解码 CSV 数据源：{csv_path}，原因：{last_error}")
    return None


class CsvDataSource(DataSource):
    """提供针对点位主数据 CSV 的查询能力。"""

    def __init__(
        self,
        agent,
        *,
        dataframe,
        source_path: str,
        name: str = "opcae_lookup",
        description: str = "Point master data in CSV",
    ):
        self._agent = agent
        self.dataframe = dataframe
        self.source_path = source_path
        self.name = name
        self.description = description

    def run(
        self,
        query: str,
        *,
        callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    ):
        if self._agent is None:
            return {"output": "[error] CSV LLM agent 未初始化或不可用", "data": [], "source": self.name}
        config = {}
        if callbacks:
            config["callbacks"] = list(callbacks)
        return self._agent.invoke({"input": query}, config=config)

    def probe(self) -> str:
        try:
            sample = self.dataframe.head(3)
            try:
                preview = sample.to_markdown(index=False)
            except Exception:
                preview = sample.to_dict(orient="records")
            return f"sample rows:\n{preview}"
        except Exception as exc:
            return f"[probe-error] {exc}"


class _FilterCond(BaseModel):
    column: str = Field(..., description="列名")
    op: Literal["equals", "contains", "startswith", "endswith"] = Field("equals", description="操作类型")
    value: str = Field(..., description="匹配值")


class _FindRowsInput(BaseModel):
    where: List[_FilterCond] = Field(default_factory=list, description="过滤条件列表")
    select: Optional[List[str]] = Field(default=None, description="返回列集合")
    limit: int = Field(20, description="最大返回行数")


class _HeadInput(BaseModel):
    limit: int = Field(5, description="最大返回行数")
    select: Optional[List[str]] = Field(default=None, description="返回列集合")


class _GroupCountInput(BaseModel):
    by: List[str] = Field(..., description="分组列")
    where: List[_FilterCond] = Field(default_factory=list, description="过滤条件列表")
    limit: int = Field(20, description="最大返回组数")


class SimpleCsvToolsAgent:
    """一个安全的 CSV 工具调用代理，只允许对内存 DataFrame 进行受限操作。"""

    def __init__(self, llm: ChatOpenAI, df):
        self.llm = llm
        self.df = df
        self.tools = self._build_tools()

    def _build_tools(self) -> List[StructuredTool]:
        df = self.df

        def _csv_columns() -> Dict[str, Any]:
            cols = [str(c) for c in list(df.columns)]
            return {"columns": cols}

        def _csv_head(limit: int = 5, select: Optional[List[str]] = None) -> Dict[str, Any]:
            d = df
            if select:
                cols = [c for c in select if c in d.columns]
                if cols:
                    d = d[cols]
            sample = d.head(max(1, int(limit)))
            return {"rows": sample.to_dict(orient="records"), "count": int(sample.shape[0])}

        def _apply_where(d0, where: List[Dict[str, Any]]):
            d = d0
            for cond in where or []:
                try:
                    col = str(cond.get("column"))
                    if col not in d.columns:
                        continue
                    val = str(cond.get("value", ""))
                    op = str(cond.get("op", "contains")).lower()
                    series = d[col].astype(str)
                    if op == "equals":
                        mask = series.str.strip() == val.strip()
                    elif op == "startswith":
                        mask = series.str.startswith(val, na=False)
                    elif op == "endswith":
                        mask = series.str.endswith(val, na=False)
                    else:
                        mask = series.str.contains(val, case=False, na=False)
                    d = d[mask]
                except Exception:
                    continue
            return d

        def _csv_find_rows(where: List[_FilterCond], select: Optional[List[str]] = None, limit: int = 20) -> Dict[str, Any]:
            d = df
            where_dicts = [c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else dict(c)) for c in (where or [])]
            d = _apply_where(d, where_dicts)
            if select:
                cols = [c for c in select if c in d.columns]
                if cols:
                    d = d[cols]
            out = d.head(max(1, int(limit)))
            return {"rows": out.to_dict(orient="records"), "count": int(out.shape[0])}

        def _csv_group_count(by: List[str], where: List[_FilterCond] = [], limit: int = 20) -> Dict[str, Any]:
            d = df
            where_dicts = [c.model_dump() if hasattr(c, "model_dump") else (c.dict() if hasattr(c, "dict") else dict(c)) for c in (where or [])]
            d = _apply_where(d, where_dicts)
            by_cols = [c for c in (by or []) if c in d.columns]
            if not by_cols:
                return {"groups": [], "error": "no valid group-by columns"}
            try:
                g = d.groupby(by_cols).size().reset_index(name="count").sort_values("count", ascending=False).head(max(1, int(limit)))
                return {"groups": g.to_dict(orient="records"), "count": int(g.shape[0])}
            except Exception as exc:
                return {"groups": [], "error": str(exc)}

        return [
            StructuredTool.from_function(_csv_columns, name="csv_columns", description="列出 CSV 列名"),
            StructuredTool.from_function(_csv_head, args_schema=_HeadInput, name="csv_head", description="返回前N行，可选择返回列"),
            StructuredTool.from_function(_csv_find_rows, args_schema=_FindRowsInput, name="csv_find_rows", description="按条件过滤并返回行"),
            StructuredTool.from_function(_csv_group_count, args_schema=_GroupCountInput, name="csv_group_count", description="按列分组并统计数量"),
        ]

    def invoke(self, payload: Any, config: Optional[Dict[str, Any]] = None):
        query = payload.get("input") if isinstance(payload, dict) else str(payload)
        system = (
            "You are a CSV tool-calling assistant for the in-memory DataFrame 'opcae_lookup'. "
            "Use the provided tools to retrieve structured candidates (bridge keys like point_id/tag/name/desc). "
            "Do not access any local files or run arbitrary code. Respond concisely in the user's language."
        )
        human = query
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{q}")])
        messages = prompt.format_messages(q=human)
        bound_llm = self.llm.bind_tools(self.tools)
        resp = bound_llm.invoke(messages, config=config) if config else bound_llm.invoke(messages)
        tool_calls = getattr(resp, "tool_calls", None) or getattr(resp, "additional_kwargs", {}).get("tool_calls")
        data_out = None
        if tool_calls:
            for call in tool_calls:
                name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else None)
                args = getattr(call, "args", None) or (call.get("args") if isinstance(call, dict) else {})
                call_id = getattr(call, "id", None) or (call.get("id") if isinstance(call, dict) else name)
                tool = next((t for t in self.tools if t.name == name), None)
                if tool:
                    try:
                        data_out = tool.invoke(args)
                    except Exception as exc:
                        data_out = {"error": str(exc)}
                    tm = ToolMessage(content=json.dumps(data_out, ensure_ascii=False), tool_call_id=str(call_id))
                    follow_messages = list(messages) + [AIMessage(content=getattr(resp, "content", ""), tool_calls=tool_calls), tm]
                    final = self.llm.invoke(follow_messages, config=config) if config else self.llm.invoke(follow_messages)
                    return {"output": getattr(final, "content", str(final)), "data": data_out}
        return {"output": getattr(resp, "content", str(resp)), "data": data_out}


def build_csv_source(base_url: str, max_iterations: int = 10):
    csv_path = OPCAE_CSV_PATH
    df = load_opcae_dataframe(csv_path)
    if df is None:
        return None
    llm = build_llm(base_url)
    agent = SimpleCsvToolsAgent(llm, df)
    desc = f"Point master data CSV ({_default_dataframe_description(df)})"
    return CsvDataSource(agent=agent, dataframe=df, source_path=csv_path, description=desc)