from typing import Any, Dict, Optional, Sequence

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

from app.sources.base import DataSource
from app.llm.factory import build_llm
from app.config.settings import DB_URI, PG_DB, PG_HOST, PG_PORT, SQL_AGENT_MAX_ITERATIONS, SQL_TABLE_INFO_SAMPLE_ROWS, SQL_TABLE_INFO_CACHE_SECONDS


class SqlDataSource(DataSource):
    """将 LangChain SQL Agent 包装为 DataSource 接口。"""

    def __init__(
        self,
        agent,
        db: SQLDatabase,
        name: str = "sql_database",
        description: str = "Industrial PostgreSQL database",
    ):
        self._agent = agent
        self._db = db
        self.name = name
        self.description = description

    def run(
        self,
        query: str,
        *,
        callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    ):
        config = {}
        if callbacks:
            config["callbacks"] = list(callbacks)
        return self._agent.invoke({"input": query}, config=config)

    def probe(self) -> str:
        try:
            tables = sorted(list(self._db.get_usable_table_names()))
            preview = ", ".join(tables[:5]) if tables else "no tables detected"
            more = "" if len(tables) <= 5 else f" (+{len(tables) - 5} more)"
            return f"tables preview: {preview}{more}"
        except Exception as exc:
            return f"[probe-error] {exc}"


def build_sql_source(base_url: str, max_iterations: int = SQL_AGENT_MAX_ITERATIONS) -> SqlDataSource:
    """构建 SQL 数据源，屏蔽不应访问的表并返回包装后的 DataSource。"""
    # 降低 get_table_info 的采样行数，加速 Agent 生成与执行
    db = SQLDatabase.from_uri(
        DB_URI,
        ignore_tables=["point_data"],
        sample_rows_in_table_info=SQL_TABLE_INFO_SAMPLE_ROWS,
    )
    # 轻量缓存 table_info，减少重复自省开销
    try:
        import time
        _orig_get_table_info = db.get_table_info
        _ti_cache: Dict[str, tuple] = {}

        def _cached_get_table_info(*args, **kwargs):
            # 以表名列表、采样数作为键；缺省则用 "ALL" 作为键
            table_names = None
            if args:
                table_names = args[0]
            else:
                table_names = kwargs.get("table_names")
            sample_rows = getattr(db, "sample_rows_in_table_info", None)
            key = (
                "ALL" if table_names in (None, [], tuple()) else tuple(sorted(list(table_names)))
            )
            ckey = f"{key}|{sample_rows}"
            entry = _ti_cache.get(ckey)
            now = time.time()
            if entry and (now - entry[0]) <= SQL_TABLE_INFO_CACHE_SECONDS:
                return entry[1]
            val = _orig_get_table_info(*args, **kwargs)
            _ti_cache[ckey] = (now, val)
            return val

        db.get_table_info = _cached_get_table_info  # type: ignore
    except Exception:
        pass
    llm = build_llm(base_url)
    agent = create_sql_agent(
        llm,
        db=db,
        agent_type="tool-calling",
        verbose=False,
        max_iterations=max_iterations,
    )
    desc = f"PostgreSQL {PG_DB} @ {PG_HOST}:{PG_PORT}"
    return SqlDataSource(agent, db=db, description=desc)