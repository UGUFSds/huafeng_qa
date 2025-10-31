import json
import os
import sys
from typing import Optional, Sequence

from langchain_core.callbacks.base import BaseCallbackHandler

# Ensure project root is on sys.path when running as a script from scripts/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from huafeng.router import RoutingOrchestrator, register_data_sources, AVAILABLE_SOURCES
from huafeng.sources.base import DataSource


class DummyLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages, config=None):
        class R:
            content = "summary"
            tool_calls = []

        return R()


class DummySource(DataSource):
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    def run(self, query: str, *, callbacks: Optional[Sequence[BaseCallbackHandler]] = None):
        return {"output": f"[{self.name}] echoed", "query": query}

    def probe(self) -> str:
        return "ok"


def main():
    register_data_sources(
        [
            DummySource("opcae_lookup", "Point master data CSV (dummy)"),
            DummySource("sql_database", "Industrial SQL DB (dummy)"),
        ]
    )
    orchestrator = RoutingOrchestrator(DummyLLM(), AVAILABLE_SOURCES)
    data = orchestrator.execute("测试：查询某个点位的描述和近期报警次数", lang="zh")
    print(
        json.dumps(
            {
                "plan": data.get("plan"),
                "probe": data.get("probe"),
                "outputs": data.get("outputs"),
                "final_text": data.get("final_text"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()