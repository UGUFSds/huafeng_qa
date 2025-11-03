from typing import Optional, Sequence

from langchain_core.callbacks.base import BaseCallbackHandler


class DataSource:
    """Minimal interface so不同来源可以统一被主路由管理。"""

    name: str
    description: str

    def run(
        self,
        query: str,
        *,
        callbacks: Optional[Sequence[BaseCallbackHandler]] = None,
    ):
        raise NotImplementedError

    def short_info(self) -> str:
        return f"{self.name}: {self.description}"

    def probe(self) -> str:
        return "probe not implemented"