import json
from typing import Any, Dict, Optional

from langchain_openai import ChatOpenAI

from huafeng.config.settings import (
    API_KEY,
    MODEL,
    ENABLE_AUTO_TOOL_CHOICE,
    TOOL_CALL_PARSER,
    EXTRA_HEADERS_JSON,
)


def build_llm(base_url: str) -> ChatOpenAI:
    """Construct ChatOpenAI with provider-compatible options and optional headers."""
    model_kwargs: Dict[str, Any] = {}
    try:
        if ENABLE_AUTO_TOOL_CHOICE:
            val = ENABLE_AUTO_TOOL_CHOICE.lower() in {"1", "true", "yes", "on"}
            model_kwargs["enable_auto_tool_choice"] = val
        if TOOL_CALL_PARSER:
            model_kwargs["tool_call_parser"] = TOOL_CALL_PARSER
    except Exception:
        pass
    headers: Optional[Dict[str, str]] = None
    if EXTRA_HEADERS_JSON:
        try:
            headers = json.loads(EXTRA_HEADERS_JSON)
        except Exception:
            headers = None
    return ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=base_url,
        temperature=0.0,
        model_kwargs=model_kwargs,
        default_headers=headers,
    )