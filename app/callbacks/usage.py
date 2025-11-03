import time
from langchain_core.callbacks.base import BaseCallbackHandler


class TokenUsageHandler(BaseCallbackHandler):
    """Collects token usage and LLM-only runtime per single run."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0
        self.llm_runtime_sec = 0.0
        self._llm_timer_stack = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.llm_calls += 1
        self._llm_timer_stack.append(time.perf_counter())

    def on_llm_end(self, response, **kwargs):
        # accumulate LLM generation time
        try:
            start_ts = self._llm_timer_stack.pop() if self._llm_timer_stack else None
            if start_ts is not None:
                self.llm_runtime_sec += (time.perf_counter() - start_ts)
        except Exception:
            pass
        # accumulate tokens
        try:
            usage = response.llm_output.get("token_usage")
        except Exception:
            usage = None
        if usage:
            self.prompt_tokens += usage.get("prompt_tokens", 0)
            self.completion_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get(
                "total_tokens",
                usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0),
            )
        else:
            # Fallback: approximate tokens from output text
            approx_tokens = 0
            try:
                import tiktoken

                enc = tiktoken.get_encoding("cl100k_base")
                for gen_list in getattr(response, "generations", []):
                    for gen in gen_list:
                        text = getattr(gen, "text", "")
                        approx_tokens += len(enc.encode(text))
            except Exception:
                for gen_list in getattr(response, "generations", []):
                    for gen in gen_list:
                        text = getattr(gen, "text", "")
                        approx_tokens += max(1, len(text) // 4)
            self.completion_tokens += approx_tokens
            self.total_tokens += approx_tokens