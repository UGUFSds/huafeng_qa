import os
import json
import random
from typing import Any, Dict, List, Optional

from app.config.settings import (
    EVALS_ENABLED,
    EVALS_PROVIDER,
    EVALS_MODEL,
    EVALS_SAMPLING_RATE,
    EVALS_LOG_PATH,
)
from app.config.settings import BASE_URL, API_KEY


def _ensure_dir(path: str):
    try:
        d = os.path.dirname(path)
        if d and not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
    except Exception:
        pass


def _extract_references_from_outputs(outputs: List[Dict[str, Any]]) -> List[str]:
    refs: List[str] = []
    if not outputs:
        return refs
    for out in outputs:
        if out.get("source") == "csv_lookup":
            raw = out.get("raw") or {}
            data = raw.get("data") if isinstance(raw, dict) else None
            rows = []
            if isinstance(data, dict):
                if isinstance(data.get("rows"), list):
                    rows = data.get("rows")
                elif isinstance(data.get("groups"), list):
                    rows = data.get("groups")
            if rows:
                for r in rows[:10]:
                    if isinstance(r, dict):
                        try:
                            # flatten a few key fields for reference text
                            parts = []
                            for k in ("point_name", "code", "desc", "table_name", "tag", "name"):
                                v = r.get(k)
                                if isinstance(v, str) and v.strip():
                                    parts.append(f"{k}:{v.strip()}")
                            if not parts:
                                # fallback to compact json
                                parts.append(json.dumps(r, ensure_ascii=False))
                            refs.append(" | ".join(parts))
                        except Exception:
                            refs.append(json.dumps(r, ensure_ascii=False))
    return refs


def maybe_run_evals(question: str, final_text: str, outputs: List[Dict[str, Any]], meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Run Phoenix Evals on a single QA sample, safely and optionally.

    - Respects EVALS_ENABLED and sampling rate.
    - Gracefully no-ops if phoenix.evals is unavailable or provider credentials are missing.
    - Logs results to EVALS_LOG_PATH when successful.
    """
    try:
        if not EVALS_ENABLED:
            return None
        if EVALS_SAMPLING_RATE <= 0.0:
            return None
        if random.random() > EVALS_SAMPLING_RATE:
            return None

        # Try import Phoenix Evals
        try:
            from phoenix.evals import create_classifier
            from phoenix.evals.evaluators import HallucinationEvaluator
            from phoenix.evals.llm import LLM
        except Exception:
            # Library not installed or incompatible
            return None

        # Prepare judge LLM using explicit api_key/base_url to support OpenAI-compatible providers (e.g., DeepSeek)
        judge = None
        try:
            judge = LLM(provider=EVALS_PROVIDER, model=EVALS_MODEL, client="langchain", api_key=API_KEY, base_url=BASE_URL)
        except Exception:
            # Fallback: some providers require '/v1' suffix on base_url
            try:
                fb_url = BASE_URL.rstrip("/") + "/v1"
                judge = LLM(provider=EVALS_PROVIDER, model=EVALS_MODEL, client="langchain", api_key=API_KEY, base_url=fb_url)
            except Exception:
                # If provider/model misconfigured, skip
                return None

        # Build dataset row
        refs = _extract_references_from_outputs(outputs)
        dataset_row = {
            "input": question,
            "output": final_text,
            "reference": refs,
        }

        results: Dict[str, Any] = {"ok": True, "provider": EVALS_PROVIDER, "model": EVALS_MODEL, "explanations": {}}

        # Helpfulness classifier (yes/no)
        try:
            helpfulness = create_classifier(
                name="helpfulness",
                prompt_template=(
                    "你是评审专家，请判断系统给出的回答是否对用户问题有帮助。\n"
                    "问题: {input}\n回答: {output}\n"
                    "只输出两类之一：'helpful' 或 'not_helpful'，并给出一句理由。"
                ),
                llm=judge,
            )
            helpful_res = helpfulness.evaluate([dataset_row])
            results["helpfulness"] = helpful_res.results[0].label if helpful_res and helpful_res.results else None
            results["explanations"]["helpfulness"] = getattr(helpful_res.results[0], "explanation", None) if helpful_res and helpful_res.results else None
        except Exception:
            results["helpfulness"] = None

        # Hallucination evaluator (requires references)
        if refs:
            try:
                hallu = HallucinationEvaluator(llm=judge)
                hallu_res = hallu.evaluate([dataset_row])
                results["hallucination"] = hallu_res.results[0].label if hallu_res and hallu_res.results else None
                results["explanations"]["hallucination"] = getattr(hallu_res.results[0], "explanation", None) if hallu_res and hallu_res.results else None
            except Exception:
                results["hallucination"] = None

        # Persist to file
        try:
            _ensure_dir(EVALS_LOG_PATH)
            with open(EVALS_LOG_PATH, "a", encoding="utf-8") as f:
                rec = {
                    "input": question,
                    "output": final_text,
                    "reference": refs,
                    "meta": meta or {},
                    "evals": results,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

        return results
    except Exception:
        return None