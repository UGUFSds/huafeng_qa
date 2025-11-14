from typing import Any, Dict
from opentelemetry import trace as trace_api
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
try:
    from openinference_semantic_conventions import SEMRESATTRS_PROJECT_NAME
except Exception:
    SEMRESATTRS_PROJECT_NAME = "openinference.project.name"
from openinference.instrumentation.langchain import LangChainInstrumentor
from app.llm.factory import build_llm
from app.config.settings import PHOENIX_PROJECT_NAME

def init(endpoint: str) -> None:
    resource = Resource.create({SEMRESATTRS_PROJECT_NAME: PHOENIX_PROJECT_NAME or "default"})
    tp = TracerProvider(resource=resource)
    trace_api.set_tracer_provider(tp)
    tp.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint)))
    LangChainInstrumentor().instrument()

def evaluate(base_url: str, question: str, final_text: str, lang: str) -> Dict[str, Any]:
    llm = build_llm(base_url)
    prompt = (
        "请对回答进行评估，输出JSON，字段：relevance(1-5), completeness(1-5), clarity(1-5), rationale。"
        "问题：" + question + "\n回答：" + final_text + "\nJSON："
    )
    o = llm.invoke(prompt)
    if isinstance(o, str):
        text = o
    else:
        text = getattr(o, "content", str(o))
    try:
        import json
        return json.loads(text)
    except Exception:
        return {"relevance": None, "completeness": None, "clarity": None, "rationale": text}

def record_eval(evals: Dict[str, Any], rep: Dict[str, Any]) -> None:
    tracer = trace_api.get_tracer("huafeng-qa")
    with tracer.start_as_current_span("post_query_eval") as span:
        span.set_attribute("evals.relevance", evals.get("relevance"))
        span.set_attribute("evals.completeness", evals.get("completeness"))
        span.set_attribute("evals.clarity", evals.get("clarity"))
        span.set_attribute("evals.rationale", str(evals.get("rationale")))
        for k, v in (rep.get("metrics") or {}).items():
            span.set_attribute("metrics." + k, v)
        span.set_attribute("question", rep.get("question"))
        span.set_attribute("final_text", rep.get("final_text"))
