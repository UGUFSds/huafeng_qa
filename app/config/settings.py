import os
from dotenv import load_dotenv

# Load environment from .env if present
load_dotenv()

# Helper: read the first non-empty env value from a list of keys
def _env_any(keys, default=None):
    for key in keys:
        val = os.getenv(key)
        if val not in (None, ""):
            return val
    return default

# LLM configuration (supports new LLM_* and legacy HUAFENG_* prefixes)
BASE_URL = _env_any(["LLM_BASE_URL", "HUAFENG_DEEPSEEK_BASE_URL"], default="https://api.deepseek.com").rstrip("/")
API_KEY = _env_any(["LLM_API_KEY", "HUAFENG_DEEPSEEK_API_KEY"]) 
PROVIDER = _env_any(["LLM_PROVIDER", "HUAFENG_DEEPSEEK_PROVIDER"], default="api").strip().lower()
INTERNAL_BASE_URL = _env_any(["LLM_INTERNAL_BASE_URL", "HUAFENG_DEEPSEEK_INTERNAL_BASE_URL"], default=BASE_URL).rstrip("/")
ENABLE_AUTO_TOOL_CHOICE = _env_any(["LLM_ENABLE_AUTO_TOOL_CHOICE", "HUAFENG_DEEPSEEK_ENABLE_AUTO_TOOL_CHOICE"], default="").strip()
TOOL_CALL_PARSER = _env_any(["LLM_TOOL_CALL_PARSER", "HUAFENG_DEEPSEEK_TOOL_CALL_PARSER"], default="").strip()
EXTRA_HEADERS_JSON = _env_any(["LLM_EXTRA_HEADERS_JSON", "HUAFENG_DEEPSEEK_EXTRA_HEADERS_JSON"], default="").strip()

MODEL = _env_any(["LLM_MODEL", "HUAFENG_TEXT2SQL_MODEL", "HUAFENG_ANALYSIS_MODEL"], default="deepseek-chat")
LLM_REQUEST_TIMEOUT = int(_env_any(["LLM_REQUEST_TIMEOUT"], default="25"))
LLM_MAX_TOKENS = int(_env_any(["LLM_MAX_TOKENS"], default="512"))

# Postgres connection settings (supports new POSTGRES_* and legacy HUAFENG_* prefixes)
PG_HOST = _env_any(["POSTGRES_HOST", "HUAFENG_LOCAL_POSTGRES_HOST"], default="127.0.0.1")
PG_PORT = _env_any(["POSTGRES_PORT", "HUAFENG_LOCAL_POSTGRES_PORT"], default="5433")
PG_DB = _env_any(["POSTGRES_DB", "HUAFENG_LOCAL_POSTGRES_DB"], default="app_db")
PG_USER = _env_any(["POSTGRES_USER", "HUAFENG_LOCAL_POSTGRES_USER"], default="postgres")
PG_PASSWORD = _env_any(["POSTGRES_PASSWORD", "HUAFENG_LOCAL_POSTGRES_PASSWORD"], default="postgres")
PG_SSLMODE = _env_any(["POSTGRES_SSLMODE", "HUAFENG_POSTGRES_SSLMODE"], default="disable")

DB_URI = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"
)

# CSV data source path (supports legacy HUAFENG_*)
OPCAE_CSV_PATH = _env_any(["OPCAE_CSV_PATH", "HUAFENG_OPCAE_CSV_PATH"], default=os.path.join("data", "point_data.csv"))

# Router/runtime optimization toggles
# Use LLM planner only when explicitly enabled (and multiple candidates)
USE_LLM_PLANNER = _env_any(["ROUTER_USE_LLM_PLANNER"], default="1").strip() in ("1", "true", "True")
# Skip LLM summarizer when only one source contributed
SUMMARIZE_SINGLE_SOURCE = _env_any(["ROUTER_SUMMARIZE_SINGLE_SOURCE"], default="0").strip() in ("1", "true", "True")
# Globally enable/disable LLM summarizer for multi-source
USE_LLM_SUMMARIZER = _env_any(["ROUTER_USE_LLM_SUMMARIZER"], default="1").strip() in ("1", "true", "True")
# Enable/disable source probing to save time
ROUTER_ENABLE_PROBE = _env_any(["ROUTER_ENABLE_PROBE"], default="1").strip() in ("1", "true", "True")
# 并发执行多源以提高吞吐（在存在 CSV→SQL 依赖时自动串行）
ROUTER_PARALLEL_EXECUTE = _env_any(["ROUTER_PARALLEL_EXECUTE"], default="1").strip() in ("1", "true", "True")
# Short caches in seconds to avoid repeated introspection/prompts
PROBE_CACHE_SECONDS = int(_env_any(["ROUTER_PROBE_CACHE_SECONDS"], default="120"))
PLAN_CACHE_SECONDS = int(_env_any(["ROUTER_PLAN_CACHE_SECONDS"], default="300"))
REWRITE_CACHE_SECONDS = int(_env_any(["ROUTER_REWRITE_CACHE_SECONDS"], default="600"))
RESULT_CACHE_SECONDS = int(_env_any(["ROUTER_RESULT_CACHE_SECONDS"], default="600"))

# SQL agent iteration cap and table sampling for faster planning/execution
SQL_AGENT_MAX_ITERATIONS = int(_env_any(["SQL_AGENT_MAX_ITERATIONS"], default="12"))
SQL_TABLE_INFO_SAMPLE_ROWS = int(_env_any(["SQL_TABLE_INFO_SAMPLE_ROWS"], default="0"))
SQL_TABLE_INFO_CACHE_SECONDS = int(_env_any(["SQL_TABLE_INFO_CACHE_SECONDS"], default="1200"))

# CSV agent behavior
CSV_AGENT_SECOND_PASS = _env_any(["CSV_AGENT_SECOND_PASS"], default="1").strip() in ("1", "true", "True")
CSV_FALLBACK_AUTO_FILTER = _env_any(["CSV_FALLBACK_AUTO_FILTER"], default="1").strip() in ("1", "true", "True")

# Clarification (disambiguation) behavior to avoid enumerating queries
CLARIFY_ENABLED = _env_any(["ROUTER_CLARIFY_ENABLED"], default="1").strip() in ("1", "true", "True")
CLARIFY_CANDIDATE_THRESHOLD = int(_env_any(["ROUTER_CLARIFY_CANDIDATE_THRESHOLD"], default="4"))
CLARIFY_MAX_OPTIONS = int(_env_any(["ROUTER_CLARIFY_MAX_OPTIONS"], default="8"))

# SQL query year guardrails
# 强制在未明确年份时仅查询当年数据；可通过环境变量关闭或调整时间列名
SQL_ENFORCE_CURRENT_YEAR_IF_UNSPECIFIED = _env_any(["SQL_ENFORCE_CURRENT_YEAR_IF_UNSPECIFIED"], default="1").strip() in ("1", "true", "True")
SQL_YEAR_COLUMN_NAME = _env_any(["SQL_YEAR_COLUMN_NAME"], default="ts").strip()

# Per-query report directory（可选）。若设置，service 将为每次查询写出 JSON 报告。
EVALS_REPORT_DIR = _env_any(["EVALS_REPORT_DIR", "HUAFENG_EVALS_REPORT_DIR"], default="").strip()

# Phoenix observability
PHOENIX_ENABLED = _env_any(["PHOENIX_ENABLED"], default="0").strip() in ("1", "true", "True")
PHOENIX_ENDPOINT = _env_any(["PHOENIX_ENDPOINT"], default="http://127.0.0.1:6006/v1/traces").strip()
PHOENIX_PROJECT_NAME = _env_any(["PHOENIX_PROJECT_NAME"], default="huafeng-qa").strip()
COST_RATE_PER_1K_PROMPT = float(_env_any(["COST_RATE_PER_1K_PROMPT"], default="0"))
COST_RATE_PER_1K_COMPLETION = float(_env_any(["COST_RATE_PER_1K_COMPLETION"], default="0"))
ROUTER_STRICT_AFTER_CLARIFICATION = _env_any(["ROUTER_STRICT_AFTER_CLARIFICATION"], default="1").strip() in ("1", "true", "True")
COST_RATE_PER_1K_PROMPT = float(_env_any(["COST_RATE_PER_1K_PROMPT"], default="0"))
COST_RATE_PER_1K_COMPLETION = float(_env_any(["COST_RATE_PER_1K_COMPLETION"], default="0"))
