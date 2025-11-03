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
# Short caches in seconds to avoid repeated introspection/prompts
PROBE_CACHE_SECONDS = int(_env_any(["ROUTER_PROBE_CACHE_SECONDS"], default="60"))
PLAN_CACHE_SECONDS = int(_env_any(["ROUTER_PLAN_CACHE_SECONDS"], default="120"))
REWRITE_CACHE_SECONDS = int(_env_any(["ROUTER_REWRITE_CACHE_SECONDS"], default="180"))

# SQL agent iteration cap and table sampling for faster planning/execution
SQL_AGENT_MAX_ITERATIONS = int(_env_any(["SQL_AGENT_MAX_ITERATIONS"], default="12"))
SQL_TABLE_INFO_SAMPLE_ROWS = int(_env_any(["SQL_TABLE_INFO_SAMPLE_ROWS"], default="0"))