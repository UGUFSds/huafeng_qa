import os
from dotenv import load_dotenv

# Load environment from .env if present
load_dotenv()

# DeepSeek and LLM configuration
BASE_URL = os.getenv("HUAFENG_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("HUAFENG_DEEPSEEK_API_KEY")
PROVIDER = os.getenv("HUAFENG_DEEPSEEK_PROVIDER", "api").strip().lower()
INTERNAL_BASE_URL = os.getenv("HUAFENG_DEEPSEEK_INTERNAL_BASE_URL", BASE_URL).rstrip("/")
ENABLE_AUTO_TOOL_CHOICE = os.getenv("HUAFENG_DEEPSEEK_ENABLE_AUTO_TOOL_CHOICE", "").strip()
TOOL_CALL_PARSER = os.getenv("HUAFENG_DEEPSEEK_TOOL_CALL_PARSER", "").strip()
EXTRA_HEADERS_JSON = os.getenv("HUAFENG_DEEPSEEK_EXTRA_HEADERS_JSON", "").strip()

MODEL = (
    os.getenv("HUAFENG_TEXT2SQL_MODEL")
    or os.getenv("HUAFENG_ANALYSIS_MODEL")
    or "deepseek-chat"
)

# Postgres connection settings
PG_HOST = os.getenv("HUAFENG_LOCAL_POSTGRES_HOST", "127.0.0.1")
PG_PORT = os.getenv("HUAFENG_LOCAL_POSTGRES_PORT", "5433")
PG_DB = os.getenv("HUAFENG_LOCAL_POSTGRES_DB", "huafeng_db")
PG_USER = os.getenv("HUAFENG_LOCAL_POSTGRES_USER", "postgres")
PG_PASSWORD = os.getenv("HUAFENG_LOCAL_POSTGRES_PASSWORD", "postgres")
PG_SSLMODE = os.getenv("HUAFENG_POSTGRES_SSLMODE", "disable")

DB_URI = (
    f"postgresql+psycopg2://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}?sslmode={PG_SSLMODE}"
)

# CSV data source path
OPCAE_CSV_PATH = os.getenv("HUAFENG_OPCAE_CSV_PATH", os.path.join("data", "point_data.csv"))