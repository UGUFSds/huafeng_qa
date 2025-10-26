import os
from dotenv import load_dotenv

# Load .env
load_dotenv()

BASE_URL = os.getenv("HUAFENG_DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
API_KEY = os.getenv("HUAFENG_DEEPSEEK_API_KEY")
MODEL = (
    os.getenv("HUAFENG_TEXT2SQL_MODEL")
    or os.getenv("HUAFENG_ANALYSIS_MODEL")
    or "deepseek-chat"
)

print(
    f"[env] BASE_URL={BASE_URL}, MODEL={MODEL}, API_KEY_SUFFIX={'***' + (API_KEY[-4:] if API_KEY else 'NONE')}"
)

if not API_KEY:
    raise RuntimeError("HUAFENG_DEEPSEEK_API_KEY 未设置，无法调用 DeepSeek API")

from langchain_openai import ChatOpenAI


def try_invoke(base_url: str):
    # Instantiate LangChain OpenAI-compatible client for DeepSeek
    llm = ChatOpenAI(
        model=MODEL,
        api_key=API_KEY,
        base_url=base_url,
        temperature=0.0,
    )
    # Minimal test prompt
    resp = llm.invoke("请用一个词回答：2+2=?")
    print("[ok] content:", resp.content)


try:
    # First attempt with BASE_URL
    try_invoke(BASE_URL)
except Exception as e1:
    print("[warn] 首次调用失败：", e1)
    # Fallback: some deployments require /v1 path suffix
    fallback_url = BASE_URL + "/v1"
    print("[info] 尝试使用 fallback base_url:", fallback_url)
    try:
        try_invoke(fallback_url)
    except Exception as e2:
        print("[error] fallback 调用仍失败：", e2)
        raise