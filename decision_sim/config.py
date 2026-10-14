"""Central configuration. Secrets are read from the .env file, never hard-coded."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
try:  # Streamlit Cloud: keys come from the app's Secrets instead of .env
    import streamlit as st
    for _k in ("GROQ_API_KEY", "GROQ_MODEL", "GROQ_FALLBACK_MODEL"):
        if _k in st.secrets and not os.getenv(_k):
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass  # no secrets.toml locally is fine

DATA_PATH = ROOT / "data" / "decisions.csv"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# A non-Llama model hosted on Groq. Change in .env if you want another one.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b").strip()
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
