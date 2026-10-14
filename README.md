# 🧭 AI Decision Simulator ("What Would Past-Me Think")

A ChatGPT-style AI decision coach built in pure Python (Streamlit): streaming answers, saved chat history, clarifying questions, scored options, risks and next steps, powered by Groq, with an offline data engine as backup.
Free to run: local dataset + rule-based engine, with optional Groq AI for natural conversation.
No local LLM, no llama.cpp/GPT4All, no paid service.

## Run it (3 steps)
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`   (or double-click `run.bat` on Windows / `./run.sh` on Mac/Linux)

The Groq key is read from `.env` (`GROQ_API_KEY=...`). Without it the app still works in rule-based mode.

## Project layout
```
app.py                  Streamlit UI (chat, log outcome, insights, about)
decision_sim/
  config.py             settings + .env loading
  schema.py             allowed values / labels
  data_gen.py           builds the sample decisions.csv
  store.py              Polars read/write/validate
  stats.py              DuckDB summaries
  engine.py             features -> FAISS/scikit-learn search -> Bayesian Beta forecast + confidence
  explain.py            rule-based explanation templates (works offline)
  understand.py         Groq / keyword parser: text -> structured decision
  llm.py                Groq client with retry + graceful fallback
data/decisions.csv      1,000 SIMULATED sample decisions (replace with your own!)
tests/test_core.py      self-test
```

## How it works
**AI-first (default):** Message -> Groq (`openai/gpt-oss-120b`, fallback `gpt-oss-20b`) identifies your real
options, scores each 0-100, and returns verdict, confidence, pros/cons, key factors, risks + mitigations,
next steps and a "past-me / future-me" view. Your own logged outcomes are sent along as "past-me" memory.

**Data engine (reference + offline fallback):** similar past decisions in `decisions.csv` (weighted by similarity,
recency, and your own logged entries x2) -> Beta posterior -> chance of a good outcome + 90% range + confidence.
Shown in a collapsible "historical pattern" panel, and used automatically if Groq is unavailable.
The Insights page runs a live leave-one-out accuracy test.

## Using your own data
Keep the columns: `decision_id, decision_type, context, choice, previous_experience, risk_level,
time_available, expected_result, actual_result, satisfaction, outcome_score, date`
(see `decision_sim/schema.py` for allowed values). Or use the **Log outcome** page in the app.

## Security
Never share or commit `.env`. If a key was ever pasted somewhere public, revoke it at console.groq.com/keys and create a new one.
