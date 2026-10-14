"""Groq API client: structured decision analysis + streaming chat, with graceful fallback."""
import json
import re
import time
from datetime import date

import requests

from . import config

LAST_ERROR = ""


def available() -> bool:
    return bool(config.GROQ_API_KEY)


def _clean(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _payload(model, messages, temperature, max_tokens, effort, json_mode=False, stream=False) -> dict:
    p = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if model.startswith("openai/gpt-oss"):
        p["reasoning_effort"] = effort
    if json_mode:
        p["response_format"] = {"type": "json_object"}
    if stream:
        p["stream"] = True
    return p


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.GROQ_API_KEY}"}


def chat(messages: list[dict], temperature=0.3, max_tokens=600, json_mode=False,
         effort="low") -> str | None:
    """Returns the model text, or None on any failure (caller falls back to rule-based logic)."""
    global LAST_ERROR
    if not available():
        LAST_ERROR = "No GROQ_API_KEY set"
        return None
    for model in dict.fromkeys([config.GROQ_MODEL, config.GROQ_FALLBACK_MODEL]):
        payload = _payload(model, messages, temperature, max_tokens, effort, json_mode)
        status = None
        for attempt in range(3):
            try:
                r = requests.post(config.GROQ_URL, timeout=90, json=payload, headers=_headers())
            except Exception as e:  # network down, timeout ...
                LAST_ERROR = f"{type(e).__name__}: {e}"
                return None
            status = r.status_code
            if status == 200:
                LAST_ERROR = ""
                try:
                    return _clean(r.json()["choices"][0]["message"].get("content", "")) or None
                except Exception as e:
                    LAST_ERROR = f"Bad response: {e}"
                    return None
            LAST_ERROR = f"Groq {status}: {r.text[:160]}"
            if status == 400 and "response_format" in payload:
                payload.pop("response_format")  # JSON mode rejected -> retry as plain text
                continue
            if status in (429, 500, 502, 503) and attempt == 0:
                time.sleep(2)  # one polite retry
                continue
            break
        if status not in (400, 404, 429, 503):   # these justify trying the next model
            return None
    return None


def stream(messages: list[dict], temperature=0.5, max_tokens=1600, effort="low"):
    """Yields text chunks as Groq generates them. Yields nothing on failure (LAST_ERROR is set)."""
    global LAST_ERROR
    if not available():
        LAST_ERROR = "No GROQ_API_KEY set"
        return
    for model in dict.fromkeys([config.GROQ_MODEL, config.GROQ_FALLBACK_MODEL]):
        payload = _payload(model, messages, temperature, max_tokens, effort, stream=True)
        try:
            with requests.post(config.GROQ_URL, json=payload, headers=_headers(), stream=True, timeout=90) as r:
                if r.status_code != 200:
                    LAST_ERROR = f"Groq {r.status_code}: {r.text[:160]}"
                    if r.status_code in (400, 404, 429, 500, 502, 503):
                        continue
                    return
                LAST_ERROR = ""
                r.encoding = "utf-8"
                for line in r.iter_lines(decode_unicode=True):
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content")
                    except Exception:
                        continue
                    if delta:
                        yield delta
                return
        except Exception as e:
            LAST_ERROR = f"{type(e).__name__}: {e}"
            return


def extract_json(text: str | None) -> dict | None:
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


SYSTEM_EXTRACT = """You convert a user's message into structured data for a decision simulator.
Return ONLY one JSON object, no prose, with these keys:
is_decision (true if the user is asking for help with a choice/decision, else false),
decision_type (one of: career, finance, education, health, relationships, purchase, travel, productivity, other),
choice (the option the user is leaning to: act_now | wait | decline | compromise, or null if unclear),
previous_experience (none | negative | mixed | positive, or null if not mentioned),
risk_level (low | medium | high, or null),
time_available (hours | days | weeks | months, or null),
summary (max 12 words describing the decision).
Never guess values that were not stated; use null instead."""


def extract(text: str, history: list[dict]) -> dict | None:
    ctx = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in history[-4:])
    out = chat([{"role": "system", "content": SYSTEM_EXTRACT},
                {"role": "user", "content": f"Recent chat:\n{ctx}\n\nNew message:\n{text}"}],
               temperature=0, max_tokens=300)
    return extract_json(out)


SYSTEM_GENERAL = """You are 'Past-Me', a friendly assistant that helps people make better decisions.
Answer briefly (max 120 words). If the user hasn't described a decision, invite them to describe one
(what they're deciding, options, risk, how much time they have, any past experience).
If a previous analysis is given, use it to answer follow-up questions; never invent numbers."""


def general(history: list[dict], last_analysis: str) -> str | None:
    msgs = [{"role": "system", "content": SYSTEM_GENERAL + (f"\n\nLast analysis:\n{last_analysis}" if last_analysis else "")}]
    msgs += [{"role": m["role"], "content": m["content"][:800]} for m in history[-6:]]
    return chat(msgs, temperature=0.5, max_tokens=300)


# ---------------------------------------------------------------- step 1: structured analysis
SYSTEM_ADVISE = """You are the analysis engine of 'Past-Me', an expert decision coach (behavioural economist +
seasoned mentor). Read the whole conversation and analyse the user's CURRENT decision carefully and honestly.

How to think:
- Identify the REAL options (2-4) in the user's own words, e.g. "Accept the startup offer",
  "Negotiate a higher salary first", "Stay in the current job". If only one option is stated, add the most
  sensible alternatives (wait, negotiate, a small trial, decline).
- Score each option 0-100 = probability it leads to a good outcome for THIS person. Weigh goals, money,
  size of the downside, reversibility, time pressure, past experience, emotional cost, opportunity cost and
  second-order effects. Use well-known base rates and expert consensus where relevant.
- Be calibrated: spread scores when one option is clearly better; keep them close only for a genuine toss-up.
  Never invent facts about the user. State assumptions explicitly.
- The user's own logged past decisions ("past-me") are real evidence about them: use their lessons.
- Health, legal and large financial decisions: mention consulting a qualified professional in the summary.

Stage:
- "clarify" ONLY if the decision is too vague to analyse responsibly (e.g. "should I quit?" with no context)
  AND the assistant has NOT already asked clarifying questions about this decision. Then put the 2-4 most
  decision-relevant questions in "questions"; options may be a rough draft.
- otherwise "analyze" (prefer this; list what you assumed).
- is_decision=false for greetings, thanks, small talk, or general questions that need no option scoring;
  follow-up questions about an earlier analysis ("why?", "explain the risks") are also false.
  A "what if ..." that changes the scenario IS a decision: re-analyse it.

Return ONLY one JSON object with these keys:
{"is_decision": true/false, "stage": "clarify|analyze",
 "title": "max 7 words", "decision_type": "career|finance|education|health|relationships|purchase|travel|productivity|other",
 "leaning": "option the user currently prefers or null", "leaning_category": "act_now|wait|decline|compromise|null",
 "previous_experience": "none|negative|mixed|positive|null", "risk_level": "low|medium|high|null",
 "time_available": "hours|days|weeks|months|null",
 "verdict": "Go for it|Proceed with caution|Hold off", "recommendation": "exact name of the best option",
 "confidence": 0-100, "summary": "2 sentences: verdict + main reason", "reasoning": "4-6 sentences",
 "options": [{"name": "", "score": 0-100, "risk": "low|medium|high", "reversible": true/false,
              "pros": ["2-4 specific"], "cons": ["2-4 specific"]}],
 "factors": [{"name": "", "impact": "positive|negative|neutral", "weight": 1-5, "note": "short"}],
 "risks": [{"risk": "", "mitigation": ""}], "next_steps": ["3-5 concrete actions"],
 "questions": ["questions that would sharpen the analysis"], "assumptions": ["what you assumed"],
 "past_me": "one sentence: what their past experience says", "future_me": "one sentence: the 5-year view",
 "follow_ups": ["3 short messages the USER could send next, first person, max 9 words each"]}"""

VERDICTS = ("Go for it", "Proceed with caution", "Hold off")


def _int(v, lo=0, hi=100, default=50) -> int:
    try:
        return int(max(lo, min(hi, round(float(v)))))
    except Exception:
        return default


def _strs(v, n=6) -> list[str]:
    return [str(x).strip() for x in (v if isinstance(v, list) else []) if str(x).strip()][:n]


def normalize_advice(d: dict | None) -> dict | None:
    """Validate the model's JSON so the UI never breaks on a sloppy answer."""
    if not isinstance(d, dict):
        return None
    follow = _strs(d.get("follow_ups"), 3)
    if not d.get("is_decision", True):
        return {"is_decision": False, "stage": "chat", "follow_ups": follow}
    stage = "clarify" if d.get("stage") == "clarify" else "analyze"
    opts = []
    for o in d.get("options") or []:
        if isinstance(o, dict) and str(o.get("name", "")).strip():
            opts.append({"name": str(o["name"]).strip(), "score": _int(o.get("score")),
                         "risk": o.get("risk") if o.get("risk") in ("low", "medium", "high") else "medium",
                         "reversible": bool(o.get("reversible", True)),
                         "pros": _strs(o.get("pros"), 5), "cons": _strs(o.get("cons"), 5)})
    if not opts and stage == "analyze":
        return None
    opts.sort(key=lambda o: -o["score"])
    names = [o["name"] for o in opts]
    rec = str(d.get("recommendation") or "").strip()
    if names and rec not in names:
        rec = next((n for n in names if rec and (rec.lower() in n.lower() or n.lower() in rec.lower())), names[0])
    top = next((o["score"] for o in opts if o["name"] == rec), 50)
    verdict = d.get("verdict") if d.get("verdict") in VERDICTS else (
        "Go for it" if top >= 70 else "Proceed with caution" if top >= 50 else "Hold off")
    factors = [{"name": str(f.get("name", "")).strip(),
                "impact": f.get("impact") if f.get("impact") in ("positive", "negative", "neutral") else "neutral",
                "weight": _int(f.get("weight"), 1, 5, 3), "note": str(f.get("note") or "").strip()}
               for f in d.get("factors") or [] if isinstance(f, dict) and str(f.get("name", "")).strip()][:6]
    risks = [{"risk": str(r.get("risk", "")).strip(), "mitigation": str(r.get("mitigation") or "").strip()}
             for r in d.get("risks") or [] if isinstance(r, dict) and str(r.get("risk", "")).strip()][:5]
    return {"is_decision": True, "stage": stage, "title": str(d.get("title") or "Your decision").strip()[:80],
            "decision_type": d.get("decision_type"), "leaning": d.get("leaning"),
            "leaning_category": d.get("leaning_category"),
            "previous_experience": d.get("previous_experience"), "risk_level": d.get("risk_level"),
            "time_available": d.get("time_available"), "verdict": verdict, "recommendation": rec,
            "rec_score": top, "confidence": _int(d.get("confidence"), default=60),
            "summary": str(d.get("summary") or "").strip(), "reasoning": str(d.get("reasoning") or "").strip(),
            "options": opts, "factors": factors, "risks": risks,
            "next_steps": _strs(d.get("next_steps")), "questions": _strs(d.get("questions"), 4),
            "assumptions": _strs(d.get("assumptions"), 5), "follow_ups": follow,
            "past_me": str(d.get("past_me") or "").strip(), "future_me": str(d.get("future_me") or "").strip()}


def advise(text: str, history: list[dict], past_me: str = "", last_analysis: str = "") -> dict | None:
    """Full AI decision analysis. Returns a validated dict, or None if Groq failed."""
    ctx = "\n".join(f"{m['role']}: {m['content'][:700]}" for m in history[-8:])
    user = (f"Today: {date.today().isoformat()}\n\nConversation so far:\n{ctx or '(none)'}\n\n"
            f"Previous analysis:\n{last_analysis or '(none)'}\n\n"
            f"My own logged past decisions (past-me):\n{past_me or '(none logged yet)'}\n\n"
            f"New message:\n{text}")
    out = chat([{"role": "system", "content": SYSTEM_ADVISE}, {"role": "user", "content": user}],
               temperature=0.25, max_tokens=4000, json_mode=True, effort="medium")
    return normalize_advice(extract_json(out))


# ---------------------------------------------------------------- step 2: streamed conversational answer
SYSTEM_CHAT = """You are Past-Me, a brilliant, warm and brutally honest AI decision coach. You talk like a
trusted mentor: specific to the user's situation, practical, never generic, never preachy.
Format with Markdown: short paragraphs, **bold** key points, bullet lists; a table only when comparing options.

You receive an ANALYSIS (JSON) computed for this conversation. Stay 100% consistent with it: same
recommendation, same verdict, same scores. The app shows a visual card with all scores, pros/cons, risks and
steps right below your message, so do NOT dump every number; explain the thinking.

If ANALYSIS.stage == "clarify": acknowledge the situation in 1-2 sentences, share your initial instinct in one
sentence, then ask the questions as a short numbered list and explain in a few words why each one matters.

If ANALYSIS.stage == "analyze", use exactly this shape:
**Bottom line:** <one-sentence recommendation>
### Why
2-4 bullets with the decisive reasons, tied to the user's details.
### How I'd play it
a short concrete plan (what to do this week).
### Watch out for
1-3 bullets on the biggest risks and how to protect yourself.
Then one short closing line that invites them to refine the scenario.

If ANALYSIS is none (general chat or follow-up): answer directly and helpfully. For follow-ups, use the
previous analysis in the conversation and reason step by step about the user's question.

Rules: 120-280 words. Never invent statistics or facts about the user. If you assumed something, say so
briefly. For health, legal or major financial decisions, add one line recommending a qualified professional."""

_GROUND_KEYS = ("stage", "title", "verdict", "recommendation", "confidence", "summary", "reasoning", "options",
                "risks", "next_steps", "questions", "assumptions", "past_me", "future_me")


def stream_reply(conversation: list[dict], adv: dict | None, last_analysis: str = ""):
    if adv and adv.get("is_decision"):
        ground = json.dumps({k: adv.get(k) for k in _GROUND_KEYS})
    else:
        ground = "none" + (f"\n\nPrevious analysis in this conversation:\n{last_analysis}" if last_analysis else "")
    msgs = [{"role": "system", "content": f"{SYSTEM_CHAT}\n\nANALYSIS:\n{ground}"}]
    msgs += [{"role": m["role"], "content": m["content"][:2500]} for m in conversation[-10:]]
    yield from stream(msgs)
