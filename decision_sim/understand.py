"""Turns free text into structured decision features.
Groq does the smart parsing; a keyword parser is the always-available fallback."""
import re

from . import llm
from .schema import (CHOICES, DECISION_TYPES, DEFAULTS, EXPERIENCE, RISK, TIME)

_TYPE_WORDS = {
    "career": ["job", "offer", "salary", "promotion", "resign", "quit", "hire", "interview", "boss",
               "career", "startup", "freelanc", "manager", "colleague"],
    "finance": ["invest", "stock", "loan", "saving", "budget", "crypto", "debt", "mortgage", "money",
                "shares", "insurance", "loss", "profit"],
    "education": ["study", "course", "degree", "university", "exam", "college", "master", "learn",
                  "school", "scholarship", "phd"],
    "health": ["doctor", "diet", "gym", "exercise", "sleep", "surgery", "health", "weight", "medical",
               "therapy", "workout"],
    "relationships": ["marry", "marriage", "relationship", "friend", "partner", "breakup", "family",
                      "girlfriend", "boyfriend", "wife", "husband", "parents"],
    "purchase": ["buy", "purchase", "phone", "laptop", "car", "house", "flat", "furniture", "price"],
    "travel": ["trip", "travel", "visit", "relocate", "abroad", "visa", "vacation", "flight", "move to"],
    "productivity": ["habit", "schedule", "routine", "project", "deadline", "focus", "plan", "task"],
}
_CHOICE_WORDS = {
    "wait": ["wait", "postpone", "later", "delay", "not now", "hold off", "sleep on it"],
    "decline": ["decline", "reject", "say no", "refuse", "turn down", "skip", "pass on", "not to"],
    "compromise": ["negotiate", "middle", "part-time", "part time", "compromise", "trial", "test first",
                   "gradually", "small amount", "halfway"],
    "act_now": ["accept", "go ahead", "do it now", "start now", "buy now", "take the", "say yes",
                "sign", "commit", "go for it"],
}
_EXP_WORDS = {
    "negative": ["failed", "bad experience", "regret", "went badly", "burned", "lost money",
                 "didn't work", "did not work", "went wrong", "mistake before"],
    "mixed": ["mixed", "some good", "so-so", "hit or miss", "up and down"],
    "positive": ["worked well", "went well", "successful", "good experience", "positive experience",
                 "did this before and", "worked out before"],
    "none": ["first time", "never done", "no experience", "never tried", "new to"],
}
_RISK_WORDS = {
    "high": ["risky", "high risk", "big risk", "all my savings", "dangerous", "life-changing",
             "scary", "huge risk", "could lose everything"],
    "low": ["safe", "low risk", "small", "minor", "no risk", "harmless"],
    "medium": ["moderate", "medium risk", "some risk"],
}
_TIME_WORDS = {
    "hours": ["today", "tonight", "hours", "right now", "asap", "urgent", "immediately", "in an hour"],
    "days": ["tomorrow", "this week", "few days", "days"],
    "weeks": ["weeks", "next month", "end of the month", "couple of weeks"],
    "months": ["months", "next year", "no rush", "long term", "long-term", "plenty of time"],
}
_CUES = ["should i", "shall i", "whether to", "thinking of", "thinking about", "considering",
         "can't decide", "cannot decide", "help me decide", "or should", "what if i", "worth it",
         "decision", "decide", "choose", "torn between"]


def _first(text: str, table: dict, order=None):
    for key in (order or table):
        if any(re.search(r"\b" + re.escape(w) + r"\b", text) if w.isalpha() else w in text for w in table[key]):
            return key
    return None


def rule_parse(text: str) -> dict:
    t = text.lower()
    scores = {k: sum(w in t for w in ws) for k, ws in _TYPE_WORDS.items()}
    best = max(scores, key=scores.get)
    dtype = best if scores[best] > 0 else None
    cue = any(c in t for c in _CUES)
    choice = _first(t, _CHOICE_WORDS, ["wait", "decline", "compromise", "act_now"])
    if choice is None and "should i" in t:
        choice = "act_now"
    return {"is_decision": cue or dtype is not None, "decision_type": dtype, "choice": choice,
            "previous_experience": _first(t, _EXP_WORDS, ["negative", "mixed", "positive", "none"]),
            "risk_level": _first(t, _RISK_WORDS, ["high", "low", "medium"]),
            "time_available": _first(t, _TIME_WORDS, ["hours", "days", "weeks", "months"]),
            "summary": text.strip()[:80]}


def _valid(v, allowed):
    return v if isinstance(v, str) and v in allowed else None


def understand(text: str, history: list[dict], use_llm: bool = True) -> dict:
    rule = rule_parse(text)
    ai = llm.extract(text, history) if (use_llm and llm.available()) else None
    is_decision = bool(ai.get("is_decision")) if ai else rule["is_decision"]
    out = {"is_decision": is_decision, "used_ai": ai is not None,
           "summary": (ai or {}).get("summary") or rule["summary"], "assumptions": []}
    if not is_decision:
        return out
    allowed = {"decision_type": DECISION_TYPES, "choice": CHOICES, "previous_experience": EXPERIENCE,
               "risk_level": RISK, "time_available": TIME}
    for field, opts in allowed.items():
        val = _valid((ai or {}).get(field), opts) or rule.get(field)
        if val is None and field in DEFAULTS:
            val = DEFAULTS[field]
            out["assumptions"].append(f"{field.replace('_', ' ')} = {val}")
        if val is None and field == "decision_type":
            val = "other"
        out[field] = val
    return out
