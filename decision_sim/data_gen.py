"""Creates a realistic SAMPLE dataset (decisions.csv) so the app works on day one.

The outcomes are simulated from sensible rules (experience, risk, timing) plus noise.
Replace/extend it with your own real decisions using the 'Log outcome' page.
"""
import math
from datetime import date, timedelta

import numpy as np

from .schema import (CHOICES, DECISION_TYPES, EXPERIENCE, RISK, TIME, SUCCESS_THRESHOLD)

_CONTEXT = {
    "career": ["Job offer at a new company", "Asked to lead a team", "Considering a career switch"],
    "finance": ["Investing a portion of savings", "Taking a small loan", "Starting a side business"],
    "education": ["Enrolling in an online course", "Applying for a master's degree", "Changing my study plan"],
    "health": ["Starting a new fitness routine", "Changing my diet", "Booking a health check-up"],
    "relationships": ["Having a serious conversation", "Moving in together", "Reconnecting with an old friend"],
    "purchase": ["Buying a new laptop", "Replacing my phone", "Renting a bigger place"],
    "travel": ["Planning a trip abroad", "Relocating to another city", "Accepting a work trip"],
    "productivity": ["Adopting a new daily schedule", "Taking on an extra project", "Dropping a low-value task"],
    "other": ["A personal decision", "A family decision", "A general life choice"],
}
_EXP = {"none": 0.0, "negative": -0.9, "mixed": -0.1, "positive": 0.9}
_RISK_CHOICE = {
    "act_now": {"low": 0.5, "medium": -0.1, "high": -0.9},
    "wait": {"low": -0.2, "medium": 0.3, "high": 0.5},
    "decline": {"low": -0.6, "medium": 0.0, "high": 0.4},
    "compromise": {"low": 0.1, "medium": 0.4, "high": 0.3},
}
_TIME_CHOICE = {
    "act_now": {"hours": -0.3, "days": 0.0, "weeks": 0.1, "months": 0.2},
    "wait": {"hours": -0.8, "days": -0.3, "weeks": 0.2, "months": 0.4},
    "decline": {"hours": 0.1, "days": 0.0, "weeks": 0.0, "months": -0.1},
    "compromise": {"hours": -0.4, "days": 0.1, "weeks": 0.3, "months": 0.3},
}


def generate(n: int = 1000, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    type_bias = {t: float(rng.normal(0, 0.25)) for t in DECISION_TYPES}
    today = date.today()
    rows = []
    for i in range(1, n + 1):
        t = str(rng.choice(DECISION_TYPES, p=[.18, .14, .11, .11, .10, .13, .08, .10, .05]))
        c = str(rng.choice(CHOICES, p=[.38, .22, .18, .22]))
        e = str(rng.choice(EXPERIENCE, p=[.30, .20, .20, .30]))
        r = str(rng.choice(RISK, p=[.35, .40, .25]))
        tm = str(rng.choice(TIME, p=[.12, .28, .35, .25]))
        z = 0.1 + 1.8 * (type_bias[t] + _EXP[e] + _RISK_CHOICE[c][r] + _TIME_CHOICE[c][tm])
        p = 1 / (1 + math.exp(-z))
        good = rng.random() < p
        score = int(np.clip(rng.normal(76, 10) if good else rng.normal(42, 13), 0, 100))
        sat = int(np.clip(round(score / 10 + rng.normal(0, 1)), 1, 10))
        guess = rng.random()
        expected = "positive" if guess < 0.55 else ("mixed" if guess < 0.8 else "negative")
        actual = "positive" if score >= SUCCESS_THRESHOLD else ("mixed" if score >= 45 else "negative")
        d = today - timedelta(days=int(rng.integers(5, 1100)))
        rows.append({
            "decision_id": f"D{i:04d}", "decision_type": t,
            "context": str(rng.choice(_CONTEXT[t])), "choice": c, "previous_experience": e,
            "risk_level": r, "time_available": tm, "expected_result": expected,
            "actual_result": actual, "satisfaction": sat, "outcome_score": score,
            "date": d.isoformat(),
        })
    return rows
