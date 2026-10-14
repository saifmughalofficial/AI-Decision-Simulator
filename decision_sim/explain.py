"""Rule-based explanation engine: template sentences driven by the numbers. Works offline, free."""
from .engine import Forecast
from .schema import CHOICE_LABELS


def _pct(x): return f"{round(x * 100)}%"


def rule_summary(fc: Forecast) -> str:
    if fc.p >= 0.65:
        s = "Similar past decisions usually produced a positive outcome."
    elif fc.p >= 0.45:
        s = "Similar past decisions gave mixed results, so this could go either way."
    else:
        s = "Similar past decisions often ended badly, so this option carries real risk."
    return s


def _confidence_line(fc: Forecast) -> str:
    return {
        "High": "Confidence is high: many close matches point in the same direction.",
        "Medium": "Confidence is medium: the pattern is fairly clear, but not overwhelming.",
        "Low": ("Confidence is low: the past results are mixed or there are few close matches, "
                "so treat this forecast as a rough guide only."),
    }[fc.confidence]


def _tips(exp, risk, time, choice) -> list[str]:
    t = []
    if exp == "negative":
        t.append("You had a bad experience with something similar before. Write down what went wrong "
                 "then, and check that it's actually different this time.")
    elif exp == "positive":
        t.append("A positive past experience is one of the strongest predictors of success. That helps you here.")
    elif exp == "none":
        t.append("Since this is new to you, a small test or advice from someone experienced lowers the uncertainty.")
    if risk == "high":
        t.append("Risk is high: decide the maximum loss you can accept *before* you commit, and set an exit point.")
    elif risk == "low":
        t.append("Risk is low, so the cost of being wrong is small. Don't over-analyse it.")
    if time == "hours":
        t.append("You have very little time. Rushed choices score lower on average, so take even 10 minutes to list pros and cons.")
    elif time == "months":
        t.append("You have plenty of time. Use it to gather information or run a small trial first.")
    if choice == "wait" and time == "hours":
        t.append("Waiting when time is nearly gone usually means the decision gets made for you.")
    return t


def build(f: dict, fc: Forecast, cmp: dict[str, Forecast], best: str, assumptions: list[str]) -> str:
    lines = [f"**{rule_summary(fc)}**", ""]
    if f.get("choice"):
        lines.append(
            f"I compared *{CHOICE_LABELS[f['choice']]}* with the {fc.n_used} closest past decisions "
            f"(effective evidence ≈ {fc.eff_n:.0f} decisions). About **{_pct(fc.p)}** ended well "
            f"(score ≥ 60), with an expected score of **{fc.exp_score:.0f}/100**. "
            f"The likely range is {_pct(fc.lo)}–{_pct(fc.hi)}.")
        lines += ["", _confidence_line(fc), ""]
    lines.append("**Your options compared** (chance of a good outcome):")
    for c, x in sorted(cmp.items(), key=lambda kv: -kv[1].p):
        mark = " ⭐ best overall" if c == best else ""
        lines.append(f"- {CHOICE_LABELS[c]}: {_pct(x.p)} (confidence {x.confidence}){mark}")
    if f.get("choice") and best != f["choice"]:
        gap = cmp[best].p - cmp[f["choice"]].p
        if gap >= 0.08:
            lines += ["", f"💡 Your history suggests **{CHOICE_LABELS[best]}** may work better "
                          f"(+{round(gap * 100)} points) than your current leaning."]
    tips = _tips(f["previous_experience"], f["risk_level"], f["time_available"], f.get("choice"))
    if tips:
        lines += ["", "**Things to keep in mind**"] + [f"- {t}" for t in tips]
    if assumptions:
        lines += ["", "*Assumed because you didn't say: " + "; ".join(assumptions) + ".*"]
    return "\n".join(lines)
