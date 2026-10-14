"""Shared vocabulary: allowed values, labels and column order."""

DECISION_TYPES = ["career", "finance", "education", "health", "relationships",
                  "purchase", "travel", "productivity", "other"]
CHOICES = ["act_now", "wait", "decline", "compromise"]
EXPERIENCE = ["none", "negative", "mixed", "positive"]
RISK = ["low", "medium", "high"]
TIME = ["hours", "days", "weeks", "months"]
RESULTS = ["negative", "mixed", "positive"]

SUCCESS_THRESHOLD = 60  # outcome_score >= 60 counts as "went well"

COLUMNS = ["decision_id", "decision_type", "context", "choice", "previous_experience",
           "risk_level", "time_available", "expected_result", "actual_result",
           "satisfaction", "outcome_score", "date"]

CHOICE_LABELS = {
    "act_now": "Go ahead now",
    "wait": "Wait / postpone",
    "decline": "Decline / skip",
    "compromise": "Middle path / trial run",
}
TYPE_ICONS = {"career": "💼", "finance": "💰", "education": "🎓", "health": "🩺",
              "relationships": "❤️", "purchase": "🛒", "travel": "✈️",
              "productivity": "⏱️", "other": "🧩"}
DEFAULTS = {"previous_experience": "none", "risk_level": "medium", "time_available": "weeks"}
