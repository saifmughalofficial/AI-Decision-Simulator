"""Similarity search + Bayesian forecast (no LLM needed, pure maths).

Pipeline:  rows -> feature vectors -> nearest past decisions (FAISS or scikit-learn)
           -> similarity/recency/own-data weights -> Beta posterior -> forecast + confidence
"""
from dataclasses import asdict, dataclass
from datetime import date, datetime

import numpy as np
from scipy.stats import beta as beta_dist

from .schema import (CHOICES, DECISION_TYPES, EXPERIENCE, RISK, SUCCESS_THRESHOLD, TIME)

W = {"type": 1.0, "choice": 1.5, "exp": 1.2, "risk": 1.0, "time": 0.8}
_ORD = {
    "exp": {"negative": 0.0, "mixed": 0.4, "none": 0.55, "positive": 1.0},
    "risk": {"low": 0.0, "medium": 0.5, "high": 1.0},
    "time": {"hours": 0.0, "days": 0.33, "weeks": 0.67, "months": 1.0},
}


def _onehot(v, options):
    a = np.zeros(len(options), dtype=np.float32)
    if v in options:
        a[options.index(v)] = 1.0
    return a


def encode(dtype, choice, exp, risk, time) -> np.ndarray:
    return np.concatenate([
        _onehot(dtype, DECISION_TYPES) * W["type"],
        _onehot(choice, CHOICES) * W["choice"],
        np.array([_ORD["exp"].get(exp, .55) * W["exp"],
                  _ORD["risk"].get(risk, .5) * W["risk"],
                  _ORD["time"].get(time, .67) * W["time"]], dtype=np.float32),
    ]).astype(np.float32)


@dataclass
class Forecast:
    p: float
    lo: float
    hi: float
    exp_score: float
    n_used: int
    eff_n: float
    avg_sim: float
    confidence: str
    conf_score: float
    verdict: str
    neighbors: list

    def to_dict(self):
        return asdict(self)


def verdict_for(p: float) -> str:
    return "Favourable" if p >= 0.65 else ("Uncertain" if p >= 0.45 else "Risky")


class Engine:
    def __init__(self, rows: list[dict]):
        if len(rows) < 10:
            raise ValueError("Need at least 10 past decisions in data/decisions.csv")
        self.rows = rows
        self.n = len(rows)
        self.X = np.stack([encode(r["decision_type"], r["choice"], r["previous_experience"],
                                  r["risk_level"], r["time_available"]) for r in rows])
        self.score = np.array([r["outcome_score"] for r in rows], dtype=float)
        self.success = (self.score >= SUCCESS_THRESHOLD).astype(float)
        today = date.today()
        ages = []
        for r in rows:
            try:
                ages.append(max((today - datetime.strptime(str(r["date"])[:10], "%Y-%m-%d").date()).days, 0))
            except Exception:
                ages.append(0)
        self.recency = 0.5 ** (np.array(ages) / 1095.0)          # half-life: 3 years
        self.boost = np.array([2.0 if str(r["decision_id"]).startswith("U") else 1.0 for r in rows])
        self.backend = "numpy"
        self._faiss = self._nn = None
        try:
            import faiss  # optional, faster on big data
            self._faiss = faiss.IndexFlatL2(self.X.shape[1])
            self._faiss.add(self.X)
            self.backend = "FAISS"
        except Exception:
            from sklearn.neighbors import NearestNeighbors
            self._nn = NearestNeighbors(algorithm="brute", metric="euclidean").fit(self.X)
            self.backend = "scikit-learn"

    # ---------- search ----------
    def _search(self, q, k, exclude=None):
        kk = min(k + (1 if exclude is not None else 0), self.n)
        if self._faiss is not None:
            d2, idx = self._faiss.search(q[None, :], kk)
            d2, idx = d2[0], idx[0]
        else:
            d, idx = self._nn.kneighbors(q[None, :], n_neighbors=kk)
            d2, idx = d[0] ** 2, idx[0]
        if exclude is not None:
            keep = idx != exclude
            d2, idx = d2[keep], idx[keep]
        return idx[:k], d2[:k]

    # ---------- forecast ----------
    def _core(self, vec, k=60, exclude=None, want_neighbors=False) -> Forecast:
        idx, d2 = self._search(vec, k, exclude)
        sim = np.exp(-d2 / 1.5)
        w = sim * self.recency[idx] * self.boost[idx]
        s = self.success[idx]
        a, b = 1.0 + float((w * s).sum()), 1.0 + float((w * (1 - s)).sum())
        p = a / (a + b)
        lo, hi = (float(x) for x in beta_dist.ppf([0.05, 0.95], a, b))
        sw = float(w.sum())
        exp_score = float((w * self.score[idx]).sum() / sw) if sw > 0 else 50.0
        eff = sw ** 2 / float((w ** 2).sum()) if sw > 0 else 0.0
        avg_sim = float(sim[:5].mean()) if len(sim) else 0.0

        c = float(np.clip(1 - (hi - lo) / 0.55, 0, 1)) * min(1.0, eff / 10.0)
        label = "High" if c >= 0.5 else ("Medium" if c >= 0.28 else "Low")
        if 0.40 <= p <= 0.60 or avg_sim < 0.20:      # mixed evidence or weak matches -> low confidence
            label = "Low"
        nb = []
        if want_neighbors:
            for j in np.argsort(-w)[:6]:
                r = self.rows[int(idx[j])]
                nb.append({"id": r["decision_id"], "type": r["decision_type"], "choice": r["choice"],
                           "experience": r["previous_experience"], "risk": r["risk_level"],
                           "time": r["time_available"], "outcome": r["actual_result"],
                           "score": r["outcome_score"], "similarity_%": round(float(sim[j]) * 100),
                           "context": r["context"], "date": str(r["date"])[:10]})
        return Forecast(p, lo, hi, exp_score, int(len(idx)), eff, avg_sim, label, c, verdict_for(p), nb)

    def forecast(self, dtype, choice, exp, risk, time, k=60) -> Forecast:
        return self._core(encode(dtype, choice, exp, risk, time), k, want_neighbors=True)

    def compare(self, dtype, exp, risk, time, k=60) -> dict[str, Forecast]:
        return {c: self._core(encode(dtype, c, exp, risk, time), k) for c in CHOICES}

    @staticmethod
    def rank_key(f: Forecast) -> float:          # prefer good AND certain outcomes
        return f.p - 0.25 * (f.hi - f.lo)

    def best_choice(self, cmp: dict[str, Forecast]) -> str:
        return max(cmp, key=lambda c: self.rank_key(cmp[c]))

    # ---------- honesty check ----------
    def backtest(self, n=300, seed=0, k=60) -> dict:
        """Leave-one-out test: predict past decisions from all the OTHER decisions."""
        rng = np.random.default_rng(seed)
        ids = rng.choice(self.n, min(n, self.n), replace=False)
        ps = np.array([self._core(self.X[i], k, exclude=int(i)).p for i in ids])
        ys = self.success[ids]
        acc = float(((ps >= 0.5) == (ys == 1)).mean())
        base = float(max(ys.mean(), 1 - ys.mean()))
        return {"tested": int(len(ids)), "accuracy": acc, "baseline": base,
                "brier": float(((ps - ys) ** 2).mean())}
