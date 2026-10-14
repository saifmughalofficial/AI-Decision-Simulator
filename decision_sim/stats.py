"""Summary tables. Uses DuckDB (SQL) and falls back to plain Python if unavailable."""
from collections import defaultdict

from .schema import SUCCESS_THRESHOLD

_ALLOWED = {"decision_type", "choice", "previous_experience", "risk_level", "time_available"}


def summary_by(rows: list[dict], col: str) -> list[dict]:
    """Success rate and average score grouped by one column."""
    if col not in _ALLOWED:
        raise ValueError("unsupported column")
    try:
        import duckdb
        import polars as pl
        con = duckdb.connect()
        con.register("d", pl.DataFrame(rows).to_arrow())
        q = (f"select {col} as group_name, count(*) as decisions, "
             f"round(avg(outcome_score),1) as avg_score, "
             f"round(100*avg(cast(outcome_score >= {SUCCESS_THRESHOLD} as int)),1) as success_pct "
             f"from d group by 1 order by success_pct desc")
        return con.execute(q).pl().to_dicts()
    except Exception:
        g = defaultdict(list)
        for r in rows:
            g[r[col]].append(r["outcome_score"])
        out = [{"group_name": k, "decisions": len(v), "avg_score": round(sum(v) / len(v), 1),
                "success_pct": round(100 * sum(x >= SUCCESS_THRESHOLD for x in v) / len(v), 1)}
               for k, v in g.items()]
        return sorted(out, key=lambda x: -x["success_pct"])
