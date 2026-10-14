"""Reads/writes decisions.csv with Polars (atomic writes, validated rows)."""
import os

import polars as pl

from . import config
from .data_gen import generate
from .schema import (CHOICES, COLUMNS, DECISION_TYPES, EXPERIENCE, RESULTS, RISK, TIME)


def ensure_data() -> None:
    if not config.DATA_PATH.exists():
        config.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        _write(pl.DataFrame(generate()))


def _write(df: pl.DataFrame) -> None:
    tmp = config.DATA_PATH.with_suffix(".tmp")
    df.select(COLUMNS).write_csv(tmp)
    os.replace(tmp, config.DATA_PATH)


def load() -> pl.DataFrame:
    """Load and validate. Bad rows are dropped instead of crashing the app."""
    ensure_data()
    df = pl.read_csv(config.DATA_PATH, schema_overrides={
        "outcome_score": pl.Int64, "satisfaction": pl.Int64})
    return df.filter(
        pl.col("decision_type").is_in(DECISION_TYPES)
        & pl.col("choice").is_in(CHOICES)
        & pl.col("previous_experience").is_in(EXPERIENCE)
        & pl.col("risk_level").is_in(RISK)
        & pl.col("time_available").is_in(TIME)
        & pl.col("outcome_score").is_between(0, 100)
    )


def rows() -> list[dict]:
    return load().to_dicts()


def append_decision(row: dict) -> str:
    """Add one real, user-logged decision. IDs starting with 'U' get extra weight."""
    if row.get("actual_result") not in RESULTS:
        raise ValueError("actual_result must be negative/mixed/positive")
    df = load()
    row = dict(row)
    row["decision_id"] = f"U{df.height + 1:04d}"
    new = pl.DataFrame([row], schema=df.schema)
    _write(pl.concat([df, new]))
    return row["decision_id"]
