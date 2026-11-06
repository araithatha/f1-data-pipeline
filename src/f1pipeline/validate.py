"""Data quality checks run at the end of every pipeline run.

Hard checks (keys, nulls, impossible values) fail the run. Soft checks
(coverage ratios) are reported so drift in the source APIs is visible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger(__name__)

PRIMARY_KEYS = {
    "dim_race": ["race_id"],
    "dim_driver": ["driver_id"],
    "dim_constructor": ["constructor_id"],
    "fact_result": ["race_id", "driver_id"],
    "fact_qualifying": ["race_id", "driver_id"],
    "fact_lap": ["race_id", "driver_number", "lap_number"],
    "fact_pit_stop": ["race_id", "driver_id", "stop_number"],
    "fact_practice": ["race_id", "driver_id", "session_name"],
}

# Facts anyone can look up, used as end-to-end sanity checks.
KNOWN_WINNERS = {
    "2023_01": "max_verstappen",  # 2023 Bahrain GP
    "2024_01": "max_verstappen",  # 2024 Bahrain GP
    "2024_02": "max_verstappen",  # 2024 Saudi Arabian GP
    "2024_03": "sainz",           # 2024 Australian GP
    "2024_06": "norris",          # 2024 Miami GP
    "2019_21": "hamilton",        # 2019 Abu Dhabi GP
}


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    hard: bool = True


def run_checks(tables: dict[str, pd.DataFrame]) -> list[Check]:
    checks: list[Check] = []

    for table, keys in PRIMARY_KEYS.items():
        df = tables.get(table)
        if df is None or df.empty:
            checks.append(Check(f"{table} not empty", False, "table is empty"))
            continue
        dupes = int(df.duplicated(keys).sum())
        nulls = int(df[keys].isna().any(axis=1).sum())
        checks.append(Check(f"{table} primary key unique", dupes == 0, f"{dupes} duplicate rows on {keys}"))
        checks.append(Check(f"{table} keys not null", nulls == 0, f"{nulls} rows with null keys"))

    fr = tables["fact_result"]
    winners = fr[fr["finish_position"] == 1].set_index("race_id")["driver_id"]
    for rid, expected in KNOWN_WINNERS.items():
        if rid in winners.index:
            got = winners[rid]
            checks.append(Check(f"winner {rid}", got == expected, f"expected {expected}, got {got}"))

    one_winner = fr[fr["finish_position"] == 1].groupby("race_id").size()
    checks.append(Check("exactly one winner per race", bool((one_winner == 1).all()), f"{int((one_winner != 1).sum())} races off"))

    orphans = set(fr["race_id"]) - set(tables["dim_race"]["race_id"])
    checks.append(Check("every result has a race", not orphans, f"{len(orphans)} orphan race_ids"))

    laps = tables.get("fact_lap")
    if laps is not None and not laps.empty:
        bad_times = int((laps["lap_time_s"] <= 0).sum())
        checks.append(Check("lap times positive", bad_times == 0, f"{bad_times} non-positive lap times"))

        mapped = laps["driver_id"].notna().mean()
        checks.append(Check("laps mapped to a driver", mapped >= 0.99, f"{mapped:.2%} mapped", hard=False))

        if "compound" in laps:
            with_tyre = laps["compound"].notna().mean()
            checks.append(Check("laps with tyre compound", with_tyre >= 0.95, f"{with_tyre:.2%} have a compound", hard=False))

        clean = laps["is_clean_lap"].mean()
        checks.append(Check("share of clean laps", 0.6 <= clean <= 0.98, f"{clean:.1%} flagged clean", hard=False))

        # Disqualified drivers are recorded with 0 laps completed, so leave them out.
        eligible = fr[fr["status"] != "Disqualified"][["race_id", "driver_id", "laps_completed"]]
        max_lap = laps.merge(eligible, on=["race_id", "driver_id"], how="inner")
        over = int((max_lap["lap_number"] > max_lap["laps_completed"] + 1).sum())
        checks.append(Check("lap numbers within laps completed", over == 0, f"{over} laps beyond the driver's race distance", hard=False))

    dr = tables["dim_race"]
    if "wx_wet_race" in dr:
        cov = dr["wx_precip_mm"].notna().mean()
        checks.append(Check("races with Open-Meteo weather", cov >= 0.98, f"{cov:.1%} covered", hard=False))
        both = dr.dropna(subset=["wet_race", "wx_wet_race"]) if "wet_race" in dr else pd.DataFrame()
        if not both.empty:
            agree = (both["wet_race"].astype(bool) == both["wx_wet_race"].astype(bool)).mean()
            checks.append(Check("Open-Meteo vs track rain sensor agree (2023+)", agree >= 0.85,
                                f"{agree:.1%} of {len(both)} races agree", hard=False))

    ps = tables.get("fact_pit_stop")
    if ps is not None and not ps.empty:
        races_with_stops = ps["race_id"].nunique() / dr["race_id"].nunique()
        checks.append(Check("races with pit-stop data", races_with_stops >= 0.95, f"{races_with_stops:.1%} of races", hard=False))

    return checks


def report(checks: list[Check]) -> str:
    lines = ["| Check | Result | Detail |", "|---|---|---|"]
    for c in checks:
        status = "PASS" if c.passed else ("FAIL" if c.hard else "WARN")
        lines.append(f"| {c.name} | {status} | {c.detail} |")
    return "\n".join(lines)
