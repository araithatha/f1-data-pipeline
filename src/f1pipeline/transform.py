"""Transform step: turn raw API records into clean, analysis-ready tables.

Output is a small star schema saved as CSV:

    dim_race, dim_driver, dim_constructor          (dimensions)
    fact_result, fact_qualifying, fact_pit_stop    (driver x race, 2014+, Jolpica)
    fact_lap, fact_practice                        (lap-level and practice pace, 2023+, OpenF1)

dim_race also carries race-window weather from Open-Meteo for every season.

Every table uses the same keys: ``race_id`` ("2024_01"), ``driver_id`` and
``constructor_id`` (Jolpica's ids, e.g. "max_verstappen", "red_bull").
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

CLEAN_LAP_THRESHOLD = 1.07  # laps slower than 107% of the race median are treated as non-representative


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def race_id(season: int | str, rnd: int | str) -> str:
    return f"{int(season)}_{int(rnd):02d}"


def parse_lap_time(value: Any) -> float:
    """'1:29.708' -> 89.708, '59.1' -> 59.1, '' / None -> NaN."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.nan
    text = str(value).strip()
    if not text:
        return np.nan
    try:
        if ":" in text:
            minutes, seconds = text.split(":", 1)
            return int(minutes) * 60 + float(seconds)
        return float(text)
    except ValueError:
        return np.nan


def _to_int(value: Any) -> float:
    try:
        return int(value)
    except (TypeError, ValueError):
        return np.nan


# --------------------------------------------------------------------------- #
# Jolpica tables
# --------------------------------------------------------------------------- #
def build_dim_race(schedule: list[dict]) -> pd.DataFrame:
    rows = []
    for r in schedule:
        c = r["Circuit"]
        loc = c["Location"]
        ts = f"{r['date']}T{r.get('time', '00:00:00Z')}".replace("Z", "+00:00")
        rows.append(
            {
                "race_id": race_id(r["season"], r["round"]),
                "season": int(r["season"]),
                "round": int(r["round"]),
                "race_name": r["raceName"],
                "race_date": r["date"],
                "race_start_utc": pd.Timestamp(ts),
                "circuit_id": c["circuitId"],
                "circuit_name": c["circuitName"],
                "locality": loc["locality"],
                "country": loc["country"],
                "latitude": float(loc["lat"]),
                "longitude": float(loc["long"]),
            }
        )
    return pd.DataFrame(rows).sort_values("race_id").reset_index(drop=True)


def build_dims_from_results(results: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    drivers, constructors = {}, {}
    for race in sorted(results, key=lambda r: (int(r["season"]), int(r["round"]))):
        for res in race["Results"]:
            d, c = res["Driver"], res["Constructor"]
            # Later seasons overwrite earlier ones, so the dims hold each id's latest info.
            drivers[d["driverId"]] = {
                "driver_id": d["driverId"],
                "driver_code": d.get("code"),
                "permanent_number": _to_int(d.get("permanentNumber")),
                "first_name": d["givenName"],
                "last_name": d["familyName"],
                "full_name": f"{d['givenName']} {d['familyName']}",
                "date_of_birth": d.get("dateOfBirth"),
                "nationality": d.get("nationality"),
            }
            constructors[c["constructorId"]] = {
                "constructor_id": c["constructorId"],
                "constructor_name": c["name"],
                "nationality": c.get("nationality"),
            }
    return (
        pd.DataFrame(drivers.values()).sort_values("driver_id").reset_index(drop=True),
        pd.DataFrame(constructors.values()).sort_values("constructor_id").reset_index(drop=True),
    )


def build_fact_result(results: list[dict]) -> pd.DataFrame:
    rows = []
    for race in results:
        rid = race_id(race["season"], race["round"])
        for res in race["Results"]:
            pos_text = res.get("positionText", "")
            fl = res.get("FastestLap", {})
            rows.append(
                {
                    "race_id": rid,
                    "driver_id": res["Driver"]["driverId"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "car_number": _to_int(res.get("number")),
                    "grid": _to_int(res.get("grid")),
                    "finish_position": _to_int(res.get("position")),
                    "position_text": pos_text,
                    "is_classified": pos_text.isdigit(),
                    "points": float(res.get("points", 0)),
                    "laps_completed": _to_int(res.get("laps")),
                    "status": res.get("status"),
                    "race_time_ms": _to_int(res.get("Time", {}).get("millis")),
                    "fastest_lap_rank": _to_int(fl.get("rank")),
                    "fastest_lap_s": parse_lap_time(fl.get("Time", {}).get("time")),
                }
            )
    df = pd.DataFrame(rows)
    # Grid 0 means a pit-lane start: treat it as starting behind the whole field.
    starters = df.groupby("race_id")["driver_id"].transform("count")
    df["grid_adj"] = np.where(df["grid"] == 0, starters, df["grid"])
    df["positions_gained"] = np.where(
        df["is_classified"], df["grid_adj"] - df["finish_position"], np.nan
    )
    df["is_podium"] = df["finish_position"] <= 3
    df["is_top10"] = df["finish_position"] <= 10
    df["is_win"] = df["finish_position"] == 1
    return df.sort_values(["race_id", "finish_position"]).reset_index(drop=True)


def build_fact_qualifying(qualifying: list[dict]) -> pd.DataFrame:
    rows = []
    for race in qualifying:
        rid = race_id(race["season"], race["round"])
        for q in race["QualifyingResults"]:
            q1, q2, q3 = (parse_lap_time(q.get(k)) for k in ("Q1", "Q2", "Q3"))
            rows.append(
                {
                    "race_id": rid,
                    "driver_id": q["Driver"]["driverId"],
                    "constructor_id": q["Constructor"]["constructorId"],
                    "quali_position": _to_int(q.get("position")),
                    "q1_s": q1,
                    "q2_s": q2,
                    "q3_s": q3,
                    "best_quali_s": np.nanmin([q1, q2, q3]) if not all(map(np.isnan, [q1, q2, q3])) else np.nan,
                    "reached_q3": not np.isnan(q3),
                }
            )
    df = pd.DataFrame(rows)
    pole = df.groupby("race_id")["best_quali_s"].transform("min")
    df["gap_to_pole_pct"] = (df["best_quali_s"] - pole) / pole * 100
    return df.sort_values(["race_id", "quali_position"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# OpenF1 tables
# --------------------------------------------------------------------------- #
def _match_race_id(session: dict, dim_race: pd.DataFrame) -> str | None:
    start = pd.Timestamp(session["date_start"])
    same_season = dim_race[dim_race["season"] == session["year"]]
    if same_season.empty:
        return None
    diffs = (same_season["race_start_utc"] - start).abs()
    best = diffs.idxmin()
    return same_season.loc[best, "race_id"] if diffs[best] <= pd.Timedelta(days=1) else None


def _expand_stints(stints: list[dict]) -> pd.DataFrame:
    rows = []
    for s in stints:
        if s.get("lap_start") is None or s.get("lap_end") is None:
            continue
        age0 = s.get("tyre_age_at_start") or 0
        for lap in range(int(s["lap_start"]), int(s["lap_end"]) + 1):
            rows.append(
                {
                    "driver_number": s["driver_number"],
                    "lap_number": lap,
                    "stint_number": s["stint_number"],
                    "compound": (s.get("compound") or "UNKNOWN").upper(),
                    "tyre_age_laps": age0 + (lap - int(s["lap_start"])),
                    "tyre_age_at_stint_start": age0,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Where stints overlap by a lap (it happens in the feed), keep the later stint.
    return df.sort_values("stint_number").drop_duplicates(["driver_number", "lap_number"], keep="last")


def build_openf1_tables(
    bundles: list[dict], dim_race: pd.DataFrame, fact_result: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (fact_lap, fact_pit_stop, race_weather)."""
    lap_frames, pit_frames, weather_rows = [], [], []
    number_to_driver = fact_result.set_index(["race_id", "car_number"])["driver_id"].to_dict()

    for b in bundles:
        session = b["session"]
        rid = _match_race_id(session, dim_race)
        if rid is None:
            log.warning("No Jolpica race for OpenF1 session %s (%s)", session["session_key"], session["country_name"])
            continue

        laps = pd.DataFrame(b["laps"])
        if laps.empty:
            log.warning("No laps for %s", rid)
            continue
        laps = laps.rename(
            columns={
                "lap_duration": "lap_time_s",
                "duration_sector_1": "sector1_s",
                "duration_sector_2": "sector2_s",
                "duration_sector_3": "sector3_s",
                "i1_speed": "speed_trap_i1_kph",
                "i2_speed": "speed_trap_i2_kph",
                "st_speed": "speed_trap_st_kph",
            }
        )
        keep = [
            "driver_number", "lap_number", "date_start", "lap_time_s", "sector1_s", "sector2_s",
            "sector3_s", "speed_trap_i1_kph", "speed_trap_i2_kph", "speed_trap_st_kph", "is_pit_out_lap",
        ]
        laps = laps[[c for c in keep if c in laps.columns]].copy()
        laps["date_start"] = pd.to_datetime(laps["date_start"], utc=True, errors="coerce", format="ISO8601")

        stints = _expand_stints(b["stints"])
        if not stints.empty:
            laps = laps.merge(stints, on=["driver_number", "lap_number"], how="left")

        pits = pd.DataFrame(b["pit"])
        if not pits.empty:
            pits = pits.dropna(subset=["lap_number"])
            pit_laps = set(zip(pits["driver_number"], pits["lap_number"].astype(int)))
            laps["is_pit_in_lap"] = [(d, l) in pit_laps for d, l in zip(laps["driver_number"], laps["lap_number"])]
            pf = pits.rename(columns={"pit_duration": "pit_lane_time_s", "stop_duration": "stationary_time_s"})
            pf["race_id"] = rid
            pf["driver_id"] = [number_to_driver.get((rid, n)) for n in pf["driver_number"]]
            pf["lap_number"] = pf["lap_number"].astype(int)
            for col in ("pit_lane_time_s", "stationary_time_s"):
                if col not in pf.columns:
                    pf[col] = np.nan
            pit_frames.append(
                pf[["race_id", "driver_id", "driver_number", "lap_number", "pit_lane_time_s", "stationary_time_s"]]
            )
        else:
            laps["is_pit_in_lap"] = False

        weather = pd.DataFrame(b["weather"])
        if not weather.empty:
            weather["date"] = pd.to_datetime(weather["date"], utc=True, format="ISO8601")
            weather = weather.sort_values("date")
            wcols = {"air_temperature": "air_temp_c", "track_temperature": "track_temp_c", "rainfall": "rainfall", "humidity": "humidity_pct", "wind_speed": "wind_speed_ms"}
            w = weather[["date", *wcols]].rename(columns=wcols)
            timed = laps.dropna(subset=["date_start"]).sort_values("date_start")
            untimed = laps[laps["date_start"].isna()]
            timed = pd.merge_asof(timed, w, left_on="date_start", right_on="date", direction="backward").drop(columns="date")
            laps = pd.concat([timed, untimed], ignore_index=True)
            race_window = weather[
                (weather["date"] >= pd.Timestamp(session["date_start"]))
                & (weather["date"] <= pd.Timestamp(session["date_end"]))
            ]
            if race_window.empty:
                race_window = weather
            weather_rows.append(
                {
                    "race_id": rid,
                    "openf1_session_key": session["session_key"],
                    "avg_air_temp_c": round(race_window["air_temperature"].mean(), 1),
                    "avg_track_temp_c": round(race_window["track_temperature"].mean(), 1),
                    "max_track_temp_c": race_window["track_temperature"].max(),
                    "wet_race": bool((race_window["rainfall"] > 0).any()),
                }
            )

        laps["race_id"] = rid
        laps["driver_id"] = [number_to_driver.get((rid, n)) for n in laps["driver_number"]]
        lap_frames.append(laps)

    fact_lap = pd.concat(lap_frames, ignore_index=True) if lap_frames else pd.DataFrame()
    if not fact_lap.empty:
        fact_lap["is_pit_out_lap"] = fact_lap["is_pit_out_lap"].fillna(False).astype(bool)
        fact_lap["is_pit_in_lap"] = fact_lap["is_pit_in_lap"].fillna(False).astype(bool)
        median = fact_lap.groupby("race_id")["lap_time_s"].transform("median")
        fact_lap["is_clean_lap"] = (
            fact_lap["lap_time_s"].notna()
            & (fact_lap["lap_number"] > 1)
            & ~fact_lap["is_pit_out_lap"]
            & ~fact_lap["is_pit_in_lap"]
            & (fact_lap["lap_time_s"] <= median * CLEAN_LAP_THRESHOLD)
        )
        for col in ("stint_number", "tyre_age_laps", "tyre_age_at_stint_start",
                    "speed_trap_i1_kph", "speed_trap_i2_kph", "speed_trap_st_kph"):
            if col in fact_lap:
                fact_lap[col] = pd.to_numeric(fact_lap[col], errors="coerce").round().astype("Int64")
        fact_lap["date_start"] = fact_lap["date_start"].dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        lead = ["race_id", "driver_id", "driver_number", "lap_number"]
        fact_lap = fact_lap[lead + [c for c in fact_lap.columns if c not in lead]]
        fact_lap = fact_lap.sort_values(["race_id", "driver_number", "lap_number"]).reset_index(drop=True)

    fact_pit = pd.concat(pit_frames, ignore_index=True) if pit_frames else pd.DataFrame()
    if not fact_pit.empty:
        fact_pit = fact_pit.sort_values(["race_id", "driver_number", "lap_number"]).reset_index(drop=True)

    return fact_lap, fact_pit, pd.DataFrame(weather_rows)


# --------------------------------------------------------------------------- #
# Pit stops (Jolpica, 2014+), weather (Open-Meteo, 2014+), practice (OpenF1, 2023+)
# --------------------------------------------------------------------------- #
def build_fact_pit_stop(pitstops: list[dict]) -> pd.DataFrame:
    rows = []
    for race in pitstops:
        rid = race_id(race["season"], race["round"])
        for s in race["stops"]:
            rows.append(
                {
                    "race_id": rid,
                    "driver_id": s["driverId"],
                    "stop_number": _to_int(s.get("stop")),
                    "lap_number": _to_int(s.get("lap")),
                    "pit_lane_time_s": parse_lap_time(s.get("duration")),
                }
            )
    df = pd.DataFrame(rows, columns=["race_id", "driver_id", "stop_number", "lap_number", "pit_lane_time_s"])
    # Pit-lane length differs by circuit, so compare each stop with that race's median stop.
    # Stops more than 2x the median are red-flag or repair stops, not normal service.
    median = df.groupby("race_id")["pit_lane_time_s"].transform("median")
    df["pit_time_vs_race_median"] = df["pit_lane_time_s"] / median
    df["is_normal_stop"] = df["pit_time_vs_race_median"].between(0.5, 2.0)
    return df.sort_values(["race_id", "driver_id", "stop_number"]).reset_index(drop=True)


WET_RACE_PRECIP_MM = 0.5  # total rain over the race window that counts as a wet race


def build_race_weather(weather: list[dict]) -> pd.DataFrame:
    """Summarise hourly weather over the ~2-hour race window (start hour to start + 2h)."""
    rows = []
    for w in weather:
        hourly = w["data"].get("hourly") if isinstance(w["data"], dict) else None
        if not hourly:
            continue
        hw = pd.DataFrame(hourly)
        hw["time"] = pd.to_datetime(hw["time"], utc=True)
        start = pd.Timestamp(w["start"].replace("Z", "+00:00")).floor("h")
        window = hw[(hw["time"] >= start) & (hw["time"] <= start + pd.Timedelta(hours=2))]
        if window.empty:
            continue
        precip = float(window["precipitation"].sum())
        rows.append(
            {
                "race_id": race_id(w["season"], w["round"]),
                "wx_air_temp_c": round(window["temperature_2m"].mean(), 1),
                "wx_precip_mm": round(precip, 2),
                "wx_humidity_pct": round(window["relative_humidity_2m"].mean(), 1),
                "wx_wind_kph": round(window["wind_speed_10m"].mean(), 1),
                "wx_wet_race": precip >= WET_RACE_PRECIP_MM,
            }
        )
    return pd.DataFrame(rows)


def build_fact_practice(
    practice: list[dict], meeting_to_race: dict[int, str], fact_result: pd.DataFrame
) -> pd.DataFrame:
    """One row per driver per practice session: best lap and long-run pace.

    Long-run pace is the median of a driver's representative laps (within 107%
    of their own best, not pit-out laps). It needs at least 5 such laps; teams
    run race simulations on heavier fuel, so this reads closer to race pace than
    a single fastest lap.
    """
    number_to_driver = fact_result.set_index(["race_id", "car_number"])["driver_id"].to_dict()
    rows = []
    for p in practice:
        s = p["session"]
        rid = meeting_to_race.get(s["meeting_key"])
        laps = pd.DataFrame(p["laps"])
        if rid is None or laps.empty or "lap_duration" not in laps:
            continue
        laps = laps.dropna(subset=["lap_duration"])
        if "is_pit_out_lap" in laps:
            laps = laps[~laps["is_pit_out_lap"].fillna(False).astype(bool)]
        for number, g in laps.groupby("driver_number"):
            best = g["lap_duration"].min()
            rep = g[g["lap_duration"] <= best * 1.07]["lap_duration"]
            rows.append(
                {
                    "race_id": rid,
                    "driver_id": number_to_driver.get((rid, number)),
                    "driver_number": number,
                    "session_name": s["session_name"],
                    "laps_completed": len(g),
                    "best_lap_s": best,
                    "longrun_median_s": rep.median() if len(rep) >= 5 else np.nan,
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Stand-in drivers who ran only in practice don't map to a race entry; drop them.
    df = df.dropna(subset=["driver_id"])
    grp = df.groupby(["race_id", "session_name"])
    df["best_lap_gap_pct"] = (df["best_lap_s"] / grp["best_lap_s"].transform("min") - 1) * 100
    df["longrun_gap_pct"] = (df["longrun_median_s"] / grp["longrun_median_s"].transform("min") - 1) * 100
    df["best_lap_rank"] = grp["best_lap_s"].rank(method="min")
    return df.sort_values(["race_id", "session_name", "best_lap_rank"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def transform_all(raw: dict[str, Any]) -> dict[str, pd.DataFrame]:
    dim_race = build_dim_race(raw["schedule"])
    dim_driver, dim_constructor = build_dims_from_results(raw["results"])
    fact_result = build_fact_result(raw["results"])
    fact_qualifying = build_fact_qualifying(raw["qualifying"])
    # OpenF1 pit data only covers 2023+, so pit stops come from Jolpica (2014+) instead.
    fact_lap, _openf1_pits, race_weather = build_openf1_tables(raw["openf1"], dim_race, fact_result)
    fact_pit_stop = build_fact_pit_stop(raw["pitstops"])

    meeting_to_race = {}
    for b in raw["openf1"]:
        rid = _match_race_id(b["session"], dim_race)
        if rid:
            meeting_to_race[b["session"]["meeting_key"]] = rid
    fact_practice = build_fact_practice(raw["practice"], meeting_to_race, fact_result)

    # Only keep races that have actually been run, then attach race-level weather:
    # OpenF1 track sensors (2023+) and Open-Meteo reanalysis (all seasons).
    dim_race = dim_race[dim_race["race_id"].isin(fact_result["race_id"])]
    if not race_weather.empty:
        dim_race = dim_race.merge(race_weather, on="race_id", how="left")
    wx = build_race_weather(raw["weather"])
    if not wx.empty:
        dim_race = dim_race.merge(wx, on="race_id", how="left")
    dim_race["race_start_utc"] = dim_race["race_start_utc"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    return {
        "dim_race": dim_race.reset_index(drop=True),
        "dim_driver": dim_driver,
        "dim_constructor": dim_constructor,
        "fact_result": fact_result,
        "fact_qualifying": fact_qualifying,
        "fact_lap": fact_lap,
        "fact_pit_stop": fact_pit_stop,
        "fact_practice": fact_practice,
    }
