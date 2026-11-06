"""Extract step: pull raw records from the Jolpica-F1 and OpenF1 APIs.

Jolpica-F1 (successor to the Ergast API) supplies the historical, relational
side: schedule, circuits, race results and qualifying, back to 1950.

OpenF1 supplies lap-level timing for 2023 onward: lap and sector times, tyre
stints, pit stops, weather, practice sessions and the driver/car-number mapping.

Open-Meteo's historical archive (ERA5 reanalysis) supplies race-day weather at
each circuit for every season, so weather is available before 2023 too.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from .http import CachedClient

log = logging.getLogger(__name__)

JOLPICA_PAGE = 100


# --------------------------------------------------------------------------- #
# Jolpica-F1
# --------------------------------------------------------------------------- #
def _jolpica_paged(client: CachedClient, path: str, table: str, list_key: str) -> list[dict]:
    """Fetch every page of a Jolpica endpoint and merge race-nested rows.

    Jolpica paginates on the inner rows (e.g. individual results), so a single
    race can be split across two pages. We merge pages back together by round.
    """
    races: dict[str, dict] = {}
    offset = 0
    while True:
        payload = client.get(path, {"limit": JOLPICA_PAGE, "offset": offset})
        mr = payload["MRData"]
        for race in mr[table]["Races"]:
            existing = races.setdefault(race["round"], {**race, list_key: []})
            existing[list_key].extend(race.get(list_key, []))
        offset += JOLPICA_PAGE
        if offset >= int(mr["total"]):
            break
    return list(races.values())


def jolpica_schedule(client: CachedClient, season: int) -> list[dict]:
    payload = client.get(f"{season}/races.json", {"limit": JOLPICA_PAGE})
    return payload["MRData"]["RaceTable"]["Races"]


def jolpica_results(client: CachedClient, season: int) -> list[dict]:
    return _jolpica_paged(client, f"{season}/results.json", "RaceTable", "Results")


def jolpica_qualifying(client: CachedClient, season: int) -> list[dict]:
    return _jolpica_paged(client, f"{season}/qualifying.json", "RaceTable", "QualifyingResults")


def jolpica_pitstops(client: CachedClient, season: int, rnd: int) -> list[dict]:
    """Every pit stop in one race (Jolpica has pit data from 2011)."""
    races = _jolpica_paged(client, f"{season}/{rnd}/pitstops.json", "RaceTable", "PitStops")
    return races[0]["PitStops"] if races else []


# --------------------------------------------------------------------------- #
# Open-Meteo (historical weather, ERA5 reanalysis)
# --------------------------------------------------------------------------- #
WEATHER_VARS = "temperature_2m,precipitation,relative_humidity_2m,wind_speed_10m"


def openmeteo_race_weather(client: CachedClient, lat: float, lon: float, race_start_utc: str) -> dict:
    """Hourly weather at the circuit on race day (UTC), from the Open-Meteo archive."""
    start = datetime.fromisoformat(race_start_utc.replace("Z", "+00:00"))
    day = start.date().isoformat()
    next_day = (start + timedelta(days=1)).date().isoformat()
    return client.get(
        "archive",
        {
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "start_date": day,
            "end_date": next_day,
            "hourly": WEATHER_VARS,
            "timezone": "GMT",
        },
    )


# --------------------------------------------------------------------------- #
# OpenF1
# --------------------------------------------------------------------------- #
def openf1_race_sessions(client: CachedClient, season: int) -> list[dict]:
    """Completed, non-cancelled Grand Prix race sessions (sprints excluded)."""
    sessions = client.get("sessions", {"year": season, "session_name": "Race"})
    now = datetime.now(timezone.utc)
    done = []
    for s in sessions:
        end = datetime.fromisoformat(s["date_end"])
        if s.get("is_cancelled") or end > now:
            continue
        done.append(s)
    return done


def openf1_practice_sessions(client: CachedClient, season: int) -> list[dict]:
    """Completed free-practice sessions (FP1-FP3); pre-season testing days excluded."""
    sessions = client.get("sessions", {"year": season, "session_type": "Practice"})
    now = datetime.now(timezone.utc)
    return [
        s for s in sessions
        if s["session_name"].startswith("Practice")
        and not s.get("is_cancelled")
        and datetime.fromisoformat(s["date_end"]) < now
    ]


OPENF1_SESSION_ENDPOINTS = ("drivers", "laps", "stints", "pit", "weather")


def openf1_session_bundle(client: CachedClient, session_key: int) -> dict[str, list[dict]]:
    """All lap-level tables for one race session."""
    return {
        name: client.get(name, {"session_key": session_key}) for name in OPENF1_SESSION_ENDPOINTS
    }


def extract_all(
    jolpica: CachedClient,
    openf1: CachedClient,
    openmeteo: CachedClient,
    history_seasons: list[int],
    timing_seasons: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schedule": [], "results": [], "qualifying": [], "pitstops": [],
        "weather": [], "openf1": [], "practice": [],
    }

    for season in history_seasons:
        log.info("Jolpica %s: schedule, results, qualifying, pit stops", season)
        schedule = jolpica_schedule(jolpica, season)
        results = jolpica_results(jolpica, season)
        out["schedule"].extend(schedule)
        out["results"].extend(results)
        out["qualifying"].extend(jolpica_qualifying(jolpica, season))
        for race in results:  # only races that have been run
            stops = jolpica_pitstops(jolpica, season, int(race["round"]))
            out["pitstops"].append({"season": season, "round": int(race["round"]), "stops": stops})

        run_rounds = {r["round"] for r in results}
        for race in schedule:
            if race["round"] not in run_rounds:
                continue
            loc = race["Circuit"]["Location"]
            ts = f"{race['date']}T{race.get('time', '12:00:00Z')}"
            wx = openmeteo_race_weather(openmeteo, float(loc["lat"]), float(loc["long"]), ts)
            out["weather"].append({"season": season, "round": int(race["round"]), "start": ts, "data": wx})

    for season in timing_seasons:
        sessions = openf1_race_sessions(openf1, season)
        log.info("OpenF1 %s: %d completed races", season, len(sessions))
        for s in sessions:
            log.info("  %s %s (session %s)", season, s["country_name"], s["session_key"])
            out["openf1"].append({"session": s, **openf1_session_bundle(openf1, s["session_key"])})

        practice = openf1_practice_sessions(openf1, season)
        log.info("OpenF1 %s: %d practice sessions", season, len(practice))
        for s in practice:
            out["practice"].append({"session": s, "laps": openf1.get("laps", {"session_key": s["session_key"]})})

    return out
