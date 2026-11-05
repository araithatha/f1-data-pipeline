"""Extract step: pull raw records from the Jolpica-F1 and OpenF1 APIs.

Jolpica-F1 (successor to the Ergast API) supplies the historical, relational
side: schedule, circuits, race results and qualifying, back to 1950.

OpenF1 supplies lap-level timing for 2023 onward: lap and sector times, tyre
stints, pit stops, weather and the driver/car-number mapping.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
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


OPENF1_SESSION_ENDPOINTS = ("drivers", "laps", "stints", "pit", "weather")


def openf1_session_bundle(client: CachedClient, session_key: int) -> dict[str, list[dict]]:
    """All lap-level tables for one race session."""
    return {
        name: client.get(name, {"session_key": session_key}) for name in OPENF1_SESSION_ENDPOINTS
    }


def extract_all(
    jolpica: CachedClient,
    openf1: CachedClient,
    history_seasons: list[int],
    timing_seasons: list[int],
) -> dict[str, Any]:
    out: dict[str, Any] = {"schedule": [], "results": [], "qualifying": [], "openf1": []}

    for season in history_seasons:
        log.info("Jolpica %s: schedule, results, qualifying", season)
        out["schedule"].extend(jolpica_schedule(jolpica, season))
        out["results"].extend(jolpica_results(jolpica, season))
        out["qualifying"].extend(jolpica_qualifying(jolpica, season))

    for season in timing_seasons:
        sessions = openf1_race_sessions(openf1, season)
        log.info("OpenF1 %s: %d completed races", season, len(sessions))
        for s in sessions:
            log.info("  %s %s (session %s)", season, s["country_name"], s["session_key"])
            out["openf1"].append({"session": s, **openf1_session_bundle(openf1, s["session_key"])})

    return out
