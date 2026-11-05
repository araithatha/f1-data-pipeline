"""Unit tests for the transform layer, using small hand-built fixtures (no network)."""

import math

import pandas as pd
import pytest

from f1pipeline import transform as t


@pytest.mark.parametrize(
    "raw, expected",
    [("1:29.708", 89.708), ("59.1", 59.1), ("2:00.000", 120.0), ("", math.nan), (None, math.nan), ("DNF", math.nan)],
)
def test_parse_lap_time(raw, expected):
    got = t.parse_lap_time(raw)
    assert (math.isnan(got) and math.isnan(expected)) or got == pytest.approx(expected)


def test_race_id_is_zero_padded():
    assert t.race_id("2024", "3") == "2024_03"


def _result(driver, number, grid, position, pos_text=None, status="Finished"):
    return {
        "number": str(number),
        "position": str(position),
        "positionText": pos_text or str(position),
        "points": "0",
        "grid": str(grid),
        "laps": "57",
        "status": status,
        "Driver": {"driverId": driver, "givenName": driver.title(), "familyName": "X"},
        "Constructor": {"constructorId": "team", "name": "Team"},
    }


def test_fact_result_pit_lane_start_and_positions_gained():
    races = [
        {
            "season": "2024",
            "round": "1",
            "Results": [
                _result("a", 1, 2, 1),
                _result("b", 2, 1, 2),
                _result("c", 3, 0, 3),  # pit-lane start
                _result("d", 4, 3, 4, pos_text="R", status="Retired"),
            ],
        }
    ]
    df = t.build_fact_result(races).set_index("driver_id")
    assert df.loc["c", "grid_adj"] == 4            # starts behind the 4-car field
    assert df.loc["c", "positions_gained"] == 1
    assert df.loc["a", "positions_gained"] == 1
    assert not df.loc["d", "is_classified"]
    assert math.isnan(df.loc["d", "positions_gained"])
    assert df["is_win"].sum() == 1


def test_qualifying_best_time_and_gap_to_pole():
    races = [
        {
            "season": "2024",
            "round": "1",
            "QualifyingResults": [
                {"position": "1", "Q1": "1:30.000", "Q2": "1:29.500", "Q3": "1:29.000",
                 "Driver": {"driverId": "a"}, "Constructor": {"constructorId": "t"}},
                {"position": "2", "Q1": "1:30.100", "Q2": "", "Q3": "",
                 "Driver": {"driverId": "b"}, "Constructor": {"constructorId": "t"}},
            ],
        }
    ]
    df = t.build_fact_qualifying(races).set_index("driver_id")
    assert df.loc["a", "best_quali_s"] == pytest.approx(89.0)
    assert df.loc["a", "gap_to_pole_pct"] == pytest.approx(0.0)
    assert df.loc["b", "gap_to_pole_pct"] == pytest.approx((90.1 - 89.0) / 89.0 * 100)
    assert bool(df.loc["a", "reached_q3"]) and not bool(df.loc["b", "reached_q3"])


def test_expand_stints_tyre_age_and_overlap():
    stints = [
        {"driver_number": 1, "stint_number": 1, "lap_start": 1, "lap_end": 3, "compound": "SOFT", "tyre_age_at_start": 2},
        {"driver_number": 1, "stint_number": 2, "lap_start": 3, "lap_end": 4, "compound": "hard", "tyre_age_at_start": 0},
    ]
    df = t._expand_stints(stints).set_index("lap_number")
    assert df.loc[1, "tyre_age_laps"] == 2
    assert df.loc[2, "tyre_age_laps"] == 3
    assert df.loc[3, "compound"] == "HARD"      # overlapping lap goes to the later stint
    assert df.loc[4, "tyre_age_laps"] == 1
    assert len(df) == 4


def test_match_race_id_tolerates_timezone_offsets():
    dim = pd.DataFrame(
        {"race_id": ["2024_01", "2024_02"], "season": [2024, 2024],
         "race_start_utc": pd.to_datetime(["2024-03-02T15:00:00Z", "2024-03-09T17:00:00Z"])}
    )
    assert t._match_race_id({"year": 2024, "date_start": "2024-03-09T17:00:00+00:00"}, dim) == "2024_02"
    assert t._match_race_id({"year": 2024, "date_start": "2024-05-01T12:00:00+00:00"}, dim) is None
