"""Train, compare, select and evaluate the race prediction model.

    python -m f1model.run

Writes leaderboards, test results and per-driver probabilities to
projects/02_race_prediction/results/. Charts are made by f1model.figures.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import chaos, features, models
from .models import CANDIDATES, TEST_SEASONS, VALIDATION_SEASONS

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "projects" / "02_race_prediction" / "results"
log = logging.getLogger("f1model")


def official_order_metrics(df: pd.DataFrame, score_col: str) -> dict[str, float]:
    """Compare a predicted order (lower = better) with the official result, retirements included."""
    m = models.race_metrics(df.assign(_truth=df["finish_position"]), score_col, truth_col="_truth")
    return {f"official_{k}": v for k, v in m.items()}


def group_importance(df: pd.DataFrame, cand, params, feats, season: int, n_repeats: int = 5) -> pd.DataFrame:
    """Permutation importance by feature group: how much does test Spearman drop when a group is scrambled?"""
    rng = np.random.default_rng(models.SEED)
    train = df[(df["season"] >= models.FIRST_TRAIN_SEASON) & (df["season"] < season)]
    test = df[df["season"] == season].copy()
    tr = train.dropna(subset=["finisher_rank"]).sort_values(["race_order", "finisher_rank"])
    model = cand.build(**params)
    if cand.ranker:
        rel = (tr["n_finishers"] - tr["finisher_rank"]).clip(lower=0).astype(int)
        model.fit(tr[feats], rel, group=tr.groupby("race_order", sort=False).size().to_numpy())
        predict = lambda X: -model.predict(X)
    else:
        model.fit(tr[feats], tr["finisher_rank"])
        predict = model.predict
    base = models.race_metrics(test.assign(score=predict(test[feats])), "score")["spearman"]
    rows = []
    for group, cols in features.FEATURES.items():
        cols = [c for c in cols if c in feats]
        drops = []
        for _ in range(n_repeats):
            shuffled = test.copy()
            # Shuffle within each race so the scrambled values stay realistic for that race.
            for c in cols:
                shuffled[c] = shuffled.groupby("race_id")[c].transform(lambda s: s.sample(frac=1, random_state=int(rng.integers(1e9))).to_numpy())
            drops.append(base - models.race_metrics(shuffled.assign(score=predict(shuffled[feats])), "score")["spearman"])
        rows.append({"feature_group": group, "spearman_drop": float(np.mean(drops)), "n_features": len(cols)})
    return pd.DataFrame(rows).sort_values("spearman_drop", ascending=False)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    started = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    df = features.build_feature_table()
    df.to_csv(OUT / "feature_table.csv", index=False)
    feats = features.ALL_FEATURES
    log.info("feature table: %d driver-races, %d races, %d features", len(df), df["race_id"].nunique(), len(feats))

    # ---- Stage 1: compare pace models on validation seasons ------------------
    board, best_params = models.compare_models(df, feats)
    board.to_csv(OUT / "model_comparison_validation.csv", index=False)
    best_name = board.loc[0, "model"]
    best = next(c for c in CANDIDATES if c.name == best_name)
    params = best_params[best_name]
    log.info("selected pace model: %s %s", best_name, params)

    # Ablation: drop one feature group at a time (validation seasons).
    base_val = board.loc[0, "spearman"]
    ablation = []
    for group, cols in features.FEATURES.items():
        kept = [f for f in feats if f not in cols]
        oof = models.walk_forward(df, best, params, kept, VALIDATION_SEASONS)
        ablation.append({"dropped_group": group, "spearman": models.race_metrics(oof, "score")["spearman"]})
    ablation = pd.DataFrame(ablation).assign(change=lambda d: d["spearman"] - base_val).sort_values("change")
    ablation.to_csv(OUT / "ablation_validation.csv", index=False)

    # ---- Stage 2: DNF model ---------------------------------------------------
    dnf_board, dnf_name = chaos.compare_dnf_models(df, VALIDATION_SEASONS)
    dnf_board.to_csv(OUT / "dnf_model_comparison_validation.csv", index=False)
    make_dnf = chaos.DNF_CANDIDATES[dnf_name]
    log.info("selected DNF model: %s", dnf_name)

    # Fit the race-day noise on out-of-fold validation predictions.
    val = models.walk_forward(df, best, params, feats, VALIDATION_SEASONS)
    val["p_dnf"] = chaos.dnf_walk_forward(df, make_dnf, VALIDATION_SEASONS)
    val["mu"] = chaos.standardise_scores(val)
    sigma, wet_mult, noise_grid = chaos.fit_noise(val)
    noise_grid.to_csv(OUT / "noise_fit_validation.csv", index=False)
    log.info("fitted noise: sigma=%.2f, wet multiplier=%.2f", sigma, wet_mult)

    # Comparison systems, fitted the same way: grid order + base-rate DNFs, and the model without a DNF layer.
    base_rate = chaos.dnf_walk_forward(df, chaos.DNF_CANDIDATES["Base rate (baseline)"], VALIDATION_SEASONS)
    val_grid = val.assign(score=val["grid_adj"], p_dnf=base_rate)
    val_grid["mu"] = chaos.standardise_scores(val_grid)
    g_sigma, g_wet, _ = chaos.fit_noise(val_grid)
    val_nodnf = val.assign(p_dnf=0.0)
    n_sigma, n_wet, _ = chaos.fit_noise(val_nodnf)

    # ---- Held-out test: 2025 and 2026, walk-forward --------------------------
    test = models.walk_forward(df, best, params, feats, TEST_SEASONS)
    test["p_dnf"] = chaos.dnf_walk_forward(df, make_dnf, TEST_SEASONS)
    test["mu"] = chaos.standardise_scores(test)
    test_sim = chaos.simulate_all(test, sigma, wet_mult)

    test_grid = test.assign(score=test["grid_adj"], p_dnf=chaos.dnf_walk_forward(df, chaos.DNF_CANDIDATES["Base rate (baseline)"], TEST_SEASONS))
    test_grid["mu"] = chaos.standardise_scores(test_grid)
    grid_sim = chaos.simulate_all(test_grid, g_sigma, g_wet)
    nodnf_sim = chaos.simulate_all(test.assign(p_dnf=0.0), n_sigma, n_wet)

    # Every candidate on the test seasons too (selection was already made on validation).
    test_board = []
    for cand in CANDIDATES:
        oof = models.walk_forward(df, cand, best_params[cand.name], feats, TEST_SEASONS)
        test_board.append({"model": cand.name, **models.race_metrics(oof, "score")})
    test_board = pd.DataFrame(test_board).sort_values("spearman", ascending=False)
    test_board.to_csv(OUT / "model_comparison_test.csv", index=False)

    # Practice-pace ablation on the test seasons (practice data exists from 2023).
    no_fp = [f for f in feats if f not in features.FEATURES["practice"]]
    test_nofp = models.walk_forward(df, best, params, no_fp, TEST_SEASONS)

    summary = {
        "selected_pace_model": best_name,
        "pace_model_params": params,
        "selected_dnf_model": dnf_name,
        "noise": {"sigma": sigma, "wet_multiplier": wet_mult},
        "validation_seasons": VALIDATION_SEASONS,
        "test_seasons": TEST_SEASONS,
        "test_races": int(test["race_id"].nunique()),
        "pace_model_test": models.race_metrics(test, "score"),
        "pace_model_test_without_practice": models.race_metrics(test_nofp, "score"),
        "grid_baseline_test": models.race_metrics(test.assign(score=test["grid_adj"]), "score"),
        "full_system_test_official_order": official_order_metrics(test_sim, "expected_position"),
        "grid_baseline_test_official_order": official_order_metrics(test_sim.assign(g=test_sim["grid_adj"]), "g"),
        "favourite_wins_test": float(test_sim.loc[test_sim.groupby("race_id")["p_win"].idxmax(), "is_win"].mean()),
        "pole_sitter_wins_test": float(test_sim.loc[test_sim.groupby("race_id")["grid_adj"].idxmin(), "is_win"].mean()),
        "probabilistic_test": {
            "full_system": chaos.probabilistic_scores(test_sim),
            "grid_baseline_sim": chaos.probabilistic_scores(grid_sim),
            "without_dnf_layer": chaos.probabilistic_scores(nodnf_sim),
        },
        "by_season": {
            int(s): {
                **models.race_metrics(g, "score"),
                "grid_spearman": models.race_metrics(g.assign(score=g["grid_adj"]), "score")["spearman"],
                **chaos.probabilistic_scores(test_sim[test_sim["season"] == s]),
            }
            for s, g in test.groupby("season")
        },
    }

    # Calibration: predicted probability bins vs how often it happened.
    calib = []
    for event in ["win", "podium", "top10"]:
        bins = pd.cut(test_sim[f"p_{event}"], [0, 0.05, 0.15, 0.3, 0.5, 0.7, 0.85, 1.0], include_lowest=True)
        c = test_sim.groupby(bins, observed=True).agg(predicted=(f"p_{event}", "mean"), observed=(f"is_{event}", "mean"),
                                                       n=(f"is_{event}", "size")).reset_index()
        calib.append(c.assign(event=event, bin=c[f"p_{event}"].astype(str)).drop(columns=f"p_{event}"))
    pd.concat(calib).to_csv(OUT / "calibration_test.csv", index=False)

    importance = group_importance(df, best, params, feats, season=2025)
    importance.to_csv(OUT / "feature_group_importance_test2025.csv", index=False)

    cols = ["race_id", "season", "round", "race_name", "driver_id", "team_id", "grid_adj", "finish_position", "status",
            "p_dnf", "p_win", "p_podium", "p_top10", "expected_position", "p10_position", "p90_position"]
    test_sim[cols].sort_values(["race_id", "expected_position"]).to_csv(OUT / "test_predictions.csv", index=False)

    summary["runtime_s"] = round(time.monotonic() - started, 1)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    log.info("done in %.0fs", summary["runtime_s"])
    print(json.dumps({k: summary[k] for k in ["selected_pace_model", "pace_model_test", "grid_baseline_test",
                                               "favourite_wins_test", "pole_sitter_wins_test"]}, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
