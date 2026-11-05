"""Pipeline entry point.

    python -m f1pipeline.run                         # defaults: results 2014+, lap timing 2023+
    python -m f1pipeline.run --history-from 2018 --timing-from 2024
    python -m f1pipeline.run --refresh               # ignore the raw cache and re-download

Raw API responses are cached in data/raw/, so re-runs only download what is new.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from . import extract, transform, validate
from .http import CachedClient

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

JOLPICA_URL = "https://api.jolpi.ca/ergast/f1"
OPENF1_URL = "https://api.openf1.org/v1"


def main(argv: list[str] | None = None) -> int:
    this_year = datetime.now(timezone.utc).year
    p = argparse.ArgumentParser(description="Build the F1 analytics dataset.")
    p.add_argument("--history-from", type=int, default=2014, help="first season for results/qualifying (Jolpica)")
    p.add_argument("--timing-from", type=int, default=2023, help="first season for lap timing (OpenF1, 2023+)")
    p.add_argument("--to", type=int, default=this_year, help="last season to include")
    p.add_argument("--refresh", action="store_true", help="re-download the current season instead of using the cache")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    log = logging.getLogger("f1pipeline")
    started = time.monotonic()

    jolpica = CachedClient(JOLPICA_URL, RAW / "jolpica", min_interval_s=1.0)
    openf1 = CachedClient(OPENF1_URL, RAW / "openf1", min_interval_s=2.1)

    if args.refresh:
        # Only the current season changes; clear its cached index files so new races are picked up.
        for f in list((RAW / "jolpica").glob(f"{args.to}_*")) + list((RAW / "openf1").glob(f"sessions__*year={args.to}*")):
            f.unlink()

    history = list(range(args.history_from, args.to + 1))
    timing = list(range(max(args.timing_from, 2023), args.to + 1))

    raw = extract.extract_all(jolpica, openf1, history, timing)
    tables = transform.transform_all(raw)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(PROCESSED / f"{name}.csv", index=False)
        log.info("wrote %-16s %8d rows", name, len(df))

    checks = validate.run_checks(tables)
    summary = validate.report(checks)
    manifest = {
        "built_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "history_seasons": [history[0], history[-1]],
        "timing_seasons": [timing[0], timing[-1]] if timing else [],
        "row_counts": {k: len(v) for k, v in tables.items()},
        "api_calls": {"jolpica": jolpica.calls_made, "openf1": openf1.calls_made},
        "cache_hits": {"jolpica": jolpica.cache_hits, "openf1": openf1.cache_hits},
        "runtime_s": round(time.monotonic() - started, 1),
    }
    (PROCESSED / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    (PROCESSED / "_quality_report.md").write_text(f"# Data quality report\n\nBuilt {manifest['built_at_utc']}\n\n{summary}\n")
    print("\n" + summary)

    failed = [c for c in checks if c.hard and not c.passed]
    if failed:
        log.error("%d hard data-quality checks failed", len(failed))
        return 1
    log.info("done in %.0fs (%d API calls, %d cache hits)", manifest["runtime_s"],
             jolpica.calls_made + openf1.calls_made, jolpica.cache_hits + openf1.cache_hits)
    return 0


if __name__ == "__main__":
    sys.exit(main())
