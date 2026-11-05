"""HTTP client with on-disk caching, rate limiting and retries.

Every API response is saved as raw JSON under ``data/raw/<source>/``. Re-running
the pipeline reads from that cache instead of calling the API again, so a run
can be interrupted and resumed, and the raw layer can be rebuilt offline.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import requests

log = logging.getLogger(__name__)


class CachedClient:
    """A small polite API client: one request at a time, with a minimum gap."""

    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        min_interval_s: float = 1.0,
        max_retries: int = 5,
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._last_call = 0.0
        self._session = requests.Session()
        self._session.headers["User-Agent"] = "f1-data-portfolio/1.0 (personal project)"
        self.calls_made = 0
        self.cache_hits = 0

    def _cache_path(self, path: str, params: dict[str, Any] | None) -> Path:
        query = urlencode(sorted((params or {}).items()))
        readable = (path.strip("/").replace("/", "_") + ("__" + query if query else ""))[:120]
        digest = hashlib.sha1(f"{path}?{query}".encode()).hexdigest()[:10]
        safe = "".join(c if c.isalnum() or c in "-_=." else "-" for c in readable)
        return self.cache_dir / f"{safe}__{digest}.json"

    def get(self, path: str, params: dict[str, Any] | None = None, use_cache: bool = True) -> Any:
        cache_file = self._cache_path(path, params)
        if use_cache and cache_file.exists():
            self.cache_hits += 1
            return json.loads(cache_file.read_text())

        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(1, self.max_retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException as exc:
                log.warning("Request error %s (attempt %d): %s", url, attempt, exc)
                time.sleep(2**attempt)
                continue

            self.calls_made += 1
            if resp.status_code == 200:
                data = resp.json()
                cache_file.write_text(json.dumps(data))
                return data
            if resp.status_code == 404:
                # OpenF1 returns 404 for "no rows match"; treat as empty.
                cache_file.write_text("[]")
                return []
            if resp.status_code in (429, 500, 502, 503, 504):
                backoff = float(resp.headers.get("Retry-After", 0)) or min(60, 5 * 2 ** (attempt - 1))
                log.warning("HTTP %s from %s, retrying in %.0fs", resp.status_code, url, backoff)
                time.sleep(backoff)
                continue
            resp.raise_for_status()

        raise RuntimeError(f"Gave up on {url} after {self.max_retries} attempts")
