"""評価役のJ-Quants API (V2) クライアント（開発役のコードは使わない）。

- APIキーは evaluator/.env の JQUANTS_API_KEY から読む。ログ・例外に出さない
- ホールドアウト開始日（project/config/base.yaml）以降の日付を含むリクエストは送らない
- レート制限：一般 120件/分・/fins/ 60件/分（Standard）の 7 割に抑える
"""
from __future__ import annotations

import os
import time
from datetime import date
from pathlib import Path

import requests
import yaml
from dotenv import load_dotenv

EVAL_ROOT = Path(__file__).resolve().parents[1]          # evaluator/evaluation
ROOT = EVAL_ROOT.parent                                   # evaluator
BASE_URL = "https://api.jquants.com/v2"


def holdout_start() -> date:
    cfg = yaml.safe_load((ROOT / "project" / "config" / "base.yaml").read_text(encoding="utf-8"))
    return date.fromisoformat(str(cfg["data"]["holdout_start"]))


HOLDOUT_START = holdout_start()
LAST_ALLOWED = date.fromordinal(HOLDOUT_START.toordinal() - 1)


class HoldoutViolation(RuntimeError):
    pass


def _check_dates(params: dict) -> None:
    for k in ("date", "to", "from"):
        if k in params and params[k]:
            d = date.fromisoformat(str(params[k])[:10])
            if d >= HOLDOUT_START:
                raise HoldoutViolation(f"{k}={d} はホールドアウト開始日 {HOLDOUT_START} 以降のため取得しない")
    if "date" not in params and "to" not in params:
        # 期間指定なしは最新（ホールドアウト期間）まで返るため禁止
        raise HoldoutViolation("date も to も指定のないリクエストは送らない")


class Client:
    def __init__(self, per_min_general: float = 120 * 0.7, per_min_fins: float = 60 * 0.7):
        load_dotenv(ROOT / ".env")
        key = os.environ.get("JQUANTS_API_KEY")
        if not key:
            raise RuntimeError("JQUANTS_API_KEY が evaluator/.env にない")
        self._key = key
        self._s = requests.Session()
        self._interval = {"general": 60 / per_min_general, "fins": 60 / per_min_fins}
        self._last = {"general": 0.0, "fins": 0.0}

    def __repr__(self) -> str:
        return "Client()"

    def _wait(self, bucket: str) -> None:
        dt = time.monotonic() - self._last[bucket]
        if dt < self._interval[bucket]:
            time.sleep(self._interval[bucket] - dt)
        self._last[bucket] = time.monotonic()

    def get(self, path: str, **params) -> list[dict]:
        params = {k: v for k, v in params.items() if v is not None}
        _check_dates(params)
        bucket = "fins" if path.startswith("/fins") else "general"
        rows: list[dict] = []
        n_err = 0
        while True:
            self._wait(bucket)
            try:
                r = self._s.get(BASE_URL + path, params=params, headers={"x-api-key": self._key}, timeout=60)
            except (requests.ConnectionError, requests.Timeout) as e:
                n_err += 1
                if n_err > 5:
                    raise RuntimeError(f"通信エラーが続いた: {path} {params} {type(e).__name__}") from None
                time.sleep(30)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                n_err += 1
                if n_err > 5:
                    raise RuntimeError(f"HTTP {r.status_code} が続いた: {path} {params}")
                time.sleep(120 if r.status_code == 429 else 30)
                continue
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code} {path} {params}: {r.text[:300]}")
            body = r.json()
            rows.extend(body.get("data", []))
            pk = body.get("pagination_key")
            if not pk:
                return rows
            params = {**params, "pagination_key": pk}
