"""評価役のデータ取得。ホールドアウト開始日の前日までだけを取る。

使い方: python src/fetch.py general   （calendar, topix, master, bars）
        python src/fetch.py fins      （summary, earnings_date）
- 1日1ファイル（Parquet）。保存済みの日は取らない（再実行で続きから）
- 保存は一時ファイル → 名前の変更（途中で止まっても壊れたファイルを残さない）
"""
from __future__ import annotations

import csv
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jq import EVAL_ROOT, LAST_ALLOWED, Client  # noqa: E402

START = date(2016, 10, 5)          # 開発役と同じ開始日（Standardの範囲 2016-10-04〜 の翌日）
END = LAST_ALLOWED                 # 2025-09-28
RAW = EVAL_ROOT / "data" / "raw"


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def log(ds: str, key: str, n: int) -> None:
    p = RAW / ds / "_fetch_log.csv"
    new = not p.exists()
    with p.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["key", "rows", "fetched_at"])
        w.writerow([key, n, datetime.now().isoformat(timespec="seconds")])


def fetch_range(c: Client, ds: str, path: str) -> None:
    out = RAW / ds / f"{START}_{END}.parquet"
    if out.exists():
        return
    rows = c.get(path, **{"from": START.isoformat(), "to": END.isoformat()})
    save(pd.DataFrame(rows), out)
    log(ds, out.stem, len(rows))
    print(ds, len(rows), flush=True)


def fetch_daily(c: Client, ds: str, path: str, days: list[str]) -> None:
    todo = [d for d in days if not (RAW / ds / f"{d}.parquet").exists()]
    print(ds, "todo", len(todo), "of", len(days), flush=True)
    for i, d in enumerate(todo):
        rows = c.get(path, date=d)
        save(pd.DataFrame(rows), RAW / ds / f"{d}.parquet")
        log(ds, d, len(rows))
        if i % 100 == 0:
            print(ds, d, len(rows), f"{i}/{len(todo)}", flush=True)


def business_days() -> list[str]:
    cal = pd.read_parquet(RAW / "calendar" / f"{START}_{END}.parquet")
    return sorted(cal.loc[cal["HolDiv"].astype(str) == "1", "Date"].astype(str))


def main(which: str) -> None:
    if which == "general":
        c = Client(per_min_general=60, per_min_fins=1)
        fetch_range(c, "calendar", "/markets/calendar")
        fetch_range(c, "topix", "/indices/bars/daily/topix")
        bdays = business_days()
        fetch_daily(c, "master", "/equities/master", bdays)
        fetch_daily(c, "bars", "/equities/bars/daily", bdays)
    elif which == "fins":
        c = Client(per_min_general=1, per_min_fins=50)
        n = (END - START).days + 1
        cdays = [(START + timedelta(days=i)).isoformat() for i in range(n)]
        fetch_daily(c, "summary", "/fins/summary", cdays)
        fetch_daily(c, "earnings_date", "/fins/earnings-date", cdays)
    print("done", which, flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
