"""フェーズ2の照合：評価役の独立実装（bt2.py）でベースラインを計算し、開発役の phase2_baselines.json と並べる。

使い方: python src/phase2_run.py [--margin]   （--margin：貸借信用区分「その他」を除く＝phase2b と照合）
出力: data/phase2_check[_b].json、data/phase2_weekly/<戦略>_<予算>_<コスト>.csv、trades の csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402

PROJECT = EVAL_ROOT.parent / "project"


def main() -> None:
    margin = "--margin" in sys.argv
    tag = "_b" if margin else ""
    dev = json.loads((PROJECT / "reports" / f"phase2{'b' if margin else ''}_baselines.json").read_text(encoding="utf-8"))
    outdir = EVAL_ROOT / "data" / f"phase2_weekly{tag}"
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    pc = bt2.Precomp(bt2.load_market(), exclude_margin_other=margin)
    res: dict = {"exclude_margin_other": margin, "strategies": {}}
    for kind in ("momentum_20d", "reversal_5d"):
        for bm in ("min_equity", "fixed"):
            for cost, ck in ((0.0, "no_cost"), (0.003, "cost_0.3%"), (0.005, "cost_0.5%")):
                r = bt2.simulate(pc, kind, cost, budget_mode=bm)
                r.weekly.to_csv(outdir / f"{kind}_{bm}_{ck}.csv", index=False)
                if ck == "cost_0.3%":
                    r.trades.to_csv(outdir / f"{kind}_{bm}_{ck}_trades.csv", index=False)
                m = bt2.metrics(r.weekly["ret"], r.weekly["week_end"])
                m["orders"] = r.orders
                m["fill_rate"] = r.fills / r.orders if r.orders else None
                m["cash_week_ratio"] = float((r.weekly["positions_end"] == 0).mean())
                d = dev["baselines"][kind][bm][ck]
                res["strategies"][f"{kind}|{bm}|{ck}"] = {
                    "evaluator": m,
                    "developer": {k: d.get(k) for k in ("annual_return", "total_return", "sharpe", "max_drawdown",
                                                         "worst_week", "orders", "fill_rate", "cash_week_ratio")},
                }
                print(f"{kind:13s} {bm:10s} {ck:9s} 評価役 {m['annual_return']:+.4f} 開発役 {d['annual_return']:+.4f}"
                      f"  注文 {r.orders}/{d.get('orders')}  約定率 {m['fill_rate']:.3f}/{d.get('fill_rate')}"
                      f"  ({time.time() - t0:.0f}s)", flush=True)
    (EVAL_ROOT / "data" / f"phase2_check{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
