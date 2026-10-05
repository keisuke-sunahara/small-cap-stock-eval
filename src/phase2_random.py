"""フェーズ2の照合：ランダム選択（1,000回）を評価役の実装で計算し、開発役の分布と比べる。

乱数の使い方は開発役と違うので、1回ずつは一致しない。分布（平均・パーセント点）が近いかを確かめる。
使い方: python src/phase2_random.py [--margin] [--runs 1000]
出力: data/phase2_random[_b].json（各回の年率と週次リターンの平均）、data/phase2_random_weekly[_b].npy（週次リターン）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402


def main() -> None:
    margin = "--margin" in sys.argv
    runs = int(sys.argv[sys.argv.index("--runs") + 1]) if "--runs" in sys.argv else 1000
    tag = "_b" if margin else ""
    pc = bt2.Precomp(bt2.load_market(), exclude_margin_other=margin)
    out: dict = {"runs": runs, "exclude_margin_other": margin}
    weekly = {}
    t0 = time.time()
    for bm in ("min_equity", "fixed"):
        for cost, ck in ((0.0, "no_cost"), (0.003, "cost_0.3%"), (0.005, "cost_0.5%")):
            anns, fills, W = [], [], []
            for s in range(runs):
                r = bt2.simulate(pc, "random", cost, budget_mode=bm, seed=10_000 + s, record=False)
                m = bt2.metrics(r.weekly["ret"])
                anns.append(m["annual_return"])
                fills.append(r.fills / r.orders)
                W.append(r.weekly["ret"].to_numpy())
            a = np.array(anns)
            out[f"{bm}|{ck}"] = {"mean_annual": float(a.mean()),
                                 "pct": {q: float(np.quantile(a, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
                                 "fill_rate_mean": float(np.mean(fills)), "annual": a.tolist()}
            weekly[f"{bm}|{ck}"] = np.array(W)
            print(f"{bm} {ck} mean {a.mean():+.4f}  ({time.time() - t0:.0f}s)", flush=True)
    (EVAL_ROOT / "data" / f"phase2_random{tag}.json").write_text(json.dumps(out), encoding="utf-8")
    np.savez_compressed(EVAL_ROOT / "data" / f"phase2_random_weekly{tag}.npz", **{k.replace("|", "__").replace("%", "pct"): v for k, v in weekly.items()})


if __name__ == "__main__":
    main()
