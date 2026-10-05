"""依頼の確認：株式数の補正の後のベースライン（開発役の reports/phase3r_baselines.json）を、評価役のエンジン（bt2.py）と
評価役が独立に実装した株式数の補正（data/processed/shares_fixed.npz）で再計算して照合する。

- モメンタム・リバーサル × 予算2通り × コスト4通り（年率・シャープ・最悪の週・注文数・約定率、損益分岐のコスト）
- ユニバース平均 × コスト4通り
- ランダム：予算固定・基本ルール × 一律0.3%・銘柄ごとのコスト（評価役は各200回。開発役は1,000回の平均）
- Rank IC（週の最初の営業日の始値 → 最終営業日の終値、調整後。予測日のユニバース内の順位相関）
結果：data/phase4_bt_check.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402

DEV = EVAL_ROOT.parent / "project" / "reports" / "phase3r_baselines.json"
OUT = EVAL_ROOT / "data" / "phase4_bt_check.json"
COSTS = {"no_cost": 0.0, "cost_0.3%": 0.003, "cost_0.5%": 0.005, "cost_tick": "tick"}
RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def rank_ic(pc: bt2.Precomp, kind: str) -> dict:
    ics = []
    for w in pc.weeks:
        p, a, b = w[0] - 1, w[0], w[-1]
        idx = np.flatnonzero(pc.universe[p])
        s = pc.score[kind][p, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = (pc.C[b, idx] / pc.cumF[b, idx]) / (pc.O[a, idx] / pc.cumF[a, idx]) - 1
        ok = np.isfinite(s) & np.isfinite(r)
        ics.append(spearmanr(s[ok], r[ok]).statistic if ok.sum() > 2 else np.nan)
    ics = np.array(ics)
    m, sd = np.nanmean(ics), np.nanstd(ics, ddof=1)
    return {"mean": float(m), "std": float(sd), "t": float(m / sd * np.sqrt(np.isfinite(ics).sum())), "weeks": int(np.isfinite(ics).sum())}


def main() -> None:
    t0 = time.time()
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    dev = json.load(open(DEV, encoding="utf-8"))
    out = {"universe": {}, "universe_average": {}, "baselines": {}, "random": {}, "rank_ic": {}}
    preds = [w[0] - 1 for w in pc.weeks]
    sizes = pc.universe[preds].sum(axis=1)
    out["universe"] = {"eval": [int(sizes.min()), float(np.median(sizes)), int(sizes.max())],
                       "dev": dev["meta"]["universe_size_per_pred"], "weeks": len(pc.weeks), "dev_weeks": dev["meta"]["weeks"]}
    ua = bt2.universe_average(pc, costs=tuple(COSTS.values()))
    yrs = bt2.years_of(ua.assign(pred_date=[pc.dates[w[0] - 1] for w in pc.weeks]))
    bench = None
    for name, c in COSTS.items():
        m = bt2.metrics(ua[f"ret_{c}"], years=yrs)
        if name == "no_cost":
            bench = m["annual_return"]
        d = dev["universe_average"][name]
        out["universe_average"][name] = {"eval": m["annual_return"], "dev": d["annual_return"],
                                         "eval_worst": m["worst_week"], "dev_worst": d["worst_week"]}
    maxdiff = 0.0
    for kind in ("momentum_20d", "reversal_5d"):
        for bm in ("fixed", "min_equity"):
            for name, c in COSTS.items():
                res = bt2.simulate(pc, kind, c, budget_mode=bm, record=False)
                w = res.weekly
                m = bt2.metrics(w["ret"], years=bt2.years_of(w))
                d = dev["baselines"][kind][bm][name]
                rec = {"eval_ann": m["annual_return"], "dev_ann": d["annual_return"],
                       "eval_sharpe": m["sharpe"], "dev_sharpe": d["sharpe"],
                       "eval_worst": m["worst_week"], "dev_worst": d["worst_week"],
                       "eval_orders": res.orders, "dev_orders": d.get("orders"),
                       "eval_fill": res.fills / max(res.orders, 1), "dev_fill": d.get("fill_rate")}
                if bm == "fixed" and name == "cost_0.3%":
                    rec["eval_breakeven_vs_universe"] = bt2.breakeven_cost(w, bench)
                    rec["dev_breakeven_vs_universe"] = d.get("breakeven_one_way_cost_vs_universe")
                    rec["eval_breakeven_vs_zero"] = bt2.breakeven_cost(w, 0.0)
                    rec["dev_breakeven_vs_zero"] = d.get("breakeven_one_way_cost_vs_zero")
                maxdiff = max(maxdiff, abs(rec["eval_ann"] - rec["dev_ann"]), abs(rec["eval_sharpe"] - rec["dev_sharpe"]),
                              abs(rec["eval_worst"] - rec["dev_worst"]))
                out["baselines"][f"{kind}/{bm}/{name}"] = rec
                print(kind, bm, name, f"eval {m['annual_return']:.6f} dev {d['annual_return']:.6f}",
                      res.orders, d.get("orders"), flush=True)
    out["baselines_max_abs_diff"] = maxdiff
    for kind in ("momentum_20d", "reversal_5d"):
        out["rank_ic"][kind] = {"eval": rank_ic(pc, kind), "dev": dev["rank_ic"][kind]}
        print(kind, out["rank_ic"][kind], flush=True)
    for bm in ("fixed", "min_equity"):
        for name in ("cost_0.3%", "cost_tick"):
            a = []
            for s in range(RUNS):
                r = bt2.simulate(pc, "random", COSTS[name], budget_mode=bm, seed=30_000 + s, record=False)
                a.append(bt2.metrics(r.weekly["ret"], years=bt2.years_of(r.weekly))["annual_return"])
            a = np.array(a)
            d = dev["random"][bm][name]
            out["random"][f"{bm}/{name}"] = {"eval_mean": float(a.mean()), "eval_se": float(a.std(ddof=1) / np.sqrt(RUNS)),
                                             "runs": RUNS, "dev_mean": d["mean"]["annual_return"], "dev_runs": d["runs"],
                                             "z": float((a.mean() - d["mean"]["annual_return"]) / (a.std(ddof=1) / np.sqrt(RUNS)))}
            print(bm, name, out["random"][f"{bm}/{name}"], f"{time.time() - t0:.0f}s", flush=True)
    out["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
