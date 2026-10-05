"""フェーズ3の評価：最新の project/CLAUDE.md 第5章（案A・銘柄ごとのコスト・損益分岐・暦日の年率）で
評価役のエンジン（bt2.py）を動かし、開発役の reports/phase2c_baselines.json と照合する。
結果：evaluation/data/phase3_bt_check.json
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

DEV = EVAL_ROOT.parent / "project" / "reports" / "phase2c_baselines.json"
OUT = EVAL_ROOT / "data" / "phase3_bt_check.json"
COSTS = {"no_cost": 0.0, "cost_0.3%": 0.003, "cost_0.5%": 0.005, "cost_tick": "tick"}


def main() -> None:
    t0 = time.time()
    mk = bt2.load_market()
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    dev = json.load(open(DEV, encoding="utf-8"))
    out = {"universe": {}, "universe_average": {}, "baselines": {}}
    preds = [w[0] - 1 for w in pc.weeks]
    sizes = pc.universe[preds].sum(axis=1)
    out["universe"] = {"min": int(sizes.min()), "median": float(np.median(sizes)), "max": int(sizes.max()),
                       "case_a_events": int(pc.case_a_known.sum()),
                       "dev": dev["meta"]["universe_size_per_pred"]}
    # 案Aでユニバースが変わった予測日の件数
    pc0 = bt2.Precomp(mk, exclude_margin_other=True, case_a=False)
    diff = (pc0.universe[preds] != pc.universe[preds]).sum()
    out["universe"]["pred_day_cells_changed_by_case_a"] = int(diff)
    known = np.argwhere(pc.case_a_known)
    out["universe"]["case_a_events_list"] = [[str(pc.dates[t]), str(pc.codes[j])] for t, j in known]

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
                out["baselines"][f"{kind}/{bm}/{name}"] = rec
                print(kind, bm, name, f"eval {m['annual_return']:.4f} dev {d['annual_return']:.4f}",
                      res.orders, d.get("orders"), flush=True)
                w.to_csv(EVAL_ROOT / "data" / "phase3_weekly" / f"{kind}_{bm}_{name}.csv", index=False)
    out["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    (EVAL_ROOT / "data" / "phase3_weekly").mkdir(parents=True, exist_ok=True)
    main()
