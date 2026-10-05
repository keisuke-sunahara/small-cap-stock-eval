"""フェーズ4前半：コストの仮定が成績の判断を変えるか（評価役）。予算固定で、一律の片道コスト c を変えた年率と、
データ費用（年6.6%）・配当の見積もり（phase4_production.py、税引き後）を足し引きした年率。損益分岐の c も求める。
週次リターン(c) = (損益 + 払ったコスト − c × 売買代金) ÷ 運用資金（bt2.breakeven_cost と同じ考え方）。
結果：data/phase4_cost_curve.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402
from phase4_model_check import load_scores  # noqa: E402

FEE = 19_800 / bt2.CAPITAL


def main() -> None:
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    prod = json.load(open(EVAL_ROOT / "data/phase4_production.json", encoding="utf-8"))
    div = prod["dividend_EXP-002"]["pct_of_capital_per_year"] * (1 - 0.20315)
    out = {"data_fee": FEE, "dividend_after_tax_est": div, "exp": {}}
    for e in ("EXP-002", "EXP-001"):
        M, _ = load_scores(pc, e)
        pc.score[e] = M
        w = bt2.simulate(pc, e, 0.003, budget_mode="fixed", record=False).weekly
        yrs = bt2.years_of(w)
        pnl = w["ret"].to_numpy() * bt2.CAPITAL + w["cost_paid"].to_numpy()
        tv = w["traded"].to_numpy()

        def ann(c: float) -> float:
            r = (pnl - c * tv) / bt2.CAPITAL
            tot = np.prod(1 + r)
            return tot ** (1 / yrs) - 1 if tot > 0 else -1.0

        curve = {f"{c:.4f}": {"annual": ann(c), "after_fee": ann(c) - FEE, "after_fee_plus_div": ann(c) - FEE + div}
                 for c in (0.0, 0.0005, 0.001, 0.0015, 0.002, 0.0025, 0.003)}

        def solve(target_fn) -> float | None:
            lo, hi = -0.01, 0.01
            if target_fn(lo) < 0:
                return None
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if target_fn(mid) > 0 else (lo, mid)
            return (lo + hi) / 2

        out["exp"][e] = {"curve": curve, "turnover_per_year": float(tv.sum() / bt2.CAPITAL / yrs),
                         "breakeven_after_fee": solve(lambda c: ann(c) - FEE),
                         "breakeven_after_fee_plus_div": solve(lambda c: ann(c) - FEE + div)}
        print(e, json.dumps(out["exp"][e], indent=0)[:900])
    (EVAL_ROOT / "data/phase4_cost_curve.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
