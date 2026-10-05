"""v2 計画の確認（2026-10-05）：開発役の E0・E0b の数字のうち、評価役のデータだけで確かめられるものを独立に計算する。

開発役のコードは使わない。株価は評価役が自分で取得したもの（2016-10-05〜2025-09-26。ホールドアウト 2025-09-29 以降は未取得）。
成績・Rank IC・目的変数・リターンの平均は計算しない（開発役の E0 と同じ制約。月ごとの「ばらつき」だけを出す）。

(1) E0 (3)：v2 の保有期間（予測日 P の2営業日後 B の始値 → 翌々月の第1営業日 S' の始値、調整後）の月次リターンの、
    ユニバース内の標準偏差 σ（月ごと）の中央値。N 銘柄の等金額での1年の超過リターンのぶれ σ√(12/N) と、検証期間の年率の平均の標準誤差
    ※ 平均は計算も出力もしない（del）。ばらつきだけ
(2) E0b (1)：今のかぶミニの取扱一覧（project/logs/v2_e0/kabumini_yoritsuki_2026-10-05.csv、コード・銘柄名・市場の3列）が、
    2025-09-26 の評価役のユニバースの何割か。時価総額の帯ごと
(3) E0 (4)・5.1：かぶミニの成行の買いの余力拘束（（基準値段 + 制限値幅）× 1.0022、1円未満切上げ）で、1回の注文で入る資金の割合
(4) E0 (2)：50万円・N=20 で100株を買えるユニバースの割合（2025-09-26 と予測日の中央値）
(5) バッファの機械的な性質：予測日の上位20%の銘柄数（=保有を続ける条件の枠）と N=20 の比

結果：data/v2_plan_check.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402

PROJ = EVAL_ROOT.parent / "project"
LINEUP = PROJ / "logs" / "v2_e0" / "kabumini_yoritsuki_2026-10-05.csv"
OUT = EVAL_ROOT / "data" / "v2_plan_check.json"
HOLDOUT = "2025-09-29"
FIRST_P, LAST_P = "2018-09", "2025-08"
CHECK = "2025-09-26"
SPREAD = 0.0022
CAPITAL = 500_000
N = 20

# 東証の制限値幅（基準値段 未満 → 値幅）
LIMIT_WIDTH = [(100, 30), (200, 50), (500, 80), (700, 100), (1000, 150), (1500, 300), (2000, 400), (3000, 500),
               (5000, 700), (7000, 1000), (10000, 1500), (15000, 3000), (20000, 4000), (30000, 5000),
               (50000, 7000), (70000, 10000), (100000, 15000), (150000, 30000), (200000, 40000),
               (300000, 50000), (500000, 70000), (700000, 100000), (1000000, 150000)]


def limit_width(p: float) -> float:
    for up, w in LIMIT_WIDTH:
        if p < up:
            return w
    return float("nan")


def main() -> None:
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data" / "processed" / "shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    assert pc.dates[-1] < HOLDOUT
    dates = pc.dates
    months = pd.Series(dates).str[:7].to_numpy()
    P = [int(np.flatnonzero(months == mo).max()) for mo in sorted(set(months)) if FIRST_P <= mo <= LAST_P]
    out: dict = {"data_last_date": str(dates[-1]), "prediction_dates": len(P), "first_P": str(dates[P[0]]),
                 "last_P": str(dates[P[-1]])}

    def next_month_first(t: int) -> int:
        k = t + 1
        while months[k] == months[t]:
            k += 1
        return k

    # (1) σ（ばらつきだけ）
    Oadj = pc.O / pc.cumF
    stds, iqrs, ns = [], [], []
    for t in P[:-1]:
        S = next_month_first(t)
        B = S + 1
        S2 = next_month_first(S)
        js = np.flatnonzero(pc.universe[t])
        entry = Oadj[B, js]
        exit_ = np.where(np.isnan(Oadj[S2, js]), pc.cadj_ff[S2 - 1, js], Oadj[S2, js])
        ok = ~np.isnan(entry) & ~np.isnan(exit_)
        r = exit_[ok] / entry[ok] - 1
        stds.append(float(np.std(r, ddof=1)))
        q75, q25 = np.quantile(r, [0.75, 0.25])
        iqrs.append(float((q75 - q25) / 1.349))
        ns.append(int(ok.sum()))
        del r
    sigma, sigma_iqr = float(np.median(stds)), float(np.median(iqrs))
    n_months = len(stds)
    years = n_months / 12
    noise = []
    for n in (3, 10, 20, 30):
        one = sigma * math.sqrt(12 / n)
        noise.append({"N": n, "one_year_sd": round(one, 4), "one_year_sd_iqr": round(sigma_iqr * math.sqrt(12 / n), 4),
                      "se_annual_mean_over_period": round(one / math.sqrt(years), 4),
                      "se_annual_mean_over_period_iqr": round(sigma_iqr * math.sqrt(12 / n) / math.sqrt(years), 4),
                      "se_12_months": round(one, 4)})
    out["e0_3_sigma"] = {"months": n_months, "years": round(years, 3), "stocks_per_month_median": int(np.median(ns)),
                         "monthly_idio_std_median": round(sigma, 4),
                         "monthly_idio_std_quantiles": {str(q): round(float(np.quantile(stds, q)), 4) for q in (0, .25, .5, .75, 1)},
                         "monthly_idio_std_iqr_median": round(sigma_iqr, 4), "noise": noise,
                         "months_with_std_over_0.2": int(sum(s > 0.2 for s in stds))}

    # (2) 取扱一覧（3列）× 2025-09-26 のユニバース
    lu = pd.read_csv(LINEUP, encoding="utf-8-sig", dtype=str)
    assert list(lu.columns) == ["コード", "銘柄名", "市場"], list(lu.columns)
    buy = {c + "0" for c in lu["コード"]}
    t = int(np.flatnonzero(dates == CHECK)[0])
    js = np.flatnonzero(pc.universe[t])
    codes = pc.codes[js]
    inlist = np.isin(codes, list(buy))
    mcap = mk["shares"][t, js] * pc.C[t, js]
    bands = [("50億円未満", 0, 5e9), ("50〜100億円", 5e9, 1e10), ("100〜200億円", 1e10, 2e10), ("200〜500億円", 2e10, 5.0000001e10)]
    out["e0b_1_coverage"] = {
        "lineup_rows": int(len(lu)), "universe": int(len(js)), "buyable": int(inlist.sum()), "share": round(float(inlist.mean()), 4),
        "by_market_cap": [{"band": b, "n": int(((mcap >= lo) & (mcap < hi)).sum()),
                           "share": round(float(inlist[(mcap >= lo) & (mcap < hi)].mean()), 4)} for b, lo, hi in bands],
        "by_prediction_date_median": round(float(np.median([np.isin(pc.codes[np.flatnonzero(pc.universe[p])], list(buy)).mean() for p in P])), 4),
    }

    # (3) 余力拘束で入る割合
    fr_month = []
    for p in P:
        js_ = np.flatnonzero(pc.universe[p])
        c = pc.C[p, js_]
        bind = np.ceil((c + np.vectorize(limit_width)(c)) * (1 + SPREAD))
        fr_month.append(float(np.mean(c / bind)))
    f1 = float(np.median(fr_month))
    out["e0_binding"] = {"one_stage_invested_fraction_median": round(f1, 4), "two_stage_approx": round(f1 + (1 - f1) * f1, 4),
                         "min_month": round(min(fr_month), 4), "max_month": round(max(fr_month), 4)}

    # (4) 50万円・N=20 で100株を買える割合（v1 の方法：指値 = 終値×1.02 を呼値で切り下げ、100株×指値 ≤ 予算、注文 ≤ 20日平均売買代金の1%）
    def lot_share(p: int) -> float:
        js_ = np.flatnonzero(pc.universe[p])
        lim = np.array([bt2.floor_tick(c * 1.02) for c in pc.C[p, js_]])
        budget = CAPITAL / N
        sh = np.floor(budget / (lim * 100) + 1e-9) * 100
        ok = (sh >= 100) & (sh * lim <= pc.adv[p, js_] * 0.01 + 1e-9)
        return float(ok.mean())
    out["e0_2_lot100_500k_N20"] = {"share_2025_09_26": round(lot_share(t), 4),
                                   "median_prediction_dates": round(float(np.median([lot_share(p) for p in P])), 4)}
    # かぶミニ：1銘柄 24,250 円で100株以上になる銘柄の割合（株価 ≤ 242.5円）
    c = pc.C[t, js]
    out["kabumini_500k_N20"] = {"budget_per_stock": CAPITAL * 0.97 / N,
                                "share_100_or_more_shares_2025_09_26": round(float((np.floor(CAPITAL * 0.97 / N / c) >= 100).mean()), 4),
                                "share_price_above_budget": round(float((c > CAPITAL * 0.97 / N).mean()), 4),
                                "share_order_over_1pct_adv": round(float((CAPITAL * 0.97 / N > pc.adv[t, js] * 0.01).mean()), 4)}

    # (5) 上位20%の枠と N
    sizes = [int(pc.universe[p].sum()) for p in P]
    out["universe_size"] = {"min": min(sizes), "median": int(np.median(sizes)), "max": max(sizes),
                            "top20pct_slots_median": int(np.median([math.floor(s * 0.2 + 1e-9) for s in sizes]))}

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
