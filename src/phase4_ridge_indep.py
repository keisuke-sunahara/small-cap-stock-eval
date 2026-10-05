"""フェーズ4前半の確認6：EXP-002（リッジ回帰）を、experiments/EXP-002_ridge/plan.md・config.yaml、experiments/README.md 第2章、
project/CLAUDE.md 第5章の文章を仕様として、評価役のデータとコードで学習し直し、開発役の点数（scores.parquet）と照合する。
開発役のコードは使わない（リッジ回帰も numpy で自分で解く）。

- 特徴量：data/phase4_features_daily.npz（評価役が全営業日について計算。予測日の値は開発役と全セル一致を確認済み）。
  日ごとにユニバース内の順位（同順位は平均の順位 ÷ 値のある銘柄数）。欠損は 0.5
- 目的変数：起点 t の翌営業日の始値 → t の5営業日後の終値（調整後。終値が無い日は直前の終値）のリターンの、t のユニバース内の
  順位（平均の順位 ÷ 銘柄数）。翌営業日に始値が無い行は使わず、順位の銘柄数にも入れない。
  案A：目的変数の期間に、判定の対象の係数の日（起点の2〜5営業日後）がある行の目的変数は使わない
- 学習の範囲：月ごとに、最初の予測日 F の6営業日前まで、F の4年前の日付以降の最初の営業日（2017-01-01 以降）から
- リッジ回帰：alpha = 1.0、切片あり（切片には罰則をかけない）。X・y を平均で中心化して (X'X + αI)β = X'y を解く
- 点数を付ける日 d のモデル：F ≤ d となる最も新しい月
結果：data/phase4_ridge_indep.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402

PROJ = EVAL_ROOT.parent / "project"
FEATS = ["ret_5d", "ret_20d", "ret_60d", "vol_ratio_5_60", "volatility_20d", "dist_high_60d", "log_turnover_20d",
         "log_mcap", "ep_fcst", "bp", "sales_growth_yoy", "op_chg_to_sales_yoy", "bdays_since_report", "op_surprise_fy",
         "op_fcst_rev", "cdays_to_next_earnings"]
ALPHA = 1.0
H = 5


def bad_coef_days(pc: bt2.Precomp) -> np.ndarray:
    """案Aの判定の対象になった係数の日：判定の日 t_e（係数の日以降で最初に売買が成立した日）ごとに、その前に売買が成立した日 s より後で
    t_e 以前の、係数が1でない日。"""
    C, F = pc.C, pc.F
    out = np.zeros(C.shape, dtype=bool)
    for te, j in zip(*np.nonzero(pc.case_a_known)):
        t = te
        while t >= 0:
            if abs(F[t, j] - 1) > 1e-12:
                out[t, j] = True
            if t < te and not np.isnan(C[t, j]):
                break
            t -= 1
    return out


def main() -> None:
    t0 = time.time()
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    dates = pc.dates
    T, N = pc.C.shape
    di = {d: i for i, d in enumerate(dates)}
    z = np.load(EVAL_ROOT / "data/phase4_features_daily.npz")
    rt, rj = z["t"].astype(int), z["j"].astype(int)
    # ---- 特徴量の順位（日ごと）
    X = np.empty((len(rt), len(FEATS)))
    bounds = np.r_[0, np.flatnonzero(np.diff(rt)) + 1, len(rt)]
    assert np.all(np.diff(rt) >= 0)
    for k, n in enumerate(FEATS):
        v = z[n]
        col = np.full(len(v), 0.5)
        for a, b in zip(bounds[:-1], bounds[1:]):
            x = v[a:b]
            ok = np.isfinite(x)
            if ok.sum():
                col[a:b][ok] = rankdata(x[ok], method="average") / ok.sum()
        X[:, k] = col
    print("ranks done", f"{time.time() - t0:.0f}s", flush=True)
    # ---- 目的変数
    bad = bad_coef_days(pc)
    badw = np.zeros((T, N), dtype=bool)   # 起点 t の2〜5営業日後に判定の対象の係数の日がある
    for k in range(2, H + 1):
        badw[:-k] |= bad[k:]
    variants = {}
    for mode in ("rank_then_drop", "drop_then_rank"):
        y = np.full(len(rt), np.nan)
        for a, b in zip(bounds[:-1], bounds[1:]):
            t = rt[a]
            if t + H >= T:
                continue
            js = rj[a:b]
            with np.errstate(invalid="ignore", divide="ignore"):
                r = pc.cadj_ff[t + H, js] / (pc.O[t + 1, js] / pc.cumF[t + 1, js]) - 1
            ok = np.isfinite(r)
            if mode == "drop_then_rank":
                ok &= ~badw[t, js]
            if ok.sum():
                yy = np.full(len(js), np.nan)
                yy[ok] = rankdata(r[ok], method="average") / ok.sum()
                yy[badw[t, js]] = np.nan
                y[a:b] = yy
        variants[mode] = y
    print("target done", f"{time.time() - t0:.0f}s", "bad rows", int(badw[rt, rj].sum()), flush=True)
    # ---- 月ごとの学習
    preds = [w[0] - 1 for w in pc.weeks]
    months = sorted({str(dates[p])[:7] for p in preds})
    first = {m: min(p for p in preds if str(dates[p])[:7] == m) for m in months}
    firsts = np.array([first[m] for m in months])
    dev = pd.read_parquet(PROJ / "logs/backtest/EXP-002/scores.parquet")
    dsp = pd.read_csv(PROJ / "logs/backtest/EXP-002/splits.csv")
    ci = {c: j for j, c in enumerate(pc.codes)}
    dev_t = dev["date"].map(di).to_numpy()
    dev_j = dev["code"].map(ci).to_numpy()
    dev_m = np.searchsorted(firsts, dev_t, side="right") - 1
    row_index = {(t, j): i for i, (t, j) in enumerate(zip(rt, rj))}
    dev_row = np.array([row_index.get((t, j), -1) for t, j in zip(dev_t, dev_j)])
    out = {"rows_features": int(len(rt)), "dev_rows_not_in_eval_universe": int((dev_row < 0).sum()), "variants": {}}
    for mode, y in variants.items():
        score = np.full(len(dev), np.nan)
        train_rows = []
        for i, m in enumerate(months):
            F = first[m]
            lo_date = max(pd.Timestamp(str(dates[F])) - pd.DateOffset(years=4), pd.Timestamp("2017-01-01"))
            lo = int(np.searchsorted(dates, lo_date.strftime("%Y-%m-%d")))
            hi = F - 6
            sel = (rt >= lo) & (rt <= hi) & np.isfinite(y)
            Xt, yt = X[sel], y[sel]
            mx, my = Xt.mean(axis=0), yt.mean()
            Xc = Xt - mx
            beta = np.linalg.solve(Xc.T @ Xc + ALPHA * np.eye(Xc.shape[1]), Xc.T @ (yt - my))
            b0 = my - mx @ beta
            rows = np.flatnonzero((dev_m == i) & (dev_row >= 0))
            score[rows] = X[dev_row[rows]] @ beta + b0
            train_rows.append({"month": m, "eval_rows": int(sel.sum()),
                               "dev_rows": int(dsp.loc[dsp["month"] == m, "train_rows"].iloc[0])})
        diff = np.abs(score - dev["score"].to_numpy())
        tr = pd.DataFrame(train_rows)
        cors = []
        for d_, g in pd.DataFrame({"d": dev_t, "a": score, "b": dev["score"].to_numpy()}).groupby("d"):
            cors.append(spearmanr(g.a, g.b).statistic)
        by_month = pd.DataFrame({"m": dev_m, "diff": diff}).groupby("m")["diff"].max()
        out["variants"][mode] = {"max_abs_diff": float(np.nanmax(diff)), "median_abs_diff": float(np.nanmedian(diff)),
                                 "rows_diff_gt_1e-9": int((diff > 1e-9).sum()), "rows": int(len(diff)),
                                 "rank_corr_by_day_min": float(np.min(cors)), "rank_corr_by_day_mean": float(np.mean(cors)),
                                 "train_rows_equal_months": int((tr.eval_rows == tr.dev_rows).sum()), "months": len(tr),
                                 "train_rows_max_abs_diff": int((tr.eval_rows - tr.dev_rows).abs().max()),
                                 "months_maxdiff_gt_1e-9": [months[k] for k, v in by_month.items() if v > 1e-9][:20]}
        print(mode, out["variants"][mode], f"{time.time() - t0:.0f}s", flush=True)
        # 評価役の点数で売買を計算し、開発役の週次リターンと比べる（予算固定・基本ルール × 一律0.3%・銘柄ごと）
        M = np.full((T, N), np.nan)
        M[dev_t, dev_j] = score
        pc.score["ridge_eval"] = M
        dwr = pd.read_csv(PROJ / "logs/backtest/EXP-002/weekly_returns.csv")
        bt = {}
        for bm in ("fixed", "min_equity"):
            for cname, c in (("cost_0.3%", 0.003), ("cost_tick", "tick")):
                w = bt2.simulate(pc, "ridge_eval", c, budget_mode=bm, record=False).weekly
                d = np.abs(w["ret"].to_numpy() - dwr[f"{bm}|{cname}"].to_numpy())
                bt[f"{bm}|{cname}"] = {"weeks_diff_gt_1e-9": int((d > 1e-9).sum()), "max_abs_diff": float(d.max()),
                                       "annual": bt2.metrics(w["ret"], years=bt2.years_of(w))["annual_return"]}
        out["variants"][mode]["backtest_with_eval_scores"] = bt
        print(mode, bt, flush=True)
    (EVAL_ROOT / "data/phase4_ridge_indep.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
