"""依頼4の確認：開発役の特徴量の値（project/logs/backtest/features/all_v1.parquet、予測日364日 × ユニバース × 16個）と、
評価役の独立計算（data/phase3_features_indep.npz、src/phase3_features_indep.py）を銘柄・日ごとに照合する。

- 評価役の値は、フェーズ3の独立計算（その時点の記録だけ、「未定」は予定日不明）に、株式数の補正（評価役が CLAUDE.md の文章から
  独立に実装した src/phase4_shares_fix.py の data/processed/shares_fixed.npz）を反映して log_mcap・bp・ep_fcst を計算し直したもの
- ユニバース：評価役の bt2.Precomp（「その他」の除外・案A）に、補正後の株数を使う。開発役の行（日付・銘柄）と集合として照合する
- 値：両方にある行で、NaN の位置と値（相対誤差 1e-9 または絶対誤差 1e-12 以内）を照合する
- 第8章②：開示ごとに「値が変わった日」の照合（bdays_since_report が 0 になった日＝決算短信の利用開始日の週）は、
  評価役の bdays_since_report が開発役と全件一致すれば同時に確かめられる（予測日の時点で最新の短信の利用開始日が一致する）
結果：data/phase4_features_check.json・phase4_features_mismatch.csv
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

DEV = EVAL_ROOT.parent / "project" / "logs" / "backtest" / "features" / "all_v1.parquet"
FEATS = ["ret_5d", "ret_20d", "ret_60d", "vol_ratio_5_60", "volatility_20d", "dist_high_60d", "log_turnover_20d",
         "log_mcap", "ep_fcst", "bp", "sales_growth_yoy", "op_chg_to_sales_yoy", "bdays_since_report", "op_surprise_fy",
         "op_fcst_rev", "cdays_to_next_earnings"]


def main() -> None:
    mk = bt2.load_market()
    shares_old = mk["shares"]
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    dates, codes = pc.dates, pc.codes
    z = np.load(EVAL_ROOT / "data/phase3_features_indep.npz")
    preds = z["preds"]
    dev = pd.read_parquet(DEV)
    dev_dates = np.array(sorted(dev.date.unique()))
    di = {d: i for i, d in enumerate(dates)}
    pos = {int(p): k for k, p in enumerate(preds)}
    sel = [pos[di[d]] for d in dev_dates]          # 開発役の364日が、評価役の454日に含まれていること
    P = preds[sel]
    # ---- 株数の補正を反映
    Cff = pd.DataFrame(mk["C"]).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        mc_old = (shares_old * Cff)[P]
        mc_new = (mk["shares"] * Cff)[P]
    ev = {n: z[n][sel].copy() for n in FEATS}
    with np.errstate(invalid="ignore", divide="ignore"):
        ev["log_mcap"] = np.log(mc_new)
        ev["bp"] = ev["bp"] * mc_old / mc_new
        ev["ep_fcst"] = ev["ep_fcst"] * mc_old / mc_new
    U = pc.universe[P]
    # ---- ユニバースの照合
    ci = {c: j for j, c in enumerate(codes)}
    dk = np.array([np.searchsorted(dev_dates, d) for d in dev.date])
    dj = dev.code.map(ci)
    unknown_codes = int(dj.isna().sum())
    dj = dj.fillna(-1).astype(int).to_numpy()
    dev_mask = np.zeros_like(U)
    ok = dj >= 0
    dev_mask[dk[ok], dj[ok]] = True
    only_eval = U & ~dev_mask
    only_dev = dev_mask & ~U
    res = {"dates": len(dev_dates), "dev_rows": int(len(dev)), "eval_rows": int(U.sum()), "dev_codes_unknown_to_eval": unknown_codes,
           "universe_only_eval": int(only_eval.sum()), "universe_only_dev": int(only_dev.sum()),
           "universe_only_eval_list": [f"{dev_dates[k]}|{codes[j]}" for k, j in zip(*np.nonzero(only_eval))][:50],
           "universe_only_dev_list": [f"{dev_dates[k]}|{codes[j]}" for k, j in zip(*np.nonzero(only_dev))][:50],
           "features": {}}
    # ---- 値の照合（両方にある行）
    both = ok & U[dk, np.maximum(dj, 0)]
    k_, j_ = dk[both], dj[both]
    mism_rows = []
    for n in FEATS:
        a = dev[n].to_numpy()[both]
        b = ev[n][k_, j_]
        na, nb = np.isnan(a), np.isnan(b)
        close = np.isclose(a, b, rtol=1e-9, atol=1e-12)
        nan_mis = na != nb
        val_mis = ~na & ~nb & ~close
        res["features"][n] = {"rows": int(len(a)), "both_nan": int((na & nb).sum()), "nan_only_dev": int((na & ~nb).sum()),
                              "nan_only_eval": int((~na & nb).sum()), "value_mismatch": int(val_mis.sum()),
                              "max_abs_diff_matched": float(np.nanmax(np.abs(a - b)[~na & ~nb & close])) if (~na & ~nb & close).any() else 0.0}
        for i in np.flatnonzero(nan_mis | val_mis):
            mism_rows.append({"feature": n, "date": dev_dates[k_[i]], "code": codes[j_[i]], "dev": a[i], "eval": b[i]})
    mm = pd.DataFrame(mism_rows)
    mm.to_csv(EVAL_ROOT / "data/phase4_features_mismatch.csv", index=False)
    (EVAL_ROOT / "data/phase4_features_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("_list")}, ensure_ascii=False, indent=1))
    if len(mm):
        print(mm.groupby("feature").size())
        print(mm.head(30).to_string())


if __name__ == "__main__":
    main()
