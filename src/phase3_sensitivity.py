"""フェーズ3：分布の比較で、開示の時刻の扱いの誤りをどこまで見つけられるか（感度）。

評価役の計算（決算発表からの営業日数 bdays_since_report、純資産の B/P の値のある割合）に、わざと次の誤りを入れ、
開発役の集計（値のある割合・分位点）との差が出るかを見る。
- A：大引けの時刻の変更を無視（いつも15:30で判定。2024-11-05 より前の 15:00〜15:29 の開示を当日から使う）
- B：大引け後の開示もすべて当日から使う
- C：土日・祝日の開示を、直前の営業日（金曜など）から使う
- D：決算発表予定日を公表日の当日から使う（cdays_to_next_earnings）
結果：data/phase3_sensitivity.json
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
from market2 import availability_date  # noqa: E402

DEV = json.load(open(EVAL_ROOT.parent / "project" / "reports" / "phase3_features.json", encoding="utf-8"))
QS = ("0.01", "0.05", "0.25", "0.5", "0.75", "0.95", "0.99")


def main() -> None:
    mk = bt2.load_market()
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=False)
    dates = pc.dates
    T, N = len(dates), len(pc.codes)
    ci = {c: j for j, c in enumerate(pc.codes)}
    di = {d: i for i, d in enumerate(dates)}
    bd = np.array(dates)
    z = np.load(EVAL_ROOT / "data/phase3_features_indep.npz")
    preds, U = z["preds"], z["universe"]
    s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                        columns=["DiscDate", "DiscTime", "Code", "DiscNo", "DocType"])
    s = s[s.Code.isin(ci) & s.DocType.str.contains("FinancialStatements", na=False)].copy()
    d = s.DiscDate.astype(str).to_numpy()
    hh = s.DiscTime.astype(str).str[:5].to_numpy()
    is_bd = np.isin(d, bd)
    same = np.searchsorted(bd, d, side="left")
    nxt = np.searchsorted(bd, d, side="right")
    prev = nxt - 1
    base = availability_date(s.DiscDate, s.DiscTime, list(dates)).map(di).fillna(T).astype(int).to_numpy()
    variants = {
        "正しい扱い": base,
        "A：いつも15:30で判定": np.where(is_bd & (hh < "15:30"), same, nxt),
        "B：大引け後も当日": np.where(is_bd, same, nxt),
        "C：休日の開示を前の営業日から": np.where(is_bd, base, prev),
    }
    out = {}
    j_all = s.Code.map(ci).to_numpy()
    for name, av in variants.items():
        changed = int((av != base).sum())
        df = pd.DataFrame({"av": av, "j": j_all}).query("av < @T")
        last = np.full((T, N), -1.0)
        last[df.av.to_numpy(), df.j.to_numpy()] = df.av.to_numpy()
        last = np.maximum.accumulate(last, axis=0)
        b = np.where(last >= 0, np.arange(T)[:, None] - last, np.nan)[preds]
        v = b[U]
        v = v[np.isfinite(v)]
        q = {k: float(np.quantile(v, float(k))) for k in QS}
        dev_q = DEV["features"]["bdays_since_report"]["quantiles"]
        out[name] = {"disclosures_changed": changed, "quantiles": q,
                     "quantiles_differ_from_dev": [k for k in QS if abs(q[k] - dev_q[k]) > 1e-9],
                     "share_zero": float((v == 0).mean()), "mean": float(v.mean())}
    # D：決算発表予定日を公表日の当日から（PubDate ≤ 予測日）
    e = pd.read_parquet(EVAL_ROOT / "data/processed/earnings_date_all.parquet")
    e = e[e.Code.isin(ci) & (e.SchDate.astype(str).str.len() == 10)].copy()
    e["j"] = e.Code.map(ci)
    e = e.sort_values("PubDate", kind="stable")
    nx = np.full((len(preds), N), np.nan)
    for k, p in enumerate(preds):
        dd = str(dates[p])
        x = e[e.PubDate <= dd].groupby(["j", "FYE", "FQName"]).tail(1)
        x = x[x.SchDate > dd]
        m = x.groupby("j").SchDate.min()
        nx[k, m.index.to_numpy()] = (pd.to_datetime(m.to_numpy()) - pd.Timestamp(dd)).days
    cov = float(np.mean([np.isfinite(nx[k][U[k]]).mean() for k in range(len(preds))]))
    v = nx[U]
    v = v[np.isfinite(v)]
    q = {k: float(np.quantile(v, float(k))) for k in QS}
    dv = DEV["features"]["cdays_to_next_earnings"]
    out["D：予定日を公表日の当日から"] = {"coverage": cov, "dev_coverage": dv["coverage_mean"], "quantiles": q,
                                    "quantiles_differ_from_dev": [k for k in QS if abs(q[k] - dv["quantiles"][k]) > 1e-9]}
    (EVAL_ROOT / "data/phase3_sensitivity.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, r in out.items():
        print(k, json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
