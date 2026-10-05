"""フェーズ4前半の補足（評価役）：EXP-001 と EXP-002 の Rank IC の週ごとの差の t値、EXP-001 の約定のマーケット・インパクトの見積もり、
EXP-001 と EXP-002 の点数の順位相関（予測日）。結果：data/phase4_extra.json"""
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from jq import EVAL_ROOT
from phase2_cost import sigma20
from phase4_model_check import load_scores, rank_ic

mk = bt2.load_market()
mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
out = {}
S = {e: load_scores(pc, e)[0] for e in ("EXP-001", "EXP-002")}
ic = {e: np.array(rank_ic(pc, S[e])["weekly"]) for e in S}
d = ic["EXP-001"] - ic["EXP-002"]
d = d[np.isfinite(d)]
out["ic_diff"] = {"mean": float(d.mean()), "t": float(d.mean() / d.std(ddof=1) * np.sqrt(len(d)))}
cors = []
for w in pc.weeks:
    p = w[0] - 1
    idx = np.flatnonzero(pc.universe[p])
    a, b = S["EXP-001"][p, idx], S["EXP-002"][p, idx]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() > 2:
        cors.append(spearmanr(a[ok], b[ok]).statistic)
out["score_rank_corr_001_002"] = {"mean": float(np.mean(cors)), "min": float(np.min(cors))}
# 実際に注文した銘柄の重なり
o1 = pd.read_csv(EVAL_ROOT / "data/phase4_model_weekly/EXP-001_trades.csv", dtype={"code": str})
o2 = pd.read_csv(EVAL_ROOT / "data/phase4_model_weekly/EXP-002_trades.csv", dtype={"code": str})
s1 = set(zip(o1.buy_date, o1.code)); s2 = set(zip(o2.buy_date, o2.code))
out["order_overlap"] = {"EXP-001": len(s1), "EXP-002": len(s2), "both": len(s1 & s2)}
sig = sigma20(pc)
va = np.nan_to_num(pc.mk["Va"], nan=0.0)
di = {x: i for i, x in enumerate(pc.dates)}; ci = {c: j for j, c in enumerate(pc.codes)}
tr = o1[o1.how != "not_filled"]
rows = []
for x in tr.itertuples():
    j, tb, ts = ci[x.code], di[x.buy_date], di[x.sell_date]
    rows.append((sig[tb - 1, j], x.buy_price * x.shares_buy / va[tb, j] if va[tb, j] > 0 else np.nan,
                 x.sell_price * x.shares_sell / va[ts, j] if va[ts, j] > 0 else np.nan, x.buy_price))
g = np.array(rows)
imp = {}
for s in (0.05, 0.10, 0.15):
    a = np.r_[g[:, 0] * np.sqrt(np.clip(g[:, 1], None, 1) / s), g[:, 0] * np.sqrt(np.clip(g[:, 2], None, 1) / s)]
    imp[str(s)] = {"mean": float(np.nanmean(a)), "median": float(np.nanmedian(a))}
out["impact_EXP-001"] = {"trades": len(g), "price_median": float(np.median(g[:, 3])), "sigma20_median": float(np.nanmedian(g[:, 0])),
                         "sqrt_model_one_way": imp}
print(json.dumps(out, indent=1))
(EVAL_ROOT / "data/phase4_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
# 点数の上位3のうち100株を買えない（指値 × 100株 > 30万円 ÷ 3）割合（EXP-002）
M = S["EXP-002"]
picks = unaff = 0
for w in pc.weeks:
    p = w[0] - 1
    rk = pc.ranking(p, "EXP-002") if "EXP-002" in pc.score else None
    if rk is None:
        pc.score["EXP-002"] = M
        rk = pc.ranking(p, "EXP-002")
    for j in rk[:3]:
        picks += 1
        unaff += bt2.floor_tick(pc.C[p, j] * 1.02) * 100 > 100_000
out["top3_unaffordable_EXP-002"] = {"picks": picks, "unaffordable": int(unaff), "ratio": unaff / picks}
print(out["top3_unaffordable_EXP-002"])
(EVAL_ROOT / "data/phase4_extra.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
