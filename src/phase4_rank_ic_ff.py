"""Rank IC の照合（実現リターンの終値に、終値が無い日は直前の終値を使う版）。開発役の weekly_realized の扱いに合わせた確認。"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from jq import EVAL_ROOT
mk = bt2.load_market()
mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
dev = json.load(open(EVAL_ROOT.parent / "project/reports/phase3r_baselines.json", encoding="utf-8"))["rank_ic"]
out = {}
for kind in ("momentum_20d", "reversal_5d"):
    ics = []
    for w in pc.weeks:
        p, a, b = w[0] - 1, w[0], w[-1]
        idx = np.flatnonzero(pc.universe[p])
        s = pc.score[kind][p, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = pc.cadj_ff[b, idx] / (pc.O[a, idx] / pc.cumF[a, idx]) - 1
        ok = np.isfinite(s) & np.isfinite(r)
        ics.append(spearmanr(s[ok], r[ok]).statistic)
    ics = np.array(ics)
    out[kind] = {"eval_mean": float(ics.mean()), "eval_std": float(ics.std(ddof=1)),
                 "eval_t": float(ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))), "dev": dev[kind]}
    print(kind, out[kind])
(EVAL_ROOT / "data/phase4_rank_ic_ff.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
