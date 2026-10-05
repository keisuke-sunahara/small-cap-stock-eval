"""補正の後にも残る株数のずれ（監査用の検出：前後の両方の短信と10%超ずれ、前後は2%以内でそろう）が、
予測日（364日）のユニバースにどれだけ影響するかを数える。正しい値は前後の短信の幾何平均とみなす（監査用。未来の短信を使う）。
結果：data/phase4_shares_residual.json"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2, phase4_shares_fix as f
from jq import EVAL_ROOT
mk = bt2.load_market()
s, cumF = f.build_records(mk)
s = f.apply_fix(s, cumF)
T, N = cumF.shape
shares = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
true = shares.copy()
rows = []
for code, g in s.groupby("Code", sort=False):
    g = g.drop_duplicates("av", keep="last").reset_index(drop=True)
    x = g.base.to_numpy()
    for k in range(1, len(g) - 1):
        a, b = np.log(x[k] / x[k - 1]), np.log(x[k] / x[k + 1])
        if abs(a) > np.log(1.1) and abs(b) > np.log(1.1) and abs(np.log(x[k - 1] / x[k + 1])) < 0.02:
            j, t0, t1 = int(g.j.iat[k]), int(g.av.iat[k]), int(g.av.iat[k + 1])
            ok = np.sqrt(x[k - 1] * x[k + 1])
            true[t0:t1, j] = shares[t0:t1, j] * ok / x[k]
            j_ = j
            F = mk["F"][:, j]
            near = [str(mk["dates"][t]) for t in range(max(int(g.pe.iat[k]) - 10, 0), min(t1, T)) if abs(F[t] - 1) > 1e-12]
            rows.append({"code": code, "disc": g.DiscDate.iat[k], "ratio": float(np.exp(a)), "fix": g.fix.iat[k], "splits_near": near})
mk2 = dict(mk); mk2["shares"] = shares
pc = bt2.Precomp(mk2, exclude_margin_other=True, case_a=True)
mk3 = dict(mk); mk3["shares"] = true
pc3 = bt2.Precomp(mk3, exclude_margin_other=True, case_a=True)
P = [w[0] - 1 for w in pc.weeks]
U, U3 = pc.universe[P], pc3.universe[P]
r = pd.DataFrame(rows)
out = {"residual_reports": len(r), "codes": int(r.code.nunique()),
       "with_split_nearby": int((r.splits_near.str.len() > 0).sum()),
       "ratio_dist": r.ratio.describe().to_dict(),
       "pred_universe_wrongly_in": int((U & ~U3).sum()), "pred_universe_wrongly_out": int((~U & U3).sum()),
       "pred_universe_cells": int(U.sum()),
       "pred_cells_in_universe_with_wrong_shares": int((U & (np.abs(np.log(true / shares)) > 1e-9))[P].sum()) if False else
           int((U & ~np.isclose(true[P], shares[P], equal_nan=True)).sum())}
r.to_csv(EVAL_ROOT / "data/phase4_shares_residual.csv", index=False)
(EVAL_ROOT / "data/phase4_shares_residual.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
print(r.to_string())
