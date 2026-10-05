"""株式数の補正の範囲・許容幅を変えた場合の比較（監査用。「次の短信」を使うので売買には使えない）。
開発役の設定（許容5%、(a) の終わりは開示日）に対して、許容10%・15%、(a) の終わりを利用開始日にした場合で、
- 補正した件数、前後の短信とずれたまま残る件数（neighbor_audit）
- 補正が誤りの疑い（補正後の値より補正前の値の方が次の短信に近い）
を数える。結果：data/phase4_shares_variants.json
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2, phase4_shares_fix as f
from jq import EVAL_ROOT

mk = bt2.load_market()
s0, cumF = f.build_records(mk)
res = {}
for tol in (0.05, 0.10, 0.15, 0.25):
    for end in ("disc", "av"):
        s = f.apply_fix(s0, cumF, tol=tol, end_col=end)
        s["flag_raw"] = f.neighbor_audit(s, "base_raw")
        s["flag_fix"] = f.neighbor_audit(s, "base")
        worse, better, nonext = [], 0, 0
        for code, g in s.groupby("Code", sort=False):
            g = g.drop_duplicates("av", keep="last").reset_index(drop=True)
            for k in np.flatnonzero(g.fix.ne("")):
                if k + 1 >= len(g):
                    nonext += 1
                    continue
                nx = g.base.iat[k + 1]
                a, b = abs(np.log(g.base.iat[k] / nx)), abs(np.log(g.base_raw.iat[k] / nx))
                if a > b:
                    worse.append(f"{code}|{g.DiscDate.iat[k]}|{np.exp(a):.3f}")
                else:
                    better += 1
        res[f"tol{tol}_{end}"] = {"fixed": int((s.fix != "").sum()), "flag_after": int(s.flag_fix.sum()),
                                  "fix_closer_to_next": better, "fix_worse_than_raw": len(worse), "no_next": nonext,
                                  "worse_list": worse}
        print(tol, end, {k: v for k, v in res[f"tol{tol}_{end}"].items() if k != "worse_list"}, flush=True)
(EVAL_ROOT / "data/phase4_shares_variants.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
