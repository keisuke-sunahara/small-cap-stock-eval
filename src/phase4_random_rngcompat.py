"""ランダム選択の照合：乱数の引き方を開発役に合わせ（bt2.simulate の rng_compat）、乱数シード0〜999で開発役の
reports/phase3r_baselines.json の平均・分位点と比べる。エンジンが同じなら、平均は計算の誤差の範囲で一致するはず。
結果：data/phase4_random_rngcompat.json"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from jq import EVAL_ROOT
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
mk = bt2.load_market()
mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
dev = json.load(open(EVAL_ROOT.parent / "project/reports/phase3r_baselines.json", encoding="utf-8"))["random"]
COSTS = {"cost_0.3%": 0.003, "cost_tick": "tick"}
out, t0 = {}, time.time()
for bm in ("fixed", "min_equity"):
    for name, c in COSTS.items():
        a = np.array([bt2.metrics((r := bt2.simulate(pc, "random", c, budget_mode=bm, seed=s, record=False, rng_compat=True)).weekly["ret"],
                                  years=bt2.years_of(r.weekly))["annual_return"] for s in range(runs)])
        d = dev[bm][name]
        out[f"{bm}/{name}"] = {"eval_mean": float(a.mean()), "dev_mean": d["mean"]["annual_return"],
                               "eval_pct": {q: float(np.quantile(a, float(q))) for q in d["percentiles_annual_return"]},
                               "dev_pct": d["percentiles_annual_return"], "runs": runs}
        print(bm, name, out[f"{bm}/{name}"]["eval_mean"], d["mean"]["annual_return"], f"{time.time()-t0:.0f}s", flush=True)
(EVAL_ROOT / "data/phase4_random_rngcompat.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
