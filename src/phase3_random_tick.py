"""フェーズ3の照合：ランダム選択・銘柄ごとのコスト（max(0.3%, 1呼値÷約定価格)）・案A込みで200回。開発役の平均と比べる。"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from jq import EVAL_ROOT
runs = int(sys.argv[1]) if len(sys.argv) > 1 else 200
pc = bt2.Precomp(bt2.load_market(), exclude_margin_other=True, case_a=True)
out = {}
t0 = time.time()
for bm in ("fixed", "min_equity"):
    a = []
    for s in range(runs):
        r = bt2.simulate(pc, "random", "tick", budget_mode=bm, seed=20_000 + s, record=False)
        a.append(bt2.metrics(r.weekly["ret"], years=bt2.years_of(r.weekly))["annual_return"])
    a = np.array(a)
    out[bm] = {"mean": float(a.mean()), "se": float(a.std(ddof=1) / np.sqrt(runs)), "annual": a.tolist()}
    print(bm, a.mean(), a.std(ddof=1) / np.sqrt(runs), f"{time.time()-t0:.0f}s", flush=True)
(EVAL_ROOT / "data" / "phase3_random_tick.json").write_text(json.dumps(out), encoding="utf-8")
