"""フェーズ3の確認：分割・併合の権利落ち日（AdjFactor の日）が期末日以前で、効力発生日が期末日より後のとき、
決算短信の期末発行済株式数は分割前の数なのに、CLAUDE.md の式（期末の株数 × 期末日までの累積積 ÷ t までの累積積）では
分割の調整がされず、株数（時価総額）が分割比率の分だけ小さくなるかを、評価役のデータで確かめる。
結果：data/phase3_shares_check.json・phase3_shares_cases.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from jq import EVAL_ROOT

mk = bt2.load_market()
pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
dates, codes, F = pc.dates, pc.codes, pc.F
ci = {c: j for j, c in enumerate(codes)}
s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                    columns=["DiscDate", "DiscTime", "Code", "DocType", "CurPerEn", "ShOutFY"])
s = s[s.DocType.str.contains("FinancialStatements", na=False)].copy()
s["sh"] = pd.to_numeric(s.ShOutFY, errors="coerce")
s = s[(s.sh > 0) & s.Code.isin(ci)]
# 期末日ごとに最後の開示（訂正後）の株数
s = s.sort_values(["Code", "CurPerEn", "DiscDate", "DiscTime"]).drop_duplicates(["Code", "CurPerEn"], keep="last")
rows = []
ev_t, ev_j = np.nonzero(np.abs(F - 1) > 1e-12)
for t, j in zip(ev_t, ev_j):
    f = F[t, j]
    if 0.8 < f < 1.25:   # 小さな係数（株式無償割当の端数など）は除く
        continue
    code, e = codes[j], dates[t]
    g = s[s.Code == code]
    pe = g.CurPerEn.to_numpy()
    k = np.searchsorted(pe, e, side="left")   # e 以後で最初の期末
    if k == 0 or k >= len(g):
        continue
    p, prev = pe[k], pe[k - 1]
    # 権利落ち日 e から期末日 p までの営業日数
    bd_e, bd_p = t, np.searchsorted(dates, p, side="right") - 1
    ratio = g.sh.iloc[k] / g.sh.iloc[k - 1]
    nxt = g.sh.iloc[k + 1] / g.sh.iloc[k] if k + 1 < len(g) else np.nan
    rows.append({"code": code, "ex_date": e, "factor": f, "period_end": p, "bdays_ex_to_pe": bd_p - bd_e,
                 "sh_ratio_at_pe": ratio, "sh_ratio_next": nxt, "expected": 1 / f})
df = pd.DataFrame(rows)
df["reflected"] = np.isclose(df.sh_ratio_at_pe, df.expected, rtol=0.05)
df["not_reflected"] = np.isclose(df.sh_ratio_at_pe, 1.0, rtol=0.05) & ~df.reflected
df.to_csv(EVAL_ROOT / "data/phase3_shares_cases.csv", index=False)
out = {"events": len(df)}
for lo, hi in [(0, 0), (1, 1), (2, 3), (4, 20), (21, 9999)]:
    x = df[(df.bdays_ex_to_pe >= lo) & (df.bdays_ex_to_pe <= hi)]
    out[f"ex_to_pe_{lo}-{hi}bd"] = {"n": len(x), "reflected": int(x.reflected.sum()), "not_reflected": int(x.not_reflected.sum())}
# 影響：not_reflected の事例で、株数が小さく出る期間（その期末の短信が使える日 → 次の期末の短信が使える日）の
# ユニバースへの影響を、正しい株数（÷ 係数）で作り直して比べる
shares = mk["shares"].copy()
fixed = shares.copy()
bad = df[df.not_reflected]
for _, r in bad.iterrows():
    j = ci[r.code]
    g = s[s.Code == r.code]
    # その期末の株数が使われている日：market2 の株数 × 係数 ≈ 分割前の株数になっている日
    row = g[g.CurPerEn == r.period_end].iloc[0]
    sh_pe = row.sh
    t0 = int(np.searchsorted(dates, r.ex_date))
    seg = np.where(np.isclose(shares[:, j], sh_pe * pc.cumF[t0, j] / pc.cumF[:, j] / (pc.cumF[t0, j] / pc.cumF[t0 - 1, j]) , rtol=1e-6))[0]
    # 上の式は「期末日までの累積積」が係数を含むときの値 = sh_pe × cumF[pe]/cumF[t] と同じ。素直に計算し直す
    pe_idx = np.searchsorted(dates, r.period_end, side="right") - 1
    wrong = sh_pe * pc.cumF[pe_idx, j] / pc.cumF[:, j]
    days = np.where(np.isclose(shares[:, j], wrong, rtol=1e-9))[0]
    days = days[days > pe_idx]
    fixed[days, j] = shares[days, j] / r.factor
C = mk["C"]
mc_bad, mc_fix = shares * C, fixed * C
uni = pc.universe
preds = np.array([w[0] - 1 for w in pc.weeks])
affected = (fixed != shares) & ~np.isnan(shares)
out["not_reflected_events"] = int(len(bad))
out["stock_days_affected"] = int(affected.sum())
out["stock_days_affected_in_universe"] = int((affected & uni).sum())
with np.errstate(invalid="ignore"):
    wrongly_in = uni & affected & (mc_fix > bt2.MAX_MCAP)
out["universe_days_wrongly_included_mcap_over_50bn"] = int(wrongly_in.sum())
out["pred_days_wrongly_included"] = int(wrongly_in[preds].sum())
out["pred_days_affected_in_universe"] = int((affected & uni)[preds].sum())
out["pred_universe_cells_total"] = int(uni[preds].sum())
out["examples"] = bad.head(10).to_dict("records")
(EVAL_ROOT / "data/phase3_shares_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "examples"}, ensure_ascii=False, indent=1))
print(bad.head(10).to_string())
