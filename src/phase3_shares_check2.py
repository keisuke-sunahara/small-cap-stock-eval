"""フェーズ3の確認（その2）：開示済みの発行済株式数を「データの最初の日の株数の基準」に直した値
X_k = 期末の株数 × 期末日までの累積積（CLAUDE.md の式の分子）が、前後の短信と比べて分割比率だけずれている短信を探す。
X は分割では変わらず、増資・自社株の消却でだけ変わるはずなので、前後と比べて ±40% 以上ずれ、前後どうしはそろっている短信を
「株数の基準の誤り」とみなす。そのずれの期間を正しい値（前後の平均）に直してユニバースへの影響を数える。
結果：data/phase3_shares_check2.json・phase3_shares_bad_reports.csv
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2
from market2 import availability_date
from jq import EVAL_ROOT

mk = bt2.load_market()
pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
dates, codes, F, cumF = pc.dates, pc.codes, pc.F, pc.cumF
T = len(dates)
ci = {c: j for j, c in enumerate(codes)}
di = {d: i for i, d in enumerate(dates)}
s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                    columns=["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "CurPerEn", "ShOutFY"])
s = s[s.DocType.str.contains("FinancialStatements", na=False)].copy()
s["sh"] = pd.to_numeric(s.ShOutFY, errors="coerce")
s = s[(s.sh > 0) & s.Code.isin(ci)].copy()
s["avail"] = availability_date(s.DiscDate, s.DiscTime, list(dates))
s = s[s.avail.isin(di)].copy()
s["av"] = s.avail.map(di)
s["pe"] = np.clip(np.searchsorted(dates, s.CurPerEn.astype(str).to_numpy(), side="right") - 1, 0, T - 1)
s["j"] = s.Code.map(ci)
s["X"] = s.sh.to_numpy() * cumF[s.pe.to_numpy(), s.j.to_numpy()]
s = s.sort_values(["Code", "av", "DiscDate", "DiscTime", "DiscNo"]).drop_duplicates(["Code", "av"], keep="last")
bad_rows = []
for code, g in s.groupby("Code", sort=False):
    X = g.X.to_numpy(); n = len(X)
    for k in range(1, n - 1):
        a, b = X[k] / X[k - 1], X[k] / X[k + 1]
        if (abs(np.log(a)) > np.log(1.4) and abs(np.log(b)) > np.log(1.4) and np.sign(np.log(a)) == np.sign(np.log(b))
                and abs(np.log(X[k - 1] / X[k + 1])) < np.log(1.15)):
            r = g.iloc[k]
            j = int(r.j)
            # 期末日の後〜開示日に権利落ちがある分割（遡って反映された疑い）か、期末日以前の直前の分割か
            ev_after = [(dates[t], F[t, j]) for t in range(int(r.pe) + 1, int(r.av) + 1) if abs(F[t, j] - 1) > 1e-12]
            ev_before = [(dates[t], F[t, j]) for t in range(max(int(r.pe) - 5, 0), int(r.pe) + 1) if abs(F[t, j] - 1) > 1e-12]
            bad_rows.append({"code": code, "period_end": r.CurPerEn, "disc": r.DiscDate, "ratio_vs_prev": a,
                             "ratio_vs_next": b, "split_after_pe_before_disc": ev_after, "split_0to5bd_before_pe": ev_before,
                             "j": j, "av": int(r.av), "av_next": int(g.av.iloc[k + 1]), "X_ok": float(np.sqrt(X[k - 1] * X[k + 1])),
                             "X_bad": float(X[k])})
bad = pd.DataFrame(bad_rows)
bad.drop(columns=["j", "av", "av_next"]).to_csv(EVAL_ROOT / "data/phase3_shares_bad_reports.csv", index=False)
cat = {"split_after_pe_before_disc（遡って反映）": int((bad.split_after_pe_before_disc.str.len() > 0).sum()),
       "split_0to5bd_before_pe（効力発生日が期末後）": int(((bad.split_after_pe_before_disc.str.len() == 0) & (bad.split_0to5bd_before_pe.str.len() > 0)).sum()),
       "その他": int(((bad.split_after_pe_before_disc.str.len() == 0) & (bad.split_0to5bd_before_pe.str.len() == 0)).sum())}
shares = mk["shares"]
fixed = shares.copy()
for _, r in bad.iterrows():
    fixed[r.av:r.av_next, r.j] = shares[r.av:r.av_next, r.j] * r.X_ok / r.X_bad
C = mk["C"]
uni = pc.universe
preds = np.array([w[0] - 1 for w in pc.weeks])
aff = fixed != shares
with np.errstate(invalid="ignore"):
    mc_fix = fixed * C
    true_uni = uni | (pc.listed & ~np.isnan(C) & (C >= bt2.MIN_PRICE) & (pc.adv >= bt2.MIN_TURNOVER) & (mc_fix <= bt2.MAX_MCAP)
                      & ~mk["margin_other"])
    wrong_in = uni & aff & (mc_fix > bt2.MAX_MCAP)
    wrong_out = ~uni & aff & (mc_fix <= bt2.MAX_MCAP) & pc.listed & ~np.isnan(C) & (C >= bt2.MIN_PRICE) & (pc.adv >= bt2.MIN_TURNOVER) & ~mk["margin_other"]
out = {"reports_checked": int(len(s)), "bad_reports": int(len(bad)), "codes": int(bad.code.nunique()),
       "by_cause": cat,
       "overstated（株数が大きすぎ）": int((bad.ratio_vs_prev > 1).sum()), "understated（小さすぎ）": int((bad.ratio_vs_prev < 1).sum()),
       "stock_days_affected": int(aff.sum()),
       "pred_days: wrongly_included（本当は500億円超）": int(wrong_in[preds].sum()),
       "pred_days: wrongly_excluded（本当は500億円以下）": int(wrong_out[preds].sum()),
       "pred_universe_cells_total": int(uni[preds].sum()),
       "pred_days_in_universe_with_wrong_shares（log_mcap・E/P・B/P がずれる）": int((aff & uni)[preds].sum()),
       "by_year_bad_reports": bad.disc.str[:4].value_counts().sort_index().to_dict()}
(EVAL_ROOT / "data/phase3_shares_check2.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
print(bad.drop(columns=["j", "av", "av_next", "X_ok", "X_bad"]).head(12).to_string())
