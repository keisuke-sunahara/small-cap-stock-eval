"""フェーズ3の独立検証：開発役の特徴量（reports/phase3.md 第2章・registry の説明文）を、評価役が自分のデータと
自分のコードで「その予測日の時点で入手できた記録だけ」から計算し、開発役の集計（reports/phase3_features.json）と比べる。

- 予測日 p ごとに、利用開始日（market2.availability_date：大引け前の開示は当日、大引け以降・休日は次の営業日）≤ p の
  記録だけを残してから計算する（as-of を日ごとに作り直すので、構造上、未来の記録は入らない）
- 決算発表予定日は、公表日（PubDate）< p のものだけ（公表日の次の営業日から使う）。同じ四半期の最新の公表が「未定」なら
  その四半期の予定日は不明とする（registry の説明文どおり。開発役のコードは「未定」の記録を捨てている）
- ユニバース：予測日の一覧で「その他」を除く。案Aは開発役の集計の時点（ce0d3b3）では未実装なので使わない
結果：data/phase3_features_indep.json（評価役の集計と開発役の集計の比較）、data/phase3_features_indep.npz（予測日 × 銘柄の値）
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from fetch import RAW  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402
from market2 import availability_date  # noqa: E402

DEV = EVAL_ROOT.parent / "project" / "reports" / "phase3_features.json"
NUM = ["Sales", "OP", "NP", "Eq", "FSales", "FOP", "FNP", "NxFSales", "NxFOP", "NxFNp"]


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    out[k:] = a[:-k]
    return out


def main() -> None:
    t0 = time.time()
    mk = bt2.load_market()
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=False)
    dates, codes = pc.dates, pc.codes
    T, N = len(dates), len(codes)
    ci = {c: j for j, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(dates)}
    # 予測日：各 ISO 週の最終営業日（＝次の週の予測日）。2017-01-01 以降、データの最後の週の前まで
    iso = pd.to_datetime(pd.Series(dates)).dt.isocalendar()
    wk = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
    preds = np.array([i for i in range(T - 1) if wk[i + 1] != wk[i] and dates[i] >= "2017-01-01"])
    U = pc.universe[preds]
    print("preds", len(preds), dates[preds[0]], dates[preds[-1]], "universe median", np.median(U.sum(1)), flush=True)

    # ---------------- 値動き
    C, H = mk["C"], mk["H"]
    cumF = pc.cumF
    q = pc.cadj_ff
    feats: dict[str, np.ndarray] = {}
    for k in (5, 20, 60):
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"ret_{k}d"] = (q / lag(q, k) - 1)[preds]
    vo = np.full((T, N), np.nan)
    for d in dates:
        b = pd.read_parquet(RAW / "bars" / f"{d}.parquet", columns=["Code", "Vo"])
        b = b[b.Code.isin(ci)]
        vo[di[d], b.Code.map(ci).to_numpy()] = pd.to_numeric(b.Vo, errors="coerce").fillna(0.0).to_numpy()
    vq = pd.DataFrame(vo * cumF)
    feats["vol_ratio_5_60"] = np.log((vq.rolling(5, min_periods=5).mean() + 1)
                                     / (vq.rolling(60, min_periods=60).mean() + 1)).to_numpy()[preds]
    lr = pd.DataFrame(np.log(q)).diff()
    feats["volatility_20d"] = lr.rolling(20, min_periods=20).std().to_numpy()[preds]
    hi = pd.DataFrame(H / cumF).rolling(60, min_periods=1).max().to_numpy()
    ok60 = ~np.isnan(lag(q, 59))
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["dist_high_60d"] = np.where(ok60, q / hi - 1, np.nan)[preds]
    va = pd.DataFrame(np.where(pc.listed, np.nan_to_num(mk["Va"], nan=0.0), np.nan))
    feats["log_turnover_20d"] = np.log1p(va.rolling(20, min_periods=20).mean().to_numpy())[preds]
    print("price done", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 財務
    mcap = (mk["shares"] * pd.DataFrame(C).ffill().to_numpy())[preds]
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["log_mcap"] = np.log(mcap)
    s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                        columns=["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "CurPerType", "CurPerEn",
                                 "CurFYEn", "NxtFYEn", *NUM])
    s = s[s.Code.isin(ci)].copy()
    for c in NUM:
        s[c] = pd.to_numeric(s[c].replace("", np.nan), errors="coerce")
    s["avail"] = availability_date(s.DiscDate, s.DiscTime, list(dates))
    s = s[s.avail.isin(di)].copy()
    s["av"] = s.avail.map(di)
    s = s.sort_values(["av", "DiscDate", "DiscTime", "DiscNo"], kind="stable").reset_index(drop=True)
    s["order"] = np.arange(len(s))
    s["j"] = s.Code.map(ci)
    is_st = s.DocType.str.contains("FinancialStatements", na=False)
    st = s[is_st].copy()
    pe = pd.to_datetime(st.CurPerEn, errors="coerce")
    st["ym"] = pe.dt.year * 12 + pe.dt.month
    # 前年同期：同じ銘柄・同じ期間の種類・期末の月が12か月前で、自分より前に開示された最後の短信
    sp, op = np.full(len(st), np.nan), np.full(len(st), np.nan)
    for _, g in st.groupby(["j", "CurPerType"]):
        ym, order = g.ym.to_numpy(), g.order.to_numpy()
        S, O = g.Sales.to_numpy(), g.OP.to_numpy()
        pos = st.index.get_indexer(g.index)
        for i in range(len(g)):
            m = np.flatnonzero((ym == ym[i] - 12) & (order < order[i]))
            if len(m):
                sp[pos[i]], op[pos[i]] = S[m[-1]], O[m[-1]]
    st["Sp"], st["Op"] = sp, op
    with np.errstate(invalid="ignore", divide="ignore"):
        st["sg"] = np.where(st.Sp > 0, st.Sales / st.Sp - 1, np.nan)
        st["oc"] = np.where(st.Sp > 0, (st.OP - st.Op) / st.Sp, np.nan)
    print("yoy done", f"{time.time() - t0:.0f}s", flush=True)
    # 会社予想（通期）の記録
    is_rev = s.DocType.eq("EarnForecastRevision")
    is_fy = is_st & s.CurPerType.eq("FY")
    fc = pd.DataFrame({"order": s.order, "av": s.av, "j": s.j,
                       "fy": np.where(is_fy, s.NxtFYEn.astype(str), s.CurFYEn.astype(str)),
                       "fs": np.where(is_fy, s.NxFSales, s.FSales), "fo": np.where(is_fy, s.NxFOP, s.FOP),
                       "fn": np.where(is_fy, s.NxFNp, s.FNP)})
    keep = (is_st | (is_rev & ~fc[["fs", "fo", "fn"]].isna().all(axis=1))) & (fc.fy.str.len() == 10)
    fc = fc[keep].copy()
    fc["fy_n"] = fc.fy.str.replace("-", "").astype(np.int64)
    fc["run_max"] = fc.groupby("j")["fy_n"].cummax()
    fc = fc[fc.fy_n >= fc.run_max].copy()
    fc["fo_prev"] = fc.groupby(["j", "fy"])["fo"].shift(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        fc["rev"] = np.where(fc.fs > 0, (fc.fo - fc.fo_prev) / fc.fs, np.nan)
    # 本決算の上振れ：同じ期の、本決算より前の最後の予想
    fy = st[st.CurPerType.eq("FY")].copy()
    fcg = {k: g for k, g in fc.groupby(["j", "fy"])}
    surp = []
    for r in fy.itertuples():
        g = fcg.get((r.j, str(r.CurFYEn)))
        if g is None:
            surp.append(np.nan)
            continue
        g = g[g.order < r.order]
        surp.append((r.OP - g.fo.iloc[-1]) / abs(r.Sales) if len(g) and abs(r.Sales) > 0 else np.nan)
    fy["sp"] = surp
    st["avf"] = st.av.astype(float)

    def asof_at(df: pd.DataFrame, col: str, p: int) -> np.ndarray:
        """利用開始日 ≤ p の記録のうち、銘柄ごとに最後のもの（av・開示日・時刻・番号の順）。"""
        x = df[df.av <= p].groupby("j").tail(1)
        out = np.full(N, np.nan)
        out[x.j.to_numpy()] = x[col].to_numpy(dtype=float)
        return out

    for n in ["bp", "ep_fcst", "sales_growth_yoy", "op_chg_to_sales_yoy", "bdays_since_report", "op_surprise_fy",
              "op_fcst_rev"]:
        feats[n] = np.full((len(preds), N), np.nan)
    for k, p in enumerate(preds):
        with np.errstate(invalid="ignore", divide="ignore"):
            feats["bp"][k] = asof_at(st, "Eq", p) / mcap[k]
            feats["ep_fcst"][k] = asof_at(fc, "fn", p) / mcap[k]
        feats["sales_growth_yoy"][k] = asof_at(st, "sg", p)
        feats["op_chg_to_sales_yoy"][k] = asof_at(st, "oc", p)
        feats["bdays_since_report"][k] = p - asof_at(st, "avf", p)
        feats["op_surprise_fy"][k] = asof_at(fy, "sp", p)
        feats["op_fcst_rev"][k] = asof_at(fc, "rev", p)
    print("fund done", f"{time.time() - t0:.0f}s", flush=True)
    # 次の決算発表予定（公表日 < 予測日。四半期ごとに最新の公表。「未定」ならその四半期は不明）
    e = pd.read_parquet(EVAL_ROOT / "data/processed/earnings_date_all.parquet")
    e = e[e.Code.isin(ci)].copy()
    e["j"] = e.Code.map(ci)
    e = e.sort_values("PubDate", kind="stable")
    nx = np.full((len(preds), N), np.nan)
    nx_drop = np.full((len(preds), N), np.nan)   # 開発役の扱い（「未定」の記録を捨てる）での値
    e_known = e[e.SchDate.astype(str).str.len() == 10]
    for k, p in enumerate(preds):
        d = str(dates[p])
        for src, dst in ((e, nx), (e_known, nx_drop)):
            x = src[src.PubDate < d].groupby(["j", "FYE", "FQName"]).tail(1)
            x = x[(x.SchDate.astype(str).str.len() == 10) & (x.SchDate > d)]
            m = x.groupby("j").SchDate.min()
            dst[k, m.index.to_numpy()] = (pd.to_datetime(m.to_numpy()) - pd.Timestamp(d)).days
    feats["cdays_to_next_earnings"] = nx
    print("sched done", f"{time.time() - t0:.0f}s", flush=True)

    dev = json.load(open(DEV, encoding="utf-8"))
    res = {"preds": int(len(preds)), "period": [str(dates[preds[0]]), str(dates[preds[-1]])],
           "universe_median": float(np.median(U.sum(1))), "dev_meta": dev["meta"], "features": {}}
    for n, a in list(feats.items()) + [("cdays_to_next_earnings（未定を捨てる）", nx_drop)]:
        cov = np.array([np.isfinite(a[k][U[k]]).mean() for k in range(len(preds))])
        v = a[U]
        v = v[np.isfinite(v)]
        qs = {str(qq): float(np.quantile(v, qq)) for qq in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)}
        dv = dev["features"][n.split("（")[0]]
        res["features"][n] = {"eval_cov": float(cov.mean()), "dev_cov": dv["coverage_mean"],
                              "eval_cov_min": float(cov.min()), "dev_cov_min": dv["coverage_min"],
                              "eval_cov_min_date": str(dates[preds[cov.argmin()]]), "dev_cov_min_date": dv["coverage_min_date"],
                              "eval_q": qs, "dev_q": dv["quantiles"]}
    (EVAL_ROOT / "data/phase3_features_indep.json").write_text(json.dumps(res, ensure_ascii=False, indent=1),
                                                              encoding="utf-8")
    np.savez_compressed(EVAL_ROOT / "data/phase3_features_indep.npz", preds=preds, universe=U, **feats)
    for n, r in res["features"].items():
        print(f"{n:28s} cov {r['eval_cov']:.4f}/{r['dev_cov']:.4f}  min {r['eval_cov_min']:.3f}/{r['dev_cov_min']:.3f}  "
              + "  ".join(f"q{qq} {r['eval_q'][qq]:.4g}/{r['dev_q'][qq]:.4g}" for qq in ("0.01", "0.05", "0.5", "0.95", "0.99")))
    print("elapsed", time.time() - t0)


if __name__ == "__main__":
    main()
