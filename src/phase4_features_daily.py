"""フェーズ4前半の確認6の準備：特徴量 all_v1（16個）を、全営業日（2017-01-04〜2025-09-26）× ユニバースについて、
評価役のデータと評価役のコードで計算する（学習の行に使うため）。

計算の中身は src/phase3_features_indep.py（予測日だけ。フェーズ4計画の評価で開発役の値と全セル一致）と同じ定義で、
日ごとの as-of を「記録の番号を前に埋める」方法で全営業日に広げたもの。株数は評価役の補正（shares_fixed.npz）。
- 財務：利用開始日（market2.availability_date）≤ p の記録のうち、銘柄ごとに最後のもの
- 決算発表予定日：公表日 < p の、四半期ごとに最新の記録。「未定」ならその四半期は不明。予定日 > p の最も近いもの
- ユニバース：bt2.Precomp（「その他」の除外・案A・補正後の株数）
結果：data/phase4_features_daily.npz（t・j・16個の値。ユニバースの行だけ）
"""
from __future__ import annotations

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

NUM = ["Sales", "OP", "NP", "Eq", "FSales", "FOP", "FNP", "NxFSales", "NxFOP", "NxFNp"]
FEATS = ["ret_5d", "ret_20d", "ret_60d", "vol_ratio_5_60", "volatility_20d", "dist_high_60d", "log_turnover_20d",
         "log_mcap", "ep_fcst", "bp", "sales_growth_yoy", "op_chg_to_sales_yoy", "bdays_since_report", "op_surprise_fy",
         "op_fcst_rev", "cdays_to_next_earnings"]
START = "2017-01-01"


def lag(a: np.ndarray, k: int) -> np.ndarray:
    out = np.full(a.shape, np.nan)
    out[k:] = a[:-k]
    return out


def asof_matrix(df: pd.DataFrame, T: int, N: int) -> np.ndarray:
    """df（av・j、行は order の順＝av の順）の各日 p・銘柄 j で、av ≤ p の最後の記録の行番号（無ければ −1）。"""
    df = df.reset_index(drop=True)
    last = df.groupby(["av", "j"]).tail(1)
    M = np.full((T, N), -1, dtype=np.int64)
    M[last.av.to_numpy(), last.j.to_numpy()] = last.index.to_numpy()
    return np.maximum.accumulate(M, axis=0)


def gather(df: pd.DataFrame, col: str, M: np.ndarray) -> np.ndarray:
    v = np.append(df[col].to_numpy(dtype=float), np.nan)   # −1 → NaN
    return v[np.where(M >= 0, M, len(v) - 1)]


def main() -> None:
    t0 = time.time()
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    dates, codes = pc.dates, pc.codes
    T, N = len(dates), len(codes)
    ci = {c: j for j, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(dates)}
    feats: dict[str, np.ndarray] = {}
    C, H = mk["C"], mk["H"]
    cumF = pc.cumF
    q = pc.cadj_ff
    for k in (5, 20, 60):
        with np.errstate(invalid="ignore", divide="ignore"):
            feats[f"ret_{k}d"] = q / lag(q, k) - 1
    vo = np.full((T, N), np.nan)
    for d in dates:
        b = pd.read_parquet(RAW / "bars" / f"{d}.parquet", columns=["Code", "Vo"])
        b = b[b.Code.isin(ci)]
        vo[di[d], b.Code.map(ci).to_numpy()] = pd.to_numeric(b.Vo, errors="coerce").fillna(0.0).to_numpy()
    vq = pd.DataFrame(vo * cumF)
    feats["vol_ratio_5_60"] = np.log((vq.rolling(5, min_periods=5).mean() + 1)
                                     / (vq.rolling(60, min_periods=60).mean() + 1)).to_numpy()
    del vo, vq
    lr = pd.DataFrame(np.log(q)).diff()
    feats["volatility_20d"] = lr.rolling(20, min_periods=20).std().to_numpy()
    hi = pd.DataFrame(H / cumF).rolling(60, min_periods=1).max().to_numpy()
    ok60 = ~np.isnan(lag(q, 59))
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["dist_high_60d"] = np.where(ok60, q / hi - 1, np.nan)
    va = pd.DataFrame(np.where(pc.listed, np.nan_to_num(mk["Va"], nan=0.0), np.nan))
    feats["log_turnover_20d"] = np.log1p(va.rolling(20, min_periods=20).mean().to_numpy())
    print("price done", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 財務
    mcap = mk["shares"] * pd.DataFrame(C).ffill().to_numpy()
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
    st, fc, fy = st.reset_index(drop=True), fc.reset_index(drop=True), fy.reset_index(drop=True)
    Mst, Mfc, Mfy = asof_matrix(st, T, N), asof_matrix(fc, T, N), asof_matrix(fy, T, N)
    with np.errstate(invalid="ignore", divide="ignore"):
        feats["bp"] = gather(st, "Eq", Mst) / mcap
        feats["ep_fcst"] = gather(fc, "fn", Mfc) / mcap
    feats["sales_growth_yoy"] = gather(st, "sg", Mst)
    feats["op_chg_to_sales_yoy"] = gather(st, "oc", Mst)
    feats["bdays_since_report"] = np.arange(T)[:, None] - gather(st, "avf", Mst)
    feats["op_surprise_fy"] = gather(fy, "sp", Mfy)
    feats["op_fcst_rev"] = gather(fc, "rev", Mfc)
    del Mst, Mfc, Mfy
    print("fund done", f"{time.time() - t0:.0f}s", flush=True)

    # ---------------- 次の決算発表予定（公表日 < p。四半期ごとに最新の公表。「未定」ならその四半期は不明）
    e = pd.read_parquet(EVAL_ROOT / "data/processed/earnings_date_all.parquet")
    e = e[e.Code.isin(ci)].copy()
    e["j"] = e.Code.map(ci)
    e = e.sort_values("PubDate", kind="stable").reset_index(drop=True)
    e["next_pub"] = e.groupby(["j", "FYE", "FQName"])["PubDate"].shift(-1)
    # 同じ四半期で同じ公表日の記録が複数あれば、最後の記録だけが有効（tail(1) と同じ）
    e = e[e.next_pub.isna() | (e.next_pub != e.PubDate)].copy()
    valid = e.SchDate.astype(str).str.len() == 10
    e = e[valid].copy()
    start = np.searchsorted(dates, e.PubDate.astype(str).to_numpy(), side="right")        # p > 公表日
    end_pub = np.where(e.next_pub.isna(), T - 1,
                       np.searchsorted(dates, e.next_pub.fillna("9999").astype(str).to_numpy(), side="right") - 1)  # p ≤ 次の公表日
    end_sch = np.searchsorted(dates, e.SchDate.astype(str).to_numpy(), side="left") - 1    # p < 予定日
    end = np.minimum(end_pub, end_sch)
    sch_ord = pd.to_datetime(e.SchDate).map(pd.Timestamp.toordinal).to_numpy()
    best = np.full((T, N), np.iinfo(np.int64).max, dtype=np.int64)
    jj = e.j.to_numpy()
    for a, b, j, o in zip(start, end, jj, sch_ord):
        if a <= b:
            seg = best[a:b + 1, j]
            np.minimum(seg, o, out=seg)
    d_ord = np.array([pd.Timestamp(str(d)).toordinal() for d in dates])
    nx = np.where(best == np.iinfo(np.int64).max, np.nan, (best - d_ord[:, None]).astype(float))
    feats["cdays_to_next_earnings"] = nx
    print("sched done", f"{time.time() - t0:.0f}s", flush=True)

    t_lo = int(np.searchsorted(dates, START))
    U = pc.universe.copy()
    U[:t_lo] = False
    tt, jj = np.nonzero(U)
    out = {"t": tt.astype(np.int32), "j": jj.astype(np.int32)}
    for n in FEATS:
        out[n] = feats[n][tt, jj].astype(np.float64)
    np.savez_compressed(EVAL_ROOT / "data/phase4_features_daily.npz", **out)
    print("saved rows", len(tt), f"{time.time() - t0:.0f}s")
    # 自己点検：予測日の値を、フェーズ3の独立計算（予測日だけ、全セルが開発役と一致済み）と比べる
    z = np.load(EVAL_ROOT / "data/phase3_features_indep.npz")
    preds = z["preds"]
    shares_old = bt2.load_market()["shares"]
    Cff = pd.DataFrame(C).ffill().to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = (shares_old * Cff)[preds] / mcap[preds]
    Up = U[preds]
    for n in FEATS:
        a = z[n].copy()
        if n in ("bp", "ep_fcst"):
            a = a * ratio
        if n == "log_mcap":
            a = np.log(mcap[preds])
        b = feats[n][preds]
        x, y = a[Up], b[Up]
        nm = np.isnan(x) != np.isnan(y)
        vm = ~np.isnan(x) & ~np.isnan(y) & ~np.isclose(x, y, rtol=1e-9, atol=1e-12)
        print(f"{n:26s} nan_mismatch {nm.sum():6d} value_mismatch {vm.sum():6d}")


if __name__ == "__main__":
    main()
