"""フェーズ2の独立検証用：評価役のデータから「日付×銘柄」の行列を作る（開発役のコードは使わない）。

- 価格は調整前（実際の値）。分割・併合は AdjFactor（その日の朝に効く係数）で別に扱う
- Adj* 列・MktCap 列は読まない
- 発行済株式数は決算短信の ShOutFY を「開示の利用開始日」から使い、期末日より後の分割・併合を AdjFactor で直す
- 結果は data/processed/market2.npz（約1分で作り直せる）

使い方: python src/market2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import END, RAW, START  # noqa: E402
from jq import EVAL_ROOT, HOLDOUT_START  # noqa: E402

OUT = EVAL_ROOT / "data" / "processed" / "market2.npz"
COMMON_MKT = {"東証一部", "東証二部", "マザーズ", "JASDAQ スタンダード", "JASDAQ グロース",
              "プライム", "スタンダード", "グロース"}
CLOSE_CHANGE = "2024-11-05"


def business_days() -> list[str]:
    cal = pd.read_parquet(RAW / "calendar" / f"{START}_{END}.parquet")
    cal["HolDiv"] = cal["HolDiv"].astype(str)
    return sorted(cal.loc[cal["HolDiv"].isin(["1", "2"]), "Date"])


def availability_date(disc_date: pd.Series, disc_time: pd.Series, bdays: list[str]) -> pd.Series:
    """開示を予測に使える最初の営業日（その日の引け後の予測から使える）。
    大引け（2024-11-05以降15:30、それ以前15:00）より前の開示なら当日、以降・非営業日・時刻なしなら翌営業日。"""
    bd = np.array(bdays)
    d = disc_date.astype(str).to_numpy()
    t = disc_time.fillna("99:99:99").astype(str).str[:5].to_numpy()
    close = np.where(d >= CLOSE_CHANGE, "15:30", "15:00")
    is_bd = np.isin(d, bd)
    before_close = is_bd & (t < close)
    # 当日（営業日かつ引け前）→ d、それ以外 → d より後の最初の営業日
    idx_same = np.searchsorted(bd, d, side="left")
    idx_next = np.searchsorted(bd, d, side="right")
    idx = np.where(before_close, idx_same, idx_next)
    out = np.where(idx < len(bd), bd[np.minimum(idx, len(bd) - 1)], "9999-99-99")
    return pd.Series(out, index=disc_date.index)


def main() -> None:
    bdays = business_days()
    assert bdays[-1] < HOLDOUT_START.isoformat()
    di = {d: i for i, d in enumerate(bdays)}

    bars, master = [], []
    for d in bdays:
        b = pd.read_parquet(RAW / "bars" / f"{d}.parquet",
                            columns=["Date", "Code", "O", "H", "L", "C", "UL", "LL", "Va", "AdjFactor"])
        bars.append(b)
        m = pd.read_parquet(RAW / "master" / f"{d}.parquet", columns=["Date", "Code", "MktNm", "ProdCat", "Mrgn"])
        master.append(m)
    bars = pd.concat(bars, ignore_index=True)
    master = pd.concat(master, ignore_index=True)
    master["common"] = (master["ProdCat"] == "011") & master["MktNm"].isin(COMMON_MKT) & master["Code"].str.endswith("0")
    codes = np.array(sorted(master.loc[master["common"], "Code"].unique()))
    ci = {c: i for i, c in enumerate(codes)}
    T, N = len(bdays), len(codes)
    print(f"days={T} codes={N}")

    def mat(df: pd.DataFrame, col: str, dtype=float, fill=np.nan) -> np.ndarray:
        a = np.full((T, N), fill, dtype=dtype)
        sub = df[df["Code"].isin(ci)]
        r = sub["Date"].map(di).to_numpy()
        c = sub["Code"].map(ci).to_numpy()
        v = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        a[r, c] = v.astype(dtype) if dtype is not float else v
        return a

    O, H, L, C = (mat(bars, k) for k in ("O", "H", "L", "C"))
    Va = mat(bars, "Va")
    UL = mat(bars, "UL", float, 0.0)
    LL = mat(bars, "LL", float, 0.0)
    F = mat(bars, "AdjFactor", float, 1.0)
    F = np.where(np.isnan(F), 1.0, F)
    m_c = master[master["common"]].copy()
    m_c["one"] = 1.0
    listed = mat(m_c, "one", float, 0.0) > 0
    m_c["mo"] = (m_c["Mrgn"].astype(str) == "3").astype(float)
    margin_other = mat(m_c, "mo", float, 0.0) > 0

    # ---------- 発行済株式数（その時点で開示済み） ----------
    fins = []
    for p in sorted((RAW / "summary").glob("*.parquet")):
        if p.stem >= HOLDOUT_START.isoformat():
            raise RuntimeError(p)
        if pq.read_metadata(p).num_rows == 0:
            continue
        df = pd.read_parquet(p, columns=["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "CurPerEn", "ShOutFY"])
        if len(df):
            fins.append(df)
    fins = pd.concat(fins, ignore_index=True)
    fins = fins[fins["DocType"].str.contains("FinancialStatements", na=False)]
    fins["sh"] = pd.to_numeric(fins["ShOutFY"], errors="coerce")
    fins = fins[(fins["sh"] > 0) & fins["Code"].isin(ci)].copy()
    fins["avail"] = availability_date(fins["DiscDate"], fins["DiscTime"], bdays)
    fins = fins[fins["avail"].isin(di)]
    fins = fins.sort_values(["Code", "avail", "DiscDate", "DiscTime", "DiscNo"])
    # 期末日以前の最後の営業日のインデックス（期末日より後の係数で株数を直す）
    bd_arr = np.array(bdays)
    fins["pe_idx"] = np.searchsorted(bd_arr, fins["CurPerEn"].astype(str).to_numpy(), side="right") - 1
    fins["av_idx"] = fins["avail"].map(di)
    # cumF[t] = 係数の累積積（0日目から t 日目まで）。期末日 pe より後〜t の係数の積 = cumF[t]/cumF[pe]
    cumF = np.cumprod(F, axis=0)
    shares = np.full((T, N), np.nan)
    for code, g in fins.groupby("Code", sort=False):
        j = ci[code]
        # 同じ利用開始日に複数あれば最後（開示番号の大きい方）
        g = g.drop_duplicates("av_idx", keep="last")
        av = g["av_idx"].to_numpy()
        sh = g["sh"].to_numpy()
        pe = np.maximum(g["pe_idx"].to_numpy(), 0)
        for k in range(len(av)):
            t0 = av[k]
            t1 = av[k + 1] if k + 1 < len(av) else T
            base = cumF[pe[k], j]
            shares[t0:t1, j] = sh[k] * base / cumF[t0:t1, j]   # 係数0.5（1→2分割）なら株数は2倍
    np.savez_compressed(OUT, dates=np.array(bdays), codes=codes, O=O, H=H, L=L, C=C, Va=Va, UL=UL, LL=LL,
                        F=F, listed=listed, margin_other=margin_other, shares=shares)
    print("saved", OUT)


if __name__ == "__main__":
    OUT.parent.mkdir(parents=True, exist_ok=True)
    main()
