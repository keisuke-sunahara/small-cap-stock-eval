"""フェーズ1の照合：評価役が自分で取得したデータから、開発役の報告の数字を独立に計算する。

開発役のコードは使わない。結果は data/phase1_check.json に保存し、会話には要点だけを出す。
使い方: python src/phase1_check.py
"""
from __future__ import annotations

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch import END, RAW, START  # noqa: E402
from jq import EVAL_ROOT, HOLDOUT_START  # noqa: E402

OUT = EVAL_ROOT / "data" / "phase1_check.json"
COMMON_MKT = {"東証一部", "東証二部", "マザーズ", "JASDAQ スタンダード", "JASDAQ グロース",
              "プライム", "スタンダード", "グロース"}


def read_daily(ds: str, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(p for p in (RAW / ds).glob("*.parquet"))
    dfs, empty = [], []
    for p in files:
        d = p.stem
        assert d < HOLDOUT_START.isoformat(), p   # 念のため：ホールドアウト期間のファイルは存在しないはず
        if pq.read_metadata(p).num_rows == 0:
            empty.append(d)
            continue
        df = pd.read_parquet(p, columns=columns) if columns else pd.read_parquet(p)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True), empty


def main() -> None:
    res: dict = {"period": [START.isoformat(), END.isoformat()], "holdout_start": HOLDOUT_START.isoformat()}

    # ---------- カレンダー ----------
    cal = pd.read_parquet(RAW / "calendar" / f"{START}_{END}.parquet")
    cal["HolDiv"] = cal["HolDiv"].astype(str)
    res["calendar"] = {
        "rows": len(cal), "min": cal["Date"].min(), "max": cal["Date"].max(),
        "holdiv_counts": cal["HolDiv"].value_counts().to_dict(),
    }
    bdays = sorted(cal.loc[cal["HolDiv"].isin(["1", "2"]), "Date"])
    res["calendar"]["business_days"] = len(bdays)
    res["calendar"]["business_days_per_year"] = pd.Series(bdays).str[:4].value_counts().sort_index().to_dict()
    cdays = [(START + timedelta(days=i)).isoformat() for i in range((END - START).days + 1)]

    # ---------- 取得の欠け ----------
    cov = {}
    for ds, days in [("master", bdays), ("bars", bdays), ("summary", cdays), ("earnings_date", cdays)]:
        have = {p.stem for p in (RAW / ds).glob("*.parquet")}
        miss = [d for d in days if d not in have]
        extra = sorted(have - set(days))
        cov[ds] = {"target_days": len(days), "missing_days": len(miss), "missing_sample": miss[:5],
                   "files_outside_target": extra[:5]}
    res["coverage"] = cov

    # ---------- TOPIX ----------
    tp = pd.read_parquet(RAW / "topix" / f"{START}_{END}.parquet")
    res["topix"] = {"rows": len(tp), "missing_bdays": sorted(set(bdays) - set(tp["Date"]))[:10],
                    "null_close": int(tp["C"].isna().sum()), "max_date": tp["Date"].max(),
                    "abs_daily_return_max": round(float(tp.sort_values("Date")["C"].pct_change().abs().max()), 4)}

    # ---------- 上場銘柄一覧 ----------
    m, m_empty = read_daily("master", ["Date", "Code", "CoName", "Mkt", "MktNm", "ProdCat", "Mrgn"])
    m["common"] = (m["ProdCat"] == "011") & m["MktNm"].isin(COMMON_MKT) & m["Code"].str.endswith("0")
    per_day = m.groupby("Date").agg(rows=("Code", "size"), common=("common", "sum"))
    last = m["Date"].max()
    snap = m[m["Date"] == last]
    res["master"] = {
        "empty_days": m_empty,
        "rows_per_day": {"min": int(per_day["rows"].min()), "max": int(per_day["rows"].max())},
        "common_per_day": {"min": int(per_day["common"].min()), "max": int(per_day["common"].max()),
                           "first": int(per_day["common"].iloc[0]), "last": int(per_day["common"].iloc[-1])},
        "last_date": last,
        "snapshot_last": {"rows": len(snap), "common": int(snap["common"].sum()),
                          "ProdCat": snap["ProdCat"].value_counts().to_dict(),
                          "Mkt": snap["MktNm"].value_counts().to_dict()},
        "market_names_all": sorted(m["MktNm"].unique().tolist()),
        "prodcat011_code_not_ending_0_last": snap.loc[(snap["ProdCat"] == "011") & ~snap["Code"].str.endswith("0"),
                                                      "CoName"].tolist(),
        "prodcat011_other_market_last": snap.loc[(snap["ProdCat"] == "011") & ~snap["MktNm"].isin(COMMON_MKT),
                                                 "MktNm"].value_counts().to_dict(),
        "common_drop_over_10pct_days": per_day.index[per_day["common"].pct_change() < -0.10].tolist(),
    }
    first = m[m["Date"] == m["Date"].min()]
    res["master"]["snapshot_first"] = {"date": first["Date"].iloc[0], "rows": len(first),
                                       "common": int(first["common"].sum())}

    # ---------- 株価四本値 ----------
    b, b_empty = read_daily("bars", ["Date", "Code", "O", "H", "L", "C", "UL", "LL", "Vo", "Va", "AdjFactor"])
    b = b.merge(m[["Date", "Code", "common"]], on=["Date", "Code"], how="left", indicator=True)
    px = ["O", "H", "L", "C"]
    no_trade = b[px].isna().all(axis=1)
    any_null = b[px + ["Vo", "Va"]].isna().any(axis=1)
    all_null = b[px + ["Vo", "Va"]].isna().all(axis=1)
    t = b[~no_trade]
    rpd = b.groupby("Date").size()
    nt_day = no_trade.groupby(b["Date"]).mean()
    res["bars"] = {
        "empty_days": b_empty,
        "total_rows": len(b),
        "rows_per_day": {"min": int(rpd.min()), "max": int(rpd.max()), "median": float(rpd.median())},
        "no_trade_rows": int(no_trade.sum()),
        "no_trade_ratio": round(float(no_trade.mean()), 5),
        "no_trade_ratio_common": round(float(no_trade[b["common"] == True].mean()), 5),  # noqa: E712
        "no_trade_ratio_common_2025": round(float(no_trade[(b["common"] == True) & (b["Date"] >= "2025")].mean()), 5),  # noqa: E712
        "days_no_trade_over_50pct": {k: round(v, 3) for k, v in nt_day[nt_day > 0.5].items()},
        "partial_null_rows": int((any_null & ~all_null).sum()),
        "invalid": {
            "nonpositive_price": int((t[px] <= 0).any(axis=1).sum()),
            "low_gt_high": int((t["L"] > t["H"]).sum()),
            "open_or_close_outside_range": int(((t["O"] > t["H"]) | (t["O"] < t["L"]) |
                                                (t["C"] > t["H"]) | (t["C"] < t["L"])).sum()),
            "zero_volume_with_price": int((t["Vo"] <= 0).sum()),
        },
        "vwap_outside_range_rows": int((((t["Va"] / t["Vo"]) > t["H"] * 1.0001) |
                                        ((t["Va"] / t["Vo"]) < t["L"] * 0.9999)).sum()),
        "limit_flags": {"UL_rows": int((b["UL"].astype(str) == "1").sum()),
                        "LL_rows": int((b["LL"].astype(str) == "1").sum())},
        "rows_not_in_master": int((b["_merge"] == "left_only").sum()),
    }
    # 一覧にあって株価がない（普通株）
    mm = m[m["common"]].merge(b[["Date", "Code"]], on=["Date", "Code"], how="left", indicator=True)
    res["bars"]["common_in_master_without_bars_rows"] = int((mm["_merge"] == "left_only").sum())
    res["bars"]["common_bars_without_master_rows"] = 0  # 普通株の判定は一覧に依存するため、下の rows_not_in_master で代替
    res["bars"]["rows_not_in_master_sample"] = b.loc[b["_merge"] == "left_only", ["Date", "Code"]].head(5).values.tolist()

    # ---------- 分割・併合（調整係数の累積積） ----------
    b = b.sort_values(["Code", "Date"]).reset_index(drop=True)
    b["AdjFactor"] = b["AdjFactor"].astype(float)
    ev = b[b["AdjFactor"] != 1.0]
    res["bars"]["adjfactor_null"] = int(b["AdjFactor"].isna().sum())
    res["bars"]["adjfactor_events"] = {
        "count": len(ev), "codes": int(ev["Code"].nunique()),
        "values_top": ev["AdjFactor"].round(4).astype(str).value_counts().head(10).to_dict(),
    }
    # 累積積（その日までの情報だけ）で調整した終値。売買不成立の日の係数も累積に含める
    b["cumF"] = b.groupby("Code")["AdjFactor"].cumprod()
    b["adjC"] = b["C"] / b["cumF"]
    # 直前の「売買が成立した日」の調整後終値との比
    g = b["Code"]
    for col in ["adjC", "C", "cumF"]:
        tr = b[col].where(b["C"].notna())
        b["prev_" + col] = tr.groupby(g).ffill().groupby(g).shift(1)
    # 直前の成立日から今日までの係数の積（今日を含む）
    b["adj_ret"] = b["adjC"] / b["prev_adjC"] - 1
    b["raw_ret"] = b["C"] / b["prev_C"] - 1
    r = b.dropna(subset=["adj_ret"])
    # 分割の日：その日か、直前の成立日以降に係数≠1がある日
    b["F_since_prev"] = b["cumF"] / b["prev_cumF"]

    split_days = b[(b["F_since_prev"].round(6) != 1.0) & b["adj_ret"].notna()]
    res["bars"]["split_days"] = {
        "n": len(split_days),
        "abs_raw_return_median": round(float(split_days["raw_ret"].abs().median()), 4),
        "abs_adj_return_median": round(float(split_days["adj_ret"].abs().median()), 4),
        "adj_return_over_50pct": int((split_days["adj_ret"].abs() > 0.5).sum()),
    }
    big = r[r["adj_ret"].abs() > 0.5]
    res["bars"]["adjusted_daily_returns"] = {
        "n": len(r),
        "abs_over_50pct": len(big),
        "abs_over_50pct_common": int((big["common"] == True).sum()),  # noqa: E712
        "abs_over_50pct_up": int((big["adj_ret"] > 0).sum()),
        "abs_over_50pct_with_limit_flag": int(((big["UL"].astype(str) == "1") | (big["LL"].astype(str) == "1")).sum()),
        "quantiles": {str(q): round(float(r["adj_ret"].quantile(q)), 4) for q in [0.001, 0.01, 0.5, 0.99, 0.999]},
        "extreme_by_year": big["Date"].str[:4].value_counts().sort_index().to_dict(),
    }
    big.sort_values("adj_ret").head(15)[["Date", "Code", "C", "prev_C", "adj_ret", "UL", "LL", "common"]].to_csv(
        EVAL_ROOT / "data" / "phase1_big_moves_sample.csv", index=False)
    # API の調整後株価（Adj*）は使わない。ここでも読まない（ホールドアウト期間の分割を含むため）

    # ---------- 上場廃止・新規上場 ----------
    last_bday = b["Date"].max()
    first_bday = b["Date"].min()
    span = b.groupby("Code")["Date"].agg(["min", "max"])
    ended = span[span["max"] < last_bday]
    started = span[span["min"] > first_bday]
    traded = b[b["C"].notna()].groupby("Code")["Date"].max()
    ever_common = set(b.loc[b["common"] == True, "Code"])  # noqa: E712
    res["delisted"] = {
        "codes_total": len(span),
        "codes_ended_before_last_day": len(ended),
        "codes_ended_common": int(ended.index.isin(ever_common).sum()),
        "ended_by_year": ended["max"].str[:4].value_counts().sort_index().to_dict(),
        "started_after_first_day_by_year": started["min"].str[:4].value_counts().sort_index().to_dict(),
        "codes_never_traded": int(len(span) - len(traded)),
        "ended_but_in_master_on_last_day": int(ended.index.isin(set(snap["Code"])).sum()),
        # 一覧から消えた日と株価が終わった日のずれ
    }
    mspan = m.groupby("Code")["Date"].max()
    gap = (mspan.reindex(ended.index) != ended["max"]).sum()
    res["delisted"]["ended_codes_master_last_date_differs"] = int(gap)

    # ---------- 財務情報 ----------
    s, s_empty = read_daily("summary", ["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "ShOutFY"])
    tm = s["DiscTime"].astype(str).str[:5]
    fmt_ok = tm.str.match(r"^\d{2}:\d{2}$")
    fin = s["DocType"].str.contains("FinancialStatements")
    dow = pd.to_datetime(s["DiscDate"]).dt.dayofweek
    res["summary"] = {
        "rows": len(s), "empty_days": len(s_empty),
        "rows_per_year": s["DiscDate"].str[:4].value_counts().sort_index().to_dict(),
        "DocType_kinds": int(s["DocType"].nunique()),
        "DiscTime_bad_format": int((~fmt_ok).sum()),
        "DiscTime_share": {"before_15:00": round(float((tm < "15:00").mean()), 4),
                           "15:00_to_15:30": round(float(((tm >= "15:00") & (tm < "15:30")).mean()), 4),
                           "15:30_or_later": round(float((tm >= "15:30").mean()), 4)},
        "duplicate_DiscNo": int(s["DiscNo"].duplicated().sum()),
        "dates_with_weekend_disclosure": int(s.loc[dow >= 5, "DiscDate"].nunique()),
        "ShOutFY_blank_ratio_in_FinancialStatements": round(float(
            (s.loc[fin, "ShOutFY"].isna() | (s.loc[fin, "ShOutFY"].astype(str) == "")).mean()), 4),
        "max_DiscDate": s["DiscDate"].max(),
    }
    snap_common = set(snap.loc[snap["common"], "Code"])
    recent = set(s.loc[fin & (s["DiscDate"] > (date.fromisoformat(last) - timedelta(days=365)).isoformat()), "Code"])
    res["summary"]["common_with_statement_in_last_365d"] = round(len(snap_common & recent) / len(snap_common), 4)

    # ---------- 決算発表予定日 ----------
    e, e_empty = read_daily("earnings_date", ["PubDate", "SchDate", "FQName", "Code"])
    sch = pd.to_datetime(e["SchDate"].replace("", None), errors="coerce")
    pub = pd.to_datetime(e["PubDate"])
    res["earnings_date"] = {
        "rows": len(e), "empty_days": len(e_empty),
        "rows_per_year": e["PubDate"].str[:4].value_counts().sort_index().to_dict(),
        "SchDate_blank_ratio": round(float(sch.isna().mean()), 4),
        "SchDate_before_PubDate": int((sch < pub).sum()),
        "FQName": e["FQName"].value_counts().to_dict(),
        "lead_days_median": float((sch - pub).dt.days.median()),
        "max_PubDate": e["PubDate"].max(),
        "max_SchDate_year": None,  # 予定日（未来の日付）の中身は見ない
    }
    rec_e = set(e.loc[e["PubDate"] > (date.fromisoformat(last) - timedelta(days=365)).isoformat(), "Code"])
    res["earnings_date"]["common_with_schedule_in_last_365d"] = round(len(snap_common & rec_e) / len(snap_common), 4)

    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
