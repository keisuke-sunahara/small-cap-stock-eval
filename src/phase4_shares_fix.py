"""フェーズ3の指摘（低1）への対応の確認：発行済株式数の基準のずれの補正。

1. project/CLAUDE.md 第5章「株式数の基準のずれの補正（2026-10-05 承認）」の文章だけから、評価役が補正を独立に実装する
   （開発役の src/backtest/market.py の share_basis は使わない）
2. 評価役の補正の結果と、開発役の一覧（project/reports/phase3_shares_bad_reports.csv の fix 列）を短信ごとに照合する
3. 評価役のフェーズ3の一覧（data/phase3_shares_bad_reports.csv、179件）と開発役の一覧を照合し、
   評価役の一覧の各短信が補正されたか・されなかった理由を分類する
4. 補正の正しさの監査（監査用に「次の短信」も使う。売買には使わない）：補正の前後で、前後の短信との一致がどう変わるか
5. 補正後の株数の行列を data/processed/shares_fixed.npz に保存する（特徴量・ベースラインの照合に使う）

結果：data/phase4_shares_fix.json・phase4_shares_fix_reports.csv・phase4_shares_vs_eval.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402
from market2 import availability_date  # noqa: E402

DEV_CSV = EVAL_ROOT.parent / "project" / "reports" / "phase3_shares_bad_reports.csv"
EVAL_CSV = EVAL_ROOT / "data" / "phase3_shares_bad_reports.csv"
TOL = 0.05          # config/base.yaml の shares_split_fix_tol（文面の「5%」）
BEFORE_DAYS = 5     # 文面の「期末日の5営業日前」


def build_records(mk: dict) -> tuple[pd.DataFrame, np.ndarray]:
    dates, codes, F = mk["dates"], mk["codes"], mk["F"]
    T = len(dates)
    ci = {c: j for j, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(dates)}
    cumF = np.cumprod(F, axis=0)
    s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                        columns=["DiscDate", "DiscTime", "Code", "DiscNo", "DocType", "CurPerEn", "ShOutFY"])
    s = s[s.DocType.str.contains("FinancialStatements", na=False)].copy()
    s["sh"] = pd.to_numeric(s.ShOutFY, errors="coerce")
    s = s[(s.sh > 0) & s.Code.isin(ci)].copy()
    s["avail"] = availability_date(s.DiscDate, s.DiscTime, list(dates))
    s = s[s.avail.isin(di)].copy()
    s["av"] = s.avail.map(di)
    s["pe"] = np.clip(np.searchsorted(dates, s.CurPerEn.astype(str).to_numpy(), side="right") - 1, 0, T - 1)
    s["disc"] = np.searchsorted(dates, s.DiscDate.astype(str).to_numpy(), side="right") - 1   # 開示日以前の最後の営業日
    s["j"] = s.Code.map(ci)
    s = s.sort_values(["Code", "av", "DiscDate", "DiscTime", "DiscNo"], kind="stable").reset_index(drop=True)
    return s, cumF


def apply_fix(s: pd.DataFrame, cumF: np.ndarray, tol: float = TOL, before_days: int = BEFORE_DAYS,
              end_col: str = "disc") -> pd.DataFrame:
    """CLAUDE.md の文章どおり：直前の短信（補正後）と5%超ずれたら、(a) 期末日の翌営業日〜開示日、(b) 期末日の5営業日前〜前の営業日
    の累積積で計算し直した値のうち直前に最も近いものが5%以内ならそれを使う。直前の短信をその比率で戻すと1つ前と5%以内で合うなら補正しない。
    「直前の短信」は、使える順（利用開始日・開示日・時刻・番号）で1つ前の短信（同じ日の訂正も1件と数える）。
    「5%」は対数の差（|log(比)| ≤ log 1.05）で判定する。"""
    s = s.copy()
    base_raw = s.sh.to_numpy() * cumF[s.pe.to_numpy(), s.j.to_numpy()]
    base = base_raw.copy()
    fix = np.array([""] * len(s), dtype=object)
    lim = np.log1p(tol)
    for _, idx in s.groupby("Code", sort=False).indices.items():
        prev = prev2 = np.nan
        for i in idx:
            j, pe, dc, sh = int(s.j.iat[i]), int(s.pe.iat[i]), int(s[end_col].iat[i]), float(s.sh.iat[i])
            if np.isfinite(prev) and abs(np.log(base[i] / prev)) > lim:
                cand = list(range(max(pe - before_days, 0), pe)) + list(range(pe + 1, dc + 1))
                if cand:
                    vals = sh * cumF[cand, j]
                    dev = np.abs(np.log(vals / prev))
                    k = int(np.argmin(dev))
                    ratio = cumF[cand[k], j] / cumF[pe, j]
                    ahead = np.isfinite(prev2) and abs(np.log(prev / ratio / prev2)) <= lim
                    if dev[k] <= lim and not ahead:
                        base[i] = vals[k]
                        fix[i] = "after" if cand[k] > pe else "before"
            prev2, prev = prev, base[i]
    s["base_raw"], s["base"], s["fix"] = base_raw, base, fix
    return s


def shares_matrix(s: pd.DataFrame, cumF: np.ndarray, col: str) -> np.ndarray:
    T, N = cumF.shape
    out = np.full((T, N), np.nan)
    for _, g in s.groupby("Code", sort=False):
        g = g.drop_duplicates("av", keep="last")
        j = int(g.j.iat[0])
        av, b = g.av.to_numpy(), g[col].to_numpy()
        for k in range(len(av)):
            t1 = av[k + 1] if k + 1 < len(av) else T
            out[av[k]:t1, j] = b[k] / cumF[av[k]:t1, j]
    return out


def neighbor_audit(s: pd.DataFrame, col: str) -> pd.Series:
    """監査用（未来の短信も使う）：前後の短信の両方と10%超ずれ、前後どうしは2%以内でそろっている短信（開発役の診断と同じ定義）。
    同じ利用開始日の重複は最後の1件だけで見る。"""
    flag = pd.Series(False, index=s.index)
    for _, g in s.groupby("Code", sort=False):
        g = g.drop_duplicates("av", keep="last")
        x = np.log(g[col].to_numpy())
        for k in range(1, len(g) - 1):
            a, b = x[k] - x[k - 1], x[k] - x[k + 1]
            if abs(a) > np.log(1.1) and abs(b) > np.log(1.1) and abs(x[k - 1] - x[k + 1]) < 0.02:
                flag[g.index[k]] = True
    return flag


def main() -> None:
    mk = bt2.load_market()
    s, cumF = build_records(mk)
    s = apply_fix(s, cumF)
    s["key"] = s.Code + "|" + s.DiscDate.astype(str) + "|" + s.CurPerEn.astype(str)
    fixed = s[s.fix != ""]
    # ---------- 開発役の一覧との照合
    dv = pd.read_csv(DEV_CSV, dtype={"code": str}, encoding="utf-8-sig").rename(columns={"code": "Code"})
    dv["key"] = dv.Code + "|" + dv.DiscDate.astype(str) + "|" + dv.CurPerEn.astype(str)
    dv_fix = dv[dv.fix.fillna("").ne("")]
    ev_keys, dv_keys = set(fixed.key), set(dv_fix.key)
    both = ev_keys & dv_keys
    m = fixed.set_index("key").loc[sorted(both)]
    d = dv_fix.drop_duplicates("key").set_index("key").loc[sorted(both)]
    same_value = np.isclose(m.base.to_numpy(), d.base_fixed.to_numpy(), rtol=1e-9)
    same_type = (m.fix.to_numpy() == d.fix.to_numpy())
    # ---------- 評価役のフェーズ3の一覧（179件）との照合
    ev3 = pd.read_csv(EVAL_CSV, dtype={"code": str})
    ev3["key"] = ev3.code + "|" + ev3.disc.astype(str) + "|" + ev3.period_end.astype(str)
    ev3["cls"] = np.where(ev3.split_after_pe_before_disc.str.len() > 2, "A:期末後〜開示日に分割",
                          np.where(ev3.split_0to5bd_before_pe.str.len() > 2, "B:期末前5営業日に分割", "C:近くに分割なし"))
    ev3["dev_fixed"] = ev3.key.isin(dv_keys)
    ev3["dev_listed"] = ev3.key.isin(set(dv.key))
    ev3["eval_rule_fixed"] = ev3.key.isin(ev_keys)
    sidx = s.drop_duplicates("key", keep="last").set_index("key")
    ev3["fixed_base"] = ev3.key.map(sidx.base)
    # 補正後の値が「正しい値」（前後の短信の幾何平均＝評価役の X_ok）と5%以内で合うか
    ev3["fix_matches_neighbors"] = np.abs(np.log(ev3.fixed_base / ev3.X_ok)) <= np.log(1.05)
    # 監査：補正の前後で、前後の短信とずれている短信の数
    s["flag_raw"] = neighbor_audit(s, "base_raw")
    s["flag_fix"] = neighbor_audit(s, "base")
    # 補正した短信のうち、補正が「誤り」の疑い（補正前は前後とそろっていたのに、補正後にずれた）
    s["fix_made_worse"] = (s.fix != "") & s.flag_fix & ~s.flag_raw
    # 開発役が補正しなかった評価役の一覧の短信について、理由を調べる（どの範囲に分割があったか）
    F = mk["F"]
    rows = []
    for r in ev3[~ev3.dev_fixed].itertuples():
        g = s[s.key == r.key]
        if not len(g):
            rows.append({"key": r.key, "reason": "記録なし"})
            continue
        g = g.iloc[-1]
        j, pe, dc = int(g.j), int(g.pe), int(g.disc)
        ev_a = [(str(mk["dates"][t]), float(F[t, j])) for t in range(pe + 1, dc + 1) if abs(F[t, j] - 1) > 1e-12]
        ev_b = [(str(mk["dates"][t]), float(F[t, j])) for t in range(max(pe - 10, 0), pe + 1) if abs(F[t, j] - 1) > 1e-12]
        rows.append({"key": r.key, "cls": r.cls, "ratio_vs_prev": r.ratio_vs_prev, "splits_a": ev_a, "splits_pe_minus10_to_pe": ev_b})
    nf = pd.DataFrame(rows)
    ev3.drop(columns=["split_after_pe_before_disc", "split_0to5bd_before_pe"]).to_csv(
        EVAL_ROOT / "data/phase4_shares_vs_eval.csv", index=False)
    nf.to_csv(EVAL_ROOT / "data/phase4_shares_not_fixed.csv", index=False)
    s[(s.fix != "") | s.flag_raw | s.flag_fix].drop(columns=["ShOutFY"]).to_csv(
        EVAL_ROOT / "data/phase4_shares_fix_reports.csv", index=False)
    shares_fix = shares_matrix(s, cumF, "base")
    shares_raw = shares_matrix(s, cumF, "base_raw")
    same_raw = np.array_equal(np.nan_to_num(shares_raw, nan=-1), np.nan_to_num(mk["shares"], nan=-1))
    np.savez_compressed(EVAL_ROOT / "data/processed/shares_fixed.npz", shares=shares_fix)
    out = {
        "eval_rule_fixed": {"reports": int(len(fixed)), "codes": int(fixed.Code.nunique()),
                            "by_type": fixed.fix.value_counts().to_dict()},
        "dev_fixed": {"reports": int(len(dv_fix)), "codes": int(dv_fix.Code.nunique()),
                      "by_type": dv_fix.fix.value_counts().to_dict()},
        "match": {"both": len(both), "eval_only": sorted(ev_keys - dv_keys), "dev_only": sorted(dv_keys - ev_keys),
                  "same_value_in_both": int(same_value.sum()), "same_type_in_both": int(same_type.sum())},
        "raw_shares_matrix_equals_market2": bool(same_raw),
        "eval_phase3_list": {
            "reports": int(len(ev3)), "codes": int(ev3.code.nunique()),
            "by_class": ev3.cls.value_counts().to_dict(),
            "in_dev_csv": int(ev3.dev_listed.sum()),
            "dev_fixed_by_class": ev3[ev3.dev_fixed].cls.value_counts().to_dict(),
            "dev_not_fixed_by_class": ev3[~ev3.dev_fixed].cls.value_counts().to_dict(),
            "dev_fixed_and_matches_neighbors": int((ev3.dev_fixed & ev3.fix_matches_neighbors).sum()),
            "dev_fixed_not_matching_neighbors": ev3[ev3.dev_fixed & ~ev3.fix_matches_neighbors].key.tolist(),
        },
        "dev_csv": {"rows": int(len(dv)), "codes": int(dv.Code.nunique()),
                    "detected_before": int(dv.detected_before.sum()),
                    "detected_before_codes": int(dv[dv.detected_before].Code.nunique()),
                    "not_in_eval_list_detected": int((dv.detected_before & ~dv.key.isin(set(ev3.key))).sum())},
        "audit_neighbors": {"flag_before_fix": int(s.flag_raw.sum()), "flag_after_fix": int(s.flag_fix.sum()),
                            "fix_made_worse": s[s.fix_made_worse].key.tolist(),
                            "fixed_and_unflagged_before": int(((s.fix != "") & ~s.flag_raw).sum())},
    }
    (EVAL_ROOT / "data/phase4_shares_fix.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str),
                                                         encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
