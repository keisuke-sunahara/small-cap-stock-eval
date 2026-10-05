"""フェーズ4前半の確認3：学習の範囲（logs/backtest/<EXP>/splits.csv）と点数を付けた日の関係に、未来の情報が入っていないか。

評価役の営業日（評価役のデータ）で、experiments/README.md 第2章・CLAUDE.md 第5章の文章どおりの範囲を作り、照合する。
- 予測日：各週の最初の営業日の前の営業日（2018-09-28〜2025-09-19、364日）。月ごとの「最初の予測日」F
- 学習の最後の起点：F の6営業日前（5営業日の空白）。その起点の目的変数は5営業日後の終値（F の1営業日前）で終わる
- 学習の最初の起点：F の4年前の日付以降の最初の営業日、かつ 2017-01-01 以降（EXP-005 は拡大窓で 2017-01-01 以降の最初の営業日）
- 点数を付けた日 d のモデル：F ≤ d となる最も新しい月。月ごとの点数の日数・行数を splits.csv と照合する
- 点数の日・学習の目的変数の終わりが、ホールドアウトの開始日（2025-09-29）より前か
結果：data/phase4_splits_check.json
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

PROJ = EVAL_ROOT.parent / "project"
HOLDOUT = "2025-09-29"


def main() -> None:
    mk = bt2.load_market()
    dates = mk["dates"]
    pc = bt2.Precomp(mk, exclude_margin_other=False, case_a=False)
    di = {d: i for i, d in enumerate(dates)}
    preds = [w[0] - 1 for w in pc.weeks]
    pm = pd.Series([str(dates[p])[:7] for p in preds])
    first = {m: preds[i] for i, m in reversed(list(enumerate(pm)))}
    months = sorted(first)
    out = {}
    for exp in ["EXP-001", "EXP-002", "EXP-003", "EXP-004", "EXP-005"]:
        sp = pd.read_csv(PROJ / "logs/backtest" / exp / "splits.csv")
        sc = pd.read_parquet(PROJ / "logs/backtest" / exp / "scores.parquet", columns=["date"])
        expanding = exp == "EXP-005"
        bad = []
        # 点数の日 → モデルの月
        firsts = np.array([first[m] for m in months])
        sd = sc["date"].map(di).to_numpy()
        k = np.searchsorted(firsts, sd, side="right") - 1
        cnt_rows = pd.Series(k).value_counts()
        cnt_days = pd.Series(k).groupby(k).apply(lambda x: len(set(sd[x.index]))) if len(k) else pd.Series()
        days_by_model = {i: np.unique(sd[k == i]) for i in range(len(months))}
        for i, m in enumerate(months):
            F = first[m]
            exp_last = dates[F - 6]
            start = pd.Timestamp(str(dates[F])) - pd.DateOffset(years=4)
            lo = max(start, pd.Timestamp("2017-01-01")) if not expanding else pd.Timestamp("2017-01-01")
            exp_first = dates[np.searchsorted(dates, lo.strftime("%Y-%m-%d"))]
            r = sp[sp["month"] == m]
            if len(r) != 1:
                bad.append({"month": m, "issue": "splits.csv に無い"})
                continue
            r = r.iloc[0]
            target_end = dates[di[r["train_last"]] + 5]
            first_scored = dates[days_by_model[i].min()] if len(days_by_model[i]) else None
            row = {"month": m, "first_pred": str(dates[F]), "train_first": r["train_first"], "exp_first": str(exp_first),
                   "train_last": r["train_last"], "exp_last": str(exp_last), "target_end": str(target_end),
                   "first_scored": str(first_scored), "score_days": int(r["score_days"]), "eval_days": int(cnt_days.get(i, 0)),
                   "score_rows": int(r["score_rows"]), "eval_rows": int(cnt_rows.get(i, 0))}
            issues = []
            if r["train_first"] != str(exp_first):
                issues.append("train_first")
            if r["train_last"] != str(exp_last):
                issues.append("train_last")
            if first_scored is not None and not (str(target_end) < str(first_scored)):
                issues.append("target_end >= first_scored")
            if row["score_days"] != row["eval_days"] or row["score_rows"] != row["eval_rows"]:
                issues.append("score count")
            if issues:
                row["issues"] = issues
                bad.append(row)
        out[exp] = {"months": len(sp), "eval_months": len(months), "bad": bad,
                    "max_score_date": str(sc["date"].max()), "max_train_last": str(sp["train_last"].max()),
                    "max_target_end": str(dates[di[sp["train_last"].max()] + 5]),
                    "holdout_ok": bool(sc["date"].max() < HOLDOUT and dates[di[sp["train_last"].max()] + 5] < HOLDOUT),
                    "min_gap_bdays_target_end_to_first_scored": None}
        gaps = []
        for i, m in enumerate(months):
            r = sp[sp["month"] == m].iloc[0]
            if len(days_by_model[i]):
                gaps.append(int(days_by_model[i].min() - (di[r["train_last"]] + 5)))
        out[exp]["min_gap_bdays_target_end_to_first_scored"] = min(gaps)
        print(exp, len(bad), out[exp]["max_score_date"], out[exp]["max_target_end"], "min gap", min(gaps))
        for b in bad[:5]:
            print("  ", b)
    (EVAL_ROOT / "data/phase4_splits_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
