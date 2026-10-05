"""フェーズ4前半（モデルの実験）の独立検証：開発役の点数（logs/backtest/<EXP>/scores.parquet）だけを使い、
評価役のエンジン（bt2.py）で「点数 → 銘柄選択 → 注文 → 約定 → 損益」を計算し直して、開発役の数字と照合する。

- 点数のファイルの日・銘柄が、評価役のユニバースと一致するか（予測日・継続の判断の日）
- Rank IC（予測日の点数と翌週の実現リターン。終値が無い日は直前の終値）、平均・t値・年ごと
- 予算固定・基本ルール × コスト4通り：週次リターン（開発役の weekly_returns.csv と週ごとに照合）・年率・超過・最大DD・年ごとの超過
- 予算固定・一律0.3% の注文と約定を、開発役の orders.csv・trades.csv と照合
- データ費用（年6.6%）を引いた年率
- モデルの確定の手順（README 第4章）：EXP-001 − EXP-002 の週次の差の t値
結果：data/phase4_model_check.json、data/phase4_model_weekly/<EXP>.csv
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402

PROJ = EVAL_ROOT.parent / "project"
OUT = EVAL_ROOT / "data" / "phase4_model_check.json"
WDIR = EVAL_ROOT / "data" / "phase4_model_weekly"
COSTS = {"no_cost": 0.0, "cost_0.3%": 0.003, "cost_0.5%": 0.005, "cost_tick": "tick"}
DATA_FEE = 19_800 / bt2.CAPITAL
EXPS = sys.argv[1:] or ["EXP-002", "EXP-001", "EXP-003", "EXP-004", "EXP-005"]
FULL = {"EXP-002", "EXP-001"}


def load_scores(pc: bt2.Precomp, exp: str) -> tuple[np.ndarray, dict]:
    s = pd.read_parquet(PROJ / "logs/backtest" / exp / "scores.parquet")
    di = {d: i for i, d in enumerate(pc.dates)}
    ci = {c: j for j, c in enumerate(pc.codes)}
    t = s["date"].map(di).to_numpy()
    j = s["code"].map(ci).to_numpy()
    M = np.full(pc.C.shape, np.nan)
    M[t.astype(int), j.astype(int)] = s["score"].to_numpy()
    # 照合：点数のある日 × 銘柄 と 評価役のユニバース
    days = np.unique(t.astype(int))
    has = ~np.isnan(M[days])
    uni = pc.universe[days]
    chk = {"rows": int(len(s)), "days": int(len(days)), "missing_code_or_date": int(np.isnan(t).sum() + np.isnan(j).sum()),
           "in_scores_not_universe": int((has & ~uni).sum()), "in_universe_not_scores": int((~has & uni).sum())}
    need = sorted({w[0] - 1 for w in pc.weeks} | {w[-1] - 1 for w in pc.weeks if len(w) >= 2})
    chk["needed_days"] = len(need)
    chk["needed_days_without_scores"] = [str(pc.dates[d]) for d in need if d not in set(days)]
    return M, chk


def rank_ic(pc: bt2.Precomp, M: np.ndarray) -> dict:
    ics, yrs = [], []
    for w in pc.weeks:
        p, a, b = w[0] - 1, w[0], w[-1]
        idx = np.flatnonzero(pc.universe[p])
        s = M[p, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = pc.cadj_ff[b, idx] / (pc.O[a, idx] / pc.cumF[a, idx]) - 1
        ok = np.isfinite(s) & np.isfinite(r)
        ics.append(spearmanr(s[ok], r[ok]).statistic if ok.sum() > 2 else np.nan)
        yrs.append(str(pc.dates[p])[:4])
    ics = np.array(ics)
    ok = np.isfinite(ics)
    m, sd = ics[ok].mean(), ics[ok].std(ddof=1)
    by = pd.Series(ics).groupby(yrs).mean().to_dict()
    return {"mean": float(m), "std": float(sd), "t": float(m / sd * np.sqrt(ok.sum())), "weeks": int(ok.sum()),
            "by_year": {k: float(v) for k, v in by.items()}, "weekly": ics.tolist()}


def deciles(pc: bt2.Precomp, M: np.ndarray) -> list[float]:
    acc = [[] for _ in range(10)]
    for w in pc.weeks:
        p, a, b = w[0] - 1, w[0], w[-1]
        idx = np.flatnonzero(pc.universe[p])
        s = M[p, idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            r = pc.cadj_ff[b, idx] / (pc.O[a, idx] / pc.cumF[a, idx]) - 1
        ok = np.isfinite(s) & np.isfinite(r)
        if ok.sum() < 10:
            continue
        q = pd.qcut(pd.Series(s[ok]).rank(method="first"), 10, labels=False).to_numpy()
        for k in range(10):
            acc[k].append(r[ok][q == k].mean())
    return [float(np.mean(a)) for a in acc]


def by_year_excess(w: pd.DataFrame, ua_ret: pd.Series) -> dict:
    yr = w["week_end"].astype(str).str[:4].to_numpy()   # 開発役と同じく、週の最終営業日の年
    out = {}
    for y in sorted(set(yr)):
        m = yr == y
        a = float((1 + w["ret"][m]).prod() - 1)
        u = float((1 + ua_ret[m]).prod() - 1)
        out[y] = {"return": a, "universe": u, "excess": a - u}
    return out


def main() -> None:
    t0 = time.time()
    WDIR.mkdir(parents=True, exist_ok=True)
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    ua = bt2.universe_average(pc, costs=(0.0,))
    ua_ret = ua["ret_0.0"].reset_index(drop=True)
    wk_meta = pd.DataFrame({"pred_date": [pc.dates[w[0] - 1] for w in pc.weeks], "week_end": [pc.dates[w[-1]] for w in pc.weeks]})
    yrs = bt2.years_of(wk_meta)
    bench = bt2.metrics(ua_ret, years=yrs)["annual_return"]
    out = {"years": yrs, "universe_avg_no_cost_annual": bench, "data_fee": DATA_FEE, "exp": {}}
    weekly_all = {}
    for exp in EXPS:
        M, chk = load_scores(pc, exp)
        pc.score[exp] = M
        dev = json.load(open(next((PROJ / "experiments").glob(f"{exp}_*")) / "metrics.json", encoding="utf-8"))
        dwr = pd.read_csv(PROJ / "logs/backtest" / exp / "weekly_returns.csv")
        rec = {"score_check": chk}
        ric = rank_ic(pc, M)
        rec["rank_ic"] = {"eval": {k: v for k, v in ric.items() if k != "weekly"},
                          "dev": {k: dev["rank_ic"][k] for k in ("mean", "std", "t", "weeks", "by_year")},
                          "weekly_max_abs_diff": float(np.nanmax(np.abs(np.array(ric["weekly"]) - np.array(dev["rank_ic"]["weekly"]))))}
        print(exp, "IC", ric["mean"], ric["t"], "dev", dev["rank_ic"]["mean"], dev["rank_ic"]["t"], chk, flush=True)
        if exp in FULL:
            rec["deciles"] = {"eval": deciles(pc, M), "dev": dev["deciles_mean_weekly_return"]}
        combos = [(bm, c) for bm in ("fixed", "min_equity") for c in COSTS] if exp in FULL else [("fixed", "cost_0.3%")]
        rec["bt"] = {}
        wdf = pd.DataFrame({"week_end": wk_meta["week_end"]})
        for bm, cname in combos:
            res = bt2.simulate(pc, exp, COSTS[cname], budget_mode=bm, record=(bm == "fixed" and cname == "cost_0.3%"))
            w = res.weekly
            m = bt2.metrics(w["ret"], years=yrs)
            d = dev["backtest"][bm][cname]
            key = f"{bm}|{cname}"
            diff = np.abs(w["ret"].to_numpy() - dwr[key].to_numpy())
            wdf[key] = w["ret"].to_numpy()
            r = {"eval": {"annual": m["annual_return"], "excess": m["annual_return"] - bench,
                          "after_data_fee": m["annual_return"] - DATA_FEE, "max_dd": m["max_drawdown"],
                          "worst_week": m["worst_week"], "sharpe": m["sharpe"], "orders": res.orders,
                          "fill_rate": res.fills / max(res.orders, 1),
                          "turnover_per_year": float(w["traded"].sum() / bt2.CAPITAL / yrs) if bm == "fixed" else None,
                          "by_year": by_year_excess(w, ua_ret)},
                 "dev": {"annual": d["annual_return"], "excess": d.get("annual_excess_vs_universe"), "max_dd": d["max_drawdown"],
                         "worst_week": d["worst_week"], "orders": d.get("orders"), "fill_rate": d.get("fill_rate"),
                         "turnover_per_year": d.get("turnover_per_year"),
                         "by_year_excess": {y: v["excess"] for y, v in d.get("by_year", {}).items()}},
                 "weekly_max_abs_diff": float(diff.max()), "weeks_diff_gt_1e-9": int((diff > 1e-9).sum())}
            if bm == "fixed" and cname == "cost_0.3%":
                r["eval"]["breakeven_vs_universe"] = bt2.breakeven_cost(w, bench)
                r["dev"]["breakeven_vs_universe"] = d.get("breakeven_one_way_cost_vs_universe")
                # 注文・約定の照合
                tr = res.trades
                dor = pd.read_csv(PROJ / "logs/backtest" / exp / "orders.csv", dtype={"code": str})
                dtr = pd.read_csv(PROJ / "logs/backtest" / exp / "trades.csv", dtype={"code": str})
                ev_orders = set(zip(tr["buy_date"].astype(str), tr["code"].astype(str)))
                dv_orders = set(zip(dor["buy_date"].astype(str), dor["code"].astype(str)))
                ev_fill = set(zip(tr.loc[tr["how"] != "not_filled", "buy_date"].astype(str), tr.loc[tr["how"] != "not_filled", "code"].astype(str)))
                db = dtr[dtr["side"] == "buy"]
                dv_fill = set(zip(db["date"].astype(str), db["code"].astype(str)))
                ev_sell = tr[tr["how"] != "not_filled"]
                ev_sells = set(zip(ev_sell["sell_date"].astype(str), ev_sell["code"].astype(str)))
                ds = dtr[dtr["side"] == "sell"]
                dv_sells = set(zip(ds["date"].astype(str), ds["code"].astype(str)))
                r["orders_match"] = {"eval": len(ev_orders), "dev": len(dv_orders), "only_eval": sorted(ev_orders - dv_orders)[:10],
                                     "only_dev": sorted(dv_orders - ev_orders)[:10]}
                r["fills_match"] = {"eval": len(ev_fill), "dev": len(dv_fill), "only_eval": sorted(ev_fill - dv_fill)[:10],
                                    "only_dev": sorted(dv_fill - ev_fill)[:10]}
                r["sells_match"] = {"eval": len(ev_sells), "dev": len(dv_sells), "only_eval": sorted(ev_sells - dv_sells)[:10],
                                    "only_dev": sorted(dv_sells - ev_sells)[:10]}
                tr.to_csv(WDIR / f"{exp}_trades.csv", index=False)
            rec["bt"][key] = r
            print(exp, key, f"eval {m['annual_return']:.6f} dev {d['annual_return']:.6f} wkdiff {diff.max():.2e}",
                  f"{time.time() - t0:.0f}s", flush=True)
        wdf.to_csv(WDIR / f"{exp}.csv", index=False)
        weekly_all[exp] = wdf
        out["exp"][exp] = rec
    # モデルの確定の手順（README 第4章）
    if {"EXP-001", "EXP-002"} <= set(weekly_all):
        sel = {}
        for cname in ("cost_0.3%", "cost_tick"):
            dd = weekly_all["EXP-001"][f"fixed|{cname}"] - weekly_all["EXP-002"][f"fixed|{cname}"]
            sel[cname] = {"mean_weekly_diff": float(dd.mean()), "t": float(dd.mean() / dd.std(ddof=1) * np.sqrt(len(dd))),
                          "weeks": int(len(dd))}
        out["selection"] = sel
        print("selection", sel)
    out["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
