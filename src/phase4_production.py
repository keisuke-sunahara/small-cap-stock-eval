"""フェーズ4前半：確定したモデル（EXP-002）と参考の EXP-001 の、30万円での本番想定の見込みとストレステスト（評価役）。

週次リターンは src/phase4_model_check.py が評価役のエンジンで計算したもの（開発役と全週一致を確認済み）。
1. 30万円・予算固定の1年間の損益（円）：週の損益を足し合わせる（予算固定は毎週の予算が一定なので、円の損益は足し算）。
   データ費用（年19,800円）を引き、利益が出た年だけ税（20.315%）を引く
   - すべての開始週からの52週（重なりあり）と、4週ブロックの再標本化（ブートストラップ、10,000回）
   - 1年で損をしている確率、最悪の週
2. 自分の注文による値動き（マーケット・インパクト）の見積もり：EXP-002 の実際の約定に、平方根モデル（フェーズ2と同じ。
   σ20 × √(注文 ÷ (s × その日の売買代金))、s＝寄付・引けの板寄せの割合の仮定 0.05〜0.15）
3. 配当の見積もり：保有した日数 × 直近の本決算の年間配当 ÷ 買値（分割は調整係数で直す）。データに配当は入っていないため
4. ストレステスト：コスト2倍（0.6%）、寄付の後に不利な価格（始値と高値の中間）で買う、
   大きな下落局面（2020-02〜03、2024-08）、開始月を1か月ずつずらした1年間の成績
5. 参考：資金を100万円・300万円にした場合（データ費用の割合が下がり、点数の上位の株を買えるようになる）
結果：data/phase4_production.json
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
from jq import EVAL_ROOT  # noqa: E402
from phase2_cost import sigma20  # noqa: E402
from phase4_model_check import load_scores  # noqa: E402

WDIR = EVAL_ROOT / "data" / "phase4_model_weekly"
OUT = EVAL_ROOT / "data" / "phase4_production.json"
FEE = 19_800
TAX = 0.20315
CAP = 300_000


def annual_yen(pnl_weeks: np.ndarray, fee: float = FEE) -> float:
    x = pnl_weeks.sum() - fee
    return x * (1 - TAX) if x > 0 else x


def dist(a: np.ndarray) -> dict:
    a = np.asarray(a)
    return {"p05": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)), "mean": float(a.mean()),
            "prob_loss": float((a < 0).mean())}


def year_stats(ret: np.ndarray, rng: np.random.Generator, cap: float = CAP, fee: float = FEE) -> dict:
    pnl = ret * cap
    roll = np.array([annual_yen(pnl[i:i + 52], fee) for i in range(len(pnl) - 51)])
    n, B = len(pnl), 4
    boots = []
    for _ in range(10_000):
        starts = rng.integers(0, n - B + 1, size=13)
        boots.append(annual_yen(np.concatenate([pnl[s:s + B] for s in starts]), fee))
    return {"rolling_52w": dist(roll), "bootstrap_52w": dist(np.array(boots)),
            "mean_weekly_yen": float(pnl.mean()), "annual_yen_simple_before_fee": float(pnl.mean() * 52),
            "worst_week_yen": float(pnl.min()), "best_week_yen": float(pnl.max()),
            "weeks_loss_over_10000": int((pnl < -10_000).sum()), "weeks": int(n)}


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(20261005)
    mk = bt2.load_market()
    mk["shares"] = np.load(EVAL_ROOT / "data/processed/shares_fixed.npz")["shares"]
    pc = bt2.Precomp(mk, exclude_margin_other=True, case_a=True)
    di = {d: i for i, d in enumerate(pc.dates)}
    ci = {c: j for j, c in enumerate(pc.codes)}
    out: dict = {}
    W = {e: pd.read_csv(WDIR / f"{e}.csv") for e in ("EXP-002", "EXP-001")}
    # ---- 1. 1年間の損益（円）
    out["yen"] = {}
    for e, w in W.items():
        for k in ("fixed|cost_0.3%", "fixed|cost_tick", "fixed|no_cost"):
            out["yen"][f"{e}/{k}"] = year_stats(w[k].to_numpy(), rng)
            r = out["yen"][f"{e}/{k}"]
            print(e, k, {kk: round(v) for kk, v in r["rolling_52w"].items() if kk != "prob_loss"},
                  "P(loss) roll", round(r["rolling_52w"]["prob_loss"], 3), "boot", round(r["bootstrap_52w"]["prob_loss"], 3),
                  "worst wk", round(r["worst_week_yen"]), flush=True)
    # 開始月を1か月ずつずらした1年間（月の最初の週から52週）
    w = W["EXP-002"]
    ym = w["week_end"].str[:7]
    firsts = [int(np.flatnonzero(ym == m)[0]) for m in sorted(ym.unique())]
    out["by_start_month"] = {}
    for e, ww in W.items():
        for k in ("fixed|cost_0.3%", "fixed|cost_tick"):
            v = [annual_yen(ww[k].to_numpy()[i:i + 52] * CAP) for i in firsts if i + 52 <= len(ww)]
            out["by_start_month"][f"{e}/{k}"] = {"n": len(v), "min": float(min(v)), "max": float(max(v)),
                                                 "median": float(np.median(v)), "share_positive": float(np.mean(np.array(v) > 0))}
    # 大きな下落局面
    out["crash"] = {}
    for e, ww in W.items():
        for name, (a, b) in {"2020-02〜03": ("2020-02-01", "2020-03-31"), "2024-08": ("2024-07-29", "2024-08-30")}.items():
            m = (ww["week_end"] >= a) & (ww["week_end"] <= b)
            out["crash"][f"{e}/{name}"] = {"weeks": int(m.sum()), "yen_0.3%": float(ww.loc[m, "fixed|cost_0.3%"].sum() * CAP),
                                           "yen_no_cost": float(ww.loc[m, "fixed|no_cost"].sum() * CAP)}
    # ---- 2. マーケット・インパクトの見積もり（EXP-002 の約定）
    tr = pd.read_csv(WDIR / "EXP-002_trades.csv", dtype={"code": str})
    tr = tr[tr["how"] != "not_filled"]
    sig = sigma20(pc)
    va = np.nan_to_num(pc.mk["Va"], nan=0.0)
    rows = []
    for x in tr.itertuples():
        j, tb, ts = ci[x.code], di[x.buy_date], di[x.sell_date]
        qb, qs = x.buy_price * x.shares_buy, x.sell_price * x.shares_sell
        rows.append({"sig": sig[tb - 1, j], "qb": qb / va[tb, j] if va[tb, j] > 0 else np.nan,
                     "qs": qs / va[ts, j] if va[ts, j] > 0 else np.nan, "price": x.buy_price,
                     "tick": bt2.tick_size(x.buy_price) / x.buy_price, "q_adv": qb / pc.adv[tb - 1, j]})
    g = pd.DataFrame(rows)
    imp = {}
    for s in (0.05, 0.10, 0.15):
        a = np.r_[g.sig * np.sqrt(g.qb.clip(upper=1) / s), g.sig * np.sqrt(g.qs.clip(upper=1) / s)]
        imp[str(s)] = {"mean": float(np.nanmean(a)), "median": float(np.nanmedian(a))}
    out["impact_EXP-002"] = {"trades": int(len(g)), "price_median": float(g.price.median()),
                             "tick_ratio_median": float(g.tick.median()), "q_adv_median": float(g.q_adv.median()),
                             "sigma20_median": float(g.sig.median()), "sqrt_model_one_way": imp}
    print("impact", out["impact_EXP-002"], flush=True)
    # ---- 3. 配当の見積もり（EXP-002、予算固定・0.3% の約定）
    s = pd.read_parquet(EVAL_ROOT / "data/processed/summary_all.parquet",
                        columns=["DiscDate", "DiscTime", "Code", "DocType", "CurPerType", "DivAnn"])
    s = s[s.Code.isin(ci) & s.DocType.str.contains("FinancialStatements", na=False) & s.CurPerType.eq("FY")].copy()
    from market2 import availability_date
    s["avail"] = availability_date(s.DiscDate, s.DiscTime, list(pc.dates))
    s = s[s.avail.isin(di)].copy()
    s["av"] = s.avail.map(di)
    s["DivAnn"] = pd.to_numeric(s["DivAnn"].replace("", np.nan), errors="coerce")
    s = s.sort_values(["av", "DiscDate", "DiscTime"])
    by = {c: g_ for c, g_ in s.groupby("Code")}
    div_yen, n_known = 0.0, 0
    for x in tr.itertuples():
        g_ = by.get(x.code)
        tb, ts = di[x.buy_date], di[x.sell_date]
        if g_ is None:
            continue
        h = g_[g_.av < tb]
        if not len(h) or not np.isfinite(h.DivAnn.iloc[-1]):
            continue
        j = ci[x.code]
        dps = h.DivAnn.iloc[-1] * pc.cumF[tb, j] / pc.cumF[int(h.av.iloc[-1]), j]
        days = (pd.Timestamp(x.sell_date) - pd.Timestamp(x.buy_date)).days + 1
        div_yen += dps * x.shares_buy * days / 365.25
        n_known += 1
    yrs = 364 * 7 / 365.25
    out["dividend_EXP-002"] = {"trades_with_dps": n_known, "trades": int(len(tr)), "yen_total": div_yen,
                               "yen_per_year_before_tax": div_yen / yrs, "yen_per_year_after_tax": div_yen / yrs * (1 - TAX),
                               "pct_of_capital_per_year": div_yen / yrs / CAP}
    print("dividend", out["dividend_EXP-002"], flush=True)
    # ---- 4. ストレステスト（EXP-002 と EXP-001、予算固定）
    out["stress"] = {}
    for e in ("EXP-002", "EXP-001"):
        M, _ = load_scores(pc, e)
        pc.score[e] = M
        base = W[e]["fixed|cost_0.3%"].to_numpy()
        r6 = bt2.simulate(pc, e, 0.006, budget_mode="fixed", record=False).weekly["ret"].to_numpy()
        st = {"cost_0.6%": {"annual_yen_simple": float(r6.mean() * 52 * CAP), "rolling": dist(
            [annual_yen(r6[i:i + 52] * CAP) for i in range(len(r6) - 51)])}}
        # 1割の週の見送りは計算しない（エンジンの変更が要る。期待値がマイナスの戦略では、見送りは損益を0に近づけるだけ）
        # 寄付の後に不利な価格で買う：買値を (始値 + 高値) ÷ 2 にした場合。週次の損益に、約定した買いごとの差額を加える
        trf = pd.read_csv(WDIR / f"{e}_trades.csv", dtype={"code": str}) if (WDIR / f"{e}_trades.csv").exists() else None
        if trf is not None:
            trf = trf[trf["how"] != "not_filled"]
            extra = np.zeros(len(base))
            wk_of = {str(d): i for i, d in enumerate(W[e]["week_end"])}
            wstart = pd.read_csv(WDIR / f"{e}.csv")["week_end"]
            for x in trf.itertuples():
                tb = di[x.buy_date]
                j = ci[x.code]
                hi = pc.H[tb, j]
                adv_px = (x.buy_price + hi) / 2 if np.isfinite(hi) else x.buy_price
                # その買いの週（週の最終営業日が買った日以降で最初の週）
                wi = int(np.searchsorted(wstart.to_numpy().astype(str), str(x.buy_date)))
                extra[wi] -= (adv_px - x.buy_price) * x.shares_buy * 1.003 / CAP
            ra = base + extra
            st["buy_mid_open_high"] = {"annual_yen_simple": float(ra.mean() * 52 * CAP),
                                       "extra_cost_per_buy_pct_median": None,
                                       "rolling": dist([annual_yen(ra[i:i + 52] * CAP) for i in range(len(ra) - 51)])}
        out["stress"][e] = st
        print("stress", e, {k: v.get("annual_yen_simple", v.get("annual_yen_simple_mean")) for k, v in st.items()}, flush=True)
    # ---- 5. 参考：資金を増やした場合（予算固定、一律0.3%・銘柄ごと）
    ua = bt2.universe_average(pc, costs=(0.0,))
    wk_meta = pd.DataFrame({"pred_date": [pc.dates[w_[0] - 1] for w_ in pc.weeks], "week_end": [pc.dates[w_[-1]] for w_ in pc.weeks]})
    yrs_c = bt2.years_of(wk_meta)
    out["capital"] = {}
    for capv in (300_000, 1_000_000, 3_000_000):
        bt2.CAPITAL = capv
        for e in ("EXP-002", "EXP-001"):
            for cname, c in (("cost_0.3%", 0.003), ("cost_tick", "tick"), ("no_cost", 0.0)):
                res = bt2.simulate(pc, e, c, budget_mode="fixed", record=False)
                m = bt2.metrics(res.weekly["ret"], years=yrs_c)
                out["capital"][f"{capv}/{e}/{cname}"] = {"annual": m["annual_return"], "data_fee_pct": FEE / capv,
                                                         "after_fee": m["annual_return"] - FEE / capv,
                                                         "simple_annual_pct": float(res.weekly["ret"].mean() * 52),
                                                         "orders": res.orders}
                print("capital", capv, e, cname, round(m["annual_return"], 4), round(m["annual_return"] - FEE / capv, 4), flush=True)
    bt2.CAPITAL = CAP
    out["elapsed_sec"] = time.time() - t0
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("saved", OUT)


if __name__ == "__main__":
    main()
