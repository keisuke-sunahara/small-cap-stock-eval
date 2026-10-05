"""フェーズ2：「片道0.3%のコスト」の現実性の評価（評価役）。

1. ベースラインが実際に出した注文（約定した買い・売り）について、株価、呼値の単位 ÷ 株価、注文金額 ÷ 売買代金、
   値動きの大きさ（20日の日次リターンの標準偏差）、スプレッドの推定（Abdi-Ranaldo 2017、日足の高値・安値・終値から）を集計
2. コストの見積もり（手数料は楽天証券ゼロコースで0円を確認済み）
   - 寄付・引けの板寄せ（オークション）で約定する場合：スプレッドは払わない。自分の注文による価格の動き（マーケット・インパクト）を
     平方根モデル  impact = Y × σ × sqrt(注文金額 ÷ オークションの売買代金)  で見積もる。
     オークションの売買代金は日中の売買代金の s 倍と仮定（s は J-Quants の Standard では取れないため仮定。0.05〜0.15 で感度を見る）
   - 寄付に間に合わず日中の連続売買で約定する場合：スプレッドの半分 ＋ インパクト
3. コストの水準ごとのベースラインの成績（0〜0.5%）。評価役の実装で計算
出力: data/phase2_cost.json、data/phase2_cost_orders.csv
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bt2  # noqa: E402
from jq import EVAL_ROOT  # noqa: E402


def ar_spread(pc: bt2.Precomp, window: int = 60) -> np.ndarray:
    """Abdi & Ranaldo (2017) のスプレッド推定（2日ごと、負は0、window 日の平均）。分割調整した価格で計算。"""
    cf = pc.cumF
    c = np.log(pc.C / cf)
    h = np.log(pc.H / cf)
    lo = np.log(pc.L / cf)
    eta = (h + lo) / 2
    s2 = 4 * (c[:-1] - eta[:-1]) * (c[:-1] - eta[1:])
    s2 = np.vstack([np.full((1, c.shape[1]), np.nan), s2])
    s = np.sqrt(np.clip(s2, 0, None))
    return pd.DataFrame(s).rolling(window, min_periods=window // 2).mean().to_numpy()


def sigma20(pc: bt2.Precomp) -> np.ndarray:
    lr = np.log(pc.cadj_ff)
    d = np.vstack([np.full((1, lr.shape[1]), np.nan), np.diff(lr, axis=0)])
    d = np.where(np.isnan(pc.C), np.nan, d)
    return pd.DataFrame(d).rolling(20, min_periods=15).std().to_numpy()


def main() -> None:
    pc = bt2.Precomp(bt2.load_market())
    spr = ar_spread(pc)
    sig = sigma20(pc)
    va = np.nan_to_num(pc.mk["Va"], nan=0.0)
    di = {d: i for i, d in enumerate(pc.dates)}
    ci = {c: i for i, c in enumerate(pc.codes)}
    rows = []
    sources = [("momentum_20d", None), ("reversal_5d", None)] + [("random", s) for s in range(30)]
    for kind, seed in sources:
        r = bt2.simulate(pc, kind, 0.003, budget_mode="fixed", seed=seed)
        tr = r.trades[r.trades["how"] != "not_filled"]
        for _, x in tr.iterrows():
            j = ci[x["code"]]
            tb = di[x["buy_date"]]
            ts = di[x["sell_date"]]
            p = tb - 1
            q_buy = x["buy_price"] * x["shares_buy"]
            q_sell = x["sell_price"] * x["shares_sell"]
            rows.append({"src": kind, "seed": seed, "code": x["code"], "buy_date": x["buy_date"],
                         "price": x["buy_price"], "tick_ratio": bt2.tick_size(x["buy_price"]) / x["buy_price"],
                         "q_buy": q_buy, "adv20": pc.adv[p, j], "q_adv": q_buy / pc.adv[p, j],
                         "q_va_buyday": q_buy / va[tb, j] if va[tb, j] > 0 else np.nan,
                         "q_va_sellday": q_sell / va[ts, j] if va[ts, j] > 0 else np.nan,
                         "sigma20": sig[p, j], "spread_ar": spr[p, j], "how": x["how"]})
    df = pd.DataFrame(rows)
    df.to_csv(EVAL_ROOT / "data" / "phase2_cost_orders.csv", index=False)

    def q(s: pd.Series) -> dict:
        s = s.dropna()
        return {k: float(s.quantile(v)) for k, v in (("p10", .1), ("p25", .25), ("median", .5), ("p75", .75), ("p90", .9))} | {"mean": float(s.mean())}

    out: dict = {"n_trades": {k: int(v) for k, v in df.groupby("src").size().items()}}
    for src, g in df.groupby("src"):
        o = {c: q(g[c]) for c in ("price", "tick_ratio", "q_buy", "q_adv", "q_va_buyday", "q_va_sellday", "sigma20", "spread_ar")}
        # インパクトの見積もり（片道）。Y=1（保守的）。オークションの売買代金 = s × その日の売買代金
        est = {}
        for s in (0.05, 0.10, 0.15):
            imp_buy = g["sigma20"] * np.sqrt(g["q_va_buyday"].clip(upper=1) / s)
            imp_sell = g["sigma20"] * np.sqrt(g["q_va_sellday"].clip(upper=1) / s)
            est[f"auction_share_{s}"] = {"impact_buy": q(imp_buy), "impact_sell": q(imp_sell),
                                         "one_way_mean": float(np.nanmean(np.r_[imp_buy, imp_sell])),
                                         "one_way_median": float(np.nanmedian(np.r_[imp_buy, imp_sell]))}
        cont = g["spread_ar"] / 2 + g["sigma20"] * np.sqrt(g["q_va_buyday"].clip(upper=1))
        est["continuous_half_spread_plus_impact"] = q(cont)
        o["cost_estimates"] = est
        # 株価帯別
        bands = pd.cut(g["price"], [0, 100, 200, 300, 500, 1000, 1e9])
        o["by_price_band"] = {str(b): {"n": int(len(h)), "tick_ratio_median": float(h["tick_ratio"].median()),
                                        "spread_ar_median": float(h["spread_ar"].median()),
                                        "impact_s0.10_median": float(np.nanmedian(h["sigma20"] * np.sqrt(h["q_va_buyday"].clip(upper=1) / 0.10)))}
                              for b, h in g.groupby(bands, observed=True)}
        out[src] = o

    # コストの水準ごとの成績
    levels = [0.0, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005]
    sens = {}
    for kind in ("momentum_20d", "reversal_5d"):
        for bm in ("fixed", "min_equity"):
            sens[f"{kind}|{bm}"] = {str(c): bt2.metrics(bt2.simulate(pc, kind, c, budget_mode=bm, record=False).weekly["ret"])["annual_return"] for c in levels}
    rnd = {}
    for c in levels:
        a = [bt2.metrics(bt2.simulate(pc, "random", c, budget_mode="fixed", seed=20_000 + s, record=False).weekly["ret"])["annual_return"] for s in range(200)]
        rnd[str(c)] = {"mean": float(np.mean(a)), "p05": float(np.quantile(a, .05)), "p95": float(np.quantile(a, .95))}
    sens["random_fixed_200runs"] = rnd
    ua = bt2.universe_average(pc, costs=tuple(levels))
    sens["universe_average"] = {str(c): bt2.metrics(ua[f"ret_{c}"])["annual_return"] for c in levels}
    out["sensitivity_annual_return"] = sens
    (EVAL_ROOT / "data" / "phase2_cost.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved")


if __name__ == "__main__":
    main()
