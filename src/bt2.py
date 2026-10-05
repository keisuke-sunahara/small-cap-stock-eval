"""フェーズ2の独立検証：基本の売買ルールの週次バックテスト（評価役が project/CLAUDE.md 第5章の文章から書いたもの）。

開発役のコード（project/src/）は使わない。細部で文章に書かれていないことは project/docs/DECISIONS.md の記述に合わせ、
それでも決まらないものは評価役が決めた（各所にコメント）。

主な関数
- load_market()       : market2.py が作った行列を読む
- Precomp             : ユニバース・20日平均売買代金・点数などの前計算
- universe_average()  : ベースライン4（ユニバースの等金額平均）
- simulate()          : ベースライン1〜3（ランダム・モメンタム・リバーサル）
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from jq import EVAL_ROOT  # noqa: E402

MARKET = EVAL_ROOT / "data" / "processed" / "market2.npz"
TRADE_START = "2018-10-01"
CAPITAL = 300_000
N_HOLD = 3
LIMIT_UP = 0.02
LOT = 100
MAX_ADV = 0.01
ADV_WIN = 20
MAX_MCAP = 50e9
MIN_TURNOVER = 30e6
MIN_PRICE = 50


# ---------------------------------------------------------------- 呼値（TOPIX100 以外の表）
_TICKS = [(3000, 1), (5000, 5), (30000, 10), (50000, 50), (300000, 100), (500000, 500),
          (3_000_000, 1000), (5_000_000, 5000), (30_000_000, 10000), (50_000_000, 50000)]


def tick_size(price: float) -> int:
    for upper, t in _TICKS:
        if price <= upper:
            return t
    return 100000


def floor_tick(x: float) -> float:
    """x 以下で最大の、呼値に合った価格。境目（3,000円など）は呼値の倍数なので、x の呼値で切り下げればよい。"""
    t = tick_size(x)
    # 浮動小数点の誤差（例：1000*1.02=1020.0000000000001）で1単位ずれないよう、小さな余裕を足す
    return math.floor(x / t + 1e-9) * t


# ---------------------------------------------------------------- 案A（調整係数の誤りとみられる銘柄の除外）
CASE_A_THRESHOLD = 0.30
CASE_A_DAYS = 60


def case_a_events(C: np.ndarray, F: np.ndarray, threshold: float = CASE_A_THRESHOLD) -> np.ndarray:
    """project/CLAUDE.md 第5章の文章から：係数が1でない日（売買不成立の日を含む）の後で最初に売買が成立した日 t に、
    その前に売買が成立した日 s の終値からの調整後リターン（C_t/cumF_t ÷ C_s/cumF_s − 1）が ±threshold を超えたら True。
    係数の日が t 自身（売買が成立した日）でも対象とする（「以降」と読む）。"""
    T, N = C.shape
    traded = ~np.isnan(C)
    cumF = np.cumprod(F, axis=0)
    q = C / cumF
    ev = np.cumsum(np.abs(F - 1) > 1e-12, axis=0)
    idx = np.where(traded, np.arange(T)[:, None], -1)
    last_tr = np.maximum.accumulate(idx, axis=0)          # t 以前で最後に売買が成立した日
    prev_tr = np.full((T, N), -1)
    prev_tr[1:] = last_tr[:-1]                              # t より前で最後に売買が成立した日 s
    ok = traded & (prev_tr >= 0)
    s_ = np.maximum(prev_tr, 0)
    cols = np.arange(N)[None, :].repeat(T, 0)
    spans = ev - ev[s_, cols] > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        r = q / q[s_, cols] - 1
    return ok & spans & (np.abs(r) > threshold)


def exclusion_window(known: np.ndarray, days: int) -> np.ndarray:
    """known の日から days 営業日（その日を含む）True。"""
    c = np.cumsum(known, axis=0)
    before = np.zeros_like(c)
    before[days:] = c[:-days]
    return (c - before) > 0


# ---------------------------------------------------------------- データ
def load_market() -> dict:
    z = np.load(MARKET, allow_pickle=False)
    return {k: z[k] for k in z.files}


@dataclass
class Precomp:
    mk: dict
    exclude_margin_other: bool = False
    case_a: bool = False          # 調整係数が株価の動きと合わない銘柄を60営業日除く（project/CLAUDE.md 第5章、2026-10-04 承認）
    dates: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        mk = self.mk
        self.dates = mk["dates"]
        self.codes = mk["codes"]
        O, C, F = mk["O"], mk["C"], mk["F"]
        self.O, self.H, self.L, self.C = O, mk["H"], mk["L"], C
        self.F = F
        self.LL = mk["LL"] > 0
        self.UL = mk["UL"] > 0
        self.listed = mk["listed"]
        self.cumF = np.cumprod(F, axis=0)
        cadj = C / self.cumF
        self.cadj_ff = pd.DataFrame(cadj).ffill().to_numpy()
        # Mark[t,j]：t 日時点の株数の基準での、直近の終値（売買不成立の日は前の終値を分割で直したもの）
        self.mark = self.cadj_ff * self.cumF
        va = np.nan_to_num(mk["Va"], nan=0.0)
        va = np.where(self.listed, va, 0.0)
        cs = np.cumsum(va, axis=0)
        adv = np.full_like(va, np.nan)
        adv[ADV_WIN - 1:] = cs[ADV_WIN - 1:] - np.vstack([np.zeros((1, va.shape[1])), cs[:-ADV_WIN]])
        adv /= ADV_WIN
        lc = np.cumsum(self.listed.astype(int), axis=0)
        full = np.zeros_like(self.listed)
        full[ADV_WIN - 1:] = (lc[ADV_WIN - 1:] - np.vstack([np.zeros((1, va.shape[1]), int), lc[:-ADV_WIN]])) == ADV_WIN
        self.adv = np.where(full, adv, np.nan)
        mcap = mk["shares"] * C
        uni = (self.listed & ~np.isnan(C) & (C >= MIN_PRICE) & (self.adv >= MIN_TURNOVER)
               & ~np.isnan(mcap) & (mcap <= MAX_MCAP))
        if self.exclude_margin_other:
            uni &= ~mk["margin_other"]
        self.case_a_known = case_a_events(C, F)
        if self.case_a:
            uni &= ~exclusion_window(self.case_a_known, CASE_A_DAYS)
        self.universe = uni
        # 点数（大きいほど上位）
        with np.errstate(divide="ignore", invalid="ignore"):
            r20 = self.cadj_ff / np.vstack([np.full((20, C.shape[1]), np.nan), self.cadj_ff[:-20]]) - 1
            r5 = self.cadj_ff / np.vstack([np.full((5, C.shape[1]), np.nan), self.cadj_ff[:-5]]) - 1
        self.score = {"momentum_20d": r20, "reversal_5d": -r5}
        # 最後に一覧に載っていた日
        T = len(self.dates)
        last = np.where(self.listed.any(axis=0), T - 1 - np.argmax(self.listed[::-1], axis=0), -1)
        self.last_listed = last
        # 週（ISO 週）
        dt = pd.to_datetime(pd.Series(self.dates))
        iso = dt.dt.isocalendar()
        wk = (iso["year"].astype(int) * 100 + iso["week"].astype(int)).to_numpy()
        start = int(np.searchsorted(self.dates, TRADE_START))
        weeks = []
        i = start
        while i < T:
            j = i
            while j + 1 < T and wk[j + 1] == wk[i]:
                j += 1
            weeks.append(list(range(i, j + 1)))
            i = j + 1
        # 最後の週がデータの終わりで切れていないか（ホールドアウト前の最後の週は 2025-09-22〜26 の5日）
        self.weeks = weeks

    def ranking(self, t: int, kind: str, rng: np.random.Generator | None = None) -> np.ndarray:
        idx = np.flatnonzero(self.universe[t])
        if kind == "random":
            return rng.permutation(idx)
        s = self.score[kind][t, idx]
        ok = ~np.isnan(s)
        idx, s = idx[ok], s[ok]
        order = np.lexsort((idx, -s))   # 点数の大きい順、同点はコード順
        return idx[order]


# ---------------------------------------------------------------- 約定
def buy_fill(pc: Precomp, j: int, t: int, limit: float) -> float | None:
    """寄付の指値。始値 < 指値 なら始値で約定。始値なし（寄らず・売買停止）、分割・併合の日は約定しない。"""
    o = pc.O[t, j]
    if np.isnan(o) or pc.F[t, j] != 1.0 or not pc.listed[t, j]:
        return None
    return float(o) if o < limit else None


def sell_close_ok(pc: Precomp, j: int, t: int) -> bool:
    """引成の売り。終値なし、またはストップ安のフラグがあり安値で引けた場合は売れない。"""
    c = pc.C[t, j]
    if np.isnan(c):
        return False
    if pc.LL[t, j] and c <= pc.L[t, j]:
        return False
    return True


@dataclass
class Pos:
    j: int
    t_buy: int
    shares_buy: float
    price_buy: float
    pending: bool = False    # 売れずに、次に売買が成立する日の始値で売る待ち

    def shares_at(self, pc: Precomp, t: int) -> float:
        return self.shares_buy * pc.cumF[self.t_buy, self.j] / pc.cumF[t, self.j]


@dataclass
class Result:
    weekly: pd.DataFrame
    trades: pd.DataFrame
    orders: int
    fills: int


def cost_rate(cost, price: float) -> float:
    """片道のコスト率。cost が数値なら一定。"tick" なら max(0.3%, 1呼値 ÷ 約定価格)（project/CLAUDE.md 第5章）。
    (base, k) なら base + k × 呼値の単位 ÷ 価格（フェーズ2で評価役が使った感度の計算）。"""
    if cost == "tick":
        return max(0.003, tick_size(price) / price)
    if isinstance(cost, tuple):
        base, k = cost
        return base + k * tick_size(price) / price
    return cost


def simulate(pc: Precomp, kind: str, cost, budget_mode: str = "min_equity",
             continuation: str = "prev_day", seed: int | None = None, record: bool = True,
             dev_compat=False, rng_compat: bool = False) -> Result:
    """rng_compat=True は照合用：ランダム選択の乱数の引き方を開発役に合わせる（順位を付けるたびに全銘柄に一様乱数を振り、
    継続の判断は売れ残りでない保有がある週だけ行う）。分布は同じで、乱数の並びだけが変わる。
    dev_compat=True は照合用：開発役の実装の細部（売れ残りの銘柄は注文の対象から外す、評価額は調整前の終値の
    直前値 × 分割で直した株数）に合わせる。本来のルールの計算は dev_compat=False。"""
    rng = np.random.default_rng(seed) if kind == "random" else None
    T = len(pc.dates)
    cash = float(CAPITAL)
    pos: list[Pos] = []
    rows, trades = [], []
    n_orders = n_fills = 0
    rank_cache: dict[int, np.ndarray] = {}

    def rank(t: int) -> np.ndarray:
        if rng_compat and kind == "random":
            sc = rng.random(len(pc.codes))
            idx = np.flatnonzero(pc.universe[t])
            return idx[np.lexsort((idx, -sc[idx]))]
        if t not in rank_cache:
            rank_cache[t] = pc.ranking(t, kind, rng)
        return rank_cache[t]

    dc = {"value", "skip", "eps"} if dev_compat is True else set(dev_compat or ())
    c_ff = pd.DataFrame(pc.C).ffill().to_numpy() if "value" in dc else None

    def equity(t: int) -> float:
        if c_ff is not None:
            return cash + sum(p.shares_at(pc, t) * c_ff[t, p.j] for p in pos)
        return cash + sum(p.shares_at(pc, t) * pc.mark[t, p.j] for p in pos)

    def affordable(j: int, t: int, budget: float) -> tuple[float, int] | None:
        limit = floor_tick(pc.C[t, j] * (1 + LIMIT_UP))
        sh = math.floor(budget / (limit * LOT) + (1e-9 if "eps" in dc else 0.0)) * LOT
        if sh <= 0:
            return None
        if limit * sh > MAX_ADV * pc.adv[t, j]:
            return None
        return limit, sh

    flow = {"traded": 0.0, "cost": 0.0}

    def close_pos(p: Pos, t: int, price: float, how: str) -> None:
        nonlocal cash
        sh = p.shares_at(pc, t)
        cr = cost_rate(cost, price)
        flow["traded"] += sh * price
        flow["cost"] += sh * price * cr
        cash += sh * price * (1 - cr)
        if record:
            trades.append({"code": pc.codes[p.j], "buy_date": pc.dates[p.t_buy], "buy_price": p.price_buy,
                           "shares_buy": p.shares_buy, "sell_date": pc.dates[t], "sell_price": price,
                           "shares_sell": sh, "how": how})
        pos.remove(p)

    def process_day_open(t: int) -> None:
        """売れ残りの売り（始値）と、上場廃止の処理。t は営業日。"""
        for p in list(pos):
            if not pc.listed[t, p.j]:
                # 上場廃止：最後に一覧に載っていた日の直近の終値で売ったものとする
                tl = pc.last_listed[p.j]
                close_pos(p, t, float(pc.mark[tl, p.j]), "delisted")
                continue
            if p.pending and not np.isnan(pc.O[t, p.j]):
                close_pos(p, t, float(pc.O[t, p.j]), "next_open")

    for wi, days in enumerate(pc.weeks):
        d1, dk = days[0], days[-1]
        p_day = d1 - 1
        e_start = equity(p_day)
        # ---- 予測日の引け後：注文 ----
        held = {p.j for p in pos if not p.pending}
        slots = N_HOLD - len(held)
        orders = []
        if len(days) >= 2 and slots > 0:
            base = min(CAPITAL, e_start) if budget_mode == "min_equity" else CAPITAL
            budget = base / N_HOLD
            avail_cash = cash if budget_mode == "min_equity" else math.inf
            skip = {p.j for p in pos}   # 売れ残り（pending）も保有銘柄なので選ばない（CLAUDE.md 第5章「保有していない銘柄」）
            for j in rank(p_day):
                if j in skip:
                    continue
                a = affordable(j, p_day, budget)
                if a is None:
                    continue
                limit, sh = a
                need = limit * sh * (1 + cost_rate(cost, limit))
                if need > avail_cash:
                    continue
                avail_cash -= need
                orders.append((j, limit, sh))
                if len(orders) == slots:
                    break
        # ---- 週の各営業日 ----
        for t in days:
            process_day_open(t)
            if t == d1:
                for j, limit, sh in orders:
                    n_orders += 1
                    px = buy_fill(pc, j, t, limit)
                    if px is None:
                        if record:
                            trades.append({"code": pc.codes[j], "buy_date": pc.dates[t], "buy_price": np.nan,
                                           "shares_buy": sh, "limit": limit, "how": "not_filled"})
                        continue
                    n_fills += 1
                    cr = cost_rate(cost, px)
                    flow["traded"] += px * sh
                    flow["cost"] += px * sh * cr
                    cash -= px * sh * (1 + cr)
                    pos.append(Pos(j, t, sh, px))
        # ---- 最終営業日：継続の判断と引成の売り ----
        if continuation == "none" or (rng_compat and not any(not p.pending for p in pos)):
            keep = set()
        else:
            q = dk - 1 if continuation == "prev_day" else dk
            e_q = equity(q)
            base_q = min(CAPITAL, e_q) if budget_mode == "min_equity" else CAPITAL
            held_now = {p.j for p in pos if not p.pending}
            keep, cnt = set(), 0
            for j in rank(q):
                if cnt >= N_HOLD:
                    break
                if j in held_now:
                    keep.add(j)
                    cnt += 1
                elif affordable(j, q, base_q / N_HOLD) is not None:
                    cnt += 1
        for p in list(pos):
            if p.pending or p.j in keep:
                continue
            if sell_close_ok(pc, p.j, dk):
                close_pos(p, dk, float(pc.C[dk, p.j]), "close")
            else:
                p.pending = True
        e_end = equity(dk)
        if budget_mode == "min_equity":
            r = e_end / e_start - 1 if e_start > 0 else 0.0
        else:
            r = (e_end - e_start) / CAPITAL
        n_held = len(pos)
        rows.append({"week_start": pc.dates[d1], "week_end": pc.dates[dk], "pred_date": pc.dates[p_day],
                     "n_days": len(days), "ret": r, "equity_start": e_start, "equity_end": e_end,
                     "orders": len(orders), "positions_end": n_held,
                     "traded": flow["traded"], "cost_paid": flow["cost"]})
        flow["traded"] = flow["cost"] = 0.0
    weekly = pd.DataFrame(rows)
    return Result(weekly, pd.DataFrame(trades), n_orders, n_fills)


# ---------------------------------------------------------------- ベースライン4
def universe_average(pc: Precomp, costs=(0.0, 0.003, 0.005)) -> pd.DataFrame:
    rows = []
    T = len(pc.dates)
    for days in pc.weeks:
        d1, dk = days[0], days[-1]
        p_day = d1 - 1
        U = np.flatnonzero(pc.universe[p_day])
        rec = {"week_start": pc.dates[d1], "week_end": pc.dates[dk], "n_days": len(days), "n_universe": len(U)}
        if len(days) < 2:
            for c in costs:
                rec[f"ret_{c}"] = 0.0
            rec["fill_rate"] = np.nan
            rows.append(rec)
            continue
        rets = {c: [] for c in costs}
        nf = 0
        for j in U:
            limit = floor_tick(pc.C[p_day, j] * (1 + LIMIT_UP))
            px = buy_fill(pc, j, d1, limit)
            if px is None:
                for c in costs:
                    rets[c].append(0.0)
                continue
            nf += 1
            # 売り：最終営業日の引け。売れなければ次に売買が成立した日の始値。途中で上場廃止なら最後の終値
            exit_px, t_exit = None, None
            for t in days[1:] if len(days) > 1 else []:
                if not pc.listed[t, j]:
                    tl = pc.last_listed[j]
                    exit_px, t_exit = pc.mark[tl, j], tl
                    break
            if exit_px is None:
                if sell_close_ok(pc, j, dk):
                    exit_px, t_exit = float(pc.C[dk, j]), dk
                else:
                    t = dk + 1
                    while t < T:
                        if not pc.listed[t, j]:
                            tl = pc.last_listed[j]
                            exit_px, t_exit = pc.mark[tl, j], tl
                            break
                        if not np.isnan(pc.O[t, j]):
                            exit_px, t_exit = float(pc.O[t, j]), t
                            break
                        t += 1
                    if exit_px is None:   # データの終わりまで売れない：最後の値で評価
                        exit_px, t_exit = pc.mark[T - 1, j], T - 1
            gross = exit_px * pc.cumF[d1, j] / pc.cumF[t_exit, j] / px   # 株数の変化を反映した比
            for c in costs:
                rets[c].append(gross * (1 - cost_rate(c, exit_px)) / (1 + cost_rate(c, px)) - 1)
        for c in costs:
            rec[f"ret_{c}"] = float(np.mean(rets[c])) if len(U) else 0.0
        rec["fill_rate"] = nf / len(U) if len(U) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 指標
def metrics(r: pd.Series, week_end: pd.Series | None = None, years: float | None = None) -> dict:
    """years を渡すと年率 = 倍率^(1/years) − 1（project/CLAUDE.md 第5章：最初の予測日〜最後の週の最終営業日の暦日 ÷ 365.25）。
    省略時は 52 ÷ 週の数。"""
    r = pd.Series(r, dtype=float).reset_index(drop=True)
    n = len(r)
    eq = (1 + r).cumprod()
    total = eq.iloc[-1] - 1
    expo = (1 / years) if years else (52 / n)
    ann = (1 + total) ** expo - 1 if total > -1 else -1.0
    sd = r.std(ddof=1)
    dd = (eq / eq.cummax().clip(lower=1.0) - 1).min()
    out = {"weeks": n, "annual_return": ann, "total_return": total,
           "sharpe": r.mean() / sd * math.sqrt(52) if sd > 0 else float("nan"),
           "max_drawdown": dd, "worst_week": r.min(), "best_week": r.max()}
    if week_end is not None:
        yr = pd.Series(week_end).str[:4].reset_index(drop=True)
        out["by_year"] = {y: float((1 + r[yr == y]).prod() - 1) for y in sorted(yr.unique())}
    return out


def years_of(weekly: pd.DataFrame) -> float:
    """最初の予測日から最後の週の最終営業日までの暦日 ÷ 365.25。"""
    a = pd.Timestamp(str(weekly["pred_date"].iloc[0]))
    b = pd.Timestamp(str(weekly["week_end"].iloc[-1]))
    return (b - a).days / 365.25


def breakeven_cost(weekly: pd.DataFrame, bench_ann: float, capital: float = CAPITAL) -> float | None:
    """予算固定の結果から、年率（暦日で年率化）がベンチマークと等しくなる一律の片道コスト（二分法）。
    週次リターン(c) = (損益 + 払ったコスト − c × 売買代金) ÷ 運用資金。"""
    yrs = years_of(weekly)
    pnl = weekly["ret"].to_numpy() * capital + weekly["cost_paid"].to_numpy()
    tv = weekly["traded"].to_numpy()

    def f(c):
        r = (pnl - c * tv) / capital
        tot = np.prod(1 + r)
        a = tot ** (1 / yrs) - 1 if tot > 0 else -1.0
        return a - bench_ann
    lo, hi = -0.05, 0.10
    if f(lo) <= 0:
        return None
    if f(hi) >= 0:
        return hi
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(mid) > 0 else (lo, mid)
    return (lo + hi) / 2
