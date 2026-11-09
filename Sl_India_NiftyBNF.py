"""
NIFTY 50 / BANK NIFTY - intraday trend checklist.

Imported by app.py as a strategy module (`Sl_India_NiftyBNF.render()`), the same
way the other Sl_* screeners plug into the dashboard.  It also runs on its own:

    Terminal:   python Sl_India_NiftyBNF.py
                python Sl_India_NiftyBNF.py --capital 500000 --risk-pct 0.5
                python Sl_India_NiftyBNF.py --force        # ignore time gates
                python Sl_India_NiftyBNF.py --color never  # no ANSI

    Browser:    streamlit run Sl_India_NiftyBNF.py

The analysis engine is shared by all three; only the rendering differs.

    Terminal:   python nifty_banknifty_app.py
                python nifty_banknifty_app.py --capital 500000 --risk-pct 0.5
                python nifty_banknifty_app.py --force        # ignore time gates
                python nifty_banknifty_app.py --color never  # no ANSI

    Browser:    streamlit run nifty_banknifty_app.py

Streamlit is only imported when the app is actually launched that way, so the
terminal version runs with just pandas / numpy / yfinance installed.

Engine notes (unchanged from the CLI version):

 1. VWAP is honest.  Yahoo reports ZERO volume for ^NSEI and ^NSEBANK, so the
    original `Volume.replace(0, 1)` turned "VWAP" into a plain running average
    of typical price.  Here we detect that, fall back to the matching ETF
    (NIFTYBEES / BANKBEES) volume as a proxy, and label whichever mode was used.

 2. No forming candle.  The last 15m bar is dropped unless it has actually
    closed, so a 09:49 run no longer scores a 4-minute-old candle.

 3. Entry / stop / target are all measured from the SAME price - the live one.
    The original reported an entry at VWAP while measuring the target from the
    live price, which inflated R:R (1:2.5 printed on a setup that was 1:0.5).

 4. Scoring is symmetric: four equal votes, majority + margin.

 5. Structural stop, Wilder ATR, ADX chop filter, session time gates,
    cost-aware R:R, and position sizing from the NSE lot size.

 6. A price-vs-level vote abstains when price is within 0.1 ATR of the level, so
    a half-point poke through VWAP no longer flips the whole verdict.

 7. No market entry when the first pivot is less than 1 ATR away - that is
    buying into resistance, and the R:R to that level is what you get first.

Data: Yahoo intraday is ~15 min delayed and gives only ~60 days of 15m history.
Fine for study and for planning; NOT good enough to execute or to backtest on.
For live signals use a broker feed (Kite Connect / Fyers / Upstox / Angel One /
Dhan).  Nothing here is investment advice.
"""

from __future__ import annotations

import argparse
import os
import sys
import time as _time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import yfinance as yf

IST = "Asia/Kolkata"
MARKET_OPEN = (9, 15)
NO_ENTRY_BEFORE = (9, 45)      # let the opening auction noise settle
NO_ENTRY_AFTER = (14, 45)      # leave time for the target to be reached
SQUARE_OFF = (15, 15)          # MIS positions get auto-squared ~15:20 with penalty

GREEN, RED, YELLOW, DIM, BOLD = "32", "31", "33", "2", "1"
COLOUR = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def colour(text: str, code: str) -> str:
    """ANSI colour, silently dropped when the output is piped or NO_COLOR is set."""
    return f"\033[{code}m{text}\033[0m" if COLOUR else text


@dataclass(frozen=True)
class Instrument:
    name: str
    symbol: str
    proxy: str          # ETF used only for a volume proxy
    lot_size: int       # NSE market lot, revised Jan 2026
    note: str


INSTRUMENTS = (
    Instrument("NIFTY 50", "^NSEI", "NIFTYBEES.NS", 65,
               "weekly options expire Tuesday; monthly on last Tuesday"),
    Instrument("BANK NIFTY", "^NSEBANK", "BANKBEES.NS", 30,
               "NO weekly options since 20-Nov-2024; monthly only (last Tuesday)"),
)

DEFAULTS = dict(
    capital=500_000.0,
    risk_pct=0.5,
    atr_mult=1.5,
    min_rr=1.5,
    min_room_atr=1.0,    # require this much room to the first level before entering
    cost_points=10.0,    # round-trip cost in index points for one futures lot:
                         # ~7 pts of brokerage/STT/exchange/GST/stamp on a 65-lot
                         # at 24,600, plus ~1-2 pts slippage each side
    atr_floor_pct=0.04,  # skip dead sessions: 15m ATR below 0.04% of price
    adx_floor=18.0,      # below this the market is usually chop
)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def fetch(symbol: str, period: str, interval: str, retries: int = 3) -> pd.DataFrame:
    last_err = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(symbol).history(period=period, interval=interval,
                                           auto_adjust=False)
            if df is not None and not df.empty:
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                return df.tz_convert(IST).sort_index()
        except Exception as exc:                       # noqa: BLE001
            last_err = exc
        _time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not fetch {symbol} {interval}: {last_err}")


def drop_forming_bar(df: pd.DataFrame, minutes: int, now: pd.Timestamp) -> pd.DataFrame:
    """Yahoo labels a bar by its START.  Keep only bars that have fully closed."""
    if df.empty:
        return df
    last = df.index[-1]
    if last + pd.Timedelta(minutes=minutes) <= now:
        return df
    # the 15:15 bar is only 15 minutes long; it is complete after 15:30
    if last.time() >= pd.Timestamp("15:15").time() and now.time() >= pd.Timestamp("15:30").time():
        return df
    return df.iloc[:-1]


def attach_vwap(df: pd.DataFrame, proxy: pd.DataFrame | None) -> tuple[pd.DataFrame, str]:
    """Session VWAP that never silently pretends volume exists."""
    out = df.copy()
    out["TP"] = (out.High + out.Low + out.Close) / 3.0
    vol = out["Volume"].astype(float).fillna(0.0)
    mode = "exchange volume"

    if vol.sum() <= 0:                                  # index: always the case
        if proxy is not None and not proxy.empty:
            pv = proxy["Volume"].reindex(out.index).fillna(0.0).astype(float)
            if pv.sum() > 0:
                vol, mode = pv, "ETF volume proxy"
        if vol.sum() <= 0:
            vol, mode = pd.Series(1.0, index=out.index), "session mean (no volume)"

    day = out.index.normalize()
    weighted = (out["TP"] * vol)
    out["VWAP"] = weighted.groupby(day).cumsum() / vol.groupby(day).cumsum()
    return out, mode


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["Close"].shift(1)
    return pd.concat([df.High - df.Low,
                      (df.High - prev_close).abs(),
                      (df.Low - prev_close).abs()], axis=1).max(axis=1)


def wilder_atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, down = df.High.diff(), -df.Low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr = wilder_atr(df, n)
    pdi = 100 * plus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    mdi = 100 * minus_dm.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #
def previous_session(df: pd.DataFrame, today) -> pd.DataFrame:
    days = sorted({d.date() for d in df.index})
    prior = [d for d in days if d < today]
    if not prior:
        return pd.DataFrame()
    return df[[d.date() == prior[-1] for d in df.index]]


def floor_pivots(high: float, low: float, close: float) -> dict[str, float]:
    pp = (high + low + close) / 3.0
    return dict(
        PP=pp,
        R1=2 * pp - low, S1=2 * pp - high,
        R2=pp + (high - low), S2=pp - (high - low),
        R3=high + 2 * (pp - low), S3=low - 2 * (high - pp),
    )


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_votes(l15: pd.Series, l1h: pd.Series, tol: float) -> list[tuple[str, int, str]]:
    """Four equal votes. A price-vs-level vote abstains (0) inside the noise band."""
    def vs_level(label: str, close: float, level: float) -> tuple[str, int, str]:
        delta = close - level
        if abs(delta) <= tol:
            return (f"{label} (at level)", 0, f"{delta:+,.1f}")
        return (label, 1 if delta > 0 else -1, f"{delta:+,.1f}")

    return [
        vs_level("close > VWAP", l15.Close, l15.VWAP),
        vs_level("close > 20 SMA", l15.Close, l15.SMA_20),
        ("9 EMA > 21 EMA", 1 if l15.EMA_9 > l15.EMA_21 else -1,
         f"{l15.EMA_9 - l15.EMA_21:+,.1f}"),
        # 1h VWAP is NOT used as a vote: with one completed 1h bar in the session
        # it equals that bar's own (H+L+C)/3, so the "vote" degenerates into
        # "is the close above this bar's midpoint". The 20 SMA needs 20 hourly
        # bars (~3 sessions) and is valid from the first minute of the day.
        ("1h close > 1h 20 SMA", 1 if l1h.Close > l1h.SMA_20 else -1,
         f"{l1h.Close - l1h.SMA_20:+,.0f}"),
    ]


# --------------------------------------------------------------------------- #
# setup
# --------------------------------------------------------------------------- #
def build_setup(price: float, atr: float, session_low: float, session_high: float,
                pivots: dict[str, float], direction: int, cfg: dict) -> tuple[dict | None, str]:
    """Entry, stop and target all measured from the SAME price - the live one."""
    atr_stop = cfg["atr_mult"] * atr
    note = ""
    if direction > 0:
        structural = session_low - 0.1 * atr
        dist = price - structural
        # use the structure only when it sits a sane distance away
        if atr_stop <= dist <= 2.5 * atr:
            stop = structural
        else:
            stop = price - atr_stop
            if dist > 2.5 * atr:
                note = (f"session low {session_low:,.0f} is {dist / atr:.1f} ATR away - "
                        f"a stop under it is too wide for a 15m trade")
        risk = price - stop
        candidates = sorted(t for k, t in pivots.items() if k.startswith("R") and t > price)
    else:
        structural = session_high + 0.1 * atr
        dist = structural - price
        if atr_stop <= dist <= 2.5 * atr:
            stop = structural
        else:
            stop = price + atr_stop
            if dist > 2.5 * atr:
                note = (f"session high {session_high:,.0f} is {dist / atr:.1f} ATR away - "
                        f"a stop above it is too wide for a 15m trade")
        risk = stop - price
        candidates = sorted((t for k, t in pivots.items() if k.startswith("S") and t < price),
                            reverse=True)

    if risk <= 0:
        return None, "stop distance is zero"

    cost = cfg["cost_points"]
    target, source = None, "pivot"
    for cand in candidates:
        if (abs(cand - price) - cost) / (risk + cost) >= cfg["min_rr"]:
            target, source = cand, "pivot"
            break
    if target is None:
        target = price + 2 * risk if direction > 0 else price - 2 * risk
        source = "fallback 2R (no pivot gave the required R:R)"

    reward = abs(target - price)
    return dict(
        direction=direction, entry=price, stop=stop, target=target, target_source=source,
        risk=risk, reward=reward,
        gross_rr=reward / risk,
        net_rr=(reward - cost) / (risk + cost),
        cost=cost, note=note,
    ), note


def size_position(setup: dict, cfg: dict, lot: int) -> tuple[int, float, float, float]:
    budget = cfg["capital"] * cfg["risk_pct"] / 100.0
    per_lot = setup["risk"] * lot
    lots = int(budget // per_lot) if per_lot > 0 else 0
    needed = per_lot / (cfg["risk_pct"] / 100.0)
    return lots, lots * per_lot, per_lot, needed


# --------------------------------------------------------------------------- #
# evaluation - the shared engine, no printing
# --------------------------------------------------------------------------- #
@dataclass
class Evaluation:
    inst: Instrument
    now: pd.Timestamp
    n_bars: int
    last_bar: pd.Timestamp
    price_age: float
    vwap_mode: str

    price: float
    vwap: float
    sma20: float
    atr: float
    adx: float
    bb_up: float
    bb_dn: float

    h1_close: float
    h1_sma20: float
    h1_vwap: float
    n_1h_today: int

    prev_high: float
    prev_low: float
    prev_close: float
    pivots: dict

    votes: list = field(default_factory=list)
    net: int = 0
    cast: int = 0
    direction: int = 0
    notes: list = field(default_factory=list)
    blocks: list = field(default_factory=list)

    setup: dict | None = None
    lots: int = 0
    risk_rs: float = 0.0
    per_lot: float = 0.0
    needed: float = 0.0
    first_level: float | None = None
    first_key: str = ""
    first_pts: float = 0.0
    target_key: str = ""
    risk_pct: float = 0.0

    chart: pd.DataFrame | None = None


def evaluate(inst: Instrument, cfg: dict, force: bool = False,
             now: pd.Timestamp | None = None,
             fetch_fn=None) -> Evaluation:
    """Run the whole checklist for one instrument and return a plain result."""
    fetch_fn = fetch_fn or fetch
    now = now or pd.Timestamp.now(tz=IST)

    d15 = fetch_fn(inst.symbol, "5d", "15m")
    d1h = fetch_fn(inst.symbol, "10d", "1h")
    proxy = None
    try:
        proxy = fetch_fn(inst.proxy, "5d", "15m")
    except RuntimeError:
        pass

    d15 = drop_forming_bar(d15, 15, now)
    d1h = drop_forming_bar(d1h, 60, now)
    if len(d15) < 40 or len(d1h) < 25:
        raise RuntimeError("not enough history after dropping the forming bar")

    d15, vwap_mode = attach_vwap(d15, proxy)
    d1h, _ = attach_vwap(d1h, proxy)

    for df, ema_fast, ema_slow in ((d15, 9, 21), (d1h, 9, 21)):
        df["SMA_20"] = df["Close"].rolling(20).mean()
        df["SD_20"] = df["Close"].rolling(20).std()
        df["BB_UP"] = df["SMA_20"] + 2 * df["SD_20"]
        df["BB_DN"] = df["SMA_20"] - 2 * df["SD_20"]
        df[f"EMA_{ema_fast}"] = df["Close"].ewm(span=ema_fast, adjust=False).mean()
        df[f"EMA_{ema_slow}"] = df["Close"].ewm(span=ema_slow, adjust=False).mean()
        df["ATR"] = wilder_atr(df)
        df["ADX"] = adx(df)

    d15 = d15.dropna(subset=["SMA_20", "ATR", "ADX"])
    d1h = d1h.dropna(subset=["SMA_20", "VWAP"])
    if len(d15) < 5 or len(d1h) < 2:
        raise RuntimeError("indicators still warming up; try again later")

    l15, p15 = d15.iloc[-1], d15.iloc[-2]
    l1h = d1h.iloc[-1]

    prev = previous_session(d15, now.date())
    if prev.empty:
        raise RuntimeError("no completed prior session in the window")
    pivots = floor_pivots(prev.High.max(), prev.Low.min(), prev.Close.iloc[-1])

    price_age = (now - (d15.index[-1] + pd.Timedelta(minutes=15))).total_seconds() / 60
    n_1h_today = sum(1 for b in d1h.index if b.date() == now.date())

    ev = Evaluation(
        inst=inst, now=now, n_bars=len(d15), last_bar=d15.index[-1],
        price_age=price_age, vwap_mode=vwap_mode,
        price=float(l15.Close), vwap=float(l15.VWAP), sma20=float(l15.SMA_20),
        atr=float(l15.ATR), adx=float(l15.ADX), bb_up=float(l15.BB_UP),
        bb_dn=float(l15.BB_DN),
        h1_close=float(l1h.Close), h1_sma20=float(l1h.SMA_20),
        h1_vwap=float(l1h.VWAP), n_1h_today=n_1h_today,
        prev_high=float(prev.High.max()), prev_low=float(prev.Low.min()),
        prev_close=float(prev.Close.iloc[-1]), pivots=pivots,
    )

    # --- votes ------------------------------------------------------------- #
    ev.risk_pct = cfg["risk_pct"]
    ev.votes = score_votes(l15, l1h, tol=0.1 * l15.ATR)
    ev.net = sum(v for _, v, _ in ev.votes)
    ev.cast = sum(1 for _, v, _ in ev.votes if v != 0)
    ev.direction = 1 if ev.net >= 2 else -1 if ev.net <= -2 else 0

    # --- gates ------------------------------------------------------------- #
    if not force:
        if now.time() < pd.Timestamp(f"{NO_ENTRY_BEFORE[0]:02d}:{NO_ENTRY_BEFORE[1]:02d}").time():
            ev.blocks.append("before 09:45 - opening auction noise")
        if now.time() > pd.Timestamp(f"{NO_ENTRY_AFTER[0]:02d}:{NO_ENTRY_AFTER[1]:02d}").time():
            ev.blocks.append("after 14:45 - no time left for the target")
    if ev.direction == 0:
        ev.blocks.append(f"no majority - net {ev.net:+d} across {ev.cast} vote(s) cast")
    if l15.ADX < cfg["adx_floor"]:
        ev.blocks.append(f"ADX {l15.ADX:.1f} < {cfg['adx_floor']} - ranging/chop")
    if l15.ATR / l15.Close * 100 < cfg["atr_floor_pct"]:
        ev.blocks.append(f"ATR only {l15.ATR / l15.Close * 100:.2f}% of price - dead session")
    if abs(ev.net) == 2 and ev.cast == 4 and ev.direction != 0:
        ev.blocks.append("split 3-1 vote - needs a confirmed trigger")

    if np.sign(p15.EMA_9 - p15.EMA_21) != np.sign(l15.EMA_9 - l15.EMA_21):
        ev.notes.append("9/21 EMA cross on this bar (unconfirmed)")
    if l15.Close > l15.BB_UP or l15.Close < l15.BB_DN:
        ev.notes.append("price outside the 2-sigma band - a breakout entry here is chasing")

    # --- setup ------------------------------------------------------------- #
    if ev.direction != 0:
        today = [b for b in d15.index if b.date() == now.date()]
        setup, note = build_setup(l15.Close, l15.ATR, d15.loc[today, "Low"].min(),
                                  d15.loc[today, "High"].max(), pivots, ev.direction, cfg)
        if note:
            ev.notes.append(note)
        ev.setup = setup

        if setup is None:
            ev.blocks.append("stop distance is zero - bad structure")
        elif setup["net_rr"] < cfg["min_rr"]:
            ev.blocks.append(f"net R:R {setup['net_rr']:.2f} < {cfg['min_rr']} after costs")
        else:
            side = "R" if ev.direction > 0 else "S"
            ahead = [t for k, t in pivots.items()
                     if k.startswith(side)
                     and (t > l15.Close if ev.direction > 0 else t < l15.Close)]
            if ahead:
                near = min(ahead, key=lambda t: abs(t - l15.Close))
                if abs(near - l15.Close) < cfg["min_room_atr"] * l15.ATR:
                    ev.blocks.append(f"price is sitting on {near:,.2f} - only "
                                     f"{abs(near - l15.Close):.0f} pts "
                                     f"({abs(near - l15.Close) / l15.ATR:.1f} ATR) of room; "
                                     f"wait for a close beyond it")

        if setup is not None:
            (ev.lots, ev.risk_rs, ev.per_lot,
             ev.needed) = size_position(setup, cfg, inst.lot_size)
            ev.target_key = next((k for k, v in pivots.items()
                                  if abs(v - setup["target"]) < 0.01), "2R")
            ordered = sorted(
                (t for k, t in pivots.items()
                 if k.startswith("R" if setup["direction"] > 0 else "S")
                 and (t > setup["entry"] if setup["direction"] > 0 else t < setup["entry"])),
                reverse=setup["direction"] < 0)
            if ordered:
                ev.first_level = ordered[0]
                ev.first_pts = abs(ordered[0] - setup["entry"])
                ev.first_key = next((k for k, v in pivots.items()
                                     if abs(v - ordered[0]) < 0.01), "level")

    today_mask = [b.date() == now.date() for b in d15.index]
    if any(today_mask):
        ev.chart = d15.loc[today_mask, ["Close", "VWAP", "SMA_20"]].copy()

    return ev


def setup_rows(ev: Evaluation) -> list[tuple[str, str, str]]:
    """The trade table as (item, value, detail) - shared by CLI and browser."""
    s = ev.setup
    rows = [
        ("Entry", f"{s['entry']:,.2f}", "market / on trigger"),
        ("Stop loss", f"{s['stop']:,.2f}",
         f"{s['risk']:.1f} pts ({s['risk'] / s['entry'] * 100:.2f}%)"),
        (f"Target {ev.target_key}", f"{s['target']:,.2f}",
         f"{s['reward'] / ev.atr:.1f} ATR"
         + (" - needs a trend day" if s["reward"] / ev.atr > 3 else "")),
        ("R:R", f"1:{s['gross_rr']:.2f}",
         f"1:{s['net_rr']:.2f} net of {s['cost']:.0f} pts"),
    ]
    if ev.first_level is not None and abs(ev.first_level - s["target"]) > 0.01:
        rows.append((f"First {ev.first_key}", f"{ev.first_level:,.2f}",
                     f"{ev.first_pts:.0f} pts  ->  1:{ev.first_pts / s['risk']:.2f}"))
        if s["reward"] > ev.first_pts + 1e-9:
            blend = 0.5 * ev.first_pts + 0.5 * s["reward"]
            rows.append(("Scale-out", "50 / 50",
                         f"blended 1:{blend / s['risk']:.2f}  "
                         f"/  1:{(blend - s['cost']) / (s['risk'] + s['cost']):.2f} net"))
    rows.append(("Size", f"{ev.lots} lot(s)",
                 f"Rs {ev.per_lot:,.0f}/lot; need Rs {ev.needed / 1e5:,.1f}L at "
                 f"{ev.risk_pct:g}%"
                 if ev.lots == 0 else f"Rs {ev.risk_rs:,.0f} at {ev.risk_pct:g}% risk"))
    return rows


# --------------------------------------------------------------------------- #
# terminal front end
# --------------------------------------------------------------------------- #
def render_table(headers: tuple, rows: list, widths: tuple) -> list[str]:
    bar = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    head = "| " + " | ".join(f"{h:<{w}}" for h, w in zip(headers, widths)) + " |"
    body = ["| " + " | ".join(f"{str(c):<{w}}" for c, w in zip(r, widths)) + " |"
            for r in rows]
    return [bar, head, bar.replace("-", "="), *body, bar]


def render_cli(ev: Evaluation, cfg: dict) -> None:
    print()
    print(colour("=" * 74, DIM))
    print(" " + colour(f"{ev.inst.name:<11}{ev.inst.symbol:<10}{ev.now:%d-%b-%Y  %H:%M} IST",
                       BOLD))
    print(" " + colour(f"{ev.n_bars} bars | last closed {ev.last_bar:%H:%M} "
                       f"({ev.price_age:.0f}m old) | VWAP: {ev.vwap_mode}", DIM))
    print(colour("=" * 74, DIM))
    print(f" Price    {ev.price:>10,.2f}   VWAP {ev.vwap:>10,.2f}"
          f"   20SMA {ev.sma20:>10,.2f}")
    print(f" ATR      {ev.atr:>10,.1f}   ADX  {ev.adx:>10,.1f}")
    vwap_txt = (f"1h VWAP {ev.h1_vwap:,.2f}" if ev.n_1h_today >= 2 else "1h VWAP n/a")
    print(f" 1h       {ev.h1_close:>10,.2f}   vs 20SMA {ev.h1_sma20:>10,.2f}"
          f" ({ev.price / ev.h1_sma20 - 1:+.1%})   {vwap_txt}")
    print(f" Prev day  H {ev.prev_high:,.2f}  L {ev.prev_low:,.2f}  C {ev.prev_close:,.2f}")
    print(" Levels  " + " ".join(f"{k} {ev.pivots[k]:,.0f}"
                                 for k in ("S3", "S2", "S1", "PP", "R1", "R2", "R3")))

    print()
    for label, vote, detail in ev.votes:
        tag = "BULL" if vote > 0 else "BEAR" if vote < 0 else "FLAT"
        hue = GREEN if vote > 0 else RED if vote < 0 else YELLOW
        print(f"   {colour(f'{tag:<4}', hue)}  {label:<24}{detail:>9}")

    if ev.notes:
        print()
        for n in ev.notes:
            print("   " + colour("! ", YELLOW) + colour(n, DIM))

    print()
    if ev.blocks:
        print(" " + colour("VERDICT   NO TRADE", "1;" + YELLOW))
        for b in ev.blocks:
            print("   " + colour("- " + b, DIM))
        if ev.setup:
            print("\n   " + colour("reference only   ", DIM)
                  + f"entry {ev.setup['entry']:,.2f}   stop {ev.setup['stop']:,.2f}   "
                    f"target {ev.setup['target']:,.2f}   1:{ev.setup['gross_rr']:.2f}")
        return

    side = "LONG" if ev.setup["direction"] > 0 else "SHORT"
    hue = GREEN if ev.setup["direction"] > 0 else RED
    print(" " + colour(f"VERDICT   {side}", "1;" + hue)
          + colour(f"   {abs(ev.net)}/{ev.cast} votes", DIM))
    print()
    for line in render_table(("Item", "Value", "Detail"), setup_rows(ev), (13, 11, 40)):
        print(" " + line)
    print()
    print("   " + colour(ev.inst.note, DIM))
    print("   " + colour(f"square off by {SQUARE_OFF[0]:02d}:{SQUARE_OFF[1]:02d} IST "
                          f"regardless", DIM))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NIFTY / BANKNIFTY intraday checklist")
    p.add_argument("--capital", type=float, default=DEFAULTS["capital"])
    p.add_argument("--risk-pct", type=float, default=DEFAULTS["risk_pct"])
    p.add_argument("--atr-mult", type=float, default=DEFAULTS["atr_mult"])
    p.add_argument("--min-rr", type=float, default=DEFAULTS["min_rr"])
    p.add_argument("--cost-points", type=float, default=DEFAULTS["cost_points"])
    p.add_argument("--force", action="store_true", help="ignore time-of-day gates")
    p.add_argument("--color", choices=("auto", "always", "never"), default="auto",
                   help="ANSI colour in the terminal output")
    args = p.parse_args(argv)

    global COLOUR
    COLOUR = (args.color == "always") or (
        args.color == "auto" and sys.stdout.isatty() and os.environ.get("NO_COLOR") is None)

    cfg = dict(DEFAULTS, capital=args.capital, risk_pct=args.risk_pct,
               atr_mult=args.atr_mult, min_rr=args.min_rr, cost_points=args.cost_points)

    rc = 0
    for inst in INSTRUMENTS:
        try:
            render_cli(evaluate(inst, cfg, force=args.force), cfg)
        except Exception as exc:                        # noqa: BLE001
            print(f"\n{inst.name}: analysis failed - {exc}")
            rc = 1
    print()
    return rc


# --------------------------------------------------------------------------- #
# browser front end
# --------------------------------------------------------------------------- #
_CACHED_FETCH = None


def _cached_fetch():
    """Cache Yahoo calls for 60s inside Streamlit; plain fetch elsewhere."""
    global _CACHED_FETCH
    if _CACHED_FETCH is None:
        _CACHED_FETCH = fetch
        try:
            import streamlit as st
            _CACHED_FETCH = st.cache_data(ttl=60, show_spinner=False)(fetch)
        except Exception:                               # noqa: BLE001
            pass
    return _CACHED_FETCH


def _vote_rows_html(votes) -> str:
    hues = {"BULL": "#16a34a", "BEAR": "#dc2626", "FLAT": "#d97706"}
    rows = []
    for label, vote, detail in votes:
        tag = "BULL" if vote > 0 else "BEAR" if vote < 0 else "FLAT"
        rows.append(
            f"<tr>"
            f"<td style='color:{hues[tag]};font-weight:700;width:70px'>{tag}</td>"
            f"<td>{label}</td>"
            f"<td style='text-align:right;font-variant-numeric:tabular-nums;"
            f"color:#6b7280'>{detail}</td>"
            f"</tr>")
    return ("<table style='width:100%;border-collapse:collapse;font-size:0.95rem'>"
            + "".join(rows) + "</table>")


def render() -> None:
    """Draw the whole page. Called by app.py, or by `streamlit run` on this file."""
    import streamlit as st

    with st.sidebar:
        st.header("Settings")
        capital = st.number_input("Capital (Rs)", min_value=50_000, max_value=100_000_000,
                                  value=int(DEFAULTS["capital"]), step=50_000)
        risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=5.0,
                                   value=float(DEFAULTS["risk_pct"]), step=0.1)
        atr_mult = st.slider("Stop = x ATR", 0.5, 4.0, float(DEFAULTS["atr_mult"]), 0.25)
        min_rr = st.slider("Min net R:R", 0.5, 4.0, float(DEFAULTS["min_rr"]), 0.1)
        cost_points = st.number_input("Round-trip cost (index pts)", min_value=0.0,
                                      max_value=100.0, value=float(DEFAULTS["cost_points"]),
                                      step=1.0)
        force = st.checkbox("Ignore time-of-day gates", value=False,
                            help="Study mode: scores outside 09:45-14:45")
        picks = st.multiselect("Indices", [i.name for i in INSTRUMENTS],
                               default=[i.name for i in INSTRUMENTS])
        if st.button("Refresh now", type="primary"):
            st.cache_data.clear()
            st.rerun()
        st.caption("Yahoo Finance, ~15 min delayed. 60s cache. Not investment advice.")

    cfg = dict(DEFAULTS, capital=float(capital), risk_pct=float(risk_pct),
               atr_mult=float(atr_mult), min_rr=float(min_rr),
               cost_points=float(cost_points))

    st.subheader("NIFTY / BANK NIFTY - Intraday Trend Checklist")
    st.markdown(
        "Four-vote confluence on 15m with a 1h regime filter, floor pivots, an ATR or "
        "structural stop, cost-aware R:R and lot-based position sizing. "
        "Yahoo data, ~15 min delayed - for planning, not for execution.")
    st.caption(f"{pd.Timestamp.now(tz=IST):%A %d %B %Y, %H:%M} IST")

    chosen = [i for i in INSTRUMENTS if i.name in picks] or list(INSTRUMENTS)

    for inst in chosen:
        try:
            ev = evaluate(inst, cfg, force=force, fetch_fn=_cached_fetch())
        except Exception as exc:                        # noqa: BLE001
            st.error(f"{inst.name}: analysis failed - {exc}")
            continue

        st.divider()
        st.subheader(f"{inst.name}   `{inst.symbol}`")

        c = st.columns(4)
        c[0].metric("Price", f"{ev.price:,.2f}")
        c[1].metric("VWAP", f"{ev.vwap:,.2f}", f"{ev.price - ev.vwap:+,.1f}")
        c[2].metric("ATR (14)", f"{ev.atr:,.1f}")
        c[3].metric("ADX (14)", f"{ev.adx:,.1f}")

        st.caption(f"{ev.n_bars} bars · last closed {ev.last_bar:%H:%M} "
                   f"({ev.price_age:.0f}m old) · VWAP source: {ev.vwap_mode} · "
                   f"lot {inst.lot_size} · prev day H {ev.prev_high:,.2f} "
                   f"L {ev.prev_low:,.2f} C {ev.prev_close:,.2f}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Signals**")
            st.markdown(_vote_rows_html(ev.votes), unsafe_allow_html=True)
        with right:
            st.markdown("**Levels**")
            lv = "  ·  ".join(f"{k} {ev.pivots[k]:,.0f}"
                              for k in ("S3", "S2", "S1", "PP", "R1", "R2", "R3"))
            st.markdown(f"<span style='color:#6b7280'>{lv}</span>",
                        unsafe_allow_html=True)
            if ev.chart is not None and len(ev.chart) > 2:
                st.line_chart(ev.chart, height=180)

        for n in ev.notes:
            st.caption(f"⚠ {n}")

        if ev.blocks:
            st.warning("**VERDICT — NO TRADE**\n\n"
                       + "\n".join(f"- {b}" for b in ev.blocks))
            if ev.setup:
                st.caption(f"reference only — entry {ev.setup['entry']:,.2f} · "
                           f"stop {ev.setup['stop']:,.2f} · "
                           f"target {ev.setup['target']:,.2f} · "
                           f"1:{ev.setup['gross_rr']:.2f}")
        else:
            side = "LONG" if ev.setup["direction"] > 0 else "SHORT"
            banner = (st.success if side == "LONG" else st.error)
            banner(f"**VERDICT — {side}**   ({abs(ev.net)}/{ev.cast} votes)")
            st.dataframe(pd.DataFrame(setup_rows(ev),
                                      columns=["Item", "Value", "Detail"]),
                         hide_index=True)
            st.caption(f"{inst.note} · square off by "
                       f"{SQUARE_OFF[0]:02d}:{SQUARE_OFF[1]:02d} IST regardless")


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def _running_in_streamlit() -> bool:
    if "streamlit.runtime.scriptrunner" in sys.modules:
        return True
    try:
        from streamlit.runtime import exists            # type: ignore
        return bool(exists())
    except Exception:                                   # noqa: BLE001
        return False


if __name__ == "__main__":
    if _running_in_streamlit():
        import streamlit as st
        st.set_page_config(page_title="NIFTY / BANKNIFTY checklist", layout="wide")
        render()
    else:
        sys.exit(main())
