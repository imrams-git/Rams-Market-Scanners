"""
US Quantitative Scanner  (rev 3 -- download reliability fixes)

Runs a multi-timeframe unmitigated-imbalance / squeeze scan over a US equity
universe, then produces a tiered summary (deterministic in pandas) and a
handoff packet you can paste into the chat for a top-5 buy recommendation.

USAGE
    pip install yfinance pandas numpy pytz   # streamlit optional
    python 2_US_matching.py
    streamlit run 2_US_matching.py

    At the end of a run the script writes output/AI_PACKET_<ts>.md. Attach or
    paste that file into the chat to get the top-5 buy recommendation. There is
    no API key and no external AI call in this script.

--------------------------------------------------------------------------
WHAT CHANGED FROM THE PREVIOUS VERSION
--------------------------------------------------------------------------
1.  SECURITY  The hard-coded API key is gone entirely (the Gemini call was
    removed in rev 8). If you never revoked the key that was in the original
    script, revoke it now -- it was stored in plaintext.

2.  '4Hr' TIME FRAME NOW WORKS.  yfinance has no "4h" interval (valid:
    1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). The old code sent "4h"
    to Yahoo, got an empty frame back and silently skipped the whole
    time frame. Now 1h bars are fetched and resampled to 4h.

3.  FUNDAMENTALS FIXED.  `yf.Ticker(sym, session=self.session)` referenced
    a `session` attribute that does not exist on the class, so every PE and
    Days2Earn was "N/A" -- which also disabled the earnings filter and one
    of the TopPick gates. Now fetched properly (and in parallel).

4.  BEARISH SIDE REACHABLE.  The early `return []` when price <= 50DMA made
    `is_steady_bear` permanently False, so ImbLow rows could only ever come
    from the squeeze bypass. The trend gate is now applied after both flags
    are computed.

5.  RVOL NO LONGER EXPLODES ON HOLIDAYS.  The time-of-day pro-rata assumed
    the last daily bar was *today's* partial bar. On a market holiday (or
    before Yahoo posts the new bar) it compared a full session's volume to
    ~8% of the average, producing 10-15x readings for every symbol. The
    pro-rata is now applied only when the last bar really is today.

6.  ROLLING AVERAGES EXCLUDE THE BAR BEING TESTED.  `Volume / mean(20)` with
    the tested bar inside the window caps a genuine 2x spike at ~1.8x.
    Baselines now use shift(1).

7.  PER-TIMEFRAME DISTANCE FLOOR.  A flat 1.5% floor fought the 5x-intraday-
    ATR cap on 30m bars, so 30m contributed almost nothing. Floors are now
    scaled per time frame.

8.  DETERMINISTIC TIERS.  The four tiers are computed in pandas and added as a
    `Tier` column, so the ranking is reproducible rather than a model's opinion.
    Same output formatting rules as before.

9.  SPEED + VISIBILITY.  Batch downloads run with a thread pool (was
    threads=False, ~2,160 serial requests), failed batches are retried and
    then *reported* instead of being silently skipped, daily indicators are
    computed once per symbol instead of once per symbol per time frame, the
    unbroken-level check is vectorised, and the bare `except:` clauses that
    also swallowed Ctrl+C are gone.

10. "POSSIBLY DELISTED" FLOOD FIXED.  Two causes, both addressed:
    a) yfinance keeps a sqlite cookie jar (cookies.db) in a user cache
       directory. If that folder is missing or read-only, every request dies
       with OperationalError('unable to open database file') and Yahoo then
       reports live tickers (MSFT, NVDA, ...) as "possibly delisted". The
       cache is now pinned to a writable "cache" folder we create next to the
       script (override with YF_CACHE_DIR).
    b) 8 concurrent threads x 4 time frames x 540 tickers was enough to get
       throttled. Concurrency defaults to 4 (set YF_WORKERS=1 for fully serial,
       which is now viable because the request count halved), gaps between
       batches are jittered, failed batches back off (2s, 4s), and -- biggest
       win -- each batch now
       downloads 30m ONCE and derives 1h and 4h locally, while the 1d scan
       reuses the 90d frame already in memory. That is ~2 requests per symbol
       instead of ~4, roughly 1,070 instead of 2,140.
    yfinance's per-ticker ERROR logging is silenced (set YF_VERBOSE=1 to get it
    back); a compact coverage line reports exactly which symbols came back empty.

11. RATE LIMITS SURVIVED, NOT FOUGHT.  Yahoo throttles per IP, and a cold scan
    is ~1,600 requests, so a run can die halfway through and look like a
    filtering problem. Now:
    a) Every downloaded bar is cached on disk under cache/bars/<interval>/ and
       reused while still fresh (long TTL when the market is closed, because
       the bars cannot change). A second run on the same session is free.
    b) YF_MAX_REQUESTS caps requests per run (0 = unlimited). When the cap is
       hit the run finishes with what it has and tells you to run again --
       the cache makes the next run continue rather than restart.
    c) Three consecutive empty batches abort the phase with a clear
       rate-limit message instead of grinding through hundreds of failures.
    d) A "Filter funnel" table shows, per time frame, how many symbols clear
       each stage, so "too strict?" is answerable from the output.

12. SERIAL BY DEFAULT (this was my mistake).  The original script used
    threads=False; I changed it to 4 concurrent downloads and that burst is
    what got the intraday phase refused while the daily phase still worked.
    Downloads are serial again -- `threads=False` is passed explicitly because
    an integer is interpreted differently across yfinance releases -- and the
    curl-level retry (yf.set_config(retries=2)) is gone, since it re-sends
    requests with no backoff exactly when Yahoo is already refusing them.
    YF_WORKERS=4 raises concurrency if your IP tolerates it.

13. QUALITY TRIM.  The raw scan emits ~100 rows over ~60 symbols, with the same
    name repeated across time frames. Now, after the scan:
      - hard gates on every row: price >= MIN_PRICE ($5), 20d dollar volume >=
        MIN_DOLLAR_VOL ($20M), 0 < PE <= MAX_PE (60), no earnings inside
        MIN_DAYS_EARN (5) days, RVOL_D >= MIN_RVOL_D (0.5), and the target no
        further than MAX_TARGET_ATR (4) x that time frame's ATR;
      - one row per symbol per direction, chosen by most time frames agreeing,
        then nearest level, then oldest unbroken level;
      - a symbol that fires both ways is dropped as chopping;
      - two separate ranked lists (LONG / SHORT) instead of one mixed table;
      - every dropped row keeps its reason in the CSV "Status" column, and the
        terminal prints a count per reason, so nothing is hidden.
    All thresholds are overridable by the environment variables in brackets.

14. NO EXTERNAL AI CALL.  The Gemini integration is gone: no google-genai
    dependency, no API key, no network call to a model. Instead every run
    writes output/AI_PACKET_<timestamp>.md -- the ranked long and short tables
    plus sector, trend context, the exclusion counts and a legend. Attach or
    paste that file into the chat to get the top-5 buy recommendation.

15. RSI-H / RSI-D / RSI-W AND COILED Y/N.  Every row now carries RSI(14) on the
    hourly, daily and weekly charts next to the signal time frame's own RSI.
    The weekly series is resampled from the same 1y of daily data already
    fetched, so the three extra columns cost no additional requests; it covers
    roughly the last year (about 50 weekly bars). `Is_Coiled` is printed as
    Y/N rather than True/False.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import warnings
import os
import time
import random
import logging
import tempfile
import pickle
import pytz
from datetime import datetime
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

# Safely import streamlit without forcing browser execution context
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

warnings.filterwarnings("ignore")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

# Intervals Yahoo actually serves. Anything else must be resampled locally.
# Yahoo serves 4h even though it is not in yfinance's documented list, so it
# is requested natively and only resampled from 30m if Yahoo returns nothing.
YF_INTERVALS = {"1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "4h",
                "1d", "5d", "1wk", "1mo", "3mo"}
RESAMPLE_MAP = {}                    # requested -> fetched, then resampled
RESAMPLE_RULE = {"30m": "30min", "1Hr": "1h", "4Hr": "4h", "1d": "1D"}
EASTERN = pytz.timezone("US/Eastern")


def _market_open_now() -> bool:
    """True only during a US regular session (used for cache freshness)."""
    now = datetime.now(EASTERN)
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=5, second=0, microsecond=0)
    return open_t <= now <= close_t


class BarCache:
    """Per-symbol OHLCV on disk, so re-runs do not re-hit Yahoo.

    This is the difference between a scan you can iterate on and one that gets
    you rate-limited: a second run on the same session costs zero requests.
    Freshness is TTL-based, and the TTL is long when the market is closed
    because the bars cannot change until the next session.
    """

    def __init__(self, root: str):
        self.root = root

    def _path(self, key: str, symbol: str) -> str:
        return os.path.join(self.root, key, f"{symbol}.pkl")

    @staticmethod
    def _fresh(ts: float, interval: str) -> bool:
        age = time.time() - ts
        if interval == "1d":
            return age < (3600 if _market_open_now() else 12 * 3600)
        return age < (300 if _market_open_now() else 12 * 3600)

    def load(self, key: str, symbols: List[str], interval: str = None) -> dict:
        interval = interval or key.split("_")[0]
        out = {}
        for sym in symbols:
            try:
                with open(self._path(key, sym), "rb") as fh:
                    blob = pickle.load(fh)
                if self._fresh(blob["ts"], interval) and not blob["df"].empty:
                    out[sym] = blob["df"]
            except Exception:            # corrupt/foreign pickle: just refetch
                continue
        return out

    def store(self, key: str, frame) -> int:
        if frame is None or getattr(frame, "empty", True):
            return 0
        if not isinstance(frame.columns, pd.MultiIndex):
            return 0
        try:
            os.makedirs(os.path.join(self.root, key), exist_ok=True)
        except OSError:
            return 0
        written = 0
        for sym in frame.columns.get_level_values(0).unique():
            sub = frame[sym].dropna(how="all")
            if sub.empty:
                continue
            try:
                with open(self._path(key, sym), "wb") as fh:
                    pickle.dump({"ts": time.time(), "df": sub}, fh)
                written += 1
            except OSError:
                pass
        return written


def _configure_yfinance():
    """Point yfinance's cookie/tz cache at a writable folder.

    yfinance keeps a sqlite cookie jar (cookies.db) in a user cache directory.
    If that folder is missing or read-only, every request fails with
    OperationalError('unable to open database file') and Yahoo then reports
    perfectly live tickers as "possibly delisted". That single failure mode
    accounts for most of a scan where half the universe 'looks delisted'.
    """
    candidates = []
    if os.environ.get("YF_CACHE_DIR"):
        candidates.append(os.environ["YF_CACHE_DIR"])
    try:
        candidates.append(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "cache"))
    except NameError:
        pass
    candidates.append(os.path.join(os.path.expanduser("~"), "cache"))
    candidates.append(os.path.join(tempfile.gettempdir(), "cache"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            probe = os.path.join(path, ".probe")
            with open(probe, "w") as fh:
                fh.write("ok")
            os.remove(probe)
        except OSError:
            continue
        try:
            if hasattr(yf, "set_tz_cache_location"):
                yf.set_tz_cache_location(path)      # 0.2.4x and 1.x
            elif hasattr(yf, "set_cache_location"):
                yf.set_cache_location(path)
            return path
        except Exception:
            continue
    return None


YF_CACHE_DIR = _configure_yfinance()

# NOTE: deliberately no yf.set_config(retries=N) here. A curl-level retry has no
# backoff and multiplies requests precisely when Yahoo is already refusing them;
# get_batch_data() does its own paced retry instead.

# yfinance logs one ERROR line per failed ticker plus a "Failed downloads"
# block. We compute our own coverage summary instead, so keep it quiet unless
# YF_VERBOSE is set.
if not os.environ.get("YF_VERBOSE"):
    logging.getLogger("yfinance").setLevel(logging.CRITICAL)
    logging.getLogger("peewee").setLevel(logging.CRITICAL)


class VolumeAlertChecker:
    def __init__(
        self,
        length: int = 10,
        high_volume_threshold: float = 2.0,
        imbalance_threshold: float = 0.70,
        min_body_ratio: float = 0.60,
        min_distance_pct: float = 1.5,
        trend_lookback: int = 5,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier: float = 5.0,
        squeeze_length: int = 20,
        squeeze_multiplier: float = 1.5,
        min_distance_by_tf: dict = None,
        max_workers: int = None,
        fundamentals_workers: int = 4,
        base_intraday: str = "30m",
        daily_period: str = "1y",
        cache_dir: str = None,
        max_requests: int = None
    ):
        self.length = length
        self.high_volume_threshold = high_volume_threshold
        self.imbalance_threshold = imbalance_threshold
        self.min_body_ratio = min_body_ratio
        self.min_distance_pct = min_distance_pct
        self.trend_lookback = trend_lookback
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.timeframes = {"30m": "30m", "1Hr": "1h", "4Hr": "4h", "1d": "1d"}
        self.squeeze_length = squeeze_length
        self.squeeze_multiplier = squeeze_multiplier
        # A 1.5% floor is unreachable inside a 5x-ATR cap on 30m bars, so the
        # floor is scaled per time frame.
        self.min_distance_by_tf = min_distance_by_tf or {
            "30m": 0.6, "1Hr": 0.9, "4Hr": 1.2, "1d": 1.5
        }

        # ---- quality gates applied to the FINAL list (see build_lists) ----
        def _env_float(name, default):
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return float(default)

        self.min_price = _env_float("MIN_PRICE", 5.0)          # $ per share
        self.min_dollar_vol = _env_float("MIN_DOLLAR_VOL", 2.0e7)   # 20d $ volume
        self.max_pe = _env_float("MAX_PE", 60.0)
        self.min_rvol_d = _env_float("MIN_RVOL_D", 0.5)
        self.min_days_to_earnings = _env_float("MIN_DAYS_EARN", 5)
        self.max_target_atr = _env_float("MAX_TARGET_ATR", 4.0)
        # keep the handoff packet small enough to paste into a chat
        self.max_packet_rows = int(_env_float("MAX_PACKET_ROWS", 40))
        # SERIAL by default. Concurrency is what trips Yahoo's limiter: the
        # original script downloaded one ticker at a time and worked, while
        # 4-at-a-time bursts got the whole intraday phase refused. The request
        # count is already halved elsewhere, so serial is affordable.
        # Raise it with YF_WORKERS=4 only if your IP tolerates it.
        if max_workers is None:
            try:
                max_workers = int(os.environ.get("YF_WORKERS", 1))
            except ValueError:
                max_workers = 1
        self.max_workers = max(1, int(max_workers))
        self.fundamentals_workers = fundamentals_workers
        # One intraday download per batch feeds every intraday time frame,
        # which cuts Yahoo requests by ~3x versus fetching each separately.
        self.base_intraday = base_intraday
        # 90d left only ~12 valid 50DMA readings, so the slope was always NaN
        self.daily_period = daily_period
        self._daily_cache = {}
        self._failed = Counter()
        self._scanned = Counter()
        self._missing = {}
        self._last_error = None
        self._funnel = Counter()
        self._rsi_cache = {}
        self._hourly_frame = None
        self.last_run = {}
        # None = unknown, True = Yahoo serves 4h, False = derive it from 30m.
        # YF_NATIVE_4H=0 skips the extra 4h request (cuts a cold scan by a third).
        self._allow_native_4h = os.environ.get("YF_NATIVE_4H", "1") != "0"
        self.native_4h = None if self._allow_native_4h else False

        # On-disk bar cache: the same session is never fetched twice.
        if cache_dir is None:
            try:
                cache_dir = os.path.join(
                    os.path.dirname(os.path.abspath(__file__)), "cache", "bars")
            except NameError:
                cache_dir = os.path.join(tempfile.gettempdir(), "bars")
        try:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache = BarCache(cache_dir)
        except OSError:
            self.cache = None
        self.cache_dir = cache_dir

        # Yahoo rate-limits per IP. Cap requests per run so a scan degrades
        # gracefully instead of being blocked halfway through; the cache means
        # the next run picks up where this one stopped.
        if max_requests is None:
            try:
                max_requests = int(os.environ.get("YF_MAX_REQUESTS", 0))
            except ValueError:
                max_requests = 0
        self.max_requests = max(0, int(max_requests))
        self._requests_made = 0
        self._cache_hits = 0
        self._fail_streak = 0
        self._budget_stop = False

    # ------------------------------------------------------------------ utils
    def get_time_adjusted_rvol(self, current_vol, avg_vol, bar_date=None):
        """Relative volume, pro-rated by elapsed session time.

        Only pro-rates when `bar_date` is today's session; otherwise the last
        daily bar is a *complete* prior session and must be compared to a full
        average (this is what blew up to 10-15x on market holidays).
        """
        if avg_vol is None or not np.isfinite(avg_vol) or avg_vol <= 0:
            return 0.0
        raw = float(current_vol) / float(avg_vol)
        now_et = datetime.now(EASTERN)
        today = now_et.date()

        if bar_date is None or bar_date != today:
            return raw
        if now_et.weekday() >= 5:
            return raw

        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        if now_et < market_open or now_et >= market_close:
            # Today's bar cannot be partial yet (or the session is over).
            return raw

        elapsed_minutes = (now_et - market_open).total_seconds() / 60.0
        fraction = min(max(elapsed_minutes / 390.0, 1.0 / 390.0), 1.0)
        pro_rated_avg_vol = avg_vol * fraction
        return float(current_vol) / pro_rated_avg_vol if pro_rated_avg_vol > 0 else 0.0

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(100)

    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _resample_to(self, df: pd.DataFrame, rule: str) -> pd.DataFrame:
        """Aggregate intraday OHLCV to a coarser bar (e.g. 1h -> 4h)."""
        agg = {"Open": "first", "High": "max", "Low": "min",
               "Close": "last", "Volume": "sum"}

        def _one(sub: pd.DataFrame) -> pd.DataFrame:
            if not isinstance(sub.index, pd.DatetimeIndex):
                return sub
            cols = {k: v for k, v in agg.items() if k in sub.columns}
            try:
                out = sub.resample(rule, origin="start", label="left",
                                   closed="left").agg(cols)
            except (TypeError, ValueError):
                out = sub.resample(rule, label="left", closed="left").agg(cols)
            # empty bins get Volume=0 and NaN OHLC -- drop them, not just all-NaN
            if "Close" in out.columns:
                out = out[out["Close"].notna()]
            return out.dropna(how="all")

        if isinstance(df.columns, pd.MultiIndex):
            out = {}
            for sym in df.columns.get_level_values(0).unique():
                sub = df[sym].dropna(how="all")
                if sub.empty:
                    continue
                agg_df = _one(sub)
                if not agg_df.empty:
                    out[sym] = agg_df
            if not out:
                return pd.DataFrame()
            return pd.concat(out, axis=1)
        return _one(df)

    @staticmethod
    def _pause(base: float = 0.35):
        """Small randomised gap between batches (fixed sleeps look like a bot)."""
        time.sleep(base + random.uniform(0, base))

    @staticmethod
    def _symbols_with_data(frame, requested) -> set:
        """Which of the requested symbols actually came back with prices."""
        if frame is None or getattr(frame, "empty", True):
            return set()
        try:
            if isinstance(frame.columns, pd.MultiIndex):
                present = set(frame.columns.get_level_values(0).unique())
                out = set()
                for sym in requested:
                    if sym in present and frame[sym]["Close"].notna().any():
                        out.add(sym)
                return out
            return set(requested) if frame["Close"].notna().any() else set()
        except (KeyError, AttributeError):
            return set()

    def _budget_left(self) -> int:
        if not self.max_requests:
            return 10 ** 9
        return max(0, self.max_requests - self._requests_made)

    def get_batch_data(self, symbols: List[str], timeframe: str,
                       period: str = "60d", retries: int = 2):
        fetch_tf = RESAMPLE_MAP.get(timeframe, timeframe)
        if fetch_tf not in YF_INTERVALS:
            raise ValueError(
                f"interval '{timeframe}' is not served by Yahoo; add it to "
                f"RESAMPLE_MAP instead of passing it through")
        symbols = list(dict.fromkeys(symbols))

        # 1. serve whatever we already have on disk (zero requests)
        cache_key = f"{fetch_tf}_{period}"
        cached = self.cache.load(cache_key, symbols, fetch_tf) if self.cache else {}
        self._cache_hits += len(cached)
        missing = [s for s in symbols if s not in cached]

        # 2. fetch only what is missing, within the per-run request budget
        fetched = {}
        if missing:
            budget = self._budget_left()
            if budget <= 0:
                self._budget_stop = True
                missing = []
            else:
                missing = missing[:budget]

        if missing:
            # Do not retry into a wall: if the previous batch came back empty,
            # one attempt is enough before the caller's fail-streak aborts.
            if self._fail_streak > 0:
                retries = 0
            for attempt in range(retries + 1):
                try:
                    data = yf.download(
                        tickers=missing, period=period, interval=fetch_tf,
                        group_by="ticker",
                        # bool False = truly serial on every yfinance version;
                        # an int is interpreted differently across releases.
                        threads=(self.max_workers if self.max_workers > 1 else False),
                        progress=False
                    )
                except Exception as exc:           # not bare: Ctrl+C still works
                    # Keep the reason: a swallowed exception looks exactly like
                    # "Yahoo has no data" and is impossible to debug otherwise.
                    self._last_error = f"{type(exc).__name__}: {exc}"
                    data = pd.DataFrame()
                self._requests_made += len(missing)

                if data is not None and not data.empty:
                    have = self._symbols_with_data(data, missing)
                    if have:
                        if self.cache:
                            self.cache.store(cache_key, data)
                        for sym in have:
                            try:
                                fetched[sym] = data[sym].dropna(how="all")
                            except KeyError:
                                pass
                        break
                if attempt < retries:
                    # exponential backoff + jitter: 2s, 4s
                    time.sleep((2.0 * (2 ** attempt)) + random.uniform(0, 1.5))
            if not fetched and not self._last_error:
                self._last_error = (f"{fetch_tf}/{period}: Yahoo returned no rows for "
                                    f"{len(missing)} symbols (no exception raised)")

        merged = dict(cached)
        merged.update(fetched)
        if not merged:
            return pd.DataFrame()
        out = pd.concat(merged, axis=1)
        out.columns.names = ["Ticker", "Price"]
        if fetch_tf != timeframe:
            out = self._resample_to(out, timeframe)
        return out

    # ------------------------------------------------------------- indicators
    def is_coiled_squeeze(self, df_d: pd.DataFrame) -> bool:
        """Bollinger Bands inside Keltner Channels on the daily chart."""
        if df_d is None or len(df_d) < self.squeeze_length:
            return False
        close = df_d['Close']
        ma = close.rolling(window=self.squeeze_length).mean()
        std = close.rolling(window=self.squeeze_length).std()
        bb_upper = ma + (2 * std)
        bb_lower = ma - (2 * std)

        high_low = df_d['High'] - df_d['Low']
        high_cp = (df_d['High'] - close.shift()).abs()
        low_cp = (df_d['Low'] - close.shift()).abs()
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=self.squeeze_length).mean()

        kc_upper = ma + (self.squeeze_multiplier * atr)
        kc_lower = ma - (self.squeeze_multiplier * atr)

        try:
            current_squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and \
                              (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        except IndexError:
            return False
        return bool(current_squeeze)

    def _daily_context(self, daily_data: pd.DataFrame, symbol: str):
        """Daily features, computed once per symbol (not once per time frame)."""
        if symbol in self._daily_cache:
            return self._daily_cache[symbol]

        ctx = None
        try:
            if isinstance(daily_data.columns, pd.MultiIndex):
                if symbol not in daily_data.columns.levels[0]:
                    self._daily_cache[symbol] = None
                    return None
                df_d = daily_data[symbol].copy().dropna()
            else:
                df_d = daily_data.copy().dropna()

            if len(df_d) < 55:
                self._daily_cache[symbol] = None
                return None

            close = df_d['Close']
            volume = df_d['Volume']
            dma50 = float(close.rolling(window=50).mean().iloc[-1])
            # Baseline excludes the bar being measured -> no self-inclusion bias
            avg_vol = volume.shift(1).rolling(window=20).mean()

            last_ts = df_d.index[-1]
            last_date = last_ts.date() if hasattr(last_ts, "date") else last_ts

            def _ratio(i):
                base = avg_vol.iloc[i]
                if not np.isfinite(base) or base <= 0:
                    return 0.0
                return float(volume.iloc[i]) / float(base)

            dollar_vol = (close * volume).shift(1).rolling(window=20).mean()
            dma50_series = close.rolling(window=50).mean()
            prev_dma50 = float(dma50_series.iloc[-21]) if len(dma50_series) > 21 else float("nan")
            ctx = {
                "dma50": dma50,
                "pct_50dma": float((close.iloc[-1] / dma50 - 1.0) * 100.0)
                if np.isfinite(dma50) and dma50 else float("nan"),
                "slope_50dma": float((dma50 / prev_dma50 - 1.0) * 100.0)
                if np.isfinite(dma50) and np.isfinite(prev_dma50) and prev_dma50
                else float("nan"),
                "close": float(close.iloc[-1]),
                "adv20": float(dollar_vol.iloc[-1])
                if np.isfinite(dollar_vol.iloc[-1]) else float("nan"),
                "rvol_today": self.get_time_adjusted_rvol(
                    volume.iloc[-1], avg_vol.iloc[-1], last_date),
                "rvol_d_1": _ratio(-2),
                "rvol_d_2": _ratio(-3),
                "is_coiled": self.is_coiled_squeeze(df_d),
            }
        except Exception:
            ctx = None

        self._daily_cache[symbol] = ctx
        return ctx

    def _mtf_rsi(self, symbol, hourly_frame, daily_frame) -> dict:
        """RSI on the 1Hr, daily and weekly charts for one symbol.

        Weekly bars are resampled from the daily history we already hold, so
        this costs no extra requests. Memoised per symbol per run.
        """
        if symbol in self._rsi_cache:
            return self._rsi_cache[symbol]

        out = {"RSI_H": np.nan, "RSI_D": np.nan, "RSI_W": np.nan}
        need = self.rsi_period + 2

        def _last_rsi(frame):
            try:
                if frame is None or len(frame) < need:
                    return np.nan
                close = frame["Close"] if "Close" in frame.columns else frame
                close = close.dropna()
                if len(close) < need:
                    return np.nan
                value = float(self.calculate_rsi(close, self.rsi_period).iloc[-1])
                return value if np.isfinite(value) else np.nan
            except Exception:
                return np.nan

        def _slice(frame):
            if frame is None or getattr(frame, "empty", True):
                return None
            try:
                if isinstance(frame.columns, pd.MultiIndex):
                    if symbol not in frame.columns.levels[0]:
                        return None
                    return frame[symbol].dropna(how="all")
                return frame
            except Exception:
                return None

        out["RSI_H"] = _last_rsi(_slice(hourly_frame))

        daily = _slice(daily_frame)
        out["RSI_D"] = _last_rsi(daily)
        if daily is not None and "Close" in daily.columns:
            try:
                weekly = daily["Close"].resample("W-FRI").last().dropna()
                out["RSI_W"] = _last_rsi(weekly.to_frame("Close"))
            except (TypeError, ValueError):
                pass

        self._rsi_cache[symbol] = out
        return out

    # --------------------------------------------------------------- analysis
    def _rate_limit_note(self) -> str:
        return ("\n!! Three batches in a row returned nothing -- that is Yahoo "
                "rate-limiting this IP, not a bad symbol list.\n"
                "   Stop here, wait 15-60 minutes, and run again: everything already\n"
                "   downloaded is cached, so the next run only fetches what is missing.\n"
                "   Set YF_MAX_REQUESTS=400 to stay under the limit in one go.")

    def _count(self, tf_name, stage, n=1):
        self._funnel[f"{tf_name}|{stage}"] += n

    def analyze_symbol(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame,
                       symbol: str, n_bars: int, tf_name: str = None):
        if isinstance(intraday_data.columns, pd.MultiIndex):
            if symbol not in intraday_data.columns.levels[0]:
                return []
            df = intraday_data[symbol].copy().dropna()
        else:
            df = intraday_data.copy().dropna()

        if len(df) < 20:
            self._count(tf_name, "short")
            return []

        ctx = self._daily_context(daily_data, symbol)
        if ctx is None:
            self._count(tf_name, "no_daily")
            return []

        cmp_price = float(df['Close'].iloc[-1])
        dma50 = ctx["dma50"]
        is_coiled = ctx["is_coiled"]

        has_trend = np.isfinite(dma50)
        vol_ok = (ctx["rvol_d_1"] >= 1.0) and (ctx["rvol_d_2"] >= 1.0)
        # Both flags are computed before any early return, so the bearish
        # branch is actually reachable.
        is_steady_bull = has_trend and (cmp_price >= dma50) and vol_ok
        is_steady_bear = has_trend and (cmp_price <= dma50) and vol_ok

        if not (is_steady_bull or is_steady_bear or is_coiled):
            self._count(tf_name, "daily_gate")
            return []
        self._count(tf_name, "passed_daily")

        df = df.copy()
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['ATR'] = self.calculate_atr(df, self.atr_period)
        df['AvgVol'] = df['Volume'].shift(1).rolling(self.length).mean()
        df['RVOL_H'] = df['Volume'] / df['AvgVol']

        current_rsi = float(df['RSI'].iloc[-1])
        current_atr = float(df['ATR'].iloc[-1])
        current_rvol_h = float(df['RVOL_H'].iloc[-1])
        if not np.isfinite(current_atr) or current_atr <= 0:
            return []

        # Vectorised "was this level breached afterwards?"
        # future_max[t] = max High strictly after bar t; future_min[t] = min Low.
        future_max = df['High'][::-1].cummax()[::-1].shift(-1)
        future_min = df['Low'][::-1].cummin()[::-1].shift(-1)

        min_dist = self.min_distance_by_tf.get(tf_name, self.min_distance_pct)

        recent_bars = df.iloc[-n_bars:]
        high_vol_indices = recent_bars[recent_bars["RVOL_H"] > self.high_volume_threshold].index
        self._count(tf_name, "bars", len(high_vol_indices))

        valid_levels = []
        for i in reversed(high_vol_indices):
            row = df.loc[i]
            rng = row["High"] - row["Low"]
            if not np.isfinite(rng) or rng == 0:
                self._count(tf_name, "zero_range")
                continue

            color = "ImbLow" if row["Close"] >= row["Open"] else "ImbHigh"

            # Squeeze names bypass the strict trend gate
            if not is_coiled:
                if color == "ImbHigh" and not is_steady_bull:
                    self._count(tf_name, "trend")
                    continue
                if color == "ImbLow" and not is_steady_bear:
                    self._count(tf_name, "trend")
                    continue

            # Momentum safety check
            if color == "ImbHigh" and not (55 <= current_rsi <= 68):
                self._count(tf_name, "rsi")
                continue
            if color == "ImbLow" and not (25 <= current_rsi <= 45):
                self._count(tf_name, "rsi")
                continue

            buy_p = (row["Close"] - row["Low"]) / rng
            if color == "ImbLow" and buy_p < self.imbalance_threshold:
                self._count(tf_name, "imbalance")
                continue
            if color == "ImbHigh" and (1 - buy_p) < self.imbalance_threshold:
                self._count(tf_name, "imbalance")
                continue

            # RULE 1: unbroken level
            level = row["High"] if color == "ImbHigh" else row["Low"]
            if color == "ImbHigh" and (future_max.loc[i] > level):
                self._count(tf_name, "broken")
                continue
            if color == "ImbLow" and (future_min.loc[i] < level):
                self._count(tf_name, "broken")
                continue

            dist = abs(level - cmp_price)
            if dist > (current_atr * self.atr_multiplier):
                self._count(tf_name, "atr_cap")
                continue
            pct_diff = (dist / cmp_price) * 100
            if pct_diff < min_dist:
                self._count(tf_name, "too_close")
                continue

            self._count(tf_name, "kept")

            bars_ago = len(df) - df.index.get_loc(i) - 1
            atr_pct = (current_atr / cmp_price) * 100.0
            mtf = self._mtf_rsi(symbol, self._hourly_frame, daily_data)
            valid_levels.append([
                symbol, color, round(cmp_price, 2), round(float(level), 2),
                round(pct_diff, 2), round(atr_pct, 2), int(bars_ago),
                round(current_rsi, 2),
                round(mtf["RSI_H"], 2), round(mtf["RSI_D"], 2), round(mtf["RSI_W"], 2),
                round(current_rvol_h, 2), round(ctx["rvol_today"], 2), is_coiled
            ])
        return valid_levels

    # ----------------------------------------------------------- fundamentals
    def _fundamentals_one(self, sym: str) -> dict:
        pe_str, days_to_earn, sector = "N/A", "N/A", "N/A"
        for attempt in range(2):
            try:
                ticker = yf.Ticker(sym)          # no bogus `session=` kwarg
                info = ticker.info or {}
                pe = info.get('trailingPE') or info.get('forwardPE')
                if pe and pe > 0:
                    pe_str = f"{pe:.1f}"
                sector = info.get('sector') or "N/A"

                calendar = ticker.calendar
                if isinstance(calendar, dict) and calendar.get('Earnings Date'):
                    dates = calendar['Earnings Date']
                    if len(dates) > 0:
                        next_date = dates[0].date() if hasattr(dates[0], 'date') else dates[0]
                        days_to_earn = (next_date - datetime.now(EASTERN).date()).days
                return {"PE": pe_str, "Days2Earn": days_to_earn, "Sector": sector}
            except Exception:
                time.sleep(0.5 * (attempt + 1))
        return {"PE": pe_str, "Days2Earn": days_to_earn, "Sector": sector}

    def add_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        unique_symbols = list(df['Symbol'].unique())
        fund_data = {}

        if unique_symbols:
            print(f"\nFetching Fundamental Data for {len(unique_symbols)} matched symbols...")
            with ThreadPoolExecutor(max_workers=self.fundamentals_workers) as pool:
                results = list(pool.map(self._fundamentals_one, unique_symbols))
            fund_data = dict(zip(unique_symbols, results))
            resolved = sum(1 for v in fund_data.values() if v.get("PE") != "N/A")
            print(f"  PE resolved for {resolved}/{len(unique_symbols)} symbols")

        df = df.copy()
        df['PE'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('PE', 'N/A'))
        df['Days2Earn'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('Days2Earn', 'N/A'))
        df['Sector'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('Sector', 'N/A'))
        return df

    # ---------------------------------------------------------------- tiers
    @staticmethod
    def _yn(value) -> str:
        try:
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                return "-"
            return "Y" if bool(value) else "N"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _fmt(value, digits: int = 1) -> str:
        try:
            out = float(value)
            return f"{out:.{digits}f}" if np.isfinite(out) else "-"
        except (TypeError, ValueError):
            return "-"

    @staticmethod
    def _num(value, default=np.nan):
        try:
            out = float(value)
            return out if np.isfinite(out) else default
        except (TypeError, ValueError):
            return default

    def assign_tiers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deterministic replacement for the tier rules that were only in the
        LLM prompt. 1 = highest conviction, 4 = rejected / chopping."""
        if df.empty:
            return df.assign(Tier=pd.Series(dtype="int64"))

        d = df.copy()
        if "Days2Earn" in d.columns:
            d["_earn_safe"] = d["Days2Earn"].map(
                lambda v: True if str(v) == "N/A"
                else (self._num(v) > 15 or self._num(v) < -2)
            )
        else:                      # tiers can be built before fundamentals land
            d["_earn_safe"] = True

        agg = d.groupby("Symbol").agg(
            n_tf=("TF", "nunique"),
            has_low=("Imb", lambda s: (s == "ImbLow").any()),
            has_high=("Imb", lambda s: (s == "ImbHigh").any()),
            max_ago=("Ago", "max"),
            max_rvol_d=("RVOL_D", "max"),
            any_coiled=("Is_Coiled", "any"),
            earn_safe=("_earn_safe", "all"),
        )

        tier = pd.Series(4, index=agg.index, dtype="int64")

        t2 = (agg["max_ago"] > 5) & (agg["max_rvol_d"] > 0.8) & \
             agg["earn_safe"] & (~agg["has_low"])
        tier[t2] = 2

        t1 = t2 & (agg["max_ago"] > 20) & (agg["n_tf"] >= 2) & \
             (agg["any_coiled"] | (agg["max_rvol_d"] > 1.0))
        tier[t1] = 1

        t3 = (tier == 4) & (agg["max_rvol_d"] > 3.0) & agg["earn_safe"] & (~agg["has_low"])
        tier[t3] = 3

        # Mixed signals in the same name = chopping -> reject
        tier[agg["has_low"] & agg["has_high"]] = 4

        d["Tier"] = d["Symbol"].map(tier).astype("int64")
        return d.drop(columns=["_earn_safe"])

    # ------------------------------------------------------------------ run
    def run(self, symbols: List[str], n_bars: int = 100):
        rows = []
        batch_size = 40
        self._daily_cache = {}
        self._failed = Counter()
        self._scanned = Counter()
        self._missing = {tf: [] for tf in self.timeframes}
        self._last_error = None
        self._funnel = Counter()
        self.native_4h = None if self._allow_native_4h else False
        self._requests_made = 0
        self._cache_hits = 0
        self._fail_streak = 0
        self._budget_stop = False

        try:
            import yfinance as _yf
            print(f"yfinance {_yf.__version__} | cache {self.cache_dir}")
        except Exception:
            print(f"cache {self.cache_dir}")

        if YF_CACHE_DIR:
            print(f"yfinance cache: {YF_CACHE_DIR}")
        else:
            print("WARNING: no writable cache folder found; yfinance will keep "
                  "failing with 'unable to open database file'. Set YF_CACHE_DIR.")

        print(f"workers={self.max_workers} | request budget="
              f"{self.max_requests or 'unlimited'} | native 4h="
              f"{self._allow_native_4h} | cache={self.cache_dir}")
        print("Fetching Daily Data and filtering delisted tickers...")
        fixed_symbols = list(dict.fromkeys(sym.replace(".", "-") for sym in symbols))
        self._fail_streak = 0

        daily_data_chunks = []
        have_daily = set()

        for i in range(0, len(fixed_symbols), batch_size):
            batch = fixed_symbols[i: i + batch_size]
            batch_df = self.get_batch_data(batch, "1d", self.daily_period)

            if batch_df is None or batch_df.empty:
                self._failed["1d"] += 1
                if self._budget_stop:
                    break
                self._fail_streak += 1
                if self._fail_streak >= 3:
                    print(self._rate_limit_note())
                    break
                continue
            self._fail_streak = 0

            daily_data_chunks.append(batch_df)
            have_daily |= self._symbols_with_data(batch_df, batch)
            self._pause()

        valid_symbols = [s for s in fixed_symbols if s in have_daily]
        missing_daily = [s for s in fixed_symbols if s not in have_daily]

        if not daily_data_chunks:
            print("Failed to fetch any baseline daily data.")
            print("  Check the cache folder above and that Yahoo is reachable.")
            return pd.DataFrame()

        full_daily_data = pd.concat(daily_data_chunks, axis=1)
        print(f"{len(valid_symbols)} of {len(fixed_symbols)} symbols returned daily data "
              f"({len(missing_daily)} missing).")
        if missing_daily:
            shown = ", ".join(missing_daily[:12])
            print(f"  missing: {shown}" + (" ..." if len(missing_daily) > 12 else ""))

        # ---- Intraday: ONE download per batch feeds every intraday time frame
        intraday_tfs = [tf for tf in self.timeframes if tf != "1d"]
        base_tf = self.base_intraday
        print(f"Scanning intraday ({base_tf} base -> {', '.join(intraday_tfs)})...")
        self._fail_streak = 0

        for i in range(0, len(valid_symbols), batch_size):
            batch = valid_symbols[i: i + batch_size]
            base = self.get_batch_data(batch, base_tf)

            if base is None or base.empty:
                self._failed[base_tf] += 1
                for tf in intraday_tfs:
                    self._missing[tf].extend(batch)
                if self._budget_stop:
                    break
                self._fail_streak += 1
                if self._fail_streak >= 3:
                    print(self._rate_limit_note())
                    break
                self._pause(1.0)
                continue
            self._fail_streak = 0

            frames = {}
            for tf in intraday_tfs:
                if tf == base_tf:
                    frames[tf] = base
                elif tf == "4Hr" and self.native_4h is not False:
                    # Prefer Yahoo's own 4h bars; its grid is denser than a
                    # 30m->4h resample and matches the original script.
                    native = self.get_batch_data(batch, "4h", retries=1)
                    if native is not None and not native.empty:
                        frames[tf] = native
                        self.native_4h = True
                    else:
                        frames[tf] = self._resample_to(base, RESAMPLE_RULE[tf])
                        if self.native_4h is None:
                            self.native_4h = False
                            print("  (Yahoo returned no native 4h bars - "
                                  "deriving 4Hr from 30m instead)")
                else:
                    frames[tf] = self._resample_to(base, RESAMPLE_RULE.get(tf, tf))

            self._hourly_frame = frames.get("1Hr")
            for sym in batch:
                # compute once here so the 1d pass (which has no hourly frame)
                # still reports RSI-H from the cache
                self._mtf_rsi(sym, self._hourly_frame, full_daily_data)

            for tf in intraday_tfs:
                data = frames[tf]
                have = self._symbols_with_data(data, batch)
                self._scanned[tf] += len(have)
                self._missing[tf].extend([s for s in batch if s not in have])
                for sym in batch:
                    if sym not in have:
                        continue
                    for res in self.analyze_symbol(
                            data, full_daily_data, sym, n_bars, tf):
                        rows.append([res[0], tf] + res[1:])
            self._pause()

        # ---- Daily time frame reuses the 90d frame we already hold
        if "1d" in self.timeframes:
            print("Scanning 1d...")
            have = self._symbols_with_data(full_daily_data, valid_symbols)
            self._scanned["1d"] = len(have)
            self._missing["1d"].extend([s for s in valid_symbols if s not in have])
            for sym in valid_symbols:
                if sym not in have:
                    continue
                for res in self.analyze_symbol(
                        full_daily_data, full_daily_data, sym, n_bars, "1d"):
                    rows.append([res[0], "1d"] + res[1:])

        # Silent gaps were the worst part of the old version -- report them.
        print("\nCoverage: " + ", ".join(
            f"{tf}={self._scanned.get(tf, 0)}/{len(valid_symbols)}"
            for tf in self.timeframes))
        gaps = {tf: len(set(v)) for tf, v in self._missing.items() if v}
        if gaps:
            print("Symbols missing data: " + ", ".join(
                f"{tf}={n}" for tf, n in gaps.items()))
            worst = max(gaps, key=gaps.get)
            sample = sorted(set(self._missing[worst]))[:12]
            print(f"  e.g. {worst}: {', '.join(sample)}")
            if worst in (base_tf, "1d") and gaps[worst] > 0.2 * max(len(valid_symbols), 1):
                print("  Over 20% missing -> Yahoo is throttling or the cookie cache "
                      "is broken. Try a smaller batch (max_workers=2) or wait a few "
                      "minutes between runs.")
        if self._failed:
            print("Batches with NO data: " + ", ".join(
                f"{tf}={n}" for tf, n in self._failed.items()))
        if self._last_error:
            print(f"Last download exception: {self._last_error}")
        if self._budget_stop:
            print(f"Request budget of {self.max_requests} reached. Progress is cached -- "
                  "run again to continue where this stopped.")
        print(f"Yahoo requests: {self._requests_made} | served from cache: "
              f"{self._cache_hits} symbols")
        self.print_funnel()

        if not rows:
            # Still write a packet and publish artefacts: "nothing matched" is a
            # result, and the browser UI needs to show that the scan completed.
            print("No stocks matched.")
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
            os.makedirs(output_dir, exist_ok=True)
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            empty = pd.DataFrame(columns=["Symbol", "TF", "Imb", "CMP", "Unbrk",
                                          "%Diff", "ATR%", "Ago", "RSI",
                                          "RSI_H", "RSI_D", "RSI_W", "RVOL_H",
                                          "RVOL_D", "Is_Coiled", "Tier", "PE",
                                          "Days2Earn", "Sector", "Status"])
            packet = self.write_ai_packet(empty, empty, empty, Counter(),
                                          output_dir, timestamp_str)
            self.last_run = {"longs": empty, "shorts": empty, "annotated": empty,
                             "exclusions": Counter(), "csv": "", "packet": packet,
                             "coverage": {tf: f"0/{len(valid_symbols)}"
                                          for tf in self.timeframes},
                             "requests": self._requests_made,
                             "cache_hits": self._cache_hits, "funnel": dict(self._funnel)}
            if packet:
                print(f"AI packet written: {packet}")
            return empty

        df = pd.DataFrame(rows, columns=["Symbol", "TF", "Imb", "CMP", "Unbrk",
                                         "%Diff", "ATR%", "Ago", "RSI",
                                         "RSI_H", "RSI_D", "RSI_W",
                                         "RVOL_H", "RVOL_D", "Is_Coiled"])
        df = df.drop_duplicates()
        # Fundamentals must land before tiering (tiers use the earnings window)
        df = self.add_fundamentals(df)
        df = self.assign_tiers(df)

        # Earnings proximity: downgrade instead of dropping, so the name stays visible
        def downgrade_earnings_risk(row):
            days = self._num(row.get('Days2Earn'))
            if np.isfinite(days) and 0 <= days <= 7 and row['Tier'] <= 3:
                return 4
            return row['Tier']

        df['Tier'] = df.apply(downgrade_earnings_risk, axis=1)

        longs, shorts, annotated, exclusions = self.build_lists(df)
        annotated = annotated.sort_values(
            ["Status", "Confl", "%Diff"], ascending=[True, False, True])

        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(output_dir, exist_ok=True)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f"US_Matching_{timestamp_str}.csv")
        annotated.to_csv(filepath, index=False)

        # Terminal Print Output
        self.print_ranked(longs, "LONG SETUPS  (ImbHigh, one row per symbol)", GREEN)
        self.print_ranked(shorts, "SHORT SETUPS  (ImbLow, one row per symbol)", RED)
        self.print_exclusions(exclusions, len(df), len(longs) + len(shorts))
        self.print_tier_summary(annotated)
        print(f"\nResults saved to {filepath}")

        # Streamlit Browser Render Output
        # ==================================================
        # AI HANDOFF PACKET (no API key, no external call)
        # ==================================================
        packet = self.write_ai_packet(annotated, longs, shorts, exclusions,
                                      output_dir, timestamp_str)
        self.last_run = {
            "longs": longs, "shorts": shorts, "annotated": annotated,
            "exclusions": exclusions, "csv": filepath, "packet": packet,
            "coverage": {tf: f"{self._scanned.get(tf, 0)}/{len(valid_symbols)}"
                         for tf in self.timeframes},
            "requests": self._requests_made, "cache_hits": self._cache_hits,
            "funnel": dict(self._funnel),
        }
        if packet:
            print(f"\nAI packet written: {packet}")
            print("  Attach that file (or paste its contents) into the chat and ask")
            print("  for the top 5 buys.")

        return annotated

    # ------------------------------------------------------------- printing
    # ------------------------------------------------------------- AI handoff
    @staticmethod
    def _packet_table(frame) -> str:
        """Markdown table of the ranked rows -- everything needed to rank them."""
        if frame is None or frame.empty:
            return "_none_"
        cols = [("Symbol", "Symbol"), ("Sector", "Sector"), ("TF", "TF"),
                ("CMP", "CMP"), ("Unbrk", "Target"), ("%Diff", "%Diff"),
                ("ATRx", "ATRx"), ("Ago", "Ago"), ("RSI", "RSI"),
                ("RSI_H", "RSI-H"), ("RSI_D", "RSI-D"), ("RSI_W", "RSI-W"),
                ("RVOL_D", "RVOL_D"), ("Confl", "Confl"), ("PE", "PE"),
                ("Days2Earn", "D2Earn"), ("Pct50", "%vs50DMA"),
                ("Slope50", "50DMA slope%"), ("Is_Coiled", "Coiled")]
        head = "| " + " | ".join(h for _, h in cols) + " |"
        sep = "|" + "|".join("---" for _ in cols) + "|"
        lines = [head, sep]
        for _, r in frame.iterrows():
            atr_pct = r.get("ATR%")
            try:
                atrx = f"{float(r['%Diff']) / float(atr_pct):.1f}" if atr_pct else "n/a"
            except (TypeError, ValueError, ZeroDivisionError):
                atrx = "n/a"
            cells = []
            for key, _ in cols:
                if key == "ATRx":
                    cells.append(atrx)
                elif key == "Is_Coiled":
                    cells.append(VolumeAlertChecker._yn(r.get(key)))
                elif key in ("RSI", "RSI_H", "RSI_D", "RSI_W"):
                    cells.append(VolumeAlertChecker._fmt(r.get(key)))
                elif key in ("%Diff", "CMP", "Unbrk", "RVOL_D", "Pct50", "Slope50"):
                    try:
                        cells.append(f"{float(r[key]):.2f}")
                    except (TypeError, ValueError):
                        cells.append("n/a")
                else:
                    cells.append(str(r.get(key, "n/a")))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    def write_ai_packet(self, annotated, longs, shorts, exclusions,
                        output_dir, timestamp_str) -> str:
        """Write the handoff file to paste into the chat for a top-5 pick."""
        path = os.path.join(output_dir, f"AI_PACKET_{timestamp_str}.md")
        now_et = datetime.now(EASTERN).strftime("%Y-%m-%d %H:%M ET")
        state = "open" if _market_open_now() else "closed"
        L = []
        A = L.append
        A(f"# Quant scan packet - {now_et} (US market {state})")
        A("")
        A(f"- Universe with daily data: {self._scanned.get('1d', 0)} symbols")
        A(f"- Rows produced by the scan: {len(annotated)}")
        A(f"- After quality gates: {len(longs) + len(shorts)} "
          f"({len(longs)} long, {len(shorts)} short)")
        A("")
        def _trim(frame):
            if frame is None or frame.empty or len(frame) <= self.max_packet_rows:
                return frame, ""
            return (frame.head(self.max_packet_rows),
                    f"_showing the top {self.max_packet_rows} of {len(frame)}_\n")

        longs_t, note_l = _trim(longs)
        shorts_t, note_s = _trim(shorts)

        A("## LONG setups (ImbHigh - price above an unmitigated resistance level)")
        A("")
        A(note_l)
        A(self._packet_table(longs_t))
        A("")
        A("## SHORT setups (ImbLow - price above an unmitigated support level)")
        A("")
        A(note_s)
        A(self._packet_table(shorts_t))
        A("")
        if exclusions:
            A("## Rows removed by the quality gates")
            A("")
            for reason, n in exclusions.most_common():
                A(f"- {n}: {reason}")
            A("")
        A("## Legend")
        A("")
        A("- **Confl** - number of time frames agreeing on the same direction (higher is stronger)")
        A("- **ATRx** - distance to target divided by that time frame's ATR "
          "(lower = closer, more likely to be reached soon)")
        A("- **Ago** - bars since the unmitigated level formed; price has not traded through it since")
        A("- **RVOL_D** - today's daily volume vs its 20-day average")
        A("- **Pct50 / 50DMA slope%** - distance from the 50-day average, and that average's slope over 20 sessions")
        A("- **D2Earn** - days to earnings; negative means earnings already passed")
        A("- **RSI-H / RSI-D / RSI-W** - RSI(14) on the hourly, daily and weekly charts "
          "(weekly resampled from the same 1y of daily data, so it covers roughly the last year)")
        A("- **Coiled** - Y when Bollinger Bands sit inside Keltner Channels on the daily chart (a squeeze)")
        A("")
        A("## What I want back")
        A("")
        A("Recommend the **top 5 buys** from the LONG table above. For each: one line on why it")
        A("is the best of the set, the level to watch, and the main risk. Do not pick two names")
        A("from the same sector. If fewer than five are worth taking, say so and explain why.")
        A("This is a screen, not investment advice.")
        A("")
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("\n".join(L))
        except OSError as exc:
            print(f"Could not write AI packet: {exc}")
            return ""
        return path

    def print_funnel(self):
        """Show where candidates drop out: the answer to 'is it too strict?'"""
        if not self._funnel:
            return
        cols = [("passed_daily", "daily"), ("bars", "bars"), ("trend", "trend"),
                ("rsi", "rsi"), ("imbalance", "imb"), ("broken", "brk"),
                ("atr_cap", "atr"), ("too_close", "dist"), ("kept", "kept")]
        header = f"{'TF':<5}{'syms':>6}" + "".join(f"{h:>7}" for _, h in cols)
        print("\nFilter funnel (where candidates drop out)")
        print(header)
        print("-" * len(header))
        for tf in self.timeframes:
            row = f"{tf:<5}{self._scanned.get(tf, 0):>6}"
            row += "".join(f"{self._funnel.get(tf + '|' + k, 0):>7}" for k, _ in cols)
            print(row)
        print("  daily = symbols passing the 50DMA/volume/squeeze gate; bars = bars with\n"
              "  RVOL > threshold in the last n_bars. Each bar then stops at the first\n"
              "  filter it fails: trend -> rsi -> imb -> brk -> atr -> dist -> kept")

    def build_lists(self, df):
        """Hard quality gates, then one row per symbol per direction.

        Order is deliberately simple and predictable: most time frames agreeing
        first, then the nearest level, then the oldest unbroken level.
        Returns (longs, shorts, annotated_df, exclusion_counts).
        """
        reasons = Counter()
        if df.empty:
            return df, df, df, reasons

        d = df.copy()
        conf = d.groupby(["Symbol", "Imb"])["TF"].nunique().to_dict()
        d["Confl"] = [conf.get((sym, imb), 1) for sym, imb in zip(d["Symbol"], d["Imb"])]
        d["Pct50"] = [round((self._daily_cache.get(sym) or {}).get("pct_50dma", np.nan), 1)
                      for sym in d["Symbol"]]
        d["Slope50"] = [round((self._daily_cache.get(sym) or {}).get("slope_50dma", np.nan), 1)
                        for sym in d["Symbol"]]

        status = []
        for _, r in d.iterrows():
            ctx = self._daily_cache.get(r["Symbol"]) or {}
            price = ctx.get("close", np.nan)
            adv = ctx.get("adv20", np.nan)
            pe = self._num(r.get("PE"))
            days = self._num(r.get("Days2Earn"))
            atr_pct = self._num(r.get("ATR%"))
            pct = self._num(r.get("%Diff"))
            rvol_d = self._num(r.get("RVOL_D"))

            reason = ""
            if not np.isfinite(price) or price < self.min_price:
                reason = f"price below ${self.min_price:g}"
            elif not np.isfinite(adv) or adv < self.min_dollar_vol:
                reason = "20d dollar volume too thin"
            elif not np.isfinite(pe) or pe <= 0 or pe > self.max_pe:
                reason = f"PE missing or above {self.max_pe:g}"
            elif np.isfinite(days) and 0 <= days <= self.min_days_to_earnings:
                reason = f"earnings within {self.min_days_to_earnings:g} days"
            elif not np.isfinite(rvol_d) or rvol_d < self.min_rvol_d:
                reason = f"RVOL_D below {self.min_rvol_d:g}"
            elif np.isfinite(atr_pct) and atr_pct > 0 and pct > self.max_target_atr * atr_pct:
                reason = f"target beyond {self.max_target_atr:g} x ATR"
            status.append(reason)

        d["Status"] = status
        for reason in status:
            if reason:
                reasons[reason] += 1

        passed = d[d["Status"] == ""].copy()
        passed = passed.sort_values(["Confl", "%Diff", "Ago"],
                                    ascending=[False, True, False])
        picked = passed.drop_duplicates(subset=["Symbol", "Imb"], keep="first")

        dropped = set(passed.index) - set(picked.index)
        if dropped:
            reasons["duplicate time frames collapsed"] = len(dropped)
            d.loc[list(dropped), "Status"] = "duplicate (other time frame kept)"

        longs = picked[picked["Imb"] == "ImbHigh"].sort_values(
            ["Confl", "%Diff"], ascending=[False, True]).reset_index(drop=True)
        shorts = picked[picked["Imb"] == "ImbLow"].sort_values(
            ["Confl", "%Diff"], ascending=[False, True]).reset_index(drop=True)

        both = set(longs["Symbol"]) & set(shorts["Symbol"])
        if both:
            reasons["conflicting directions (chopping)"] = len(both)
            d.loc[d["Symbol"].isin(both) & (d["Status"] == ""), "Status"] = "conflicting directions"
            longs = longs[~longs["Symbol"].isin(both)].reset_index(drop=True)
            shorts = shorts[~shorts["Symbol"].isin(both)].reset_index(drop=True)

        keep = d["Status"] == ""
        d.loc[keep & (d["Imb"] == "ImbHigh"), "Status"] = "LONG"
        d.loc[keep & (d["Imb"] == "ImbLow"), "Status"] = "SHORT"
        return longs, shorts, d, reasons

    def print_exclusions(self, reasons, total, kept):
        if not reasons:
            return
        print(f"\nQuality gates: {kept} of {total} rows kept, "
              f"{total - kept} dropped")
        for reason, n in reasons.most_common():
            print(f"   {n:>4}  {reason}")

    def print_tier_summary(self, df):
        if df.empty or "Tier" not in df.columns:
            return
        labels = {1: "Tier 1 (high conviction)", 2: "Tier 2 (buys)",
                  3: "Tier 3 (momentum)", 4: "Tier 4 (warnings)"}
        print("\nTier summary")
        print("-" * 60)
        for t in (1, 2, 3, 4):
            names = sorted(df.loc[df["Tier"] == t, "Symbol"].unique())
            print(f"{labels[t]:<26} {len(names):>3}  {', '.join(names[:12])}"
                  + (" ..." if len(names) > 12 else ""))

    def print_ranked(self, frame, title, color):
        if frame is None or frame.empty:
            print(f"\n{title}\n  (nothing passed the quality gates)")
            return
        print(f"\n{title}")
        header = (f"{'#':>3} {'Symbol':<7} {'TF':<4} {'CMP':>9} {'Target':>9} "
                  f"{'%Diff':>6} {'ATRx':>5} {'Ago':>4} {'RSI':>5} "
                  f"{'RSI-H':>5} {'RSI-D':>5} {'RSI-W':>5} {'RV(D)':>6} "
                  f"{'Coil':>4} {'Cf':>3} {'PE':>7} {'D2Ern':>6}")
        print(header)
        print("-" * len(header))
        for n, (_, r) in enumerate(frame.iterrows(), start=1):
            atr_pct = self._num(r.get("ATR%"))
            atrx = (self._num(r.get("%Diff")) / atr_pct) if atr_pct else float("nan")
            print(color + f"{n:>3} {r['Symbol']:<7} {r['TF']:<4} {r['CMP']:>9.2f} "
                          f"{r['Unbrk']:>9.2f} {r['%Diff']:>6.2f} {atrx:>5.1f} "
                          f"{int(r['Ago']):>4} {r['RSI']:>5.1f} "
                          f"{self._fmt(r.get('RSI_H')):>5} {self._fmt(r.get('RSI_D')):>5} "
                          f"{self._fmt(r.get('RSI_W')):>5} {r['RVOL_D']:>6.1f} "
                          f"{self._yn(r.get('Is_Coiled')):>4} {int(r['Confl']):>3} "
                          f"{str(r['PE']):>7} {str(r['Days2Earn']):>6}" + RESET)
        print("  Cf = time frames agreeing; ATRx = distance to target in ATRs; "
              "RSI-H/D/W = hourly, daily, weekly RSI")

# ==================================================
def _read_text(path):
    """Read a run artefact for download; empty string if it is missing."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except (OSError, TypeError):
        return ""


def _render_app(checker, symbols):
    """Browser UI. Results live in session_state so widget clicks do not
    re-run a multi-minute scan."""
    try:
        st.set_page_config(page_title="US Quant Scanner", layout="wide")
    except Exception:
        pass

    st.title("US Quantitative Scanner")
    st.caption("Unmitigated-level scan across 30m / 1Hr / 4Hr / 1d. "
               "First run fetches everything; later runs reuse the on-disk cache.")

    if st.button("Run Scan", type="primary"):
        with st.spinner("Scanning... this can take a few minutes on a cold cache."):
            checker.run(symbols, n_bars=100)
        st.session_state["scan"] = dict(checker.last_run)

    scan = st.session_state.get("scan")
    if not scan:
        st.info("No results yet. Press Run Scan.")
        return

    longs, shorts = scan.get("longs"), scan.get("shorts")
    n_long = 0 if longs is None else len(longs)
    n_short = 0 if shorts is None else len(shorts)
    st.success(f"{n_long} long setups, {n_short} short setups after the quality gates.")

    st.subheader("Long setups")
    if n_long:
        st.dataframe(longs, use_container_width=True, hide_index=True)
    else:
        st.write("None passed the gates.")

    st.subheader("Short setups")
    if n_short:
        st.dataframe(shorts, use_container_width=True, hide_index=True)
    else:
        st.write("None passed the gates.")

    col1, col2 = st.columns(2)
    packet_text = _read_text(scan.get("packet"))
    with col1:
        st.download_button(
            "Download AI packet (.md)", packet_text or "no packet",
            file_name=os.path.basename(scan.get("packet") or "AI_PACKET.md"),
            mime="text/markdown", disabled=not packet_text)
        st.caption("Attach this file to the chat for the top-5 pick.")
    with col2:
        csv_text = _read_text(scan.get("csv"))
        st.download_button(
            "Download full results (.csv)", csv_text or "no csv",
            file_name=os.path.basename(scan.get("csv") or "US_Matching.csv"),
            mime="text/csv", disabled=not csv_text)
        st.caption("Every row, including what was excluded and why.")

    exclusions = scan.get("exclusions") or {}
    if exclusions:
        with st.expander("Rows removed by the quality gates"):
            for reason, n in sorted(exclusions.items(), key=lambda kv: -kv[1]):
                st.write(f"{n} - {reason}")

    if packet_text:
        with st.expander("Preview the AI packet"):
            st.markdown(packet_text)


# ==================================================
def main():
    symbols = [
        "A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACI", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
        "AEP", "AFL", "AFRM", "AGIO", "AIG", "AIZ", "AJG", "AKAM", "ALAB", "ALB", "ALGN", "ALK", "ALL", "ALLE", "ALLY",
        "ALNY", "AMAT", "AMC", "AMD", "AME", "AMGN", "AMN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD",
        "APH", "APP", "APTV", "ARE", "ARM", "ASML", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZN", "AZO", "BA",
        "BABA", "BAC", "BALL", "BAX", "BBAI", "BBY", "BDX", "BEN", "BIDU", "BIIB", "BILI", "BIO", "BJ", "BKNG", "BKR",
        "BLK", "BLMN", "BMRN", "BMY", "BR", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAH", "CAKE", "CARR", "CAT",
        "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHH", "CHRW", "CHTR",
        "CI", "CINF", "CINT", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO",
        "COST", "COTY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSX", "CTAS", "CTSH", "CTVA", "CUBE",
        "CVS", "CVX", "CZR", "D", "DAL", "DASH", "DD", "DDOG", "DE", "DG", "DHR", "DIS", "DKNG", "DLR", "DLTR",
        "DOCU", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EAT", "EBAY", "ECL",
        "ED", "EFX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
        "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FITB", "FLS",
        "FMC", "FOX", "FOXA", "FRSH", "FRT", "FSLR", "FTI", "FTNT", "GD", "GE", "GEHC", "GEN", "GEV", "GGG", "GILD",
        "GIS", "GL", "GLD", "GLW", "GM", "GME", "GNRC", "GOOG", "GOOGL", "GPC", "GRMN", "GS", "GWW", "H",
        "HAL", "HAS", "HBAN", "HCA", "HD", "HIG", "HII", "HLT", "HON", "HOOD", "HP", "HPE", "HPQ", "HST",
        "HSY", "HUBB", "HUM", "HUN", "HWM", "IBM", "ICE", "IDXX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
        "IR", "IRM", "ISRG", "ITW", "IVZ", "IWM", "JBHT", "JCI", "JD", "JKHY", "JNJ", "JPM", "KDP",
        "KEY", "KEYS", "KFY", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KNX", "KO", "KR", "L", "LCID", "LDOS",
        "LHX", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAN",
        "MAR", "MAS", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META", "MGM", "MKC", "MKTX",
        "MLM", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRVL", "MS", "MSCI",
        "MSFT", "MSTR", "MTB", "MTCH", "MTD", "MU", "NBR", "NCLH", "NDAQ", "NEE", "NEM", "NET", "NFLX", "NKE", "NMIH",
        "NNN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NU", "NUE", "NVDA", "NWS", "NWSA", "NXPI", "O",
        "ODFL", "OGN", "OKE", "OKTA", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PBI", "PCAR", "PDD",
        "PEG", "PENN", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PKG", "PLD", "PLTR", "PM", "PNC", "PNR",
        "PODD", "POOL", "PPG", "PRU", "PSA", "PSX", "PTC", "PTON", "PYPL", "QCOM", "QQQ", "QSR", "RBLX", "RCL", "RDW",
        "REG", "REGN", "RF", "RGNX", "RHI", "RIVN", "RL", "RMD", "RNG", "ROK", "ROKU", "ROL", "ROP", "ROST", "RS",
        "RSG", "RTX", "SBAC", "SBUX", "SCHW", "SEDG", "SFM", "SHOP", "SHW", "SIRI", "SJM", "SLB", "SLG", "SMCI", "SNA",
        "SNAP", "SNOW", "SNPS", "SO", "SPG", "SPGI", "SPY", "SRE", "SRPT", "STE", "STLD", "STT", "STX", "STZ", "SWK",
        "SWKS", "SYF", "SYK", "SYM", "SYY", "T", "TDG", "TEAM", "TECH", "TEL", "TEM", "TER", "TFC", "TGT", "TJX",
        "TMO", "TMUS", "TPR", "TRGP", "TRI", "TRIP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSM", "TT", "TTWO", "TWLO",
        "TXN", "TXRH", "TXT", "TYL", "U", "UA", "UAA", "UAL", "UDR", "UHS", "ULTA", "UNH", "UNM", "UNP", "UPS",
        "UPST", "URI", "V", "VFC", "VICI", "VLO", "VMC", "VNO", "VRSK", "VRSN", "VRT", "VRTX", "VTR", "VTRS", "VZ",
        "WAT", "WBA", "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC", "WH", "WHR", "WM", "WMB", "WMT", "WPC", "WRB",
        "WST", "WTW", "WY", "WYNN", "X", "XEL", "XOM", "XRAY", "XRX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZM",
        "ZS", "ZTS", "SPCX"
    ]

    checker = VolumeAlertChecker()

    if HAS_STREAMLIT and st.runtime.exists():
        _render_app(checker, symbols)
    else:
        checker.run(symbols, n_bars=100)


# ==================================================
if __name__ == "__main__":
    main()
