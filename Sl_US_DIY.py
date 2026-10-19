import os
import math
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# ---------------------------------------------------------
# DIRECTORY SETUP & CACHING LOGIC
# ---------------------------------------------------------
CACHE_DIR = "cache"
OUTPUT_DIR = "output"

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def fetch_data_with_cache(ticker: str, period: str, interval: str) -> pd.DataFrame:
    """Fetches data from Yahoo Finance. Caches the result daily."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    cache_filename = f"{ticker}_{period}_{interval}_{today_str}.pkl"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if os.path.exists(cache_path):
        return pd.read_pickle(cache_path)

    df = yf.download(ticker, period=period, interval=interval, progress=False)
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        
    if not df.empty and len(df) > 0:
        df.to_pickle(cache_path)
        
    return df

def save_results_to_csv(buy_results: list, sell_results: list) -> str:
    """Saves Buy and Sell screener results to a single timestamped CSV file with headers."""
    if not buy_results and not sell_results:
        return ""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(OUTPUT_DIR, f"DIY_US_Coiled_results_{timestamp}.csv")
    
    def process_to_dataframe(results_list, type_str):
        flat_data = []
        for item in results_list:
            row = {
                'Stock': item['Clean Ticker'],
                'Symbol': item['Ticker'],
                'CMP (USD)': item['Price ($)'],
                'Coiled': item['Coiled'],
                'BBB': item['BBB'],
                f'{type_str} Signals': item[f'{type_str} Signals'],
                'Total Indicators': item['Total Indicators'],
                'RSI (1H)': item['RSI (1H)'],
                'RSI (Daily)': item['RSI (Daily)'],
                'RSI (Weekly)': item['RSI (Weekly)']
            }
            for sig_name, sig_status in item['Signals Details'].items():
                row[sig_name] = type_str if sig_status else "Neutral/Opposite"
            flat_data.append(row)
        return pd.DataFrame(flat_data)

    # Write to a single file sequentially
    with open(filepath, 'w', newline='') as f:
        if buy_results:
            f.write("BUY CANDIDATES (COILED)\n")
            df_buy = process_to_dataframe(buy_results, "Buy")
            df_buy.to_csv(f, index=False)
            f.write("\n")  # Blank line separator
            
        if sell_results:
            f.write("SELL CANDIDATES (COILED)\n")
            df_sell = process_to_dataframe(sell_results, "Sell")
            df_sell.to_csv(f, index=False)

    return filepath

def is_streamlit_running():
    try:
        from streamlit.runtime.scriptrunner import get_script_run_context
        return get_script_run_context() is not None
    except Exception:
        return False

# ---------------------------------------------------------
# TECHNICAL INDICATORS CALCULATION ENGINE
# ---------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period).mean()

def wma(series: pd.Series, period: int) -> pd.Series:
    weights = np.arange(1, period + 1)
    return series.rolling(period).apply(
        lambda s: np.dot(s, weights) / weights.sum() if len(s) == period else np.nan, raw=True
    )

def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close_prev = df['Close'].shift(1)
    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def calc_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> bool:
    hl2 = (df['High'] + df['Low']) / 2
    atr = calc_atr(df, period)
    upper_band = (hl2 + multiplier * atr).values
    lower_band = (hl2 - multiplier * atr).values
    close = df['Close'].values
    n = len(df)
    
    if n < period: return False
    trend = 1
    final_upper, final_lower = upper_band.copy(), lower_band.copy()
    
    for i in range(1, n):
        if close[i-1] > final_lower[i-1]: final_lower[i] = max(lower_band[i], final_lower[i-1])
        if close[i-1] < final_upper[i-1]: final_upper[i] = min(upper_band[i], final_upper[i-1])
        if trend == 1: trend = -1 if close[i] < final_lower[i] else 1
        else: trend = 1 if close[i] > final_upper[i] else -1
            
    return trend == 1

def calc_range_filter(df: pd.DataFrame, per: int = 100, mult: float = 3.0) -> bool:
    close = df['Close']
    if len(close) < per: return False
    wper = per * 2 - 1
    avrng = (close - close.shift(1)).abs().ewm(span=per, adjust=False).mean()
    smrng = avrng.ewm(span=wper, adjust=False).mean() * mult
    
    filt = close.values.copy()
    for i in range(1, len(close)):
        c, r, prev = close.values[i], smrng.values[i], filt[i-1]
        if c > prev: filt[i] = prev if (c - r < prev) else (c - r)
        else: filt[i] = prev if (c + r > prev) else (c + r)
    return filt[-1] > filt[-2]

def calc_halftrend(df: pd.DataFrame, amplitude: int = 2) -> bool:
    if len(df) < 100: return False
    close, high, low = df['Close'].values, df['High'].values, df['Low'].values
    trend, next_trend = 0, 0
    max_low, min_high = low[0], high[0]
    
    for i in range(1, len(df)):
        h_slice, l_slice = high[max(0, i-amplitude):i+1], low[max(0, i-amplitude):i+1]
        h_ma, l_ma = high[max(0, i-amplitude+1):i+1].mean(), low[max(0, i-amplitude+1):i+1].mean()
        high_price, low_price = h_slice.max(), l_slice.min()
        
        if next_trend == 1:
            max_low = max(low_price, max_low)
            if h_ma < max_low and close[i] < low[i-1]: trend, next_trend, min_high = 1, 0, high_price
        else:
            min_high = min(high_price, min_high)
            if l_ma > min_high and close[i] > high[i-1]: trend, next_trend, max_low = 0, 1, low_price
    return trend == 0

def calc_chandelier_exit(df: pd.DataFrame, length: int = 22, mult: float = 3.0) -> tuple:
    atr = calc_atr(df, length)
    high_max = df['High'].rolling(length).max()
    low_min = df['Low'].rolling(length).min()
    long_stop = high_max - mult * atr
    short_stop = low_min + mult * atr
    c = df['Close'].iloc[-1]
    return c > long_stop.iloc[-1], c < short_stop.iloc[-1]

def calc_hull_ma(close: pd.Series, length: int = 55) -> pd.Series:
    wma_half = wma(close, length // 2)
    wma_full = wma(close, length)
    raw_hma = 2 * wma_half - wma_full
    return wma(raw_hma, int(math.sqrt(length)))

def calc_mcginley_dynamic(close: pd.Series, length: int = 14) -> pd.Series:
    mg = close.values.copy()
    for i in range(1, len(close)):
        denom = length * math.pow(max(close.values[i] / mg[i-1], 1e-4), 4)
        mg[i] = mg[i-1] + (close.values[i] - mg[i-1]) / max(denom, 1e-4)
    return pd.Series(mg, index=close.index)

def calc_is_coiled(df: pd.DataFrame, period: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> bool:
    """
    Determines if the stock is 'coiled' (consolidation/squeeze).
    Uses the classic Squeeze logic: Bollinger Bands are completely inside Keltner Channels.
    """
    if len(df) < period:
        return False
        
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Bollinger Bands
    sma_20 = close.rolling(period).mean()
    std_20 = close.rolling(period).std()
    bb_upper = sma_20 + bb_mult * std_20
    bb_lower = sma_20 - bb_mult * std_20
    
    # Keltner Channels
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_20 = tr.rolling(period).mean()
    kc_upper = sma_20 + kc_mult * atr_20
    kc_lower = sma_20 - kc_mult * atr_20
    
    # Is Coiled condition (Current bar or within recent 3 bars)
    coiled_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    # Check if coiled currently OR coiled in the previous 3 bars
    is_recently_coiled = coiled_series.iloc[-4:].any()
    return bool(is_recently_coiled)

def calc_bbb_status(df: pd.DataFrame, period: int = 20, bb_mult: float = 2.0, kc_mult: float = 1.5) -> str:
    """
    Determines if the stock is actively breaking the Upper Bollinger Band
    after being coiled, incorporating false-breakout filters (Volume & Band Expansion).
    """
    if len(df) < 30:
        return "N"

    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    # Bollinger Bands
    sma_20 = close.rolling(period).mean()
    std_20 = close.rolling(period).std()
    bb_upper = sma_20 + bb_mult * std_20
    bb_lower = sma_20 - bb_mult * std_20
    bb_width = (bb_upper - bb_lower) / sma_20.replace(0, np.nan)

    # Keltner Channels
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr_20 = tr.rolling(period).mean()
    kc_upper = sma_20 + kc_mult * atr_20
    kc_lower = sma_20 - kc_mult * atr_20

    # 1. Coiled in recent bars (bars -4 to -2)
    is_coiled_series = (bb_upper < kc_upper) & (bb_lower > kc_lower)
    was_recently_coiled = is_coiled_series.iloc[-4:-1].any() or is_coiled_series.iloc[-1]

    # 2. Upper Bollinger Band Cross
    curr_close = close.iloc[-1]
    curr_bbu = bb_upper.iloc[-1]
    prev_close = close.iloc[-2]
    prev_bbu = bb_upper.iloc[-2]
    is_breaking_upper_bb = (curr_close > curr_bbu) and (prev_close <= prev_bbu * 1.01)

    # 3. False-Breakout Filter Enhancements:
    # A. Volume expansion filter (Volume > 1.25x of 20 SMA volume)
    vol_sma20 = volume.rolling(20).mean().iloc[-1]
    vol_conf = volume.iloc[-1] > (1.25 * vol_sma20) if not pd.isna(vol_sma20) and vol_sma20 > 0 else True

    # B. Band Width Expansion (Volatility expanding upward)
    band_expanding = bb_width.iloc[-1] >= bb_width.iloc[-2]

    if was_recently_coiled and is_breaking_upper_bb and vol_conf and band_expanding:
        return "Y"
    return "N"

# ---------------------------------------------------------
# TECHNICAL EVALUATION ENGINE (DUAL BUY/SELL)
# ---------------------------------------------------------

def evaluate_signals(df: pd.DataFrame) -> tuple:
    if len(df) < 60: return {}, {}

    close, high, low, volume = df['Close'], df['High'], df['Low'], df['Volume']
    hlc3, hl2 = (high + low + close) / 3, (high + low) / 2
    
    buys, sells = {}, {}

    # 1. EMA 200 Filter
    ema200 = ema(close, 200)
    buys['EMA 200 Filter'] = close.iloc[-1] > ema200.iloc[-1]
    sells['EMA 200 Filter'] = close.iloc[-1] < ema200.iloc[-1]

    # 2. 2 EMA Cross
    ema50 = ema(close, 50)
    buys['2 EMA Cross (50/200)'] = ema50.iloc[-1] > ema200.iloc[-1]
    sells['2 EMA Cross (50/200)'] = ema50.iloc[-1] < ema200.iloc[-1]

    # 3. 3 EMA Ribbon
    ema9, ema21, ema55 = ema(close, 9), ema(close, 21), ema(close, 55)
    buys['3 EMA Ribbon'] = ema9.iloc[-1] > ema21.iloc[-1] > ema55.iloc[-1]
    sells['3 EMA Ribbon'] = ema9.iloc[-1] < ema21.iloc[-1] < ema55.iloc[-1]

    # 4. Range Filter
    rf_up = calc_range_filter(df, 100, 3.0)
    buys['Range Filter'] = rf_up
    sells['Range Filter'] = not rf_up

    # 5. Supertrend
    st_up = calc_supertrend(df, 10, 3.0)
    buys['Supertrend'] = st_up
    sells['Supertrend'] = not st_up

    # 6. HalfTrend
    ht_up = calc_halftrend(df, 2)
    buys['HalfTrend'] = ht_up
    sells['HalfTrend'] = not ht_up

    # 7. Donchian Breakout
    hh15, ll15 = high.rolling(15).max().shift(1), low.rolling(15).min().shift(1)
    buys['Donchian Breakout'] = close.iloc[-1] > hh15.iloc[-1]
    sells['Donchian Breakout'] = close.iloc[-1] < ll15.iloc[-1]

    # 8. Rate of Change
    roc = (close - close.shift(9)) / close.shift(9) * 100
    buys['ROC (9)'] = roc.iloc[-1] > 0
    sells['ROC (9)'] = roc.iloc[-1] < 0

    # 9. TSI
    pc = close.diff()
    tsi = 100 * (ema(ema(pc, 25), 13) / ema(ema(pc.abs(), 25), 13).replace(0, np.nan))
    tsi_sig = ema(tsi, 13)
    buys['TSI Signal Cross'] = tsi.iloc[-1] > tsi_sig.iloc[-1]
    sells['TSI Signal Cross'] = tsi.iloc[-1] < tsi_sig.iloc[-1]

    # 10. McGinley Dynamic
    mg = calc_mcginley_dynamic(close, 14)
    buys['McGinley Dynamic'] = close.iloc[-1] > mg.iloc[-1]
    sells['McGinley Dynamic'] = close.iloc[-1] < mg.iloc[-1]

    # 11. DPO
    dpo = close - sma(close, 10).shift(6)
    buys['DPO'] = dpo.iloc[-1] > 0
    sells['DPO'] = dpo.iloc[-1] < 0

    # 12. Ichimoku
    span_a = (((high.rolling(9).max() + low.rolling(9).min())/2 + (high.rolling(26).max() + low.rolling(26).min())/2) / 2).shift(26)
    span_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    buys['Ichimoku Cloud'] = close.iloc[-1] > span_a.iloc[-1] and close.iloc[-1] > span_b.iloc[-1]
    sells['Ichimoku Cloud'] = close.iloc[-1] < span_a.iloc[-1] and close.iloc[-1] < span_b.iloc[-1]

    # 13. VWAP
    vwap = (volume * hlc3).cumsum() / volume.cumsum().replace(0, np.nan)
    buys['VWAP'] = close.iloc[-1] > vwap.iloc[-1]
    sells['VWAP'] = close.iloc[-1] < vwap.iloc[-1]

    # 14. Chandelier Exit
    ce_buy, ce_sell = calc_chandelier_exit(df, 22, 3.0)
    buys['Chandelier Exit'] = ce_buy
    sells['Chandelier Exit'] = ce_sell

    # 15. CCI
    cci_ma = sma(hlc3, 20)
    cci_dev = (hlc3 - cci_ma).abs().rolling(20).mean()
    cci = (hlc3 - cci_ma) / (0.015 * cci_dev.replace(0, np.nan))
    buys['CCI'] = cci.iloc[-1] > 0
    sells['CCI'] = cci.iloc[-1] < 0

    # 16. DMI / ADX
    atr14 = calc_atr(df, 14)
    plus_di = 100 * ema(high.diff().clip(lower=0), 14) / atr14.replace(0, np.nan)
    minus_di = 100 * ema((-low.diff()).clip(lower=0), 14) / atr14.replace(0, np.nan)
    adx = ema(100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan), 14)
    buys['DMI / ADX'] = (plus_di.iloc[-1] > minus_di.iloc[-1]) and (adx.iloc[-1] >= 20)
    sells['DMI / ADX'] = (minus_di.iloc[-1] > plus_di.iloc[-1]) and (adx.iloc[-1] >= 20)

    # 17. Stochastic
    lowest_l, highest_h = low.rolling(14).min(), high.rolling(14).max()
    k_smooth = sma(100 * (close - lowest_l) / (highest_h - lowest_l).replace(0, np.nan), 3)
    d_smooth = sma(k_smooth, 3)
    buys['Stochastic %K > %D'] = k_smooth.iloc[-1] > d_smooth.iloc[-1]
    sells['Stochastic %K > %D'] = k_smooth.iloc[-1] < d_smooth.iloc[-1]

    # 18. RSI
    rsi = calc_rsi(close, 14)
    buys['RSI > RSI MA'] = rsi.iloc[-1] > sma(rsi, 14).iloc[-1]
    sells['RSI > RSI MA'] = rsi.iloc[-1] < sma(rsi, 14).iloc[-1]
    buys['RSI Level (50)'] = rsi.iloc[-1] > 50
    sells['RSI Level (50)'] = rsi.iloc[-1] < 50

    # 19. MACD
    macd_line = ema(close, 12) - ema(close, 26)
    macd_signal = ema(macd_line, 9)
    buys['MACD Cross'] = macd_line.iloc[-1] > macd_signal.iloc[-1]
    sells['MACD Cross'] = macd_line.iloc[-1] < macd_signal.iloc[-1]

    # 20. SSL Channel
    buys['SSL Channel'] = close.iloc[-1] > sma(high, 10).iloc[-1]
    sells['SSL Channel'] = close.iloc[-1] < sma(low, 10).iloc[-1]

    # 21. CMF
    mfv = ((close - low) - (high - close)) / (high - low).replace(0, np.nan) * volume
    cmf = mfv.rolling(20).sum() / volume.rolling(20).sum().replace(0, np.nan)
    buys['Chaikin Money Flow'] = cmf.iloc[-1] > 0
    sells['Chaikin Money Flow'] = cmf.iloc[-1] < 0

    # 22. Volume Surge
    vol_surge = volume.iloc[-1] > sma(volume, 20).iloc[-1]
    buys['Volume Surge'] = vol_surge
    sells['Volume Surge'] = vol_surge

    # 23. Awesome Oscillator
    ao = sma(hl2, 5) - sma(hl2, 34)
    buys['Awesome Oscillator'] = ao.iloc[-1] > 0
    sells['Awesome Oscillator'] = ao.iloc[-1] < 0

    # 24. Wolfpack ID
    buys['Wolfpack ID'] = ema(close, 3).iloc[-1] > ema(close, 8).iloc[-1]
    sells['Wolfpack ID'] = ema(close, 3).iloc[-1] < ema(close, 8).iloc[-1]

    # 25. Hull Suite Trend
    hma = calc_hull_ma(close, 55)
    buys['Hull Suite Trend'] = hma.iloc[-1] > hma.iloc[-3] if len(hma) > 3 else False
    sells['Hull Suite Trend'] = hma.iloc[-1] < hma.iloc[-3] if len(hma) > 3 else False

    # 26. Vortex
    str_sum = calc_atr(df, 1).rolling(14).sum()
    vip = (high - low.shift(1)).abs().rolling(14).sum() / str_sum.replace(0, np.nan)
    vim = (low - high.shift(1)).abs().rolling(14).sum() / str_sum.replace(0, np.nan)
    buys['Vortex VI'] = vip.iloc[-1] > vim.iloc[-1]
    sells['Vortex VI'] = vip.iloc[-1] < vim.iloc[-1]

    # 27. Choppiness
    chop = 100 * np.log10(calc_atr(df, 1).rolling(14).sum() / (high.rolling(14).max() - low.rolling(14).min()).replace(0, np.nan)) / np.log10(14)
    buys['Choppiness Filter (<61.8)'] = chop.iloc[-1] < 61.8
    sells['Choppiness Filter (<61.8)'] = chop.iloc[-1] < 61.8

    # 28. WAE
    wae_t1 = (ema(close, 20) - ema(close, 40)).diff() * 150
    bb_band = 2.0 * close.rolling(20).std()
    buys['Waddah Attar Explosion'] = (wae_t1.iloc[-1] > 0) and (wae_t1.iloc[-1] > bb_band.iloc[-1])
    sells['Waddah Attar Explosion'] = (wae_t1.iloc[-1] < 0) and (abs(wae_t1.iloc[-1]) > bb_band.iloc[-1])

    # 29. BBPT
    atr5 = calc_atr(df, 5)
    bull_t = (close - low.rolling(50).min()) / atr5.replace(0, np.nan)
    bear_t = (high.rolling(50).max() - close) / atr5.replace(0, np.nan)
    buys['Bull Bear Power Trend'] = (bull_t.iloc[-1] - bear_t.iloc[-1]) >= 2.0
    sells['Bull Bear Power Trend'] = (bull_t.iloc[-1] - bear_t.iloc[-1]) <= -2.0

    # 30. Bollinger %B
    bb_dev = 2.0 * close.rolling(20).std()
    pct_b = (close - (sma(close, 20) - bb_dev)) / (2 * bb_dev).replace(0, np.nan)
    buys['Bollinger %B'] = pct_b.iloc[-1] > 0.5
    sells['Bollinger %B'] = pct_b.iloc[-1] < 0.5

    return buys, sells

# ---------------------------------------------------------
# MULTI-TIMEFRAME RSI (1H, 1D, 1W)
# ---------------------------------------------------------

def fetch_multitimeframe_rsi(ticker: str) -> dict:
    res = {'RSI_1H': np.nan, 'RSI_D': np.nan, 'RSI_W': np.nan}
    try:
        df_1h = fetch_data_with_cache(ticker, "1mo", "1h")
        if not df_1h.empty and len(df_1h) >= 15: res['RSI_1H'] = round(float(calc_rsi(df_1h['Close'], 14).iloc[-1]), 2)
            
        df_d = fetch_data_with_cache(ticker, "1y", "1d")
        if not df_d.empty and len(df_d) >= 15: res['RSI_D'] = round(float(calc_rsi(df_d['Close'], 14).iloc[-1]), 2)
            
        df_w = fetch_data_with_cache(ticker, "2y", "1wk")
        if not df_w.empty and len(df_w) >= 15: res['RSI_W'] = round(float(calc_rsi(df_w['Close'], 14).iloc[-1]), 2)
    except Exception:
        pass
    return res

def run_screener(tickers: list, min_signals: int = 20, progress_callback=None):
    buy_results, sell_results = [], []
    total = len(tickers)
    
    for idx, ticker in enumerate(tickers):
        formatted_ticker = ticker
        
        try:
            df = fetch_data_with_cache(formatted_ticker, "1y", "1d")
            if df.empty or len(df) < 60: continue
            
            # Check for Coiled / Consolidation first
            is_coiled = calc_is_coiled(df)
            if not is_coiled:
                # Strictly filter out non-coiled assets per user request
                continue
                
            buys, sells = evaluate_signals(df)
            buy_count = sum(1 for v in buys.values() if v)
            sell_count = sum(1 for v in sells.values() if v)
            total_ind = len(buys)
            
            # Continue strictly only if signals pass the threshold
            if buy_count < min_signals and sell_count < min_signals:
                continue
            
            rsi_dict = fetch_multitimeframe_rsi(formatted_ticker)
            last_price = round(float(df['Close'].iloc[-1]), 2)
            bbb_flag = calc_bbb_status(df)
            
            base_data = {
                'Ticker': formatted_ticker,
                'Clean Ticker': formatted_ticker,
                'Price ($)': last_price,
                'Coiled': True,
                'BBB': bbb_flag,
                'Total Indicators': total_ind,
                'RSI (1H)': rsi_dict['RSI_1H'],
                'RSI (Daily)': rsi_dict['RSI_D'],
                'RSI (Weekly)': rsi_dict['RSI_W']
            }

            if buy_count >= min_signals:
                b_item = base_data.copy()
                b_item['Buy Signals'] = buy_count
                b_item['Signals Details'] = buys
                buy_results.append(b_item)

            if sell_count >= min_signals:
                s_item = base_data.copy()
                s_item['Sell Signals'] = sell_count
                s_item['Signals Details'] = sells
                sell_results.append(s_item)

        except Exception:
            pass
            
        if progress_callback:
            progress_callback(idx + 1, total)

    # Sort BBB == 'Y' first, then by signals count descending
    buy_results.sort(key=lambda x: (1 if x['BBB'] == 'Y' else 0, x['Buy Signals']), reverse=True)
    sell_results.sort(key=lambda x: (1 if x['BBB'] == 'Y' else 0, x['Sell Signals']), reverse=True)
    return buy_results, sell_results

# ---------------------------------------------------------
# US STOCKS LIST (Top 100 Popular US Equities & ETFs)
# ---------------------------------------------------------

US_STOCKS = [
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

# ---------------------------------------------------------
# RUNTIME DISPATCHER
# ---------------------------------------------------------

if is_streamlit_running():
    import streamlit as st

    st.set_page_config(page_title="US Technical Screener", layout="wide", page_icon="🇺🇸")
    st.title("🇺🇸 US Coiled Stock Screener")
    st.markdown("Filtering for **Coiled** assets (Bollinger Bands inside Keltner Channels) with **≥20 Signals**.")

    with st.sidebar:
        st.header("⚙️ Screener Settings")
        min_sigs = st.slider("Minimum Buy/Sell Confirmations:", min_value=1, max_value=30, value=20)
        
    initial_stocks = ", ".join(US_STOCKS)

    stock_input = st.text_area("Stocks to Scan (comma separated):", initial_stocks, height=150)
    
    if st.button("🚀 Run Coiled Screener", use_container_width=True):
        tickers = [t.strip().upper() for t in stock_input.split(",") if t.strip()]
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_ui(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Scanning {current}/{total} US stocks...")

        with st.spinner("Processing indicators..."):
            buys_list, sells_list = run_screener(tickers, min_signals=min_sigs, progress_callback=update_ui)
            status_text.empty()
            progress_bar.empty()

        combined_csv = save_results_to_csv(buys_list, sells_list)
        if combined_csv: st.info(f"✅ Combined CSV export saved to `{combined_csv}`")

        def style_buy_row(row):
            count = int(row['Buy Signals'].split(' / ')[0])
            color = 'background-color: rgba(0, 255, 0, 0.2); color: #00FF00' if count >= 20 else ''
            return [color] * len(row)

        def style_sell_row(row):
            count = int(row['Sell Signals'].split(' / ')[0])
            color = 'background-color: rgba(255, 0, 0, 0.2); color: #FF0000' if count >= 20 else ''
            return [color] * len(row)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("🟢 BUY Candidates (Coiled)")
            if buys_list:
                df_b = pd.DataFrame([{
                    'Stock': i['Clean Ticker'],
                    'CMP ($)': i['Price ($)'],
                    'BBB': i['BBB'],
                    'Buy Signals': f"{i['Buy Signals']} / {i['Total Indicators']}",
                    'RSI(1H)': i['RSI (1H)'],
                    'RSI(D)': i['RSI (Daily)'],
                    'RSI(W)': i['RSI (Weekly)']
                } for i in buys_list])
                st.dataframe(df_b.style.apply(style_buy_row, axis=1), use_container_width=True)
            else:
                st.warning(f"No coiled stocks with ≥ {min_sigs} Buy signals.")

        with c2:
            st.subheader("🔴 SELL Candidates (Coiled)")
            if sells_list:
                df_s = pd.DataFrame([{
                    'Stock': i['Clean Ticker'],
                    'CMP ($)': i['Price ($)'],
                    'BBB': i['BBB'],
                    'Sell Signals': f"{i['Sell Signals']} / {i['Total Indicators']}",
                    'RSI(1H)': i['RSI (1H)'],
                    'RSI(D)': i['RSI (Daily)'],
                    'RSI(W)': i['RSI (Weekly)']
                } for i in sells_list])
                st.dataframe(df_s.style.apply(style_sell_row, axis=1), use_container_width=True)
            else:
                st.warning(f"No coiled stocks with ≥ {min_sigs} Sell signals.")

else:
    # CLI Terminal Execution
    def main():
        stocks_to_scan = US_STOCKS
        print("\n" + "="*95)
        print("          US MARKET - COILED INDICATOR STRATEGY SCREENER")
        print("="*95)
        print(f"[*] Scanning {len(stocks_to_scan)} US stocks (Filtering for Coiled & >= 20 Signals)...\n")

        buys_list, sells_list = run_screener(stocks_to_scan, min_signals=20)

        # ANSI Colors
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        # --- BUY TABLE ---
        print("\n" + "="*95)
        print("🟢 BUY CANDIDATES (COILED & >= 20 Signals)")
        print("="*95)
        header_b = f"{'Rank':<5}{'Stock':<12}{'CMP ($)':<12}{'BBB':<6}{'Buy Signals':<16}{'RSI(1H)':<10}{'RSI(D)':<10}{'RSI(W)':<10}"
        print(header_b)
        print("-" * len(header_b))

        if not buys_list:
            print("[-] No stocks passed the BUY screener.")
        else:
            for rank, item in enumerate(buys_list, start=1):
                sigs = f"{item['Buy Signals']}/{item['Total Indicators']}"
                bbb = item.get('BBB', 'N')
                r1h = f"{item['RSI (1H)']:.2f}" if not pd.isna(item['RSI (1H)']) else "N/A"
                rd = f"{item['RSI (Daily)']:.2f}" if not pd.isna(item['RSI (Daily)']) else "N/A"
                rw = f"{item['RSI (Weekly)']:.2f}" if not pd.isna(item['RSI (Weekly)']) else "N/A"
                
                row_str = f"{rank:<5}{item['Ticker']:<12}${item['Price ($)']:<11.2f}{bbb:<6}{sigs:<16}{r1h:<10}{rd:<10}{rw:<10}"
                
                # Apply Green color if signals >= 20
                if item['Buy Signals'] >= 20:
                    print(f"{GREEN}{row_str}{RESET}")
                else:
                    print(row_str)

        # --- SELL TABLE ---
        print("\n\n" + "="*95)
        print("🔴 SELL CANDIDATES (COILED & >= 20 Signals)")
        print("="*95)
        header_s = f"{'Rank':<5}{'Stock':<12}{'CMP ($)':<12}{'BBB':<6}{'Sell Signals':<16}{'RSI(1H)':<10}{'RSI(D)':<10}{'RSI(W)':<10}"
        print(header_s)
        print("-" * len(header_s))

        if not sells_list:
            print("[-] No stocks passed the SELL screener.")
        else:
            for rank, item in enumerate(sells_list, start=1):
                sigs = f"{item['Sell Signals']}/{item['Total Indicators']}"
                bbb = item.get('BBB', 'N')
                r1h = f"{item['RSI (1H)']:.2f}" if not pd.isna(item['RSI (1H)']) else "N/A"
                rd = f"{item['RSI (Daily)']:.2f}" if not pd.isna(item['RSI (Daily)']) else "N/A"
                rw = f"{item['RSI (Weekly)']:.2f}" if not pd.isna(item['RSI (Weekly)']) else "N/A"
                
                row_str = f"{rank:<5}{item['Ticker']:<12}${item['Price ($)']:<11.2f}{bbb:<6}{sigs:<16}{r1h:<10}{rd:<10}{rw:<10}"
                
                # Apply Red color if signals >= 20
                if item['Sell Signals'] >= 20:
                    print(f"{RED}{row_str}{RESET}")
                else:
                    print(row_str)

        # Save to single CSV
        combined_file = save_results_to_csv(buys_list, sells_list)
        print("\n" + "="*95)
        if combined_file: print(f"[+] Combined detailed CSV saved: {combined_file}")
        print("="*95 + "\n")

    if __name__ == '__main__':
        main()
