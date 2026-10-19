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
    filepath = os.path.join(OUTPUT_DIR, f"DIY_India_results_{timestamp}.csv")
    
    def process_to_dataframe(results_list, type_str):
        flat_data = []
        for item in results_list:
            row = {
                'Stock': item['Clean Ticker'],
                'Symbol': item['Ticker'],
                'CMP (INR)': item['Price (₹)'],
                'Coiled': item['Coiled'],
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
    
    # Is Coiled condition (Current bar)
    coiled = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
    return bool(coiled)

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
        formatted_ticker = ticker if (ticker.endswith(".NS") or ticker.endswith(".BO")) else f"{ticker}.NS"
        
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
            
            base_data = {
                'Ticker': formatted_ticker,
                'Clean Ticker': formatted_ticker.replace('.NS', '').replace('.BO', ''),
                'Price (₹)': last_price,
                'Coiled': True,
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

    buy_results.sort(key=lambda x: x['Buy Signals'], reverse=True)
    sell_results.sort(key=lambda x: x['Sell Signals'], reverse=True)
    return buy_results, sell_results

# ---------------------------------------------------------
# INDIAN STOCKS LIST (Combined & Deduplicated)
# ---------------------------------------------------------

INDIAN_STOCKS = [
        'ADANIENT.NS', 'ADANIPORTS.NS', 'APOLLOHOSP.NS', 'ASIANPAINT.NS',
        'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJAJFINSV.NS', 'BAJFINANCE.NS',
        'BHARTIARTL.NS', 'BPCL.NS', 'BRITANNIA.NS', 'CIPLA.NS',
        'COALINDIA.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS',
        'GRASIM.NS', 'HCLTECH.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS',
        'HEROMOTOCO.NS', 'HINDALCO.NS', 'HINDUNILVR.NS', 'ICICIBANK.NS',
        'INDUSINDBK.NS', 'INFY.NS', 'ITC.NS', 'JSWSTEEL.NS',
        'KOTAKBANK.NS', 'LT.NS', 'M&M.NS', 'MARUTI.NS',
        'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
        'RELIANCE.NS', 'SBIN.NS', 'SBILIFE.NS', 'SHREECEM.NS',
        'SUNPHARMA.NS', 'TATACONSUM.NS', 'TATASTEEL.NS',
        'TCS.NS', 'TECHM.NS', 'TITAN.NS', 'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS',
        'BANDHANBNK.NS', 'FEDERALBNK.NS', 'IDFCFIRSTB.NS', 'PNB.NS',
        'RBLBANK.NS', 'AUBANK.NS', 'MMTC.NS', 'HEG.NS', 'HFCL.NS', 'SCI.NS',
        'GRAPHITE.NS', 'HBLENGINE.NS', 'NTPCGREEN.NS', 'FORCEMOT.NS',
        'SAMMAANCAP.NS', 'HINDCOPPER.NS', 'NLCINDIA.NS', 'FACT.NS',
        'ASTRAZEN.NS', 'GPIL.NS', 'ANANDRATHI.NS', 'DEEPAKNTR.NS', 'SWANCORP.NS',
        'FINPIPE.NS', 'WELCORP.NS', 'SUNDARMFIN.NS', 'ACMESOLAR.NS', 'INOXINDIA.NS',
        'NCC.NS', 'USHAMART.NS', 'RCF.NS', 'MAHABANK.NS', 'LALPATHLAB.NS',
        'SAREGAMA.NS', 'ATUL.NS', 'GMDCLTD.NS', 'HUDCO.NS', 'CREDITACC.NS',
        'INDIACEM.NS', 'BIKAJI.NS', 'CRISIL.NS', 'HONASA.NS', 'MARICO.NS',
        'TATACOMM.NS', 'SAIL.NS', 'TITAGARH.NS', 'JINDALSTEL.NS', 'AIIL.NS',
        'COROMANDEL.NS', 'HINDPETRO.NS', 'CCL.NS', 'NAM-INDIA.NS', 'JUBLINGREA.NS',
        'JKCEMENT.NS', 'BALKRISIND.NS', 'BEML.NS', 'PGHH.NS', 'OIL.NS',
        'KPRMILL.NS', 'KEI.NS', 'SUMICHEM.NS', 'APLAPOLLO.NS', 'INDUSTOWER.NS',
        'INDIANB.NS', 'MANYAVAR.NS', 'NSLNISP.NS', 'SHYAMMETL.NS', 'IOC.NS',
        'GESHIP.NS', 'NEULANDLAB.NS', 'WHIRLPOOL.NS', 'VOLTAS.NS',
        'CHAMBLFERT.NS', 'WOCKPHARMA.NS', 'BERGEPAINT.NS', 'ENDURANCE.NS',
        'SWIGGY.NS', 'JUBLFOOD.NS', 'TEJASNET.NS', 'MAZDOCK.NS', 'SYNGENE.NS',
        'RKFORGE.NS', 'IGL.NS', 'POLYMED.NS', 'SCHAEFFLER.NS', 'SUPREMEIND.NS',
        'INDIGO.NS', 'TMPV.NS', 'IGIL.NS', 'BHARTIHEXA.NS', 'FLUOROCHEM.NS',
        'DCMSHRIRAM.NS', 'VIJAYA.NS', 'RAMCOCEM.NS', 'ZFCVINDIA.NS', 'JKTYRE.NS',
        'PIDILITIND.NS', 'COFORGE.NS', 'DALBHARAT.NS', 'MOTHERSON.NS', 'ESCORTS.NS',
        'HAL.NS', 'BAYERCROP.NS', 'NBCC.NS', 'BDL.NS', 'NH.NS',
        'INDHOTEL.NS', 'INDGN.NS', 'ETERNAL.NS', 'NAVINFLUOR.NS', 'KEC.NS',
        'MAHSCOOTER.NS', 'BANKBARODA.NS', 'MAHSEAMLES.NS', 'ABSLAMC.NS',
        'KFINTECH.NS', 'VMM.NS', 'GODREJPROP.NS', 'CANBK.NS', 'LTF.NS',
        'ICICIPRULI.NS', 'MCX.NS', 'VENTIVE.NS', 'VBL.NS', 'SOLARINDS.NS',
        'LLOYDSME.NS', 'AKUMS.NS', 'PVRINOX.NS', 'GLENMARK.NS', 'MANKIND.NS',
        'POONAWALLA.NS', 'JYOTHYLAB.NS', 'HYUNDAI.NS', 'GODREJCP.NS', 'ALKEM.NS',
        'PTCIL.NS', 'CHOLAFIN.NS', 'ABB.NS', 'IDBI.NS', 'HONAUT.NS', 'BSE.NS',
        'DABUR.NS', 'DMART.NS', 'FIRSTCRY.NS', 'THERMAX.NS', 'UNOMINDA.NS',
        'PIIND.NS', 'ABLBL.NS', 'MFSL.NS', 'BHARATFORG.NS', 'ALKYLAMINE.NS',
        'TATACHEM.NS', 'AJANTPHARM.NS', 'HOMEFIRST.NS', 'GVT&D.NS', 'IOB.NS',
        'CONCORDBIO.NS', 'LAURUSLABS.NS', 'BEL.NS', 'ABFRL.NS', 'PAYTM.NS',
        'SIGNATURE.NS', 'TRITURBINE.NS', 'POLICYBZR.NS', 'ASHOKLEY.NS',
        'CHOICEIN.NS', 'GLAXO.NS', 'GICRE.NS', 'GRSE.NS', 'ELECON.NS',
        'CANFINHOME.NS', 'POWERINDIA.NS', 'SRF.NS', 'INTELLECT.NS', 'ERIS.NS',
        'GLAND.NS', 'EIHOTEL.NS', 'BOSCHLTD.NS', 'AGARWALEYE.NS', 'TORNTPHARM.NS',
        '360ONE.NS', 'MOTILALOFS.NS', 'UCOBANK.NS', 'NYKAA.NS', 'HDFCAMC.NS',
        'NEWGEN.NS', 'NAVA.NS', 'LICI.NS', 'ICICIGI.NS', 'YESBANK.NS', 'PAGEIND.NS',
        'VEDL.NS', 'GAIL.NS', 'ZEEL.NS', 'BBTC.NS', 'IDEA.NS', 'ACC.NS',
        'BANKINDIA.NS', 'DELHIVERY.NS', 'LINDEINDIA.NS', 'VGUARD.NS',
        'OBEROIRLTY.NS', 'GODREJAGRO.NS', 'RECLTD.NS', 'SAPPHIRE.NS',
        'SHRIRAMFIN.NS', 'J&KBANK.NS', 'ECLERX.NS', 'SUZLON.NS', 'CONCOR.NS',
        'AMBUJACEM.NS', 'ASTERDM.NS', 'MAXHEALTH.NS', 'INDIAMART.NS', 'BLUEDART.NS',
        'JYOTICNC.NS', 'AWL.NS', 'BHEL.NS', 'TARIL.NS', 'AARTIIND.NS',
        'ZYDUSLIFE.NS', 'LATENTVIEW.NS', 'CERA.NS', 'BIOCON.NS', 'CAMS.NS',
        'NIACL.NS', 'CHALET.NS', 'MANAPPURAM.NS', 'LODHA.NS', 'CUB.NS',
        'PRESTIGE.NS', 'TBOTEK.NS', 'APARINDS.NS', 'SAGILITY.NS', 'ABCAPITAL.NS',
        'ATHERENERG.NS', 'LICHSGFIN.NS', 'RAINBOW.NS', 'NIVABUPA.NS', 'TIINDIA.NS',
        'ATGL.NS', 'PATANJALI.NS', 'FSL.NS', 'JIOFIN.NS', 'UNIONBANK.NS',
        'CHOLAHLDNG.NS', 'MRPL.NS', 'CENTURYPLY.NS', 'BLS.NS', 'TECHNOE.NS',
        'GODIGIT.NS', 'LTTS.NS', 'EXIDEIND.NS', 'DEEPAKFERT.NS', 'MGL.NS', 'UBL.NS',
        'HAPPSTMNDS.NS', 'JBCHEPHARM.NS', 'RHIM.NS', 'LUPIN.NS', 'TATAELXSI.NS',
        'WELSPUNLIV.NS', 'AEGISLOG.NS', 'GILLETTE.NS', 'AADHARHFC.NS', 'PFIZER.NS',
        'DATAPATTNS.NS', 'DLF.NS', 'CYIENT.NS', 'IRCTC.NS', 'ZENTEC.NS',
        'BLUEJET.NS', 'POLYCAB.NS', 'IIFL.NS', 'COLPAL.NS', 'REDINGTON.NS',
        'TATAPOWER.NS', 'BRIGADE.NS', 'CAPLIPOINT.NS', 'NAUKRI.NS', 'PFC.NS',
        'GMRAIRPORT.NS', 'CROMPTON.NS', 'COCHINSHIP.NS', 'PERSISTENT.NS',
        'CENTRALBK.NS', 'BAJAJHFL.NS', 'TATATECH.NS', '3MINDIA.NS', 'GODREJIND.NS',
        'SUNDRMFAST.NS', 'KPIL.NS', 'HSCL.NS', 'SKFINDIA.NS', 'TIMKEN.NS',
        'MUTHOOTFIN.NS', 'CARBORUNIV.NS', 'ZENSARTECH.NS', 'SIEMENS.NS', 'SBFC.NS',
        'BALRAMCHIN.NS', 'CGPOWER.NS', 'PPLPHARMA.NS', 'DOMS.NS', 'SAILIFE.NS',
        'ARE&M.NS', 'AFCONS.NS', 'VTL.NS', 'CDSL.NS', 'KALYANKJIL.NS', 'CESC.NS',
        'CEATLTD.NS', 'CHENNPETRO.NS', 'PNBHOUSING.NS', 'GSPL.NS', 'DEVYANI.NS',
        'BATAINDIA.NS', 'TVSMOTOR.NS', 'TRIDENT.NS', 'KPITTECH.NS', 'TATAINVEST.NS',
        'LEMONTREE.NS', 'AAVAS.NS', 'SOBHA.NS', 'JSWINFRA.NS', 'AFFLE.NS',
        'ADANIENSOL.NS', 'COHANCE.NS', 'HEXT.NS', 'IREDA.NS', 'ADANIPOWER.NS',
        'MRF.NS', 'UTIAMC.NS', 'INOXWIND.NS', 'NATCOPHARM.NS', 'OFSS.NS',
        'CLEAN.NS', 'JSWENERGY.NS', 'MSUMI.NS', 'MEDANTA.NS', 'KAJARIACER.NS',
        'IEX.NS', 'MPHASIS.NS', 'JSL.NS', 'ABREL.NS', 'SJVN.NS', 'SCHNEIDER.NS',
        'JINDALSAW.NS', 'GODFRYPHLP.NS', 'METROPOLIS.NS', 'ENRIN.NS', 'ADANIGREEN.NS',
        'TRENT.NS', 'JPPOWER.NS', 'ITCHOTELS.NS', 'IRB.NS', 'CGCL.NS', 'NUVAMA.NS',
        'APLLTD.NS', 'SONACOMS.NS', 'RAILTEL.NS', 'TORNTPOWER.NS', 'ALOKINDS.NS',
        'EIDPARRY.NS', 'NATIONALUM.NS', 'ABBOTINDIA.NS', 'RRKABEL.NS', 'KIMS.NS',
        'SBICARD.NS', 'PETRONET.NS', 'NMDC.NS', 'FORTIS.NS', 'AMBER.NS',
        'NUVOCO.NS', 'NHPC.NS', 'EMCURE.NS', 'FIVESTAR.NS', 'CAMPUS.NS',
        'CASTROLIND.NS', 'KSB.NS', 'ONESOURCE.NS', 'ELGIEQUIP.NS', 'ENGINERSIN.NS',
        'IPCALAB.NS', 'KIRLOSBROS.NS', 'PGEL.NS', 'NETWEB.NS', 'ITI.NS',
        'JUBLPHARMA.NS', 'MAPMYINDIA.NS', 'PREMIERENE.NS', 'THELEELA.NS',
        'RADICO.NS', 'AIAENG.NS', 'BSOFT.NS', 'SUNTV.NS', 'KARURVYSYA.NS',
        'M&MFIN.NS', 'MINDACORP.NS', 'AEGISVOPAK.NS', 'HINDZINC.NS', 'JBMA.NS',
        'GRANULES.NS', 'TRIVENI.NS', 'PRAJIND.NS', 'GRAVITA.NS', 'RITES.NS',
        'DIXON.NS', 'FINCABLES.NS', 'SONATSOFTW.NS', 'APTUS.NS', 'IFCI.NS',
        'ANANTRAJ.NS', 'OLAELEC.NS', 'IKS.NS', 'RPOWER.NS', 'CUMMINSIND.NS',
        'LTFOODS.NS', 'IRCON.NS', 'OLECTRA.NS', 'KAYNES.NS', 'TTML.NS',
        'SARDAEN.NS', 'ASTRAL.NS', 'DBREALTY.NS', 'WAAREEENER.NS', 'BLUESTARCO.NS',
        'STARHEALTH.NS', 'APOLLOTYRE.NS', 'EMAMILTD.NS', 'UNITDSPR.NS',
        'PHOENIXLTD.NS', 'AUROPHARMA.NS', 'HAVELLS.NS', 'BASF.NS', 'JMFINANCIL.NS',
        'BAJAJHLDNG.NS'
]

# ---------------------------------------------------------
# RUNTIME DISPATCHER
# ---------------------------------------------------------

if is_streamlit_running():
    import streamlit as st

    st.set_page_config(page_title="NSE India Technical Screener", layout="wide", page_icon="🇮🇳")
    st.title("🇮🇳 Indian Stock Market Technical Indicator Screener (NSE/BSE)")
    st.markdown("Filtering for **Coiled** assets (Bollinger Bands inside Keltner Channels) with **≥20 Signals**.")

    with st.sidebar:
        st.header("⚙️ Screener Settings")
        min_sigs = st.slider("Minimum Buy/Sell Confirmations:", min_value=1, max_value=30, value=20)
        
    initial_stocks = ", ".join(INDIAN_STOCKS)

    stock_input = st.text_area("Stocks to Scan (comma separated):", initial_stocks, height=150)
    
    if st.button("🚀 Run Coiled Screener", use_container_width=True):
        tickers = [t.strip().upper() for t in stock_input.split(",") if t.strip()]
        progress_bar = st.progress(0.0)
        status_text = st.empty()

        def update_ui(current, total):
            progress_bar.progress(current / total)
            status_text.text(f"Scanning {current}/{total} Indian stocks...")

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
                    'CMP (₹)': i['Price (₹)'],
                    'Buy Signals': f"{i['Buy Signals']} / {i['Total Indicators']}",
                    'RSI(D)': i['RSI (Daily)']
                } for i in buys_list])
                st.dataframe(df_b.style.apply(style_buy_row, axis=1), use_container_width=True)
            else:
                st.warning(f"No coiled stocks with ≥ {min_sigs} Buy signals.")

        with c2:
            st.subheader("🔴 SELL Candidates (Coiled)")
            if sells_list:
                df_s = pd.DataFrame([{
                    'Stock': i['Clean Ticker'],
                    'CMP (₹)': i['Price (₹)'],
                    'Sell Signals': f"{i['Sell Signals']} / {i['Total Indicators']}",
                    'RSI(D)': i['RSI (Daily)']
                } for i in sells_list])
                st.dataframe(df_s.style.apply(style_sell_row, axis=1), use_container_width=True)
            else:
                st.warning(f"No coiled stocks with ≥ {min_sigs} Sell signals.")

else:
    # CLI Terminal Execution
    def main():
        stocks_to_scan = INDIAN_STOCKS
        print("\n" + "="*85)
        print("          NSE INDIA - COILED INDICATOR STRATEGY SCREENER")
        print("="*85)
        print(f"[*] Scanning {len(stocks_to_scan)} NSE stocks (Filtering for Coiled & >= 20 Signals)...\n")

        buys_list, sells_list = run_screener(stocks_to_scan, min_signals=20)

        # ANSI Colors
        GREEN = '\033[92m'
        RED = '\033[91m'
        RESET = '\033[0m'

        # --- BUY TABLE ---
        print("\n" + "="*85)
        print("🟢 BUY CANDIDATES (COILED & >= 20 Signals)")
        print("="*85)
        header_b = f"{'Rank':<5}{'Stock':<14}{'CMP ($)':<12}{'Buy Signals':<16}{'RSI(1H)':<10}{'RSI(D)':<10}{'RSI(W)':<10}"
        print(header_b)
        print("-" * len(header_b))

        if not buys_list:
            print("[-] No stocks passed the BUY screener.")
        else:
            for rank, item in enumerate(buys_list, start=1):
                sigs = f"{item['Buy Signals']}/{item['Total Indicators']}"
                r1h = f"{item['RSI (1H)']:.2f}" if not pd.isna(item['RSI (1H)']) else "N/A"
                rd = f"{item['RSI (Daily)']:.2f}" if not pd.isna(item['RSI (Daily)']) else "N/A"
                rw = f"{item['RSI (Weekly)']:.2f}" if not pd.isna(item['RSI (Weekly)']) else "N/A"
                
                row_str = f"{rank:<5}{item['Ticker']:<14}₹{item['Price (₹)']:<11.2f}{sigs:<16}{r1h:<10}{rd:<10}{rw:<10}"
                
                # Apply Green color if signals >= 20
                if item['Buy Signals'] >= 20:
                    print(f"{GREEN}{row_str}{RESET}")
                else:
                    print(row_str)

        # --- SELL TABLE ---
        print("\n\n" + "="*85)
        print("🔴 SELL CANDIDATES (COILED & >= 20 Signals)")
        print("="*85)
        header_s = f"{'Rank':<5}{'Stock':<14}{'CMP ($)':<12}{'Sell Signals':<16}{'RSI(1H)':<10}{'RSI(D)':<10}{'RSI(W)':<10}"
        print(header_s)
        print("-" * len(header_s))

        if not sells_list:
            print("[-] No stocks passed the SELL screener.")
        else:
            for rank, item in enumerate(sells_list, start=1):
                sigs = f"{item['Sell Signals']}/{item['Total Indicators']}"
                r1h = f"{item['RSI (1H)']:.2f}" if not pd.isna(item['RSI (1H)']) else "N/A"
                rd = f"{item['RSI (Daily)']:.2f}" if not pd.isna(item['RSI (Daily)']) else "N/A"
                rw = f"{item['RSI (Weekly)']:.2f}" if not pd.isna(item['RSI (Weekly)']) else "N/A"
                
                row_str = f"{rank:<5}{item['Ticker']:<14}₹{item['Price (₹)']:<11.2f}{sigs:<16}{r1h:<10}{rd:<10}{rw:<10}"
                
                # Apply Red color if signals >= 20
                if item['Sell Signals'] >= 20:
                    print(f"{RED}{row_str}{RESET}")
                else:
                    print(row_str)

        # Save to single CSV
        combined_file = save_results_to_csv(buys_list, sells_list)
        print("\n" + "="*85)
        if combined_file: print(f"[+] Combined detailed CSV saved: {combined_file}")
        print("="*85 + "\n")

    if __name__ == '__main__':
        main()
