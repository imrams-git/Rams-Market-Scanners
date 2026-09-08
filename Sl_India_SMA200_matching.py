import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
from typing import List
import warnings
import os
import sys
import time
import pytz
import re
from io import StringIO
from google import genai

# Safely import streamlit without forcing browser execution context
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

# Suppress warnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"

os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6JcW2HiHVhlfeHLXcx_Xs5xR8DSXP3WgDiIw7a_wiCuEA"


# ==========================================================
# PART 1: NIFTY 500 SMA & RSI SCANNER COMPONENTS
# ==========================================================

def calculate_rsi(series, period=14):
    """Calculates Wilder's Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def fetch_nifty_500_tickers():
    url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))
        return [f"{sym}.NS" for sym in df['Symbol'].tolist()]
    except Exception as e:
        print(f"Error fetching Nifty 500: {e}. Using fallback list.")
        return ["RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS"]

def scan_stocks(progress_callback=None):
    if not progress_callback:
        print("Fetching Nifty 500 stock list...")
        
    tickers = fetch_nifty_500_tickers()
    total_tickers = len(tickers)
    
    if not progress_callback:
        print(f"Successfully loaded {total_tickers} symbols. Starting scan...\n")
    
    successful_matches = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    
    # Create local output cache directory
    os.makedirs("output", exist_ok=True)
    today_str = datetime.now().strftime("%Y-%m-%d")
    filename = f"output/nifty500_data_{today_str}.csv"
    
    # Check for local daily CSV cache to prevent re-downloading and open-file crashes
    if os.path.exists(filename):
        if not progress_callback:
            print(f"Local data found for {today_str}. Loading from disk...")
        df_bulk = pd.read_csv(filename, header=[0, 1], index_col=0, parse_dates=True)
    else:
        if not progress_callback:
            print("No local data found. Initiating BATCH download for all symbols...")
        # Download all 500 tickers at once in a single bulk request
        df_bulk = yf.download(tickers, start=start_date, end=end_date, progress=False, threads=True)
        df_bulk.to_csv(filename)
        if not progress_callback:
            print(f"Data saved to {filename} for reuse today.")

    if not progress_callback:
        print("\nDownload/Load complete. Running mathematical filters...")
    
    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, total_tickers, ticker.replace(".NS", ""))
            
        try:
            # Check if data exists in bulk dataframe
            if 'Close' not in df_bulk.columns or ticker not in df_bulk['Close'].columns:
                continue
                
            df_daily = pd.DataFrame({
                'Open': df_bulk['Open'][ticker],
                'Close': df_bulk['Close'][ticker],
                'Volume': df_bulk['Volume'][ticker]
            }).dropna()
            
            if df_daily.empty or len(df_daily) < 200:
                continue

            # --- STAGE 1: Price ---
            current_price = float(df_daily['Close'].iloc[-1])
            prev_price = float(df_daily['Close'].iloc[-2])
            
            if current_price <= 50 or current_price <= prev_price: 
                continue

            # --- STAGE 2: Volume & Liquidity ---
            df_daily['Vol_Avg'] = df_daily['Volume'].rolling(window=20).mean()
            current_vol = float(df_daily['Volume'].iloc[-1])
            avg_vol = float(df_daily['Vol_Avg'].iloc[-1])
            
            if (avg_vol * current_price) < 250000000: # Rs 25Cr minimum turnover
                continue

            now = datetime.now()
            market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
            market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
            
            if market_open < now < market_close:
                elapsed = (now - market_open).total_seconds() / 60
                expected_vol = avg_vol * (elapsed / 375)
            else:
                expected_vol = avg_vol

            volume_threshold = expected_vol * 0.5
            if current_vol <= volume_threshold:
                continue

            # --- STAGE 3: 200 SMA ---
            df_daily['SMA_200'] = df_daily['Close'].rolling(window=200).mean()
            current_sma200 = float(df_daily['SMA_200'].iloc[-1])
            
            lower_bound = current_sma200 * 1.0
            upper_bound = current_sma200 * 1.1
            if not (lower_bound <= current_price <= upper_bound):
                continue

            # --- STAGE 4: RSI ---
            df_daily['RSI'] = calculate_rsi(df_daily['Close'], period=14)
            daily_rsi = float(df_daily['RSI'].iloc[-1])
            if daily_rsi < 50:
                continue

            df_weekly = df_daily['Close'].resample('W-SUN').last().to_frame()
            df_weekly['RSI'] = calculate_rsi(df_weekly['Close'], period=14)
            weekly_rsi = float(df_weekly['RSI'].iloc[-1])
            if weekly_rsi < 50:
                continue

            df_monthly = df_daily['Close'].resample('ME').last().to_frame()
            df_monthly['RSI'] = calculate_rsi(df_monthly['Close'], period=14)
            monthly_rsi = float(df_monthly['RSI'].iloc[-1])
            if monthly_rsi < 50:
                continue

            # --- STAGE 5: Fundamentals ---
            ticker_obj = yf.Ticker(ticker)
            info = ticker_obj.info
            
            market_cap = info.get('marketCap', 0)
            if market_cap is None or market_cap < 1000000000:
                continue
                
            calendar = ticker_obj.calendar
            skip_earnings = False
            if calendar is not None and 'Earnings Date' in calendar:
                earnings_dates = calendar['Earnings Date']
                if earnings_dates:
                    next_earnings = earnings_dates[0]
                    if isinstance(next_earnings, datetime):
                        next_earnings = next_earnings.date()
                    days = (next_earnings - datetime.now().date()).days
                    if 0 <= days <= 30:
                        skip_earnings = True
            
            if skip_earnings:
                continue

            # --- Calculate % Difference from SMA200 ---
            pct_diff = ((current_price - current_sma200) / current_sma200) * 100

            # --- Success ---
            clean_ticker = ticker.replace(".NS", "")
            successful_matches.append({
                "Ticker": clean_ticker,
                "Price": round(current_price, 2),
                "200 SMA": round(current_sma200, 2),
                "%diff": round(pct_diff, 2),
                "D-RSI": round(daily_rsi, 2),
                "W-RSI": round(weekly_rsi, 2),
                "M-RSI": round(monthly_rsi, 2)
            })
            
            if not progress_callback:
                print(f"{clean_ticker},",end="")

        except Exception as e:
            if not progress_callback:
                print(f" [!] Crash on {ticker}: {type(e).__name__} - {e}")
            continue
            
    return successful_matches


# ==========================================================
# PART 2: IMBALANCE CHECKER & GEMINI AI COMPONENTS
# ==========================================================

def colorize_terminal_output(text: str) -> str:
    colors = {
        "[GREEN]": "\033[92m", "[RED]": "\033[91m", "[YELLOW]": "\033[93m",
        "[CYAN]": "\033[96m", "[BLUE]": "\033[94m", "[RESET]": "\033[0m"
    }
    colored_text = text
    for tag, ansi_code in colors.items():
        colored_text = colored_text.replace(tag, ansi_code)
    colored_text = re.sub(r'(\033\[[0-9;]*m[^\n]+)', r'\1\033[0m', colored_text)
    return colored_text

def strip_tags_for_file(text: str) -> str:
    return re.sub(r'\[(GREEN|RED|YELLOW|CYAN|BLUE|RESET)\]', '', text)


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
        squeeze_multiplier: float = 1.5
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
        
        # Create a persistent network session to prevent Mac OS file limit crashes
        self.session = requests.Session()
        self.session.headers.update({'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)'})

    def get_time_adjusted_rvol(self, current_vol, avg_vol):
        # INDIAN MARKET HOURS: 9:15 AM to 3:30 PM IST (375 total minutes)
        ist = pytz.timezone('Asia/Kolkata')
        now_ist = datetime.now(ist)
        
        if now_ist.weekday() >= 5:
            return current_vol / avg_vol if avg_vol > 0 else 0
        
        market_open = now_ist.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_ist.replace(hour=15, minute=30, second=0, microsecond=0)
        
        if now_ist < market_open: 
            return 0
            
        if now_ist >= market_close: 
            return current_vol / avg_vol if avg_vol > 0 else 0
        
        elapsed_minutes = (now_ist - market_open).total_seconds() / 60.0
        fraction_of_day = elapsed_minutes / 375.0
        pro_rated_avg_vol = avg_vol * fraction_of_day
        
        return current_vol / pro_rated_avg_vol if pro_rated_avg_vol > 0 else 0

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)    
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(100)

    def calculate_atr(self, df, period=14):
        high_low = df['High'] - df['Low']
        high_cp = np.abs(df['High'] - df['Close'].shift())
        low_cp = np.abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def get_batch_data(self, symbols: List[str], timeframe: str, period: str = "60d"):
        for attempt in range(3):
            try:
                df = yf.download(
                    tickers=symbols, 
                    period=period, 
                    interval=timeframe, 
                    group_by="ticker", 
                    threads=False,   
                    session=self.session,
                    progress=False
                )
                if not df.empty:
                    return df
            except Exception as e:
                print(f"⚠️ Yahoo blocked attempt {attempt + 1}. Retrying...")
                time.sleep(2 * (attempt + 1))                
        return pd.DataFrame()

    def is_coiled_squeeze(self, df_d: pd.DataFrame) -> bool:
        if len(df_d) < self.squeeze_length:
            return False
        close = df_d['Close']
        
        ma = close.rolling(window=self.squeeze_length).mean()
        std = close.rolling(window=self.squeeze_length).std()
        bb_upper = ma + (2 * std)
        bb_lower = ma - (2 * std)
        
        high_low = df_d['High'] - df_d['Low']
        high_cp = abs(df_d['High'] - close.shift())
        low_cp = abs(df_d['Low'] - close.shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=self.squeeze_length).mean()
        
        kc_upper = ma + (self.squeeze_multiplier * atr)
        kc_lower = ma - (self.squeeze_multiplier * atr)
        
        current_squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        return bool(current_squeeze)

    def analyze_symbol(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame, weekly_data: pd.DataFrame, symbol: str, n_bars: int):
        # Safely extract Intraday Data
        if isinstance(intraday_data.columns, pd.MultiIndex):
            if symbol not in intraday_data.columns.levels[0]: return []
            df = intraday_data[symbol].copy().dropna()
        else:
            df = intraday_data.copy().dropna()

        # Safely extract Daily Data
        if isinstance(daily_data.columns, pd.MultiIndex):
            if symbol not in daily_data.columns.levels[0]: return []
            df_d = daily_data[symbol].copy().dropna()
        else:
            df_d = daily_data.copy().dropna()

        # Safely extract Weekly Data
        if isinstance(weekly_data.columns, pd.MultiIndex):
            if symbol not in weekly_data.columns.levels[0]: return []
            df_w = weekly_data[symbol].copy().dropna()
        else:
            df_w = weekly_data.copy().dropna()
            
        # Calculate RSI for Daily and Weekly
        df_d['RSI'] = self.calculate_rsi(df_d['Close'], self.rsi_period)
        if len(df_w) > self.rsi_period:
            df_w['RSI'] = self.calculate_rsi(df_w['Close'], self.rsi_period)
        
        current_rsi_d = df_d['RSI'].iloc[-1] if not df_d.empty else 50
        current_rsi_w = df_w['RSI'].iloc[-1] if not df_w.empty and 'RSI' in df_w.columns else 50

        df_d['50_DMA'] = df_d['Close'].rolling(window=50).mean()
        current_50_dma = df_d['50_DMA'].iloc[-1]
        cmp_price = df['Close'].iloc[-1]
                
        df_d['AvgVol_20'] = df_d['Volume'].rolling(window=20).mean()
        rvol_d_today = self.get_time_adjusted_rvol(df_d['Volume'].iloc[-1], df_d['AvgVol_20'].iloc[-1])
        rvol_d_1_ago = df_d['Volume'].iloc[-2] / df_d['AvgVol_20'].iloc[-2]
        rvol_d_2_ago = df_d['Volume'].iloc[-3] / df_d['AvgVol_20'].iloc[-3]

        is_coiled = self.is_coiled_squeeze(df_d)

        is_steady_bull = (
            (cmp_price >= current_50_dma) and
            (rvol_d_1_ago >= 1.0) and 
            (rvol_d_2_ago >= 1.0)
        )
        is_steady_bear = (
            (cmp_price <= current_50_dma) and
            (rvol_d_1_ago >= 1.0) and 
            (rvol_d_2_ago >= 1.0)
        )

        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['ATR'] = self.calculate_atr(df, self.atr_period)
        df["AvgVol"] = df["Volume"].rolling(self.length).mean()
        df["RVOL_H"] = df["Volume"] / df["AvgVol"] 

        current_rsi_h = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        current_rvol_h = df['RVOL_H'].iloc[-1]

        recent_bars = df.iloc[-n_bars:]
        high_vol_indices = recent_bars[recent_bars["RVOL_H"] > self.high_volume_threshold].index

        valid_levels = []
        for i in reversed(high_vol_indices):
            row = df.loc[i]
            rng = row["High"] - row["Low"]
            if rng == 0: continue

            color = "ImbLow" if row["Close"] >= row["Open"] else "ImbHigh"
            
            buy_p = (row["Close"] - row["Low"]) / rng
            future = df.loc[i:].iloc[1:]
            level = row["High"] if color == "ImbHigh" else row["Low"]
            dist = abs(level - cmp_price)
            pct_diff = (dist / cmp_price) * 100

            bars_ago = len(df) - df.index.get_loc(i) - 1
            valid_levels.append([symbol, color, round(cmp_price, 2), round(level, 2), 
                                 round(pct_diff, 2), bars_ago, round(current_rsi_h, 2), 
                                 round(current_rsi_d, 2), round(current_rsi_w, 2),
                                 round(current_rvol_h, 2), round(rvol_d_today, 2), is_coiled])
        return valid_levels

    def add_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        unique_symbols = df['Symbol'].unique()
        fund_data = {}
        
        if len(unique_symbols) > 0:
            print(f"\nFetching Fundamental Data for {len(unique_symbols)} matched symbols...")
            
        for sym in unique_symbols:
            pe_str = "N/A"
            days_to_earn = "N/A"
            try:
                ticker = yf.Ticker(sym, session=self.session)
                info = ticker.info
                pe = info.get('trailingPE') or info.get('forwardPE')
                if pe and pe > 0: pe_str = f"{pe:.1f}"
                
                calendar = ticker.calendar
                if calendar and isinstance(calendar, dict) and 'Earnings Date' in calendar:
                    dates = calendar['Earnings Date']
                    if len(dates) > 0:
                        next_date = dates[0].date() if hasattr(dates[0], 'date') else dates[0]
                        days_to_earn = (next_date - datetime.now().date()).days
            except:
                pass
            fund_data[sym] = {"PE": pe_str, "Days2Earn": days_to_earn}
            time.sleep(0.5)
            
        df['PE'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('PE', 'N/A'))
        df['Days2Earn'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('Days2Earn', 'N/A'))
        return df

    def print_unified_table(self, df: pd.DataFrame):
        if df.empty: 
            print("\nNo stocks matched in Imbalance Scan.")
            return
            
        print(f"\n🎯 INDIAN MARKET TARGETED SCAN RESULTS (Unified Table)")
        header = f"{'Symbol':<14} {'TF':<4} {'Imb':<8} {'CMP':>8} {'Unbrk':>8} {'%Diff':>7} {'Ago':>4} {'RSI(H)':>6} {'RSI(D)':>6} {'RSI(W)':>6} {'RV(H)':>5} {'RV(D)':>5} {'PE':>7} {'D2Earn':>6} {'IsCoiled':>5}"
        print("=" * len(header))
        print(header)
        print("=" * len(header))
        
        prev_sym = ""
        for _, r in df.iterrows():
            # Print a subtle divider when the symbol changes
            if prev_sym and prev_sym != r['Symbol']:
                print("-" * len(header))
                
            row_color = GREEN if r['Imb'] == "ImbHigh" else RED
            print(row_color + f"{r['Symbol']:<14} {r['TF']:<4} {r['Imb']:<8} {r['CMP']:>8.2f} {r['Unbrk']:>8.2f} {r['%Diff']:>7.2f} {int(r['Ago']):>4} {r['RSI(H)']:>6.1f} {r['RSI(D)']:>6.1f} {r['RSI(W)']:>6.1f} {r['RVOL_H']:>5.1f} {r['RVOL_D']:>5.1f} {str(r['PE']):>7} {str(r['Days2Earn']):>6} {str(r['Is_Coiled']):>5}" + RESET)
            
            prev_sym = r['Symbol']
        print("=" * len(header))


    def run(self, symbols: List[str], n_bars: int = 100):
        rows = []
        batch_size = 40
        
        # Auto-append .NS if user types a raw ticker like 'RELIANCE' instead of 'RELIANCE.NS'
        formatted_symbols = [s if "." in s else f"{s}.NS" for s in symbols]
        
        print(f"\nFetching Daily Data for NSE...")
        valid_symbols = []
        daily_data_chunks = []
        
        for i in range(0, len(formatted_symbols), batch_size):
            batch = formatted_symbols[i : i + batch_size]
            batch_df = self.get_batch_data(batch, "1d", "90d")
            
            if not batch_df.empty:
                daily_data_chunks.append(batch_df)
                if isinstance(batch_df.columns, pd.MultiIndex):
                    valid_symbols.extend(list(set(batch_df.columns.get_level_values(0))))
                else:
                    valid_symbols.extend(batch)
            time.sleep(0.5)

        if not daily_data_chunks:
            print("Failed to fetch any baseline daily data.")
            return pd.DataFrame()

        full_daily_data = pd.concat(daily_data_chunks, axis=1)

        print(f"Fetching Weekly Data for RSI(W)...")
        weekly_data_chunks = []
        for i in range(0, len(valid_symbols), batch_size):
            batch = valid_symbols[i : i + batch_size]
            batch_df_w = self.get_batch_data(batch, "1wk", "2y")
            if not batch_df_w.empty:
                weekly_data_chunks.append(batch_df_w)
            time.sleep(0.5)
            
        full_weekly_data = pd.concat(weekly_data_chunks, axis=1) if weekly_data_chunks else pd.DataFrame()

        for tf_name, tf_interval in self.timeframes.items():
            print(f"Scanning {tf_name}...")
            for i in range(0, len(valid_symbols), batch_size):
                batch = valid_symbols[i : i + batch_size]
                intraday_data = self.get_batch_data(batch, tf_interval)
                if intraday_data.empty: continue
                for sym in batch:
                    results = self.analyze_symbol(intraday_data, full_daily_data, full_weekly_data, sym, n_bars)
                    for res in results:
                        rows.append([res[0], tf_name] + res[1:])
                time.sleep(0.5)

        if not rows: 
            print("No stocks matched.")
            return pd.DataFrame() 

        df = pd.DataFrame(rows, columns=["Symbol", "TF", "Imb", "CMP", "Unbrk", "%Diff", "Ago", "RSI(H)", "RSI(D)", "RSI(W)", "RVOL_H", "RVOL_D", "Is_Coiled"])
        df = df.drop_duplicates()
        df = df.sort_values(by=["Symbol", "TF", "%Diff"]).drop_duplicates(subset=["Symbol", "TF"], keep="first")
        
        top_criteria = (df['RVOL_D'] >= 1.0) & (df['RSI(H)'] >= 60) & (df['RSI(H)'] <= 70) & (df['Ago'] >= 30) & (df['Is_Coiled'])
        df['TopPick'] = top_criteria
        
        df = self.add_fundamentals(df)
        
        def filter_earnings_risk(row):
            try:
                days_to_earn = int(row['Days2Earn'])
                if 0 <= days_to_earn <= 7 and row['TopPick']:
                    return False
            except:
                pass
            return True
            
        df = df[df.apply(filter_earnings_risk, axis=1)]
        df = df.sort_values(by=["Is_Coiled", "TopPick", "RVOL_D"], ascending=False)
        df = df.drop(columns=["TopPick"])

        output_dir = 'output'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f"NSE_Pipeline_Results_{timestamp_str}.csv")
        df.to_csv(filepath, index=False)

        self.print_unified_table(df)
        print(f"\n✅ Results saved to {filepath}")

        if HAS_STREAMLIT and st.runtime.exists():
            st.success(f"Results saved to `{filepath}`")
            st.subheader("📊 Master Scan Results")
            st.dataframe(df, use_container_width=True)
            
        # ==================================================
        # AUTOMATED GEMINI INTEGRATION & LOGGING
        # ==================================================
        print("\n🤖 Sending scan data to Gemini for automated analysis...")
        try:
            client = genai.Client()
            csv_data_string = df.to_csv(index=False)

            prompt = f"""
            You are an elite quantitative trading assistant. Analyze the following scanner dataset and organize the output strictly into these four clean tiers:

            1. 🚀 The Tier 1 Structural Alpha (High Conviction Buys)
               - Requirements: Multi-timeframe confluence, Ago > 20. 
               - If IsCoiled = True, accept ANY RVOL_D. If IsCoiled = False, requires RVOL_D > 0.8.
               - Days2Earn > 15 or Days2Earn < -2.
               - All TF: Must have 'ImbHigh' signals.
            2. 🏆 The Tier 2 Structural Alpha (Buys)
               - Requirements: Ago > 5, RVOL_D > 0.5, IsCoiled = True or IsCoiled = False, Days2Earn > 15 or Days2Earn < -2.
               - All TF except one Must have 'ImbHigh' signals.
            3. 🎯 The High-Volume Momentum Plays
               - Requirements: RVOL_D > 3.0, Days2Earn > 15 or Days2Earn < -2.
            4. ⚠️ The Warnings & Rejections
               - Requirements: Low volume, structural conflicts, earnings risks, poor risk profiles, OR mixed signals.

            Formatting Rules:
            - **CRITICAL:** You MUST output Tier 4 (Warnings & Rejections) as a clean Markdown table with the following columns: | Symbol | Target | Diff % | ❌ Rejection Reason |
            - For Tiers 1, 2, and 3, use clear bullet points formatting the data like this: `**AAPL** | 🎯 Target: 155.00 | 📏 Diff: 3.33%`
            - Wrap key section headers in visual tags: [GREEN], [YELLOW], [CYAN], [RED].
            - DO NOT print color tags at the end of lines (e.g., avoid '[/CYAN]'). Keep lines clean.
            - Keep descriptions concise, punchy, and professional.
            - If a tier has no setups, simply write "No qualifying setups found at this time."
            - Note: The earnings column is named "Days2Earn". "N/A" is treated as > 15 (safe).
            
            Scan Data:
            {csv_data_string}
            """
            
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=prompt,
            )

            raw_gemini_output = response.text
            clean_log_file = strip_tags_for_file(raw_gemini_output)

            # Print Colorized Output to Terminal
            terminal_display = colorize_terminal_output(raw_gemini_output)
            print("\n" + terminal_display + "\n")

            # Render in Browser if Streamlit is active
            if HAS_STREAMLIT and st.runtime.exists():
                st.subheader("🤖 Gemini AI Quantitative Analysis")
                st.markdown(raw_gemini_output)

            ai_filepath = os.path.join(output_dir, f"Gemini_Analysis_{timestamp_str}.txt")
            with open(ai_filepath, "w", encoding="utf-8") as f:
                f.write(clean_log_file)

            print(f"✅ Clean Gemini report saved to {ai_filepath}")

        except Exception as e:
            print(f"\n❌ Error communicating with Gemini API: {e}")

        return df


# ==========================================================
# PART 3: MAIN EXECUTION CONTROLLER (PIPELINE LINK)
# ==========================================================

def main():
    if HAS_STREAMLIT and st.runtime.exists():
        st.set_page_config(page_title="Nifty 500 Advanced Pipeline", layout="wide")
        st.title("📈 End-to-End Nifty 500 Algorithmic Pipeline")
        st.write("Step 1: Scans Nifty 500 for stocks bouncing 10% above 200 SMA with confirmed multi-timeframe RSI (>50).")
        st.write("Step 2: Takes passing stocks and scans for unmitigated imbalances (30m, 1h, 4h, 1d) & coiled setups.")
        
        if st.button("🚀 Run Full Pipeline Now", type="primary"):
            
            # --- STEP 1: SMA 200 SCAN ---
            st.subheader("Step 1: Executing Nifty 500 Trend & Support Scan")
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, ticker_name):
                pct = current / total
                progress_bar.progress(pct)
                status_text.text(f"Scanning symbol {current}/{total}: {ticker_name}")
                
            matches = scan_stocks(progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if not matches:
                st.warning("0 results found in Step 1 matching current technical parameters. Pipeline halted.")
                return
                
            df_sma_results = pd.DataFrame(matches)
            df_sma_results = df_sma_results.sort_values(by="%diff", ascending=True)
            st.success(f"Step 1 Complete! Found {len(df_sma_results)} stocks bouncing off the 200 SMA.")
            st.dataframe(df_sma_results, use_container_width=True)
            
            # --- STEP 2: IMBALANCE SCAN & GEMINI ---
            st.subheader("Step 2: Identifying Imbalances & AI Analysis")
            symbols_to_scan = df_sma_results["Ticker"].tolist()
            
            with st.spinner(f"Running targeted volume/imbalance scan on {len(symbols_to_scan)} stocks..."):
                checker = VolumeAlertChecker()
                checker.run(symbols_to_scan, n_bars=100)

    else:
        # Standard Terminal Execution
        print("\n" + "="*60)
        print("🚀 STEP 1: SCANNING NIFTY 500 FOR 200 SMA BOUNCE")
        print("="*60)
        
        matches = scan_stocks()
        
        if not matches:
            print("\n0 results from Step 1. (The code is functioning perfectly, but no stocks meet all macro criteria today). Pipeline Halted.")
            return
            
        results_df = pd.DataFrame(matches)
        results_df = results_df.sort_values(by="%diff", ascending=True)
        print("\n" + "="*60)
        print(f"STEP 1 COMPLETE: FOUND {len(results_df)} MATCHES")
        print("="*60)
        print(results_df.to_string(index=False))

        # Extract tickers and feed directly into Step 2
        symbols_to_scan = results_df["Ticker"].tolist()
        
        print("\n" + "="*60)
        print(f"🚀 STEP 2: RUNNING IMBALANCE SCAN ON {len(symbols_to_scan)} STOCKS")
        print("="*60)
        
        checker = VolumeAlertChecker()
        checker.run(symbols_to_scan, n_bars=100)


if __name__ == "__main__":
    main()
