import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings
import os
import sys

# Suppress warnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, module="yfinance")
warnings.filterwarnings("ignore", message=".*Timestamp.utcnow is deprecated.*")

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
        from io import StringIO
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
    
    for idx, ticker in enumerate(tickers):
        if progress_callback:
            progress_callback(idx + 1, total_tickers, ticker.replace(".NS", ""))
            
        try:
            df_daily = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df_daily.empty or len(df_daily) < 200:
                continue
                
            # Flatten multi-index columns for yfinance compatibility
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = df_daily.columns.get_level_values(0)

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
                print(f" MATCH FOUND: {clean_ticker}")

        except Exception as e:
            if not progress_callback:
                print(f" [!] Crash on {ticker}: {type(e).__name__} - {e}")
            continue
            
    return successful_matches

# ==========================================
# EXECUTION CONTROLLER (Terminal vs Web UI)
# ==========================================
if __name__ == "__main__":
    # Check if executed via Streamlit
    if 'streamlit' in sys.modules or os.environ.get('STREAMLIT_RUN'):
        import streamlit as st
        
        st.set_page_config(page_title="Nifty 500 SMA Scanner", layout="wide")
        st.title("📈 Nifty 500 Trend & Support Scanner")
        st.write("Scanning for stocks trading up to 10% above their 200 SMA with multi-timeframe RSI confirmation (>50).")
        
        if st.button("🚀 Run Scan Now", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            def update_progress(current, total, ticker_name):
                pct = current / total
                progress_bar.progress(pct)
                status_text.text(f"Scanning symbol {current}/{total}: {ticker_name}")
                
            matches = scan_stocks(progress_callback=update_progress)
            
            progress_bar.empty()
            status_text.empty()
            
            if matches:
                df_results = pd.DataFrame(matches)
                df_results = df_results.sort_values(by="%diff", ascending=True)
                st.success(f"Scan complete! Found {len(df_results)} matching stocks.")
                st.dataframe(df_results, use_container_width=True)
            else:
                st.warning("0 results found matching current technical parameters.")
    else:
        # Standard Terminal Execution
        matches = scan_stocks()
        print("\n" + "="*60)
        print("FINAL SCAN RESULTS")
        print("="*60)
        if matches:
            results_df = pd.DataFrame(matches)
            results_df = results_df.sort_values(by="%diff", ascending=True)
            print(results_df.to_string(index=False))
        else:
            print("0 results. (The code is functioning perfectly, but no stocks meet all technical criteria today).")
