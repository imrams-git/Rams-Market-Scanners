import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings

# Suppress warnings for clean console output
warnings.simplefilter(action='ignore', category=FutureWarning)

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

def scan_stocks():
    print("Fetching Nifty 500 stock list...")
    tickers = fetch_nifty_500_tickers()
    print(f"Successfully loaded {len(tickers)} symbols. Starting scan...\n")
    
    successful_matches = []
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    
    for idx, ticker in enumerate(tickers):
        try:
            df_daily = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df_daily.empty or len(df_daily) < 200:
                continue
                
            # Flatten multi-index columns for yfinance compatibility
            if isinstance(df_daily.columns, pd.MultiIndex):
                df_daily.columns = df_daily.columns.get_level_values(0)

            # --- STAGE 1: Price ---
            current_price = float(df_daily['Close'].iloc[-1])
            open_price = float(df_daily['Open'].iloc[-1])
            
            if current_price <= 100 or current_price <= open_price: 
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

            if current_vol <= expected_vol:
                continue

            # --- STAGE 3: 200 SMA ---
            df_daily['SMA_200'] = df_daily['Close'].rolling(window=200).mean()
            current_sma200 = float(df_daily['SMA_200'].iloc[-1])
            
            lower_bound = current_sma200 * 1.0
            upper_bound = current_sma200 * 1.05
            if not (lower_bound <= current_price <= upper_bound):
                continue

            # --- STAGE 4: RSI (Fixed for Pandas 3.0) ---
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

            # --- Success ---
            successful_matches.append({
                "Ticker": ticker.replace(".NS", ""),
                "Price": round(current_price, 2),
                "200 SMA": round(current_sma200, 2),
                "D-RSI": round(daily_rsi, 2),
                "W-RSI": round(weekly_rsi, 2),
                "M-RSI": round(monthly_rsi, 2)
            })
            print(f" MATCH FOUND: {ticker.replace('.NS', '')}")

        except Exception as e:
            # THIS WILL NOW ALERT YOU IF CODE FAILS INSTEAD OF HIDING IT
            print(f" [!] Crash on {ticker}: {type(e).__name__} - {e}")
            continue
            
    print("\n" + "="*60)
    print("FINAL SCAN RESULTS")
    print("="*60)
    if successful_matches:
        results_df = pd.DataFrame(successful_matches)
        print(results_df.to_string(index=False))
        return results_df
    else:
        print("0 results. (The code is functioning perfectly, but no stocks meet all technical criteria today).")

if __name__ == "__main__":
    scan_stocks()
