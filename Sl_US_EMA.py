import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from tabulate import tabulate
import pytz

# Safely import streamlit and runtime checking
try:
    import streamlit as st
    from streamlit import runtime
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False

def is_streamlit_running():
    """Returns True if the script is running inside a Streamlit web server."""
    if HAS_STREAMLIT:
        return runtime.exists()
    return False

def get_time_adjusted_rvol(current_vol, avg_vol):
    eastern = pytz.timezone('US/Eastern')
    now_et = datetime.now(eastern)
        
    # FIX: If it is Saturday (5) or Sunday (6), return the standard end-of-day calculation
    if now_et.weekday() >= 5:
        return current_vol / avg_vol if avg_vol > 0 else 0
        
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
    if now_et < market_open: 
        return 0
            
    if now_et >= market_close: 
        return current_vol / avg_vol if avg_vol > 0 else 0
        
    elapsed_minutes = (now_et - market_open).total_seconds() / 60.0
    fraction_of_day = elapsed_minutes / 390.0
    pro_rated_avg_vol = avg_vol * fraction_of_day
        
    return current_vol / pro_rated_avg_vol if pro_rated_avg_vol > 0 else 0

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    for i in range(period, len(series)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def screen_stocks(tickers: list[str]) -> pd.DataFrame:
    results = []
    
    # 1. Setup Cache Directory
    cache_dir = "EMAOutput"
    os.makedirs(cache_dir, exist_ok=True)
    today = datetime.now().date()
    
    total_tickers = len(tickers)
    is_web = is_streamlit_running()
    
    # Initialize appropriate UI elements based on environment
    if is_web:
        progress_bar = st.progress(0, text=f"Initializing scan for {total_tickers} tickers...")
    else:
        print(f"Scanning {total_tickers} tickers. Using cache directory: '{cache_dir}/'")
    
    for i, ticker in enumerate(tickers):
        if is_web:
            progress_bar.progress((i + 1) / total_tickers, text=f"Scanning {ticker} ({i+1}/{total_tickers})")
            
        cache_file = os.path.join(cache_dir, f"{ticker}.csv")
        df = pd.DataFrame()
        
        try:
            # 2. Cache Retrieval Logic
            if os.path.exists(cache_file):
                # Check if the file was modified today
                file_date = datetime.fromtimestamp(os.path.getmtime(cache_file)).date()
                if file_date == today:
                    df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            
            if df.empty:
                stock = yf.Ticker(ticker)
                df = stock.history(period="2y")
                if not df.empty:
                    df.to_csv(cache_file)
            
            df = df.dropna()
            if len(df) < 200:
                continue

            # Calculate Moving Averages
            df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
            df['EMA_100'] = df['Close'].ewm(span=100, adjust=False).mean()
            df['EMA_200'] = df['Close'].ewm(span=200, adjust=False).mean()

            df['Vol_SMA_20'] = df['Volume'].rolling(window=20).mean()
            df['RSI_14'] = calculate_rsi(df['Close'], period=14)

            # Volatility Contraction / Coil Gauge
            df['Coil_Range_10D'] = ((df['High'].rolling(10).max() - df['Low'].rolling(10).min()) / df['Close']) * 100

            curr = df.iloc[-1]
            prev = df.iloc[-2]

            # --- BREAKOUT LOGIC ---
            # 1. Fresh Crossover Detection (Lookback: Last 5 Sessions)
            cross_100 = (df['EMA_50'] > df['EMA_100']) & (df['EMA_50'].shift(1) <= df['EMA_100'].shift(1))
            cross_200 = (df['EMA_50'] > df['EMA_200']) & (df['EMA_50'].shift(1) <= df['EMA_200'].shift(1))
            
            fresh_cross_100 = cross_100.iloc[-5:].any()
            fresh_cross_200 = cross_200.iloc[-5:].any()

            # 2. Price Requirement (Price above all 3 EMAs)
            price_above_emas = (curr['Close'] > curr['EMA_50']) and (curr['Close'] > curr['EMA_100']) and (curr['Close'] > curr['EMA_200'])

            # 3. Specific Crossover Conditions (The "OR" Logic)
            # Cond A: 50 crossed 100 recently AND 50 is above 200
            cond_a = fresh_cross_100 and (curr['EMA_50'] > curr['EMA_200'])
            # Cond B: 50 crossed 200 recently AND 50 is above 100
            cond_b = fresh_cross_200 and (curr['EMA_50'] > curr['EMA_100'])

            ema_condition = price_above_emas and (cond_a or cond_b)

            prior_5d_high = df['High'].iloc[-6:-1].max()
            price_breakout = curr['Close'] > prior_5d_high
            trigger = price_breakout or (prev['Close'] > df['High'].iloc[-7:-2].max())
            
            # Use the new time-adjusted volume calculation here
            vol_ratio = get_time_adjusted_rvol(curr['Volume'], curr['Vol_SMA_20'])
            
            # Apply time-adjusted ratio to the expansion filter
            vol_expansion = vol_ratio >= 0.4
            is_coiled = curr['Coil_Range_10D'] <= 15.0
            rsi_valid = 50 <= curr['RSI_14'] <= 80

            if ema_condition and is_coiled and rsi_valid:
                dist_ema50 = ((curr['Close'] - curr['EMA_50']) / curr['EMA_50']) * 100

                # Probability Scoring
                vol_score = min(35, (vol_ratio / 2.0) * 35)
                coil_score = max(0, 35 - (curr['Coil_Range_10D'] * 3.5))
                rsi_distance = abs(curr['RSI_14'] - 68)
                rsi_score = max(0, 30 - (rsi_distance * 2))

                results.append({
                    "Ticker": ticker,
                    "Price ($)": f"{curr['Close']:.2f}",
                    "Score": round(vol_score + coil_score + rsi_score, 1),
                    "Vol Ratio": f"{vol_ratio:.2f}x",
                    "10D Coil": f"{curr['Coil_Range_10D']:.1f}%",
                    "RSI (14)": f"{curr['RSI_14']:.1f}",
                    "50 EMA Dist": f"+{dist_ema50:.1f}%"
                })

        except Exception as e:
            continue
            
    if is_web:
        progress_bar.empty()

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    return df_results


def run_terminal(watchlist):
    """Executes the standard print-to-terminal logic."""
    ranked_table = screen_stocks(watchlist)
    if not ranked_table.empty:
        print("\n=== TOP HIGH-PROBABILITY BREAKOUT CANDIDATES ===")
        print(tabulate(ranked_table, headers="keys", tablefmt="fancy_grid", showindex=False))
    else:
        print("\nNo stocks currently match the breakout criteria.")

def run_streamlit(watchlist):
    """Executes the Streamlit GUI logic."""
    st.set_page_config(page_title="US Breakout Screener", layout="wide")
    st.title("📈 US High-Probability Breakout Screener")
    st.markdown("Scans a defined US watchlist for EMA Crossovers, Volatility Contraction, and RSI Momentum.")

    if st.button("Run Breakout Scan", type="primary"):
        ranked_table = screen_stocks(watchlist)
        if not ranked_table.empty:
            st.success(f"Successfully found {len(ranked_table)} high-probability candidates!")
            st.dataframe(ranked_table, use_container_width=True, hide_index=True)
        else:
            st.warning("No stocks currently match the breakout criteria.")

if __name__ == "__main__":
    watchlist = [
        "A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACI", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
        "AEP", "AFL", "AFRM", "AGIO", "AIG", "AIZ", "AJG", "AKAM", "ALAB", "ALB", "ALGN", "ALK", "ALL", "ALLE", "ALLY",
        "ALNY", "AMAT", "AMC", "AMD", "AME", "AMGN", "AMN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD",
        "APH", "APP", "APTV", "ARE", "ARM", "ASML", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZN", "AZO", "BA",
        "BABA", "BAC", "BALL", "BAX", "BBAI", "BBY", "BDX", "BEN", "BIDU", "BIIB", "BILI", "BIO", "BJ", "BKNG", "BKR",
        "BLK", "BLMN", "BMRN", "BMY", "BR", "BRO", "BSX", "BWA", "BX", "BXP", "C", "CAH", "CAKE", "CARR", "CAT",
        "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF", "CFG", "CHD", "CHH", "CHRW", "CHTR",
        "CI", "CINF", "CINT", "CL", "CLX", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO",
        "COST", "COTY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSX", "CTAS", "CTRA", "CTSH", "CTVA", "CUBE",
        "CVS", "CVX", "CZR", "D", "DAL", "DASH", "DD", "DDOG", "DE", "DG", "DHR", "DIS", "DKNG", "DLR", "DLTR",
        "DOCU", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EAT", "EBAY", "ECL",
        "ED", "EFX", "EL", "ELV", "EMN", "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN",
        "EW", "EXC", "EXPD", "EXPE", "EXR", "F", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FIS", "FITB", "FLS",
        "FMC", "FOX", "FOXA", "FRSH", "FRT", "FSLR", "FTI", "FTNT", "GD", "GE", "GEHC", "GEN", "GEV", "GGG", "GILD",
        "GIS", "GL", "GLD", "GLPG", "GLW", "GM", "GME", "GNRC", "GOOG", "GOOGL", "GPC", "GRMN", "GS", "GWW", "H",
        "HAL", "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HON", "HOOD", "HP", "HPE", "HPQ", "HST",
        "HSY", "HUBB", "HUM", "HUN", "HWM", "IBM", "ICE", "IDXX", "IFF", "ILMN", "INCY", "INTC", "INTU", "INVH", "IP",
        "IR", "IRM", "ISRG", "ITW", "IVZ", "IWM", "JBHT", "JCI", "JD", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP",
        "KEY", "KEYS", "KFY", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KNX", "KO", "KR", "L", "LCID", "LDOS",
        "LHX", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA", "MAN",
        "MAR", "MAS", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META", "MGM", "MKC", "MKTX",
        "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MRVL", "MS", "MSCI",
        "MSFT", "MSTR", "MTB", "MTCH", "MTD", "MU", "NBR", "NCLH", "NDAQ", "NEE", "NEM", "NET", "NFLX", "NKE", "NMIH",
        "NNN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NU", "NUE", "NVDA", "NWS", "NWSA", "NXPI", "O",
        "ODFL", "OGN", "OKE", "OKTA", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PBI", "PCAR", "PDD",
        "PEG", "PENN", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PKG", "PKI", "PLD", "PLTR", "PM", "PNC", "PNR",
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
        "ZS", "ZTS"
    ]

    # Dynamically run the correct interface mode
    if is_streamlit_running():
        run_streamlit(watchlist)
    else:
        run_terminal(watchlist)
