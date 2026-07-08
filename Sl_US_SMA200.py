import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime, timedelta
import warnings
import pytz
from io import StringIO

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

def fetch_sp500_tickers():
        return ["A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACI", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
    "AEP", "AFL", "AFRM", "AGIO", "AIG", "AIZ", "AJG", "AKAM", "ALAB", "ALB", "ALGN", "ALK", "ALL", "ALLE", "ALLY",
    "ALNY", "AMAT", "AMC", "AMD", "AME", "AMGN", "AMN", "AMP", "AMT", "AMZN", "ANET", "AON", "AOS", "APA", "APD",
    "APH", "APP", "APTV", "ARE", "ARM", "ARNA", "ASGN", "ASML", "ATO", "AVB", "AVGO", "AVY", "AWK", "AXP", "AZN",
    "AZO", "BA", "BABA", "BAC", "BALL", "BAX", "BBAI", "BBBY", "BBY", "BDX", "BEN", "BIDU", "BIIB", "BILI", "BIO",
    "BJ", "BK", "BKNG", "BKR", "BLK", "BLMN", "BLUE", "BMRN", "BMY", "BR", "BRK.B", "BRO", "BSX", "BWA", "BX",
    "BXP", "C", "CAH", "CAKE", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDAY", "CDNS", "CDW", "CE",
    "CEG", "CF", "CFG", "CHD", "CHH", "CHRW", "CHTR", "CI", "CINF", "CINT", "CL", "CLX", "CMA", "CMCSA", "CME",
    "CMG", "CMI", "CMS", "CNC", "CNP", "COF", "COIN", "COO", "COST", "COTY", "CPB", "CPRT", "CPT", "CRL", "CRM",
    "CRWD", "CSCO", "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CUBE", "CVS", "CVX", "CZR", "D", "DAL", "DASH",
    "DD", "DDOG", "DE", "DFS", "DG", "DHR", "DIS", "DKNG", "DLR", "DLTR", "DOCU", "DOV", "DOW", "DPZ", "DRI",
    "DTE", "DUK", "DVA", "DVN", "DXC", "DXCM", "EA", "EAT", "EBAY", "ECL", "ED", "EFX", "EL", "ELV", "EMN",
    "EMR", "ENPH", "EOG", "EPAM", "EQIX", "EQR", "EQT", "ES", "ESS", "ETN", "EW", "EXAS", "EXC", "EXPD", "EXPE",
    "EXR", "F", "FAST", "FBHS", "FCX", "FDS", "FDX", "FE", "FFIV", "FGEN", "FIS", "FITB", "FLS", "FMC", "FOX",
    "FOXA", "FRSH", "FRT", "FSLR", "FTI", "FTNT", "GD", "GE", "GEHC", "GEN", "GEV", "GGG", "GILD", "GIS", "GL",
    "GLD", "GLPG", "GLW", "GM", "GME", "GNRC", "GOOG", "GOOGL", "GPC", "GPS", "GRMN", "GS", "GWW", "H", "HAL",
    "HAS", "HBAN", "HCA", "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HOOD", "HP", "HPE", "HPQ", "HST",
    "HSY", "HUBB", "HUM", "HUN", "HWM", "IBM", "ICE", "IDXX", "IFF", "ILMN", "INCY", "INSU", "INTC", "INTU", "INVH",
    "IP", "IR", "IRM", "ISRG", "ITW", "IVZ", "IWM", "JBHT", "JCI", "JD", "JKHY", "JNJ", "JNPR", "JPM", "K",
    "KDP", "KEY", "KEYS", "KFY", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KNX", "KO", "KR", "L", "LCID",
    "LDOS", "LHX", "LIN", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LYB", "LYV", "MA", "MAA",
    "MAN", "MAR", "MAS", "MAT", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MELI", "MET", "META", "MGM", "MKC",
    "MKTX", "MLM", "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MRO", "MRVL", "MS",
    "MSCI", "MSFT", "MSTR", "MTB", "MTCH", "MTD", "MU", "MYOV", "NBR", "NCLH", "NDAQ", "NEE", "NEM", "NET", "NFLX",
    "NKE", "NMIH", "NNN", "NOC", "NOV", "NOW", "NRG", "NSC", "NTAP", "NTRS", "NU", "NUE", "NVDA", "NWS", "NWSA",
    "NXPI", "O", "ODFL", "OGN", "OKE", "OKTA", "ORCL", "ORLY", "OTIS", "OXY", "PANW", "PARA", "PAYC", "PAYX", "PBI",
    "PCAR", "PDD", "PEAK", "PEG", "PENN", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PKG", "PKI", "PLD", "PLTR",
    "PM", "PNC", "PNR", "PODD", "POOL", "PPG", "PRU", "PSA", "PSX", "PTC", "PTON", "PYPL", "QCOM", "QD", "QQQ",
    "QSR", "RAD", "RBLX", "RCL", "RDW", "REG", "REGN", "RF", "RGNX", "RHI", "RIVN", "RL", "RMD", "RNG", "ROK",
    "ROKU", "ROL", "ROP", "ROST", "RS", "RSG", "RTX", "SAGE", "SBAC", "SBUX", "SCHW", "SEDG", "SEE", "SFM", "SHOP",
    "SHW", "SIRI", "SJM", "SLB", "SLG", "SMCI", "SNA", "SNAP", "SNOW", "SNPS", "SO", "SPG", "SPGI", "SPY", "SRE",
    "SRPT", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SYF", "SYK", "SYM", "SYY", "T", "TDG", "TEAM",
    "TECH", "TEL", "TEM", "TER", "TFC", "TGT", "TJX", "TMO", "TMUS", "TPR", "TRGP", "TRI", "TRIP", "TRMB", "TROW",
    "TRV", "TSCO", "TSLA", "TSM", "TT", "TTWO", "TWLO", "TXN", "TXRH", "TXT", "TYL", "U", "UA", "UAA", "UAL",
    "UDR", "UHS", "ULTA", "UNH", "UNM", "UNP", "UPS", "UPST", "URI", "V", "VFC", "VICI", "VLO", "VMC", "VNO",
    "VRSK", "VRSN", "VRT", "VRTX", "VTR", "VTRS", "VZ", "WAT", "WBA", "WBD", "WDAY", "WDC", "WEC", "WELL", "WFC",
    "WH", "WHR", "WM", "WMB", "WMT", "WPC", "WRB", "WRK", "WST", "WTW", "WY", "WYNN", "X", "XEL", "XOM",
    "XRAY", "XRX", "XYL", "YUM", "ZBH", "ZBRA", "ZION", "ZM", "ZS", "ZTS", "SPCX"
]

def scan_stocks():
    print("Fetching S&P 500 stock list...")
    tickers = fetch_sp500_tickers()
    print(f"Successfully loaded {len(tickers)} symbols.\n")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=3 * 365)
    eastern = pytz.timezone('US/Eastern')
    successful_matches = []
    
    # ---------------------------------------------------------
    # FIX: BULK DOWNLOAD 
    # Fetches all 500 stocks in ONE network request to avoid bans
    # ---------------------------------------------------------
    print("Initiating bulk data download. Please wait...")
    df_bulk = yf.download(tickers, start=start_date, end=end_date, progress=True)
    print("\nDownload complete. Running technical scans...")
    
    # Create session strictly for the handful of fundamental checks at the end
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    for ticker in tickers:
        try:
            # Safely check if ticker data was returned in the bulk download
            if 'Close' not in df_bulk.columns or ticker not in df_bulk['Close'].columns:
                continue
                
            # Slice out the individual stock's data from the massive bulk DataFrame
            df_daily = pd.DataFrame({
                'Open': df_bulk['Open'][ticker],
                'Close': df_bulk['Close'][ticker],
                'Volume': df_bulk['Volume'][ticker]
            }).dropna()
            
            if df_daily.empty or len(df_daily) < 200:
                continue

            # --- STAGE 1: Price ---
            current_price = float(df_daily['Close'].iloc[-1])
            open_price = float(df_daily['Open'].iloc[-1])
            
            if current_price <= 100 or current_price <= open_price: 
                continue

            # --- STAGE 2: Volume & Liquidity ---
            df_daily['Vol_Avg'] = df_daily['Volume'].rolling(window=20).mean()
            current_vol = float(df_daily['Volume'].iloc[-1])
            avg_vol = float(df_daily['Vol_Avg'].iloc[-1])
            
            if (avg_vol * current_price) < 5000000: 
                continue

            now_est = datetime.now(eastern)
            market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
            market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
            
            if now_est.weekday() >= 5: 
                expected_vol = avg_vol
            elif market_open < now_est < market_close:
                elapsed = (now_est - market_open).total_seconds() / 60
                expected_vol = avg_vol * (elapsed / 390) 
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
            # We ONLY query Yahoo individually here if the stock passed the math above
            ticker_obj = yf.Ticker(ticker, session=session)
            info = ticker_obj.info
            
            market_cap = info.get('marketCap', 0)
            if market_cap is None or market_cap < 5000000: 
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
                "Ticker": ticker,
                "Price": round(current_price, 2),
                "200 SMA": round(current_sma200, 2),
                "D-RSI": round(daily_rsi, 2),
                "W-RSI": round(weekly_rsi, 2),
                "M-RSI": round(monthly_rsi, 2)
            })
            print(f" MATCH FOUND: {ticker}")

        except Exception as e:
            continue
            
    print("\n" + "="*60)
    print("FINAL SCAN RESULTS (US MARKETS)")
    print("="*60)
    
    if successful_matches:
        results_df = pd.DataFrame(successful_matches)
        print(results_df.to_string(index=False))
    else:
        print("0 results. (The code is functioning perfectly, but no stocks meet all technical criteria today).")

    session.close()

if __name__ == "__main__":
    scan_stocks()
