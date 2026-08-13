#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║         US STOCK PROFIT OPPORTUNITY ANALYZER             ║
║  Scores stocks via Sector-Normalized Z-Scores (Quant)    ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import warnings
import concurrent.futures
warnings.filterwarnings("ignore")

try:
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from tabulate import tabulate
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    print("Missing dependencies. Run:\n  pip install yfinance pandas numpy tabulate colorama streamlit")
    sys.exit(1)

# ─────────────────────────────────────────────
#  STOCK UNIVERSE
# ─────────────────────────────────────────────
US_SECTORS = {
{
    "Healthcare": [
        "A",
        "ABBV",
        "ABT",
        "AGIO",
        "ALGN",
        "ALNY",
        "AMGN",
        "AMN",
        "AZN",
        "BAX",
        "BDX",
        "BIIB",
        "BIO",
        "BMRN",
        "BMY",
        "BSX",
        "CAH",
        "CI",
        "CNC",
        "COO",
        "CRL",
        "CVS",
        "DHR",
        "DVA",
        "DXCM",
        "ELV",
        "EW",
        "GEHC",
        "GILD",
        "HCA",
        "HUM",
        "IDXX",
        "ILMN",
        "INCY",
        "ISRG",
        "JNJ",
        "LLY",
        "MCK",
        "MDT",
        "MOH",
        "MRK",
        "MRNA",
        "MTD",
        "OGN",
        "PFE",
        "PODD",
        "REGN",
        "RGNX",
        "RMD",
        "SRPT",
        "STE",
        "SYK",
        "TECH",
        "TEM",
        "TMO",
        "UHS",
        "UNH",
        "VRTX",
        "VTRS",
        "WAT",
        "WST",
        "XRAY",
        "ZBH",
        "ZTS"
    ],
    "Industrials": [
        "AAL",
        "ALK",
        "ALLE",
        "AME",
        "AOS",
        "BA",
        "CARR",
        "CAT",
        "CHRW",
        "CMI",
        "CPRT",
        "CSX",
        "CTAS",
        "DAL",
        "DE",
        "DOV",
        "EFX",
        "EMR",
        "ETN",
        "EXPD",
        "FAST",
        "FDX",
        "FLS",
        "GD",
        "GE",
        "GEV",
        "GGG",
        "GNRC",
        "GWW",
        "HII",
        "HON",
        "HUBB",
        "HWM",
        "IR",
        "ITW",
        "JBHT",
        "JCI",
        "KFY",
        "KNX",
        "LHX",
        "LMT",
        "LUV",
        "MAN",
        "MAS",
        "MMM",
        "NOC",
        "NSC",
        "ODFL",
        "OTIS",
        "PBI",
        "PCAR",
        "PH",
        "PNR",
        "POOL",
        "RDW",
        "RHI",
        "ROK",
        "RSG",
        "RTX",
        "SNA",
        "SWK",
        "SYM",
        "TDG",
        "TRI",
        "TT",
        "TXT",
        "UAL",
        "UNP",
        "UPS",
        "URI",
        "VRSK",
        "VRT",
        "WM",
        "XRX",
        "XYL",
        "SPCX"
    ],
    "Consumer Cyclical": [
        "AAP",
        "ABNB",
        "AMZN",
        "APTV",
        "AVY",
        "AZO",
        "BABA",
        "BALL",
        "BBY",
        "BKNG",
        "BLMN",
        "BWA",
        "CAKE",
        "CCL",
        "CHH",
        "CMG",
        "CZR",
        "DASH",
        "DKNG",
        "DPZ",
        "DRI",
        "EAT",
        "EBAY",
        "EXPE",
        "F",
        "GM",
        "GME",
        "GPC",
        "H",
        "HAS",
        "HD",
        "HLT",
        "IP",
        "JD",
        "KMX",
        "LCID",
        "LOW",
        "LULU",
        "LVS",
        "MAR",
        "MAT",
        "MCD",
        "MELI",
        "MGM",
        "NCLH",
        "NKE",
        "ORLY",
        "PDD",
        "PENN",
        "PKG",
        "PTON",
        "QSR",
        "RCL",
        "RIVN",
        "RL",
        "ROL",
        "ROST",
        "SBUX",
        "TJX",
        "TPR",
        "TRIP",
        "TSCO",
        "TSLA",
        "TXRH",
        "UA",
        "UAA",
        "ULTA",
        "VFC",
        "WH",
        "WHR",
        "WYNN",
        "YUM"
    ],
    "Technology": [
        "AAPL",
        "ACN",
        "ADBE",
        "ADI",
        "ADP",
        "ADSK",
        "AKAM",
        "ALAB",
        "AMAT",
        "AMD",
        "ANET",
        "APH",
        "ARM",
        "ASML",
        "AVGO",
        "BBAI",
        "BR",
        "CDNS",
        "CDW",
        "CINT",
        "CRM",
        "CRWD",
        "CSCO",
        "CTSH",
        "DDOG",
        "DOCU",
        "DXC",
        "ENPH",
        "EPAM",
        "FFIV",
        "FIS",
        "FRSH",
        "FSLR",
        "FTNT",
        "GEN",
        "GLW",
        "GRMN",
        "HPE",
        "HPQ",
        "IBM",
        "INTC",
        "INTU",
        "JKHY",
        "KEYS",
        "KLAC",
        "LDOS",
        "LRCX",
        "MCHP",
        "MPWR",
        "MRVL",
        "MSFT",
        "MSTR",
        "MU",
        "NET",
        "NOW",
        "NTAP",
        "NVDA",
        "NXPI",
        "OKTA",
        "ORCL",
        "PANW",
        "PARA",
        "PAYC",
        "PAYX",
        "PLTR",
        "PTC",
        "QCOM",
        "RNG",
        "ROP",
        "SEDG",
        "SHOP",
        "SMCI",
        "SNOW",
        "SNPS",
        "STX",
        "SWKS",
        "TEAM",
        "TEL",
        "TER",
        "TRMB",
        "TSM",
        "TWLO",
        "TXN",
        "TYL",
        "U",
        "VRSN",
        "WDAY",
        "WDC",
        "ZBRA",
        "ZM",
        "ZS"
    ],
    "Consumer Defensive": [
        "ACI",
        "ADM",
        "BJ",
        "CHD",
        "CL",
        "CLX",
        "COST",
        "COTY",
        "CPB",
        "DG",
        "DLTR",
        "EL",
        "GIS",
        "HSY",
        "KDP",
        "KHC",
        "KMB",
        "KO",
        "KR",
        "MDLZ",
        "MKC",
        "MNST",
        "MO",
        "PEP",
        "PG",
        "PM",
        "SFM",
        "SJM",
        "STZ",
        "SYY",
        "TGT",
        "WMT"
    ],
    "Utilities": [
        "AEE",
        "AEP",
        "ATO",
        "AWK",
        "CEG",
        "CMS",
        "CNP",
        "D",
        "DTE",
        "DUK",
        "ED",
        "ES",
        "EXC",
        "FE",
        "LNT",
        "NEE",
        "NRG",
        "PEG",
        "SO",
        "SRE",
        "WEC",
        "XEL"
    ],
    "Financial Services": [
        "AFL",
        "AFRM",
        "AIG",
        "AIZ",
        "AJG",
        "ALL",
        "ALLY",
        "AMP",
        "AON",
        "AXP",
        "BAC",
        "BEN",
        "BLK",
        "BRO",
        "BX",
        "C",
        "CB",
        "CBOE",
        "CFG",
        "CINF",
        "CME",
        "COF",
        "COIN",
        "FDS",
        "FITB",
        "GL",
        "GS",
        "HBAN",
        "HIG",
        "HOOD",
        "ICE",
        "IVZ",
        "JPM",
        "KEY",
        "L",
        "MA",
        "MCO",
        "MET",
        "MKTX",
        "MS",
        "MSCI",
        "MTB",
        "NDAQ",
        "NMIH",
        "NTRS",
        "NU",
        "PFG",
        "PGR",
        "PNC",
        "PRU",
        "PYPL",
        "RF",
        "SCHW",
        "SPGI",
        "STT",
        "SYF",
        "TFC",
        "TROW",
        "TRV",
        "UNM",
        "UPST",
        "V",
        "WFC",
        "WRB",
        "WTW",
        "ZION"
    ],
    "Basic Materials": [
        "ALB",
        "APD",
        "CE",
        "CF",
        "CTVA",
        "DD",
        "DOW",
        "ECL",
        "EMN",
        "FCX",
        "FMC",
        "HUN",
        "IFF",
        "LIN",
        "LYB",
        "MLM",
        "MOS",
        "NEM",
        "NUE",
        "PPG",
        "RS",
        "SHW",
        "STLD",
        "VMC"
    ],
    "Communication Services": [
        "AMC",
        "APP",
        "BIDU",
        "BILI",
        "CHTR",
        "CMCSA",
        "DIS",
        "EA",
        "FOX",
        "FOXA",
        "GOOG",
        "GOOGL",
        "LYV",
        "META",
        "MTCH",
        "NFLX",
        "NWS",
        "NWSA",
        "RBLX",
        "ROKU",
        "SIRI",
        "SNAP",
        "T",
        "TMUS",
        "TTWO",
        "VZ",
        "WBD"
    ],
    "Real Estate": [
        "AMT",
        "ARE",
        "AVB",
        "BXP",
        "CBRE",
        "CCI",
        "CPT",
        "CUBE",
        "DLR",
        "EQIX",
        "EQR",
        "ESS",
        "EXR",
        "FRT",
        "HST",
        "INVH",
        "IRM",
        "KIM",
        "MAA",
        "NNN",
        "O",
        "PLD",
        "PSA",
        "REG",
        "SBAC",
        "SLG",
        "SPG",
        "UDR",
        "VICI",
        "VNO",
        "VTR",
        "WELL",
        "WPC",
        "WY"
    ],
    "Energy": [
        "APA",
        "BKR",
        "CVX",
        "DVN",
        "EOG",
        "EQT",
        "FTI",
        "HAL",
        "HP",
        "KMI",
        "MPC",
        "NBR",
        "NOV",
        "OKE",
        "OXY",
        "PSX",
        "SLB",
        "TRGP",
        "VLO",
        "WMB",
        "XOM"
    ],
    "Unknown": [
        "ARNA",
        "ASGN",
        "BBBY",
        "BK",
        "BLUE",
        "BRK.B",
        "CDAY",
        "CMA",
        "CTLT",
        "CTRA",
        "DFS",
        "EXAS",
        "FBHS",
        "FGEN",
        "GLD",
        "GLPG",
        "GPS",
        "HES",
        "HOLX",
        "INSU",
        "IWM",
        "JNPR",
        "K",
        "MMC",
        "MRO",
        "MYOV",
        "PEAK",
        "PKI",
        "QD",
        "QQQ",
        "RAD",
        "SAGE",
        "SEE",
        "SPY",
        "WBA",
        "WRK",
        "X"
    ]
}
US_SECTORS["all"] = list({t for tickers in US_SECTORS.values() for t in tickers})

# ─────────────────────────────────────────────
#  SCORING WEIGHTS
# ─────────────────────────────────────────────
WEIGHTS = {
    "pe_score":          0.15,
    "peg_score":         0.15,
    "rev_score":         0.15,
    "earn_score":        0.15,
    "margin_score":      0.10,
    "roe_score":         0.10,
    "momentum_score":    0.15,  
    "analyst_score":     0.05,  
}

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def safe(val, default=np.nan):
    if val is None:
        return default
    try:
        v = float(val)
        return default if (v != v) else v 
    except (TypeError, ValueError):
        return default

def clamp01(x):
    if pd.isna(x):
        return 0.3 
    return max(0.0, min(1.0, x))

def score_momentum(current, low52, high52):
    if pd.isna(current) or pd.isna(low52) or pd.isna(high52) or high52 == low52:
        return 0.3
    pct = (current - low52) / (high52 - low52)
    return clamp01(1 - abs(pct - 0.625) * 1.4)

def score_analyst(rec_mean):
    if pd.isna(rec_mean):
        return 0.3
    return clamp01(1 - (rec_mean - 1) / 4)

def fetch_raw_data(ticker: str) -> dict | None:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("regularMarketPrice") is None:
            return None

        return {
            "Ticker":        ticker,
            "Name":          info.get("shortName", ticker)[:22],
            "Sector":        info.get("sector", "Unknown"),
            "Price":         safe(info.get("regularMarketPrice") or info.get("currentPrice")),
            "Low52":         safe(info.get("fiftyTwoWeekLow")),
            "High52":        safe(info.get("fiftyTwoWeekHigh")),
            "PE":            safe(info.get("trailingPE") or info.get("forwardPE")),
            "PEG":           safe(info.get("pegRatio")),
            "Rev_Pct":       safe(info.get("revenueGrowth")) * 100 if info.get("revenueGrowth") else np.nan,
            "Earn_Pct":      safe(info.get("earningsGrowth")) * 100 if info.get("earningsGrowth") else np.nan,
            "Margin_Pct":    safe(info.get("profitMargins")) * 100 if info.get("profitMargins") else np.nan,
            "ROE_Pct":       safe(info.get("returnOnEquity")) * 100 if info.get("returnOnEquity") else np.nan,
            "Analyst_Mean":  safe(info.get("recommendationMean")),
        }
    except Exception:
        return None

def z_score_to_01(z, lower_is_better=False):
    if pd.isna(z):
        return 0.3
    base_score = z / 5.0 
    if lower_is_better:
        return clamp01(0.5 - base_score)
    else:
        return clamp01(0.5 + base_score)

def calculate_normalized_scores(df: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "PE": True, 
        "PEG": True, 
        "Rev_Pct": False, 
        "Earn_Pct": False, 
        "Margin_Pct": False, 
        "ROE_Pct": False
    }

    for col, lower_better in metrics.items():
        sector_mean = df.groupby("Sector")[col].transform("mean")
        sector_std = df.groupby("Sector")[col].transform("std")
        sector_std = sector_std.replace(0, 1).fillna(1)
        
        z_col = f"{col}_Z"
        df[z_col] = (df[col] - sector_mean) / sector_std
        df[z_col] = df[z_col].fillna(0) 
        
        score_col = f"{col.split('_')[0].lower()}_score"
        df[score_col] = df[z_col].apply(lambda z: z_score_to_01(z, lower_is_better=lower_better))

    df["momentum_score"] = df.apply(lambda row: score_momentum(row["Price"], row["Low52"], row["High52"]), axis=1)
    df["analyst_score"] = df["Analyst_Mean"].apply(score_analyst)

    df["Total_Score"] = 0
    for key, weight in WEIGHTS.items():
        df["Total_Score"] += df.get(key, 0.3) * weight

    df["Total_Score"] = (df["Total_Score"] * 100).round(1)
    return df

def fetch_technicals(ticker: str) -> dict:
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        rsi_val = np.nan
        if not hist.empty and len(hist) >= 15:
            delta = hist['Close'].diff()
            up = delta.clip(lower=0)
            down = -1 * delta.clip(upper=0)
            ema_up = up.ewm(com=13, adjust=False).mean()
            ema_down = down.ewm(com=13, adjust=False).mean()
            rs = ema_up / ema_down
            rsi = 100 - (100 / (1 + rs))
            rsi_val = rsi.iloc[-1]
            
        info = stock.info
        vol = info.get("regularMarketVolume") or info.get("volume")
        avg_vol = info.get("averageVolume")
        rvol_val = (vol / avg_vol) if vol and avg_vol else np.nan
        
        iv_val = np.nan
        opts = stock.options
        if opts:
            chain = stock.option_chain(opts[0])
            calls = chain.calls
            current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', 0)
            
            if current_price > 0 and not calls.empty:
                calls['strikeDiff'] = abs(calls['strike'] - current_price)
                atm_call = calls.sort_values('strikeDiff').iloc[0]
                iv_val = atm_call.get('impliedVolatility', 0) * 100
                
        return {
            "Ticker": ticker,
            "RSI": rsi_val,
            "RVOL": rvol_val,
            "IV%": iv_val
        }
    except Exception:
        return {"Ticker": ticker, "RSI": np.nan, "RVOL": np.nan, "IV%": np.nan}

# ─────────────────────────────────────────────
#  MODULAR FUNCTION FOR STREAMLIT / TERMINAL
# ─────────────────────────────────────────────

def US_run_analysis(tickers=None, sector="stocks", top_n=10, progress_callback=None):
    """
    Core function that handles fetching and calculations. 
    Can be called from app.py or main().
    """
    if not tickers:
        tickers = [t.upper() for t in US_SECTORS.get(sector, US_SECTORS["stocks"])]

    raw_data = []
    completed = 0
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(fetch_raw_data, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            ticker = future_to_ticker[future]
            
            # Update progress callback if given (useful for Streamlit progress bars)
            if progress_callback:
                progress_callback(completed, total, f"Fetching {ticker}")
            else:
                print(f"  [{completed:03d}/{total:03d}] Fetching {ticker:<5}...", end="\r")
            
            try:
                result = future.result()
                if result: raw_data.append(result)
            except Exception:
                pass

    if not raw_data:
        return pd.DataFrame()

    df = pd.DataFrame(raw_data)
    df = calculate_normalized_scores(df)
    df = df.sort_values(by="Total_Score", ascending=False).head(top_n)

    tech_data = []
    top_tickers = df["Ticker"].tolist()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(top_tickers))) as executor:
        future_to_tech = {executor.submit(fetch_technicals, t): t for t in top_tickers}
        for future in concurrent.futures.as_completed(future_to_tech):
            try:
                res = future.result()
                tech_data.append(res)
            except Exception:
                pass
                
    tech_df = pd.DataFrame(tech_data)
    if not tech_df.empty:
        df = pd.merge(df, tech_df, on="Ticker", how="left")
    else:
        df["RSI"] = np.nan
        df["RVOL"] = np.nan
        df["IV%"] = np.nan

    return df

# ─────────────────────────────────────────────
#  MAIN (Terminal Execution)
# ─────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sector-Normalized Stock Analyzer")
    parser.add_argument("--tickers", nargs="+", help="Custom ticker list")
    parser.add_argument("--sector", default="stocks", choices=list(US_SECTORS.keys()))
    parser.add_argument("--top", type=int, default=10, help="Show top N stocks")
    args = parser.parse_args()

    print(f"\n{Fore.CYAN}{'═'*80}")
    print(f"   📈  US STOCK QUANTITATIVE ANALYZER (Z-SCORE NORMALIZED)")
    print(f"{'═'*80}{Style.RESET_ALL}")
    print(f"  Fetching data via threading...\n")

    df = US_run_analysis(tickers=args.tickers, sector=args.sector, top_n=args.top)

    if df.empty:
        print(f"{Fore.RED}No data retrieved.{Style.RESET_ALL}")
        sys.exit(1)

    table_data = []
    for rank, (_, row) in enumerate(df.iterrows(), 1):
        def fmt(val, dec=1): return f"{val:.{dec}f}" if pd.notna(val) else "—"
        
        table_data.append([
            f"#{rank}", row["Ticker"], row["Name"], row["Sector"],
            fmt(row["Price"], 2), fmt(row["PE"]), fmt(row["PEG"], 2),
            fmt(row["Rev_Pct"]), fmt(row["Margin_Pct"]), fmt(row["Total_Score"]),
            fmt(row.get("RSI"), 1), fmt(row.get("RVOL"), 2), fmt(row.get("IV%"), 1)
        ])

    headers = ["Rank", "Ticker", "Name", "Sector", "Price", "P/E", "PEG", 
               "Rev Gr%", "Margin%", "Score", "RSI", "RVOL", "IV%"]
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="rounded_outline"))

    print(f"\n{Fore.CYAN}  SIGNALS{Style.RESET_ALL}")
    print(f"  {'─'*45}")
    for _, row in df.iterrows():
        score = row['Total_Score']
        bar_len = int(score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        # Simple signal text mapping for terminal print
        sig_text = "PASS"
        if score >= 65: sig_text = f"{Fore.GREEN}★ STRONG BUY{Style.RESET_ALL}"
        elif score >= 55: sig_text = f"{Fore.CYAN}▲ BUY{Style.RESET_ALL}"
        elif score >= 45: sig_text = f"{Fore.YELLOW}● HOLD{Style.RESET_ALL}"
        else: sig_text = f"{Fore.RED}▼ PASS{Style.RESET_ALL}"

        print(f"  {row['Ticker']:<6} {bar}  {score:>5.1f}  {sig_text}")

if __name__ == "__main__":
    main()
