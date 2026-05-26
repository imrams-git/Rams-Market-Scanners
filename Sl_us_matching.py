import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import requests
import warnings
import os
import time
import pytz
from datetime import datetime

warnings.filterwarnings("ignore")

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class VolumeAlertChecker:
    def __init__(
        self,
        length: int = 10,
        high_volume_threshold: float = 2.0,
        imbalance_threshold: float = 0.70,
        min_body_ratio: float = 0.60,
        min_distance_pct: float = 0.5, 
        trend_lookback: int = 5,
        rsi_period: int = 14,
        atr_period: int = 14,
        atr_multiplier: float = 5.0
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
        self.timeframes = {"30m": "30m", "1Hr": "1h"}

    def get_time_adjusted_rvol(self, current_vol, avg_vol):
        # Force Eastern Time to align with US market hours
        eastern = pytz.timezone('US/Eastern')
        now_et = datetime.now(eastern)
        
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        
        # Pre-market: Volume is negligible, return 0 or unadjusted
        if now_et < market_open: 
            return 0
            
        # After-hours: Use the full daily average
        if now_et >= market_close: 
            return current_vol / avg_vol if avg_vol > 0 else 0
        
        # Intraday: Calculate elapsed minutes
        elapsed_minutes = (now_et - market_open).total_seconds() / 60.0
        fraction_of_day = elapsed_minutes / 390.0 # 390 minutes in a standard trading day
        
        # Pro-rate the baseline average volume
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
        try:
            return yf.download(tickers=symbols, period=period, interval=timeframe, group_by="ticker", threads=True, progress=False)
        except: return pd.DataFrame()

    def analyze_symbol(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame, symbol: str, n_bars: int):
        if symbol not in intraday_data.columns.levels[0] or symbol not in daily_data.columns.levels[0]: return []
        
        df = intraday_data[symbol].copy().dropna()
        df_d = daily_data[symbol].copy().dropna()
        
        if len(df) < 20 or len(df_d) < 11: return []

        # RVOL (D) Calculation - Time-Adjusted
        avg_vol_d = df_d['Volume'].iloc[-11:-1].mean()
        current_vol_d = df_d['Volume'].iloc[-1]
        rvol_d = self.get_time_adjusted_rvol(current_vol_d, avg_vol_d)
        
        # Technicals
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['ATR'] = self.calculate_atr(df, self.atr_period)
        df["AvgVol"] = df["Volume"].rolling(self.length).mean()
        df["RVOL_H"] = df["Volume"] / df["AvgVol"] # Current bar vs Bar Avg

        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        current_rvol_h = df['RVOL_H'].iloc[-1]
        cmp_price = df['Close'].iloc[-1]

        recent_move = cmp_price - df['Close'].iloc[-self.trend_lookback]
        is_bullish = recent_move > 0
        is_bearish = recent_move < 0

        # Analyze historical high-vol bars
        recent_bars = df.iloc[-n_bars:]
        high_vol_indices = recent_bars[recent_bars["RVOL_H"] > self.high_volume_threshold].index

        valid_levels = []
        for i in reversed(high_vol_indices):
            row = df.loc[i]
            rng = row["High"] - row["Low"]
            if rng == 0: continue

            color = "ImbLow" if row["Close"] >= row["Open"] else "ImbHigh"
            if color == "ImbHigh" and (not is_bullish or not (50 <= current_rsi <= 75)): continue
            if color == "ImbLow" and (not is_bearish or not (25 <= current_rsi <= 50)): continue

            buy_p = (row["Close"] - row["Low"]) / rng
            if color == "ImbLow" and buy_p < self.imbalance_threshold: continue
            if color == "ImbHigh" and (1 - buy_p) < self.imbalance_threshold: continue

            future = df.loc[i:].iloc[1:]
            level = row["High"] if color == "ImbHigh" else row["Low"]
            if color == "ImbHigh" and (future["High"] > level).any(): continue
            if color == "ImbLow" and (future["Low"] < level).any(): continue

            dist = abs(level - cmp_price)
            if dist > (current_atr * self.atr_multiplier): continue
            pct_diff = (dist / cmp_price) * 100
            if pct_diff < self.min_distance_pct: continue

            bars_ago = len(df) - df.index.get_loc(i) - 1
            valid_levels.append([symbol, color, round(cmp_price, 2), round(level, 2), 
                                 round(pct_diff, 2), bars_ago, round(current_rsi, 2), 
                                 round(current_rvol_h, 2), round(rvol_d, 2)])
        return valid_levels

    def add_fundamentals(self, df: pd.DataFrame) -> pd.DataFrame:
        unique_symbols = df['Symbol'].unique()
        fund_data = {}
        
        if len(unique_symbols) > 0:
            print(f"\nFetching Fundamental Data for {len(unique_symbols)} matched symbols...")
            
        for sym in unique_symbols:
            eps_str = "N/A"
            days_to_earn = "N/A"
            
            try:
                ticker = yf.Ticker(sym)
                
                # Independent Try-Catch for EPS
                try:
                    info = ticker.info
                    eps = info.get('trailingEps', None)
                    if eps is not None:
                        eps_str = f"{eps:.2f}"
                except Exception:
                    eps_str = "ERR"
                
                # Independent Try-Catch for Earnings Date
                try:
                    # Method 1: Try using the nested calendar property
                    calendar = ticker.calendar
                    if calendar is not None:
                        if isinstance(calendar, dict) and 'Earnings Date' in calendar:
                            dates = calendar['Earnings Date']
                            if len(dates) > 0:
                                next_date = dates[0].date() if hasattr(dates[0], 'date') else dates[0]
                                days_to_earn = (next_date - datetime.now().date()).days
                                
                    # Method 2: Fallback to info timestamp if calendar failed
                    if days_to_earn == "N/A" and hasattr(ticker, 'info'):
                        earn_ts = ticker.info.get('earningsTimestamp') or ticker.info.get('earningsTimestampStart')
                        if earn_ts:
                            next_date = datetime.fromtimestamp(earn_ts).date()
                            days_to_earn = (next_date - datetime.now().date()).days
                except Exception:
                    days_to_earn = "ERR"
                
            except Exception:
                pass # Main ticker assignment failed
                
            fund_data[sym] = {"EPS": eps_str, "Days2Earn": days_to_earn}
            # Increased sleep to prevent IP blocking from Yahoo Finance
            time.sleep(1) 
                
        df['EPS'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('EPS', 'N/A'))
        df['Days2Earn'] = df['Symbol'].map(lambda x: fund_data.get(x, {}).get('Days2Earn', 'N/A'))
        return df

    def run(self, symbols: List[str], n_bars: int = 100):
        rows = []
        batch_size = 40
        print(f"Fetching Daily Data for RVOL(D)...")
        full_daily_data = self.get_batch_data(symbols, "1d", "30d")

        for tf_name, tf_interval in self.timeframes.items():
            print(f"Scanning {tf_name}...")
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i : i + batch_size]
                intraday_data = self.get_batch_data(batch, tf_interval)
                if intraday_data.empty: continue
                for sym in batch:
                    results = self.analyze_symbol(intraday_data, full_daily_data, sym, n_bars)
                    for res in results:
                        rows.append([res[0], tf_name] + res[1:])
                time.sleep(1)

        if not rows: return print("No stocks matched.")

        df = pd.DataFrame(rows, columns=["Symbol", "TF", "Imb", "CMP", "Unbrk", "%Diff", "Ago", "RSI", "RVOL_H", "RVOL_D"])
        df = df.drop_duplicates()
        
        # Apply Technical Criteria to flag top picks
        top_criteria = (df['RVOL_D'] >= 1.0) & (df['RSI'] >= 60) & (df['RSI'] <= 70) & (df['Ago'] >= 30)
        df['TopPick'] = top_criteria
        
        df = self.add_fundamentals(df)
        
        # Sort by Top Pick logic first, then by RVOL_D descending
        df = df.sort_values(by=["TopPick", "RVOL_D"], ascending=[False, False])
        df = df.drop(columns=["TopPick"])

        output_dir = 'output'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filepath = os.path.join(output_dir, f"US_Matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(filepath, index=False)

        self.print_section(df, "ImbLow", RED, "🔴 BEARISH: TARGETING SUPPORT")
        self.print_section(df, "ImbHigh", GREEN, "🟢 BULLISH: TARGETING RESISTANCE")
        print(f"\n✅ Results saved to {filepath}")
        return df

    def print_section(self, df, imb_type, color, title):
        sub = df[df["Imb"] == imb_type]
        if sub.empty: return
        print(f"\n{title}")
        header = f"{'Symbol':<14} {'TF':<4} {'CMP':>8} {'Unbrk':>8} {'%Diff':>7} {'Ago':>4} {'RSI':>5} {'RV(H)':>5} {'RV(D)':>5} {'EPS':>7} {'D2Earn':>6}"
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            is_top_pick = 60 <= r['RSI'] <= 70 and r['Ago'] >= 30
            row_color = YELLOW if is_top_pick else color
            
            try:
                days_to_earn = int(r['Days2Earn'])
                if days_to_earn <= 5 and days_to_earn >= 0 and is_top_pick:
                    row_color = RED 
            except: pass

            print(row_color + f"{r['Symbol']:<14} {r['TF']:<4} {r['CMP']:>8.2f} {r['Unbrk']:>8.2f} {r['%Diff']:>7.2f} {int(r['Ago']):>4} {r['RSI']:>5.1f} {r['RVOL_H']:>5.1f} {r['RVOL_D']:>5.1f} {str(r['EPS']):>7} {str(r['Days2Earn']):>6}" + RESET)


# ==================================================
def main():
    symbols = [
        "SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
        "JNJ", "V", "PG", "AVGO", "NVDA", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", 
        "CMCSA", "ADBE", "NFLX", "XOM", "PFE", "KO", "CSCO", "PEP", "T", "ABT", "CVX", 
        "CRM", "INTC", "ABBV", "WMT", "MCD", "VZ", "ACN", "NKE", "MDT", "NEE", "COST", 
        "LIN", "BMY", "TXN", "DHR", "QCOM", "LLY", "HON", "PM", "ORCL", "AMGN", "IBM", 
        "SBUX", "MS", "RTX", "LOW", "GE", "INTU", "CAT", "BLK", "UPS", "GILD", "MMM", 
        "DE", "GS", "NOW", "PLD", "SCHW", "BA", "ADP", "AMD", "C", "CVS", "ISRG", 
        "SPGI", "MO", "BKNG", "AXP", "SYK", "ZTS", "AMT", "FIS", "MDLZ", "TJX", "TMO", 
        "BDX", "EQIX", "LMT", "PNC", "GM", "ELV", "DUK", "SO", "APD", "ICE", "CL", 
        "CCI", "NSC", "TMUS", "CSX", "ITW", "ECL", "SHW", "WM", "EMR", "CME", "TGT", 
        "HUM", "KMB", "ROST", "ADI", "ADSK", "MCO", "LRCX", "BIIB", "BSX", "MRK", 
        "HCA", "VRTX", "MAR", "AON", "AEP", "MET", "EXC", "COF", "OXY", "PGR", "STZ", 
        "EW", "APH", "REGN", "DLR", "CTSH", "ORLY", "KMI", "PCAR", "VLO", "KHC", 
        "ALL", "HIG", "VRSN", "BAX", "MNST", "PEG", "EOG", "FDX", "D", "ROK", "TEL", 
        "DXCM", "CDW", "NOC", "PAYX", "CTAS", "XEL", "CNC", "HOLX", "RMD", "HWM", 
        "MTD", "TSCO", "FTNT", "IDXX", "MTB", "BKR", "LHX", "A", "SRE", "CPRT", "WRB", 
        "RSG", "DOV", "CMS", "ED", "AJG", "WEC", "HST", "MCHP", "KMX", "PH", "EFX", 
        "CARR", "ETN", "AFL", "INCY", "ALGN", "CDNS", "COO", "MCK", "TT", "BIO", "KR", 
        "PNR", "AVY", "KEYS", "PLTR", "TEM", "SYM", "CTRA", "PPG", "SWKS", "ZBH", 
        "EXR", "DXC", "CEG", "TSM", "BIDU", "JD", "PDD", "ZM", "DOCU", "SNAP", 
        "OKTA", "PTON", "RBLX", "CRWD", "NET", "COIN", "ROKU", "TWLO", "BILI", "EA", 
        "LULU", "TTWO", "MRNA", "SNPS", "ILMN", "ASML", "NXPI", "MU", "SIRI", "TEAM", 
        "MELI", "KLAC", "EBAY", "PAYC", "MRVL", "AMAT", "FAST", "WDAY", "CHTR", "MTCH", 
        "ANET", "VRSK", "ARM", "AZN", "DASH", "HOOD", "SHOP", "TRI", "ZS", "BBAI", 
        "FRSH", "KNX", "HAL", "RDW", "BABA", "JPM", "WFC", "NU",
        "BX", "CB", "MDLZ", "GD", "AIG", "DOW", "TRV", "CTVA", "MSCI", "ADM", 
        "OTIS", "OKE", "VICI", "GWW", "WELL", "HPQ", "VMC", "STT", "HPE", "DVN", 
        "FITB", "CBRE", "O", "WDC", "WY", "AME", "DAL", "UAL", "AEE", "LVS", "RF", 
        "GLW", "XYL", "VTR", "TDG", "STX", "TROW", "AWK", "ES", "DTE", "F", "BBY", 
        "FE", "SWK", "SYY", "ZBRA", "INVH", "BRO", "MGM", "GEN", "LNT", 
        "EXPE", "CNP", "CINF", "ATO", "SJM", "DRI", "FSLR", "AKAM", "JKHY", "IRM", 
        "NRG", "MAS", "L", "TYL", "DG", "WST", "BALL", "CAH", "TRMB", "EPAM", "WAT", 
        "POOL", "MOH", "VRT", "SMCI", "RCL", "CCL", "HRL", "CAG", "ALLE", "TPR", 
        "VRSK", "MPWR", "ODFL", "LDOS", "GRMN", "TER", "HUBB", "WST",
        "MOH",  "SMCI", "RCL", "CCL", "AVAV", "QURE"
    ]

    checker = VolumeAlertChecker()
    checker.run(symbols, n_bars=100)


# ==================================================
if __name__ == "__main__":
    main()
