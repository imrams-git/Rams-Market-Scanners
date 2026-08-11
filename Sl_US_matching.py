import yfinance as yf
import pandas as pd
import numpy as np
from typing import List
import requests
import warnings
import os
import time
import pytz
import re
from datetime import datetime
from google import genai

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

os.environ["GEMINI_API_KEY"] = "AQ.Ab8RN6LlYA9_yHAfIyqIkevLwBAIfnMk0c69DocN_4ivLGPDEw"

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

    def is_coiled_squeeze(self, df_d: pd.DataFrame) -> bool:
        """
        Determines if a stock is in a quiet, coiled squeeze on the Daily chart.
        Returns True if Bollinger Bands contract inside the Keltner Channels.
        """
        if len(df_d) < self.squeeze_length:
            return False
        close = df_d['Close']
        
        # 1. Calculate Standard Bollinger Bands (20 period, 2 StdDev)
        ma = close.rolling(window=self.squeeze_length).mean()
        std = close.rolling(window=self.squeeze_length).std()
        bb_upper = ma + (2 * std)
        bb_lower = ma - (2 * std)
        
        # 2. Calculate Keltner Channels (20 period MA + 1.5 * ATR)
        # Using simple range as a proxy for True Range to keep it lightweight
        high_low = df_d['High'] - df_d['Low']
        high_cp = abs(df_d['High'] - close.shift())
        low_cp = abs(df_d['Low'] - close.shift())
        tr = pd.concat([high_low, high_cp, low_cp], axis=1).max(axis=1)
        atr = tr.rolling(window=self.squeeze_length).mean()
        
        kc_upper = ma + (self.squeeze_multiplier * atr)
        kc_lower = ma - (self.squeeze_multiplier * atr)
        
        # 3. The Squeeze Condition
        # True if both the upper and lower BBs are completely inside the KC bounds
        current_squeeze = (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        
        return bool(current_squeeze)



    def analyze_symbol(self, intraday_data: pd.DataFrame, daily_data: pd.DataFrame, symbol: str, n_bars: int):
        if symbol not in intraday_data.columns.levels[0] or symbol not in daily_data.columns.levels[0]: return []
        
        df = intraday_data[symbol].copy().dropna()
        df_d = daily_data[symbol].copy().dropna()
        
        # Ensure we have enough daily data for the 50 DMA and historical checks
        if len(df) < 20 or len(df_d) < 55: return []

        # ---------------------------------------------------------
        # NEW RULE: SUSTAINED DAILY TREND & VOLUME
        # ---------------------------------------------------------

        #Macro Trend Filter (50 DMA)
        df_d['50_DMA'] = df_d['Close'].rolling(window=50).mean()
        current_50_dma = df_d['50_DMA'].iloc[-1]
        cmp_price = df['Close'].iloc[-1]
        if cmp_price <= current_50_dma: return []
        # Calculate a 20-day baseline for Daily Volume to measure against
        df_d['AvgVol_20'] = df_d['Volume'].rolling(window=20).mean()
        
        # Get the Daily Relative Volume for today, yesterday, and the day before
        rvol_d_today = self.get_time_adjusted_rvol(df_d['Volume'].iloc[-1], df_d['AvgVol_20'].iloc[-1])
        rvol_d_1_ago = df_d['Volume'].iloc[-2] / df_d['AvgVol_20'].iloc[-2]
        rvol_d_2_ago = df_d['Volume'].iloc[-3] / df_d['AvgVol_20'].iloc[-3]

        # CHECK: Bullish Steady Accumulation
        # Price is rising day-by-day AND volume is >= 1.0 on the previous 2 days
        is_steady_bull = (
            (cmp_price >= current_50_dma) and
            (rvol_d_1_ago >= 1.0) and 
            (rvol_d_2_ago >= 1.0)
        )

        # CHECK: Bearish Steady Distribution
        # Price is falling day-by-day AND volume is >= 1.0 on the previous 2 days
        is_steady_bear = (
            (cmp_price <= current_50_dma) and
            (rvol_d_1_ago >= 1.0) and 
            (rvol_d_2_ago >= 1.0)
        )

        # If it's not steadily accumulating or distributing, skip it entirely
        if not is_steady_bull and not is_steady_bear:
            return []

        # ---------------------------------------------------------
        # EXISTING LOGIC: Find the Unbroken Magnets (Rule 1)
        # ---------------------------------------------------------
        
        is_coiled = self.is_coiled_squeeze(df_d)
        df['RSI'] = self.calculate_rsi(df['Close'], self.rsi_period)
        df['ATR'] = self.calculate_atr(df, self.atr_period)
        df["AvgVol"] = df["Volume"].rolling(self.length).mean()
        df["RVOL_H"] = df["Volume"] / df["AvgVol"] 

        current_rsi = df['RSI'].iloc[-1]
        current_atr = df['ATR'].iloc[-1]
        current_rvol_h = df['RVOL_H'].iloc[-1]

        # Find historical high-vol bars to create the unmitigated levels
        recent_bars = df.iloc[-n_bars:]
        high_vol_indices = recent_bars[recent_bars["RVOL_H"] > self.high_volume_threshold].index

        valid_levels = []
        for i in reversed(high_vol_indices):
            row = df.loc[i]
            rng = row["High"] - row["Low"]
            if rng == 0: continue

            color = "ImbLow" if row["Close"] >= row["Open"] else "ImbHigh"
            
            # Align the historical magnet with our new daily trend direction
            if color == "ImbHigh" and not is_steady_bull: continue
            if color == "ImbLow" and not is_steady_bear: continue
            
            # Momentum safety check
            if color == "ImbHigh" and not (55 <= current_rsi <= 68): continue
            if color == "ImbLow" and not (25 <= current_rsi <= 45): continue

            buy_p = (row["Close"] - row["Low"]) / rng
            if color == "ImbLow" and buy_p < self.imbalance_threshold: continue
            if color == "ImbHigh" and (1 - buy_p) < self.imbalance_threshold: continue

            # RULE 1: Unbroken Check
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
                ticker = yf.Ticker(sym)
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

    def run(self, symbols: List[str], n_bars: int = 100):
        rows = []
        batch_size = 40
        print(f"Fetching Daily Data for RVOL(D)...")
        full_daily_data = self.get_batch_data(symbols, "1d", "90d")

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
                time.sleep(0.5)

        # Added explicit return for empty results to handle it properly in Streamlit
        if not rows: 
            print("No stocks matched.")
            return pd.DataFrame() 

        df = pd.DataFrame(rows, columns=["Symbol", "TF", "Imb", "CMP", "Unbrk", "%Diff", "Ago", "RSI", "RVOL_H", "RVOL_D", "Is_Coiled"])
        df = df.drop_duplicates()
        
        # Apply Technical Criteria to flag top picks
        top_criteria = (df['RVOL_D'] >= 1.0) & (df['RSI'] >= 60) & (df['RSI'] <= 70) & (df['Ago'] >= 30) & (df['Is_Coiled'])
        df['TopPick'] = top_criteria
        
        df = self.add_fundamentals(df)
        
        # Apply final earnings proximity filter directly to the dataframe
        def filter_earnings_risk(row):
            try:
                days_to_earn = int(row['Days2Earn'])
                # If it's a top pick but has earnings in 0-7 days, it's too risky. Filter it out.
                if 0 <= days_to_earn <= 7 and row['TopPick']:
                    return False
            except:
                pass
            return True
            
        df = df[df.apply(filter_earnings_risk, axis=1)]

        # Sort by Top Pick logic first, then by Is_Coiled, RVOL_D descending
        df = df.sort_values(by=["Is_Coiled", "TopPick", "RVOL_D"], ascending=False)
        df = df.drop(columns=["TopPick"])

        output_dir = 'output'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(output_dir, f"US_Matching_{timestamp_str}.csv")
        df.to_csv(filepath, index=False)

        # Terminal Print Output
        self.print_section(df, "ImbLow", RED, "🔴 BEARISH: TARGETING SUPPORT")
        self.print_section(df, "ImbHigh", GREEN, "🟢 BULLISH: TARGETING RESISTANCE")
        print(f"\n✅ Results saved to {filepath}")

        # Streamlit Browser Render Output (Only runs if script is invoked via `streamlit run`)
        if HAS_STREAMLIT and st.runtime.exists():
            st.success(f"Results saved to `{filepath}`")
            st.subheader("🟢 Bullish Resistance Targets")
            st.dataframe(df[df["Imb"] == "ImbHigh"], use_container_width=True)
            st.subheader("🔴 Bearish Support Targets")
            st.dataframe(df[df["Imb"] == "ImbLow"], use_container_width=True)

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
               - Requirements: Multi-timeframe confluence, Ago > 20, RVOL_D > 1.0, IsCoiled = True, D2Earn > 15 or D2Earn < -2 
            2. 🏆 The Tier 2 Structural Alpha (Buys)
               - Requirements: Ago > 5, RVOL_D > 0.8, IsCoiled = True or IsCoiled = False, D2Earn > 15 or D2Earn < -2 
            3. 🎯 The High-Volume Momentum Plays
               - Requirements: RVOL_D > 3.0, D2Earn > 15  or D2Earn < -2
            4. ⚠️ The Warnings & Rejections
               - Requirements: Low volume, structural conflicts, earnings risks, or poor risk profiles.

            Formatting Rules:
            1. Wrap key section headers, symbols, or signals in visual tags like:
               - [GREEN] for Tier 1
               - [YELLOW] for Tier 2
               - [CYAN] for High-Volume Momentum
               - [RED] for Warnings & rejections 
               - [BLUE] for bearish setups/risks
            - DO NOT print color tags at the end of lines (e.g., avoid symbols like '[/CYAN]' or '[/GREEN]'). Keep lines clean.
            - Use plain ASCII borders like '----------------------------------------'.
            - Keep descriptions concise, professional, and directly tied to the dataset.

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

    def print_section(self, df, imb_type, color, title):
        sub = df[df["Imb"] == imb_type]
        if sub.empty: return
        print(f"\n{title}")
        header = f"{'Symbol':<14} {'TF':<4} {'CMP':>8} {'Unbrk':>8} {'%Diff':>7} {'Ago':>4} {'RSI':>5} {'RV(H)':>5} {'RV(D)':>5} {'PE':>7} {'D2Earn':>6} {'IsCoiled':>5}"
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            is_top_pick = 60 <= r['RSI'] <= 70 and r['Ago'] >= 30
            row_color = YELLOW if is_top_pick else color

            print(row_color + f"{r['Symbol']:<14} {r['TF']:<4} {r['CMP']:>8.2f} {r['Unbrk']:>8.2f} {r['%Diff']:>7.2f} {int(r['Ago']):>4} {r['RSI']:>5.1f} {r['RVOL_H']:>5.1f} {r['RVOL_D']:>5.1f} {str(r['PE']):>7} {str(r['Days2Earn']):>6} {str(r['Is_Coiled']):>5}" + RESET)


# ==================================================
def main():
    symbols = [
        "SPY", "QQQ", "IWM", "GLD", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
        "JNJ", "V", "PG", "AVGO", "NVDA", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", 
        "CMCSA", "ADBE", "NFLX", "XOM", "PFE", "KO", "CSCO", "PEP", "T", "ABT", "CVX", 
        "CRM", "INTC", "ABBV", "WMT", "MCD", "VZ", "ACN", "NKE", "MDT", "COST", 
        "LIN", "BMY", "TXN", "DHR", "QCOM", "LLY", "HON", "PM", "ORCL", "AMGN", "IBM", 
        "SBUX", "MS", "RTX", "LOW", "GE", "INTU", "CAT", "BLK", "UPS", "GILD", "MMM", 
        "DE", "GS", "NOW", "PLD", "SCHW", "BA", "ADP", "AMD", "C", "CVS", "ISRG", 
        "SPGI", "MO", "BKNG", "AXP", "SYK", "ZTS", "AMT", "FIS", "MDLZ", "TJX", "TMO", 
        "BDX", "EQIX", "LMT", "PNC", "GM", "ELV", "APD", "ICE", "CL", 
        "CCI", "NSC", "TMUS", "CSX", "ITW", "ECL", "SHW", "WM", "EMR", "CME", "TGT", 
        "HUM", "KMB", "ROST", "ADI", "ADSK", "MCO", "LRCX", "BIIB", "BSX", "MRK", 
        "HCA", "VRTX", "MAR", "AON", "AEP", "MET", "EXC", "COF", "OXY", "PGR", "STZ", 
        "EW", "APH", "REGN", "DLR", "CTSH", "ORLY", "KMI", "PCAR", "VLO", "KHC", 
        "ALL", "HIG", "VRSN", "BAX", "MNST", "PEG", "EOG", "FDX", "D", "ROK", "TEL", 
        "DXCM", "CDW", "NOC", "PAYX", "CTAS", "CNC", "RMD", "HWM", 
        "MTD", "TSCO", "FTNT", "IDXX", "MTB", "BKR", "LHX", "A", "SRE", "CPRT", "WRB", 
        "RSG", "DOV", "CMS", "ED", "AJG", "WEC", "HST", "MCHP", "KMX", "PH", "EFX", 
        "CARR", "ETN", "AFL", "INCY", "ALGN", "CDNS", "COO", "MCK", "TT", "BIO", "KR", 
        "PNR", "AVY", "KEYS", "PLTR", "TEM", "SYM", "CTRA", "PPG", "SWKS", "ZBH", 
        "EXR", "DXC", "TSM", "ZM", "DOCU", "SNAP", 
        "OKTA", "PTON", "RBLX", "CRWD", "NET", "COIN", "ROKU", "TWLO", "BILI", "EA", 
        "LULU", "TTWO", "MRNA", "SNPS", "ILMN", "ASML", "NXPI", "MU", "SIRI", "TEAM", 
        "MELI", "KLAC", "EBAY", "PAYC", "MRVL", "AMAT", "FAST", "WDAY", "CHTR", "MTCH", 
        "ANET", "VRSK", "ARM", "AZN", "DASH", "HOOD", "SHOP", "TRI", "ZS", "BBAI", 
        "FRSH", "KNX", "HAL", "RDW", "JPM", "WFC", "NU",
        "BX", "CB", "MDLZ", "GD", "AIG", "DOW", "TRV", "CTVA", "MSCI", "ADM", 
        "OTIS", "OKE", "VICI", "GWW", "WELL", "HPQ", "VMC", "STT", "HPE", "DVN", 
        "FITB", "CBRE", "O", "WDC", "WY", "AME", "DAL", "UAL", "AEE", "LVS", "RF", 
        "GLW", "XYL", "VTR", "TDG", "STX", "TROW", "AWK", "ES", "DTE", "F", "BBY", 
        "FE", "SWK", "SYY", "ZBRA", "INVH", "BRO", "MGM", "GEN", "LNT", 
        "EXPE", "CNP", "CINF", "ATO", "SJM", "DRI", "FSLR", "AKAM", "JKHY", "IRM", 
        "NRG", "MAS", "L", "TYL", "DG", "WST", "BALL", "CAH", "TRMB", "EPAM", "WAT", 
        "POOL", "MOH", "VRT", "SMCI", "RCL", "CCL", "HRL", "CAG", "ALLE", "TPR", 
        "VRSK", "MPWR", "ODFL", "LDOS", "GRMN", "TER", "HUBB", "WST",
        "MOH", "SMCI", "RCL", "CCL", "AVAV", "QURE", "APP", "KTOS", "VST", "RXRX", 
        "VKTX", "SNOW", "GEV", "PANW", "DDOG", "UNP", "CMG"    
]

    checker = VolumeAlertChecker()
    
    # Check if running via Streamlit browser server context
    if HAS_STREAMLIT and st.runtime.exists():
        st.title("🇮🇳 NSE Quantitative Scanner & Gemini Analysis")
        if st.button("Run Scan"):
            with st.spinner("Running scan and generating AI breakdown..."):
                checker.run(symbols, n_bars=100)
    else:
        # Standard Terminal Execution (bypasses streamlit completely)
        checker.run(symbols, n_bars=100)

# ==================================================
if __name__ == "__main__":
    main()
