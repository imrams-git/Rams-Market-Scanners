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
        self.timeframes = {"30m": "30m", "1Hr": "1h"}
        self.squeeze_length = squeeze_length
        self.squeeze_multiplier = squeeze_multiplier

    def get_time_adjusted_rvol(self, current_vol, avg_vol):
        # Force local time to calculate elapsed time accurately
        # Since the user runs this on NSE (Indian market), we need IST timezone and hours
        local_tz = pytz.timezone('Asia/Kolkata')
        now_local = datetime.now(local_tz)
        
        market_open = now_local.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now_local.replace(hour=15, minute=30, second=0, microsecond=0)
        
        # Pre-market: Volume is negligible, return 0 or unadjusted
        if now_local < market_open: 
            return 0
            
        # After-hours: Use the full daily average
        if now_local >= market_close: 
            return current_vol / avg_vol if avg_vol > 0 else 0
        
        # Intraday: Calculate elapsed minutes
        elapsed_minutes = (now_local - market_open).total_seconds() / 60.0
        fraction_of_day = elapsed_minutes / 375.0 # 375 minutes in a standard NSE trading day
        
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
        
        if len(df) < 20 or len(df_d) < 55: return []

        # Calculate the Daily 50 Moving Average
        df_d['50_DMA'] = df_d['Close'].rolling(window=50).mean()
        current_50_dma = df_d['50_DMA'].iloc[-1]
        cmp_price = df['Close'].iloc[-1]

        # Macro Trend Filter: Immediately skip if the stock is trading under its daily 50 DMA
        if cmp_price <= current_50_dma:
           return []

        # RVOL (D) Calculation - Time-Adjusted
        avg_vol_d = df_d['Volume'].iloc[-11:-1].mean()
        current_vol_d = df_d['Volume'].iloc[-1]
        rvol_d = self.get_time_adjusted_rvol(current_vol_d, avg_vol_d)

        # Check if the daily chart is actively coiling in a squeeze
        is_coiled = self.is_coiled_squeeze(df_d)
        
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

            if rvol_d < 0.8: continue

            color = "ImbLow" if row["Close"] >= row["Open"] else "ImbHigh"
            if color == "ImbHigh" and (not is_bullish or not (55 <= current_rsi <= 68)): continue
            if color == "ImbLow" and (not is_bearish or not (25 <= current_rsi <= 45)): continue

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
                                 round(current_rvol_h, 2), round(rvol_d, 2), is_coiled])
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
                time.sleep(1)

        if not rows: return print("No stocks matched.")

        df = pd.DataFrame(rows, columns=["Symbol", "TF", "Imb", "CMP", "Unbrk", "%Diff", "Ago", "RSI", "RVOL_H", "RVOL_D", "Is_Coiled"])
        df = df.drop_duplicates()
        
        # Apply Technical Criteria to flag top picks
        top_criteria = (df['RVOL_D'] >= 1.0) & (df['RSI'] >= 60) & (df['RSI'] <= 70) & (df['Ago'] >= 30 & (df['Is_Coiled']))
        df['TopPick'] = top_criteria
        
        df = self.add_fundamentals(df)
        
        # Sort by Top Pick logic first, then by Is_Coiled, RVOL_D descending
        df = df.sort_values(by=["Is_Coiled", "TopPick", "RVOL_D"], ascending=False)
        df = df.drop(columns=["TopPick"])

        output_dir = 'output'
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        filepath = os.path.join(output_dir, f"India_Matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(filepath, index=False)

        self.print_section(df, "ImbLow", RED, "🔴 BEARISH: TARGETING SUPPORT")
        self.print_section(df, "ImbHigh", GREEN, "🟢 BULLISH: TARGETING RESISTANCE")
        print(f"\n✅ Results saved to {filepath}")

    def print_section(self, df, imb_type, color, title):
        sub = df[df["Imb"] == imb_type]
        if sub.empty: return
        print(f"\n{title}")
        header = f"{'Symbol':<14} {'TF':<4} {'CMP':>8} {'Unbrk':>8} {'%Diff':>7} {'Ago':>4} {'RSI':>5} {'RV(H)':>5} {'RV(D)':>5} {'EPS':>7} {'D2Earn':>6} {'IsCoiled':>5}"
        print(header)
        print("-" * len(header))
        for _, r in sub.iterrows():
            is_top_pick = 60 <= r['RSI'] <= 70 and r['Ago'] >= 30
            row_color = YELLOW if is_top_pick else color
            
            try:
                days_to_earn = int(r['Days2Earn'])
                if days_to_earn <= 7 and days_to_earn >= 0 and is_top_pick:
                    continue 
            except: pass

            print(row_color + f"{r['Symbol']:<14} {r['TF']:<4} {r['CMP']:>8.2f} {r['Unbrk']:>8.2f} {r['%Diff']:>7.2f} {int(r['Ago']):>4} {r['RSI']:>5.1f} {r['RVOL_H']:>5.1f} {r['RVOL_D']:>5.1f} {str(r['EPS']):>7} {str(r['Days2Earn']):>6} {str(r['Is_Coiled']):>5}" + RESET)


# ==================================================
def main():
    symbols = [
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
    'HAL.NS', 'BAYERCROP.NS', 'AKZOINDIA.NS', 'NBCC.NS', 'BDL.NS', 'NH.NS',
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
    'JYOTICNC.NS', 'AWL.NS', 'BHEL.NS', 'TARIL.NS', 'GUJGASLTD.NS', 'AARTIIND.NS',
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

    checker = VolumeAlertChecker()
    checker.run(symbols, n_bars=100)


# ==================================================
if __name__ == "__main__":
    main()
