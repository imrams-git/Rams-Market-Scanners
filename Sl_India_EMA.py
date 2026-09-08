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
    local_tz = pytz.timezone('Asia/Kolkata')
    now_local = datetime.now(local_tz)
    
    if now_local.weekday() >= 5:
        return current_vol / avg_vol if avg_vol > 0 else 0

    market_open = now_local.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now_local.replace(hour=15, minute=30, second=0, microsecond=0)
    
    if now_local < market_open: 
        return 0
        
    if now_local >= market_close: 
        return current_vol / avg_vol if avg_vol > 0 else 0
    
    elapsed_minutes = (now_local - market_open).total_seconds() / 60.0
    fraction_of_day = elapsed_minutes / 375.0 
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
            
            # 3. Download if Cache is Missing or Stale
            if df.empty:
                stock = yf.Ticker(ticker)
                df = stock.history(period="2y")
                if not df.empty:
                    df.to_csv(cache_file) # Save fresh data to cache
            
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

            #if ema_condition and trigger and vol_expansion and is_coiled and rsi_valid:
            #if ema_condition and trigger and is_coiled and rsi_valid:
            if ema_condition and is_coiled and rsi_valid:
                dist_ema50 = ((curr['Close'] - curr['EMA_50']) / curr['EMA_50']) * 100

                # Probability Scoring
                vol_score = min(35, (vol_ratio / 2.0) * 35)
                coil_score = max(0, 35 - (curr['Coil_Range_10D'] * 3.5))
                rsi_distance = abs(curr['RSI_14'] - 68)
                rsi_score = max(0, 30 - (rsi_distance * 2))

                results.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "Price (₹)": f"{curr['Close']:.2f}",
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
    st.set_page_config(page_title="India Breakout Screener", layout="wide")
    st.title("📈 India High-Probability Breakout Screener")
    st.markdown("Scans a defined India NSE watchlist for EMA Crossovers, Volatility Contraction, and RSI Momentum.")

    if st.button("Run Breakout Scan", type="primary"):
        ranked_table = screen_stocks(watchlist)
        if not ranked_table.empty:
            st.success(f"Successfully found {len(ranked_table)} high-probability candidates!")
            st.dataframe(ranked_table, use_container_width=True, hide_index=True)
        else:
            st.warning("No stocks currently match the breakout criteria.")

if __name__ == "__main__":
    watchlist = [
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

    # Dynamically run the correct interface mode
    if is_streamlit_running():
        run_streamlit(watchlist)
    else:
        run_terminal(watchlist)
