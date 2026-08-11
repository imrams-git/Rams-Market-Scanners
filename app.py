import streamlit as st
import importlib
from m_stock_analyzer import run_analysis, SECTORS

# 1. Page Configuration (Called ONCE at the absolute top)
st.set_page_config(page_title="Trading Algorithm Dashboard", layout="wide")

# 2. Sidebar Navigation for selecting the script
st.sidebar.header("Select Strategy")
selected_script = st.sidebar.selectbox(
    "Choose an analysis tool:",
    ["India Volume Matching", 
     "US Volume Matching", 
     "India SMA200",
     "US SMA200",
     "M stock Analyser"])

st.sidebar.markdown("---")
st.sidebar.info(f"Active Script Configuration: **{selected_script}**")

# Main Title Area
st.title("🎛️ Algorithmic Trading Command Center")

# Shared Universal Symbol List
US_SYMBOLS = [
    "A", "AAL", "AAP", "AAPL", "ABBV", "ABNB", "ABT", "ACI", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE",
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

INDIA_SYMBOLS = [
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
    'CENTRALBK.NS', 'BAJAJHLDNG.NS'
]

# ==========================================
# ROUTING LOGIC BASED ON USER SELECTION
# ==========================================

# --- SCRIPT 1: INDIA VOLUME MATCHING ---
if selected_script == "India Volume Matching":
    st.subheader("📈 Institutional Volume Magnetism Scanner (India)")
    st.markdown("Scans for high-volume momentum candles and checks if support/resistance levels remain unbroken.")
    n_bars = st.slider("Lookback Bars", min_value=20, max_value=200, value=100)
    
    if st.button("Execute Volume Scan", type="primary"):
        with st.spinner("Processing volume imbalances..."):
            india_matching = importlib.import_module("Sl_India_matching") 
            checker = india_matching.VolumeAlertChecker()
            result_df = checker.run(INDIA_SYMBOLS, n_bars=n_bars)
            
            if result_df is not None and not result_df.empty:
                st.success("Scan Complete!")
                #    python script already prints values using streamlit
                # st.subheader("🟢 BULLISH: TARGETING RESISTANCE")
                # st.dataframe(result_df[result_df["Imb"] == "ImbHigh"], hide_index=True, use_container_width=True)
                # st.subheader("🔴 BEARISH: TARGETING SUPPORT")
                # st.dataframe(result_df[result_df["Imb"] == "ImbLow"], hide_index=True, use_container_width=True)
            else:
                st.warning("No levels matched current parameters.")

# --- SCRIPT 2: US Volume Matching ---
elif selected_script == "US Volume Matching":
    st.subheader("📈 Institutional Volume Magnetism Scanner (US)")
    st.markdown("Scans for high-volume momentum candles and checks if support/resistance levels remain unbroken.")
    n_bars = st.slider("Lookback Bars", min_value=20, max_value=200, value=100)
    
    if st.button("Execute Volume Scan", type="primary"):
        with st.spinner("Processing volume imbalances..."):
            us_matching = importlib.import_module("Sl_US_matching")
            checker = us_matching.VolumeAlertChecker()
            result_df = checker.run(US_SYMBOLS, n_bars=n_bars)
            
            if result_df is not None and not result_df.empty:
                st.success("Scan Complete!")
                # python script already prints values using streamlit
                # st.subheader("🟢 BULLISH: TARGETING RESISTANCE")
                # st.dataframe(result_df[result_df["Imb"] == "ImbHigh"], hide_index=True, use_container_width=True)
                # st.subheader("🔴 BEARISH: TARGETING SUPPORT")
                # st.dataframe(result_df[result_df["Imb"] == "ImbLow"], hide_index=True, use_container_width=True)
            else:
                st.warning("No levels matched current parameters.")

# --- SCRIPT 3: INDIA SMA200 ---
elif selected_script == "India SMA200":
    st.subheader("📈 India Stocks bouncing from SMA200 Scanner")
    st.markdown("Scans for stocks reversing after touching 200 SMA levels on daily timeframes.")
    
    if st.button("Execute SMA 200", type="primary"):
        with st.spinner("Processing SMA 200 scan..."):
            india_sma200 = importlib.import_module("Sl_India_SMA200")
            matches = india_sma200.scan_stocks()
            
            # Convert list to DataFrame safely
            result_df = pd.DataFrame(matches) if matches else pd.DataFrame()
            
            if not result_df.empty:
                st.success("Scan Complete!")
                result_df = result_df.sort_values(by="%diff", ascending=True)
                st.dataframe(result_df, hide_index=True, use_container_width=True)
            else:
                st.warning("0 results matched current technical parameters.")


# --- SCRIPT 4: US SMA200 ---
elif selected_script == "US SMA200":
    st.subheader("📈 US Stocks bouncing from SMA200 Scanner")
    st.markdown("Scans for stocks reversing after touching 200 SMA levels on daily timeframes.")
    
    if st.button("Execute SMA 200", type="primary"):
        with st.spinner("Processing SMA 200 scan..."):
            us_sma200 = importlib.import_module("Sl_US_SMA500")
            matches = us_sma200.scan_stocks()
            
            # Convert list to DataFrame safely
            result_df = pd.DataFrame(matches) if matches else pd.DataFrame()
            
            if not result_df.empty:
                st.success("Scan Complete!")
                result_df = result_df.sort_values(by="%diff", ascending=True)
                st.dataframe(result_df, hide_index=True, use_container_width=True)
            else:
                st.warning("0 results matched current technical parameters.")

# --- SCRIPT 5: M stock Analyser ---
elif selected_script == "M stock Analyser":
    st.subheader("📈 US Stock Quantitative Analyzer")
    st.markdown("Scores stocks via Sector-Normalized Z-Scores with real-time technical tracking.")

    selected_sector = st.sidebar.selectbox("Select Universe / Sector Group", list(SECTORS.keys()))
    top_n = st.sidebar.slider("Show Top N Stocks", min_value=5, max_value=30, value=10)

    if st.button("🚀 Run Analysis", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_streamlit_progress(completed, total, text):
            pct = completed / total
            progress_bar.progress(pct)
            status_text.text(f"[{completed}/{total}] {text}...")

        df = run_analysis(sector=selected_sector, top_n=top_n, progress_callback=update_streamlit_progress)

        progress_bar.empty()
        status_text.empty()

        if not df.empty:
            st.success(f"Analysis complete! Top {len(df)} stocks analyzed.")
            st.dataframe(df[["Ticker", "Name", "Sector", "Price", "PE", "PEG", "Rev_Pct", "Margin_Pct", "Total_Score", "RSI", "RVOL", "IV%"]], use_container_width=True)
            st.subheader("📊 Quant Scores Breakdown")
            st.bar_chart(df.set_index("Ticker")["Total_Score"])
        else:
            st.error("No valid stock data could be retrieved. Try again later.")
