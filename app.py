import streamlit as st
import importlib

# 1. Page Configuration
st.set_page_config(page_title="Trading Algorithm Dashboard", layout="wide")
st.title("🎛️ Algorithmic Trading Command Center")

# 2. Sidebar Navigation for selecting the script
st.sidebar.header("Select Strategy")
selected_script = st.sidebar.selectbox(
    "Choose an analysis tool:",
    ["US Volume Matching", "M stock Analyser", "India Volume Matching", "India SMA200"]
)

# Shared Universal Symbol List
US_SYMBOLS = [
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
        "MOH", "SMCI", "RCL", "CCL", "AVAV", "QURE", "APP", "KTOS", "VST", "RXRX", 
        "VKTX", "SNOW", "GEV", "PANW", "DDOG", "UNP", "CMG"    
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

st.sidebar.markdown("---")
st.sidebar.info(f"Active Script Configuration: **{selected_script}**")

# ==========================================
# ROUTING LOGIC BASED ON USER SELECTION
# ==========================================

# --- SCRIPT 1: US Volume Matching ---
if selected_script == "US Volume Matching":
    st.subheader("📈 Institutional Volume Magnetism Scanner")
    st.markdown("Scans for unmitigated high-volume levels on the 30m, 1Hr, 4Hr, 1D timeframes.")
    
    # Custom parameter inputs just for this script
    n_bars = st.slider("Lookback Bars", min_value=20, max_value=200, value=100)
    
    if st.button("Execute Volume Scan", type="primary"):
        with st.spinner("Processing volume imbalances..."):
            # Dynamically import and run your existing module
            us_matching = importlib.reload("Sl_us_matching")
            checker = us_matching.VolumeAlertChecker()
            
            # Run and capture data
            result_df = checker.run(US_SYMBOLS, n_bars=n_bars)
            
            if result_df is not None and not result_df.empty:
                st.success("Scan Complete!")
                st.subheader("🟢 BULLISH: TARGETING RESISTANCE")
                st.dataframe(result_df[result_df["Imb"] == "ImbHigh"], hide_index=True, use_container_width=True)
                st.subheader("🔴 BEARISH: TARGETING SUPPORT")
                st.dataframe(result_df[result_df["Imb"] == "ImbLow"], hide_index=True, use_container_width=True)
            else:
                st.warning("No levels matched current parameters.")

# --- SCRIPT 2: M stock Analyser ---
elif selected_script == "M stock Analyser":
    st.subheader("🦅 Mathavan's Stock Analyser")
    st.markdown("Based on Mathavan's Claude script")
    
    # Custom parameters for options script
    target_delta = st.number_input("Target OTM Delta", value=0.15, step=0.05)
    dte = st.slider("Days to Expiration (DTE)", min_value=1, max_value=60, value=45)
    
    if st.button("Scan Options Market", type="primary"):
        with st.spinner("Fetching option chains from Yahoo Finance..."):
            try:
                # Dynamically import your second script (m_stock_analyzer.py)
                opt_scan = importlib.reload("options_scanner")
                
                # Assuming your other script has a function or class to call:
                # result_df = opt_scan.run_options_analysis(US_SYMBOLS, target_delta, dte)
                # st.dataframe(result_df, hide_index=True, use_container_width=True)
                
                st.info("Placeholder: This is where your options script output will render.")
            except ModuleNotFoundError:
                st.error("Missing file: Ensure 'options_scanner.py' is in the same folder.")

# --- SCRIPT 3: INDIA VOLUME MATCHING ---
elif selected_script == "India Volume Matching":
    st.subheader("📈 Institutional Volume Magnetism Scanner")
    st.markdown("Scans for unmitigated high-volume levels on the 30m, 1Hr, 4Hr and 1D timeframes.")
    
    # Custom parameter inputs just for this script
    n_bars = st.slider("Lookback Bars", min_value=20, max_value=200, value=100)
    
    if st.button("Execute Volume Scan", type="primary"):
        with st.spinner("Processing volume imbalances..."):
            # Dynamically import and run your existing module
            india_matching = importlib.reload("Sl_india_matching")
            checker = india_matching.VolumeAlertChecker()
            
            # Run and capture data
            result_df = checker.run(INDIA_SYMBOLS, n_bars=n_bars)
            
            if result_df is not None and not result_df.empty:
                st.success("Scan Complete!")
                st.subheader("🟢 BULLISH: TARGETING RESISTANCE")
                st.dataframe(result_df[result_df["Imb"] == "ImbHigh"], hide_index=True, use_container_width=True)
                st.subheader("🔴 BEARISH: TARGETING SUPPORT")
                st.dataframe(result_df[result_df["Imb"] == "ImbLow"], hide_index=True, use_container_width=True)
            else:
                st.warning("No levels matched current parameters.")

# --- SCRIPT 4: INDIA SMA 200 ---
elif selected_script == "India SMA200":
    st.subheader("📈 India Stocks bouncing from SMA200 Scanner")
    st.markdown("Scans for stocks reversing after touhing 200 SMA levels on the daily timeframes.")
    
    # Custom parameter inputs just for this script
    n_bars = st.slider("Lookback Bars", min_value=20, max_value=200, value=100)
    
    if st.button("Execute SMA 200", type="primary"):
        with st.spinner("Processing SMA 200 scan..."):
            # Dynamically import and run your existing module
            india_sma200 = importlib.reload("Sl_India_SMA200")
            #checker = india_sma200.scan_stocks()            
            result_df = india_sma200.scan_stocks()
            
            if result_df is not None and not result_df.empty:
                st.success("Scan Complete!")
                st.dataframe(result_df, hide_index=True, use_container_width=True)
            else:
                st.warning("No levels matched current parameters.")
