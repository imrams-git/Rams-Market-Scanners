#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════╗
║       INDIAN STOCK PROFIT OPPORTUNITY ANALYZER           ║
║  Scores Nifty 500 stocks via Sector-Normalized Z-Scores  ║
╚══════════════════════════════════════════════════════════╝
"""

import sys
import os
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
    print("Missing dependencies.勸n  pip install yfinance pandas numpy tabulate colorama")
    sys.exit(1)

# ─────────────────────────────────────────────
#  STATIC SECTOR DICTIONARY
# ─────────────────────────────────────────────
INDIAN_STOCKS = {
    "Energy": [
        "ADANIENT.NS", "BPCL.NS", "COALINDIA.NS", "ONGC.NS", "RELIANCE.NS", "GMDCLTD.NS", 
        "HINDPETRO.NS", "OIL.NS", "IOC.NS", "MRPL.NS", "AEGISLOG.NS", "CHENNPETRO.NS", 
        "PETRONET.NS", "CASTROLIND.NS", "AEGISVOPAK.NS"
    ],
    "Industrials": [
        "ADANIPORTS.NS", "LT.NS", "MMTC.NS", "HEG.NS", "SCI.NS", "GRAPHITE.NS", "HBLENGINE.NS", 
        "SWANCORP.NS", "FINPIPE.NS", "INOXINDIA.NS", "NCC.NS", "TITAGARH.NS", "BEML.NS", 
        "KEI.NS", "GESHIP.NS", "MAZDOCK.NS", "RKFORGE.NS", "SUPREMEIND.NS", "INDIGO.NS", 
        "DCMSHRIRAM.NS", "ESCORTS.NS", "HAL.NS", "NBCC.NS", "BDL.NS", "KEC.NS", "PTCIL.NS", 
        "ABB.NS", "HONAUT.NS", "THERMAX.NS", "GVT&D.NS", "BEL.NS", "TRITURBINE.NS", "ASHOKLEY.NS", 
        "GRSE.NS", "ELECON.NS", "POWERINDIA.NS", "SRF.NS", "NAVA.NS", "DELHIVERY.NS", "VGUARD.NS", 
        "SUZLON.NS", "CONCOR.NS", "BLUEDART.NS", "JYOTICNC.NS", "BHEL.NS", "TARIL.NS", 
        "LATENTVIEW.NS", "CERA.NS", "APARINDS.NS", "TIINDIA.NS", "BLS.NS", "TECHNOE.NS", 
        "RHIM.NS", "DATAPATTNS.NS", "CYIENT.NS", "ZENTEC.NS", "POLYCAB.NS", "GMRAIRPORT.NS", 
        "COCHINSHIP.NS", "3MINDIA.NS", "GODREJIND.NS", "KPIL.NS", "SKFINDIA.NS", "TIMKEN.NS", 
        "CARBORUNIV.NS", "SIEMENS.NS", "CGPOWER.NS", "DOMS.NS", "ARE&M.NS", "AFCONS.NS", 
        "JSWINFRA.NS", "INOXWIND.NS", "KAJARIACER.NS", "SCHNEIDER.NS", "IRB.NS", "RRKABEL.NS", 
        "KSB.NS", "ELGIEQUIP.NS", "ENGINERSIN.NS", "KIRLOSBROS.NS", "AIAENG.NS", "PRAJIND.NS", 
        "GRAVITA.NS", "RITES.NS", "FINCABLES.NS", "CUMMINSIND.NS", "IRCON.NS", "OLECTRA.NS", 
        "ASTRAL.NS", "BLUESTARCO.NS", "HAVELLS.NS"
    ],
    "Healthcare": [
        "APOLLOHOSP.NS", "CIPLA.NS", "DIVISLAB.NS", "DRREDDY.NS", "SUNPHARMA.NS", "ASTRAZEN.NS", 
        "LALPATHLAB.NS", "NEULANDLAB.NS", "WOCKPHARMA.NS", "SYNGENE.NS", "POLYMED.NS", "VIJAYA.NS", 
        "NH.NS", "INDGN.NS", "AKUMS.NS", "GLENMARK.NS", "MANKIND.NS", "ALKEM.NS", "AJANTPHARM.NS", 
        "CONCORDBIO.NS", "LAURUSLABS.NS", "GLAXO.NS", "ERIS.NS", "GLAND.NS", "AGARWALEYE.NS", 
        "TORNTPHARM.NS", "ASTERDM.NS", "MAXHEALTH.NS", "ZYDUSLIFE.NS", "BIOCON.NS", "SAGILITY.NS", 
        "RAINBOW.NS", "JBCHEPHARM.NS", "LUPIN.NS", "PFIZER.NS", "BLUEJET.NS", "CAPLIPOINT.NS", 
        "PPLPHARMA.NS", "SAILIFE.NS", "COHANCE.NS", "NATCOPHARM.NS", "MEDANTA.NS", "METROPOLIS.NS", 
        "APLLTD.NS", "ABBOTINDIA.NS", "KIMS.NS", "FORTIS.NS", "EMCURE.NS", "ONESOURCE.NS", 
        "IPCALAB.NS", "JUBLPHARMA.NS", "GRANULES.NS", "IKS.NS", "AUROPHARMA.NS"
    ],
    "Basic Materials": [
        "ASIANPAINT.NS", "GRASIM.NS", "HINDALCO.NS", "JSWSTEEL.NS", "SHREECEM.NS", "TATASTEEL.NS", 
        "ULTRACEMCO.NS", "UPL.NS", "HINDCOPPER.NS", "FACT.NS", "GPIL.NS", "DEEPAKNTR.NS", 
        "WELCORP.NS", "USHAMART.NS", "RCF.NS", "ATUL.NS", "INDIACEM.NS", "SAIL.NS", "JINDALSTEL.NS", 
        "COROMANDEL.NS", "JUBLINGREA.NS", "JKCEMENT.NS", "SUMICHEM.NS", "APLAPOLLO.NS", "NSLNISP.NS", 
        "SHYAMMETL.NS", "CHAMBLFERT.NS", "BERGEPAINT.NS", "IGIL.NS", "FLUOROCHEM.NS", "RAMCOCEM.NS", 
        "PIDILITIND.NS", "DALBHARAT.NS", "BAYERCROP.NS", "NAVINFLUOR.NS", "MAHSEAMLES.NS", 
        "SOLARINDS.NS", "LLOYDSME.NS", "PIIND.NS", "ALKYLAMINE.NS", "TATACHEM.NS", "VEDL.NS", 
        "ACC.NS", "LINDEINDIA.NS", "AMBUJACEM.NS", "AARTIIND.NS", "CENTURYPLY.NS", "DEEPAKFERT.NS", 
        "HSCL.NS", "CLEAN.NS", "JSL.NS", "JINDALSAW.NS", "EIDPARRY.NS", "NATIONALUM.NS", 
        "NMDC.NS", "NUVOCO.NS", "HINDZINC.NS", "SARDAEN.NS", "BASF.NS"
    ],
    "Financial Services": [
        "AXISBANK.NS", "BAJAJFINSV.NS", "BAJFINANCE.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "ICICIBANK.NS", 
        "INDUSINDBK.NS", "KOTAKBANK.NS", "SBIN.NS", "SBILIFE.NS", "BANDHANBNK.NS", "FEDERALBNK.NS", 
        "IDFCFIRSTB.NS", "PNB.NS", "RBLBANK.NS", "AUBANK.NS", "SAMMAANCAP.NS", "ANANDRATHI.NS", 
        "SUNDARMFIN.NS", "MAHABANK.NS", "HUDCO.NS", "CREDITACC.NS", "CRISIL.NS", "AIIL.NS", 
        "NAM-INDIA.NS", "INDIANB.NS", "MAHSCOOTER.NS", "BANKBARODA.NS", "ABSLAMC.NS", "CANBK.NS", 
        "LTF.NS", "ICICIPRULI.NS", "MCX.NS", "POONAWALLA.NS", "CHOLAFIN.NS", "IDBI.NS", "BSE.NS", 
        "MFSL.NS", "HOMEFIRST.NS", "IOB.NS", "POLICYBZR.NS", "CHOICEIN.NS", "GICRE.NS", "CANFINHOME.NS", 
        "360ONE.NS", "MOTILALOFS.NS", "UCOBANK.NS", "HDFCAMC.NS", "LICI.NS", "ICICIGI.NS", "YESBANK.NS", 
        "BANKINDIA.NS", "RECLTD.NS", "SHRIRAMFIN.NS", "J&KBANK.NS", "NIACL.NS", "MANAPPURAM.NS", 
        "CUB.NS", "ABCAPITAL.NS", "LICHSGFIN.NS", "NIVABUPA.NS", "JIOFIN.NS", "UNIONBANK.NS", 
        "CHOLAHLDNG.NS", "GODIGIT.NS", "AADHARHFC.NS", "IIFL.NS", "PFC.NS", "CENTRALBK.NS", 
        "BAJAJHFL.NS", "MUTHOOTFIN.NS", "SBFC.NS", "CDSL.NS", "PNBHOUSING.NS", "TATAINVEST.NS", 
        "AAVAS.NS", "IREDA.NS", "UTIAMC.NS", "IEX.NS", "CGCL.NS", "NUVAMA.NS", "SBICARD.NS", 
        "FIVESTAR.NS", "KARURVYSYA.NS", "M&MFIN.NS", "APTUS.NS", "IFCI.NS", "STARHEALTH.NS", 
        "JMFINANCIL.NS", "BAJAJHLDNG.NS"
    ],
    "Consumer Cyclical": [
        "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS", "MARUTI.NS", "TITAN.NS", 
        "FORCEMOT.NS", "BALKRISIND.NS", "KPRMILL.NS", "MANYAVAR.NS", "WHIRLPOOL.NS", "VOLTAS.NS", 
        "ENDURANCE.NS", "SWIGGY.NS", "JUBLFOOD.NS", "SCHAEFFLER.NS", "TMPV.NS", "ZFCVINDIA.NS", 
        "JKTYRE.NS", "MOTHERSON.NS", "INDHOTEL.NS", "ETERNAL.NS", "VMM.NS", "VENTIVE.NS", 
        "HYUNDAI.NS", "FIRSTCRY.NS", "UNOMINDA.NS", "ABLBL.NS", "BHARATFORG.NS", "ABFRL.NS", 
        "EIHOTEL.NS", "BOSCHLTD.NS", "NYKAA.NS", "PAGEIND.NS", "SAPPHIRE.NS", "CHALET.NS", 
        "TBOTEK.NS", "ATHERENERG.NS", "EXIDEIND.NS", "WELSPUNLIV.NS", "IRCTC.NS", "CROMPTON.NS", 
        "SUNDRMFAST.NS", "VTL.NS", "KALYANKJIL.NS", "CEATLTD.NS", "DEVYANI.NS", "BATAINDIA.NS", 
        "TVSMOTOR.NS", "TRIDENT.NS", "LEMONTREE.NS", "MRF.NS", "MSUMI.NS", "TRENT.NS", 
        "ITCHOTELS.NS", "SONACOMS.NS", "ALOKINDS.NS", "AMBER.NS", "CAMPUS.NS", "THELEELA.NS", 
        "MINDACORP.NS", "JBMA.NS", "OLAELEC.NS", "APOLLOTYRE.NS"
    ],
    "Communication Services": [
        "BHARTIARTL.NS", "SAREGAMA.NS", "TATACOMM.NS", "INDUSTOWER.NS", "BHARTIHEXA.NS", 
        "PVRINOX.NS", "ZEEL.NS", "IDEA.NS", "INDIAMART.NS", "NAUKRI.NS", "AFFLE.NS", 
        "RAILTEL.NS", "SUNTV.NS", "TTML.NS"
    ],
    "Consumer Defensive": [
        "BRITANNIA.NS", "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "TATACONSUM.NS", "BIKAJI.NS", 
        "HONASA.NS", "MARICO.NS", "CCL.NS", "PGHH.NS", "VBL.NS", "JYOTHYLAB.NS", "GODREJCP.NS", 
        "DABUR.NS", "DMART.NS", "BBTC.NS", "GODREJAGRO.NS", "AWL.NS", "PATANJALI.NS", "UBL.NS", 
        "GILLETTE.NS", "COLPAL.NS", "BALRAMCHIN.NS", "GODFRYPHLP.NS", "RADICO.NS", "TRIVENI.NS", 
        "LTFOODS.NS", "EMAMILTD.NS", "UNITDSPR.NS"
    ],
    "Technology": [
        "HCLTECH.NS", "INFY.NS", "TCS.NS", "TECHM.NS", "WIPRO.NS", "HFCL.NS", "TEJASNET.NS", 
        "COFORGE.NS", "KFINTECH.NS", "PAYTM.NS", "INTELLECT.NS", "NEWGEN.NS", "ECLERX.NS", 
        "CAMS.NS", "FSL.NS", "LTTS.NS", "HAPPSTMNDS.NS", "TATAELXSI.NS", "REDINGTON.NS", 
        "PERSISTENT.NS", "TATATECH.NS", "ZENSARTECH.NS", "KPITTECH.NS", "HEXT.NS", "OFSS.NS", 
        "MPHASIS.NS", "PGEL.NS", "NETWEB.NS", "ITI.NS", "MAPMYINDIA.NS", "PREMIERENE.NS", 
        "BSOFT.NS", "DIXON.NS", "SONATSOFTW.NS", "KAYNES.NS", "WAAREEENER.NS"
    ],
    "Utilities": [
        "NTPC.NS", "POWERGRID.NS", "NTPCGREEN.NS", "NLCINDIA.NS", "ACMESOLAR.NS", "IGL.NS", 
        "GAIL.NS", "ATGL.NS", "MGL.NS", "TATAPOWER.NS", "CESC.NS", "ADANIENSOL.NS", "ADANIPOWER.NS", 
        "JSWENERGY.NS", "SJVN.NS", "ENRIN.NS", "ADANIGREEN.NS", "JPPOWER.NS", "TORNTPOWER.NS", 
        "NHPC.NS", "RPOWER.NS"
    ],
    "Real Estate": [
        "GODREJPROP.NS", "SIGNATURE.NS", "OBEROIRLTY.NS", "LODHA.NS", "PRESTIGE.NS", "DLF.NS", 
        "BRIGADE.NS", "SOBHA.NS", "ABREL.NS", "ANANTRAJ.NS", "DBREALTY.NS", "PHOENIXLTD.NS"
    ],
    "Other": [
        "AKZOINDIA.NS", "GUJGASLTD.NS", "GSPL.NS"
    ]
}

INDIA_SECTORS = {sector_name: tickers for sector_name, tickers in INDIAN_STOCKS.items()}
INDIA_SECTORS["all"] = list({t for tickers in INDIA_SECTORS.values() for t in tickers})

TICKER_TO_SECTOR = {}
for sec_name, tickers in INDIA_SECTORS.items():
    if sec_name != "all":
        for t in tickers:
            TICKER_TO_SECTOR[t.upper()] = sec_name

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

        if not info or (info.get("regularMarketPrice") is None and info.get("currentPrice") is None):
            return None

        static_sector = TICKER_TO_SECTOR.get(ticker.upper(), "Other")

        return {
            "Ticker":        ticker.replace(".NS", ""),
            "Name":          info.get("shortName", ticker)[:22],
            "Sector":        static_sector,
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

def fetch_technicals(ticker_clean: str) -> dict:
    ticker = f"{ticker_clean}.NS"
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
            "Ticker": ticker_clean,
            "RSI": rsi_val,
            "RVOL": rvol_val,
            "IV%": iv_val
        }
    except Exception:
        return {"Ticker": ticker_clean, "RSI": np.nan, "RVOL": np.nan, "IV%": np.nan}

def India_run_analysis(tickers=None, sector="all", top_n=10):
    if not tickers:
        tickers = INDIA_SECTORS.get(sector, INDIA_SECTORS["all"])

    raw_data = []
    completed = 0
    total = len(tickers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_ticker = {executor.submit(fetch_raw_data, t): t for t in tickers}
        for future in concurrent.futures.as_completed(future_to_ticker):
            completed += 1
            print(f"  [{completed:03d}/{total:03d}] Fetching...", end="\r")
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Indian Stock Quantitative Analyzer")
    parser.add_argument("--sector", default="all", choices=list(INDIA_SECTORS.keys()))
    parser.add_argument("--top", type=int, default=10, help="Show top N stocks")
    args = parser.parse_args()

    print(f"\n{Fore.CYAN}{'═'*80}")
    print(f"   📈  INDIAN STOCK QUANTITATIVE ANALYZER (NSE SECTOR-NORMALIZED)")
    print(f"{'═'*80}{Style.RESET_ALL}")
    print(f"  Fetching NSE data via threading...\n")

    df = India_run_analysis(sector=args.sector, top_n=args.top)

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

    headers = ["Rank", "Ticker", "Name", "Sector", "Price (₹)", "P/E", "PEG", 
               "Rev Gr%", "Margin%", "Score", "RSI", "RVOL", "IV%"]
    
    print("\n" + tabulate(table_data, headers=headers, tablefmt="rounded_outline"))

    # --- SIGNALS & PROGRESS BARS BLOCK ---
    print(f"\n{Fore.CYAN}  SIGNALS{Style.RESET_ALL}")
    print(f"  {'─'*45}")
    for _, row in df.iterrows():
        score = row['Total_Score']
        bar_len = int(score / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        
        sig_text = "PASS"
        if score >= 65: sig_text = f"{Fore.GREEN}★ STRONG BUY{Style.RESET_ALL}"
        elif score >= 55: sig_text = f"{Fore.CYAN}▲ BUY{Style.RESET_ALL}"
        elif score >= 45: sig_text = f"{Fore.YELLOW}● HOLD{Style.RESET_ALL}"
        else: sig_text = f"{Fore.RED}▼ PASS{Style.RESET_ALL}"

        print(f"  {row['Ticker']:<6} {bar}  {score:>5.1f}  {sig_text}")

if __name__ == "__main__":
    main()
