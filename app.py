# app.py

import streamlit as st
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
import time

# --------------- GENERAL SETTINGS --------------

st.set_page_config(page_title="Option Chain Decay Bias Dashboard", layout="wide")

# --------------- SIDEBAR CONFIGURATION --------------

st.sidebar.title("Dashboard Settings")

INDEX_MAP = {
    "Nifty": "NIFTY",
    "Bank Nifty": "BANKNIFTY",
    "Sensex": "SENSEX"  # Note: SENSEX OC not always available via NSE API, but handled for completeness
}
index_choice = st.sidebar.selectbox("Select Index", list(INDEX_MAP.keys()))
index_symbol = INDEX_MAP[index_choice]

refresh_rate = st.sidebar.slider("Auto Refresh Rate (sec)", min_value=15, max_value=300, value=60, step=15)
last_refresh = st.empty()

st.sidebar.markdown("---")
st.sidebar.markdown("##### Optional DhanHQ Credentials (for fallback):")
dhan_api_key = st.sidebar.text_input("DhanHQ Access Token", type="password")
dhan_client_id = st.sidebar.text_input("DhanHQ Client ID", type="password")
dhan_ready = bool(dhan_api_key and dhan_client_id)

# --------------- UTILITY AND API FUNCTIONS --------------

@st.cache_data(ttl=30)
def fetch_nse_oc(symbol):
    """
    Fetch option chain data from NSE for the given index symbol.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/80.0.3987.149 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br"
    }
    url = f"https://www.nseindia.com/api/option-chain-indices?symbol={symbol}"
    session = requests.Session()
    try:
        home = session.get("https://www.nseindia.com/option-chain", headers=headers, timeout=5)
        cookies = dict(home.cookies)
        resp = session.get(url, headers=headers, cookies=cookies, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as ex:
        raise RuntimeError(f"NSE OC Fetch failed: {ex}")

@st.cache_data(ttl=30)
def fetch_dhanhq_oc(symbol, expiry=None, api_key=None, client_id=None):
    """
    Fetch option chain data using DhanHQ API.
    Requires user provided api_key and client_id.
    """
    symbol_map = {
        "NIFTY": 13,
        "BANKNIFTY": 14,
        "SENSEX": 111  # Dummy ID, change based on Dhan's documentation if needed
    }
    underlying_scrip = symbol_map.get(symbol.upper(), 13)
    expiry_str = expiry if expiry else ""
    headers = {
        "access-token": api_key,
        "client-id": client_id,
        "Content-Type": "application/json"
    }
    payload = {
        "UnderlyingScrip": underlying_scrip,
        "UnderlyingSeg": "IDX_I",
        "Expiry": expiry_str
    }
    url = "https://api.dhan.co/v2/optionchain"
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    if resp.status_code != 200:
        raise RuntimeError(f"DhanHQ OC Fetch failed: {resp.text[:250]}")
    return resp.json()["data"]

def parse_nse_oc(raw, index_symbol):
    """
    Standardize NSE option chain data into DataFrame.
    """
    expiry = raw.get("records", {}).get("expiryDates", [""])[0]
    records = raw.get("records", {}).get("data", [])
    parsed = []
    for rec in records:
        strike = rec.get("strikePrice")
        ce = rec.get("CE", {})
        pe = rec.get("PE", {})
        ex_date = rec.get("expiryDate", expiry)
        parsed.append({
            "strike": strike,
            "expiry": ex_date,
            "ce_ltp": ce.get("lastPrice"), "ce_oi": ce.get("openInterest"),
            "ce_chg": ce.get("change"), "ce_theta": None, "ce_iv": ce.get("impliedVolatility"),
            "pe_ltp": pe.get("lastPrice"), "pe_oi": pe.get("openInterest"),
            "pe_chg": pe.get("change"), "pe_theta": None, "pe_iv": pe.get("impliedVolatility")
        })
    df = pd.DataFrame(parsed)
    ul = raw.get("records", {}).get("underlyingValue")
    return df, expiry, ul

def parse_dhanhq_oc(raw):
    """
    Standardize DhanHQ option chain response into DataFrame.
    """
    oc_dict = raw.get("oc", {})
    parsed = []
    expiry = raw.get("expiry_date", "")
    ul = raw.get("last_price") or np.nan
    for strike, v in oc_dict.items():
        ce = v.get("ce", {})
        pe = v.get("pe", {})
        parsed.append({
            "strike": float(strike),
            "expiry": expiry,
            "ce_ltp": ce.get("last_price"), "ce_oi": ce.get("oi"),
            "ce_chg": ce.get("last_price") - ce.get("previous_close_price", 0) if ce else None,
            "ce_theta": ce.get("greeks", {}).get("theta"),
            "ce_iv": ce.get("implied_volatility"),
            "pe_ltp": pe.get("last_price"), "pe_oi": pe.get("oi"),
            "pe_chg": pe.get("last_price") - pe.get("previous_close_price", 0) if pe else None,
            "pe_theta": pe.get("greeks", {}).get("theta"),
            "pe_iv": pe.get("implied_volatility")
        })
    df = pd.DataFrame(parsed)
    return df, expiry, ul

def bs_theta(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes theta computation per day (approx).
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                - r * K * np.exp(-r * T) * norm.cdf(d2))
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                + r * K * np.exp(-r * T) * norm.cdf(-d2))
    return theta / 365.0  # Per day

def fill_missing_theta(df, S, days, r=0.06):
    """
    Compute theta for rows where missing from API.
    """
    T = days / 365.0
    for col, opt_type in [("ce_theta", "call"), ("pe_theta", "put")]:
        missing = df[col].isnull()
        if missing.any():
            # Require strike, iv
            for idx, row in df[missing].iterrows():
                K = row["strike"]
                sigma = (row[f"{col[:2]}_iv"] or 21) / 100.0  # If missing, use 21%
                if any([val is None or np.isnan(val) for val in [S, K, sigma]]): continue
                theta = bs_theta(S, K, T, r, sigma, option_type=opt_type)
                df.at[idx, col] = theta
    return df

def get_atm_strikes(df, ul, window=2):
    """
    Returns the subset of the DataFrame near-the-money (ATM +/- window).
    """
    if df.empty: return df
    atm_strike = df.iloc[(df['strike']-ul).abs().argsort()[:1]]["strike"].values[0]
    strikes = sorted(df["strike"].unique())
    idx = strikes.index(atm_strike) if atm_strike in strikes else 0
    sel_strikes = strikes[max(0, idx-window):min(len(strikes), idx+window+1)]
    return df[df["strike"].isin(sel_strikes)]

def analyze_decay_bias(df, ul):
    """
    Compute total theta and change for CE and PE (ATM ±2), and suggest bias.
    """
    atm_df = get_atm_strikes(df, ul, window=2)
    metrics = {
        "sum_ce_theta": atm_df["ce_theta"].sum(skipna=True),
        "sum_pe_theta": atm_df["pe_theta"].sum(skipna=True),
        "sum_ce_chg": atm_df["ce_chg"].sum(skipna=True),
        "sum_pe_chg": atm_df["pe_chg"].sum(skipna=True)
    }
    bias = "Indeterminate"
    rec = ""
    if abs(metrics["sum_ce_theta"] - metrics["sum_pe_theta"]) > 0.1 * abs(metrics["sum_pe_theta"]):
        if metrics["sum_ce_theta"] < metrics["sum_pe_theta"]:
            bias = "Calls (CE)"
            rec = "Option sellers may favor writing Calls; consider Iron Condors with higher Call strikes."
        else:
            bias = "Puts (PE)"
            rec = "Option sellers may favor writing Puts; consider strategies like Bull Put Spread."
    else:
        bias = "Neutral"
        rec = "Straddle or Strangle selling may be optimal if IV and liquidity are sufficient."

    return metrics, bias, rec

# --------------- DATA FETCHING AND FALLBACK LOGIC --------------

def load_option_chain(symbol, use_fallback=False):
    """
    Wrapper that attempts primary NSE fetch, then DhanHQ fallback.
    """
    error_message = ""
    df = expiry = ul = src = None
    try:
        data = fetch_nse_oc(symbol)
        df, expiry, ul = parse_nse_oc(data, symbol)
        src = "NSE"
    except Exception as e:
        error_message = str(e)
        if dhan_ready:
            try:
                data = fetch_dhanhq_oc(symbol, api_key=dhan_api_key, client_id=dhan_client_id)
                df, expiry, ul = parse_dhanhq_oc(data)
                src = "DhanHQ"
            except Exception as e2:
                error_message += "; Fallback failed: " + str(e2)
        else:
            error_message += " | DhanHQ fallback not attempted (credentials missing)."
    return df, expiry, ul, src, error_message

# --------------- MAIN DASHBOARD LOGIC --------------

def main():
    st.title("📊 Option Chain Decay Bias Dashboard")
    st.markdown(
        """
        This dashboard provides **real-time option chain analytics** for Nifty, Bank Nifty, and Sensex, including 
        <span style="color:orange">CE/PE decay bias detection (theta analysis)</span> and actionable 
        <span style="color:green">strategy recommendations</span>. Data sources automatically failover to a backup if the primary is down.
        """,
        unsafe_allow_html=True,
    )
    load_expander = st.expander("ℹ️ How does the fallback work?", expanded=False)
    with load_expander:
        st.markdown(
            """
            - The dashboard fetches from the **NSE API** first.
            - If the NSE data is unavailable, and you supply DhanHQ credentials, it switches to DhanHQ's API.
            - If both fail, you will see a detailed error message below.
            """
        )

    index_col, ul_col, exp_col, bias_col = st.columns([2, 2, 3, 2])

    with st.spinner(f"Fetching option chain data for {index_symbol} ..."):
        df, expiry, ul, data_src, error = load_option_chain(index_symbol)
        if df is None or df.empty:
            st.error(f"Failed to fetch option chain data for {index_symbol}. {error}")
            st.stop()

    # Show underlying price
    index_col.metric(label="Index", value=index_symbol)
    ul_col.metric(label="Underlying Last Price", value=np.round(ul,2) if ul else "-")
    exp_col.metric(label="Current Expiry", value=expiry)
    bias_col.markdown(
        f"<span style='font-size:1.15em;'>Source: <strong>{data_src or 'N/A'}</strong></span>",
        unsafe_allow_html=True
    )

    # Fill missing theta
    try:
        # Estimate days to expiry
        days_to_expiry = (
            (pd.to_datetime(expiry, dayfirst=False) - datetime.now()).days
            if expiry else 2
        )
    except:
        days_to_expiry = 2
    if days_to_expiry < 1: days_to_expiry = 1
    df = fill_missing_theta(df, S=ul, days=days_to_expiry)

    # ATM rows and bias
    metrics, bias, rec = analyze_decay_bias(df, ul)
    st.divider()
    st.subheader("Decay Bias and Strategy Suggestion")
    st.markdown(
        f"""
        - **CE (Call) total theta (ATM±2):** <span style="color:orange">{metrics['sum_ce_theta']:.2f}</span>
        - **PE (Put) total theta (ATM±2):** <span style="color:blue">{metrics['sum_pe_theta']:.2f}</span>
        - **Decay Bias:** <strong style="color:purple">{bias}</strong>
        
        <span style="color:green"><b>Strategy Recommendation:</b></span> {rec}
        """, unsafe_allow_html=True
    )

    # Show ATM table for details
    st.divider()
    st.subheader(f"Detail: ATM & Near Strikes Option Data")
    atm_df = get_atm_strikes(df, ul, window=2).copy()
    atm_df_disp = atm_df[[
        "strike", "ce_ltp", "ce_theta", "ce_oi", "ce_chg", "ce_iv",
        "pe_ltp", "pe_theta", "pe_oi", "pe_chg", "pe_iv"
    ]]
    st.dataframe(atm_df_disp.rename(columns={
        "strike": "Strike",
        "ce_ltp": "CE LTP", "ce_theta": "CE Theta", "ce_oi": "CE OI", "ce_chg": "CE Chg", "ce_iv": "CE IV",
        "pe_ltp": "PE LTP", "pe_theta": "PE Theta", "pe_oi": "PE OI", "pe_chg": "PE Chg", "pe_iv": "PE IV"
    }), width=1200, height=220)

    # Visualize theta decay profile
    st.divider()
    st.subheader("Theta Decay Profile (by Strike)")
    chart_df = atm_df.set_index("strike")[["ce_theta", "pe_theta"]].rename(columns={"ce_theta":"CE", "pe_theta":"PE"})
    st.line_chart(chart_df, width=0, height=300)

    # Visualize premium change
    st.subheader("Premium Change (by Strike)")
    chg_df = atm_df.set_index("strike")[["ce_chg", "pe_chg"]].rename(columns={"ce_chg":"CE", "pe_chg":"PE"})
    st.line_chart(chg_df, width=0, height=180)

    # Last refresh
    last_refresh.markdown(f"_Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
    st.experimental_rerun() if st.session_state.get('last_run', 0) + refresh_rate < time.time() else st.session_state.update({'last_run': time.time()})
    time.sleep(refresh_rate)

if __name__ == "__main__":
    main()
