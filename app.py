# ============================================================
# INT375: DATA SCIENCE TOOLBOX: PYTHON PROGRAMMING
# PROJECT: Mutual Fund Performance Analyzer
# Student: Jaskaran Singh | Project
# Data Source: mfapi.in (REST API)
# ============================================================

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Mutual Fund Analyzer | INT375",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import Fonts ─────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ───────────────────────────────────── */
:root {
    --bg-primary: #0b0f19;
    --bg-card: rgba(17, 24, 39, 0.7);
    --bg-card-hover: rgba(25, 33, 52, 0.85);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-purple: #8b5cf6;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --border-subtle: rgba(148, 163, 184, 0.08);
    --glow-blue: 0 0 20px rgba(59, 130, 246, 0.15);
}

/* ── Global Styles ────────────────────────────────────── */
.stApp {
    font-family: 'DM Sans', sans-serif !important;
}
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
    border-right: 1px solid var(--border-subtle) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li,
section[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #f1f5f9 !important;
}

/* ── Hero Header ──────────────────────────────────────── */
.hero-container {
    position: relative;
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #0f172a 100%);
    border: 1px solid rgba(139, 92, 246, 0.15);
    border-radius: 20px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 2rem;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(59,130,246,0.06) 0%, transparent 50%),
                radial-gradient(circle at 70% 80%, rgba(139,92,246,0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #60a5fa, #a78bfa, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.5px;
    position: relative;
}
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
    position: relative;
}
.hero-subtitle a {
    color: #60a5fa;
    text-decoration: none;
    border-bottom: 1px dotted rgba(96, 165, 250, 0.4);
    transition: all 0.2s;
}
.hero-subtitle a:hover {
    color: #93bbfc;
    border-bottom-color: #93bbfc;
}

/* ── Unit Badge ───────────────────────────────────────── */
.unit-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: linear-gradient(135deg, rgba(59,130,246,0.12), rgba(139,92,246,0.12));
    color: #93c5fd;
    padding: 6px 16px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
    border: 1px solid rgba(59,130,246,0.2);
    margin-bottom: 1rem;
    text-transform: uppercase;
}

/* ── Metric Cards ─────────────────────────────────────── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 14px;
    margin: 1rem 0 1.5rem 0;
}
.metric-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.6), rgba(15,23,42,0.8));
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-blue), var(--accent-purple));
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-card:hover {
    border-color: rgba(59,130,246,0.25);
    transform: translateY(-2px);
    box-shadow: var(--glow-blue);
}
.metric-card:hover::before {
    opacity: 1;
}
.metric-icon {
    font-size: 1.6rem;
    margin-bottom: 6px;
}
.metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1.2;
}
.metric-label {
    font-size: 0.75rem;
    color: #64748b;
    font-weight: 500;
    margin-top: 4px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Section Header ───────────────────────────────────── */
.section-header {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #e2e8f0;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-subtle);
}

/* ── Info Cards / Link Cards ──────────────────────────── */
.info-card {
    background: linear-gradient(135deg, rgba(30,27,75,0.5), rgba(15,23,42,0.7));
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.info-card h4 {
    color: #e2e8f0;
    margin: 0 0 0.8rem 0;
    font-size: 1rem;
    font-weight: 600;
}
.info-card p, .info-card li {
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}
.link-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(59,130,246,0.1);
    color: #60a5fa !important;
    padding: 5px 14px;
    border-radius: 100px;
    font-size: 0.82rem;
    font-weight: 500;
    text-decoration: none !important;
    border: 1px solid rgba(59,130,246,0.2);
    transition: all 0.2s;
    margin: 3px 4px;
}
.link-pill:hover {
    background: rgba(59,130,246,0.2);
    color: #93c5fd !important;
    border-color: rgba(59,130,246,0.4);
    transform: translateY(-1px);
}

/* ── Table/Dataframe Styling ──────────────────────────── */
div[data-testid="stDataFrame"] {
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Tab Styling ──────────────────────────────────────── */
div[data-testid="stTabs"] button {
    font-weight: 600;
    font-size: 0.85rem;
    font-family: 'DM Sans', sans-serif !important;
    border-radius: 8px 8px 0 0 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    border-bottom: 2px solid var(--accent-blue) !important;
}

/* ── Sidebar Enhancements ─────────────────────────────── */
.sidebar-brand {
    text-align: center;
    padding: 0.5rem 0 1rem 0;
}
.sidebar-brand-icon {
    font-size: 2.5rem;
    display: block;
    margin-bottom: 0.3rem;
}
.sidebar-brand-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.3px;
}
.sidebar-brand-sub {
    font-size: 0.72rem;
    color: #64748b;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.5px;
}
.sidebar-links {
    background: rgba(15,23,42,0.5);
    border: 1px solid rgba(148,163,184,0.06);
    border-radius: 12px;
    padding: 0.8rem;
    margin-top: 0.5rem;
}
.sidebar-links a {
    display: block;
    color: #94a3b8 !important;
    text-decoration: none !important;
    font-size: 0.8rem;
    padding: 4px 0;
    transition: color 0.2s;
}
.sidebar-links a:hover {
    color: #60a5fa !important;
}

/* ── Footer ───────────────────────────────────────────── */
.app-footer {
    margin-top: 3rem;
    padding: 1.5rem 0;
    border-top: 1px solid var(--border-subtle);
    text-align: center;
}
.app-footer p {
    color: #475569;
    font-size: 0.78rem;
    margin: 2px 0;
}
.app-footer a {
    color: #60a5fa;
    text-decoration: none;
    border-bottom: 1px dotted rgba(96,165,250,0.3);
}
.app-footer a:hover {
    color: #93bbfc;
}

/* ── Expander Styling ─────────────────────────────────── */
details {
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    background: rgba(15,23,42,0.4) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Fund Dictionary ───────────────────────────────────────────
FUNDS = {
    "SBI Bluechip Fund":            "125497",
    "Mirae Asset Large Cap":        "118989",
    "Axis Bluechip Fund":           "120465",
    "HDFC Mid-Cap Opportunities":   "118560",
    "Parag Parikh Flexi Cap":       "122639",
}

# Links for each fund on Moneycontrol/AMFI
FUND_LINKS = {
    "SBI Bluechip Fund":            "https://api.mfapi.in/mf/125497",
    "Mirae Asset Large Cap":        "https://api.mfapi.in/mf/118989",
    "Axis Bluechip Fund":           "https://api.mfapi.in/mf/120465",
    "HDFC Mid-Cap Opportunities":   "https://api.mfapi.in/mf/118560",
    "Parag Parikh Flexi Cap":       "https://api.mfapi.in/mf/122639",
}

COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#8b5cf6', '#f43f5e']

# ============================================================
# DATA COLLECTION (REST API)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nav_data(scheme_code: str, fund_name: str):
    #Fetch historical NAV from mfapi.in REST API
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        df = pd.DataFrame(payload["data"])
        df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
        df["nav"]  = pd.to_numeric(df["nav"], errors="coerce")
        df["fund_name"] = fund_name
        df = df.sort_values("date").reset_index(drop=True)
        return df, payload["meta"], resp.status_code
    except Exception as e:
        st.error(f"❌ API Error for {fund_name}: {e}")
        return None, None, None

# ============================================================
# DATA MANIPULATION (NumPy + Pandas)
# ============================================================
def process_data(df, start, end):
    #Clean and engineer features
    df = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].copy()
    df = df.dropna(subset=["nav"])
    df["daily_return"]      = df["nav"].pct_change() * 100
    df["ma_30"]             = df["nav"].rolling(30).mean()
    df["ma_90"]             = df["nav"].rolling(90).mean()
    df["cumulative_return"] = ((df["nav"] / df["nav"].iloc[0]) - 1) * 100
    df["volatility"]        = df["daily_return"].rolling(30).std()
    return df.reset_index(drop=True)

def calculate_metrics(df):
    #Key financial metrics using NumPy
    nav   = np.array(df["nav"])
    rets  = np.array(df["daily_return"].dropna())
    days  = (df["date"].iloc[-1] - df["date"].iloc[0]).days
    yrs   = max(days / 365, 0.01)

    total_ret  = ((nav[-1] - nav[0]) / nav[0]) * 100
    cagr       = ((nav[-1] / nav[0]) ** (1 / yrs) - 1) * 100
    ann_vol    = np.std(rets) * np.sqrt(252)
    sharpe     = (np.mean(rets) * 252) / (np.std(rets) * np.sqrt(252) + 1e-9)
    max_dd     = ((nav - np.maximum.accumulate(nav)) / np.maximum.accumulate(nav) * 100).min()

    return {
        "Current NAV (₹)":       round(float(nav[-1]), 2),
        "Total Return (%)":       round(total_ret, 2),
        "CAGR (%)":               round(cagr, 2),
        "Annual Volatility (%)":  round(ann_vol, 2),
        "Sharpe Ratio":           round(sharpe, 2),
        "Max Drawdown (%)":       round(max_dd, 2),
    }

# Helper: Render metric cards as HTML
def render_metrics_html(metrics_list):
    cards = ""
    for icon, value, label in metrics_list:
        cards += f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>"""
    st.markdown(f'<div class="metric-grid">{cards}</div>', unsafe_allow_html=True)

# ── Configure matplotlib for dark theme ───────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0b0f19',
    'axes.facecolor': '#111827',
    'axes.edgecolor': '#1e293b',
    'axes.labelcolor': '#94a3b8',
    'xtick.color': '#64748b',
    'ytick.color': '#64748b',
    'text.color': '#e2e8f0',
    'grid.color': '#1e293b',
    'grid.alpha': 0.5,
    'legend.facecolor': '#111827',
    'legend.edgecolor': '#1e293b',
    'font.family': 'sans-serif',
})

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">📊</span>
        <div class="sidebar-brand-title">MF Analyzer</div>
        <div class="sidebar-brand-sub">INT375 · DATA SCIENCE TOOLBOX</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    selected_funds = st.multiselect(
        "🏦 Select Mutual Funds",
        list(FUNDS.keys()),
        default=["SBI Bluechip Fund", "Axis Bluechip Fund", "Mirae Asset Large Cap"]
    )
    st.markdown("")
    years_back = st.slider("📅 Years of Data", 1, 10, 5)
    end_dt   = datetime.now()
    start_dt = end_dt - timedelta(days=years_back * 365)

    st.markdown("---")

    page = st.radio("🗂️ Navigate", [
        "🏠 Home",
        "📥 Data Collection",
        "🧹 Data Processing",
        "📈 Visualizations",
        "🔍 EDA",
        "🏆 Fund Comparison",
        "🤖 NAV Prediction",
    ])

    st.markdown("---")

    st.markdown("""
    <div class="sidebar-links">
        <strong style="color:#e2e8f0;font-size:0.82rem;">🔗 Quick Links</strong><br>
        <a href="https://www.mfapi.in/" target="_blank">📡 mfapi.in — Data Source</a>
        <a href="https://www.amfiindia.com/net-asset-value/mutual-fund-scheme-performance/returns.html" target="_blank">🏛️ AMFI India — Official NAV</a>
        <a href="https://www.amfiindia.com/net-asset-value/mutual-fund-scheme-performance/returns.html" target="_blank">💰 AMFI — MF Performance</a>
        <a href="https://docs.python.org/3/" target="_blank">🐍 Python 3 Docs</a>
        <a href="https://docs.streamlit.io/" target="_blank">⚡ Streamlit Docs</a>
        <a href="https://pandas.pydata.org/docs/" target="_blank">🐼 Pandas Docs</a>
        <a href="https://numpy.org/doc/" target="_blank">🔢 NumPy Docs</a>
        <a href="https://matplotlib.org/stable/contents.html" target="_blank">📉 Matplotlib Docs</a>
        <a href="https://seaborn.pydata.org/" target="_blank">🎨 Seaborn Docs</a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")
    st.caption("Data: mfapi.in · AMFI India")

if not selected_funds:
    st.warning("⚠️ Please select at least one fund from the sidebar.")
    st.stop()

# ── Load & process all funds ──────────────────────────────────
all_data, all_meta, all_status = {}, {}, {}
progress = st.empty()
for i, fund in enumerate(selected_funds):
    progress.info(f"⏳ Fetching data for **{fund}**…")
    raw, meta, status = fetch_nav_data(FUNDS[fund], fund)
    if raw is not None:
        all_data[fund]   = process_data(raw, start_dt, end_dt)
        all_meta[fund]   = meta
        all_status[fund] = status
progress.empty()

if not all_data:
    st.error("No data loaded. Check your internet connection.")
    st.stop()

# ============================================================
# PAGE: HOME
# ============================================================
if page == "🏠 Home":
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">📊 Mutual Fund Performance Analyzer</div>
        <div class="hero-subtitle">
            INT375: Data Science Toolbox · Python Programming · Project<br>
            Powered by <a href="https://www.mfapi.in/" target="_blank">mfapi.in</a> REST API
            · <a href="https://www.amfiindia.com/" target="_blank">AMFI India</a> official data
        </div>
    </div>
    """, unsafe_allow_html=True)

    render_metrics_html([
        ("🏦", str(len(all_data)),    "Funds Loaded"),
        ("📅", str(years_back),       "Years of Data"),
        ("🌐", "mfapi.in",           "Data Source"),
        ("📡", "✅ Live",             "API Status"),
    ])

    # ── Project Objective & Libraries ────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>🎯 Project Objective</h4>
            <p>Analyze Indian Mutual Fund performance using Python across 5 units:</p>
            <p>
                <strong style="color:#60a5fa"> → </strong> → REST API data collection<br>
                <strong style="color:#34d399"> → </strong> → NumPy &amp; Pandas manipulation<br>
                <strong style="color:#fbbf24"> → </strong> → Matplotlib &amp; Seaborn visualization<br>
                <strong style="color:#a78bfa"> → </strong> → EDA, correlation, outlier detection<br>
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>🛠️ Libraries &amp; Resources</h4>
            <p>
                <a class="link-pill" href="https://requests.readthedocs.io/" target="_blank">📡 requests</a>
                <a class="link-pill" href="https://pandas.pydata.org/docs/" target="_blank">🐼 pandas</a>
                <a class="link-pill" href="https://numpy.org/doc/" target="_blank">🔢 numpy</a>
                <a class="link-pill" href="https://matplotlib.org/" target="_blank">📉 matplotlib</a>
                <a class="link-pill" href="https://seaborn.pydata.org/" target="_blank">🎨 seaborn</a>
                <a class="link-pill" href="https://docs.streamlit.io/" target="_blank">⚡ streamlit</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    # ── Funds Overview Table ──────────────────────────────────
    st.markdown('<div class="section-header">📋 Funds Overview</div>', unsafe_allow_html=True)

    # Fund links row
    fund_pills = " ".join([
        f'<a class="link-pill" href="{FUND_LINKS.get(f, "#")}" target="_blank">🏦 {f}</a>'
        for f in all_data.keys()
    ])
    st.markdown(f'<div style="margin-bottom:1rem">{fund_pills}</div>', unsafe_allow_html=True)

    rows = []
    for fund, df in all_data.items():
        m = calculate_metrics(df)
        m["Fund"] = fund
        rows.append(m)
    overview = pd.DataFrame(rows).set_index("Fund")
    st.dataframe(overview.style.highlight_max(subset=["CAGR (%)", "Sharpe Ratio"], color="#166534")
                                .highlight_min(subset=["Max Drawdown (%)", "Annual Volatility (%)"], color="#991b1b"),
                 use_container_width=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        <p>Built with <a href="https://docs.streamlit.io/" target="_blank">Streamlit</a> · Data from <a href="https://www.mfapi.in/" target="_blank">mfapi.in</a> · Official NAV via <a href="https://www.amfiindia.com/" target="_blank">AMFI India</a></p>
        <p>INT375: Data Science Toolbox · Python Programming · Solo Project · © 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: — DATA COLLECTION
# ============================================================
elif page == "📥 Data Collection":
    st.markdown('<div class="unit-badge">📥 Data Collection via REST API</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Extracting Data from REST API</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>📡 API Source</h4>
        <p>We use the <a href="https://www.mfapi.in/" target="_blank" style="color:#60a5fa"><strong>mfapi.in</strong></a> free REST API,
           which is a wrapper over official <a href="https://www.amfiindia.com/" target="_blank" style="color:#60a5fa">AMFI India</a> NAV data.
           No API key is required — the endpoint returns JSON with daily NAV history for any scheme code.</p>
        <p style="margin-top:0.5rem;">
            <a class="link-pill" href="https://www.mfapi.in/" target="_blank">🌐 mfapi.in</a>
            <a class="link-pill" href="https://api.mfapi.in/mf/125497" target="_blank">🔗 Example API Call</a>
            <a class="link-pill" href="https://www.amfiindia.com/net-asset-value/mutual-fund-scheme-performance/returns.html" target="_blank">🏛️ AMFI NAV Portal</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Show Code Used", expanded=True):
        st.code("""
# Extracting data from REST API
import requests
import pandas as pd

scheme_code = "125497"   # SBI Bluechip Fund
url = f"https://api.mfapi.in/mf/{scheme_code}"

response = requests.get(url, timeout=15)
data = response.json()

# Convert to DataFrame
df = pd.DataFrame(data['data'])
df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
df['nav']  = pd.to_numeric(df['nav'], errors='coerce')
df = df.sort_values('date').reset_index(drop=True)
print(df.head())
        """, language="python")

    st.markdown('<div class="section-header">🔴 Live API Response</div>', unsafe_allow_html=True)
    for fund in all_data:
        meta   = all_meta[fund]
        status = all_status[fund]
        df     = all_data[fund]
        with st.expander(f"📡 {fund} — API Response"):
            render_metrics_html([
                ("✅", str(status),                            "HTTP Status"),
                ("📊", f"{len(df):,}",                         "Records"),
                ("📅", f"{df['date'].min().strftime('%Y-%m-%d')} → {df['date'].max().strftime('%Y-%m-%d')}", "Date Range"),
            ])
            if meta:
                col1, col2 = st.columns(2)
                col1.write(f"**Fund House:** {meta.get('fund_house','N/A')}")
                col1.write(f"**Scheme Name:** {meta.get('scheme_name','N/A')}")
                col2.write(f"**Category:** {meta.get('scheme_category','N/A')}")
                fund_link = FUND_LINKS.get(fund)
                if fund_link:
                    col2.markdown(f"[🔗 View on mfapi.in]({fund_link})")

    st.markdown('<div class="section-header">📋 Raw Fetched Data (First 15 rows)</div>', unsafe_allow_html=True)
    fund_sel = st.selectbox("Select Fund", list(all_data.keys()))
    raw_display = all_data[fund_sel][["date", "nav", "fund_name"]].head(15)
    st.dataframe(raw_display, use_container_width=True)

# ============================================================
# PAGE: — DATA PROCESSING
# ============================================================
elif page == "🧹 Data Processing":
    st.markdown('<div class="unit-badge">🧹 Data Manipulation with NumPy & Pandas</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Cleaning, Engineering & Processing</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>🧰 Key Techniques</h4>
        <p>Data wrangling with
           <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pct_change.html" target="_blank" style="color:#60a5fa">pct_change()</a>,
           <a href="https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html" target="_blank" style="color:#60a5fa">rolling()</a>,
           <a href="https://numpy.org/doc/stable/reference/generated/numpy.std.html" target="_blank" style="color:#60a5fa">np.std()</a> and more.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("📄 Show Code Used", expanded=True):
        st.code("""
# Data Cleaning & Feature Engineering
import pandas as pd
import numpy as np

# Handle missing values
df = df.dropna(subset=['nav'])

# Daily returns
df['daily_return'] = df['nav'].pct_change() * 100

# Moving averages
df['ma_30'] = df['nav'].rolling(window=30).mean()
df['ma_90'] = df['nav'].rolling(window=90).mean()

# Cumulative return
df['cumulative_return'] = ((df['nav'] / df['nav'].iloc[0]) - 1) * 100

# Rolling volatility
df['volatility'] = df['daily_return'].rolling(window=30).std()

# Key metrics with NumPy
nav_array = np.array(df['nav'])
cagr = ((nav_array[-1] / nav_array[0]) ** (1/years) - 1) * 100
sharpe = (np.mean(returns) * 252) / (np.std(returns) * np.sqrt(252))
print(df.head())
        """, language="python")

    fund_sel = st.selectbox("Select Fund to Inspect", list(all_data.keys()))
    df = all_data[fund_sel]

    render_metrics_html([
        ("📋", f"{len(df):,}",   "Total Records"),
        ("❌", str(df["nav"].isna().sum()),   "Missing NAV"),
        ("🔢", str(len(df.columns)),          "Columns"),
        ("💾", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB", "Memory"),
    ])

    st.markdown('<div class="section-header">📋 Processed DataFrame (Last 20 rows)</div>', unsafe_allow_html=True)
    cols_show = ["date", "nav", "daily_return", "ma_30", "ma_90", "cumulative_return", "volatility"]
    st.dataframe(df[cols_show].tail(20).round(4), use_container_width=True)

    st.markdown('<div class="section-header">🔢 Data Types & Null Info</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**dtypes:**")
        st.dataframe(df.dtypes.rename("dtype").to_frame(), use_container_width=True)
    with c2:
        st.write("**Null counts:**")
        st.dataframe(df.isnull().sum().rename("nulls").to_frame(), use_container_width=True)

# ============================================================
# PAGE: — VISUALIZATIONS
# ============================================================
elif page == "📈 Visualizations":
    st.markdown('<div class="unit-badge">📈 Visualization with Matplotlib & Seaborn</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Charts & Plots</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <h4>📊 Visualization Libraries</h4>
        <p>
            <a class="link-pill" href="https://matplotlib.org/stable/gallery/index.html" target="_blank">📉 Matplotlib Gallery</a>
            <a class="link-pill" href="https://seaborn.pydata.org/examples/index.html" target="_blank">🎨 Seaborn Examples</a>
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["📈 NAV Trend", "📊 Returns Dist.", "🔥 Correlation Heatmap", "📉 Volatility"])

    with tab1:
        fund_sel = st.selectbox("Fund", list(all_data.keys()), key="nav_sel")
        df = all_data[fund_sel]
        fig, ax = plt.subplots(figsize=(12, 4.5))
        ax.plot(df["date"], df["nav"],    color=COLORS[0], lw=1.8, label="NAV")
        ax.plot(df["date"], df["ma_30"],  color=COLORS[2], lw=1.3, ls="--", label="30-Day MA")
        ax.plot(df["date"], df["ma_90"],  color=COLORS[3], lw=1.3, ls="--", label="90-Day MA")
        ax.fill_between(df["date"], df["nav"], alpha=0.06, color=COLORS[0])
        ax.set_title(f"{fund_sel} — NAV with Moving Averages", fontweight="bold", fontsize=12)
        ax.set_xlabel("Date"); ax.set_ylabel("NAV (₹)")
        ax.legend(facecolor='#111827', edgecolor='#1e293b', fontsize=9); ax.grid(True, alpha=0.15)
        st.pyplot(fig); plt.close()

    with tab2:
        n = len(all_data)
        fig, axes = plt.subplots(1, n, figsize=(5*n, 4.5), sharey=False)
        if n == 1: axes = [axes]
        for i, (fund, df) in enumerate(all_data.items()):
            sns.histplot(df["daily_return"].dropna(), ax=axes[i], kde=True,
                         bins=60, color=COLORS[i % len(COLORS)], alpha=0.7)
            axes[i].axvline(df["daily_return"].mean(), color="#f43f5e", ls="--", lw=1.5, label="Mean")
            axes[i].set_title(fund[:18], fontsize=10, fontweight="bold")
            axes[i].set_xlabel("Daily Return (%)")
            axes[i].legend(fontsize=8, facecolor='#111827', edgecolor='#1e293b')
        plt.suptitle("Daily Returns Distribution — Seaborn", fontweight="bold", y=1.02, fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab3:
        nav_pivot = pd.DataFrame({f: all_data[f].set_index("date")["nav"] for f in all_data})
        nav_pivot.columns = [c[:16] for c in nav_pivot.columns]
        corr = nav_pivot.corr()
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.heatmap(corr, annot=True, fmt=".3f", cmap="coolwarm",
                    vmin=-1, vmax=1, square=True, linewidths=0.5, ax=ax,
                    annot_kws={"size": 10, "weight": "bold"})
        ax.set_title("Fund NAV Correlation Heatmap", fontweight="bold", fontsize=12)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with tab4:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for i, (fund, df) in enumerate(all_data.items()):
            ax.plot(df["date"], df["volatility"], label=fund[:20], color=COLORS[i % len(COLORS)], lw=1.5)
        ax.set_title("30-Day Rolling Volatility", fontweight="bold", fontsize=12)
        ax.set_xlabel("Date"); ax.set_ylabel("Volatility (σ)")
        ax.legend(fontsize=8, facecolor='#111827', edgecolor='#1e293b'); ax.grid(True, alpha=0.15)
        st.pyplot(fig); plt.close()

# ============================================================
# PAGE: — EDA
# ============================================================
elif page == "🔍 EDA":
    st.markdown('<div class="unit-badge">🔍 Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">EDA: Summary Statistics, Correlation & Outlier Detection</div>', unsafe_allow_html=True)

    fund_sel = st.selectbox("Select Fund", list(all_data.keys()))
    df      = all_data[fund_sel]
    returns = df["daily_return"].dropna()

    # ── Summary Statistics ────────────────────────────────────
    st.markdown('<div class="section-header">📋 Summary Statistics</div>', unsafe_allow_html=True)
    with st.expander("📄 Code", expanded=False):
        st.code("""
# Summary Statistics using Pandas describe()
summary = df[['nav','daily_return','cumulative_return','volatility']].describe()
print(summary)
        """, language="python")
    summary = df[["nav", "daily_return", "cumulative_return", "volatility"]].describe()
    st.dataframe(summary.round(4), use_container_width=True)

    render_metrics_html([
        ("📈", f"{returns.mean():.4f}%", "Mean Daily Return"),
        ("📊", f"{returns.std():.4f}%",  "Std Deviation"),
        ("↗️", f"{returns.skew():.4f}",   "Skewness"),
        ("📐", f"{returns.kurtosis():.4f}", "Kurtosis"),
    ])

    # ── Outlier Detection ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🚨 Outlier Detection — IQR Method</div>', unsafe_allow_html=True)
    with st.expander("📄 Code", expanded=False):
        st.code("""
# Outlier detection using IQR method
Q1 = returns.quantile(0.25)
Q3 = returns.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
print("Number of outliers:", len(outliers))
        """, language="python")

    Q1, Q3 = returns.quantile(0.25), returns.quantile(0.75)
    IQR     = Q3 - Q1
    lower   = Q1 - 1.5 * IQR
    upper   = Q3 + 1.5 * IQR
    outliers = returns[(returns < lower) | (returns > upper)]

    render_metrics_html([
        ("📏", f"{IQR:.4f}",       "IQR"),
        ("⬇️", f"{lower:.4f}%",    "Lower Bound"),
        ("⬆️", f"{upper:.4f}%",    "Upper Bound"),
        ("⚠️", f"{len(outliers)} ({len(outliers)/len(returns)*100:.1f}%)", "Outliers"),
    ])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(y=returns,                                    ax=axes[0], color="#3b82f6", flierprops=dict(markerfacecolor='#f43f5e', markersize=4))
    axes[0].set_title("With Outliers",    fontweight="bold"); axes[0].set_ylabel("Daily Return (%)")
    sns.boxplot(y=returns[(returns >= lower) & (returns <= upper)], ax=axes[1], color="#10b981", flierprops=dict(markerfacecolor='#f43f5e', markersize=4))
    axes[1].set_title("Outliers Removed", fontweight="bold"); axes[1].set_ylabel("Daily Return (%)")
    plt.suptitle(f"Boxplots — {fund_sel[:25]}", fontweight="bold", y=1.01, fontsize=12)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Correlation & Covariance ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔗 Correlation & Covariance Matrix</div>', unsafe_allow_html=True)
    if len(all_data) > 1:
        nav_df = pd.DataFrame({f[:16]: all_data[f].set_index("date")["nav"] for f in all_data})
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Correlation:**")
            st.dataframe(nav_df.corr().round(4), use_container_width=True)
        with c2:
            st.write("**Covariance:**")
            st.dataframe(nav_df.cov().round(4), use_container_width=True)
    else:
        st.info("Select 2+ funds from sidebar to see correlation/covariance matrices.")

# ============================================================
# PAGE: FUND COMPARISON
# ============================================================
elif page == "🏆 Fund Comparison":
    st.markdown('<div class="unit-badge">🏆 BONUS · Fund Comparison & Ranking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Compare All Selected Funds</div>', unsafe_allow_html=True)

    rows = []
    for fund, df in all_data.items():
        m = calculate_metrics(df); m["Fund"] = fund; rows.append(m)
    metrics_df = pd.DataFrame(rows).set_index("Fund")
    ranked = metrics_df.sort_values("CAGR (%)", ascending=False)

    best = ranked.index[0]
    st.success(f"🏆 **Best Fund:** {best} — CAGR {ranked.loc[best,'CAGR (%)']:.2f}% | Sharpe {ranked.loc[best,'Sharpe Ratio']:.2f}")

    # Fund link pills
    fund_pills = " ".join([
        f'<a class="link-pill" href="{FUND_LINKS.get(f, "#")}" target="_blank">🏦 {f}</a>'
        for f in ranked.index
    ])
    st.markdown(f'<div style="margin:0.5rem 0 1rem 0">{fund_pills}</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header">📋 Performance Ranking Table</div>', unsafe_allow_html=True)
    st.dataframe(ranked.style.highlight_max(subset=["CAGR (%)", "Sharpe Ratio"], color="#166534")
                              .highlight_min(subset=["Annual Volatility (%)", "Max Drawdown (%)"], color="#991b1b"),
                 use_container_width=True)

    st.markdown("---")
    st.markdown('<div class="section-header">📊 Visual Comparison</div>', unsafe_allow_html=True)
    n = len(all_data)
    clrs = COLORS[:n]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    short = [f.split()[0] + " " + f.split()[1] if len(f.split()) > 1 else f[:12] for f in ranked.index]

    ranked["CAGR (%)"].plot(kind="bar", ax=axes[0], color=clrs)
    axes[0].set_title("CAGR (%)", fontweight="bold")
    axes[0].set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    axes[0].grid(axis="y", alpha=0.15)

    ranked["Annual Volatility (%)"].plot(kind="bar", ax=axes[1], color=clrs)
    axes[1].set_title("Annual Volatility (%)", fontweight="bold")
    axes[1].set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    axes[1].grid(axis="y", alpha=0.15)

    for i, fund in enumerate(ranked.index):
        axes[2].scatter(ranked.loc[fund, "Annual Volatility (%)"],
                        ranked.loc[fund, "CAGR (%)"],
                        s=220, color=clrs[i], zorder=5, label=short[i], edgecolors='white', linewidth=0.5)
    axes[2].set_xlabel("Volatility (%)"); axes[2].set_ylabel("CAGR (%)")
    axes[2].set_title("Risk vs Return", fontweight="bold")
    axes[2].legend(fontsize=7, facecolor='#111827', edgecolor='#1e293b'); axes[2].grid(True, alpha=0.15)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown("---")
    st.markdown('<div class="section-header">📈 Cumulative Returns Over Time</div>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (fund, df) in enumerate(all_data.items()):
        ax.plot(df["date"], df["cumulative_return"], lw=2, color=COLORS[i % len(COLORS)], label=fund[:22])
    ax.axhline(0, color="#475569", ls="--", lw=1, alpha=0.5)
    ax.fill_between(df["date"], 0, alpha=0.02, color="grey")
    ax.set_title("Cumulative Returns Comparison (%)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return (%)")
    ax.legend(fontsize=9, facecolor='#111827', edgecolor='#1e293b'); ax.grid(True, alpha=0.15)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        <p>Built with <a href="https://docs.streamlit.io/" target="_blank">Streamlit</a> · Data from <a href="https://www.mfapi.in/" target="_blank">mfapi.in</a> · Official NAV via <a href="https://www.amfiindia.com/" target="_blank">AMFI India</a></p>
        <p>INT375: Data Science Toolbox · Python Programming · Solo Project · © 2025</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# PAGE: NAV PREDICTION (Linear Regression)
# ============================================================
elif page == "🤖 NAV Prediction":
    st.markdown('<div class="unit-badge">🤖 Numeric Prediction using Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header">NAV Prediction using Linear Regression</div>', unsafe_allow_html=True)

    fund_sel = st.selectbox("Select Fund for Prediction", list(all_data.keys()))
    df = all_data[fund_sel].copy()

    # ── Prepare features ─────────────────────────────────────
    st.markdown('<div class="section-header">📋 Preparing Features</div>', unsafe_allow_html=True)

    with st.expander("📄 Code", expanded=False):
        st.code("""
# Create features from date
df["year"]  = df["date"].dt.year
df["month"] = df["date"].dt.month
df["day"]   = df["date"].dt.day

# Lag feature (previous day NAV)
df["nav_lag_1"] = df["nav"].shift(1)

X = df[["year", "month", "day", "nav_lag_1"]]
y = df["nav"]
        """, language="python")

    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["day"]       = df["date"].dt.day
    df["nav_lag_1"] = df["nav"].shift(1)
    df = df.dropna()

    X = df[["year", "month", "day", "nav_lag_1"]]
    y = df["nav"]

    st.write(f"**Total samples:** {len(X)}")
    st.dataframe(X.head(10), use_container_width=True)

    # ── Train-Test Split ─────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🔀 Train-Test Split</div>', unsafe_allow_html=True)

    with st.expander("📄 Code", expanded=False):
        st.code("""
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        """, language="python")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    render_metrics_html([
        ("📊", f"{len(X_train)}", "Training Samples"),
        ("🧪", f"{len(X_test)}",  "Testing Samples"),
    ])

    # ── Train Linear Regression ──────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">🧠 Train Linear Regression</div>', unsafe_allow_html=True)

    with st.expander("📄 Code", expanded=False):
        st.code("""
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Train linear regression
model = LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("R^2 score:", round(r2_score(y_test, y_pred), 4))
print("MSE:", round(mean_squared_error(y_test, y_pred), 4))
        """, language="python")

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2  = round(r2_score(y_test, y_pred), 4)
    mse = round(mean_squared_error(y_test, y_pred), 4)

    # Show coefficients
    st.write("**Model Coefficients:**", list(np.round(model.coef_, 4)))
    st.write("**Model Intercept:**", round(model.intercept_, 4))

    render_metrics_html([
        ("📈", f"{r2}",   "R² Score"),
        ("📊", f"{mse}",  "MSE"),
    ])

    if r2 > 0.9:
        st.success(f"✅ R² = {r2} — Good model fit")
    else:
        st.info(f"📊 R² = {r2}")

    # ── Regression Plot ──────────────────────────────────────
    st.markdown("---")
    st.markdown('<div class="section-header">📈 Regression Plot</div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"}, ax=ax)
    ax.set_title("Linear Regression", fontweight="bold")
    ax.set_xlabel("Actual NAV")
    ax.set_ylabel("Predicted NAV")
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Footer ────────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        <p>Built with <a href="https://docs.streamlit.io/" target="_blank">Streamlit</a> · Data from <a href="https://www.mfapi.in/" target="_blank">mfapi.in</a> · Official NAV via <a href="https://www.amfiindia.com/" target="_blank">AMFI India</a></p>
        <p>INT375: Data Science Toolbox · Python Programming · Solo Project · © 2025</p>
    </div>
    """, unsafe_allow_html=True)