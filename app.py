import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

FUEL_CSV = Path(__file__).resolve().parent / "fuel.csv"

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Macro Analyzer | Fuel · Gold · Nifty",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f0c29, #302b63, #24243e);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; padding: 6px 0; }

/* Main background */
.stApp { background: #0d1117; color: #c9d1d9; }

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #252b3b);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 24px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(88,166,255,0.15);
}
.metric-title { font-size: 0.78rem; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; margin: 6px 0; }
.metric-sub   { font-size: 0.78rem; color: #7ee787; }
.metric-sub.neg { color: #f85149; }

/* Section headers */
.section-header {
    background: linear-gradient(90deg, #58a6ff22, transparent);
    border-left: 4px solid #58a6ff;
    padding: 10px 16px;
    border-radius: 0 8px 8px 0;
    margin: 24px 0 16px;
    font-size: 1.1rem;
    font-weight: 600;
    color: #c9d1d9;
}

/* Insight box */
.insight-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px 20px;
    margin: 12px 0;
    font-size: 0.88rem;
    line-height: 1.6;
    color: #8b949e;
}
.insight-box span { color: #58a6ff; font-weight: 600; }

/* Model result card */
.result-card {
    background: linear-gradient(135deg, #161b22, #1a2332);
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px;
    margin: 8px 0;
}
.result-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; }
.result-value { font-size: 1.4rem; font-weight: 700; color: #f0f6fc; }

/* Hide default streamlit header */
#MainMenu, header, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

COLORS = {
    "fuel":  "#f59e0b",
    "gold":  "#fcd34d",
    "nifty": "#34d399",
    "bg":    "#0d1117",
    "card":  "#161b22",
    "blue":  "#58a6ff",
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_raw_data():
    # Fuel – use the bundled CSV when it contains data; otherwise use Yahoo
    # Finance so a missing or empty deployment asset does not stop the app.
    if FUEL_CSV.is_file() and FUEL_CSV.stat().st_size > 0:
        fuel = pd.read_csv(FUEL_CSV)
        fuel.columns = fuel.columns.str.lower()
        required_columns = {"date", "crude_oil_price"}
        missing_columns = required_columns - set(fuel.columns)
        if missing_columns:
            raise ValueError(
                f"{FUEL_CSV.name} is missing required columns: "
                f"{', '.join(sorted(missing_columns))}"
            )
        fuel["date"] = pd.to_datetime(fuel["date"])
    else:
        fuel = yf.download(
            "CL=F",
            start="2004-01-01",
            auto_adjust=True,
            progress=False,
        ).reset_index()
        fuel.columns = [
            c if isinstance(c, str) else c[0] for c in fuel.columns
        ]
        fuel.columns = fuel.columns.str.lower()
        fuel = fuel.rename(columns={"close": "crude_oil_price"})
        fuel["date"] = pd.to_datetime(fuel["date"])

    # Gold – yfinance daily, then monthly last close
    gold_raw = yf.download("GC=F", start="2004-01-01", auto_adjust=True, progress=False)
    gold_raw = gold_raw.reset_index()
    gold_raw.columns = [c if isinstance(c, str) else c[0] for c in gold_raw.columns]
    gold_raw.columns = gold_raw.columns.str.lower()
    gold_raw["date"] = pd.to_datetime(gold_raw["date"])

    # Nifty – yfinance daily index
    nifty_raw = yf.download("^NSEI", start="2015-01-01", auto_adjust=True, progress=False)
    nifty_raw = nifty_raw.reset_index()
    nifty_raw.columns = [c if isinstance(c, str) else c[0] for c in nifty_raw.columns]
    nifty_raw.columns = nifty_raw.columns.str.lower()
    nifty_raw["date"] = pd.to_datetime(nifty_raw["date"])

    return fuel, gold_raw, nifty_raw


@st.cache_data(show_spinner=False)
def build_features(_fuel, _gold_raw, _nifty_raw):
    fuel_monthly  = _fuel.set_index("date")["crude_oil_price"].resample("MS").last()
    gold_monthly  = _gold_raw.set_index("date")["close"].resample("MS").last()
    nifty_monthly = _nifty_raw.set_index("date")["close"].resample("MS").last()

    merged = pd.concat([fuel_monthly, gold_monthly, nifty_monthly], axis=1, join="inner")
    merged.columns = ["fuel_price", "gold_price", "nifty_close"]
    merged = merged.dropna().reset_index().rename(columns={merged.columns[0] if merged.index.name else "index": "date"})
    if "index" in merged.columns:
        merged = merged.rename(columns={"index": "date"})

    df = merged.copy()
    df["fuel_pct"]  = df["fuel_price"].pct_change() * 100
    df["gold_pct"]  = df["gold_price"].pct_change() * 100
    df["nifty_pct"] = df["nifty_close"].pct_change() * 100

    df["fuel_log_ret"]  = np.log(df["fuel_price"]  / df["fuel_price"].shift(1))
    df["gold_log_ret"]  = np.log(df["gold_price"]  / df["gold_price"].shift(1))
    df["nifty_log_ret"] = np.log(df["nifty_close"] / df["nifty_close"].shift(1))

    df["gold_fuel_ratio"] = df["gold_price"] / df["fuel_price"]
    df["fuel_pct_lag1"]   = df["fuel_pct"].shift(1)
    df["fuel_pct_lag2"]   = df["fuel_pct"].shift(2)
    df["gold_pct_lag1"]   = df["gold_pct"].shift(1)
    df["nifty_pct_lag1"]  = df["nifty_pct"].shift(1)

    df["fuel_ma3"]  = df["fuel_price"].rolling(3).mean()
    df["gold_ma3"]  = df["gold_price"].rolling(3).mean()
    df["nifty_ma3"] = df["nifty_close"].rolling(3).mean()

    df["fuel_cat"]  = pd.cut(df["fuel_pct"], bins=[-np.inf, -5, 5, np.inf], labels=["fell", "stable", "rose"])
    df["nifty_dir"] = (df["nifty_pct"] > 0).astype(int)

    df["nifty_next_pct"] = df["nifty_pct"].shift(-1)
    df["gold_next_pct"]  = df["gold_pct"].shift(-1)
    df["nifty_next_dir"] = (df["nifty_next_pct"] > 0).astype(int)

    df["year"] = df["date"].dt.year
    def label_era(yr):
        if yr <= 2016:   return "1 Oil-Crash 2015-16"
        elif yr <= 2019: return "2 Stable 2017-19"
        elif yr == 2020: return "3 COVID 2020"
        else:            return "4 Post-COVID 2021+"
    df["era"] = df["year"].apply(label_era)

    df = df.dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px;'>
        <div style='font-size:2.2rem;'>📈</div>
        <div style='font-weight:700; font-size:1.1rem; color:#58a6ff;'>Market Macro Analyzer</div>
        <div style='font-size:0.75rem; color:#8b949e; margin-top:4px;'>Fuel · Gold · Nifty 50</div>
    </div>
    <hr style='border-color:#30363d; margin:12px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "📊 EDA & Statistics", "🔗 Correlation Analysis",
         "🤖 ML Models", "🔵 Clustering", "📉 Time Series"],
        label_visibility="collapsed",
    )

    st.markdown("""
    <hr style='border-color:#30363d; margin:16px 0 8px;'>
    <div style='font-size:0.72rem; color:#8b949e; text-align:center;'>
        Data Sources<br>
        <span style='color:#58a6ff;'>Fuel</span> · Local CSV &nbsp;|&nbsp;
        <span style='color:#fcd34d;'>Gold</span> · Yahoo Finance<br>
        <span style='color:#34d399;'>Nifty</span> · Yahoo Finance (^NSEI)
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
with st.spinner("⏳ Loading market data…"):
    fuel_raw, gold_raw, nifty_raw = load_raw_data()

with st.spinner("⚙️ Building features…"):
    df = build_features(fuel_raw, gold_raw, nifty_raw)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────────────────────────────────────
LAYOUT = dict(
    paper_bgcolor=COLORS["bg"],
    plot_bgcolor="#0d1117",
    font=dict(family="Inter", color="#c9d1d9"),
    xaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    yaxis=dict(gridcolor="#21262d", zerolinecolor="#21262d"),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="#161b22", bordercolor="#30363d", borderwidth=1),
)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 — OVERVIEW
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown("""
    <h1 style='background: linear-gradient(90deg,#58a6ff,#fcd34d,#34d399);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
    font-size:2rem; font-weight:700; margin-bottom:4px;'>
    Market Macro Analyzer
    </h1>
    <p style='color:#8b949e; font-size:0.95rem; margin-bottom:24px;'>
    Analyzing the inter-relationship of Crude Oil · Gold · Nifty 50 Index (2015 – present)
    </p>
    """, unsafe_allow_html=True)

    # KPI row
    latest = df.iloc[-1]
    prev   = df.iloc[-2]
    c1, c2, c3, c4 = st.columns(4)
    def kpi(col, title, val, change, unit=""):
        arrow = "▲" if change >= 0 else "▼"
        cls   = "" if change >= 0 else "neg"
        col.markdown(f"""
        <div class='metric-card'>
            <div class='metric-title'>{title}</div>
            <div class='metric-value'>{unit}{val:.2f}</div>
            <div class='metric-sub {cls}'>{arrow} {abs(change):.2f}% MoM</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, "🛢 Crude Oil ($/bbl)",  latest.fuel_price,  latest.fuel_pct)
    kpi(c2, "🥇 Gold ($/oz)",        latest.gold_price,  latest.gold_pct)
    kpi(c3, "📈 Nifty 50 Index",     latest.nifty_close, latest.nifty_pct)
    c4.markdown(f"""
    <div class='metric-card'>
        <div class='metric-title'>📅 Data Range</div>
        <div class='metric-value' style='font-size:1rem;'>{df['date'].min().strftime('%b %Y')}</div>
        <div class='metric-sub'>→ {df['date'].max().strftime('%b %Y')} ({len(df)} months)</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-header'>📈 Price History — All 3 Assets</div>", unsafe_allow_html=True)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=["Crude Oil ($/bbl)", "Gold ($/oz)", "Nifty 50 Index"])
    fig.add_trace(go.Scatter(x=df.date, y=df.fuel_price, name="Fuel",
                             line=dict(color=COLORS["fuel"], width=2), fill="tozeroy",
                             fillcolor="rgba(245,158,11,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.date, y=df.gold_price, name="Gold",
                             line=dict(color=COLORS["gold"], width=2), fill="tozeroy",
                             fillcolor="rgba(252,211,77,0.08)"), row=2, col=1)
    fig.add_trace(go.Scatter(x=df.date, y=df.nifty_close, name="Nifty",
                             line=dict(color=COLORS["nifty"], width=2), fill="tozeroy",
                             fillcolor="rgba(52,211,153,0.08)"), row=3, col=1)
    fig.update_layout(height=520, showlegend=False, **LAYOUT)
    fig.update_annotations(font_color="#8b949e")
    st.plotly_chart(fig, use_container_width=True)

    # Economic eras
    st.markdown("<div class='section-header'>🗓 Economic Eras</div>", unsafe_allow_html=True)
    era_stats = df.groupby("era")[["fuel_price","gold_price","nifty_close"]].mean().round(2)
    fig2 = go.Figure()
    for col, color, label in [("fuel_price",COLORS["fuel"],"Fuel"), ("gold_price",COLORS["gold"],"Gold"),("nifty_close",COLORS["nifty"],"Nifty")]:
        fig2.add_trace(go.Bar(name=label, x=era_stats.index, y=era_stats[col],
                              marker_color=color, opacity=0.85))
    fig2.update_layout(barmode="group", height=350, title="Average Price by Economic Era", **LAYOUT)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"""
    <div class='insight-box'>
    <span>💡 Key Insight:</span> The dataset spans <span>{len(df)} months</span> across 4 distinct economic regimes.
    COVID-2020 caused a dramatic fuel crash (WTI briefly went negative) while Gold soared as a safe haven.
    Post-COVID era saw explosive growth in all 3 assets.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 — EDA & STATISTICS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 EDA & Statistics":
    st.markdown("<h2 style='color:#58a6ff;'>📊 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📋 Descriptive Stats", "📦 Distributions", "📈 Returns"])

    with tab1:
        cols = ["fuel_price", "gold_price", "nifty_close"]
        stats = df[cols].agg(["mean","median","std","var","min","max"])
        stats.loc["range"] = stats.loc["max"] - stats.loc["min"]
        stats.loc["mean_dev"] = df[cols].apply(lambda x: (x - x.mean()).abs().mean())
        st.dataframe(stats.round(2).style.background_gradient(cmap="Blues"), use_container_width=True)

        st.markdown("<div class='section-header'>Monthly Return Statistics</div>", unsafe_allow_html=True)
        ret_stats = df[["fuel_pct","gold_pct","nifty_pct"]].agg(["mean","std","min","max"]).round(2)
        st.dataframe(ret_stats.style.background_gradient(cmap="RdYlGn"), use_container_width=True)

        st.markdown("<div class='section-header'>Volatility (Std Dev of Returns) by Year</div>", unsafe_allow_html=True)
        vol = df.groupby("year")[["fuel_pct","gold_pct","nifty_pct"]].std().round(2)
        fig = px.imshow(vol.T, color_continuous_scale="YlOrRd",
                        title="Annual Volatility Heatmap",
                        labels=dict(x="Year", y="Asset", color="Std Dev %"))
        fig.update_layout(**LAYOUT, height=260)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.markdown("<div class='section-header'>Price Distributions</div>", unsafe_allow_html=True)
        fig = make_subplots(rows=1, cols=3, subplot_titles=["Crude Oil", "Gold", "Nifty"])
        for i, (col, color) in enumerate(zip(["fuel_price","gold_price","nifty_close"],
                                              [COLORS["fuel"],COLORS["gold"],COLORS["nifty"]]), 1):
            fig.add_trace(go.Histogram(x=df[col], name=col, marker_color=color,
                                       opacity=0.75, nbinsx=25), row=1, col=i)
        fig.update_layout(showlegend=False, height=320, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Box Plots — Monthly Returns by Era</div>", unsafe_allow_html=True)
        fig2 = make_subplots(rows=1, cols=3, subplot_titles=["Fuel %", "Gold %", "Nifty %"])
        for i, (col, color) in enumerate(zip(["fuel_pct","gold_pct","nifty_pct"],
                                              [COLORS["fuel"],COLORS["gold"],COLORS["nifty"]]), 1):
            for era in df.era.unique():
                sub = df[df.era == era]
                fig2.add_trace(go.Box(y=sub[col], name=era, marker_color=color,
                                      showlegend=(i == 1)), row=1, col=i)
        fig2.update_layout(height=380, **LAYOUT, boxmode="group")
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("<div class='section-header'>Monthly Returns Over Time</div>", unsafe_allow_html=True)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=["Fuel % Return", "Gold % Return", "Nifty % Return"])
        for i, (col, color) in enumerate(zip(["fuel_pct","gold_pct","nifty_pct"],
                                              [COLORS["fuel"],COLORS["gold"],COLORS["nifty"]]), 1):
            fig.add_trace(go.Bar(x=df.date, y=df[col], name=col,
                                 marker_color=np.where(df[col] >= 0, color, "#f85149"),
                                 opacity=0.8), row=i, col=1)
        fig.update_layout(showlegend=False, height=520, **LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Gold / Fuel Ratio</div>", unsafe_allow_html=True)
        fig3 = go.Figure(go.Scatter(x=df.date, y=df.gold_fuel_ratio,
                                    line=dict(color="#a78bfa", width=2),
                                    fill="tozeroy", fillcolor="rgba(167,139,250,0.1)"))
        fig3.update_layout(height=280, title="Gold-to-Fuel Ratio (oz of gold per barrel equivalent)",
                            **LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 — CORRELATION
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔗 Correlation Analysis":
    from scipy import stats as scipy_stats
    from scipy.stats import chi2_contingency, ttest_ind, pearsonr

    st.markdown("<h2 style='color:#58a6ff;'>🔗 Correlation Analysis</h2>", unsafe_allow_html=True)

    # Pearson heatmap
    st.markdown("<div class='section-header'>Pearson Correlation Matrix</div>", unsafe_allow_html=True)
    num_cols = ["fuel_price","gold_price","nifty_close","fuel_pct","gold_pct","nifty_pct",
                "fuel_log_ret","gold_log_ret","nifty_log_ret","gold_fuel_ratio"]
    corr = df[num_cols].corr().round(3)
    fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                    text_auto=True, aspect="auto",
                    title="Pearson Correlation — All Numeric Features")
    fig.update_layout(**LAYOUT, height=520)
    st.plotly_chart(fig, use_container_width=True)

    # Scatter plots
    st.markdown("<div class='section-header'>Scatter Plots with Regression Line</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    pairs = [("fuel_pct","nifty_pct","Fuel% vs Nifty%"),
             ("gold_pct","nifty_pct","Gold% vs Nifty%")]
    for col_, (x, y, title) in zip([c1, c2], pairs):
        r, p = pearsonr(df[x].dropna(), df[y][df[x].notna()])
        fig = px.scatter(df, x=x, y=y, color="era", trendline="ols",
                         title=f"{title} (r={r:.3f}, p={p:.3f})",
                         color_discrete_sequence=px.colors.qualitative.Pastel)
        fig.update_layout(**LAYOUT, height=360)
        col_.plotly_chart(fig, use_container_width=True)

    # T-test
    st.markdown("<div class='section-header'>Welch T-Test: Nifty returns when Fuel Rose vs Fell</div>", unsafe_allow_html=True)
    rose = df[df.fuel_cat == "rose"]["nifty_pct"].dropna()
    fell = df[df.fuel_cat == "fell"]["nifty_pct"].dropna()
    stable = df[df.fuel_cat == "stable"]["nifty_pct"].dropna()
    t, p = ttest_ind(rose, fell, equal_var=False)
    cc1, cc2, cc3 = st.columns(3)
    for c_, label, vals, color in [(cc1,"Fuel Rose 📈",rose,COLORS["nifty"]),
                                    (cc2,"Fuel Fell 📉",fell,"#f85149"),
                                    (cc3,"Fuel Stable ➡️",stable,COLORS["gold"])]:
        c_.markdown(f"""
        <div class='result-card'>
            <div class='result-label'>{label} — Mean Nifty%</div>
            <div class='result-value' style='color:{color};'>{vals.mean():.2f}%</div>
            <div style='font-size:0.78rem;color:#8b949e;'>n={len(vals)} · std={vals.std():.2f}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
    <div class='insight-box'>
    <span>T-test result:</span> t = {t:.3f}, p-value = {p:.4f}.
    {"✅ Statistically significant difference (p < 0.05)." if p < 0.05 else "⚠️ Not statistically significant (p ≥ 0.05)."}
    </div>""", unsafe_allow_html=True)

    # Chi-square
    st.markdown("<div class='section-header'>Chi-Square Test: Fuel Category vs Nifty Direction</div>", unsafe_allow_html=True)
    ct = pd.crosstab(df["fuel_cat"], df["nifty_dir"])
    chi2, p_chi, dof, _ = chi2_contingency(ct)
    st.dataframe(ct.style.background_gradient(cmap="Greens"), use_container_width=False)
    st.markdown(f"""
    <div class='insight-box'>
    <span>Chi² = {chi2:.3f}</span>, p-value = {p_chi:.4f}, dof = {dof}<br>
    {"✅ Fuel category and Nifty direction are <span>dependent</span> (p < 0.05)." if p_chi < 0.05 else "⚠️ No significant association."}
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 4 — ML MODELS
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🤖 ML Models":
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                                  accuracy_score, classification_report, confusion_matrix)
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    st.markdown("<h2 style='color:#58a6ff;'>🤖 Machine Learning Models</h2>", unsafe_allow_html=True)

    # Features
    FEAT_REG  = ["fuel_pct","gold_pct","fuel_pct_lag1","gold_pct_lag1","nifty_pct_lag1","gold_fuel_ratio"]
    FEAT_CLF  = ["fuel_pct","gold_pct","fuel_pct_lag1","gold_pct_lag1","nifty_pct_lag1"]
    TARGET_R  = "nifty_next_pct"
    TARGET_C  = "nifty_next_dir"

    sub = df[FEAT_REG + FEAT_CLF + [TARGET_R, TARGET_C]].dropna()
    X_r  = sub[FEAT_REG]
    X_c  = sub[FEAT_CLF]
    y_r  = sub[TARGET_R]
    y_c  = sub[TARGET_C]

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(X_r, y_r, test_size=0.2, random_state=42)
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(X_c, y_c, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    Xr_tr_s = scaler.fit_transform(Xr_tr)
    Xr_te_s = scaler.transform(Xr_te)

    tab1, tab2, tab3 = st.tabs(["📐 Linear Regression", "🎯 Classification", "🌳 Decision Tree"])

    with tab1:
        lr = LinearRegression().fit(Xr_tr_s, yr_tr)
        pred = lr.predict(Xr_te_s)
        mae  = mean_absolute_error(yr_te, pred)
        mse  = mean_squared_error(yr_te, pred)
        rmse = np.sqrt(mse)
        r2   = r2_score(yr_te, pred)

        st.markdown("<div class='section-header'>Linear Regression → Predict Next Month Nifty %</div>", unsafe_allow_html=True)
        cc = st.columns(4)
        for c_, lbl, val in zip(cc, ["MAE","MSE","RMSE","R²"],[mae,mse,rmse,r2]):
            c_.markdown(f"""<div class='metric-card'>
                <div class='metric-title'>{lbl}</div>
                <div class='metric-value' style='font-size:1.4rem;'>{val:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Actual vs Predicted</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=yr_te.values, name="Actual", line=dict(color=COLORS["nifty"],width=2)))
        fig.add_trace(go.Scatter(y=pred, name="Predicted", line=dict(color=COLORS["blue"],width=2,dash="dash")))
        fig.update_layout(height=340, **LAYOUT, title="Nifty Next-Month Return: Actual vs Predicted")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("<div class='section-header'>Feature Coefficients</div>", unsafe_allow_html=True)
        coef = pd.Series(lr.coef_, index=FEAT_REG).sort_values()
        fig2 = go.Figure(go.Bar(x=coef.values, y=coef.index, orientation="h",
                                 marker_color=np.where(coef.values >= 0, COLORS["nifty"], "#f85149")))
        fig2.update_layout(height=300, **LAYOUT, title="Feature Importance (Coefficients)")
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        # Logistic Regression
        log = LogisticRegression(max_iter=500, random_state=42).fit(Xc_tr, yc_tr)
        nb  = GaussianNB().fit(Xc_tr, yc_tr)

        log_acc = accuracy_score(yc_te, log.predict(Xc_te))
        nb_acc  = accuracy_score(yc_te, nb.predict(Xc_te))

        st.markdown("<div class='section-header'>Logistic Regression vs Naive Bayes</div>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.markdown(f"""<div class='metric-card'>
            <div class='metric-title'>🎯 Logistic Regression Accuracy</div>
            <div class='metric-value'>{log_acc*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)
        c2.markdown(f"""<div class='metric-card'>
            <div class='metric-title'>🎯 Naive Bayes Accuracy</div>
            <div class='metric-value'>{nb_acc*100:.1f}%</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Classification Reports</div>", unsafe_allow_html=True)
        for name, model in [("Logistic Regression", log), ("Naive Bayes", nb)]:
            report = classification_report(yc_te, model.predict(Xc_te), output_dict=True)
            rdf = pd.DataFrame(report).T.drop("accuracy", errors="ignore")
            st.markdown(f"**{name}**")
            st.dataframe(rdf.round(3).style.background_gradient(cmap="Blues", subset=["precision","recall","f1-score"]),
                         use_container_width=True)

        # Confusion matrices
        st.markdown("<div class='section-header'>Confusion Matrices</div>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        for c_, name, model in [(col1,"Logistic Reg",log),(col2,"Naive Bayes",nb)]:
            cm = confusion_matrix(yc_te, model.predict(Xc_te))
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                            title=name, labels=dict(x="Predicted",y="Actual"),
                            x=["Down","Up"], y=["Down","Up"])
            fig.update_layout(**LAYOUT, height=300)
            c_.plotly_chart(fig, use_container_width=True)

    with tab3:
        dt = DecisionTreeClassifier(max_depth=4, random_state=42).fit(Xc_tr, yc_tr)
        dt_acc = accuracy_score(yc_te, dt.predict(Xc_te))

        st.markdown("<div class='section-header'>Decision Tree Classifier</div>", unsafe_allow_html=True)
        st.markdown(f"""<div class='metric-card' style='width:220px;'>
            <div class='metric-title'>🌳 Decision Tree Accuracy</div>
            <div class='metric-value'>{dt_acc*100:.1f}%</div>
            <div class='metric-sub'>max_depth = 4</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Feature Importances</div>", unsafe_allow_html=True)
        fi = pd.Series(dt.feature_importances_, index=FEAT_CLF).sort_values(ascending=True)
        fig = go.Figure(go.Bar(x=fi.values, y=fi.index, orientation="h",
                               marker_color=COLORS["nifty"], opacity=0.85))
        fig.update_layout(**LAYOUT, height=300, title="Decision Tree Feature Importances")
        st.plotly_chart(fig, use_container_width=True)

        cm2 = confusion_matrix(yc_te, dt.predict(Xc_te))
        fig2 = px.imshow(cm2, text_auto=True, color_continuous_scale="Greens",
                         title="Decision Tree Confusion Matrix",
                         labels=dict(x="Predicted",y="Actual"),
                         x=["Down","Up"], y=["Down","Up"])
        fig2.update_layout(**LAYOUT, height=320)
        st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 5 — CLUSTERING
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🔵 Clustering":
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    st.markdown("<h2 style='color:#58a6ff;'>🔵 KMeans Clustering</h2>", unsafe_allow_html=True)

    feats = ["fuel_pct","gold_pct","nifty_pct"]
    sub   = df[feats + ["date","era"]].dropna()
    X     = StandardScaler().fit_transform(sub[feats])

    # Elbow / silhouette
    st.markdown("<div class='section-header'>Elbow Curve & Silhouette Scores</div>", unsafe_allow_html=True)
    inertias, silhouettes, K = [], [], range(2, 9)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X, km.labels_))

    fig = make_subplots(rows=1, cols=2, subplot_titles=["Elbow Curve (Inertia)", "Silhouette Score"])
    fig.add_trace(go.Scatter(x=list(K), y=inertias, mode="lines+markers",
                              line=dict(color=COLORS["blue"],width=2),
                              marker=dict(size=8, color=COLORS["blue"])), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(K), y=silhouettes, mode="lines+markers",
                              line=dict(color=COLORS["nifty"],width=2),
                              marker=dict(size=8, color=COLORS["nifty"])), row=1, col=2)
    fig.update_layout(**LAYOUT, height=320, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    n_clusters = st.slider("Select number of clusters", 2, 8, 4)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(X)
    sub = sub.copy()
    sub["cluster"] = km.labels_.astype(str)
    sil = silhouette_score(X, km.labels_)

    st.markdown(f"**Silhouette Score for k={n_clusters}: `{sil:.4f}`**")

    st.markdown("<div class='section-header'>Cluster Scatter (Fuel% vs Nifty%)</div>", unsafe_allow_html=True)
    fig2 = px.scatter(sub, x="fuel_pct", y="nifty_pct", color="cluster",
                      hover_data=["date","gold_pct","era"],
                      color_discrete_sequence=px.colors.qualitative.Vivid,
                      title=f"KMeans Clusters (k={n_clusters})")
    fig2.update_layout(**LAYOUT, height=420)
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-header'>3D Cluster View</div>", unsafe_allow_html=True)
    fig3 = px.scatter_3d(sub, x="fuel_pct", y="gold_pct", z="nifty_pct",
                          color="cluster", hover_data=["date","era"],
                          color_discrete_sequence=px.colors.qualitative.Vivid,
                          title="3D Cluster: Fuel% · Gold% · Nifty%")
    fig3.update_layout(paper_bgcolor=COLORS["bg"], font=dict(color="#c9d1d9"), height=500)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<div class='section-header'>Cluster Centroids (mean returns %)</div>", unsafe_allow_html=True)
    centroids = sub.groupby("cluster")[feats].mean().round(3)
    st.dataframe(centroids.style.background_gradient(cmap="RdYlGn", axis=None),
                 use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE 6 — TIME SERIES
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📉 Time Series":
    from statsmodels.tsa.stattools import adfuller, grangercausalitytests
    from statsmodels.tsa.seasonal import seasonal_decompose
    import io, contextlib

    st.markdown("<h2 style='color:#58a6ff;'>📉 Time Series Analysis</h2>", unsafe_allow_html=True)

    ts = df.set_index("date")[["fuel_price","gold_price","nifty_close",
                                "fuel_pct","gold_pct","nifty_pct"]].copy()

    # ADF test
    st.markdown("<div class='section-header'>Augmented Dickey-Fuller Stationarity Test</div>", unsafe_allow_html=True)
    adf_rows = []
    for col in ["fuel_price","gold_price","nifty_close","fuel_pct","gold_pct","nifty_pct"]:
        result = adfuller(ts[col].dropna())
        adf_rows.append({
            "Series": col, "ADF Stat": round(result[0], 4),
            "p-value": round(result[1], 4),
            "Stationary?": "✅ Yes" if result[1] < 0.05 else "❌ No",
        })
    adf_df = pd.DataFrame(adf_rows)
    st.dataframe(adf_df.style.applymap(
        lambda x: "color: #7ee787;" if x == "✅ Yes" else ("color: #f85149;" if x == "❌ No" else ""),
        subset=["Stationary?"]), use_container_width=True)

    # Seasonal decomposition
    st.markdown("<div class='section-header'>Seasonal Decomposition</div>", unsafe_allow_html=True)
    asset = st.selectbox("Choose asset", ["fuel_price","gold_price","nifty_close"])
    series = ts[asset].dropna()
    if len(series) >= 24:
        decomp = seasonal_decompose(series, model="multiplicative", period=12, extrapolate_trend="freq")
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                            subplot_titles=["Observed","Trend","Seasonal","Residual"])
        data_map = [(decomp.observed,COLORS["blue"]),(decomp.trend,COLORS["gold"]),
                    (decomp.seasonal,COLORS["nifty"]),(decomp.resid,"#f85149")]
        for i, (d, color) in enumerate(data_map, 1):
            fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=color, width=1.5)), row=i, col=1)
        fig.update_layout(showlegend=False, height=560, **LAYOUT, title=f"Seasonal Decomposition — {asset}")
        st.plotly_chart(fig, use_container_width=True)

    # Granger causality
    st.markdown("<div class='section-header'>Granger Causality Tests</div>", unsafe_allow_html=True)
    st.markdown("Does past **Fuel %** Granger-cause **Nifty %** (and vice versa)?")
    max_lag = st.slider("Max lag (months)", 1, 6, 3)

    granger_data = ts[["nifty_pct","fuel_pct","gold_pct"]].dropna()
    granger_rows = []
    for cause, effect in [("fuel_pct","nifty_pct"),("gold_pct","nifty_pct"),("nifty_pct","fuel_pct")]:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = grangercausalitytests(granger_data[[effect, cause]], maxlag=max_lag, verbose=True)
        best_p = min(res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag+1))
        granger_rows.append({
            "Cause → Effect": f"{cause} → {effect}",
            "Best p-value": round(best_p, 4),
            "Granger-causes?": "✅ Yes" if best_p < 0.05 else "❌ No",
        })
    g_df = pd.DataFrame(granger_rows)
    st.dataframe(g_df.style.applymap(
        lambda x: "color: #7ee787;" if x == "✅ Yes" else ("color: #f85149;" if x == "❌ No" else ""),
        subset=["Granger-causes?"]), use_container_width=True)

    # Rolling correlation
    st.markdown("<div class='section-header'>Rolling 12-Month Correlation (Fuel vs Nifty)</div>", unsafe_allow_html=True)
    roll_corr = ts["fuel_pct"].rolling(12).corr(ts["nifty_pct"])
    fig3 = go.Figure(go.Scatter(x=roll_corr.index, y=roll_corr.values,
                                 line=dict(color=COLORS["blue"], width=2),
                                 fill="tozeroy", fillcolor="rgba(88,166,255,0.1)"))
    fig3.add_hline(y=0, line_dash="dash", line_color="#8b949e", opacity=0.5)
    fig3.update_layout(**LAYOUT, height=320,
                       title="Rolling 12-Month Pearson Correlation: Fuel% vs Nifty%",
                       yaxis_range=[-1, 1])
    st.plotly_chart(fig3, use_container_width=True)
