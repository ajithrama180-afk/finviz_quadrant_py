import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Finviz Elite — 4-Quadrant Stock Classifier", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #34d399, #60a5fa);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }
    .premium { color: #34d399; }
    .speculative { color: #fbbf24; }
    .discount { color: #60a5fa; }
    .danger { color: #f87171; }
    </style>
""", unsafe_allow_html=True)

# ===== CONSTANTS =====
COL = {
    'ticker': 'Ticker',
    'company': 'Company',
    'sector': 'Sector',
    'industry': 'Industry',
    'marketCap': 'Market Cap',
    'pe': 'P/E',
    'fwdPE': 'Forward P/E',
    'peg': 'PEG',
    'ps': 'P/S',
    'pb': 'P/B',
    'epsThisY': 'EPS Growth This Year',
    'epsNextY': 'EPS Growth Next Year',
    'epsPast5Y': 'EPS Growth Past 5 Years',
    'epsNext5Y': 'EPS Growth Next 5 Years',
    'salesPast5Y': 'Sales Growth Past 5 Years',
    'epsQoQ': 'EPS Growth Quarter Over Quarter',
    'salesQoQ': 'Sales Growth Quarter Over Quarter',
    'debtEq': 'Total Debt/Equity',
    'grossM': 'Gross Margin',
    'operM': 'Operating Margin',
    'profitM': 'Profit Margin',
    'divYield': 'Dividend Yield',
    'payout': 'Payout Ratio',
    'epsTTM': 'EPS (ttm)',
    'perfWeek': 'Performance (Week)',
    'perfMonth': 'Performance (Month)',
    'perfQuarter': 'Performance (Quarter)',
    'perfHalf': 'Performance (Half Year)',
    'perfYear': 'Performance (Year)',
    'perfYTD': 'Performance (YTD)',
    'beta': 'Beta',
    'volMonth': 'Volatility (Month)',
    'sma20': '20-Day Simple Moving Average',
    'sma50': '50-Day Simple Moving Average',
    'sma200': '200-Day Simple Moving Average',
    'high52w': '52-Week High',
    'low52w': '52-Week Low',
    'rsi': 'Relative Strength Index (14)',
    'price': 'Price',
    'change': 'Change',
    'analystRecom': 'Analyst Recom',
    'sharesOut': 'Shares Outstanding',
    'relVol': 'Relative Volume'
}

FW = {
    'roe': {'label': 'ROE / ROA %', 'w': 18},
    'profitM': {'label': 'Profit Margin', 'w': 12},
    'grossM': {'label': 'Gross Margin', 'w': 8},
    'operM': {'label': 'Oper Margin', 'w': 10},
    'pe': {'label': 'P/E (lower=better)', 'w': 12},
    'peg': {'label': 'PEG (lower=better)', 'w': 8},
    'debtEq': {'label': 'Debt/Eq (lower=better)', 'w': 12},
    'epsThisY': {'label': 'EPS Growth Y', 'w': 10},
    'epsPast5Y': {'label': 'EPS Growth 5Y', 'w': 5},
    'salesPast5Y': {'label': 'Sales Growth 5Y', 'w': 5}
}

TW = {
    'sma20': {'label': 'vs SMA20', 'w': 12},
    'sma50': {'label': 'vs SMA50', 'w': 18},
    'sma200': {'label': 'vs SMA200', 'w': 22},
    'rsi': {'label': 'RSI (14)', 'w': 12},
    'w52pos': {'label': '52W Range Pos', 'w': 10},
    'perfMonth': {'label': 'Perf Month', 'w': 10},
    'perfQuarter': {'label': 'Perf Quarter', 'w': 8},
    'perfHalf': {'label': 'Perf Half Y', 'w': 8}
}

# ===== SESSION STATE =====
if 'all_stocks' not in st.session_state:
    st.session_state.all_stocks = []
if 'fund_weights' not in st.session_state:
    st.session_state.fund_weights = {k: v['w'] for k, v in FW.items()}
if 'tech_weights' not in st.session_state:
    st.session_state.tech_weights = {k: v['w'] for k, v in TW.items()}
if 'fund_cutoff' not in st.session_state:
    st.session_state.fund_cutoff = 50
if 'tech_cutoff' not in st.session_state:
    st.session_state.tech_cutoff = 50

# ===== HELPER FUNCTIONS =====
def safe_float(v):
    """Convert value to float, handling percentages and commas"""
    if v is None or v == '' or v == '-' or pd.isna(v):
        return None
    try:
        s = str(v).replace('%', '').replace(',', '').strip()
        if s == '':
            return None
        return float(s)
    except:
        return None

def norm(val, low, high):
    """Normalize value between 0 and 1"""
    if val is None:
        return None
    return np.clip((val - low) / (high - low), 0, 1)

def calc_fund_score(row):
    """Calculate fundamental score"""
    score = 0
    tw = 0
    
    # ROE
    if pd.notna(row.get('ROE')) and row.get('ROE') is not None:
        roe = safe_float(row.get('ROE'))
        if roe is not None:
            tw += FW['roe']['w']
            score += FW['roe']['w'] * norm(roe, -5, 35)
    
    # Profit Margin
    profitM = safe_float(row.get('Profit Margin'))
    if profitM is not None:
        tw += FW['profitM']['w']
        score += FW['profitM']['w'] * norm(profitM, -10, 30)
    
    # Gross Margin
    grossM = safe_float(row.get('Gross Margin'))
    if grossM is not None:
        tw += FW['grossM']['w']
        score += FW['grossM']['w'] * norm(grossM, 0, 70)
    
    # Operating Margin
    operM = safe_float(row.get('Operating Margin'))
    if operM is not None:
        tw += FW['operM']['w']
        score += FW['operM']['w'] * norm(operM, -10, 35)
    
    # P/E (lower is better)
    pe = safe_float(row.get('P/E'))
    if pe is not None and pe > 0:
        tw += FW['pe']['w']
        score += FW['pe']['w'] * (1 - norm(pe, 5, 60))
    
    # PEG (lower is better)
    peg = safe_float(row.get('PEG'))
    if peg is not None and peg > 0:
        tw += FW['peg']['w']
        score += FW['peg']['w'] * (1 - norm(peg, 0, 3))
    
    # Debt/Equity (lower is better)
    debtEq = safe_float(row.get('Total Debt/Equity'))
    if debtEq is not None and debtEq >= 0:
        tw += FW['debtEq']['w']
        score += FW['debtEq']['w'] * (1 - norm(debtEq, 0, 3))
    
    # EPS Growth This Year
    epsThisY = safe_float(row.get('EPS Growth This Year'))
    if epsThisY is not None:
        tw += FW['epsThisY']['w']
        score += FW['epsThisY']['w'] * norm(epsThisY, -20, 40)
    
    # EPS Growth Past 5Y
    epsPast5Y = safe_float(row.get('EPS Growth Past 5 Years'))
    if epsPast5Y is not None:
        tw += FW['epsPast5Y']['w']
        score += FW['epsPast5Y']['w'] * norm(epsPast5Y, -10, 30)
    
    # Sales Growth Past 5Y
    salesPast5Y = safe_float(row.get('Sales Growth Past 5 Years'))
    if salesPast5Y is not None:
        tw += FW['salesPast5Y']['w']
        score += FW['salesPast5Y']['w'] * norm(salesPast5Y, -5, 25)
    
    if tw == 0:
        return 50
    return round((score / tw) * 100)

def calc_tech_score(row):
    """Calculate technical score"""
    score = 0
    tw = 0
    
    # vs SMA20
    sma20 = safe_float(row.get('20-Day Simple Moving Average'))
    if sma20 is not None:
        tw += TW['sma20']['w']
        score += TW['sma20']['w'] * norm(sma20, -20, 20)
    
    # vs SMA50
    sma50 = safe_float(row.get('50-Day Simple Moving Average'))
    if sma50 is not None:
        tw += TW['sma50']['w']
        score += TW['sma50']['w'] * norm(sma50, -30, 30)
    
    # vs SMA200
    sma200 = safe_float(row.get('200-Day Simple Moving Average'))
    if sma200 is not None:
        tw += TW['sma200']['w']
        score += TW['sma200']['w'] * norm(sma200, -40, 40)
    
    # RSI
    rsi = safe_float(row.get('Relative Strength Index (14)'))
    if rsi is not None:
        tw += TW['rsi']['w']
        score += TW['rsi']['w'] * norm(rsi, 20, 80)
    
    # 52W Low Position
    low52w = safe_float(row.get('52-Week Low'))
    if low52w is not None:
        tw += TW['w52pos']['w']
        score += TW['w52pos']['w'] * norm(low52w, 0, 100)
    
    # Perf Month
    perfMonth = safe_float(row.get('Performance (Month)'))
    if perfMonth is not None:
        tw += TW['perfMonth']['w']
        score += TW['perfMonth']['w'] * norm(perfMonth, -20, 20)
    
    # Perf Quarter
    perfQuarter = safe_float(row.get('Performance (Quarter)'))
    if perfQuarter is not None:
        tw += TW['perfQuarter']['w']
        score += TW['perfQuarter']['w'] * norm(perfQuarter, -30, 30)
    
    # Perf Half Year
    perfHalf = safe_float(row.get('Performance (Half Year)'))
    if perfHalf is not None:
        tw += TW['perfHalf']['w']
        score += TW['perfHalf']['w'] * norm(perfHalf, -30, 30)
    
    if tw == 0:
        return 50
    return round((score / tw) * 100)

def classify_zone(fund_score, tech_score, fund_cut, tech_cut):
    """Classify stock into quadrant zone"""
    if fund_score >= fund_cut and tech_score >= tech_cut:
        return 'premium'
    elif fund_score >= fund_cut and tech_score < tech_cut:
        return 'discount'
    elif fund_score < fund_cut and tech_score >= tech_cut:
        return 'speculative'
    else:
        return 'danger'

def get_market_cap_category(market_cap):
    """Categorize market cap"""
    if pd.isna(market_cap) or market_cap is None:
        return None
    try:
        mc = safe_float(market_cap)
        if mc is None:
            return None
        if mc >= 200000:
            return 'mega'
        elif mc >= 10000:
            return 'large'
        elif mc >= 2000:
            return 'mid'
        elif mc >= 300:
            return 'small'
        else:
            return 'micro'
    except:
        return None

def process_stocks(df):
    """Process CSV data and calculate scores"""
    stocks = []
    for idx, row in df.iterrows():
        ticker = str(row.get('Ticker', '')).strip()
        if not ticker:
            continue
        
        fund_score = calc_fund_score(row)
        tech_score = calc_tech_score(row)
        zone = classify_zone(fund_score, tech_score, st.session_state.fund_cutoff, st.session_state.tech_cutoff)
        
        stock = {
            'ticker': ticker,
            'company': str(row.get('Company', ticker)).strip(),
            'sector': str(row.get('Sector', 'Unknown')).strip(),
            'fund_score': fund_score,
            'tech_score': tech_score,
            'zone': zone,
            'price': safe_float(row.get('Price')),
            'pe': safe_float(row.get('P/E')),
            'roe': safe_float(row.get('ROE')),
            'debtEq': safe_float(row.get('Total Debt/Equity')),
            'rsi': safe_float(row.get('Relative Strength Index (14)')),
            'sma200': safe_float(row.get('200-Day Simple Moving Average')),
            'perfYear': safe_float(row.get('Performance (Year)')),
            'market_cap': safe_float(row.get('Market Cap')),
        }
        stocks.append(stock)
    return stocks

# ===== TITLE & HEADER =====
st.markdown("<h1 style='text-align: center'>📊 Finviz Elite — 4-Quadrant Classifier</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray'>Upload Finviz CSV → Instant Fundamental × Technical classification</p>", unsafe_allow_html=True)

# ===== FILE UPLOAD =====
uploaded_file = st.file_uploader("📁 Drop your Finviz CSV here or click to browse", type=['csv'])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.all_stocks = process_stocks(df)
        st.success(f"✅ Loaded {len(st.session_state.all_stocks)} stocks from {uploaded_file.name}")
    except Exception as e:
        st.error(f"❌ Error parsing file: {str(e)}")

# ===== TABS =====
tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quadrants", "🔵 Scatter", "📋 Table", "⚙️ Settings"])

if st.session_state.all_stocks:
    stocks_df = pd.DataFrame(st.session_state.all_stocks)
    
    # Calculate zone counts
    zone_counts = stocks_df['zone'].value_counts().to_dict()
    
    with tab1:
        st.subheader("Stock Classification by Quadrant")
        
        # Display zone counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🟢 Premium Zone", zone_counts.get('premium', 0), "Strong + Uptrend")
        with col2:
            st.metric("🟡 Speculative", zone_counts.get('speculative', 0), "Weak + Uptrend")
        with col3:
            st.metric("🔵 Discount", zone_counts.get('discount', 0), "Strong + Downtrend")
        with col4:
            st.metric("🔴 Danger Zone", zone_counts.get('danger', 0), "Weak + Downtrend")
        
        st.markdown("---")
        
        # Filters
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            sector_filter = st.selectbox("Sector", ["All"] + sorted(stocks_df['sector'].unique().tolist()))
        with col2:
            zone_filter = st.selectbox("Zone", ["All", "premium", "discount", "speculative", "danger"])
        with col3:
            cap_filter = st.selectbox("Market Cap", ["All", "mega", "large", "mid", "small", "micro"])
        with col4:
            search_filter = st.text_input("Search", placeholder="Ticker or name...")
        
        # Apply filters
        filtered_df = stocks_df.copy()
        if sector_filter != "All":
            filtered_df = filtered_df[filtered_df['sector'] == sector_filter]
        if zone_filter != "All":
            filtered_df = filtered_df[filtered_df['zone'] == zone_filter]
        if search_filter:
            filtered_df = filtered_df[
                filtered_df['ticker'].str.contains(search_filter, case=False, na=False) |
                filtered_df['company'].str.contains(search_filter, case=False, na=False)
            ]
        
        # Display quadrants
        quad_col1, quad_col2 = st.columns(2)
        
        with quad_col1:
            st.markdown("#### 🟢 Premium Zone")
            premium = filtered_df[filtered_df['zone'] == 'premium'].sort_values('fund_score', ascending=False)
            st.dataframe(premium[['ticker', 'company', 'fund_score', 'tech_score', 'price']].head(50), use_container_width=True)
            
            st.markdown("#### 🔵 Discount / Value Trap")
            discount = filtered_df[filtered_df['zone'] == 'discount'].sort_values('fund_score', ascending=False)
            st.dataframe(discount[['ticker', 'company', 'fund_score', 'tech_score', 'price']].head(50), use_container_width=True)
        
        with quad_col2:
            st.markdown("#### 🟡 Speculative Bubble")
            speculative = filtered_df[filtered_df['zone'] == 'speculative'].sort_values('fund_score', ascending=False)
            st.dataframe(speculative[['ticker', 'company', 'fund_score', 'tech_score', 'price']].head(50), use_container_width=True)
            
            st.markdown("#### 🔴 Danger Zone")
            danger = filtered_df[filtered_df['zone'] == 'danger'].sort_values('fund_score', ascending=False)
            st.dataframe(danger[['ticker', 'company', 'fund_score', 'tech_score', 'price']].head(50), use_container_width=True)
    
    with tab2:
        st.subheader("Fundamental vs Technical Score Scatter")
        
        # Create scatter plot
        fig = go.Figure()
        
        for zone, color, symbol in [('premium', '#34d399', 'circle'), ('speculative', '#fbbf24', 'circle'),
                                     ('discount', '#60a5fa', 'circle'), ('danger', '#f87171', 'circle')]:
            zone_data = stocks_df[stocks_df['zone'] == zone]
            fig.add_trace(go.Scatter(
                x=zone_data['fund_score'],
                y=zone_data['tech_score'],
                mode='markers+text',
                name=zone.capitalize(),
                marker=dict(size=12, color=color, opacity=0.7),
                text=zone_data['ticker'],
                textposition='top center',
                textfont=dict(size=8),
                hovertemplate='<b>%{text}</b><br>Fund: %{x}<br>Tech: %{y}<extra></extra>'
            ))
        
        # Add quadrant lines
        fig.add_vline(x=st.session_state.fund_cutoff, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        fig.add_hline(y=st.session_state.tech_cutoff, line_dash="dash", line_color="rgba(255,255,255,0.3)")
        
        fig.update_layout(
            title="Fundamental vs Technical Score",
            xaxis_title="Fundamental Score (Left = Strong)",
            yaxis_title="Technical Score (Top = Uptrend)",
            hovermode='closest',
            height=600,
            template='plotly_dark',
            xaxis=dict(range=[0, 100]),
            yaxis=dict(range=[0, 100])
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("All Stocks — Scored & Classified")
        
        # Sorting and Filtering
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col1:
            sort_by = st.selectbox("Sort by", ["fund_score", "tech_score", "ticker", "price"], key="table_sort_by")
        with col2:
            sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="table_sort_order")
        with col3:
            zone_filter_multi = st.multiselect(
                "Filter by Zone",
                ["premium", "speculative", "discount", "danger"],
                default=["premium", "speculative", "discount", "danger"],
                key="table_zone_filter"
            )
        with col4:
            sector_filter_table = st.selectbox(
                "Filter by Sector",
                ["All"] + sorted(stocks_df['sector'].unique().tolist()),
                key="table_sector_filter"
            )
        
        # Apply filters
        table_filtered_df = stocks_df.copy()
        if zone_filter_multi:
            table_filtered_df = table_filtered_df[table_filtered_df['zone'].isin(zone_filter_multi)]
        if sector_filter_table != "All":
            table_filtered_df = table_filtered_df[table_filtered_df['sector'] == sector_filter_table]
        
        sorted_df = table_filtered_df.sort_values(sort_by, ascending=(sort_order == "Ascending"))
        
        st.dataframe(
            sorted_df[[
                'ticker', 'company', 'sector', 'zone', 'fund_score', 'tech_score',
                'price', 'pe', 'roe', 'debtEq', 'rsi', 'sma200', 'perfYear'
            ]],
            use_container_width=True
        )
        
        # Export button
        if st.button("⬇ Export Filtered Data as CSV", key="table_export_btn"):
            csv = sorted_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="finviz_quadrant_export.csv",
                mime="text/csv"
            )
    
    with tab4:
        st.subheader("⚙️ Scoring Weights & Thresholds")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📋 Fundamental Score Weights")
            for key, val in FW.items():
                st.session_state.fund_weights[key] = st.slider(
                    val['label'], 0, 50, st.session_state.fund_weights.get(key, val['w']),
                    key=f"fw_{key}"
                )
            
            st.session_state.fund_cutoff = st.slider(
                "Strong/Weak Cutoff", 20, 80, st.session_state.fund_cutoff,
                key="fund_cutoff_slider"
            )
        
        with col2:
            st.markdown("#### 📈 Technical Score Weights")
            for key, val in TW.items():
                st.session_state.tech_weights[key] = st.slider(
                    val['label'], 0, 50, st.session_state.tech_weights.get(key, val['w']),
                    key=f"tw_{key}"
                )
            
            st.session_state.tech_cutoff = st.slider(
                "Uptrend/Downtrend Cutoff", 20, 80, st.session_state.tech_cutoff,
                key="tech_cutoff_slider"
            )
        
        # Recalculate button
        if st.button("🔄 Recalculate All Scores", key="recalc_btn"):
            # Update FW and TW with new weights
            for key in FW:
                FW[key]['w'] = st.session_state.fund_weights[key]
            for key in TW:
                TW[key]['w'] = st.session_state.tech_weights[key]
            
            # Reprocess stocks with new weights
            stocks_df_recalc = pd.DataFrame(st.session_state.all_stocks)
            for idx, row in stocks_df_recalc.iterrows():
                st.session_state.all_stocks[idx]['fund_score'] = calc_fund_score(stocks_df_recalc.iloc[idx])
                st.session_state.all_stocks[idx]['tech_score'] = calc_tech_score(stocks_df_recalc.iloc[idx])
                st.session_state.all_stocks[idx]['zone'] = classify_zone(
                    st.session_state.all_stocks[idx]['fund_score'],
                    st.session_state.all_stocks[idx]['tech_score'],
                    st.session_state.fund_cutoff,
                    st.session_state.tech_cutoff
                )
            st.success("✅ Scores recalculated!")
            st.rerun()

else:
    st.info("📁 Upload a Finviz CSV file to get started")
