import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Finviz Elite — 4-Quadrant Stock Classifier", layout="wide")

# ===== CUSTOM STYLING =====
st.markdown("""
    <style>
    /* Global Styling */
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }
    
    /* Card Styling */
    .zone-card {
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        color: white;
        font-weight: 500;
    }
    .premium-card {
        background: linear-gradient(135deg, #34d399, #10b981);
        border-left: 4px solid #059669;
    }
    .speculative-card {
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        border-left: 4px solid #d97706;
    }
    .discount-card {
        background: linear-gradient(135deg, #60a5fa, #3b82f6);
        border-left: 4px solid #1d4ed8;
    }
    .danger-card {
        background: linear-gradient(135deg, #f87171, #ef4444);
        border-left: 4px solid #dc2626;
    }
    
    /* Metric Cards */
    .metric-card {
        text-align: center;
        padding: 20px;
        border-radius: 10px;
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        border: 1px solid #cbd5e1;
    }
    
    /* Section Headers */
    .section-header {
        border-bottom: 3px solid #e2e8f0;
        padding-bottom: 10px;
        margin: 20px 0 15px 0;
    }
    
    /* Filters Container */
    .filter-container {
        background: #f8fafc;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    /* Data Table Enhancement */
    .dataframe-wrapper {
        margin: 10px 0;
    }
    
    /* Sidebar Styling */
    .sidebar-section {
        margin: 20px 0;
        padding: 15px 0;
        border-bottom: 1px solid #e2e8f0;
    }
    </style>
""", unsafe_allow_html=True)

# ===== CONSTANTS & CONFIGURATION =====
FUNDAMENTAL_WEIGHTS = {
    'roe': {'label': 'ROE / ROA %', 'w': 18},
    'profitM': {'label': 'Profit Margin', 'w': 12},
    'grossM': {'label': 'Gross Margin', 'w': 8},
    'operM': {'label': 'Operating Margin', 'w': 10},
    'pe': {'label': 'P/E (lower=better)', 'w': 12},
    'peg': {'label': 'PEG (lower=better)', 'w': 8},
    'debtEq': {'label': 'Debt/Equity (lower=better)', 'w': 12},
    'epsThisY': {'label': 'EPS Growth This Year', 'w': 10},
    'epsPast5Y': {'label': 'EPS Growth Past 5Y', 'w': 5},
    'salesPast5Y': {'label': 'Sales Growth Past 5Y', 'w': 5}
}

TECHNICAL_WEIGHTS = {
    'sma20': {'label': '20-Day Moving Average', 'w': 12},
    'sma50': {'label': '50-Day Moving Average', 'w': 18},
    'sma200': {'label': '200-Day Moving Average', 'w': 22},
    'rsi': {'label': 'RSI (14)', 'w': 12},
    'w52pos': {'label': '52-Week Range Position', 'w': 10},
    'perfMonth': {'label': 'Monthly Performance', 'w': 10},
    'perfQuarter': {'label': 'Quarterly Performance', 'w': 8},
    'perfHalf': {'label': 'Half-Year Performance', 'w': 8}
}

ZONE_CONFIG = {
    'premium': {'color': '#34d399', 'emoji': '🟢', 'label': 'Premium Zone', 'desc': 'Strong Fundamentals + Uptrend'},
    'speculative': {'color': '#fbbf24', 'emoji': '🟡', 'label': 'Speculative', 'desc': 'Weak Fundamentals + Uptrend'},
    'discount': {'color': '#60a5fa', 'emoji': '🔵', 'label': 'Discount/Value', 'desc': 'Strong Fundamentals + Downtrend'},
    'danger': {'color': '#f87171', 'emoji': '🔴', 'label': 'Danger Zone', 'desc': 'Weak Fundamentals + Downtrend'}
}

# ===== SESSION STATE INITIALIZATION =====
def init_session_state():
    if 'all_stocks' not in st.session_state:
        st.session_state.all_stocks = []
    if 'fund_weights' not in st.session_state:
        st.session_state.fund_weights = {k: v['w'] for k, v in FUNDAMENTAL_WEIGHTS.items()}
    if 'tech_weights' not in st.session_state:
        st.session_state.tech_weights = {k: v['w'] for k, v in TECHNICAL_WEIGHTS.items()}
    if 'fund_cutoff' not in st.session_state:
        st.session_state.fund_cutoff = 50
    if 'tech_cutoff' not in st.session_state:
        st.session_state.tech_cutoff = 50
    if 'last_loaded' not in st.session_state:
        st.session_state.last_loaded = None

init_session_state()

# ===== UTILITY FUNCTIONS =====
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

def normalize(val, low, high):
    """Normalize value between 0 and 1"""
    if val is None:
        return None
    return np.clip((val - low) / (high - low), 0, 1)

def calc_fundamental_score(row):
    """Calculate fundamental score based on weighted metrics"""
    score = 0
    total_weight = 0
    metrics = {
        'roe': ('ROE', -5, 35),
        'profitM': ('Profit Margin', -10, 30),
        'grossM': ('Gross Margin', 0, 70),
        'operM': ('Operating Margin', -10, 35),
        'pe': ('P/E', 5, 60, True),  # True means lower is better
        'peg': ('PEG', 0, 3, True),
        'debtEq': ('Total Debt/Equity', 0, 3, True),
        'epsThisY': ('EPS Growth This Year', -20, 40),
        'epsPast5Y': ('EPS Growth Past 5 Years', -10, 30),
        'salesPast5Y': ('Sales Growth Past 5 Years', -5, 25),
    }
    
    for key, (col_name, low, high, *inverse) in metrics.items():
        val = safe_float(row.get(col_name))
        if val is not None and (key != 'pe' or val > 0) and (key != 'peg' or val > 0) and (key != 'debtEq' or val >= 0):
            weight = FUNDAMENTAL_WEIGHTS[key]['w']
            total_weight += weight
            norm_val = normalize(val, low, high)
            if inverse and inverse[0]:
                norm_val = 1 - norm_val
            score += weight * norm_val
    
    return round((score / total_weight) * 100) if total_weight > 0 else 50

def calc_technical_score(row):
    """Calculate technical score based on weighted metrics"""
    score = 0
    total_weight = 0
    metrics = {
        'sma20': ('20-Day Simple Moving Average', -20, 20),
        'sma50': ('50-Day Simple Moving Average', -30, 30),
        'sma200': ('200-Day Simple Moving Average', -40, 40),
        'rsi': ('Relative Strength Index (14)', 20, 80),
        'w52pos': ('52-Week Low', 0, 100),
        'perfMonth': ('Performance (Month)', -20, 20),
        'perfQuarter': ('Performance (Quarter)', -30, 30),
        'perfHalf': ('Performance (Half Year)', -30, 30),
    }
    
    for key, (col_name, low, high) in metrics.items():
        val = safe_float(row.get(col_name))
        if val is not None:
            weight = TECHNICAL_WEIGHTS[key]['w']
            total_weight += weight
            norm_val = normalize(val, low, high)
            score += weight * norm_val
    
    return round((score / total_weight) * 100) if total_weight > 0 else 50

def classify_zone(fund_score, tech_score, fund_cutoff, tech_cutoff):
    """Classify stock into quadrant zone"""
    if fund_score >= fund_cutoff and tech_score >= tech_cutoff:
        return 'premium'
    elif fund_score >= fund_cutoff and tech_score < tech_cutoff:
        return 'discount'
    elif fund_score < fund_cutoff and tech_score >= tech_cutoff:
        return 'speculative'
    else:
        return 'danger'

def process_stocks(df):
    """Process CSV data and calculate scores"""
    stocks = []
    for idx, row in df.iterrows():
        ticker = str(row.get('Ticker', '')).strip()
        if not ticker:
            continue
        
        fund_score = calc_fundamental_score(row)
        tech_score = calc_technical_score(row)
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

# ===== UI COMPONENTS =====
def render_zone_metrics(stocks_df):
    """Render zone count metrics in a grid"""
    zone_counts = stocks_df['zone'].value_counts().to_dict()
    
    col1, col2, col3, col4 = st.columns(4, gap="medium")
    
    with col1:
        count = zone_counts.get('premium', 0)
        st.metric(
            f"{ZONE_CONFIG['premium']['emoji']} Premium",
            count,
            help="Strong fundamentals + Uptrend"
        )
    with col2:
        count = zone_counts.get('speculative', 0)
        st.metric(
            f"{ZONE_CONFIG['speculative']['emoji']} Speculative",
            count,
            help="Weak fundamentals + Uptrend"
        )
    with col3:
        count = zone_counts.get('discount', 0)
        st.metric(
            f"{ZONE_CONFIG['discount']['emoji']} Discount",
            count,
            help="Strong fundamentals + Downtrend"
        )
    with col4:
        count = zone_counts.get('danger', 0)
        st.metric(
            f"{ZONE_CONFIG['danger']['emoji']} Danger",
            count,
            help="Weak fundamentals + Downtrend"
        )

def render_filter_controls():
    """Render unified filter controls"""
    st.markdown("#### 🔍 Filter & Search")
    col1, col2, col3, col4 = st.columns(4, gap="small")
    
    with col1:
        sector = st.selectbox(
            "Sector",
            ["All"] + sorted(st.session_state.sector_list),
            key="sector_filter"
        )
    with col2:
        zone = st.selectbox(
            "Zone",
            ["All", "premium", "speculative", "discount", "danger"],
            key="zone_filter"
        )
    with col3:
        search = st.text_input(
            "Search",
            placeholder="Ticker or company name...",
            key="search_filter"
        )
    with col4:
        sort_by = st.selectbox(
            "Sort by",
            ["Fund Score", "Tech Score", "Ticker", "Price"],
            key="sort_filter"
        )
    
    return sector, zone, search, sort_by

def apply_filters(df, sector_filter, zone_filter, search_filter):
    """Apply filters to dataframe"""
    filtered = df.copy()
    
    if sector_filter != "All":
        filtered = filtered[filtered['sector'] == sector_filter]
    if zone_filter != "All":
        filtered = filtered[filtered['zone'] == zone_filter]
    if search_filter:
        mask = (
            filtered['ticker'].str.contains(search_filter, case=False, na=False) |
            filtered['company'].str.contains(search_filter, case=False, na=False)
        )
        filtered = filtered[mask]
    
    return filtered

def render_quadrant_section(stocks_df):
    """Render detailed quadrant view with stocks organized by zone"""
    col1, col2 = st.columns(2, gap="medium")
    
    zones = ['premium', 'discount', 'speculative', 'danger']
    cols = [col1, col2, col1, col2]
    
    for idx, zone in enumerate(zones):
        with cols[idx]:
            zone_data = stocks_df[stocks_df['zone'] == zone].sort_values('fund_score', ascending=False).head(25)
            
            st.markdown(f"""
                <div class='zone-card {zone}-card'>
                    <strong>{ZONE_CONFIG[zone]['emoji']} {ZONE_CONFIG[zone]['label']}</strong><br>
                    <small>{ZONE_CONFIG[zone]['desc']}</small>
                </div>
            """, unsafe_allow_html=True)
            
            if len(zone_data) > 0:
                display_cols = ['ticker', 'company', 'fund_score', 'tech_score', 'price']
                st.dataframe(
                    zone_data[display_cols].rename(columns={
                        'ticker': 'Ticker',
                        'company': 'Company',
                        'fund_score': 'Fund',
                        'tech_score': 'Tech',
                        'price': 'Price'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No stocks in this zone")

def render_scatter_plot(stocks_df):
    """Render fundamental vs technical scatter plot"""
    fig = go.Figure()
    
    for zone in ['premium', 'speculative', 'discount', 'danger']:
        zone_data = stocks_df[stocks_df['zone'] == zone]
        if len(zone_data) > 0:
            fig.add_trace(go.Scatter(
                x=zone_data['fund_score'],
                y=zone_data['tech_score'],
                mode='markers+text',
                name=ZONE_CONFIG[zone]['label'],
                marker=dict(
                    size=10,
                    color=ZONE_CONFIG[zone]['color'],
                    opacity=0.7,
                    line=dict(width=1, color='white')
                ),
                text=zone_data['ticker'],
                textposition='top center',
                textfont=dict(size=9, color='white'),
                hovertemplate='<b>%{text}</b><br>Fund: %{x:.0f}<br>Tech: %{y:.0f}<extra></extra>'
            ))
    
    fig.add_vline(x=st.session_state.fund_cutoff, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Fund Cutoff")
    fig.add_hline(y=st.session_state.tech_cutoff, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Tech Cutoff")
    
    fig.update_layout(
        title="Fundamental vs Technical Score Distribution",
        xaxis_title="Fundamental Score (Left = Strong)",
        yaxis_title="Technical Score (Top = Uptrend)",
        hovermode='closest',
        height=600,
        template='plotly_dark',
        xaxis=dict(range=[0, 100]),
        yaxis=dict(range=[0, 100]),
        showlegend=True
    )
    
    return fig

# ===== MAIN APPLICATION =====
def main():
    # Header
    st.markdown("<h1 style='text-align: center; margin-bottom: 10px;'>📊 Finviz Elite</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray; margin-top: -10px; margin-bottom: 20px;'>4-Quadrant Stock Classification System</p>", unsafe_allow_html=True)
    
    # Sidebar: File Upload
    with st.sidebar:
        st.markdown("### 📥 Data Upload")
        
        uploaded_file = st.file_uploader("Upload Finviz CSV", type=['csv'], help="Export from Finviz Elite")
        
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                st.session_state.all_stocks = process_stocks(df)
                st.session_state.sector_list = [s for s in df['Sector'].unique() if pd.notna(s)]
                st.session_state.last_loaded = pd.Timestamp.now().strftime("%b %d, %Y %I:%M %p")
                st.success(f"✅ {len(st.session_state.all_stocks)} stocks loaded", icon="✓")
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}", icon="✕")
        
        st.markdown("---")
        
        FINVIZ_URL = (
            "https://elite.finviz.com/export.ashx?v=152"
            "&f=ind_stocksonly,sh_avgvol_o2000,sh_price_o50"
            "&c=1,2,3,4,6,7,9,10,11,13,14,15,16,17,18,19,20,21,22,23,"
            "24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,"
            "43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,"
            "62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,"
            "81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96"
        )
        st.link_button("📊 Export from Finviz", FINVIZ_URL, use_container_width=True)
        st.caption("Click to export CSV from Finviz Elite")
        
        if st.session_state.last_loaded:
            st.markdown("---")
            st.caption(f"🕐 Last loaded: {st.session_state.last_loaded}")
        
        st.markdown("---")
        
        # Sidebar: Settings
        with st.expander("⚙️ Settings & Weights"):
            st.markdown("**Fundamental Score Cutoff**")
            st.session_state.fund_cutoff = st.slider(
                "Strong/Weak threshold",
                20, 80, st.session_state.fund_cutoff,
                key="sidebar_fund_cutoff"
            )
            
            st.markdown("**Technical Score Cutoff**")
            st.session_state.tech_cutoff = st.slider(
                "Uptrend/Downtrend threshold",
                20, 80, st.session_state.tech_cutoff,
                key="sidebar_tech_cutoff"
            )
    
    # Main content
    if not st.session_state.all_stocks:
        st.info("📁 **Get Started:** Upload a Finviz CSV file from the sidebar to begin analysis")
        st.markdown("""
            ### How to use this tool:
            1. **Export data** from Finviz Elite using the link in the sidebar
            2. **Upload the CSV** file using the file uploader
            3. **View results** in the tabs below
            4. **Customize weights** in the settings panel
        """)
    else:
        stocks_df = pd.DataFrame(st.session_state.all_stocks)
        if 'sector_list' not in st.session_state:
            st.session_state.sector_list = stocks_df['sector'].unique().tolist()
        
        # Unified Metrics
        st.markdown("<div class='section-header'><strong>📈 Portfolio Overview</strong></div>", unsafe_allow_html=True)
        render_zone_metrics(stocks_df)
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Quadrants", "📊 Scatter Plot", "📋 Full Table", "⚙️ Advanced Settings"])
        
        with tab1:
            st.markdown("<div class='section-header'><strong>Stock Classification by Zone</strong></div>", unsafe_allow_html=True)
            
            # Filters
            sector_f, zone_f, search_f, _ = render_filter_controls()
            filtered_df = apply_filters(stocks_df, sector_f, zone_f, search_f)
            
            st.markdown("---")
            render_quadrant_section(filtered_df)
        
        with tab2:
            st.markdown("<div class='section-header'><strong>Fundamental vs Technical Analysis</strong></div>", unsafe_allow_html=True)
            
            sector_f, zone_f, search_f, _ = render_filter_controls()
            filtered_df = apply_filters(stocks_df, sector_f, zone_f, search_f)
            
            st.plotly_chart(render_scatter_plot(filtered_df), use_container_width=True)
        
        with tab3:
            st.markdown("<div class='section-header'><strong>All Stocks — Complete Data</strong></div>", unsafe_allow_html=True)
            
            col1, col2, col3, col4 = st.columns(4, gap="small")
            with col1:
                sort_by_map = {
                    'Fund Score': 'fund_score',
                    'Tech Score': 'tech_score',
                    'Ticker': 'ticker',
                    'Price': 'price'
                }
                sort_col = st.selectbox("Sort by", list(sort_by_map.keys()), key="table_sort")
            with col2:
                sort_order = st.selectbox("Order", ["Descending", "Ascending"], key="table_order")
            with col3:
                zone_multi = st.multiselect(
                    "Filter Zones",
                    ["premium", "speculative", "discount", "danger"],
                    default=["premium", "speculative", "discount", "danger"],
                    key="table_zone_multi"
                )
            with col4:
                sector_filter_table = st.selectbox(
                    "Filter Sector",
                    ["All"] + sorted(st.session_state.sector_list),
                    key="table_sector"
                )
            
            # Apply filters and sort
            table_df = stocks_df.copy()
            if zone_multi:
                table_df = table_df[table_df['zone'].isin(zone_multi)]
            if sector_filter_table != "All":
                table_df = table_df[table_df['sector'] == sector_filter_table]
            
            table_df = table_df.sort_values(
                sort_by_map[sort_col],
                ascending=(sort_order == "Ascending")
            )
            
            st.dataframe(
                table_df[[
                    'ticker', 'company', 'sector', 'zone', 'fund_score', 'tech_score',
                    'price', 'pe', 'roe', 'debtEq', 'rsi', 'sma200', 'perfYear'
                ]].rename(columns={
                    'ticker': 'Ticker',
                    'company': 'Company',
                    'sector': 'Sector',
                    'zone': 'Zone',
                    'fund_score': 'Fund Score',
                    'tech_score': 'Tech Score',
                    'price': 'Price',
                    'pe': 'P/E',
                    'roe': 'ROE',
                    'debtEq': 'Debt/Eq',
                    'rsi': 'RSI',
                    'sma200': 'SMA200',
                    'perfYear': 'YTD Perf'
                }),
                use_container_width=True,
                hide_index=True
            )
            
            # Export options
            col1, col2, col3 = st.columns(3)
            with col1:
                csv = table_df.to_csv(index=False)
                st.download_button(
                    label="⬇ Download CSV",
                    data=csv,
                    file_name="finviz_quadrant_export.csv",
                    mime="text/csv"
                )
        
        with tab4:
            st.markdown("<div class='section-header'><strong>Scoring Weights Configuration</strong></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📋 Fundamental Score Weights")
                st.info("Adjust weights to emphasize different metrics", icon="ℹ️")
                for key, config in FUNDAMENTAL_WEIGHTS.items():
                    st.session_state.fund_weights[key] = st.slider(
                        config['label'],
                        0, 50,
                        st.session_state.fund_weights.get(key, config['w']),
                        key=f"fw_{key}"
                    )
            
            with col2:
                st.markdown("#### 📈 Technical Score Weights")
                st.info("Adjust weights to emphasize different metrics", icon="ℹ️")
                for key, config in TECHNICAL_WEIGHTS.items():
                    st.session_state.tech_weights[key] = st.slider(
                        config['label'],
                        0, 50,
                        st.session_state.tech_weights.get(key, config['w']),
                        key=f"tw_{key}"
                    )
            
            st.markdown("---")
            
            if st.button("🔄 Recalculate All Scores", use_container_width=True, type="primary"):
                # Update weights in constants
                for key in FUNDAMENTAL_WEIGHTS:
                    FUNDAMENTAL_WEIGHTS[key]['w'] = st.session_state.fund_weights[key]
                for key in TECHNICAL_WEIGHTS:
                    TECHNICAL_WEIGHTS[key]['w'] = st.session_state.tech_weights[key]
                
                # Reprocess stocks
                stocks_df_recalc = pd.DataFrame(st.session_state.all_stocks)
                for idx, row in stocks_df_recalc.iterrows():
                    st.session_state.all_stocks[idx]['fund_score'] = calc_fundamental_score(stocks_df_recalc.iloc[idx])
                    st.session_state.all_stocks[idx]['tech_score'] = calc_technical_score(stocks_df_recalc.iloc[idx])
                    st.session_state.all_stocks[idx]['zone'] = classify_zone(
                        st.session_state.all_stocks[idx]['fund_score'],
                        st.session_state.all_stocks[idx]['tech_score'],
                        st.session_state.fund_cutoff,
                        st.session_state.tech_cutoff
                    )
                
                st.success("✅ Scores recalculated successfully!", icon="✓")
                st.rerun()

if __name__ == "__main__":
    main()
