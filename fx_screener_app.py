"""
Solar Finance - FX Multi-Timeframe Screener
A Streamlit dashboard for screening FX pairs across multiple timeframes (5m, 15m, 4h, 1D, 1W)
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(
    page_title="Solar Finance FX Screener",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .big-font {
        font-size:20px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# FX Pairs mapping (yfinance format)
FX_PAIRS = {
    # Major Pairs
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'USDCAD=X',
    'NZD/USD': 'NZDUSD=X',
    # Minor Pairs
    'EUR/GBP': 'EURGBP=X',
    'EUR/JPY': 'EURJPY=X',
    'GBP/JPY': 'GBPJPY=X',
    'EUR/AUD': 'EURAUD=X',
    'EUR/CAD': 'EURCAD=X',
    'AUD/JPY': 'AUDJPY=X',
    'GBP/AUD': 'GBPAUD=X',
    # Indices
    'GER40': '^GDAXI',  # DAX / Germany 40
    'US100': 'NQ=F',  # Nasdaq 100 Futures
    # Metals
    'XAUUSD': 'GC=F',  # Gold Futures
    'XAGUSD': 'SI=F',  # Silver Futures
    # Crypto
    'BTCUSD': 'BTC-USD'
}

TIMEFRAMES = {
    '5m': '5m',
    '15m': '15m',
    '4h': '1h',  # yfinance doesn't have 4h, we'll use 1h and resample
    '1D': '1d',
    '1W': '1wk'
}

def calculate_ema(data, period):
    """Calculate EMA for given period"""
    return data['Close'].ewm(span=period, adjust=False).mean()

def calculate_sentiment(df, max_diff=10.0):
    """
    Multi-Indicator Sentiment based on 20/50 EMA
    Combines Trend Points (50%) and EMA Distance (50%) for composite score
    Returns: sentiment_pct (0-100), sentiment_text, ema20, ema50
    """
    if df.empty or len(df) < 50:
        return None, None, None, None
    
    # Calculate 20 and 50 EMAs (matching Pine Script)
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Get latest values
    last_close = df['Close'].iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    
    # ========== COMPONENT 1: TREND POINTS ==========
    trend_points = 0.0
    
    # Close above short EMA
    if last_close > last_ema20:
        trend_points += 33.33
    
    # Close above long EMA
    if last_close > last_ema50:
        trend_points += 33.33
    
    # Short EMA above long EMA
    if last_ema20 > last_ema50:
        trend_points += 33.34
    
    # ========== COMPONENT 2: EMA DISTANCE ==========
    # Calculate percentage difference between EMAs
    diff = (last_ema20 - last_ema50) / last_ema50 * 100
    
    # Convert to 0-100 scale (centered at 50)
    ema_distance_raw = 50 + (diff / max_diff) * 50
    ema_distance = max(0, min(100, round(ema_distance_raw)))
    
    # ========== COMPOSITE SENTIMENT ==========
    # 50% trend points + 50% EMA distance
    composite = (trend_points * 0.50) + (ema_distance * 0.50)
    sentiment_pct = round(composite)
    
    # Determine sentiment text
    if sentiment_pct >= 70:
        sentiment_text = "Strong Bull"
    elif sentiment_pct >= 55:
        sentiment_text = "Bullish"
    elif sentiment_pct >= 45:
        sentiment_text = "Neutral"
    elif sentiment_pct >= 30:
        sentiment_text = "Bearish"
    else:
        sentiment_text = "Strong Bear"
    
    return sentiment_pct, sentiment_text, last_ema20, last_ema50

def get_market_state(df):
    """
    Determine market state based on EMA positions
    
    SIMPLIFIED LOGIC:
    - ACCUMULATION: Price above ALL EMAs (10, 20, 50)
    - DISTRIBUTION: Price below ALL EMAs (10, 20, 50)
    - RE-ACCUMULATION: Price above 50 but not above all (bullish bias, pullback)
    - RE-DISTRIBUTION: Price below 50 but not below all (bearish bias, rally)
    
    Returns: 'accumulation', 're-accumulation', 'distribution', 're-distribution'
    """
    if df.empty or len(df) < 50:
        return None
    
    # Calculate EMAs
    ema10 = df['Close'].ewm(span=10, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Get latest values
    close = df['Close'].iloc[-1]
    last_ema10 = ema10.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    
    # Determine position relative to each EMA
    above_10 = close > last_ema10
    above_20 = close > last_ema20
    above_50 = close > last_ema50
    
    # ACCUMULATION: Above ALL EMAs = strong uptrend
    if above_10 and above_20 and above_50:
        return "accumulation"
    
    # DISTRIBUTION: Below ALL EMAs = strong downtrend
    if not above_10 and not above_20 and not above_50:
        return "distribution"
    
    # Mixed states - check dominant bias using EMA50
    # If above 50 EMA = bullish bias (even if below some shorter EMAs)
    if above_50:
        return "re-accumulation"
    
    # If below 50 EMA = bearish bias (even if above some shorter EMAs)
    else:
        return "re-distribution"

def fetch_data(symbol, interval, period='5d'):
    """Fetch OHLC data from yfinance with error handling"""
    try:
        ticker = yf.Ticker(symbol)
        
        # Adjust period based on interval
        if interval == '5m':
            period = '5d'
        elif interval == '15m':
            period = '5d'
        elif interval == '1h':
            period = '60d'
        elif interval == '1d':
            period = '1y'
        elif interval == '1wk':
            period = '2y'
        
        # Add a small delay to avoid rate limiting
        time.sleep(0.1)
        
        df = ticker.history(period=period, interval=interval)
        
        # For 4h, resample 1h data
        if interval == '1h':
            df = df.resample('4h').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
        
        return df
    except Exception as e:
        # Don't show error for every failed fetch, just return empty dataframe
        return pd.DataFrame()

def is_bullish(state):
    """Check if state is bullish"""
    return state in ['accumulation', 're-accumulation']

def is_bearish(state):
    """Check if state is bearish"""
    return state in ['distribution', 're-distribution']

def analyze_pair(pair_name, symbol):
    """Analyze a single FX pair across all timeframes"""
    states = {}
    
    # Get state for each timeframe
    for tf_name, tf_interval in TIMEFRAMES.items():
        df = fetch_data(symbol, tf_interval)
        state = get_market_state(df)
        states[tf_name] = state
    
    # Calculate sentiment from 1D timeframe only
    df_daily = fetch_data(symbol, '1d')
    sentiment_pct, sentiment_text, ema20, ema50 = calculate_sentiment(df_daily)
    
    # Count alignments
    bull_count = sum(1 for s in states.values() if s and is_bullish(s))
    bear_count = sum(1 for s in states.values() if s and is_bearish(s))
    
    # Determine signal
    if bull_count == 5:
        signal = "🟢 LONG (5/5)"
        strength = 5
    elif bull_count == 4:
        signal = "🟡 Long (4/5)"
        strength = 4
    elif bear_count == 5:
        signal = "🔴 SHORT (5/5)"
        strength = -5
    elif bear_count == 4:
        signal = "🟠 Short (4/5)"
        strength = -4
    else:
        signal = "⚪ Mixed"
        strength = 0
    
    return {
        'Pair': pair_name,
        'Signal': signal,
        'Strength': strength,
        '5m': states.get('5m', '-'),
        '15m': states.get('15m', '-'),
        '4h': states.get('4h', '-'),
        '1D': states.get('1D', '-'),
        '1W': states.get('1W', '-'),
        'Alignment': f"{max(bull_count, bear_count)}/5",
        'Sentiment': f"{sentiment_pct}%" if sentiment_pct is not None else "-",
        'Sentiment_Text': sentiment_text if sentiment_text else "-",
        'Sentiment_Value': sentiment_pct if sentiment_pct is not None else 0
    }

def get_state_emoji(state):
    """Get emoji representation of state"""
    if state == 'accumulation':
        return '🟩 ACC'
    elif state == 're-accumulation':
        return '🟢 R-ACC'
    elif state == 'distribution':
        return '🟥 DIS'
    elif state == 're-distribution':
        return '🔴 R-DIS'
    else:
        return '⚪ -'

def main():
    st.title("Solar Finance - FX Multi-Timeframe Screener")
    st.markdown("**Real-time market state analysis across 5m, 15m, 4h, 1D, and 1W timeframes**")
    
    # Top navigation bar with buttons
    top_col1, top_col2, top_col3 = st.columns([1, 2, 1])
    
    with top_col1:
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            # How to Use button (list icon)
            if st.button("📋 How to Use", key="help_button"):
                st.session_state.help_visible = not st.session_state.get('help_visible', False)
        
        with button_col2:
            # Settings button (gear icon) - next to How to Use
            if st.button("⚙️ Settings", key="settings_button"):
                st.session_state.settings_visible = not st.session_state.get('settings_visible', False)
    
    with top_col3:
        # Scan button (top right)
        selected_pairs_list = st.session_state.get('selected_markets', [])
        scan_disabled = len(selected_pairs_list) == 0
        refresh_button = st.button("🔄 Scan Markets", type="primary", disabled=scan_disabled, key="scan_top")
        if scan_disabled:
            st.caption("Select markets from sidebar")
    
    # Show help modal if toggled
    if st.session_state.get('help_visible', False):
        with st.expander("FAQ", expanded=True):
            st.markdown("""
            **Usage Tips:**
            - Select Markets: Use the dropdown in sidebar to choose pairs
            - Refresh wisely: Each refresh makes 5+ calls per pair
            - Rate limit: If data stops loading, wait 15-30 minutes
            - Best times: Refresh during market opens (London/NY)
            
            **States:**
            - ACC = Accumulation (Strong Buy)
            - R-ACC = Re-Accumulation (Buy Dip)
            - DIS = Distribution (Strong Sell)
            - R-DIS = Re-Distribution (Sell Rally)
            
            **Signals:**
            - 5/5 = PERFECT alignment - Trade these!
            - 4/5 = Good alignment - Watch closely
            - Mixed = Below 4/5 - Stay out!
            """)
    
    # Show settings modal if toggled
    if st.session_state.get('settings_visible', False):
        with st.expander("Screener Settings", expanded=True):
            st.subheader("Candle Change Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                change_timeframe = st.selectbox(
                    "Timeframe for % Change:",
                    options=['5m', '15m', '4h', '1D', '1W'],
                    index=3,
                    help="Calculate absolute % change over this timeframe",
                    key="change_tf_selector"
                )
            
            with col2:
                change_period = st.number_input(
                    "Number of Candles:",
                    min_value=1,
                    max_value=500,
                    value=30,
                    step=1,
                    help="Calculate % change over this many candles back",
                    key="change_period_input"
                )
            
            st.markdown("---")
            st.subheader("Sort Results By")
            
            sort_by = st.radio(
                "Choose sorting method:",
                options=['Largest Mover', 'Fully Bullish', 'Fully Bearish'],
                index=0,
                help="Largest Mover: Highest % change | Fully Bullish: All timeframes ACC | Fully Bearish: All timeframes DIS",
                key="sort_by_selector"
            )
            
            # Store in different session state keys
            st.session_state.stored_change_timeframe = change_timeframe
            st.session_state.stored_change_period = change_period
            st.session_state.stored_sort_by = sort_by
    else:
        # Set defaults if settings not shown
        if 'stored_change_timeframe' not in st.session_state:
            st.session_state.stored_change_timeframe = '1D'
        if 'stored_change_period' not in st.session_state:
            st.session_state.stored_change_period = 30
        if 'stored_sort_by' not in st.session_state:
            st.session_state.stored_sort_by = 'Largest Mover'
    
    # Sidebar - Market Selection
    st.sidebar.header("Select Markets")
    
    # All available markets (no search)
    all_markets = list(FX_PAIRS.keys())
    filtered_markets = all_markets
    
    # Select All / Deselect All buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("Select All", key="select_all"):
            st.session_state.selected_markets = filtered_markets.copy()
            st.rerun()
    with col2:
        if st.button("Deselect All", key="deselect_all"):
            st.session_state.selected_markets = []
            st.rerun()
    
    # Market selection with multiselect
    if 'selected_markets' not in st.session_state:
        st.session_state.selected_markets = []
    
    selected_markets = st.sidebar.multiselect(
        f"Markets ({len(filtered_markets)} available):",
        options=filtered_markets,
        default=st.session_state.selected_markets,
        key="market_selector"
    )
    
    # Update session state
    st.session_state.selected_markets = selected_markets
    selected_pairs_list = selected_markets
    
    # If no markets selected, show welcome screen
    if len(selected_pairs_list) == 0 and 'results' not in st.session_state:
        st.markdown("""
        ### How to use this screener:
        1. **Select Markets**: Choose pairs/instruments from the sidebar
        2. **Configure Settings**: Click the gear icon to adjust candle analysis settings
        3. **Click Scan**: Press 'Scan Markets' in the top right to analyze your selections
        4. **View Results**: Review multi-timeframe alignment and sentiment
        
        ### Available Markets:
        - **Major FX Pairs**: EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD, USD/CAD, NZD/USD
        - **Minor FX Pairs**: EUR/GBP, EUR/JPY, GBP/JPY, EUR/AUD, EUR/CAD, AUD/JPY, GBP/AUD
        - **Indices**: GER40 (DAX), US100 (Nasdaq)
        - **Metals**: XAUUSD (Gold), XAGUSD (Silver)
        - **Crypto**: BTCUSD (Bitcoin)
        """)
        return
    
    # Main content - Scan logic
    if refresh_button or ('results' not in st.session_state and len(selected_pairs_list) > 0):
        change_timeframe = st.session_state.get('stored_change_timeframe', '1D')
        change_period = st.session_state.get('stored_change_period', 28)
        
        with st.spinner('Analyzing markets... This may take 30-60 seconds.'):
            results = []
            
            # Get list of pairs to analyze
            pairs_to_analyze = [(name, FX_PAIRS[name]) for name in selected_pairs_list]
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (pair_name, symbol) in enumerate(pairs_to_analyze):
                status_text.text(f"Analyzing {pair_name}... ({idx + 1}/{len(pairs_to_analyze)})")
                try:
                    result = analyze_pair(pair_name, symbol)
                    
                    # Add candle change analysis
                    tf_interval = TIMEFRAMES.get(change_timeframe, '1d')
                    df_change = fetch_data(symbol, tf_interval)
                    
                    if not df_change.empty and len(df_change) > change_period:
                        current_price = df_change['Close'].iloc[-1]
                        past_price = df_change['Close'].iloc[-(change_period + 1)]
                        price_change_pct = ((current_price - past_price) / past_price) * 100
                        result[f'Change_{change_timeframe}'] = round(price_change_pct, 2)
                    else:
                        result[f'Change_{change_timeframe}'] = None
                    
                    results.append(result)
                except Exception as e:
                    # Skip pairs that fail
                    st.warning(f"Could not analyze {pair_name} - skipping")
                progress_bar.progress((idx + 1) / len(pairs_to_analyze))
                # Add delay between pairs to avoid rate limiting
                time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.results = results
            st.session_state.stored_change_timeframe = change_timeframe
            st.session_state.stored_change_period = change_period
    
    # Display results
    if 'results' in st.session_state and st.session_state.results:
        df_results = pd.DataFrame(st.session_state.results)
        
        # Get sort preference
        sort_by = st.session_state.get('stored_sort_by', 'Largest Mover')
        stored_tf = st.session_state.get('stored_change_timeframe', '1D')
        change_col_name = f'Change_{stored_tf}'
        
        # Sort based on user preference
        if sort_by == 'Largest Mover':
            # Sort by absolute % change (largest movers first)
            if change_col_name in df_results.columns:
                df_results = df_results.sort_values(change_col_name, key=abs, ascending=False)
            else:
                # Fallback to strength sorting
                df_results = df_results.sort_values('Strength', key=abs, ascending=False)
        
        elif sort_by == 'Fully Bullish':
            # Filter and sort: All timeframes ACC (accumulation)
            def is_fully_bullish(row):
                return all([
                    row['5m'] == 'accumulation',
                    row['15m'] == 'accumulation',
                    row['4h'] == 'accumulation',
                    row['1D'] == 'accumulation',
                    row['1W'] == 'accumulation'
                ])
            
            df_results['is_fully_bullish'] = df_results.apply(is_fully_bullish, axis=1)
            df_results = df_results.sort_values('is_fully_bullish', ascending=False)
            df_results = df_results.drop('is_fully_bullish', axis=1)
        
        elif sort_by == 'Fully Bearish':
            # Filter and sort: All timeframes DIS (distribution)
            def is_fully_bearish(row):
                return all([
                    row['5m'] == 'distribution',
                    row['15m'] == 'distribution',
                    row['4h'] == 'distribution',
                    row['1D'] == 'distribution',
                    row['1W'] == 'distribution'
                ])
            
            df_results['is_fully_bearish'] = df_results.apply(is_fully_bearish, axis=1)
            df_results = df_results.sort_values('is_fully_bearish', ascending=False)
            df_results = df_results.drop('is_fully_bearish', axis=1)
        
        # Highlight strong signals
        st.subheader("Market Overview")
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        perfect_longs = len([r for r in st.session_state.results if r['Strength'] == 5])
        perfect_shorts = len([r for r in st.session_state.results if r['Strength'] == -5])
        watch_list = len([r for r in st.session_state.results if abs(r['Strength']) == 4])
        mixed = len([r for r in st.session_state.results if abs(r['Strength']) < 4])
        
        with col1:
            st.metric("Perfect Longs (5/5)", perfect_longs)
        with col2:
            st.metric("Perfect Shorts (5/5)", perfect_shorts)
        with col3:
            st.metric("Watch List (4/5)", watch_list)
        with col4:
            st.metric("Mixed (<4/5)", mixed)
        
        st.markdown("---")
        
        # Display detailed table
        st.subheader("Detailed Analysis")
        
        # Format the dataframe for display
        display_df = df_results.copy()
        display_df['5m'] = display_df['5m'].apply(get_state_emoji)
        display_df['15m'] = display_df['15m'].apply(get_state_emoji)
        display_df['4h'] = display_df['4h'].apply(get_state_emoji)
        display_df['1D'] = display_df['1D'].apply(get_state_emoji)
        display_df['1W'] = display_df['1W'].apply(get_state_emoji)
        
        # Format sentiment column with emoji
        def format_sentiment(row):
            sent_text = row['Sentiment_Text']
            sent_pct = row['Sentiment']
            if sent_text == "Strong Bull":
                return f"🟢🟢 {sent_pct}"
            elif sent_text == "Bullish":
                return f"🟢 {sent_pct}"
            elif sent_text == "Neutral":
                return f"⚪ {sent_pct}"
            elif sent_text == "Bearish":
                return f"🔴 {sent_pct}"
            elif sent_text == "Strong Bear":
                return f"🔴🔴 {sent_pct}"
            else:
                return f"⚪ {sent_pct}"
        
        display_df['Sentiment_Display'] = display_df.apply(format_sentiment, axis=1)
        
        # Format candle change column with color indicators
        stored_tf = st.session_state.get('stored_change_timeframe', '1D')
        stored_period = st.session_state.get('stored_change_period', 28)
        change_col_name = f'Change_{stored_tf}'
        
        if change_col_name in display_df.columns:
            def format_change(val):
                if val is None or pd.isna(val):
                    return "-"
                elif val > 0:
                    return f"🟢 +{val}%"
                elif val < 0:
                    return f"🔴 {val}%"
                else:
                    return f"⚪ {val}%"
            
            display_df[f'Change_Display'] = display_df[change_col_name].apply(format_change)
            # Drop raw change column
            display_df = display_df.drop([change_col_name], axis=1)
            # Rename for display
            change_label = f'Δ% ({stored_tf}, {stored_period}c)'
            display_df = display_df.rename(columns={'Change_Display': change_label})
        
        # Drop internal columns
        display_df = display_df.drop(['Strength', 'Sentiment', 'Sentiment_Text', 'Sentiment_Value'], axis=1)
        
        # Rename for display
        display_df = display_df.rename(columns={'Sentiment_Display': 'Sentiment'})
        
        # Reorder columns to put change column after Sentiment
        cols = display_df.columns.tolist()
        # Find sentiment column position
        sent_idx = cols.index('Sentiment')
        # Find change column (it will have the Δ% prefix)
        change_cols = [c for c in cols if c.startswith('Δ%')]
        if change_cols:
            change_col = change_cols[0]
            # Remove change column from current position
            cols.remove(change_col)
            # Insert after sentiment
            cols.insert(sent_idx + 1, change_col)
            display_df = display_df[cols]
        
        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Show timestamp and analysis info
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Candle Analysis: {stored_tf} timeframe, {stored_period} candles back")
        
        # Trade recommendations
        st.markdown("---")
        st.subheader("Trade Recommendations")
        
        # Only show 5/5 perfect alignments
        perfect_signals = [r for r in st.session_state.results if abs(r['Strength']) == 5]
        
        if perfect_signals:
            st.success("PERFECT SETUPS DETECTED - All 5 timeframes aligned!")
            for result in perfect_signals:
                if result['Strength'] > 0:
                    st.success(f"**{result['Pair']}** - {result['Signal']} | Look for LONG entries on pullbacks to support")
                else:
                    st.error(f"**{result['Pair']}** - {result['Signal']} | Look for SHORT entries on rallies to resistance")
        else:
            st.info("No perfect 5/5 alignment setups at the moment. Wait for all timeframes to align before trading.")
            
            # Show 4/5 as "watch list"
            good_signals = [r for r in st.session_state.results if abs(r['Strength']) == 4]
            if good_signals:
                st.markdown("---")
                st.markdown("**Watch List (4/5 alignment - close but not perfect):**")
                for result in good_signals:
                    if result['Strength'] > 0:
                        st.markdown(f"- {result['Pair']} - {result['Signal']} - Watch for 5th timeframe to align bullish")
                    else:
                        st.markdown(f"- {result['Pair']} - {result['Signal']} - Watch for 5th timeframe to align bearish")

if __name__ == "__main__":
    main()
