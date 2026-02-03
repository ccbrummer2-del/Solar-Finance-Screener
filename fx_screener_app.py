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

def get_market_state(df):
    """
    Determine market state based on EMA positions using state transitions
    This implements the EXACT Pine Script logic with state persistence
    Returns: 'accumulation', 're-accumulation', 'distribution', 're-distribution'
    """
    if df.empty or len(df) < 50:
        return None
    
    # Calculate EMAs for entire dataframe
    ema10 = df['Close'].ewm(span=10, adjust=False).mean()
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Start with accumulation state
    state = "accumulation"
    
    # Iterate through each bar to track state transitions (like Pine Script's var)
    for i in range(len(df)):
        close = df['Close'].iloc[i]
        e10 = ema10.iloc[i]
        e20 = ema20.iloc[i]
        e50 = ema50.iloc[i]
        
        # State transition logic - EXACT Pine Script implementation
        if state == "accumulation":
            # In accumulation, if price drops to/below 10 EMA → re-acc
            if close <= e10:
                state = "re-accumulation"
        
        elif state == "re-accumulation":
            # In re-acc, if price flushes below 50 EMA → distribution
            if close < e50:
                state = "distribution"
            # If price recovers above 10 EMA → back to accumulation
            elif close > e10:
                state = "accumulation"
        
        elif state == "distribution":
            # In distribution, if price rises to/above 20 EMA → re-dis
            if close >= e20:
                state = "re-distribution"
        
        elif state == "re-distribution":
            # In re-dis, if price flushes above 50 EMA → accumulation
            if close > e50:
                state = "accumulation"
            # If price falls back below 20 EMA → back to distribution
            elif close < e20:
                state = "distribution"
    
    # Return the final state after processing all bars
    return state

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
        'Alignment': f"{max(bull_count, bear_count)}/5"
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
    st.title("🌞 Solar Finance - FX Multi-Timeframe Screener")
    st.markdown("**Real-time market state analysis across 5m, 15m, 4h, 1D, and 1W timeframes**")
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # Pair selection
    st.sidebar.subheader("📊 Major Pairs")
    selected_pairs = {}
    selected_pairs['EUR/USD'] = st.sidebar.checkbox('EUR/USD', value=True)
    selected_pairs['GBP/USD'] = st.sidebar.checkbox('GBP/USD', value=True)
    selected_pairs['USD/JPY'] = st.sidebar.checkbox('USD/JPY', value=True)
    selected_pairs['USD/CHF'] = st.sidebar.checkbox('USD/CHF', value=True)
    selected_pairs['AUD/USD'] = st.sidebar.checkbox('AUD/USD', value=True)
    selected_pairs['USD/CAD'] = st.sidebar.checkbox('USD/CAD', value=True)
    selected_pairs['NZD/USD'] = st.sidebar.checkbox('NZD/USD', value=True)
    
    st.sidebar.subheader("📈 Minor Pairs")
    selected_pairs['EUR/GBP'] = st.sidebar.checkbox('EUR/GBP', value=False)
    selected_pairs['EUR/JPY'] = st.sidebar.checkbox('EUR/JPY', value=False)
    selected_pairs['GBP/JPY'] = st.sidebar.checkbox('GBP/JPY', value=False)
    selected_pairs['EUR/AUD'] = st.sidebar.checkbox('EUR/AUD', value=False)
    selected_pairs['EUR/CAD'] = st.sidebar.checkbox('EUR/CAD', value=False)
    selected_pairs['AUD/JPY'] = st.sidebar.checkbox('AUD/JPY', value=False)
    selected_pairs['GBP/AUD'] = st.sidebar.checkbox('GBP/AUD', value=False)
    
    st.sidebar.subheader("💹 Indices")
    selected_pairs['GER40'] = st.sidebar.checkbox('GER40 (DAX)', value=False)
    selected_pairs['US100'] = st.sidebar.checkbox('US100 (Nasdaq)', value=False)
    
    st.sidebar.subheader("🥇 Metals")
    selected_pairs['XAUUSD'] = st.sidebar.checkbox('XAUUSD (Gold)', value=False)
    selected_pairs['XAGUSD'] = st.sidebar.checkbox('XAGUSD (Silver)', value=False)
    
    st.sidebar.subheader("₿ Crypto")
    selected_pairs['BTCUSD'] = st.sidebar.checkbox('BTCUSD (Bitcoin)', value=False)
    
    # Auto-refresh
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    refresh_button = st.sidebar.button("🔄 Refresh Now")
    
    # Rate limit warning
    if auto_refresh:
        st.sidebar.warning("⚠️ Auto-refresh enabled. Too many refreshes may trigger rate limits. Use sparingly!")
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Usage Tips")
    st.sidebar.markdown("""
    - **Refresh wisely**: Each refresh makes 5 calls per pair
    - **Select fewer pairs**: Reduce load by unchecking pairs you don't trade
    - **Rate limit**: If data stops loading, wait 15-30 minutes
    - **Best times**: Refresh during market opens (London/NY)
    """)
    
    # Info section
    st.sidebar.markdown("---")
    st.sidebar.subheader("Legend")
    st.sidebar.markdown("""
    **States:**
    - 🟩 ACC = Accumulation (Strong Buy)
    - 🟢 R-ACC = Re-Accumulation (Buy Dip)
    - 🟥 DIS = Distribution (Strong Sell)
    - 🔴 R-DIS = Re-Distribution (Sell Rally)
    
    **Signals:**
    - 🟢 5/5 = PERFECT alignment - Trade these!
    - 🟡 4/5 = Good alignment - Watch closely
    - ⚪ Mixed = Below 4/5 - Stay out!
    """)
    
    # Main content
    if refresh_button or auto_refresh or 'results' not in st.session_state:
        with st.spinner('Analyzing markets... This may take 30-60 seconds.'):
            results = []
            
            # Get list of pairs to analyze
            pairs_to_analyze = [(name, symbol) for name, symbol in FX_PAIRS.items() if selected_pairs.get(name, False)]
            
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (pair_name, symbol) in enumerate(pairs_to_analyze):
                status_text.text(f"Analyzing {pair_name}... ({idx + 1}/{len(pairs_to_analyze)})")
                try:
                    result = analyze_pair(pair_name, symbol)
                    results.append(result)
                except Exception as e:
                    # Skip pairs that fail
                    st.warning(f"⚠️ Could not analyze {pair_name} - skipping")
                progress_bar.progress((idx + 1) / len(pairs_to_analyze))
                # Add delay between pairs to avoid rate limiting
                time.sleep(0.5)
            
            progress_bar.empty()
            status_text.empty()
            st.session_state.results = results
    
    # Display results
    if 'results' in st.session_state and st.session_state.results:
        df_results = pd.DataFrame(st.session_state.results)
        
        # Sort by strength (strongest signals first)
        df_results = df_results.sort_values('Strength', key=abs, ascending=False)
        
        # Highlight strong signals
        st.subheader("📊 Market Overview")
        
        # Show summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        perfect_longs = len([r for r in st.session_state.results if r['Strength'] == 5])
        perfect_shorts = len([r for r in st.session_state.results if r['Strength'] == -5])
        watch_list = len([r for r in st.session_state.results if abs(r['Strength']) == 4])
        mixed = len([r for r in st.session_state.results if abs(r['Strength']) < 4])
        
        with col1:
            st.metric("🟢 Perfect Longs (5/5)", perfect_longs)
        with col2:
            st.metric("🔴 Perfect Shorts (5/5)", perfect_shorts)
        with col3:
            st.metric("🟡 Watch List (4/5)", watch_list)
        with col4:
            st.metric("⚪ Mixed (<4/5)", mixed)
        
        st.markdown("---")
        
        # Display detailed table
        st.subheader("📈 Detailed Analysis")
        
        # Format the dataframe for display
        display_df = df_results.copy()
        display_df['5m'] = display_df['5m'].apply(get_state_emoji)
        display_df['15m'] = display_df['15m'].apply(get_state_emoji)
        display_df['4h'] = display_df['4h'].apply(get_state_emoji)
        display_df['1D'] = display_df['1D'].apply(get_state_emoji)
        display_df['1W'] = display_df['1W'].apply(get_state_emoji)
        
        # Drop strength column (internal use only)
        display_df = display_df.drop('Strength', axis=1)
        
        # Style the dataframe
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            height=400
        )
        
        # Show timestamp
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Trade recommendations
        st.markdown("---")
        st.subheader("💡 Trade Recommendations")
        
        # Only show 5/5 perfect alignments
        perfect_signals = [r for r in st.session_state.results if abs(r['Strength']) == 5]
        
        if perfect_signals:
            st.success("🎯 **PERFECT SETUPS DETECTED** - All 5 timeframes aligned!")
            for result in perfect_signals:
                if result['Strength'] > 0:
                    st.success(f"**{result['Pair']}** - {result['Signal']} | 🟢 Look for LONG entries on pullbacks to support")
                else:
                    st.error(f"**{result['Pair']}** - {result['Signal']} | 🔴 Look for SHORT entries on rallies to resistance")
        else:
            st.info("⏳ No perfect 5/5 alignment setups at the moment. Wait for all timeframes to align before trading.")
            
            # Show 4/5 as "watch list"
            good_signals = [r for r in st.session_state.results if abs(r['Strength']) == 4]
            if good_signals:
                st.markdown("---")
                st.markdown("**📋 Watch List (4/5 alignment - close but not perfect):**")
                for result in good_signals:
                    if result['Strength'] > 0:
                        st.markdown(f"- {result['Pair']} - {result['Signal']} - Watch for 5th timeframe to align bullish")
                    else:
                        st.markdown(f"- {result['Pair']} - {result['Signal']} - Watch for 5th timeframe to align bearish")
    
    # Auto-refresh logic
    if auto_refresh:
        time.sleep(30)
        st.rerun()

if __name__ == "__main__":
    main()
    
    # === CANDLE LOOKBACK ANALYZER ===
    st.markdown("---")
    st.header("📊 Candle Lookback Analyzer")
    st.markdown("**Analyze price movement over a specific number of candles**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        lookback_pair = st.selectbox(
            "Select Pair/Instrument:",
            options=list(FX_PAIRS.keys()),
            key="lookback_pair"
        )
    
    with col2:
        lookback_timeframe = st.selectbox(
            "Select Timeframe:",
            options=['5m', '15m', '4h', '1D', '1W'],
            key="lookback_tf"
        )
    
    with col3:
        lookback_candles = st.number_input(
            "Number of Candles Back:",
            min_value=1,
            max_value=500,
            value=10,
            step=1,
            key="lookback_candles"
        )
    
    analyze_button = st.button("🔍 Analyze Movement", key="analyze_lookback")
    
    if analyze_button:
        with st.spinner(f'Analyzing {lookback_pair} on {lookback_timeframe}...'):
            try:
                # Get the yfinance symbol
                symbol = FX_PAIRS[lookback_pair]
                
                # Map timeframe to yfinance interval
                tf_map = {
                    '5m': '5m',
                    '15m': '15m',
                    '4h': '1h',
                    '1D': '1d',
                    '1W': '1wk'
                }
                interval = tf_map[lookback_timeframe]
                
                # Fetch data
                df = fetch_data(symbol, interval)
                
                if df.empty or len(df) < lookback_candles:
                    st.error(f"❌ Not enough data. Only {len(df)} candles available.")
                else:
                    # Calculate price change
                    current_price = df['Close'].iloc[-1]
                    past_price = df['Close'].iloc[-(lookback_candles + 1)]
                    
                    price_change = current_price - past_price
                    price_change_pct = (price_change / past_price) * 100
                    
                    # Get high and low during period
                    period_high = df['High'].iloc[-lookback_candles:].max()
                    period_low = df['Low'].iloc[-lookback_candles:].min()
                    range_pct = ((period_high - period_low) / period_low) * 100
                    
                    # Display results in cards
                    st.success(f"✅ Analysis Complete for {lookback_pair}")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            label="Price Change",
                            value=f"{price_change_pct:.2f}%",
                            delta=f"{price_change:.5f}" if abs(price_change) < 1 else f"{price_change:.2f}"
                        )
                    
                    with col2:
                        st.metric(
                            label="Start Price",
                            value=f"{past_price:.5f}" if past_price < 10 else f"{past_price:.2f}"
                        )
                    
                    with col3:
                        st.metric(
                            label="Current Price",
                            value=f"{current_price:.5f}" if current_price < 10 else f"{current_price:.2f}"
                        )
                    
                    with col4:
                        st.metric(
                            label="Range (H-L)",
                            value=f"{range_pct:.2f}%"
                        )
                    
                    # Additional details
                    st.markdown("---")
                    st.subheader("📈 Detailed Movement")
                    
                    detail_col1, detail_col2 = st.columns(2)
                    
                    with detail_col1:
                        # Format prices properly
                        high_str = f"{period_high:.5f}" if period_high < 10 else f"{period_high:.2f}"
                        low_str = f"{period_low:.5f}" if period_low < 10 else f"{period_low:.2f}"
                        
                        st.markdown(f"""
                        **Period Analysis:**
                        - **Timeframe:** {lookback_timeframe}
                        - **Candles Analyzed:** {lookback_candles}
                        - **Period High:** {high_str}
                        - **Period Low:** {low_str}
                        - **Total Range:** {range_pct:.2f}%
                        """)
                    
                    with detail_col2:
                        # Determine trend
                        if price_change_pct > 0.5:
                            trend = "🟢 Bullish"
                            trend_desc = "Price is moving upward"
                        elif price_change_pct < -0.5:
                            trend = "🔴 Bearish"
                            trend_desc = "Price is moving downward"
                        else:
                            trend = "⚪ Neutral"
                            trend_desc = "Price is range-bound"
                        
                        # Where is current price in the range?
                        price_position = ((current_price - period_low) / (period_high - period_low)) * 100 if period_high != period_low else 50
                        
                        st.markdown(f"""
                        **Trend Assessment:**
                        - **Direction:** {trend}
                        - **Interpretation:** {trend_desc}
                        - **Price Position:** {price_position:.1f}% of range
                        - **From Low:** {((current_price - period_low) / period_low * 100):.2f}%
                        - **From High:** {((current_price - period_high) / period_high * 100):.2f}%
                        """)
                    
                    # Visual indicator
                    st.markdown("---")
                    st.markdown("**Price Position in Range:**")
                    
                    # Create a simple progress bar to show position
                    position_normalized = price_position / 100
                    st.progress(position_normalized)
                    
                    if price_position > 80:
                        st.info("💡 Price near range high - potential resistance")
                    elif price_position < 20:
                        st.info("💡 Price near range low - potential support")
                    else:
                        st.info("💡 Price in middle of range")
                    
            except Exception as e:
                st.error(f"❌ Error analyzing {lookback_pair}: {str(e)}")
