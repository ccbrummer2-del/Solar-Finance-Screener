"""
Core market analysis logic - Hybrid version with customizable timeframes
"""
import pandas as pd
from datetime import datetime

# Available timeframe options (user can select which ones to use)
AVAILABLE_TIMEFRAMES = {
    '1m': '1m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1h',
    '4h': '1h',  # Will be resampled
    '1D': '1d',
    '1W': '1wk'
}

FX_PAIRS = {
    # Major USD Pairs
    'EUR/USD': 'EURUSD=X',
    'GBP/USD': 'GBPUSD=X',
    'USD/JPY': 'USDJPY=X',
    'USD/CHF': 'USDCHF=X',
    'AUD/USD': 'AUDUSD=X',
    'USD/CAD': 'USDCAD=X',
    'NZD/USD': 'NZDUSD=X',
    
    # EUR Crosses
    'EUR/GBP': 'EURGBP=X',
    'EUR/JPY': 'EURJPY=X',
    'EUR/CHF': 'EURCHF=X',
    'EUR/AUD': 'EURAUD=X',
    'EUR/CAD': 'EURCAD=X',
    'EUR/NZD': 'EURNZD=X',
    
    # GBP Crosses
    'GBP/JPY': 'GBPJPY=X',
    'GBP/CHF': 'GBPCHF=X',
    'GBP/AUD': 'GBPAUD=X',
    'GBP/CAD': 'GBPCAD=X',
    
    # JPY Crosses
    'AUD/JPY': 'AUDJPY=X',
    'NZD/JPY': 'NZDJPY=X',
    'CAD/JPY': 'CADJPY=X',
    'CHF/JPY': 'CHFJPY=X',
    
    # AUD & NZD Crosses
    'AUD/NZD': 'AUDNZD=X',
    'AUD/CHF': 'AUDCHF=X',
    'AUD/CAD': 'AUDCAD=X',
    'NZD/CHF': 'NZDCHF=X',
    'NZD/CAD': 'NZDCAD=X',
    
    # Indices
    'GER40': '^GDAXI',
    'US100': 'NQ=F',
    'US500': 'ES=F',
    'US30': 'YM=F',
    'US2000': 'RTY=F',
    
    # Commodities
    'XAUUSD': 'GC=F',
    'XAGUSD': 'SI=F',
    'OIL': 'CL=F',
    
    # Crypto
    'BTCUSD': 'BTC-USD',
    
    # US100 Stocks - Tech
    'AAPL': 'AAPL',
    'MSFT': 'MSFT',
    'NVDA': 'NVDA',
    'GOOGL': 'GOOGL',
    'AMZN': 'AMZN',
    'META': 'META',
    'TSLA': 'TSLA',
    'AVGO': 'AVGO',
    'ORCL': 'ORCL',
    'ADBE': 'ADBE',
    'CRM': 'CRM',
    'NFLX': 'NFLX',
    'CSCO': 'CSCO',
    'INTC': 'INTC',
    'AMD': 'AMD',
    'QCOM': 'QCOM',
    'TXN': 'TXN',
    'INTU': 'INTU',
    'AMAT': 'AMAT',
    'ADI': 'ADI',
    'MU': 'MU',
    'LRCX': 'LRCX',
    'KLAC': 'KLAC',
    'SNPS': 'SNPS',
    'CDNS': 'CDNS',
    'MRVL': 'MRVL',
    'FTNT': 'FTNT',
    'PANW': 'PANW',
    'CRWD': 'CRWD',
    'TEAM': 'TEAM',
    
    # US100 Stocks - Consumer
    'COST': 'COST',
    'PEP': 'PEP',
    'SBUX': 'SBUX',
    'MDLZ': 'MDLZ',
    'MAR': 'MAR',
    'ABNB': 'ABNB',
    'BKNG': 'BKNG',
    'LULU': 'LULU',
    'ROST': 'ROST',
    'MELI': 'MELI',
    'DASH': 'DASH',
    'CPRT': 'CPRT',
    'ODFL': 'ODFL',
    
    # US100 Stocks - Healthcare/Biotech
    'AMGN': 'AMGN',
    'GILD': 'GILD',
    'VRTX': 'VRTX',
    'REGN': 'REGN',
    'BIIB': 'BIIB',
    'MRNA': 'MRNA',
    'ILMN': 'ILMN',
    'DXCM': 'DXCM',
    
    # US100 Stocks - Industrial/Energy
    'HON': 'HON',
    'ADP': 'ADP',
    'PCAR': 'PCAR',
    'PAYX': 'PAYX',
    'FAST': 'FAST',
    'CTAS': 'CTAS',
    'VRSK': 'VRSK',
    
    # US100 Stocks - Communication/Media
    'CMCSA': 'CMCSA',
    'NXPI': 'NXPI',
    'MCHP': 'MCHP',
    
    # US100 Stocks - Financial
    'PYPL': 'PYPL',
    'ADP': 'ADP',
    'PAYX': 'PAYX',
    
    # US100 Stocks - Other
    'WDAY': 'WDAY',
    'MNST': 'MNST',
    'EA': 'EA',
    'XEL': 'XEL',
    'CEG': 'CEG',
    'ANSS': 'ANSS',
    'DDOG': 'DDOG',
    'ZS': 'ZS',
    'TTWO': 'TTWO',
    'IDXX': 'IDXX',
    'CSGP': 'CSGP',
    'WBD': 'WBD',
    'GEHC': 'GEHC',
    'ON': 'ON',
    'FANG': 'FANG',
    'BKR': 'BKR',
    'DLTR': 'DLTR',
    'KDP': 'KDP',
    'SIRI': 'SIRI',
    'ZM': 'ZM',
}

# Market Categories for filtering
MARKET_CATEGORIES = {
    'FX Majors': [
        'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 
        'AUD/USD', 'USD/CAD', 'NZD/USD'
    ],
    'FX Minors': [
        'EUR/GBP', 'EUR/JPY', 'EUR/CHF', 'EUR/AUD', 'EUR/CAD', 'EUR/NZD',
        'GBP/JPY', 'GBP/CHF', 'GBP/AUD', 'GBP/CAD',
        'AUD/JPY', 'NZD/JPY', 'CAD/JPY', 'CHF/JPY',
        'AUD/NZD', 'AUD/CHF', 'AUD/CAD', 'NZD/CHF', 'NZD/CAD'
    ],
    'Stocks': [
        'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'AVGO', 
        'ORCL', 'ADBE', 'CRM', 'NFLX', 'CSCO', 'INTC', 'AMD', 'QCOM', 
        'TXN', 'INTU', 'AMAT', 'ADI', 'MU', 'LRCX', 'KLAC', 'SNPS', 
        'CDNS', 'MRVL', 'FTNT', 'PANW', 'CRWD', 'TEAM', 'COST', 'PEP', 
        'SBUX', 'MDLZ', 'MAR', 'ABNB', 'BKNG', 'LULU', 'ROST', 'MELI', 
        'DASH', 'CPRT', 'ODFL', 'AMGN', 'GILD', 'VRTX', 'REGN', 'BIIB', 
        'MRNA', 'ILMN', 'DXCM', 'HON', 'ADP', 'PCAR', 'PAYX', 'FAST', 
        'CTAS', 'VRSK', 'CMCSA', 'NXPI', 'MCHP', 'PYPL', 'WDAY', 'MNST', 
        'EA', 'XEL', 'CEG', 'ANSS', 'DDOG', 'ZS', 'TTWO', 'IDXX', 'CSGP', 
        'WBD', 'GEHC', 'ON', 'FANG', 'BKR', 'DLTR', 'KDP', 'SIRI', 'ZM'
    ],
    'Commodities/CFDs': [
        'GER40', 'US100', 'US500', 'US30', 'US2000',
        'XAUUSD', 'XAGUSD', 'OIL'
    ],
    'Cryptos': [
        'BTCUSD'
    ]
}

def calculate_sentiment(df, max_diff=10.0):
    """
    Multi-Indicator Sentiment based on 20/50 EMA (HARDCODED - NOT CUSTOMIZABLE)
    Combines Trend Points (50%) and EMA Distance (50%) for composite score
    Returns: sentiment_pct (0-100), sentiment_text, ema20, ema50
    """
    if df.empty or len(df) < 50:
        return None, None, None, None
    
    # Calculate 20 and 50 EMAs
    ema20 = df['Close'].ewm(span=20, adjust=False).mean()
    ema50 = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Get latest values
    last_close = df['Close'].iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]
    
    # COMPONENT 1: TREND POINTS
    trend_points = 0.0
    
    if last_close > last_ema20:
        trend_points += 33.33
    
    if last_close > last_ema50:
        trend_points += 33.33
    
    if last_ema20 > last_ema50:
        trend_points += 33.34
    
    # COMPONENT 2: EMA DISTANCE
    diff = (last_ema20 - last_ema50) / last_ema50 * 100
    ema_distance_raw = 50 + (diff / max_diff) * 50
    ema_distance = max(0, min(100, round(ema_distance_raw)))
    
    # COMPOSITE SENTIMENT
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
    Determine market state based on EMA positions (HARDCODED - NOT CUSTOMIZABLE)
    Uses EMA 10/20/50
    Returns: 'accumulation', 're-accumulation', 'distribution', 're-distribution'
    """
    if df.empty or len(df) < 50:
        return None
    
    # Calculate EMAs (HARDCODED PERIODS)
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
    
    # ACCUMULATION: Above ALL EMAs
    if above_10 and above_20 and above_50:
        return "accumulation"
    
    # DISTRIBUTION: Below ALL EMAs
    if not above_10 and not above_20 and not above_50:
        return "distribution"
    
    # Mixed states
    if above_50:
        return "re-accumulation"
    else:
        return "re-distribution"

class MarketAnalyzer:
    """Main analyzer class - Hybrid version"""
    
    def analyze_pair(self, pair_name, symbol, data_fetcher, settings=None):
        """
        Analyze a single pair across user-selected timeframes
        
        Args:
            pair_name: Display name of the pair
            symbol: yfinance symbol
            data_fetcher: DataFetcher instance
            settings: Dict with:
                - timeframes: List of selected timeframes (e.g., ['5m', '1D', '1W'])
                - sentiment_enabled: bool
                - sentiment_timeframe: str (e.g., '1D')
                - lookbacks: List of lookback configs
                - signal_thresholds: Dict with bullish/bearish thresholds
        """
        if settings is None:
            settings = {}
        
        # Get selected timeframes (default to old 5 TF system if not specified)
        selected_tfs = settings.get('timeframes', ['5m', '15m', '4h', '1D', '1W'])
        
        states = {}
        
        # Get state for each selected timeframe
        for tf_name in selected_tfs:
            tf_interval = AVAILABLE_TIMEFRAMES.get(tf_name, '1d')
            df = data_fetcher.fetch_data(symbol, tf_interval)
            state = get_market_state(df)
            states[tf_name] = state
        
        # Calculate sentiment if enabled
        sentiment_pct = None
        sentiment_text = None
        ema20 = None
        ema50 = None
        
        if settings.get('sentiment_enabled', True):
            sentiment_tf = settings.get('sentiment_timeframe', '1D')
            sentiment_interval = AVAILABLE_TIMEFRAMES.get(sentiment_tf, '1d')
            df_sentiment = data_fetcher.fetch_data(symbol, sentiment_interval)
            sentiment_pct, sentiment_text, ema20, ema50 = calculate_sentiment(df_sentiment)
        
        # Count alignments
        bull_count = sum(1 for s in states.values() if s in ['accumulation', 're-accumulation'])
        bear_count = sum(1 for s in states.values() if s in ['distribution', 're-distribution'])
        
        # Get signal thresholds
        total_tfs = len(selected_tfs)
        bullish_threshold = settings.get('signal_thresholds', {}).get('bullish', total_tfs)
        bearish_threshold = settings.get('signal_thresholds', {}).get('bearish', total_tfs)
        
        # Determine signal based on thresholds
        if bull_count >= bullish_threshold:
            signal = f"🟢 LONG ({bull_count}/{total_tfs})"
            strength = bull_count
        elif bear_count >= bearish_threshold:
            signal = f"🔴 SHORT ({bear_count}/{total_tfs})"
            strength = -bear_count
        else:
            signal = "⚪ Mixed"
            strength = 0
        
        result = {
            'Pair': pair_name,
            'Signal': signal,
            'Strength': strength,
            'Sentiment': f"{sentiment_pct}%" if sentiment_pct is not None else "-",
            'Sentiment_Text': sentiment_text if sentiment_text else "-",
            'Sentiment_Value': sentiment_pct if sentiment_pct is not None else 0
        }
        
        # Add individual timeframe states to result
        for tf in selected_tfs:
            result[tf] = states.get(tf, '-')
        
        # Add lookback analysis if configured
        lookbacks = settings.get('lookbacks', [])
        for idx, lookback in enumerate(lookbacks):
            if not lookback.get('enabled', False):
                continue
            
            lb_tf = lookback.get('timeframe', '1D')
            lb_period = lookback.get('periods', 30)
            lb_interval = AVAILABLE_TIMEFRAMES.get(lb_tf, '1d')
            
            df_lookback = data_fetcher.fetch_data(symbol, lb_interval)
            
            if not df_lookback.empty and len(df_lookback) > lb_period:
                current = df_lookback['Close'].iloc[-1]
                past = df_lookback['Close'].iloc[-(lb_period + 1)]
                change_pct = ((current - past) / past) * 100
                result[f'Lookback{idx+1}'] = round(change_pct, 2)
                result[f'Lookback{idx+1}_Label'] = lookback.get('label', f'LB{idx+1}')
            else:
                result[f'Lookback{idx+1}'] = None
                result[f'Lookback{idx+1}_Label'] = lookback.get('label', f'LB{idx+1}')
        
        return result
