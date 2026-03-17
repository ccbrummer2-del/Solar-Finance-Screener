"""
Data fetching logic using yfinance - Updated for hybrid system
"""
import yfinance as yf
import pandas as pd
import time

class DataFetcher:
    """Handles all data fetching with caching"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 60  # Cache data for 60 seconds
        self.last_fetch_time = {}
    
    def fetch_data(self, symbol, interval, period='5d'):
        """Fetch OHLC data from yfinance with caching"""
        cache_key = f"{symbol}_{interval}"
        current_time = time.time()
        
        # Check if we have cached data that's still fresh
        if cache_key in self.cache:
            last_time = self.last_fetch_time.get(cache_key, 0)
            if current_time - last_time < self.cache_timeout:
                return self.cache[cache_key]
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Adjust period based on interval
            if interval == '1m':
                period = '1d'
            elif interval == '5m':
                period = '5d'
            elif interval == '15m':
                period = '5d'
            elif interval == '30m':
                period = '5d'
            elif interval == '1h':
                period = '60d'
            elif interval == '1d':
                period = '1y'
            elif interval == '1wk':
                period = '2y'
            
            # Small delay to avoid rate limiting
            time.sleep(0.1)
            
            df = ticker.history(period=period, interval=interval)
            
            # Cache the result
            self.cache[cache_key] = df
            self.last_fetch_time[cache_key] = current_time
            
            return df
            
        except Exception as e:
            print(f"Error fetching {symbol} ({interval}): {e}")
            # Return cached data if available, even if stale
            return self.cache.get(cache_key, pd.DataFrame())
    
    def clear_cache(self):
        """Clear the cache"""
        self.cache = {}
        self.last_fetch_time = {}
