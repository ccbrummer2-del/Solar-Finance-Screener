"""
Background Scanner Service for Solar Terminal
Runs 24/7, scanning markets every 5 minutes and saving to database
"""
import time
import schedule
import sqlite3
from datetime import datetime
import sys
import os

# Add parent directory to path so we can import core modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.market_analyzer import MarketAnalyzer, FX_PAIRS
from core.data_fetcher import DataFetcher

class ScannerService:
    def __init__(self):
        self.analyzer = MarketAnalyzer()
        self.data_fetcher = DataFetcher()
        self.db_path = 'live_scans.db'
        self.init_db()
    
    def init_db(self):
        """Create database for storing scan results"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                signal TEXT,
                strength INTEGER,
                state_5m TEXT,
                state_15m TEXT,
                state_4h TEXT,
                state_1d TEXT,
                state_1w TEXT,
                sentiment TEXT,
                sentiment_value INTEGER,
                lookback1 REAL,
                lookback2 REAL,
                lookback3 REAL,
                lookback1_label TEXT,
                lookback2_label TEXT,
                lookback3_label TEXT,
                UNIQUE(timestamp, pair)
            )
        ''')
        conn.commit()
        conn.close()
        print(f"✅ Database initialized: {self.db_path}")
    
    def run_scan(self):
        """Run a complete market scan"""
        print(f"\n{'='*60}")
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting scan...")
        print(f"{'='*60}")
        
        results = []
        
        # Default settings (matches Solar Terminal defaults)
        settings = {
            'timeframes': ['5m', '15m', '4h', '1D', '1W'],
            'sentiment_enabled': True,
            'sentiment_timeframe': '1D',
            'lookbacks': [
                {'enabled': True, 'timeframe': '1D', 'periods': 30, 'label': 'Monthly'},
                {'enabled': True, 'timeframe': '1D', 'periods': 7, 'label': 'Weekly'},
                {'enabled': False, 'timeframe': '4h', 'periods': 20, 'label': 'Custom'}
            ],
            'signal_thresholds': {
                'bullish': 5,
                'bearish': 5
            }
        }
        
        # Scan all pairs
        total = len(FX_PAIRS)
        for idx, (pair_name, symbol) in enumerate(FX_PAIRS.items(), 1):
            try:
                print(f"  [{idx}/{total}] Analyzing {pair_name}...", end='')
                result = self.analyzer.analyze_pair(
                    pair_name, symbol, self.data_fetcher, settings
                )
                results.append(result)
                print(f" ✓")
            except Exception as e:
                print(f" ✗ Error: {e}")
        
        # Save to database
        self.save_results(results)
        
        print(f"{'='*60}")
        print(f"✅ Scan complete - {len(results)} pairs analyzed")
        print(f"{'='*60}\n")
    
    def save_results(self, results):
        """Save scan results to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        saved = 0
        for result in results:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO scan_results 
                    (timestamp, pair, signal, strength, 
                     state_5m, state_15m, state_4h, state_1d, state_1w,
                     sentiment, sentiment_value,
                     lookback1, lookback2, lookback3,
                     lookback1_label, lookback2_label, lookback3_label)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp,
                    result['Pair'],
                    result['Signal'],
                    result['Strength'],
                    result.get('5m', '-'),
                    result.get('15m', '-'),
                    result.get('4h', '-'),
                    result.get('1D', '-'),
                    result.get('1W', '-'),
                    result['Sentiment'],
                    result['Sentiment_Value'],
                    result.get('Lookback1'),
                    result.get('Lookback2'),
                    result.get('Lookback3'),
                    result.get('Lookback1_Label'),
                    result.get('Lookback2_Label'),
                    result.get('Lookback3_Label')
                ))
                saved += 1
            except Exception as e:
                print(f"  ⚠️  Error saving {result['Pair']}: {e}")
        
        conn.commit()
        conn.close()
        print(f"  💾 Saved {saved}/{len(results)} results to database")
    
    def start(self):
        """Start the scanner service"""
        print("\n" + "="*60)
        print("🌞 SOLAR TERMINAL SCANNER SERVICE")
        print("="*60)
        print("⏰ Scan interval: Every 5 minutes")
        print(f"📊 Markets: {len(FX_PAIRS)} pairs")
        print(f"💾 Database: {self.db_path}")
        print("="*60)
        print("\n🚀 Service starting...\n")
        
        # Run first scan immediately
        try:
            self.run_scan()
        except KeyboardInterrupt:
            print("\n\n⚠️  Service interrupted during first scan")
            return
        except Exception as e:
            print(f"\n\n❌ Error during first scan: {e}")
            print("Continuing anyway...\n")
        
        # Schedule scans every 5 minutes
        schedule.every(5).minutes.do(self.run_scan)
        
        print("✅ Service running. Press Ctrl+C to stop.\n")
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n" + "="*60)
            print("⏹️  Scanner Service Stopped")
            print("="*60)
            print("👋 Goodbye!\n")

if __name__ == '__main__':
    service = ScannerService()
    service.start()
