"""
FastAPI Server for Solar Terminal
Serves scan results with API key authentication
"""
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import json
from pathlib import Path
from datetime import datetime

app = FastAPI(title="Solar Terminal API", version="1.0.0")

# CORS middleware - allow desktop app to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to API keys file
API_KEYS_FILE = Path(__file__).parent / 'api_keys.json'

def load_api_keys():
    """Load API keys from JSON file"""
    if not API_KEYS_FILE.exists():
        # Create empty keys file if doesn't exist
        with open(API_KEYS_FILE, 'w') as f:
            json.dump({}, f, indent=2)
        return {}
    
    try:
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return {}

def verify_api_key(x_api_key: str = Header(None)):
    """Verify API key is valid"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="Missing API key. Include X-API-Key header."
        )
    
    api_keys = load_api_keys()
    
    if x_api_key not in api_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    
    user = api_keys[x_api_key]
    
    if not user.get('active', True):
        raise HTTPException(
            status_code=403,
            detail="API key has been deactivated"
        )
    
    return user

@app.get("/")
def root():
    """API root - health check (no auth required)"""
    return {
        "service": "Solar Terminal API",
        "version": "1.0.0",
        "status": "online"
    }

@app.get("/ping")
def ping(user = Depends(verify_api_key)):
    """Test authentication"""
    return {
        "status": "authenticated",
        "user": user['name'],
        "email": user['email']
    }

@app.get("/latest-scan")
def get_latest_scan(user = Depends(verify_api_key)):
    """Get the most recent scan results"""
    try:
        conn = sqlite3.connect('live_scans.db')
        cursor = conn.cursor()
        
        # Get latest timestamp
        cursor.execute('SELECT MAX(timestamp) FROM scan_results')
        latest_time = cursor.fetchone()[0]
        
        if not latest_time:
            return {
                "results": [],
                "timestamp": None,
                "message": "No scans available yet"
            }
        
        # Get all results from that scan
        cursor.execute('''
            SELECT pair, signal, strength, state_5m, state_15m, state_4h, 
                   state_1d, state_1w, sentiment, sentiment_value, 
                   lookback1, lookback2, lookback3,
                   lookback1_label, lookback2_label, lookback3_label
            FROM scan_results 
            WHERE timestamp = ?
        ''', (latest_time,))
        
        results = []
        for row in cursor.fetchall():
            result = {
                'Pair': row[0],
                'Signal': row[1],
                'Strength': row[2],
                '5m': row[3],
                '15m': row[4],
                '4h': row[5],
                '1D': row[6],
                '1W': row[7],
                'Sentiment': row[8],
                'Sentiment_Value': row[9]
            }
            
            # Add lookbacks if present
            if row[10] is not None:
                result['Lookback1'] = row[10]
                result['Lookback1_Label'] = row[13] or 'LB1'
            if row[11] is not None:
                result['Lookback2'] = row[11]
                result['Lookback2_Label'] = row[14] or 'LB2'
            if row[12] is not None:
                result['Lookback3'] = row[12]
                result['Lookback3_Label'] = row[15] or 'LB3'
            
            results.append(result)
        
        conn.close()
        
        return {
            "results": results,
            "timestamp": latest_time,
            "count": len(results)
        }
        
    except sqlite3.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="Scanner database not found. Is the scanner service running?"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching scan results: {str(e)}"
        )

@app.get("/last-updated")
def get_last_updated(user = Depends(verify_api_key)):
    """Get when the last scan was run"""
    try:
        conn = sqlite3.connect('live_scans.db')
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(timestamp) FROM scan_results')
        latest_time = cursor.fetchone()[0]
        conn.close()
        
        return {
            "last_updated": latest_time,
            "status": "ok" if latest_time else "no_scans"
        }
        
    except sqlite3.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="Scanner database not found"
        )

@app.get("/api-keys/info")
def get_api_key_info(user = Depends(verify_api_key)):
    """Get information about the current API key"""
    return {
        "name": user['name'],
        "email": user['email'],
        "created": user.get('created', 'unknown'),
        "tier": user.get('tier', 'standard')
    }

@app.get("/latest-stock-scan")
def get_latest_stock_scan(user = Depends(verify_api_key)):
    """Get the most recent stock scan results"""
    try:
        conn = sqlite3.connect('stock_scans.db')
        cursor = conn.cursor()
        
        # Get latest timestamp
        cursor.execute('SELECT MAX(timestamp) FROM stock_results')
        latest_time = cursor.fetchone()[0]
        
        if not latest_time:
            return {
                "results": [],
                "timestamp": None,
                "message": "No stock scans available yet"
            }
        
        # Get all results from that scan
        cursor.execute('''
            SELECT symbol, name, signal, strength,
                   state_1h, state_1d, state_1w,
                   sentiment, sentiment_value, price, change_24h
            FROM stock_results
            WHERE timestamp = ?
        ''', (latest_time,))
        
        results = []
        for row in cursor.fetchall():
            result = {
                'Pair': row[0],      # Symbol
                'Name': row[1],      # Company name
                'Signal': row[2],
                'Strength': row[3],
                '1h': row[4],
                '1D': row[5],
                '1W': row[6],
                'Sentiment': row[7],
                'Sentiment_Value': row[8],
                'Price': row[9],
                'Lookback1': row[10]  # 24h change
            }
            results.append(result)
        
        conn.close()
        
        return {
            "results": results,
            "timestamp": latest_time,
            "count": len(results)
        }
        
    except sqlite3.OperationalError:
        raise HTTPException(
            status_code=503,
            detail="Stock scanner database not found. Is the stock scanner service running?"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stock scan results: {str(e)}"
        )

if __name__ == '__main__':
    import uvicorn
    print("Starting Solar Terminal API Server...")
    print("API Keys file:", API_KEYS_FILE)
    print("\nTo manage API keys, use: python manage_keys.py")
    print("\nStarting server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
