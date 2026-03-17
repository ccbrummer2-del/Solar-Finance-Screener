"""
Stock symbols for US 500 stocks
Grouped by sector for easy management
"""

# Top 100 stocks from S&P 500 by market cap
US_500_STOCKS = {
    # Technology
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'NVDA': 'NVIDIA Corporation',
    'GOOGL': 'Alphabet Inc. (Google)',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'AVGO': 'Broadcom Inc.',
    'ORCL': 'Oracle Corporation',
    'ADBE': 'Adobe Inc.',
    'CRM': 'Salesforce Inc.',
    'CSCO': 'Cisco Systems Inc.',
    'INTC': 'Intel Corporation',
    'AMD': 'Advanced Micro Devices',
    'QCOM': 'QUALCOMM Inc.',
    'IBM': 'IBM',
    'NOW': 'ServiceNow Inc.',
    'INTU': 'Intuit Inc.',
    'TXN': 'Texas Instruments',
    'AMAT': 'Applied Materials',
    
    # Financial Services
    'BRK.B': 'Berkshire Hathaway',
    'JPM': 'JPMorgan Chase',
    'V': 'Visa Inc.',
    'MA': 'Mastercard Inc.',
    'BAC': 'Bank of America',
    'WFC': 'Wells Fargo',
    'MS': 'Morgan Stanley',
    'GS': 'Goldman Sachs',
    'AXP': 'American Express',
    'BLK': 'BlackRock Inc.',
    'C': 'Citigroup Inc.',
    'SCHW': 'Charles Schwab',
    'CB': 'Chubb Limited',
    'PGR': 'Progressive Corp',
    'MMC': 'Marsh & McLennan',
    
    # Healthcare
    'UNH': 'UnitedHealth Group',
    'JNJ': 'Johnson & Johnson',
    'LLY': 'Eli Lilly',
    'ABBV': 'AbbVie Inc.',
    'MRK': 'Merck & Co.',
    'PFE': 'Pfizer Inc.',
    'TMO': 'Thermo Fisher',
    'ABT': 'Abbott Laboratories',
    'DHR': 'Danaher Corporation',
    'BMY': 'Bristol Myers Squibb',
    'AMGN': 'Amgen Inc.',
    'GILD': 'Gilead Sciences',
    'CVS': 'CVS Health',
    'CI': 'Cigna Group',
    'ISRG': 'Intuitive Surgical',
    
    # Consumer
    'WMT': 'Walmart Inc.',
    'HD': 'Home Depot',
    'PG': 'Procter & Gamble',
    'COST': 'Costco Wholesale',
    'KO': 'Coca-Cola Company',
    'PEP': 'PepsiCo Inc.',
    'MCD': "McDonald's Corp",
    'NKE': 'Nike Inc.',
    'SBUX': 'Starbucks Corp',
    'TGT': 'Target Corporation',
    'LOW': "Lowe's Companies",
    'DIS': 'Walt Disney',
    'NFLX': 'Netflix Inc.',
    'CMCSA': 'Comcast Corp',
    'PM': 'Philip Morris',
    
    # Industrial
    'BA': 'Boeing Company',
    'CAT': 'Caterpillar Inc.',
    'HON': 'Honeywell',
    'UPS': 'United Parcel Service',
    'RTX': 'RTX Corporation',
    'DE': 'Deere & Company',
    'LMT': 'Lockheed Martin',
    'GE': 'General Electric',
    'MMM': '3M Company',
    'EMR': 'Emerson Electric',
    
    # Energy
    'XOM': 'Exxon Mobil',
    'CVX': 'Chevron Corporation',
    'COP': 'ConocoPhillips',
    'SLB': 'Schlumberger',
    'EOG': 'EOG Resources',
    'PXD': 'Pioneer Natural Resources',
    'MPC': 'Marathon Petroleum',
    'PSX': 'Phillips 66',
    
    # Communication
    'T': 'AT&T Inc.',
    'VZ': 'Verizon Communications',
    'TMUS': 'T-Mobile US',
    
    # Utilities
    'NEE': 'NextEra Energy',
    'DUK': 'Duke Energy',
    'SO': 'Southern Company',
    'D': 'Dominion Energy',
    
    # Real Estate
    'AMT': 'American Tower',
    'PLD': 'Prologis Inc.',
    'CCI': 'Crown Castle',
    'EQIX': 'Equinix Inc.',
    
    # Materials
    'LIN': 'Linde plc',
    'APD': 'Air Products',
    'SHW': 'Sherwin-Williams',
    'FCX': 'Freeport-McMoRan',
    'NEM': 'Newmont Corporation',
}

# You can add more stocks here - up to 500 total
# For now starting with top 100 most liquid stocks

def get_stock_count():
    """Get total number of stocks"""
    return len(US_500_STOCKS)

def get_stocks_by_sector():
    """Get stocks grouped by sector (for future use)"""
    # This would return organized dict by sector
    # Not implemented yet but placeholder for future
    pass
