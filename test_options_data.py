import requests
import json
from datetime import datetime, timedelta

# API credentials
API_KEY = "AKTJN56J3HTI2KWEEZ8A"
API_SECRET = "9uhVQTGFRzSRBkk1RWI0ovhBGWg3UDH2u7woInaX"

# Base URLs for Alpaca APIs
TRADING_BASE_URL = "https://api.alpaca.markets/v2"
DATA_BASE_URL = "https://data.alpaca.markets/v2"

# Headers for authentication
headers = {
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET,
    "Accept": "application/json"
}

def test_account_info():
    """Test access to account information"""
    url = f"{TRADING_BASE_URL}/account"
    response = requests.get(url, headers=headers)
    print(f"Account Info Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")

def get_options_contracts(symbol="AAPL"):
    """Get available options contracts for a symbol"""
    url = f"{TRADING_BASE_URL}/options/contracts"
    params = {"underlying_symbols": symbol}
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nOptions Contracts Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        if isinstance(data, dict) and "option_contracts" in data:
            contracts = data["option_contracts"]
            print(f"Found {len(contracts)} contracts")
            if contracts and len(contracts) > 0:
                # Display first 5 contracts (or fewer if less available)
                display_count = min(5, len(contracts))
                for i in range(display_count):
                    print(json.dumps(contracts[i], indent=2))
                return contracts
        else:
            print(f"Unexpected response format: {data}")
            return []
    else:
        print(f"Error: {response.text}")
    return []

def get_options_chain(symbol="AAPL"):
    """Get options chain data for a symbol"""
    url = f"{TRADING_BASE_URL}/options/chains"
    
    # Get current date and a date 30 days in the future
    today = datetime.now()
    expiration_date = today + timedelta(days=30)
    expiration_date_str = expiration_date.strftime("%Y-%m-%d")
    
    params = {
        "underlying_symbols": symbol,
        "expiration_date": expiration_date_str
    }
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nOptions Chain Status: {response.status_code}")
    
    if response.status_code == 200:
        chain_data = response.json()
        print(f"Found options chain data")
        if isinstance(chain_data, dict):
            print(json.dumps(chain_data, indent=2))
        elif isinstance(chain_data, list) and len(chain_data) > 0:
            display_count = min(2, len(chain_data))
            for i in range(display_count):
                print(json.dumps(chain_data[i], indent=2))
    else:
        print(f"Error: {response.text}")

def get_options_quotes(symbol_id=None):
    """Get options quotes for a specific contract"""
    if not symbol_id:
        # Try to get a contract ID first
        contracts = get_options_contracts()
        if contracts and len(contracts) > 0:
            symbol_id = contracts[0]["id"]
        else:
            print("No contract ID available")
            return
    
    url = f"{DATA_BASE_URL}/options/quotes"
    params = {
        "symbols": symbol_id
    }
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nOptions Quotes Status: {response.status_code}")
    
    if response.status_code == 200:
        quotes = response.json()
        print(json.dumps(quotes, indent=2))
    else:
        print(f"Error: {response.text}")

def get_options_trades(symbol_id=None):
    """Get options trades for a specific contract"""
    if not symbol_id:
        # Try to get a contract ID first
        contracts = get_options_contracts()
        if contracts and len(contracts) > 0:
            symbol_id = contracts[0]["id"]
        else:
            print("No contract ID available")
            return
    
    url = f"{DATA_BASE_URL}/options/trades"
    params = {
        "symbols": symbol_id
    }
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nOptions Trades Status: {response.status_code}")
    
    if response.status_code == 200:
        trades = response.json()
        print(json.dumps(trades, indent=2))
    else:
        print(f"Error: {response.text}")

def get_options_historical_data(symbol_id=None):
    """Get historical options data for a specific contract"""
    if not symbol_id:
        # Try to get a contract ID first
        contracts = get_options_contracts()
        if contracts and len(contracts) > 0:
            symbol_id = contracts[0]["id"]
        else:
            print("No contract ID available")
            return
    
    url = f"{DATA_BASE_URL}/options/bars"
    
    # Set time period
    end = datetime.now()
    start = end - timedelta(days=5)  # Last 5 days
    
    params = {
        "symbols": symbol_id,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "timeframe": "1Hour"
    }
    
    response = requests.get(url, headers=headers, params=params)
    print(f"\nOptions Historical Data Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(json.dumps(data, indent=2))
    else:
        print(f"Error: {response.text}")

if __name__ == "__main__":
    print("Testing Alpaca API connectivity for options data...")
    test_account_info()
    contracts = get_options_contracts("AAPL")
    
    if contracts and len(contracts) > 0:
        sample_contract_id = contracts[0]["id"]
        sample_contract_symbol = contracts[0]["symbol"]
        print(f"\nUsing sample contract ID: {sample_contract_id}")
        print(f"Using sample contract symbol: {sample_contract_symbol}")
        get_options_quotes(sample_contract_symbol)
        get_options_trades(sample_contract_symbol)
        get_options_historical_data(sample_contract_symbol)
        
    get_options_chain("AAPL") 