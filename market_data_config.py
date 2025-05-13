#!/usr/bin/env python
"""
Configuration for Market Data API credentials
"""

import os
import sys
from dotenv import load_dotenv

# Default token value from the instruction
DEFAULT_TOKEN = "ZFJHdWc5anZBSEtSSFIta2ItbGpKTWpCSExab21hWUdRYkt4d1J3UENKST0"

def load_market_data_credentials():
    """
    Load Market Data API credentials from environment variables or .env file.
    """
    # Try loading from .env file first
    load_dotenv()
    
    # Check if MARKET_DATA_API_TOKEN is already set in environment
    token = os.environ.get("MARKET_DATA_API_TOKEN")
    
    # If token not found in environment or .env, use the default token
    if not token:
        os.environ["MARKET_DATA_API_TOKEN"] = DEFAULT_TOKEN
        token = DEFAULT_TOKEN
        print("Using default Market Data API token")
    
    return {
        "token": token
    }

def save_market_data_credentials(token):
    """
    Save Market Data API credentials to environment variables.
    
    Args:
        token: The Market Data API token
    """
    os.environ["MARKET_DATA_API_TOKEN"] = token
    
    # Also try to save to .env file for persistence
    try:
        # Check if .env exists
        if os.path.exists(".env"):
            # Read current content
            with open(".env", "r") as f:
                content = f.read()
            
            # Check if token is already in file
            if "MARKET_DATA_API_TOKEN" in content:
                # Replace the line
                lines = content.split("\n")
                for i, line in enumerate(lines):
                    if line.startswith("MARKET_DATA_API_TOKEN="):
                        lines[i] = f"MARKET_DATA_API_TOKEN={token}"
                        break
                
                # Write back to file
                with open(".env", "w") as f:
                    f.write("\n".join(lines))
            else:
                # Append to file
                with open(".env", "a") as f:
                    f.write(f"\nMARKET_DATA_API_TOKEN={token}")
        else:
            # Create new .env file
            with open(".env", "w") as f:
                f.write(f"MARKET_DATA_API_TOKEN={token}")
                
        print("Market Data API token saved to .env file")
    except Exception as e:
        print(f"Warning: Could not save token to .env file: {e}")
    
    return {
        "token": token
    } 