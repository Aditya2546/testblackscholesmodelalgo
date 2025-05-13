#!/usr/bin/env python
"""
Data preparation script for options model training
Processes collected market data and prepares features for model training
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import glob
import json
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

# Directories
DATA_DIR = "training_data"
HISTORICAL_DIR = f"{DATA_DIR}/historical"
REALTIME_DIR = f"{DATA_DIR}/realtime"
PROCESSED_DIR = f"{DATA_DIR}/processed"
MODELS_DIR = "models"

@dataclass
class OptionFeatures:
    """
    Features for options model training
    """
    # Basic option info
    symbol: str
    option_type: str  # "C" or "P"
    strike: float
    expiration_date: str
    days_to_expiry: int
    
    # Market data
    underlying_price: float
    bid: float
    ask: float
    mid: float
    volume: float
    open_interest: float
    
    # Greeks and IV
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    
    # Calculated features
    moneyness: float  # Strike / Underlying price
    relative_strike: float  # (Strike - Underlying) / Underlying
    bid_ask_spread: float
    bid_ask_spread_percent: float
    volume_oi_ratio: float
    time_value: float
    intrinsic_value: float
    
    # Target variables
    price_change_1d: float = 0.0
    price_change_3d: float = 0.0
    price_change_5d: float = 0.0
    price_change_10d: float = 0.0
    
    # Prediction
    expected_return: float = 0.0
    signal_strength: float = 0.0

def setup_directories():
    """Create directories for processed data and models"""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Directories set up: {PROCESSED_DIR}, {MODELS_DIR}")

def load_all_realtime_data() -> pd.DataFrame:
    """
    Load all real-time data files and combine into a single DataFrame
    
    Returns:
        Combined DataFrame with all real-time data
    """
    all_files = glob.glob(f"{REALTIME_DIR}/*.csv")
    
    if not all_files:
        print(f"No data files found in {REALTIME_DIR}")
        return pd.DataFrame()
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} records from {len(all_files)} real-time data files")
    return combined_df

def load_all_historical_data() -> pd.DataFrame:
    """
    Load all historical data files and combine into a single DataFrame
    
    Returns:
        Combined DataFrame with all historical data
    """
    all_files = glob.glob(f"{HISTORICAL_DIR}/*.csv")
    
    if not all_files:
        print(f"No data files found in {HISTORICAL_DIR}")
        return pd.DataFrame()
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            dfs.append(df)
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not dfs:
        return pd.DataFrame()
    
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(combined_df)} records from {len(all_files)} historical data files")
    return combined_df

def calculate_days_to_expiry(row) -> int:
    """Calculate days to expiry from expiration date"""
    try:
        expiry_date = datetime.strptime(row["expiration_date"], "%Y-%m-%d")
        timestamp = row.get("timestamp")
        
        if isinstance(timestamp, str):
            current_date = datetime.strptime(timestamp.split(" ")[0], "%Y-%m-%d")
        else:
            current_date = datetime.now()
            
        days = (expiry_date - current_date).days
        return max(0, days)
    except Exception:
        return 0

def calculate_moneyness(row) -> float:
    """Calculate moneyness (Strike/Underlying)"""
    try:
        if row["underlying_price"] > 0:
            return row["strike"] / row["underlying_price"]
        return 1.0
    except Exception:
        return 1.0

def calculate_relative_strike(row) -> float:
    """Calculate relative strike (Strike-Underlying)/Underlying"""
    try:
        if row["underlying_price"] > 0:
            return (row["strike"] - row["underlying_price"]) / row["underlying_price"]
        return 0.0
    except Exception:
        return 0.0

def calculate_bid_ask_spread(row) -> Tuple[float, float]:
    """Calculate bid-ask spread and spread percentage"""
    try:
        spread = row["ask"] - row["bid"]
        if row["ask"] > 0:
            spread_percent = spread / row["ask"]
        else:
            spread_percent = 0.0
        return spread, spread_percent
    except Exception:
        return 0.0, 0.0

def calculate_volume_oi_ratio(row) -> float:
    """Calculate volume to open interest ratio"""
    try:
        if row["open_interest"] > 0:
            return row["volume"] / row["open_interest"]
        return 0.0
    except Exception:
        return 0.0

def calculate_intrinsic_extrinsic(row) -> Tuple[float, float]:
    """Calculate intrinsic and extrinsic (time) value"""
    try:
        # For calls: max(0, underlying - strike)
        # For puts: max(0, strike - underlying)
        if row["option_type"].upper() in ["C", "CALL"]:
            intrinsic = max(0, row["underlying_price"] - row["strike"])
        else:
            intrinsic = max(0, row["strike"] - row["underlying_price"])
            
        # Time value = mid - intrinsic
        time_value = max(0, row["mid"] - intrinsic)
        
        return intrinsic, time_value
    except Exception:
        return 0.0, 0.0

def preprocess_option_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess option data: clean, calculate features, and prepare for model training
    
    Args:
        df: Raw option data DataFrame
        
    Returns:
        Processed DataFrame with calculated features
    """
    if df.empty:
        return df
    
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Convert option_type to uppercase if it's a string
    if "option_type" in processed_df.columns and processed_df["option_type"].dtype == object:
        processed_df["option_type"] = processed_df["option_type"].str.upper()
    
    # Convert timestamp to datetime if it's a string
    if "timestamp" in processed_df.columns and processed_df["timestamp"].dtype == object:
        processed_df["timestamp"] = pd.to_datetime(processed_df["timestamp"])
    
    # Convert expiration_date to datetime if it's a string
    if "expiration_date" in processed_df.columns and processed_df["expiration_date"].dtype == object:
        processed_df["expiration_date_dt"] = pd.to_datetime(processed_df["expiration_date"])
    
    # Handle missing values
    numeric_cols = ["bid", "ask", "volume", "open_interest", "delta", "gamma", "theta", "vega", "iv"]
    for col in numeric_cols:
        if col in processed_df.columns:
            # Fill missing values with appropriate defaults
            if col in ["bid", "ask", "delta", "gamma", "theta", "vega", "iv"]:
                processed_df[col] = processed_df[col].fillna(0.0)
            elif col in ["volume", "open_interest"]:
                processed_df[col] = processed_df[col].fillna(0.0).astype(int)
    
    # Calculate mid price
    if "bid" in processed_df.columns and "ask" in processed_df.columns:
        processed_df["mid"] = (processed_df["bid"] + processed_df["ask"]) / 2
    
    # Calculate days to expiry
    processed_df["days_to_expiry"] = processed_df.apply(calculate_days_to_expiry, axis=1)
    
    # Calculate moneyness and relative strike
    processed_df["moneyness"] = processed_df.apply(calculate_moneyness, axis=1)
    processed_df["relative_strike"] = processed_df.apply(calculate_relative_strike, axis=1)
    
    # Calculate bid-ask spread
    bid_ask_results = processed_df.apply(calculate_bid_ask_spread, axis=1, result_type="expand")
    if not bid_ask_results.empty and bid_ask_results.shape[1] >= 2:
        processed_df["bid_ask_spread"] = bid_ask_results[0]
        processed_df["bid_ask_spread_percent"] = bid_ask_results[1]
    
    # Calculate volume/OI ratio
    processed_df["volume_oi_ratio"] = processed_df.apply(calculate_volume_oi_ratio, axis=1)
    
    # Calculate intrinsic and time value
    intrinsic_results = processed_df.apply(calculate_intrinsic_extrinsic, axis=1, result_type="expand")
    if not intrinsic_results.empty and intrinsic_results.shape[1] >= 2:
        processed_df["intrinsic_value"] = intrinsic_results[0]
        processed_df["time_value"] = intrinsic_results[1]
    
    # For columns that should be rename for consistency
    rename_map = {
        "iv": "implied_volatility",
        "openInterest": "open_interest",
        "underlyingPrice": "underlying_price"
    }
    
    # Rename columns that exist
    cols_to_rename = {k: v for k, v in rename_map.items() if k in processed_df.columns and v not in processed_df.columns}
    if cols_to_rename:
        processed_df = processed_df.rename(columns=cols_to_rename)
    
    return processed_df

def create_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create training features and label data from processed options data
    
    Args:
        df: Processed option data DataFrame
        
    Returns:
        DataFrame with training features and labels
    """
    # Select only necessary columns for training
    if df.empty:
        return df
    
    try:
        # Required columns for features
        required_cols = [
            "symbol", "option_type", "strike", "expiration_date", "underlying_price",
            "bid", "ask", "mid", "volume", "open_interest", "days_to_expiry",
            "implied_volatility", "delta", "gamma", "theta", "vega",
            "moneyness", "relative_strike", "bid_ask_spread", "bid_ask_spread_percent",
            "volume_oi_ratio", "intrinsic_value", "time_value"
        ]
        
        # Check if all required columns exist
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing required columns: {missing_cols}")
            # Add missing columns with default values
            for col in missing_cols:
                df[col] = 0.0
        
        # Create a feature DataFrame
        features_df = df[required_cols].copy()
        
        # One-hot encode categorical variables
        features_df = pd.get_dummies(features_df, columns=["symbol", "option_type"], drop_first=False)
        
        return features_df
    
    except Exception as e:
        print(f"Error creating training features: {e}")
        return pd.DataFrame()

def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets
    
    Args:
        df: Feature DataFrame
        test_size: Proportion of data to use for testing
        
    Returns:
        Training and testing DataFrames
    """
    if df.empty:
        return df, df
    
    # Shuffle the data
    shuffled_df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Calculate split index
    split_idx = int(len(shuffled_df) * (1 - test_size))
    
    # Split the data
    train_df = shuffled_df.iloc[:split_idx]
    test_df = shuffled_df.iloc[split_idx:]
    
    return train_df, test_df

def analyze_option_data(df: pd.DataFrame):
    """
    Analyze the option data and create visualizations
    
    Args:
        df: Processed option data DataFrame
    """
    if df.empty:
        print("No data to analyze")
        return
    
    # Create output directory for visualizations
    viz_dir = f"{PROCESSED_DIR}/visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # Set default plot style
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # 1. Distribution of implied volatility
    plt.figure(figsize=(10, 6))
    if "implied_volatility" in df.columns:
        sns.histplot(df["implied_volatility"].dropna(), kde=True)
        plt.title("Distribution of Implied Volatility")
        plt.xlabel("Implied Volatility")
        plt.ylabel("Count")
        plt.savefig(f"{viz_dir}/implied_volatility_distribution.png")
        plt.close()
    
    # 2. Distribution of option prices (mid)
    plt.figure(figsize=(10, 6))
    if "mid" in df.columns:
        sns.histplot(df["mid"].dropna(), kde=True)
        plt.title("Distribution of Option Prices")
        plt.xlabel("Option Price (Mid)")
        plt.ylabel("Count")
        plt.savefig(f"{viz_dir}/option_price_distribution.png")
        plt.close()
    
    # 3. Relationship between moneyness and implied volatility
    plt.figure(figsize=(10, 6))
    if "moneyness" in df.columns and "implied_volatility" in df.columns:
        sns.scatterplot(x="moneyness", y="implied_volatility", hue="option_type", data=df)
        plt.title("Implied Volatility vs. Moneyness")
        plt.xlabel("Moneyness (Strike/Underlying)")
        plt.ylabel("Implied Volatility")
        plt.savefig(f"{viz_dir}/iv_vs_moneyness.png")
        plt.close()
    
    # 4. Volume vs. Open Interest
    plt.figure(figsize=(10, 6))
    if "volume" in df.columns and "open_interest" in df.columns:
        sns.scatterplot(x="open_interest", y="volume", hue="option_type", data=df)
        plt.title("Volume vs. Open Interest")
        plt.xlabel("Open Interest")
        plt.ylabel("Volume")
        plt.savefig(f"{viz_dir}/volume_vs_oi.png")
        plt.close()
    
    print(f"Visualizations saved to {viz_dir}")

def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save processed data to CSV file
    
    Args:
        df: DataFrame to save
        filename: Output filename
    """
    if df.empty:
        print(f"No data to save for {filename}")
        return
    
    output_path = f"{PROCESSED_DIR}/{filename}"
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def main():
    """Main function for data preparation"""
    print("Options Data Preparation for Model Training")
    print("==========================================")
    
    # Setup directories
    setup_directories()
    
    # Load real-time data
    realtime_df = load_all_realtime_data()
    
    # Load historical data
    historical_df = load_all_historical_data()
    
    # Combine datasets if both exist
    if not realtime_df.empty and not historical_df.empty:
        # Ensure column consistency
        common_cols = set(realtime_df.columns).intersection(set(historical_df.columns))
        realtime_subset = realtime_df[list(common_cols)]
        historical_subset = historical_df[list(common_cols)]
        
        # Combine datasets
        combined_df = pd.concat([realtime_subset, historical_subset], ignore_index=True)
        print(f"Combined dataset contains {len(combined_df)} records")
    elif not realtime_df.empty:
        combined_df = realtime_df
    elif not historical_df.empty:
        combined_df = historical_df
    else:
        print("No data available for processing")
        return
    
    # Preprocess data
    processed_df = preprocess_option_data(combined_df)
    print(f"Processed data contains {len(processed_df)} records")
    
    # Save processed data
    save_processed_data(processed_df, "processed_options_data.csv")
    
    # Create training features
    features_df = create_training_features(processed_df)
    print(f"Feature dataset contains {len(features_df)} records with {features_df.shape[1]} features")
    
    # Save feature data
    save_processed_data(features_df, "option_features.csv")
    
    # Split into training and testing sets
    train_df, test_df = split_train_test(features_df)
    print(f"Training set: {len(train_df)} records, Testing set: {len(test_df)} records")
    
    # Save train/test datasets
    save_processed_data(train_df, "option_train_data.csv")
    save_processed_data(test_df, "option_test_data.csv")
    
    # Analyze data and create visualizations
    analyze_option_data(processed_df)
    
    print("Data preparation complete!")

if __name__ == "__main__":
    main() 