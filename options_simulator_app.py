#!/usr/bin/env python3
"""
Options Trading Simulator - Frontend App

A Streamlit application that provides an interactive frontend
for the high-frequency options trading system, using real historical data from Alpaca.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import subprocess
import time
import sys
import threading
import random
from typing import List, Dict, Any, Tuple, Optional
import requests

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our Alpaca market data provider
from alpaca_market_data import AlpacaMarketData
from alpaca_config import load_credentials

# Constants
DEFAULT_SYMBOLS = ["SPY", "AAPL", "TSLA", "AMZN", "MSFT", "NVDA", "GOOGL", "META", "NFLX", "AMD"]
DATA_DIR = "historical_data"
RESULTS_DIR = "training_results"
MODELS_DIR = "trained_models"
BACKUP_DIR = "results_backup"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BACKUP_DIR, exist_ok=True)

# Load Alpaca credentials
load_credentials()

# Initialize Alpaca data provider
alpaca_data = AlpacaMarketData()

# Set page config
st.set_page_config(
    page_title="Options Trading Simulator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
@st.cache_data(ttl=3600)
def fetch_stock_data(symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Fetch historical stock data from Alpaca with error handling."""
    try:
        # Format dates as strings
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        # Get historical data from Alpaca
        data = None
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use the alpaca_data object to fetch historical prices
                data = alpaca_data.get_historical_prices(
                    symbol, 
                    start_date_str, 
                    end_date_str,
                    timeframe="1D"
                )
                
                # Check if we got a valid DataFrame back
                if isinstance(data, pd.DataFrame) and not data.empty and len(data) > 5:
                    st.success(f"Successfully downloaded data for {symbol}")
                    return data
                
                # If we get here, download failed or returned empty data
                print(f"Failed to download data for {symbol}: {data}")
                time.sleep(2 * (attempt + 1))  # Exponential backoff
                
            except Exception as e:
                print(f"Download attempt {attempt+1} failed for {symbol}: {str(e)}")
                time.sleep(2 * (attempt + 1))  # Longer delay after exception
        
        # If all attempts failed
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise Exception(f"Failed to download data for {symbol} after {max_retries} attempts")
            
        return data
            
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def fetch_options_data(symbol: str, date: Optional[str] = None) -> Dict[str, Any]:
    """Fetch options chain data from Alpaca."""
    try:
        max_retries = 3
        options_data = None
        
        for attempt in range(max_retries):
            try:
                # Get options chain from Alpaca
                options_chain = alpaca_data.get_options_chain(symbol)
                
                if options_chain:
                    # If date is specified, filter for that date
                    if date and date in options_chain:
                        options_data = {
                            "calls": pd.DataFrame(options_chain[date]["calls"]).T,
                            "puts": pd.DataFrame(options_chain[date]["puts"]).T
                        }
                    else:
                        # Use the first available expiration date
                        exp_dates = list(options_chain.keys())
                        if exp_dates:
                            first_date = exp_dates[0]
                            options_data = {
                                "calls": pd.DataFrame(options_chain[first_date]["calls"]).T,
                                "puts": pd.DataFrame(options_chain[first_date]["puts"]).T
                            }
                    
                    # Convert to format that matches our application
                    if options_data:
                        # Process calls
                        calls_df = options_data["calls"]
                        if not calls_df.empty:
                            # Add additional columns if needed
                            if 'lastPrice' not in calls_df.columns:
                                calls_df['lastPrice'] = (calls_df['bid'] + calls_df['ask']) / 2
                            if 'impliedVolatility' not in calls_df.columns:
                                calls_df['impliedVolatility'] = calls_df.get('iv', 0.3)
                        
                        # Process puts
                        puts_df = options_data["puts"]
                        if not puts_df.empty:
                            # Add additional columns if needed
                            if 'lastPrice' not in puts_df.columns:
                                puts_df['lastPrice'] = (puts_df['bid'] + puts_df['ask']) / 2
                            if 'impliedVolatility' not in puts_df.columns:
                                puts_df['impliedVolatility'] = puts_df.get('iv', 0.3)
                        
                        return options_data
                
                # If we get here, no valid data was found - wait and retry
                time.sleep(2 * (attempt + 1))
                
            except Exception as e:
                print(f"Options data attempt {attempt+1} failed for {symbol}: {str(e)}")
                time.sleep(2 * (attempt + 1))
        
        # If all attempts failed
        st.warning(f"Options data for {symbol} is not available. This is likely because your Alpaca account doesn't have options data access. Check the message area below for details.")
        
        # Check for authorization errors specifically
        if options_chain is not None and 'error' in options_chain and 'code' in options_chain['error'] and options_chain['error']['code'] == 40110000:
            st.info("Alpaca API returned an authorization error (code 40110000). This typically means your account doesn't have the appropriate subscription for options data access.")
            st.info("You can sign up for options data access on the Alpaca website or continue using the application with simulated options data.")
        
        # Return empty data structure but with a special flag
        return {
            "calls": pd.DataFrame(),
            "puts": pd.DataFrame(),
            "no_data_reason": "subscription_required"
        }
        
    except Exception as e:
        st.warning(f"Options data not available: {e}")
        return {
            "calls": pd.DataFrame(),
            "puts": pd.DataFrame(),
            "no_data_reason": "error"
        }

def run_simulation(symbols: List[str], start_date: str, end_date: str, cycles: int = 1) -> subprocess.Popen:
    """Run the high-frequency trainer simulation in a subprocess."""
    cmd = [
        "python3", "high_frequency_trainer.py",
        "--symbols", *symbols,
        "--cycles", str(cycles),
        "--interval", "5",  # Short interval for demonstration
        "--start-date", start_date,
        "--end-date", end_date
    ]
    
    # Create a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    
    return process

def load_simulation_results() -> List[Dict[str, Any]]:
    """Load simulation results from the results directory."""
    results = []
    errors = []
    
    # Check if directory exists
    if not os.path.exists(RESULTS_DIR) or not os.path.isdir(RESULTS_DIR):
        return []
    
    for filename in os.listdir(RESULTS_DIR):
        if filename.startswith('simulation_results_') and filename.endswith('.json'):
            file_path = os.path.join(RESULTS_DIR, filename)
            try:
                # Check if file is not empty and is valid JSON
                if os.path.getsize(file_path) > 0:
                    with open(file_path, 'r') as f:
                        try:
                            data = json.load(f)
                            # Make sure we have required fields for a valid result
                            if isinstance(data, dict) and 'timestamp' in data:
                                data['filename'] = filename
                                results.append(data)
                            else:
                                errors.append(f"Invalid data structure in {filename}")
                        except json.JSONDecodeError as e:
                            errors.append(f"JSON decode error in {filename}: {e}")
                else:
                    errors.append(f"Empty file: {filename}")
            except Exception as e:
                errors.append(f"Error reading {filename}: {e}")
    
    # Show errors if there are any
    if errors:
        st.warning(f"Some simulation results could not be loaded: {len(errors)} file(s) had errors")
        if st.checkbox("Show detailed error information"):
            for err in errors:
                st.error(err)
    
    # Sort by timestamp
    results.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return results

def get_latest_simulation_result() -> Optional[Dict[str, Any]]:
    """Get the most recent simulation result automatically."""
    results = load_simulation_results()
    return results[0] if results else None

def plot_equity_curve(results: Dict[str, Any]) -> go.Figure:
    """Create an interactive equity curve plot."""
    # Create a synthetic equity curve based on simulation results
    trades_taken = results.get('total_trades', 0)
    if trades_taken == 0:
        return None
        
    correct_trades = results.get('correct_trades', 0)
    avg_profit = results.get('total_profit', 0) / trades_taken
    
    # Generate equity curve data
    equity = [1000]  # Start with $1000
    for i in range(trades_taken):
        # Randomly assign win/loss based on win rate to simulate curve
        if i < correct_trades:
            equity.append(equity[-1] * 1.05)  # 5% profit
        else:
            equity.append(equity[-1] * 0.97)  # 3% loss
    
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(equity))),
        y=equity,
        mode='lines',
        name='Equity Curve',
        line=dict(color='blue', width=2)
    ))
    
    # Add annotations for key metrics
    fig.add_annotation(
        x=len(equity) - 1,
        y=equity[-1],
        text=f"Final Equity: ${equity[-1]:.2f}",
        showarrow=True,
        arrowhead=1
    )
    
    # Customize layout
    fig.update_layout(
        title="Simulated Equity Curve",
        xaxis_title="Trade Number",
        yaxis_title="Account Equity ($)",
        hovermode="x unified"
    )
    
    return fig

def plot_threshold_performance(results: Dict[str, Any]) -> go.Figure:
    """Create an interactive plot of performance across confidence thresholds."""
    threshold_results = results.get('threshold_results', {})
    if not threshold_results:
        return None
        
    thresholds = []
    accuracies = []
    trades_taken = []
    
    for threshold, data in threshold_results.items():
        thresholds.append(float(threshold))
        accuracies.append(data.get('accuracy', 0) * 100)
        trades_taken.append(data.get('trades_taken', 0))
    
    # Create figure
    fig = go.Figure()
    
    # Add accuracy trace
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=accuracies,
        mode='lines+markers',
        name='Accuracy (%)',
        line=dict(color='blue', width=2),
        marker=dict(size=8)
    ))
    
    # Add trades taken trace on secondary y-axis
    fig.add_trace(go.Scatter(
        x=thresholds,
        y=trades_taken,
        mode='lines+markers',
        name='Trades Taken',
        line=dict(color='red', width=2, dash='dot'),
        marker=dict(size=8),
        yaxis='y2'
    ))
    
    # Customize layout
    fig.update_layout(
        title="Performance by Confidence Threshold",
        xaxis_title="Confidence Threshold",
        yaxis_title="Accuracy (%)",
        yaxis2=dict(
            title="Trades Taken",
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode="x unified"
    )
    
    return fig

def create_stock_chart(data: pd.DataFrame, symbol: str) -> go.Figure:
    """Create an interactive candlestick chart with volume."""
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=data.index,
        open=data['Open'],
        high=data['High'],
        low=data['Low'],
        close=data['Close'],
        name='Price'
    ))
    
    # Add volume as a bar chart on the secondary y-axis
    fig.add_trace(go.Bar(
        x=data.index,
        y=data['Volume'],
        name='Volume',
        marker_color='rgba(0, 0, 255, 0.3)',
        yaxis='y2'
    ))
    
    # Customize layout
    fig.update_layout(
        title=f"{symbol} Stock Price",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        yaxis2=dict(
            title="Volume",
            overlaying='y',
            side='right'
        ),
        legend=dict(orientation="h"),
        height=500
    )
    
    return fig

def backup_simulation_results():
    """Backup all simulation results to a timestamped zip file."""
    if not os.path.exists(RESULTS_DIR):
        return None
        
    # Create timestamp for the backup filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"results_backup_{timestamp}.zip"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    # Import zipfile module
    import zipfile
    
    try:
        # Create zip file
        with zipfile.ZipFile(backup_path, 'w') as zipf:
            # Add all files from RESULTS_DIR
            for root, _, files in os.walk(RESULTS_DIR):
                for file in files:
                    # Only add JSON files
                    if file.endswith('.json'):
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(RESULTS_DIR))
                        zipf.write(file_path, arcname)
        
        return backup_path
    except Exception as e:
        st.error(f"Backup failed: {e}")
        return None

def check_network_connectivity():
    """Check if there is internet connectivity by pinging a reliable server."""
    try:
        # Try to connect to Google's DNS server
        requests.get("https://8.8.8.8", timeout=3)
        return True
    except requests.exceptions.RequestException:
        return False

def main():
    """Main Streamlit application."""
    try:
        st.title("📈 Options Trading Simulator")
        st.write("Simulate options trading strategies with historical market data from Alpaca")
        
        # Check network connectivity at startup
        if not check_network_connectivity():
            st.error("⚠️ No internet connection detected! The app requires internet access to fetch market data.")
            st.warning("Please check your internet connection and refresh the page.")
            return
        
        # Sidebar for configuration
        st.sidebar.header("Configuration")
        
        # Select stock symbols
        default_selected = ["SPY", "AAPL", "TSLA"]
        selected_symbols = st.sidebar.multiselect(
            "Select stock symbols",
            options=DEFAULT_SYMBOLS,
            default=default_selected
        )
        
        # Date range selection
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        start_date_input = st.sidebar.date_input(
            "Start date",
            value=start_date,
            min_value=datetime(2010, 1, 1),
            max_value=end_date - timedelta(days=30)
        )
        
        end_date_input = st.sidebar.date_input(
            "End date",
            value=end_date,
            min_value=start_date_input + timedelta(days=30),
            max_value=end_date
        )
        
        # Number of training cycles
        num_cycles = st.sidebar.slider(
            "Number of training cycles",
            min_value=1,
            max_value=10,
            value=1
        )
        
        # Run simulation button
        if st.sidebar.button("Run Simulation"):
            if not selected_symbols:
                st.sidebar.error("Please select at least one stock symbol")
            else:
                with st.spinner("Running simulation..."):
                    # Create tabs for real-time output and results
                    tab1, tab2 = st.tabs(["Simulation Output", "Training Results"])
                    
                    with tab1:
                        # Create a placeholder for output
                        output_placeholder = st.empty()
                        
                        # Run simulation
                        process = run_simulation(
                            selected_symbols,
                            start_date_input.strftime('%Y-%m-%d'),
                            end_date_input.strftime('%Y-%m-%d'),
                            num_cycles
                        )
                        
                        # Create a container for output
                        output_text = ""
                        
                        # Read output in real-time
                        for line in process.stdout:
                            output_text += line
                            output_placeholder.code(output_text)
                        
                        # Wait for process to complete
                        process.wait()
                        
                        # Read any remaining output
                        for line in process.stdout:
                            output_text += line
                            output_placeholder.code(output_text)
                        
                        # Check for errors
                        if process.returncode != 0:
                            error_output = process.stderr.read()
                            st.error(f"Simulation failed with error code {process.returncode}:\n{error_output}")
                        else:
                            st.success("Simulation completed successfully!")
        
        # Main content area with tabs
        tab1, tab2, tab3 = st.tabs(["Market Data", "Simulation Results", "Trading Performance"])
        
        with tab1:
            st.header("Historical Market Data")
            
            # Select a symbol to display
            if selected_symbols:
                selected_symbol_for_chart = st.selectbox(
                    "Select a symbol to display",
                    options=selected_symbols
                )
                
                # Show loading indicator while fetching data
                with st.spinner(f"Fetching real data for {selected_symbol_for_chart}..."):
                    # Fetch and display stock data
                    data = fetch_stock_data(selected_symbol_for_chart, start_date_input, end_date_input)
                    
                    if not data.empty:
                        # Display stock chart
                        st.plotly_chart(create_stock_chart(data, selected_symbol_for_chart), use_container_width=True)
                        
                        # Stock statistics
                        st.subheader("Stock Statistics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        # Calculate basic statistics
                        if 'Close' in data.columns:
                            current_price = data['Close'].iloc[-1]
                            change_1d = data['Close'].pct_change().iloc[-1] * 100
                            high_52w = data['High'].max()
                            low_52w = data['Low'].min()
                            
                            col1.metric("Current Price", f"${current_price:.2f}")
                            col2.metric("1-Day Change", f"{change_1d:.2f}%", delta=f"{change_1d:.2f}%")
                            col3.metric("52-Week High", f"${high_52w:.2f}")
                            col4.metric("52-Week Low", f"${low_52w:.2f}")
                        
                        # Display options data if available
                        options_data = None
                        try:
                            st.subheader(f"Options Chain Data for {selected_symbol_for_chart}")
                            options_data = fetch_options_data(selected_symbol_for_chart)
                        except Exception as e:
                            st.warning(f"Cannot retrieve options data: {e}")
                        
                        if options_data and "calls" in options_data and "puts" in options_data and not options_data["calls"].empty and not options_data["puts"].empty:
                            # Create two columns for calls and puts
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Call Options")
                                st.dataframe(options_data["calls"].head(10))
                            
                            with col2:
                                st.subheader("Put Options")
                                st.dataframe(options_data["puts"].head(10))
                        else:
                            st.warning("Options data is not available for this symbol or your Alpaca account doesn't have options data access.")
                            st.info("This application can still analyze stock data. Options data requires a paid Alpaca subscription.")
                        
                        # Display raw historical data
                        with st.expander("View historical data"):
                            st.dataframe(data)
                    else:
                        st.error(f"❌ Could not retrieve data for {selected_symbol_for_chart}. Alpaca API is not responding properly.")
                        st.info("Possible solutions:")
                        st.markdown("""
                        * Check your internet connection
                        * Try a different symbol
                        * Wait a few minutes and try again (Alpaca API rate limits may apply)
                        * Try adjusting the date range to a shorter period
                        """)
                        
                        # Add a button to report the issue
                        if st.button("Try Another Symbol"):
                            # Just refresh the page
                            st.rerun()
            else:
                st.info("Please select stock symbols in the sidebar")
        
        with tab2:
            st.header("Simulation Results")
            
            # Add buttons for management operations
            col_refresh, col_cleanup, col_backup, col_manage, col_space = st.columns([1, 1.2, 1.2, 1.2, 0.4])
            with col_refresh:
                if st.button("🔄 Refresh Results"):
                    st.rerun()
            with col_cleanup:
                if st.button("🧹 Clean Corrupted Files"):
                    try:
                        # Run cleanup script
                        with st.spinner("Cleaning up corrupted files..."):
                            result = subprocess.run(
                                ["python3", "cleanup_simulation_files.py", "--delete"],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            st.success("Cleanup completed! Refreshing page...")
                            time.sleep(1)
                            st.rerun()
                    except subprocess.CalledProcessError as e:
                        st.error(f"Cleanup failed: {e.stderr}")
                    except Exception as e:
                        st.error(f"Error running cleanup: {e}")
            with col_backup:
                if st.button("💾 Backup Results"):
                    with st.spinner("Creating backup..."):
                        backup_file = backup_simulation_results()
                        if backup_file:
                            st.success(f"Backup created: {os.path.basename(backup_file)}")
                        else:
                            st.error("Backup failed")
            with col_manage:
                if st.button("♻️ Keep Recent Only"):
                    try:
                        # Run old files cleanup script
                        with st.spinner("Removing old files..."):
                            # Keep only 2 most recent files
                            result = subprocess.run(
                                ["python3", "cleanup_old_files.py", "--keep", "2", "--delete"],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            st.success("Old files removed! Refreshing page...")
                            time.sleep(1)
                            st.rerun()
                    except subprocess.CalledProcessError as e:
                        st.error(f"Cleanup failed: {e.stderr}")
                    except Exception as e:
                        st.error(f"Error running cleanup: {e}")
            
            # Automatically get the latest simulation result
            result = get_latest_simulation_result()
            
            if result:
                # Create columns for metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Win Rate",
                        f"{result.get('trade_accuracy', 0) * 100:.1f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Total Trades",
                        result.get('total_trades', 0),
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Profitable Trades",
                        result.get('correct_trades', 0),
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Total Profit",
                        f"${result.get('total_profit', 0):.2f}",
                        delta=None
                    )
                
                # Display simulation filename and timestamp
                st.info(f"Showing results from {result.get('filename', 'Unknown')} ({result.get('timestamp', 'Unknown date')})")
                
                # Display equity curve
                equity_fig = plot_equity_curve(result)
                if equity_fig:
                    st.plotly_chart(equity_fig, use_container_width=True)
                
                # Display threshold performance
                threshold_fig = plot_threshold_performance(result)
                if threshold_fig:
                    st.plotly_chart(threshold_fig, use_container_width=True)
                
                # Display raw results
                with st.expander("View raw results"):
                    st.json(result)
            else:
                st.warning("No valid simulation results available.")
                st.info("Run a simulation first by selecting symbols in the sidebar and clicking 'Run Simulation'.")
                
                # Add button to generate test results
                if st.button("Generate Test Results"):
                    try:
                        with st.spinner("Generating test results..."):
                            result = subprocess.run(
                                ["python3", "generate_test_results.py", "--count", "1"],
                                capture_output=True,
                                text=True,
                                check=True
                            )
                            st.success("Test results generated! Refreshing page...")
                            time.sleep(1)
                            st.rerun()
                    except subprocess.CalledProcessError as e:
                        st.error(f"Failed to generate test results: {e.stderr}")
                    except Exception as e:
                        st.error(f"Error generating test results: {e}")
                
                # Check if directory exists but has no valid files
                if os.path.exists(RESULTS_DIR) and os.path.isdir(RESULTS_DIR):
                    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith('.json') and f.startswith('simulation_results_')]
                    if json_files:
                        st.error("Found simulation files but they appear to be corrupted or incomplete.")
                        st.info("Try running a new simulation to create valid result files.")
        
        with tab3:
            st.header("Trading Performance Analysis")
            
            # Load all results for performance analysis
            results = load_simulation_results()
            
            if results:
                # Aggregate performance metrics across all simulations
                total_simulations = len(results)
                avg_win_rate = np.mean([r.get('trade_accuracy', 0) for r in results]) * 100
                avg_profit = np.mean([r.get('total_profit', 0) for r in results])
                total_trades = sum([r.get('total_trades', 0) for r in results])
                
                # Display aggregate metrics
                st.subheader("Aggregate Performance")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Total Simulations",
                        total_simulations,
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Average Win Rate",
                        f"{avg_win_rate:.1f}%",
                        delta=None
                    )
                
                with col3:
                    st.metric(
                        "Total Trades",
                        total_trades,
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        "Average Profit",
                        f"${avg_profit:.2f}",
                        delta=None
                    )
                
                # Create performance comparison chart
                st.subheader("Performance Comparison")
                
                # Extract data for comparison
                timestamps = [r.get('timestamp', '').split('T')[0] for r in results]
                win_rates = [r.get('trade_accuracy', 0) * 100 for r in results]
                profits = [r.get('total_profit', 0) for r in results]
                
                # Create DataFrame for comparison
                comparison_df = pd.DataFrame({
                    'Timestamp': timestamps,
                    'Win Rate (%)': win_rates,
                    'Profit ($)': profits
                })
                
                # Display comparison chart
                comparison_fig = px.bar(
                    comparison_df,
                    x='Timestamp',
                    y=['Win Rate (%)', 'Profit ($)'],
                    barmode='group',
                    title='Performance Comparison Across Simulations'
                )
                
                st.plotly_chart(comparison_fig, use_container_width=True)
                
                # Feature importance analysis if available
                st.subheader("Feature Importance Analysis")
                
                # Check if any result has feature importance
                has_feature_importance = any('feature_importance' in r for r in results)
                
                if has_feature_importance:
                    # Get the most recent result with feature importance
                    latest_result_with_features = next(
                        (r for r in results if 'feature_importance' in r),
                        None
                    )
                    
                    if latest_result_with_features:
                        # Extract feature importance
                        feature_importance = latest_result_with_features.get('feature_importance', {})
                        
                        # Convert to DataFrame for plotting
                        features_df = pd.DataFrame({
                            'Feature': list(feature_importance.keys()),
                            'Importance': list(feature_importance.values())
                        }).sort_values(by='Importance', ascending=False)
                        
                        # Create horizontal bar chart
                        fig = px.bar(
                            features_df,
                            x='Importance',
                            y='Feature',
                            orientation='h',
                            title='Feature Importance'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No feature importance data available")
            else:
                st.info("No simulation results available. Run a simulation first.")
    except Exception as e:
        # Display any unhandled errors
        st.error(f"An error occurred: {e}")
        import traceback
        st.error(traceback.format_exc())
        st.warning("Please try refreshing the page or contact support if the issue persists.")

if __name__ == "__main__":
    main() 