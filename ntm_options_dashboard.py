#!/usr/bin/env python
"""
Near-The-Money Options Analysis Dashboard

This dashboard provides a visual interface for analyzing options that are
near the current market price, with a focus on liquidity, volume, and trading potential.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objects as go
import plotly.express as px
from typing import List, Dict, Tuple
import logging
import time
import os
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import the NearTheMoneyAnalyzer and other dependencies
from near_the_money_analyzer import NearTheMoneyAnalyzer
from alpaca_market_data import AlpacaMarketData
from options_day_trader_sim import OptionSignal, Config

# Page configuration
st.set_page_config(
    page_title="Near-The-Money Options Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    # Check if Alpaca credentials are available in environment variables
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    # If not in environment, check alpaca_config.py
    if not api_key or not api_secret:
        try:
            from alpaca_config import API_KEY, API_SECRET
            api_key = API_KEY
            api_secret = API_SECRET
        except ImportError:
            st.error("Alpaca API credentials not found. Please configure them in alpaca_config.py")
            st.stop()
    
    # Initialize market data provider
    market_data = AlpacaMarketData(api_key, api_secret)
    
    # Initialize analyzer
    st.session_state.analyzer = NearTheMoneyAnalyzer(market_data)
    
    # Initialize other session variables
    st.session_state.signals = []
    st.session_state.last_refresh = None
    st.session_state.watchlist = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "TSLA", "META"]
    st.session_state.custom_strikes = {}
    st.session_state.selected_expiration = None

# Function to color RSI values
def color_score(val):
    """Color formatting for scores"""
    if val >= 0.7:
        return f'background-color: green; color: white'
    elif val >= 0.5:
        return f'background-color: lightgreen; color: black'
    elif val >= 0.3:
        return f'background-color: yellow; color: black'
    else:
        return f'background-color: red; color: white'

# Function to format option signals as DataFrame
def signals_to_dataframe(signals: List[OptionSignal]) -> pd.DataFrame:
    """Convert option signals to DataFrame for display"""
    if not signals:
        return pd.DataFrame()
    
    data = []
    for signal in signals:
        data.append({
            "Symbol": signal.symbol,
            "Type": signal.option_type.upper(),
            "Strike": signal.strike,
            "Expiration": signal.expiration,
            "Current Price": signal.current_price,
            "Entry Low": signal.entry_price_range[0],
            "Entry High": signal.entry_price_range[1],
            "Stop Loss": signal.stop_loss,
            "Target": signal.target_price,
            "Score": signal.signal_strength,
            "Volume": signal.volume,
            "Open Interest": signal.open_interest,
            "IV": signal.iv,
            "Delta": signal.delta,
            "Underlying": signal.underlying_price
        })
    
    return pd.DataFrame(data)

# Function to display option details
def display_option_details(signal: OptionSignal):
    """Display detailed information about an option"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{signal.symbol} {signal.option_type.upper()} ${signal.strike}")
        st.write(f"**Expiration:** {signal.expiration}")
        st.write(f"**Underlying Price:** ${signal.underlying_price:.2f}")
        st.write(f"**Option Price:** ${signal.current_price:.2f}")
        st.write(f"**Signal Score:** {signal.signal_strength:.2f}")
    
    with col2:
        st.write(f"**Entry Range:** ${signal.entry_price_range[0]:.2f} - ${signal.entry_price_range[1]:.2f}")
        st.write(f"**Stop Loss:** ${signal.stop_loss:.2f}")
        st.write(f"**Target Price:** ${signal.target_price:.2f}")
        st.write(f"**Volume:** {signal.volume} | **Open Interest:** {signal.open_interest}")
        st.write(f"**IV:** {signal.iv:.1f}% | **Delta:** {signal.delta:.2f}")
    
    # Calculate potential profit and loss
    avg_entry = (signal.entry_price_range[0] + signal.entry_price_range[1]) / 2
    profit_potential = (signal.target_price - avg_entry) / avg_entry * 100
    loss_potential = (signal.stop_loss - avg_entry) / avg_entry * 100
    
    # Display risk/reward
    st.write("---")
    st.write("### Risk/Reward Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Profit Potential", f"{profit_potential:.1f}%", f"${signal.target_price - avg_entry:.2f} per contract")
    
    with col2:
        st.metric("Loss Risk", f"{loss_potential:.1f}%", f"${signal.stop_loss - avg_entry:.2f} per contract")
    
    with col3:
        risk_reward = abs(profit_potential / loss_potential) if loss_potential != 0 else float('inf')
        st.metric("Risk/Reward Ratio", f"1:{risk_reward:.2f}")
    
    # Visualize the trade
    st.write("---")
    st.write("### Trade Visualization")
    
    # Create price range for visualization
    price_range = np.linspace(signal.stop_loss * 0.8, signal.target_price * 1.2, 100)
    
    # Create a DataFrame for the price levels
    price_levels = pd.DataFrame({
        'Price': price_range,
        'Level': ['Price'] * len(price_range)
    })
    
    # Add entry, stop loss, and target price lines
    key_prices = pd.DataFrame({
        'Price': [signal.entry_price_range[0], signal.entry_price_range[1], signal.stop_loss, signal.target_price, signal.current_price],
        'Level': ['Entry Low', 'Entry High', 'Stop Loss', 'Target', 'Current']
    })
    
    # Create the plot
    fig = px.line(price_levels, x='Price', y='Level', color_discrete_sequence=['gray'])
    fig.add_scatter(x=key_prices['Price'], y=key_prices['Level'], mode='markers+text', 
                    text=key_prices['Level'], textposition='top center',
                    marker=dict(size=10, color=['blue', 'blue', 'red', 'green', 'purple']))
    
    # Update layout
    fig.update_layout(
        title=f"{signal.symbol} {signal.option_type.upper()} ${signal.strike} - Price Levels",
        xaxis_title="Option Price ($)",
        yaxis_visible=False,
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Sidebar
st.sidebar.title("Near-The-Money Options Analyzer")
st.sidebar.write("Find and analyze options trading opportunities near current market prices.")

# Watchlist management
st.sidebar.subheader("Watchlist")
watchlist_input = st.sidebar.text_input("Enter symbols (comma separated)", 
                                       ",".join(st.session_state.watchlist))
st.session_state.watchlist = [s.strip().upper() for s in watchlist_input.split(",") if s.strip()]

# Analysis parameters
st.sidebar.subheader("Analysis Parameters")
moneyness_range = st.sidebar.slider("Moneyness Range (%)", 5, 20, 10, 1) / 100
min_volume = st.sidebar.number_input("Minimum Volume", 10, 1000, 50, 10)
min_open_interest = st.sidebar.number_input("Minimum Open Interest", 10, 5000, 100, 10)
max_spread_pct = st.sidebar.slider("Maximum Bid-Ask Spread (%)", 1, 20, 10, 1) / 100
days_to_expiry_range = st.sidebar.slider("Days to Expiration", 1, 180, (14, 60), 1)

# Update analyzer config
st.session_state.analyzer.config.update({
    "moneyness_range": moneyness_range,
    "min_volume": min_volume,
    "min_open_interest": min_open_interest,
    "max_spread_pct": max_spread_pct,
    "days_to_expiry_min": days_to_expiry_range[0],
    "days_to_expiry_max": days_to_expiry_range[1]
})

# Custom strike analysis
st.sidebar.subheader("Custom Strike Analysis")
selected_symbol = st.sidebar.selectbox("Select Symbol", st.session_state.watchlist)
use_custom_strikes = st.sidebar.checkbox("Use Custom Strikes")

if use_custom_strikes:
    current_price = st.session_state.analyzer.market_data.get_price(selected_symbol)
    st.sidebar.write(f"Current price: ${current_price:.2f}")
    
    custom_strikes_input = st.sidebar.text_input(
        "Enter strikes (comma separated)",
        ",".join(map(str, st.session_state.custom_strikes.get(selected_symbol, 
                                   st.session_state.analyzer._generate_strikes_around_price(current_price))))
    )
    
    try:
        st.session_state.custom_strikes[selected_symbol] = [float(s.strip()) for s in custom_strikes_input.split(",") if s.strip()]
    except ValueError:
        st.sidebar.error("Invalid strike price format. Use numbers separated by commas.")
    
    # Get available expirations
    options_chain = st.session_state.analyzer.market_data.get_options_chain(selected_symbol)
    available_expirations = list(options_chain.keys())
    
    if available_expirations:
        # Sort by date
        available_expirations.sort()
        
        # Default to a date 30-45 days out if possible
        today = datetime.datetime.now().date()
        default_idx = 0
        
        for i, exp in enumerate(available_expirations):
            exp_date = datetime.datetime.strptime(exp, "%Y-%m-%d").date()
            days = (exp_date - today).days
            if 30 <= days <= 45:
                default_idx = i
                break
        
        selected_expiration = st.sidebar.selectbox(
            "Select Expiration",
            available_expirations,
            index=min(default_idx, len(available_expirations)-1)
        )
        
        st.session_state.selected_expiration = selected_expiration

# Action buttons
st.sidebar.subheader("Actions")
refresh_button = st.sidebar.button("Refresh Analysis")
auto_refresh = st.sidebar.checkbox("Auto-refresh (1 min)", value=False)

# Main content
st.title("Near-The-Money Options Analysis Dashboard")

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["Signal Scanner", "Custom Analysis", "Market Overview"])

# Check if we need to refresh
current_time = time.time()
should_refresh = refresh_button or (
    auto_refresh and 
    (st.session_state.last_refresh is None or 
     current_time - st.session_state.last_refresh > 60)
)

# Perform analysis if needed
if should_refresh:
    with st.spinner("Analyzing options..."):
        st.session_state.signals = st.session_state.analyzer.get_option_signals(
            st.session_state.watchlist, 
            max_per_symbol=3
        )
        st.session_state.last_refresh = current_time

# Signal Scanner tab
with tab1:
    st.header("Options Trading Signals")
    
    if not st.session_state.signals:
        st.info("No signals available. Click 'Refresh Analysis' to scan for options.")
    else:
        # Convert to DataFrame
        signals_df = signals_to_dataframe(st.session_state.signals)
        
        # Display as table
        st.dataframe(
            signals_df.style.applymap(color_score, subset=["Score"]),
            height=400
        )
        
        # Display selected signal details
        st.subheader("Signal Details")
        selected_indices = []
        
        try:
            selected_rows = st.multiselect(
                "Select signals to view details",
                options=list(range(len(st.session_state.signals))),
                format_func=lambda i: (
                    f"{st.session_state.signals[i].symbol} "
                    f"{st.session_state.signals[i].option_type.upper()} "
                    f"${st.session_state.signals[i].strike} "
                    f"({st.session_state.signals[i].signal_strength:.2f})"
                )
            )
            
            for i in selected_rows:
                st.write("---")
                display_option_details(st.session_state.signals[i])
        except Exception as e:
            st.error(f"Error displaying signal details: {e}")

# Custom Analysis tab
with tab2:
    st.header("Custom Strike Analysis")
    
    if use_custom_strikes and selected_symbol and st.session_state.selected_expiration:
        with st.spinner(f"Analyzing {selected_symbol} options..."):
            # Get options for custom strikes
            custom_strikes = st.session_state.custom_strikes.get(selected_symbol, [])
            
            if not custom_strikes:
                st.warning("No custom strikes defined. Please enter strike prices in the sidebar.")
            else:
                # Get options data
                options = st.session_state.analyzer.get_options_by_strike_range(
                    selected_symbol, 
                    custom_strikes,
                    st.session_state.selected_expiration
                )
                
                # Create signals for display
                call_signals = []
                put_signals = []
                
                for option_type in ["calls", "puts"]:
                    for strike_str, option_data in options[option_type].items():
                        # Get current price
                        price = option_data.get("underlying_price")
                        
                        # Score this option
                        score = st.session_state.analyzer.score_option(
                            option_data, 
                            price, 
                            option_type[:-1]  # Remove 's' from calls/puts
                        )
                        
                        # Create signal object
                        try:
                            strike = float(strike_str)
                            expiry_date = datetime.datetime.strptime(
                                st.session_state.selected_expiration, 
                                "%Y-%m-%d"
                            ).date()
                            
                            mid_price = (option_data["bid"] + option_data["ask"]) / 2
                            signal = OptionSignal(
                                symbol=selected_symbol,
                                option_type=option_type[:-1],  # Remove 's' from calls/puts
                                strike=strike,
                                expiration=expiry_date,
                                current_price=mid_price,
                                underlying_price=price,
                                entry_price_range=(option_data["bid"], option_data["ask"]),
                                stop_loss=mid_price * 0.9,
                                target_price=mid_price * 1.2,
                                signal_strength=score,
                                volume=option_data.get("volume", 0),
                                open_interest=option_data.get("open_interest", 0),
                                iv=option_data.get("iv", 30.0),
                                delta=option_data.get("delta", 0.5)
                            )
                            
                            if option_type == "calls":
                                call_signals.append(signal)
                            else:
                                put_signals.append(signal)
                        except Exception as e:
                            st.error(f"Error creating signal for {strike_str}: {e}")
                
                # Display calls
                st.subheader(f"{selected_symbol} Calls - {st.session_state.selected_expiration}")
                
                if call_signals:
                    calls_df = signals_to_dataframe(call_signals)
                    st.dataframe(
                        calls_df.style.applymap(color_score, subset=["Score"]),
                        height=300
                    )
                    
                    # Allow selecting a call for detailed view
                    selected_call = st.selectbox(
                        "Select call option for details",
                        options=range(len(call_signals)),
                        format_func=lambda i: f"${call_signals[i].strike} - Score: {call_signals[i].signal_strength:.2f}"
                    )
                    
                    if selected_call is not None:
                        st.write("---")
                        display_option_details(call_signals[selected_call])
                else:
                    st.info(f"No call options found for {selected_symbol} at the specified strikes.")
                
                # Display puts
                st.subheader(f"{selected_symbol} Puts - {st.session_state.selected_expiration}")
                
                if put_signals:
                    puts_df = signals_to_dataframe(put_signals)
                    st.dataframe(
                        puts_df.style.applymap(color_score, subset=["Score"]),
                        height=300
                    )
                    
                    # Allow selecting a put for detailed view
                    selected_put = st.selectbox(
                        "Select put option for details",
                        options=range(len(put_signals)),
                        format_func=lambda i: f"${put_signals[i].strike} - Score: {put_signals[i].signal_strength:.2f}"
                    )
                    
                    if selected_put is not None:
                        st.write("---")
                        display_option_details(put_signals[selected_put])
                else:
                    st.info(f"No put options found for {selected_symbol} at the specified strikes.")
    else:
        st.info("Enable 'Use Custom Strikes' in the sidebar to analyze specific option contracts.")

# Market Overview tab
with tab3:
    st.header("Market Overview")
    
    # Get market data for watchlist symbols
    with st.spinner("Fetching market data..."):
        market_data = []
        
        for symbol in st.session_state.watchlist:
            try:
                price = st.session_state.analyzer.market_data.get_price(symbol)
                
                # Get additional data if available
                additional_data = {}
                try:
                    # This would ideally come from your market data provider
                    # For now, we'll use placeholder data
                    additional_data = {
                        "change_pct": np.random.uniform(-2, 2),  # Simulated daily change
                        "volume": int(np.random.uniform(1000000, 10000000)),  # Simulated volume
                        "rsi": np.random.uniform(30, 70)  # Simulated RSI
                    }
                except Exception:
                    pass
                
                market_data.append({
                    "Symbol": symbol,
                    "Price": price,
                    "Change %": additional_data.get("change_pct", 0),
                    "Volume": additional_data.get("volume", 0),
                    "RSI": additional_data.get("rsi", 50)
                })
            except Exception as e:
                st.error(f"Error fetching data for {symbol}: {e}")
        
        # Display market data
        if market_data:
            market_df = pd.DataFrame(market_data)
            
            # Color formatting for Change %
            def color_change(val):
                if val > 0:
                    return f'background-color: green; color: white'
                elif val < 0:
                    return f'background-color: red; color: white'
                return ''
            
            # Color formatting for RSI
            def color_rsi(val):
                if val >= 70:
                    return f'background-color: red; color: white'
                elif val <= 30:
                    return f'background-color: green; color: white'
                return ''
            
            # Display styled dataframe
            st.dataframe(
                market_df.style
                    .format({
                        "Price": "${:.2f}",
                        "Change %": "{:.2f}%",
                        "Volume": "{:,.0f}",
                        "RSI": "{:.1f}"
                    })
                    .applymap(color_change, subset=["Change %"])
                    .applymap(color_rsi, subset=["RSI"]),
                height=400
            )
        else:
            st.warning("No market data available.")

# Footer
st.write("---")
st.write("### About This Dashboard")
st.write("""
This dashboard uses the Near-The-Money Options Analyzer to find and evaluate options 
trading opportunities. It focuses on options that are close to the current market price 
and have good liquidity, volume, and trading potential.

The analyzer scores options based on:
- Volume and open interest
- Bid-ask spread (liquidity)
- Delta (proximity to at-the-money)
- Implied volatility
- And other factors

For best results, focus on options with a score of 0.5 or higher.
""")

# Show last refresh time
if st.session_state.last_refresh:
    last_refresh_time = datetime.datetime.fromtimestamp(st.session_state.last_refresh).strftime('%Y-%m-%d %H:%M:%S')
    st.write(f"Last refreshed: {last_refresh_time}")

# Run the dashboard with: streamlit run ntm_options_dashboard.py 