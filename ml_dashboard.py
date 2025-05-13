#!/usr/bin/env python
"""
ML-Enhanced Options Trading Dashboard

A Streamlit dashboard that integrates the machine learning model with the options trading interface.
This dashboard shows enhanced signals from the ML model along with the ability to simulate trades.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import time
import os
import sys
from typing import List, Dict, Tuple, Union, Optional

# Import our custom modules
from model_integrator import EnhancedOptionsTrader
from smart_trader import SmartTrader
from config import Config

# Set page config
st.set_page_config(
    page_title="ML-Enhanced Options Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'ml_trader' not in st.session_state:
    st.session_state.ml_trader = None
if 'smart_trader' not in st.session_state:
    st.session_state.smart_trader = SmartTrader()
if 'trades' not in st.session_state:
    st.session_state.trades = []
if 'trade_history' not in st.session_state:
    st.session_state.trade_history = []
if 'account_value' not in st.session_state:
    st.session_state.account_value = 1000.0
if 'initial_account_value' not in st.session_state:
    st.session_state.initial_account_value = 1000.0

# Header with logo and title
st.title("🤖 ML-Enhanced Options Trading Dashboard")
st.markdown("***Powered by Real Market Data & Machine Learning***")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    
    # Account settings
    st.subheader("Account")
    st.session_state.account_value = st.number_input(
        "Account Value ($)",
        min_value=1000.0,
        max_value=10000000.0,
        value=st.session_state.account_value,
        step=1000.0,
        help="Your account value for trade sizing"
    )
    
    # Risk settings
    st.subheader("Risk Management")
    risk_per_trade = st.slider(
        "Risk Per Trade (%)",
        min_value=0.5,
        max_value=5.0,
        value=2.0,
        step=0.1,
        help="Maximum percentage of account to risk on a single trade"
    )
    
    # Strategy settings
    st.subheader("Strategy Settings")
    min_ml_probability = st.slider(
        "Min ML Probability (%)",
        min_value=30,
        max_value=90,
        value=40,
        step=5,
        help="Minimum ML probability threshold for trade signals"
    )
    
    st.divider()
    
    # Initialize ML Trader
    if st.session_state.ml_trader is None or st.button("Initialize ML Trader"):
        with st.spinner("Initializing ML Trading Engine..."):
            try:
                st.session_state.ml_trader = EnhancedOptionsTrader()
                st.success("ML Trading Engine initialized!")
            except Exception as e:
                st.error(f"Error initializing ML Trading Engine: {str(e)}")
    
    # Reset button
    if st.button("Reset Dashboard"):
        st.session_state.signals = []
        st.session_state.trades = []
        st.session_state.trade_history = []
        st.success("Dashboard reset!")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    # Symbol input
    st.subheader("Scan for ML-Enhanced Trading Signals")
    
    # Symbol selection
    symbols_text = st.text_area(
        "Enter Symbols (comma separated)",
        "SPY, QQQ, AAPL, MSFT, NVDA, AMZN, TSLA, META",
        help="Enter stock symbols to scan"
    )
    symbols = [s.strip().upper() for s in symbols_text.split(",") if s.strip()]
    
    # Scan controls
    col1a, col1b, col1c = st.columns(3)
    with col1a:
        expiration_days = st.number_input(
            "Days to Expiration",
            min_value=1,
            max_value=365,
            value=30,
            step=1,
            help="Target days to expiration"
        )
    with col1b:
        max_signals = st.number_input(
            "Max Signals",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Maximum number of signals to display"
        )
    with col1c:
        # Scan button
        scan_clicked = st.button("🔍 Scan Market", type="primary")
    
    # Handle scan button click
    if scan_clicked and st.session_state.ml_trader:
        with st.spinner("Scanning market for ML-enhanced trading signals..."):
            try:
                # Get signals from ML trader
                st.session_state.signals = st.session_state.ml_trader.get_enhanced_signals(
                    symbols, 
                    expiration_days=expiration_days,
                    max_signals=max_signals,
                    min_probability=min_ml_probability/100.0
                )
                
                if not st.session_state.signals:
                    st.warning("No trading signals found. Try different symbols or lower the ML probability threshold.")
            except Exception as e:
                st.error(f"Error getting signals: {str(e)}")
    
    # Display signals if available
    if st.session_state.signals:
        st.subheader(f"Found {len(st.session_state.signals)} Enhanced Trading Signals")
        
        # Create a dataframe for the signals
        signals_data = []
        for signal in st.session_state.signals:
            signals_data.append({
                "Symbol": f"{signal['symbol']} {signal['option_type'].upper()} ${signal['strike']}",
                "ML Score": f"{signal['ml_score']:.2f}",
                "ML Probability": f"{signal['ml_probability']*100:.1f}%",
                "Signal Strength": f"{signal['signal_strength']:.2f}",
                "Current Price": f"${signal['current_price']:.2f}",
                "Underlying": f"${signal['underlying_price']:.2f}",
                "Expiration": signal['expiration']
            })
        
        # Display as table
        signals_df = pd.DataFrame(signals_data)
        st.dataframe(signals_df, hide_index=True, use_container_width=True)
        
        # Signal details and trade simulation
        st.subheader("Signal Details & Trade Simulation")
        
        for i, signal in enumerate(st.session_state.signals):
            with st.expander(f"{signal['symbol']} {signal['option_type'].upper()} ${signal['strike']} - ML Score: {signal['ml_score']:.2f}"):
                st.write(f"**ML Probability:** {signal['ml_probability']*100:.1f}%")
                st.write(f"**Signal Strength:** {signal['signal_strength']:.2f}")
                st.write(f"**Underlying Price:** ${signal['underlying_price']:.2f}")
                st.write(f"**Option Price:** ${signal['current_price']:.2f}")
                st.write(f"**Entry Range:** ${signal['entry_price_range'][0]:.2f} - ${signal['entry_price_range'][1]:.2f}")
                st.write(f"**Stop Loss:** ${signal['stop_loss']:.2f} (-{Config.STOP_LOSS_PCT*100:.1f}%)")
                st.write(f"**Target Price:** ${signal['target_price']:.2f} (+{Config.PROFIT_TARGET_PCT*100:.1f}%)")
                st.write(f"**Expiration:** {signal['expiration']}")
                
                if st.button(f"Simulate Trade #{i+1}", key=f"sim_{i}"):
                    with st.spinner("Simulating trade..."):
                        # Calculate position size based on risk
                        account_risk_amount = st.session_state.account_value * (risk_per_trade / 100)
                        option_price = signal['current_price']
                        contracts = max(1, int(account_risk_amount / (option_price * 100)))
                        
                        # Simulate the trade
                        trade_info = {
                            "symbol": signal['symbol'],
                            "option_type": signal['option_type'],
                            "strike": signal['strike'],
                            "expiry": signal['expiration'],
                            "price": signal['underlying_price'],
                            "bid": signal['entry_price_range'][0],
                            "ask": signal['entry_price_range'][1],
                            "mid": signal['current_price'],
                            "ml_probability": signal['ml_probability'],
                            "ml_score": signal['ml_score'],
                            "entry_time": datetime.now().strftime("%H:%M:%S"),
                            "contracts": contracts
                        }
                        
                        # Simulate the outcome
                        outcome = st.session_state.smart_trader.simulate_trade(
                            trade_info, 
                            risk_pct=risk_per_trade/100,
                            account_value=st.session_state.account_value
                        )
                        
                        # Update account value
                        st.session_state.account_value += outcome['pnl']
                        
                        # Add to trade history
                        st.session_state.trade_history.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "symbol": f"{signal['symbol']} {signal['option_type'].upper()} ${signal['strike']}",
                            "contracts": contracts,
                            "entry": option_price,
                            "exit": outcome['exit_price'],
                            "pnl": outcome['pnl'],
                            "return_pct": outcome['pnl'] / (option_price * contracts * 100) * 100,
                            "ml_probability": signal['ml_probability'],
                            "outcome": outcome['outcome']
                        })
                        
                        # Add current trade to active trades
                        st.session_state.trades.append({
                            "id": len(st.session_state.trades),
                            "symbol": f"{signal['symbol']} {signal['option_type'].upper()} ${signal['strike']}",
                            "contracts": contracts,
                            "entry_price": option_price,
                            "current_price": outcome['exit_price'],
                            "pnl": outcome['pnl'],
                            "return_pct": outcome['pnl'] / (option_price * contracts * 100) * 100,
                            "status": "Closed",
                            "ml_probability": signal['ml_probability'],
                            "outcome": outcome['outcome']
                        })
    else:
        st.info("Click 'Scan Market' to find ML-enhanced trading opportunities.")

with col2:
    # Account summary
    st.subheader("Account Summary")
    
    # Calculate performance metrics
    account_change = ((st.session_state.account_value / st.session_state.initial_account_value) - 1) * 100
    
    # Display metrics
    col2a, col2b = st.columns(2)
    col2a.metric("Account Value", f"${st.session_state.account_value:.2f}", f"{account_change:.2f}%")
    
    # Calculate win rate if trades exist
    win_rate = 0
    if st.session_state.trade_history:
        wins = sum(1 for trade in st.session_state.trade_history if trade['pnl'] > 0)
        win_rate = wins / len(st.session_state.trade_history) * 100
    
    col2b.metric("Win Rate", f"{win_rate:.1f}%")
    
    # Display recent trades
    st.subheader("Recent Trades")
    if st.session_state.trades:
        trades_data = []
        for trade in st.session_state.trades[-5:]:  # Show last 5 trades
            trades_data.append({
                "Symbol": trade['symbol'],
                "Contracts": trade['contracts'],
                "P&L": f"${trade['pnl']:.2f}",
                "Return": f"{trade['return_pct']:.1f}%",
                "ML Prob": f"{trade['ml_probability']*100:.1f}%",
                "Outcome": trade['outcome']
            })
        
        # Display as table
        trades_df = pd.DataFrame(trades_data)
        st.dataframe(trades_df, hide_index=True, use_container_width=True)
    else:
        st.info("No trades yet. Simulate a trade to see results.")
    
    # Performance charts
    st.subheader("Performance Analytics")
    
    if st.session_state.trade_history:
        # Prepare data for charts
        trade_returns = [trade['return_pct'] for trade in st.session_state.trade_history]
        trade_ml_probs = [trade['ml_probability']*100 for trade in st.session_state.trade_history]
        trade_outcomes = [trade['outcome'] for trade in st.session_state.trade_history]
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Plot trade returns
        ax1.bar(range(len(trade_returns)), trade_returns, color=['green' if r > 0 else 'red' for r in trade_returns])
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.set_title('Trade Returns (%)')
        ax1.set_xlabel('Trade #')
        ax1.set_ylabel('Return (%)')
        
        # Plot ML probability vs outcome
        colors = ['green' if outcome in ['PROFIT', 'PARTIAL PROFIT'] else 'red' for outcome in trade_outcomes]
        ax2.scatter(trade_ml_probs, trade_returns, c=colors, alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.set_title('ML Probability vs Return')
        ax2.set_xlabel('ML Probability (%)')
        ax2.set_ylabel('Return (%)')
        
        # Adjust layout and display
        plt.tight_layout()
        st.pyplot(fig)
        
        # Show trade distribution
        outcomes = pd.Series(trade_outcomes).value_counts()
        
        # Plot pie chart of outcomes
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['green', 'lightgreen', 'red', 'darkred']
        ax.pie(outcomes, labels=outcomes.index, autopct='%1.1f%%', colors=colors)
        ax.set_title('Trade Outcome Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Trade analytics will appear here after simulating trades.")

# Footer
st.divider()
st.markdown("*This dashboard uses real market data combined with machine learning to enhance options trading signals.*")
st.markdown("*DISCLAIMER: Simulated results are for educational purposes only. Past performance is not indicative of future results.*")

if __name__ == "__main__":
    # This will only execute when the script is run directly
    print("ML-Enhanced Options Trading Dashboard started") 