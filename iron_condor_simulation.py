#!/usr/bin/env python
"""
Iron Condor Strategy Simulation

This script provides a simplified simulation showing the P&L outcomes
of an iron condor options strategy at expiration under different
market scenarios.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Define the iron condor parameters
underlying_price = 450.0
put_short_strike = 428.0
put_long_strike = 418.0
call_short_strike = 472.0
call_long_strike = 482.0

# Option prices
put_short_price = 3.50
put_long_price = 2.20
call_short_price = 3.25
call_long_price = 2.00

# Calculate net credit
net_credit = put_short_price - put_long_price + call_short_price - call_long_price
max_profit = net_credit * 100  # Per contract
max_loss = ((put_short_strike - put_long_strike) - net_credit) * 100  # Per contract (same width on both spreads)

# Print trade details
print(f"Iron Condor Strategy on SPY (Current Price: ${underlying_price:.2f})")
print(f"------------------------------------------------")
print(f"Short Put: Strike ${put_short_strike:.2f} @ ${put_short_price:.2f}")
print(f"Long Put: Strike ${put_long_strike:.2f} @ ${put_long_price:.2f}")
print(f"Short Call: Strike ${call_short_strike:.2f} @ ${call_short_price:.2f}")
print(f"Long Call: Strike ${call_long_strike:.2f} @ ${call_long_price:.2f}")
print(f"------------------------------------------------")
print(f"Net Credit: ${net_credit:.2f} per contract (${net_credit * 100:.2f} total)")
print(f"Max Profit: ${max_profit:.2f}")
print(f"Max Loss: ${max_loss:.2f}")
print(f"Breakeven points: ${put_short_strike - net_credit:.2f} and ${call_short_strike + net_credit:.2f}")
print(f"------------------------------------------------")

# Calculate P&L at expiration for a range of underlying prices
price_range = np.arange(390, 510, 1)
pnl = np.zeros_like(price_range, dtype=float)

for i, price in enumerate(price_range):
    # Calculate payoff of each leg at expiration
    put_short_payoff = -max(0, put_short_strike - price)  # Short put (negative because we sold it)
    put_long_payoff = max(0, put_long_strike - price)     # Long put
    call_short_payoff = -max(0, price - call_short_strike)  # Short call
    call_long_payoff = max(0, price - call_long_strike)     # Long call
    
    # Calculate total payoff including the initial credit received
    total_payoff = put_short_payoff + put_long_payoff + call_short_payoff + call_long_payoff
    pnl[i] = (total_payoff + net_credit) * 100  # Per contract, in dollars

# Create a DataFrame for analysis
results = pd.DataFrame({
    'Underlying Price': price_range,
    'P&L': pnl
})

# Simulate different price scenarios
scenarios = [
    {"name": "Flat Market", "final_price": 450.0},
    {"name": "Moderate Bull", "final_price": 465.0},
    {"name": "Strong Bull", "final_price": 480.0},
    {"name": "Moderate Bear", "final_price": 435.0},
    {"name": "Strong Bear", "final_price": 415.0}
]

print("\nP&L Outcomes for Different Market Scenarios at Expiration:")
print("------------------------------------------------")
for scenario in scenarios:
    idx = np.abs(price_range - scenario["final_price"]).argmin()
    final_pnl = pnl[idx]
    pct_of_max = (final_pnl / max_profit) * 100 if final_pnl > 0 else (final_pnl / max_loss) * 100
    
    print(f"{scenario['name']} (SPY @ ${scenario['final_price']:.2f}): " 
          f"${final_pnl:.2f} ({pct_of_max:.1f}% of max {'profit' if final_pnl > 0 else 'loss'})")

# Plot the P&L diagram
plt.figure(figsize=(12, 8))
plt.plot(price_range, pnl, 'b-', linewidth=2)
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.axhline(y=max_profit, color='g', linestyle='--', alpha=0.5, label=f'Max Profit: ${max_profit:.2f}')
plt.axhline(y=-max_loss, color='r', linestyle='--', alpha=0.5, label=f'Max Loss: ${-max_loss:.2f}')
plt.axvline(x=underlying_price, color='k', linestyle='--', alpha=0.5, label=f'Current Price: ${underlying_price:.2f}')

# Add vertical lines for key price points
plt.axvline(x=put_long_strike, color='b', linestyle=':', alpha=0.5, label=f'Put Long Strike: ${put_long_strike:.2f}')
plt.axvline(x=put_short_strike, color='b', linestyle=':', alpha=0.5, label=f'Put Short Strike: ${put_short_strike:.2f}')
plt.axvline(x=call_short_strike, color='b', linestyle=':', alpha=0.5, label=f'Call Short Strike: ${call_short_strike:.2f}')
plt.axvline(x=call_long_strike, color='b', linestyle=':', alpha=0.5, label=f'Call Long Strike: ${call_long_strike:.2f}')

# Mark scenarios on the graph
for scenario in scenarios:
    idx = np.abs(price_range - scenario["final_price"]).argmin()
    plt.plot(scenario["final_price"], pnl[idx], 'ro', markersize=8)
    plt.annotate(scenario["name"], 
                xy=(scenario["final_price"], pnl[idx]),
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontweight='bold')

plt.grid(True, alpha=0.3)
plt.xlabel('SPY Price at Expiration', fontsize=12)
plt.ylabel('Profit/Loss ($)', fontsize=12)
plt.title('Iron Condor P&L at Expiration', fontsize=16)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
plt.tight_layout()

# Save the plot
plt.savefig('iron_condor_pnl.png')
print("\nP&L diagram saved as 'iron_condor_pnl.png'")

# Attempt to display the plot (may not work in all environments)
try:
    plt.show()
except:
    pass 