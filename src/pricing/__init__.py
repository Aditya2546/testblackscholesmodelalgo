"""
Options Pricing Module

This module provides a unified interface for pricing various types of financial options.
"""

from typing import Union, Dict, Any, Optional, Literal
import numpy as np

from src.pricing.black_scholes import BlackScholes, BlackScholesInputs, ArrayLike
from src.pricing.implied_volatility import ImpliedVolatilitySolver

# Define option styles
OptionStyle = Literal["european"]


def price_option(
    contract: Dict[str, Any],
    model_override: Optional[str] = None,
    steps: int = 100
) -> Dict[str, ArrayLike]:
    """
    Price an option contract, automatically selecting the appropriate pricing model.
    
    This function uses the Black-Scholes model for European options.
    
    Args:
        contract: Dictionary containing option contract details:
            - S: Underlying price
            - K: Strike price
            - T: Time to expiration (in years)
            - r: Risk-free rate (annualized)
            - sigma: Volatility (annualized)
            - option_type: "call" or "put"
            - q: Dividend yield (annualized)
            - style: "european"
        model_override: Optional specific model to use:
            - "black_scholes" - Black-Scholes model (European)
            - None - Auto-select model (default)
        steps: Number of steps for tree-based models (not used for European options)
    
    Returns:
        Dictionary containing pricing results including:
            - price: Option price
            - greeks: Option Greeks (if calculated)
            - model: Name of the model used
    """
    # Extract contract details
    S = contract["S"]
    K = contract["K"]
    T = contract["T"]
    r = contract["r"]
    sigma = contract["sigma"]
    option_type = contract["option_type"]
    q = contract.get("q", 0.0)  # Default dividend yield to 0
    style = contract.get("style", "european")  # Default to European
    
    # Ensure we're using European style
    if style != "european" and model_override != "black_scholes":
        raise ValueError(f"Only European style options are supported. Got {style}")
        
    # Use Black-Scholes for European options
    inputs = BlackScholesInputs(
        S=S,
        K=K,
        T=T,
        r=r,
        sigma=sigma,
        option_type=option_type,
        q=q
    )
    
    # Calculate price and Greeks
    greeks = BlackScholes.all_greeks(inputs)
    
    return {
        "price": greeks["price"],
        "greeks": {k: v for k, v in greeks.items() if k != "price"},
        "model": "black_scholes"
    }


def implied_volatility(contract: Dict[str, Any], market_price: ArrayLike) -> ArrayLike:
    """
    Calculate implied volatility from market prices for European options.
    
    This function works with European options using the Black-Scholes model.
    
    Args:
        contract: Dictionary containing option contract details (without sigma):
            - S: Underlying price
            - K: Strike price
            - T: Time to expiration (in years)
            - r: Risk-free rate (annualized)
            - option_type: "call" or "put"
            - q: Dividend yield (annualized) (optional)
            - style: "european" (optional, defaults to "european")
        market_price: Market price of the option
        
    Returns:
        Implied volatility
    """
    return ImpliedVolatilitySolver.calculate(contract, market_price)