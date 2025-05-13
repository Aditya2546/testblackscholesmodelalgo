"""
Implied Volatility Solver

This module provides implied volatility calculation for European options.
It uses an efficient hybrid approach combining Newton-Raphson and bisection methods.
"""

import numpy as np
from typing import Union, Tuple, Dict, Optional, Any, Callable

from src.pricing.black_scholes import BlackScholesInputs, BlackScholes, ArrayLike, OptionType


class ImpliedVolatilitySolver:
    """
    Unified implied volatility calculator for European options.
    
    This class efficiently computes implied volatility from market prices using
    a hybrid approach that combines Newton-Raphson for speed with bisection for robustness.
    """
    
    # Constants for algorithm control
    MAX_ITERATIONS = 15
    PRECISION = 1e-8
    
    @staticmethod
    def _initial_guess(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                      market_price: ArrayLike, option_type: Union[OptionType, np.ndarray]) -> ArrayLike:
        """
        Generate initial volatility guess based on option moneyness and time.
        
        Uses Brenner-Subrahmanyam (1988) approximation for ATM options,
        then adjusts based on option moneyness.
        """
        # Start with Brenner-Subrahmanyam approximation for ATM options
        moneyness = S / K
        atm_vol = np.sqrt(2 * np.pi / T) * market_price / S
        
        # Adjust for moneyness - higher vol for OTM options, lower for ITM
        if isinstance(option_type, str):
            if option_type == "call":
                return np.where(moneyness < 0.8, atm_vol * 1.5,
                       np.where(moneyness > 1.2, atm_vol * 1.3, atm_vol))
            else:  # put
                return np.where(moneyness > 1.2, atm_vol * 1.5,
                       np.where(moneyness < 0.8, atm_vol * 1.3, atm_vol))
        else:
            # Vectorized for array of option types
            call_mask = (option_type == "call")
            call_adjustment = np.where(moneyness < 0.8, atm_vol * 1.5,
                               np.where(moneyness > 1.2, atm_vol * 1.3, atm_vol))
            put_adjustment = np.where(moneyness > 1.2, atm_vol * 1.5,
                              np.where(moneyness < 0.8, atm_vol * 1.3, atm_vol))
            
            return np.where(call_mask, call_adjustment, put_adjustment)
    
    @classmethod
    def newton_raphson(cls, contract: Dict[str, Any], 
                      market_price: ArrayLike, 
                      pricer: Callable) -> Tuple[ArrayLike, ArrayLike]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            contract: Option contract dictionary (without sigma)
            market_price: Market price of the option
            pricer: Pricing function to use
            
        Returns:
            Tuple of (implied volatility, number of iterations to converge)
        """
        # Extract contract details
        S = contract["S"]
        K = contract["K"]
        T = contract["T"]
        r = contract["r"]
        option_type = contract["option_type"]
        q = contract.get("q", 0.0)
        
        # Generate initial guess
        sigma = cls._initial_guess(S, K, T, market_price, option_type)
        
        # Initialize variables for tracking iterations
        iterations = np.zeros_like(sigma, dtype=int)
        converged = np.zeros_like(sigma, dtype=bool)
        
        # Main Newton-Raphson loop
        for i in range(cls.MAX_ITERATIONS):
            # Create contract with current sigma
            current_contract = {**contract, "sigma": sigma}
            
            # Calculate option price
            result = pricer(current_contract)
            price = result["price"]
            
            # Calculate price difference
            diff = price - market_price
            
            # Check for convergence
            new_converged = np.abs(diff) < cls.PRECISION
            if np.all(new_converged):
                break
                
            # Update iterations for newly converged values
            iterations = np.where(new_converged & ~converged, i, iterations)
            converged = converged | new_converged
            
            # Calculate vega using finite differences
            delta_sigma = 0.001  # Small increment for numerical derivative
            up_contract = {**contract, "sigma": sigma + delta_sigma}
            up_price = pricer(up_contract)["price"]
            vega = (up_price - price) / delta_sigma
            
            # Handle near-zero vega case to avoid numerical issues
            vega = np.where(np.abs(vega) < 1e-10, 1e-10, vega)
            sigma_update = diff / vega
            
            # Apply update only to non-converged values
            sigma = np.where(converged, sigma, sigma - sigma_update)
            
            # Ensure sigma stays positive and reasonable
            sigma = np.maximum(sigma, 1e-8)
            sigma = np.minimum(sigma, 5.0)  # Cap at 500% volatility
            
        # Update iterations for any values that didn't converge
        iterations = np.where(~converged, cls.MAX_ITERATIONS, iterations)
        
        return sigma, iterations

    @classmethod
    def bisection(cls, contract: Dict[str, Any], 
                market_price: ArrayLike, 
                pricer: Callable) -> ArrayLike:
        """
        Fallback method using bisection when Newton-Raphson fails to converge.
        
        This is more robust but slower than Newton-Raphson.
        
        Args:
            contract: Option contract dictionary (without sigma)
            market_price: Market price of the option
            pricer: Pricing function to use
            
        Returns:
            Implied volatility
        """
        # Extract contract details
        S = contract["S"]
        
        # Initial bounds for volatility
        sigma_low = np.full_like(S, 0.001, dtype=float)
        sigma_high = np.full_like(S, 5.0, dtype=float)  # 500% volatility upper bound
        
        # Price at bounds
        low_contract = {**contract, "sigma": sigma_low}
        high_contract = {**contract, "sigma": sigma_high}
        price_low = pricer(low_contract)["price"]
        price_high = pricer(high_contract)["price"]
        
        # Initialize result array
        sigma = (sigma_low + sigma_high) / 2
        
        # Main bisection loop
        for _ in range(cls.MAX_ITERATIONS * 2):  # More iterations for bisection
            # Update middle point
            sigma = (sigma_low + sigma_high) / 2
            
            # Calculate price at middle point
            mid_contract = {**contract, "sigma": sigma}
            price_mid = pricer(mid_contract)["price"]
            
            # Calculate price difference
            diff = price_mid - market_price
            
            # Check for convergence
            if np.all(np.abs(diff) < cls.PRECISION):
                break
                
            # Update bounds
            sigma_low = np.where((diff * (price_low - market_price)) > 0, sigma, sigma_low)
            sigma_high = np.where((diff * (price_high - market_price)) > 0, sigma, sigma_high)
            
            # Update prices at bounds if needed
            mask_low = (sigma_low != low_contract["sigma"])
            if np.any(mask_low):
                low_contract = {**contract, "sigma": sigma_low}
                price_low = pricer(low_contract)["price"]
                
            mask_high = (sigma_high != high_contract["sigma"])
            if np.any(mask_high):
                high_contract = {**contract, "sigma": sigma_high}
                price_high = pricer(high_contract)["price"]
        
        return sigma
    
    @classmethod
    def calculate(cls, contract: Dict[str, Any], market_price: ArrayLike) -> ArrayLike:
        """
        Calculate implied volatility for European options.
        
        This method intelligently chooses between Newton-Raphson for speed and
        bisection for robustness, ensuring accurate results across a wide range
        of market conditions.
        
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
        # Detect extreme cases where intrinsic value exceeds market price
        S = contract["S"]
        K = contract["K"]
        option_type = contract["option_type"]
        
        # Determine if it's a call or put option
        is_call = True if option_type == "call" else False
        
        # Calculate intrinsic value
        if is_call:
            intrinsic = np.maximum(S - K, 0)
        else:
            intrinsic = np.maximum(K - S, 0)
        
        # Check if market price is below intrinsic value (arbitrage opportunity)
        if np.any(market_price < intrinsic):
            raise ValueError("Market price below intrinsic value - arbitrage opportunity")
        
        # Define pricing function based on option style
        def pricer(contract_with_sigma):
            # Use Black-Scholes model for pricing
            inputs = BlackScholesInputs(
                S=contract_with_sigma["S"],
                K=contract_with_sigma["K"],
                T=contract_with_sigma["T"],
                r=contract_with_sigma["r"],
                sigma=contract_with_sigma["sigma"],
                option_type=contract_with_sigma["option_type"],
                q=contract_with_sigma.get("q", 0.0)
            )
            
            # Calculate price and Greeks
            result = BlackScholes.all_greeks(inputs)
            
            return {
                "price": result["price"],
                "model": "black_scholes"
            }
        
        # Try Newton-Raphson method first
        try:
            sigma, iterations = cls.newton_raphson(contract, market_price, pricer)
            
            # If Newton-Raphson failed to converge for some values, use bisection for those
            if np.any(iterations >= cls.MAX_ITERATIONS):
                # Only use bisection for values that didn't converge
                # But since we can't easily mix the methods for specific elements,
                # we'll use bisection for all if any failed
                if np.isscalar(market_price) and iterations >= cls.MAX_ITERATIONS:
                    return cls.bisection(contract, market_price, pricer)
                elif not np.isscalar(market_price) and np.any(iterations >= cls.MAX_ITERATIONS):
                    return cls.bisection(contract, market_price, pricer)
            
            return sigma
        except:
            # Fall back to bisection method if Newton-Raphson fails
            return cls.bisection(contract, market_price, pricer)


class LegacyImpliedVolatility:
    """Legacy implementation of implied volatility calculation (for compatibility)."""
    
    @staticmethod
    def calculate(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                 market_price: ArrayLike, option_type: Union[OptionType, np.ndarray],
                 q: ArrayLike = 0.0) -> ArrayLike:
        """
        Legacy method for calculating implied volatility.
        
        This provides compatibility with older code that uses the standalone function
        rather than the contract-based approach.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate (annualized)
            market_price: Market price of the option
            option_type: "call" or "put"
            q: Dividend yield (annualized) (default: 0.0)
            
        Returns:
            Implied volatility
        """
        # Convert to contract format
        contract = {
            "S": S,
            "K": K,
            "T": T,
            "r": r,
            "option_type": option_type,
            "q": q,
            "style": "european"
        }
        
        return ImpliedVolatilitySolver.calculate(contract, market_price)