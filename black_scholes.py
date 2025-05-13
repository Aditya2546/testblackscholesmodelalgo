#!/usr/bin/env python
"""
Vectorized Black-Scholes Options Pricing Engine

This module provides high-performance Black-Scholes options pricing with
numpy vectorization for maximum speed when calculating option prices
and Greeks for large datasets.
"""

import numpy as np
from scipy.stats import norm
from typing import Dict, List, Tuple, Union, Optional
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BlackScholes:
    """
    Vectorized Black-Scholes options pricing calculator with Greeks.
    
    This class provides methods to calculate option prices and Greeks
    using the Black-Scholes model. All methods support vectorized 
    calculations using numpy for maximum performance.
    """
    
    @staticmethod
    def d1(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
          r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate the d1 component of Black-Scholes formula.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            d1 component
        """
        # Handle potential divide by zero
        epsilon = 1e-8
        sigma_T = sigma * np.sqrt(T) + epsilon
        
        return ((np.log(S / K) + (r + 0.5 * sigma**2) * T) / sigma_T)
    
    @staticmethod
    def d2(d1: np.ndarray, sigma: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Calculate the d2 component of Black-Scholes formula.
        
        Args:
            d1: d1 component
            sigma: Implied volatility
            T: Time to expiration in years
            
        Returns:
            d2 component
        """
        return d1 - sigma * np.sqrt(T)
    
    @staticmethod
    def call_price(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                  r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate Black-Scholes price for call options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Call option price(s)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    
    @staticmethod
    def put_price(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                 r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate Black-Scholes price for put options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Put option price(s)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    
    @staticmethod
    def delta_call(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                  r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate delta for call options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Delta values for call options
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1_val)
    
    @staticmethod
    def delta_put(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                 r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate delta for put options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Delta values for put options
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        return norm.cdf(d1_val) - 1
    
    @staticmethod
    def gamma(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
             r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate gamma (same for calls and puts).
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Gamma values
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        
        # N'(d1) = standard normal pdf
        epsilon = 1e-8
        sigma_sqrt_T = sigma * np.sqrt(T) + epsilon
        
        return norm.pdf(d1_val) / (S * sigma_sqrt_T)
    
    @staticmethod
    def vega(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
            r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate vega (same for calls and puts).
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Vega values (as percentage points per 1% change in vol)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        
        # Multiply by 0.01 to express in terms of 1% change in vol
        return 0.01 * S * np.sqrt(T) * norm.pdf(d1_val)
    
    @staticmethod
    def theta_call(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                  r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate theta for call options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Theta values for call options (per day)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        # First term: derivative of the stock term
        term1 = -(S * sigma * norm.pdf(d1_val)) / (2 * np.sqrt(T))
        
        # Second term: derivative of the exercise term
        term2 = -r * K * np.exp(-r * T) * norm.cdf(d2_val)
        
        # Theta per day (divide by 365)
        return (term1 + term2) / 365
    
    @staticmethod
    def theta_put(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
                 r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate theta for put options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Theta values for put options (per day)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        # First term: derivative of the stock term
        term1 = -(S * sigma * norm.pdf(d1_val)) / (2 * np.sqrt(T))
        
        # Second term: derivative of the exercise term
        term2 = r * K * np.exp(-r * T) * norm.cdf(-d2_val)
        
        # Theta per day (divide by 365)
        return (term1 + term2) / 365
    
    @staticmethod
    def rho_call(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
               r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate rho for call options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Rho values for call options (per 1% change in interest rate)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        # Multiply by 0.01 to express in terms of 1% change in rate
        return 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2_val)
    
    @staticmethod
    def rho_put(S: np.ndarray, K: np.ndarray, T: np.ndarray, 
              r: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """
        Calculate rho for put options.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            
        Returns:
            Rho values for put options (per 1% change in interest rate)
        """
        d1_val = BlackScholes.d1(S, K, T, r, sigma)
        d2_val = BlackScholes.d2(d1_val, sigma, T)
        
        # Multiply by 0.01 to express in terms of 1% change in rate
        return -0.01 * K * T * np.exp(-r * T) * norm.cdf(-d2_val)
    
    @staticmethod
    def implied_volatility(option_price: np.ndarray, S: np.ndarray, K: np.ndarray, 
                          T: np.ndarray, r: np.ndarray, option_type: str = 'call',
                          max_iterations: int = 100, precision: float = 1e-6) -> np.ndarray:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            option_price: Observed option price(s)
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            option_type: 'call' or 'put'
            max_iterations: Maximum number of iterations
            precision: Desired precision
            
        Returns:
            Implied volatility
        """
        # Handle array inputs
        if not isinstance(option_price, np.ndarray):
            option_price = np.array([option_price]).flatten()
        if not isinstance(S, np.ndarray):
            S = np.array([S]).flatten()
        if not isinstance(K, np.ndarray):
            K = np.array([K]).flatten()
        if not isinstance(T, np.ndarray):
            T = np.array([T]).flatten()
        if not isinstance(r, np.ndarray):
            r = np.array([r]).flatten()
        
        # Broadcast arrays
        option_price, S, K, T, r = np.broadcast_arrays(option_price, S, K, T, r)
        
        # Initial guess
        sigma = np.ones_like(option_price) * 0.2
        
        # Select pricing function
        if option_type.lower() == 'call':
            price_func = BlackScholes.call_price
            vega_func = BlackScholes.vega
        else:
            price_func = BlackScholes.put_price
            vega_func = BlackScholes.vega
        
        # Newton-Raphson iterations
        for i in range(max_iterations):
            # Calculate price and vega at current sigma
            price = price_func(S, K, T, r, sigma)
            vega = vega_func(S, K, T, r, sigma)
            
            # Calculate price difference
            diff = option_price - price
            
            # Check for convergence
            if np.all(np.abs(diff) < precision):
                break
            
            # Avoid division by zero
            vega = np.where(vega < 1e-8, 1e-8, vega)
            
            # Update sigma
            sigma = sigma + diff / (vega * 100)  # Adjustment for vega units
            
            # Enforce bounds
            sigma = np.clip(sigma, 0.001, 5.0)
        
        return sigma
    
    @staticmethod
    def price_and_greeks(S: Union[float, np.ndarray], K: Union[float, np.ndarray], 
                        T: Union[float, np.ndarray], r: Union[float, np.ndarray], 
                        sigma: Union[float, np.ndarray], option_type: str = 'call') -> Dict:
        """
        Calculate option price and all Greeks in one operation.
        
        Args:
            S: Stock price(s)
            K: Strike price(s)
            T: Time to expiration in years
            r: Risk-free interest rate
            sigma: Implied volatility
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with price and all Greeks
        """
        # Convert inputs to numpy arrays
        if not isinstance(S, np.ndarray):
            S = np.array([S]).flatten()
        if not isinstance(K, np.ndarray):
            K = np.array([K]).flatten()
        if not isinstance(T, np.ndarray):
            T = np.array([T]).flatten()
        if not isinstance(r, np.ndarray):
            r = np.array([r]).flatten()
        if not isinstance(sigma, np.ndarray):
            sigma = np.array([sigma]).flatten()
        
        # Broadcast arrays
        S, K, T, r, sigma = np.broadcast_arrays(S, K, T, r, sigma)
        
        # Choose call or put functions
        if option_type.lower() == 'call':
            price_func = BlackScholes.call_price
            delta_func = BlackScholes.delta_call
            theta_func = BlackScholes.theta_call
            rho_func = BlackScholes.rho_call
        else:
            price_func = BlackScholes.put_price
            delta_func = BlackScholes.delta_put
            theta_func = BlackScholes.theta_put
            rho_func = BlackScholes.rho_put
        
        # Calculate all values
        results = {
            'price': price_func(S, K, T, r, sigma),
            'delta': delta_func(S, K, T, r, sigma),
            'gamma': BlackScholes.gamma(S, K, T, r, sigma),
            'theta': theta_func(S, K, T, r, sigma),
            'vega': BlackScholes.vega(S, K, T, r, sigma),
            'rho': rho_func(S, K, T, r, sigma)
        }
        
        return results
    
    @staticmethod
    def days_to_years(days: Union[int, float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert days to years for time to expiration.
        
        Args:
            days: Number of days to expiration
            
        Returns:
            Time to expiration in years
        """
        return days / 365.0
    
    @staticmethod
    def pct_to_decimal(pct: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Convert percentage to decimal for rate or volatility.
        
        Args:
            pct: Percentage value (e.g., 5 for 5%)
            
        Returns:
            Decimal value (e.g., 0.05)
        """
        return pct / 100.0


def benchmark():
    """Run benchmarks to demonstrate vectorization performance"""
    print("=== Black-Scholes Pricing Engine Benchmark ===")
    
    # Create input arrays
    n = 1000000  # 1 million options
    
    S = np.ones(n) * 100  # Stock price $100
    K = np.linspace(80, 120, n)  # Strikes from $80 to $120
    T = np.ones(n) * 30 / 365  # 30 days to expiration
    r = np.ones(n) * 0.05  # 5% risk-free rate
    sigma = np.ones(n) * 0.2  # 20% volatility
    
    # Benchmark scalar (loop) approach
    print(f"Calculating prices for {n:,} options...")
    
    print("\nScalar approach (first 10 iterations only):")
    start_time = time.time()
    scalar_results = []
    for i in range(min(10, n)):
        price = (S[i] * norm.cdf((np.log(S[i]/K[i]) + (r[i] + 0.5 * sigma[i]**2) * T[i]) / (sigma[i] * np.sqrt(T[i]))) - 
                K[i] * np.exp(-r[i] * T[i]) * norm.cdf((np.log(S[i]/K[i]) + (r[i] - 0.5 * sigma[i]**2) * T[i]) / (sigma[i] * np.sqrt(T[i]))))
        scalar_results.append(price)
    scalar_time = time.time() - start_time
    print(f"Time for 10 iterations: {scalar_time:.6f} seconds")
    print(f"Estimated time for {n:,} options: {scalar_time * n / 10:.2f} seconds")
    
    # Benchmark vectorized approach
    print("\nVectorized approach:")
    start_time = time.time()
    vectorized_results = BlackScholes.call_price(S, K, T, r, sigma)
    vectorized_time = time.time() - start_time
    print(f"Time for {n:,} options: {vectorized_time:.6f} seconds")
    
    # Calculate speedup
    estimated_scalar_time = scalar_time * n / 10
    speedup = estimated_scalar_time / vectorized_time
    print(f"\nSpeedup: {speedup:.1f}x")
    
    # Benchmark all Greeks calculation
    print("\nCalculating price and all Greeks:")
    start_time = time.time()
    all_results = BlackScholes.price_and_greeks(S, K, T, r, sigma, 'call')
    all_time = time.time() - start_time
    print(f"Time for {n:,} options: {all_time:.6f} seconds")
    
    # Benchmark implied volatility
    print("\nCalculating implied volatility (10,000 options):")
    # Generate option prices for a smaller subset
    n_iv = 10000
    S_iv = np.ones(n_iv) * 100
    K_iv = np.linspace(80, 120, n_iv)
    T_iv = np.ones(n_iv) * 30 / 365
    r_iv = np.ones(n_iv) * 0.05
    sigma_iv = np.linspace(0.1, 0.6, n_iv)  # True volatilities
    
    # Generate prices using the known volatilities
    prices = BlackScholes.call_price(S_iv, K_iv, T_iv, r_iv, sigma_iv)
    
    # Now solve for implied volatility
    start_time = time.time()
    iv = BlackScholes.implied_volatility(prices, S_iv, K_iv, T_iv, r_iv, 'call')
    iv_time = time.time() - start_time
    
    # Calculate error
    error = np.mean(np.abs(iv - sigma_iv))
    print(f"Time for {n_iv:,} options: {iv_time:.6f} seconds")
    print(f"Average error: {error:.6f}")
    print(f"Convergence rate: {np.sum(np.abs(iv - sigma_iv) < 0.001) / n_iv * 100:.2f}%")


if __name__ == "__main__":
    benchmark() 