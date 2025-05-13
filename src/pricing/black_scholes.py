"""
Black-Scholes option pricing model with vectorized implementation.
Includes closed-form solutions for European options and Greeks.
"""

import numpy as np
from scipy.stats import norm
from dataclasses import dataclass
from typing import Union, Literal, Optional, Dict, Tuple, Callable

# Type aliases
ArrayLike = Union[float, np.ndarray]
OptionType = Literal["call", "put"]


@dataclass
class BlackScholesInputs:
    """Container for Black-Scholes model inputs."""
    S: ArrayLike  # Underlying price
    K: ArrayLike  # Strike price
    T: ArrayLike  # Time to expiration (in years)
    r: ArrayLike  # Risk-free rate (annualized)
    sigma: ArrayLike  # Volatility (annualized)
    option_type: Union[OptionType, np.ndarray] = "call"  # Option type
    q: ArrayLike = 0.0  # Dividend yield (annualized)


class BlackScholes:
    """
    Vectorized Black-Scholes option pricing model implementation.
    
    This class provides high-performance calculations for option prices and Greeks
    by leveraging NumPy vectorized operations. It can process multiple options 
    simultaneously for maximum throughput.
    """
    
    @staticmethod
    def _validate_inputs(inputs: BlackScholesInputs) -> None:
        """Validate inputs to avoid numerical issues."""
        # Convert option_type to arrays if needed
        if isinstance(inputs.option_type, str):
            if inputs.option_type not in ("call", "put"):
                raise ValueError("option_type must be 'call' or 'put'")
        
        # Replace zeros in time to expiration with a small number to avoid division by zero
        if hasattr(inputs.T, '__iter__'):
            inputs.T = np.where(inputs.T <= 0.0, 1e-8, inputs.T)
        elif inputs.T <= 0.0:
            inputs.T = 1e-8
            
        # Replace zeros in volatility with a small number to avoid division by zero
        if hasattr(inputs.sigma, '__iter__'):
            inputs.sigma = np.where(inputs.sigma <= 0.0, 1e-8, inputs.sigma)
        elif inputs.sigma <= 0.0:
            inputs.sigma = 1e-8

    @staticmethod
    def _calculate_d1_d2(S: ArrayLike, K: ArrayLike, T: ArrayLike, 
                         r: ArrayLike, sigma: ArrayLike, q: ArrayLike = 0.0) -> Tuple[ArrayLike, ArrayLike]:
        """Calculate d1 and d2 parameters for Black-Scholes formula."""
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return d1, d2

    @classmethod
    def price(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """
        Calculate option price using Black-Scholes formula.
        
        Args:
            inputs: Black-Scholes input parameters
            
        Returns:
            Option price(s)
        """
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, d2 = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        if isinstance(inputs.option_type, str):
            # Scalar option type
            if inputs.option_type == "call":
                return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            # Vector of option types
            call_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
            
            # Create a mask for call options
            call_mask = (inputs.option_type == "call")
            
            # Select price based on option type
            return np.where(call_mask, call_price, put_price)

    @classmethod
    def delta(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """Calculate option delta (∂V/∂S)."""
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, _ = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        if isinstance(inputs.option_type, str):
            if inputs.option_type == "call":
                return np.exp(-q * T) * norm.cdf(d1)
            else:  # put
                return np.exp(-q * T) * (norm.cdf(d1) - 1)
        else:
            call_delta = np.exp(-q * T) * norm.cdf(d1)
            put_delta = np.exp(-q * T) * (norm.cdf(d1) - 1)
            
            call_mask = (inputs.option_type == "call")
            return np.where(call_mask, call_delta, put_delta)

    @classmethod
    def gamma(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """Calculate option gamma (∂²V/∂S²)."""
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, _ = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Gamma is the same for calls and puts
        return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @classmethod
    def vega(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """Calculate option vega (∂V/∂σ), expressed as change per 1% vol move."""
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, _ = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Vega is the same for calls and puts
        # Multiply by 0.01 to express as change per 1% vol move
        return 0.01 * S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)

    @classmethod
    def theta(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """Calculate option theta (∂V/∂T), expressed as change per calendar day."""
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, d2 = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Calculate theta per year
        if isinstance(inputs.option_type, str):
            if inputs.option_type == "call":
                theta_year = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - \
                          r * K * np.exp(-r * T) * norm.cdf(d2) + \
                          q * S * np.exp(-q * T) * norm.cdf(d1)
            else:  # put
                theta_year = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + \
                          r * K * np.exp(-r * T) * norm.cdf(-d2) - \
                          q * S * np.exp(-q * T) * norm.cdf(-d1)
        else:
            call_theta = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - \
                      r * K * np.exp(-r * T) * norm.cdf(d2) + \
                      q * S * np.exp(-q * T) * norm.cdf(d1)
            
            put_theta = -np.exp(-q * T) * S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + \
                      r * K * np.exp(-r * T) * norm.cdf(-d2) - \
                      q * S * np.exp(-q * T) * norm.cdf(-d1)
            
            call_mask = (inputs.option_type == "call")
            theta_year = np.where(call_mask, call_theta, put_theta)
        
        # Convert to daily theta (calendar days)
        return theta_year / 365.0

    @classmethod
    def rho(cls, inputs: BlackScholesInputs) -> ArrayLike:
        """Calculate option rho (∂V/∂r), expressed as change per 1% rate move."""
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        _, d2 = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Scale by 0.01 to get change per 1% move in rates
        if isinstance(inputs.option_type, str):
            if inputs.option_type == "call":
                return 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2)
            else:  # put
                return -0.01 * K * T * np.exp(-r * T) * norm.cdf(-d2)
        else:
            call_rho = 0.01 * K * T * np.exp(-r * T) * norm.cdf(d2)
            put_rho = -0.01 * K * T * np.exp(-r * T) * norm.cdf(-d2)
            
            call_mask = (inputs.option_type == "call")
            return np.where(call_mask, call_rho, put_rho)
            
    @classmethod
    def all_greeks(cls, inputs: BlackScholesInputs) -> Dict[str, ArrayLike]:
        """
        Calculate all option Greeks in a single pass for efficiency.
        
        Returns:
            Dictionary containing all Greeks and option price
        """
        cls._validate_inputs(inputs)
        
        S, K, T, r, sigma, q = inputs.S, inputs.K, inputs.T, inputs.r, inputs.sigma, inputs.q
        d1, d2 = cls._calculate_d1_d2(S, K, T, r, sigma, q)
        
        # Common terms
        norm_d1 = norm.cdf(d1)
        norm_d2 = norm.cdf(d2)
        norm_neg_d1 = norm.cdf(-d1)
        norm_neg_d2 = norm.cdf(-d2)
        norm_pdf_d1 = norm.pdf(d1)
        
        exp_rT = np.exp(-r * T)
        exp_qT = np.exp(-q * T)
        
        # Pre-compute prices
        call_price = S * exp_qT * norm_d1 - K * exp_rT * norm_d2
        put_price = K * exp_rT * norm_neg_d2 - S * exp_qT * norm_neg_d1
        
        # Pre-compute Greeks
        gamma_val = exp_qT * norm_pdf_d1 / (S * sigma * np.sqrt(T))
        vega_val = 0.01 * S * exp_qT * norm_pdf_d1 * np.sqrt(T)
        
        call_delta = exp_qT * norm_d1
        put_delta = call_delta - exp_qT
        
        call_theta = -exp_qT * S * norm_pdf_d1 * sigma / (2 * np.sqrt(T)) - \
                   r * K * exp_rT * norm_d2 + q * S * exp_qT * norm_d1
        put_theta = -exp_qT * S * norm_pdf_d1 * sigma / (2 * np.sqrt(T)) + \
                  r * K * exp_rT * norm_neg_d2 - q * S * exp_qT * norm_neg_d1
        
        call_rho = 0.01 * K * T * exp_rT * norm_d2
        put_rho = -0.01 * K * T * exp_rT * norm_neg_d2
        
        # Convert theta to daily
        call_theta = call_theta / 365.0
        put_theta = put_theta / 365.0
        
        # Create result dictionary
        if isinstance(inputs.option_type, str):
            if inputs.option_type == "call":
                return {
                    "price": call_price,
                    "delta": call_delta,
                    "gamma": gamma_val,
                    "vega": vega_val,
                    "theta": call_theta,
                    "rho": call_rho
                }
            else:  # put
                return {
                    "price": put_price,
                    "delta": put_delta,
                    "gamma": gamma_val,
                    "vega": vega_val,
                    "theta": put_theta,
                    "rho": put_rho
                }
        else:
            # Handle vectorized case with multiple option types
            call_mask = (inputs.option_type == "call")
            
            return {
                "price": np.where(call_mask, call_price, put_price),
                "delta": np.where(call_mask, call_delta, put_delta),
                "gamma": gamma_val,  # Same for calls and puts
                "vega": vega_val,    # Same for calls and puts
                "theta": np.where(call_mask, call_theta, put_theta),
                "rho": np.where(call_mask, call_rho, put_rho)
            }


class ImpliedVolatility:
    """
    Fast implied volatility calculator using Newton-Raphson method.
    
    Efficiently computes implied volatility from option prices with
    vectorized implementation for high throughput.
    """
    
    # Constants for initial guesses and algorithm control
    MAX_ITERATIONS = 10
    PRECISION = 1e-8
    
    @staticmethod
    def _initial_guess(S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                      market_price: ArrayLike, option_type: Union[OptionType, np.ndarray],
                      q: ArrayLike = 0.0) -> ArrayLike:
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
    def newton_raphson(cls, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                       market_price: ArrayLike, option_type: Union[OptionType, np.ndarray],
                       q: ArrayLike = 0.0) -> Tuple[ArrayLike, ArrayLike]:
        """
        Calculate implied volatility using Newton-Raphson method.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            market_price: Market price of the option
            option_type: Option type ("call" or "put")
            q: Dividend yield
            
        Returns:
            Tuple of (implied volatility, number of iterations to converge)
        """
        # Generate initial guess
        sigma = cls._initial_guess(S, K, T, r, market_price, option_type, q)
        
        # Create inputs dataclass for Black-Scholes
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)
        
        # Initialize variables for tracking iterations
        iterations = np.zeros_like(sigma, dtype=int)
        converged = np.zeros_like(sigma, dtype=bool)
        
        # Main Newton-Raphson loop
        for i in range(cls.MAX_ITERATIONS):
            # Calculate option price and vega
            price = BlackScholes.price(inputs)
            vega = BlackScholes.vega(inputs) * 100  # Convert to per 1.0 vol move
            
            # Calculate price difference
            diff = price - market_price
            
            # Check for convergence
            new_converged = np.abs(diff) < cls.PRECISION
            if np.all(new_converged):
                break
                
            # Update iterations for newly converged values
            iterations = np.where(new_converged & ~converged, i, iterations)
            converged = converged | new_converged
            
            # Update sigma using Newton-Raphson step
            # Handle near-zero vega case to avoid numerical issues
            vega = np.where(np.abs(vega) < 1e-10, 1e-10, vega)
            sigma_update = diff / vega
            
            # Apply update only to non-converged values
            sigma = np.where(converged, sigma, sigma - sigma_update)
            
            # Ensure sigma stays positive
            sigma = np.maximum(sigma, 1e-8)
            
            # Update inputs for next iteration
            inputs.sigma = sigma
            
        # Update iterations for any values that didn't converge
        iterations = np.where(~converged, cls.MAX_ITERATIONS, iterations)
        
        return sigma, iterations
        
    @classmethod
    def bisection(cls, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                 market_price: ArrayLike, option_type: Union[OptionType, np.ndarray],
                 q: ArrayLike = 0.0) -> ArrayLike:
        """
        Fallback method using bisection when Newton-Raphson fails to converge.
        
        This is more robust but slower than Newton-Raphson.
        """
        # Initial bounds for volatility
        sigma_low = np.full_like(S, 0.001, dtype=float)
        sigma_high = np.full_like(S, 10.0, dtype=float)
        
        # Create inputs dataclass
        inputs_low = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_low, 
                                     option_type=option_type, q=q)
        inputs_high = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma_high, 
                                      option_type=option_type, q=q)
        
        # Calculate option prices at bounds
        price_low = BlackScholes.price(inputs_low)
        price_high = BlackScholes.price(inputs_high)
        
        # Initialize result array
        sigma = (sigma_low + sigma_high) / 2
        
        # Main bisection loop
        for _ in range(cls.MAX_ITERATIONS * 2):  # More iterations for bisection
            # Update middle point
            sigma = (sigma_low + sigma_high) / 2
            
            # Create new inputs and calculate price
            inputs_mid = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                         option_type=option_type, q=q)
            price_mid = BlackScholes.price(inputs_mid)
            
            # Calculate price difference
            diff = price_mid - market_price
            
            # Check for convergence
            if np.all(np.abs(diff) < cls.PRECISION):
                break
                
            # Update bounds
            sigma_low = np.where((diff * (price_low - market_price)) > 0, sigma, sigma_low)
            sigma_high = np.where((diff * (price_high - market_price)) > 0, sigma, sigma_high)
            
            # Update prices at bounds if needed
            mask_low = (sigma_low != inputs_low.sigma)
            if np.any(mask_low):
                inputs_low.sigma = sigma_low
                price_low = BlackScholes.price(inputs_low)
                
            mask_high = (sigma_high != inputs_high.sigma)
            if np.any(mask_high):
                inputs_high.sigma = sigma_high
                price_high = BlackScholes.price(inputs_high)
        
        return sigma
    
    @classmethod
    def calculate(cls, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                 market_price: ArrayLike, option_type: Union[OptionType, np.ndarray],
                 q: ArrayLike = 0.0) -> ArrayLike:
        """
        Calculate implied volatility using a hybrid approach.
        
        First tries Newton-Raphson for speed, then falls back to bisection
        for robustness if Newton-Raphson fails to converge.
        
        Args:
            S: Underlying price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            market_price: Market price of the option
            option_type: Option type ("call" or "put")
            q: Dividend yield
            
        Returns:
            Implied volatility
        """
        # Try Newton-Raphson first
        sigma, iterations = cls.newton_raphson(S, K, T, r, market_price, option_type, q)
        
        # Check for non-convergence
        non_converged = (iterations >= cls.MAX_ITERATIONS)
        
        # If all converged, return the result
        if not np.any(non_converged):
            return sigma
            
        # For non-converged values, fall back to bisection
        if isinstance(S, np.ndarray):
            # Extract non-converged inputs
            if np.isscalar(non_converged):
                # Single value case
                if non_converged:
                    sigma = cls.bisection(S, K, T, r, market_price, option_type, q)
            else:
                # Array case - only run bisection on non-converged values
                S_nc = S[non_converged]
                K_nc = K[non_converged] if isinstance(K, np.ndarray) else K
                T_nc = T[non_converged] if isinstance(T, np.ndarray) else T
                r_nc = r[non_converged] if isinstance(r, np.ndarray) else r
                market_price_nc = market_price[non_converged]
                option_type_nc = option_type[non_converged] if isinstance(option_type, np.ndarray) else option_type
                q_nc = q[non_converged] if isinstance(q, np.ndarray) else q
                
                # Calculate for non-converged values
                if np.size(S_nc) > 0:  # Only if we have non-converged values
                    sigma_nc = cls.bisection(S_nc, K_nc, T_nc, r_nc, market_price_nc, option_type_nc, q_nc)
                    
                    # Update only non-converged values
                    sigma[non_converged] = sigma_nc
        else:
            # Scalar case
            if non_converged:
                sigma = cls.bisection(S, K, T, r, market_price, option_type, q)
                
        return sigma 