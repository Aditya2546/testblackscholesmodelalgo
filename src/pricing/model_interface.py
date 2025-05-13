"""
Interface definitions for option pricing models.

Provides abstract base classes for implementing different option pricing models
with a consistent API, allowing them to be swapped at runtime.
"""

from abc import ABC, abstractmethod
from typing import Dict, Union, Tuple, Any, Optional
import numpy as np

from .black_scholes import ArrayLike


class OptionModel(ABC):
    """Abstract base class for option pricing models."""
    
    @abstractmethod
    def price(self, *args, **kwargs) -> ArrayLike:
        """Calculate option price."""
        pass
    
    @abstractmethod
    def delta(self, *args, **kwargs) -> ArrayLike:
        """Calculate option delta."""
        pass
    
    @abstractmethod
    def gamma(self, *args, **kwargs) -> ArrayLike:
        """Calculate option gamma."""
        pass
    
    @abstractmethod
    def vega(self, *args, **kwargs) -> ArrayLike:
        """Calculate option vega."""
        pass
    
    @abstractmethod
    def theta(self, *args, **kwargs) -> ArrayLike:
        """Calculate option theta."""
        pass
    
    @abstractmethod
    def rho(self, *args, **kwargs) -> ArrayLike:
        """Calculate option rho."""
        pass
    
    @abstractmethod
    def all_greeks(self, *args, **kwargs) -> Dict[str, ArrayLike]:
        """Calculate all option Greeks in a single pass."""
        pass
    
    @abstractmethod
    def implied_volatility(self, *args, **kwargs) -> ArrayLike:
        """Calculate implied volatility from market price."""
        pass


class ModelFactory:
    """Factory class for creating option pricing models."""
    
    _registered_models = {}
    
    @classmethod
    def register_model(cls, name: str, model_class: type) -> None:
        """
        Register a model class with a name.
        
        Args:
            name: The name of the model
            model_class: The class of the model
        """
        if not issubclass(model_class, OptionModel):
            raise TypeError(f"Model class must inherit from OptionModel, got {model_class}")
        cls._registered_models[name] = model_class
    
    @classmethod
    def create_model(cls, name: str, *args, **kwargs) -> OptionModel:
        """
        Create an instance of a registered model.
        
        Args:
            name: The name of the model to create
            *args, **kwargs: Arguments to pass to the model constructor
            
        Returns:
            An instance of the requested model
            
        Raises:
            ValueError: If the model name is not registered
        """
        if name not in cls._registered_models:
            available_models = ", ".join(cls._registered_models.keys())
            raise ValueError(f"Model '{name}' not found. Available models: {available_models}")
        
        model_class = cls._registered_models[name]
        return model_class(*args, **kwargs)
    
    @classmethod
    def list_models(cls) -> Dict[str, type]:
        """
        List all registered models.
        
        Returns:
            A dictionary mapping model names to model classes
        """
        return cls._registered_models.copy()


# Register the Black-Scholes model
from .black_scholes import BlackScholes, ImpliedVolatility, BlackScholesInputs


class BlackScholesModel(OptionModel):
    """Black-Scholes model implementation of the OptionModel interface."""
    
    def __init__(self):
        pass
    
    def price(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
             sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
             q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option price using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.price(inputs)
    
    def delta(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
             sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
             q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option delta using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.delta(inputs)
    
    def gamma(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
             sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
             q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option gamma using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.gamma(inputs)
    
    def vega(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
            sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
            q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option vega using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.vega(inputs)
    
    def theta(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
             sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
             q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option theta using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.theta(inputs)
    
    def rho(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
           sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
           q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate option rho using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.rho(inputs)
    
    def all_greeks(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                  sigma: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
                  q: ArrayLike = 0.0) -> Dict[str, ArrayLike]:
        """Calculate all option Greeks using Black-Scholes formula."""
        inputs = BlackScholesInputs(S=S, K=K, T=T, r=r, sigma=sigma, 
                                   option_type=option_type, q=q)
        return BlackScholes.all_greeks(inputs)
    
    def implied_volatility(self, S: ArrayLike, K: ArrayLike, T: ArrayLike, r: ArrayLike, 
                          market_price: ArrayLike, option_type: Union[str, np.ndarray] = "call", 
                          q: ArrayLike = 0.0) -> ArrayLike:
        """Calculate implied volatility using market price."""
        return ImpliedVolatility.calculate(S, K, T, r, market_price, option_type, q)


# Register the Black-Scholes model with the factory
ModelFactory.register_model("black_scholes", BlackScholesModel) 