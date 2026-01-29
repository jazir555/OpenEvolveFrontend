"""Error hierarchy for Adaptive MDAP."""

from typing import Optional, Dict, Any


class AdaptiveMDAPError(Exception):
    """Base error for Adaptive MDAP system."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ClassificationError(AdaptiveMDAPError):
    """Error during task complexity classification."""
    
    def __init__(self, message: str, subproblem_id: Optional[str] = None, 
                 feature: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.subproblem_id = subproblem_id
        self.feature = feature


class AllocationError(AdaptiveMDAPError):
    """Error during resource allocation."""
    
    def __init__(self, message: str, complexity_score: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.complexity_score = complexity_score


class ConfigurationError(AdaptiveMDAPError):
    """Error in configuration."""
    
    def __init__(self, message: str, config_key: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.config_key = config_key


class CacheError(AdaptiveMDAPError):
    """Error in caching system."""
    
    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.cache_key = cache_key
        self.operation = operation


class ExecutionError(AdaptiveMDAPError):
    """Error during task execution."""
    
    def __init__(self, message: str, strategy: Optional[str] = None,
                 subproblem_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message, details)
        self.strategy = strategy
        self.subproblem_id = subproblem_id
