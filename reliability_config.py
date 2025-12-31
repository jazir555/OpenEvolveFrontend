"""
External Configuration for Sovereign Reliability System
"""

# Retry settings (used by with_retry decorator)
RETRY_CONFIG = {
    "max_attempts": 3,
    "initial_delay": 1.0,  # seconds
    "max_delay": 10.0,     # seconds
    "exponential_base": 2.0,
    "jitter": True,
}

# Circuit breaker settings
CIRCUIT_BREAKER_CONFIG = {
    "failure_threshold": 5,
    "timeout": 30.0,  # seconds
}

# Rate limiter settings
RATE_LIMITER_CONFIG = {
    "max_requests": 100,
    "time_window": 60.0,  # seconds
}