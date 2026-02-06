
"""Rate Limiting Module (Test Compatibility)"""

import time
from typing import Any


class FixedWindowRateLimiter:
    """Fixed window rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def allow_request(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        # Remove old requests
        self.requests = [r for r in self.requests if now - r < self.window_seconds]
        if len(self.requests) < self.max_requests:
            self.requests.append(now)
            return True
        return False


class SlidingWindowRateLimiter:
    """Sliding window rate limiter."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def allow_request(self) -> bool:
        """Check if request is allowed."""
        now = time.time()
        self.requests.append(now)
        # Remove old requests
        self.requests = [r for r in self.requests if now - r < self.window_seconds]
        return len(self.requests) <= self.max_requests


class TokenBucketRateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int = 10, capacity: int = 100):
        self.rate = rate
        self.capacity = capacity
        self.tokens = capacity
    
    def consume(self) -> bool:
        """Consume a token."""
        if self.tokens > 0:
            self.tokens -= 1
            return True
        return False


class RateLimitHeaders:
    """Generator for rate limit headers."""
    
    @staticmethod
    def generate(remaining: int, limit: int, reset: int) -> dict:
        """Generate rate limit headers."""
        return {
            'X-RateLimit-Remaining': str(remaining),
            'X-RateLimit-Limit': str(limit),
            'X-RateLimit-Reset': str(reset)
        }


class RateLimitExceededHandler:
    """Handler for rate limit exceeded."""
    
    class Response:
        def __init__(self, status_code):
            self.status_code = status_code
    
    def create_response(self) -> Any:
        """Create rate limit exceeded response."""
        return self.Response(429)
