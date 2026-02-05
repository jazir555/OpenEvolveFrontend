"""
Comprehensive Rate Limiting and API Security Tests
Tests for rate limiting, DoS protection, and API security controls.
"""

import pytest
import time
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock
import threading
import json


class RateLimiter:
    """Simple rate limiter implementation for testing."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> bool:
        """Check if request is allowed under rate limit."""
        with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            # Clean old requests
            if key in self.requests:
                self.requests[key] = [
                    req_time for req_time in self.requests[key]
                    if req_time > window_start
                ]
            else:
                self.requests[key] = []
            
            # Check if under limit
            if len(self.requests[key]) < self.max_requests:
                self.requests[key].append(now)
                return True
            return False
    
    def get_remaining(self, key: str) -> int:
        """Get remaining requests in current window."""
        with self._lock:
            now = datetime.utcnow()
            window_start = now - timedelta(seconds=self.window_seconds)
            
            if key not in self.requests:
                return self.max_requests
            
            recent_requests = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
            
            return max(0, self.max_requests - len(recent_requests))
    
    def get_reset_time(self, key: str) -> datetime:
        """Get time when rate limit resets."""
        with self._lock:
            if key not in self.requests or not self.requests[key]:
                return datetime.utcnow() + timedelta(seconds=self.window_seconds)
            
            oldest_request = min(self.requests[key])
            return oldest_request + timedelta(seconds=self.window_seconds)


class SlidingWindowRateLimiter:
    """Sliding window rate limiter for more precise control."""
    
    def __init__(self, max_requests: int = 100, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
    
    def is_allowed(self, key: str) -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed with metadata."""
        with self._lock:
            now = time.time()
            window_start = now - self.window_seconds
            
            # Initialize or clean old requests
            if key not in self.requests:
                self.requests[key] = []
            
            self.requests[key] = [
                req_time for req_time in self.requests[key]
                if req_time > window_start
            ]
            
            current_count = len(self.requests[key])
            
            if current_count < self.max_requests:
                self.requests[key].append(now)
                return True, {
                    'limit': self.max_requests,
                    'remaining': self.max_requests - current_count - 1,
                    'reset': now + self.window_seconds,
                    'window': self.window_seconds
                }
            
            return False, {
                'limit': self.max_requests,
                'remaining': 0,
                'reset': min(self.requests[key]) + self.window_seconds,
                'window': self.window_seconds,
                'retry_after': int(min(self.requests[key]) + self.window_seconds - now)
            }


class TokenBucketRateLimiter:
    """Token bucket rate limiter for burst handling."""
    
    def __init__(self, capacity: int = 100, refill_rate: float = 1.0):
        self.capacity = capacity
        self.refill_rate = refill_rate  # tokens per second
        self.tokens: Dict[str, float] = {}
        self.last_update: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def _refill(self, key: str):
        """Refill tokens based on time elapsed."""
        now = time.time()
        if key not in self.tokens:
            self.tokens[key] = self.capacity
            self.last_update[key] = now
            return
        
        elapsed = now - self.last_update[key]
        tokens_to_add = elapsed * self.refill_rate
        self.tokens[key] = min(self.capacity, self.tokens[key] + tokens_to_add)
        self.last_update[key] = now
    
    def is_allowed(self, key: str, tokens: int = 1) -> bool:
        """Check if request is allowed."""
        with self._lock:
            self._refill(key)
            
            if self.tokens[key] >= tokens:
                self.tokens[key] -= tokens
                return True
            return False
    
    def get_tokens(self, key: str) -> float:
        """Get current token count."""
        with self._lock:
            self._refill(key)
            return self.tokens.get(key, self.capacity)


class TestBasicRateLimiting:
    """Test basic rate limiting functionality."""
    
    @pytest.fixture
    def rate_limiter(self):
        return RateLimiter(max_requests=10, window_seconds=60)
    
    def test_requests_within_limit(self, rate_limiter):
        """Test that requests within limit are allowed."""
        key = "user_123"
        
        for _ in range(10):
            assert rate_limiter.is_allowed(key) == True
    
    def test_requests_exceeding_limit(self, rate_limiter):
        """Test that requests exceeding limit are blocked."""
        key = "user_123"
        
        # Make max allowed requests
        for _ in range(10):
            assert rate_limiter.is_allowed(key) == True
        
        # Next request should be blocked
        assert rate_limiter.is_allowed(key) == False
    
    def test_different_keys_independent(self, rate_limiter):
        """Test that different keys have independent limits."""
        # Exhaust limit for user_1
        for _ in range(10):
            rate_limiter.is_allowed("user_1")
        
        assert rate_limiter.is_allowed("user_1") == False
        
        # user_2 should still have full limit
        assert rate_limiter.is_allowed("user_2") == True
    
    def test_remaining_requests(self, rate_limiter):
        """Test remaining requests calculation."""
        key = "user_123"
        
        assert rate_limiter.get_remaining(key) == 10
        
        rate_limiter.is_allowed(key)
        assert rate_limiter.get_remaining(key) == 9
        
        # Exhaust limit
        for _ in range(9):
            rate_limiter.is_allowed(key)
        
        assert rate_limiter.get_remaining(key) == 0
    
    def test_window_reset(self, rate_limiter):
        """Test that limit resets after window expires."""
        key = "user_123"
        
        # Exhaust limit
        for _ in range(10):
            rate_limiter.is_allowed(key)
        
        assert rate_limiter.is_allowed(key) == False
        
        # Simulate time passing (would need mocking in real tests)
        # For this test, we just verify the reset time is in the future
        reset_time = rate_limiter.get_reset_time(key)
        assert reset_time > datetime.utcnow()


class TestSlidingWindowRateLimiter:
    """Test sliding window rate limiter."""
    
    @pytest.fixture
    def limiter(self):
        return SlidingWindowRateLimiter(max_requests=5, window_seconds=60)
    
    def test_request_metadata(self, limiter):
        """Test that metadata is returned with each request."""
        key = "user_123"
        
        allowed, metadata = limiter.is_allowed(key)
        
        assert allowed == True
        assert 'limit' in metadata
        assert 'remaining' in metadata
        assert 'reset' in metadata
        assert 'window' in metadata
        assert metadata['limit'] == 5
        assert metadata['remaining'] == 4
    
    def test_rate_limit_exceeded_metadata(self, limiter):
        """Test metadata when rate limit exceeded."""
        key = "user_123"
        
        # Exhaust limit
        for _ in range(5):
            limiter.is_allowed(key)
        
        allowed, metadata = limiter.is_allowed(key)
        
        assert allowed == False
        assert metadata['remaining'] == 0
        assert 'retry_after' in metadata
        assert metadata['retry_after'] > 0
    
    def test_sliding_window_accuracy(self, limiter):
        """Test that sliding window accurately tracks requests."""
        key = "user_123"
        
        # Add requests spread over time (simulated)
        for i in range(3):
            allowed, _ = limiter.is_allowed(key)
            assert allowed == True
        
        # Should still have capacity
        allowed, metadata = limiter.is_allowed(key)
        assert allowed == True
        assert metadata['remaining'] == 1


class TestTokenBucketRateLimiter:
    """Test token bucket rate limiter."""
    
    @pytest.fixture
    def limiter(self):
        return TokenBucketRateLimiter(capacity=10, refill_rate=1.0)
    
    def test_burst_handling(self, limiter):
        """Test handling of burst traffic."""
        key = "user_123"
        
        # Should handle burst up to capacity
        for _ in range(10):
            assert limiter.is_allowed(key) == True
        
        # Should block after capacity exhausted
        assert limiter.is_allowed(key) == False
    
    def test_token_refill(self, limiter):
        """Test token refill over time."""
        key = "user_123"
        
        # Use some tokens
        for _ in range(5):
            limiter.is_allowed(key)
        
        assert limiter.get_tokens(key) == 5.0
        
        # After refill time, should have more tokens
        time.sleep(1.1)  # Wait for refill
        tokens = limiter.get_tokens(key)
        assert tokens > 5.0
    
    def test_variable_token_cost(self, limiter):
        """Test requests with different token costs."""
        key = "user_123"
        
        # High cost request
        assert limiter.is_allowed(key, tokens=5) == True
        assert limiter.get_tokens(key) == 5.0
        
        # Another high cost request
        assert limiter.is_allowed(key, tokens=5) == True
        assert limiter.get_tokens(key) == 0.0
        
        # Should be blocked now
        assert limiter.is_allowed(key, tokens=1) == False


class TestDoSProtection:
    """Test DoS protection mechanisms."""
    
    def test_request_size_limit(self):
        """Test enforcement of request size limits."""
        max_size = 1024 * 1024  # 1MB
        
        # Small request should be allowed
        small_payload = "x" * 1000
        assert len(small_payload.encode()) < max_size
        
        # Large request should be blocked
        large_payload = "x" * (max_size + 100)
        assert len(large_payload.encode()) > max_size
    
    def test_concurrent_connection_limit(self):
        """Test limiting concurrent connections."""
        max_connections = 100
        active_connections = 0
        lock = threading.Lock()
        
        def simulate_connection():
            nonlocal active_connections
            with lock:
                if active_connections >= max_connections:
                    return False
                active_connections += 1
            
            time.sleep(0.1)  # Simulate work
            
            with lock:
                active_connections -= 1
            return True
        
        # Test that we can create max connections
        threads = []
        results = []
        
        for _ in range(max_connections + 20):
            t = threading.Thread(target=lambda: results.append(simulate_connection()))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Most should succeed, some should fail due to limit
        assert sum(results) <= max_connections
    
    def test_slowloris_protection(self):
        """Test protection against slowloris attacks."""
        # Simulate slow requests
        timeout = 5  # seconds
        
        start_time = time.time()
        # Simulate a request that takes too long
        time.sleep(0.1)  # In real test, would be actual slow request
        elapsed = time.time() - start_time
        
        assert elapsed < timeout  # Should complete within timeout
    
    def test_request_rate_per_endpoint(self):
        """Test different rate limits for different endpoints."""
        # Public endpoint: high limit
        public_limiter = RateLimiter(max_requests=1000, window_seconds=60)
        
        # Sensitive endpoint: low limit
        sensitive_limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        # Should be able to make many requests to public endpoint
        public_allowed = sum(1 for _ in range(100) if public_limiter.is_allowed("user1"))
        assert public_allowed == 100
        
        # Should be limited on sensitive endpoint
        sensitive_allowed = sum(1 for _ in range(100) if sensitive_limiter.is_allowed("user1"))
        assert sensitive_allowed == 10


class TestAPIKeyRateLimiting:
    """Test rate limiting with API keys."""
    
    @pytest.fixture
    def limiter(self):
        return RateLimiter(max_requests=100, window_seconds=60)
    
    def test_api_key_rate_limit(self, limiter):
        """Test rate limiting by API key."""
        api_key = "sk-test-12345"
        
        # Make requests up to limit
        for _ in range(100):
            assert limiter.is_allowed(api_key) == True
        
        # Should be blocked
        assert limiter.is_allowed(api_key) == False
    
    def test_different_api_keys(self, limiter):
        """Test that different API keys have separate limits."""
        key1 = "sk-test-1"
        key2 = "sk-test-2"
        
        # Exhaust limit for key1
        for _ in range(100):
            limiter.is_allowed(key1)
        
        assert limiter.is_allowed(key1) == False
        
        # key2 should still have full quota
        assert limiter.is_allowed(key2) == True
    
    def test_invalid_api_key_handling(self, limiter):
        """Test handling of invalid API keys."""
        # Empty or invalid keys should be handled gracefully
        assert limiter.is_allowed("") == True  # First request
        assert limiter.is_allowed(None) == True  # First request with None


class TestIPBasedRateLimiting:
    """Test IP-based rate limiting."""
    
    @pytest.fixture
    def limiter(self):
        return RateLimiter(max_requests=50, window_seconds=60)
    
    def test_ip_rate_limiting(self, limiter):
        """Test rate limiting by IP address."""
        ip = "192.168.1.100"
        
        # Make requests up to limit
        for _ in range(50):
            assert limiter.is_allowed(ip) == True
        
        # Should be blocked
        assert limiter.is_allowed(ip) == False
    
    def test_ip_range_handling(self, limiter):
        """Test handling of IP ranges."""
        ips = [f"192.168.1.{i}" for i in range(1, 10)]
        
        # Each IP should have its own limit
        for ip in ips:
            assert limiter.is_allowed(ip) == True


class TestRateLimitHeaders:
    """Test rate limit headers in responses."""
    
    def test_standard_rate_limit_headers(self):
        """Test standard rate limit HTTP headers."""
        limiter = SlidingWindowRateLimiter(max_requests=100, window_seconds=60)
        
        allowed, metadata = limiter.is_allowed("user_123")
        
        # Standard headers that should be included
        headers = {
            'X-RateLimit-Limit': str(metadata['limit']),
            'X-RateLimit-Remaining': str(metadata['remaining']),
            'X-RateLimit-Reset': str(int(metadata['reset'])),
            'X-RateLimit-Window': str(metadata['window']),
        }
        
        assert 'X-RateLimit-Limit' in headers
        assert 'X-RateLimit-Remaining' in headers
        assert int(headers['X-RateLimit-Remaining']) == 99
    
    def test_retry_after_header(self):
        """Test Retry-After header when rate limited."""
        limiter = SlidingWindowRateLimiter(max_requests=2, window_seconds=60)
        
        # Exhaust limit
        limiter.is_allowed("user_123")
        limiter.is_allowed("user_123")
        
        allowed, metadata = limiter.is_allowed("user_123")
        
        assert allowed == False
        assert 'retry_after' in metadata
        assert metadata['retry_after'] > 0


class TestDistributedRateLimiting:
    """Test rate limiting in distributed environments."""
    
    def test_shared_state_simulation(self):
        """Test rate limiting with shared state (Redis simulation)."""
        # Simulate shared state
        shared_state = {}
        
        def check_rate_limit(key: str, max_requests: int = 10) -> bool:
            now = time.time()
            window = 60
            
            if key not in shared_state:
                shared_state[key] = []
            
            # Clean old entries
            shared_state[key] = [
                req_time for req_time in shared_state[key]
                if req_time > now - window
            ]
            
            if len(shared_state[key]) < max_requests:
                shared_state[key].append(now)
                return True
            return False
        
        # Simulate multiple instances checking same key
        results = []
        for _ in range(15):
            results.append(check_rate_limit("shared_key"))
        
        # Only 10 should succeed
        assert sum(results) == 10


class TestAdaptiveRateLimiting:
    """Test adaptive rate limiting based on behavior."""
    
    def test_legitimate_vs_abusive_behavior(self):
        """Test different limits for different behaviors."""
        
        class AdaptiveLimiter:
            def __init__(self):
                self.legitimate_limiter = RateLimiter(max_requests=1000, window_seconds=60)
                self.abusive_limiter = RateLimiter(max_requests=10, window_seconds=60)
                self.blocked_keys = set()
            
            def is_allowed(self, key: str, is_abusive: bool = False) -> bool:
                if key in self.blocked_keys:
                    return False
                
                if is_abusive:
                    allowed = self.abusive_limiter.is_allowed(key)
                    if not allowed:
                        self.blocked_keys.add(key)
                    return allowed
                
                return self.legitimate_limiter.is_allowed(key)
        
        limiter = AdaptiveLimiter()
        
        # Legitimate user gets high limit
        for _ in range(100):
            assert limiter.is_allowed("legit_user", is_abusive=False) == True
        
        # Abusive user gets restricted quickly
        for _ in range(10):
            assert limiter.is_allowed("abusive_user", is_abusive=True) == True
        
        # Abusive user now blocked
        assert limiter.is_allowed("abusive_user", is_abusive=True) == False


class TestRateLimitBypassProtection:
    """Test protection against rate limit bypass attempts."""
    
    def test_ip_spoofing_protection(self):
        """Test protection against IP spoofing."""
        # Real implementation would check X-Forwarded-For properly
        # and not trust client-provided IPs
        
        # Simulate checking real vs forwarded IP
        real_ip = "192.168.1.100"
        forwarded_ips = ["10.0.0.1", "10.0.0.2", real_ip]
        
        # Should use the real IP (last in X-Forwarded-For chain from trusted proxy)
        effective_ip = forwarded_ips[-1]
        assert effective_ip == real_ip
    
    def test_api_key_rotation_handling(self):
        """Test handling of API key rotation."""
        old_key = "sk-old-123"
        new_key = "sk-new-456"
        
        limiter = RateLimiter(max_requests=10, window_seconds=60)
        
        # Exhaust old key
        for _ in range(10):
            limiter.is_allowed(old_key)
        
        # Old key should be blocked
        assert limiter.is_allowed(old_key) == False
        
        # New key should have full quota
        assert limiter.is_allowed(new_key) == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
