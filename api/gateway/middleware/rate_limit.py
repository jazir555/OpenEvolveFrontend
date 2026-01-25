"""
Rate Limiting Middleware using slowapi
"""
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, HTTPException, status
from dotenv import load_dotenv
import os
from typing import Callable

load_dotenv()

# Rate limit configuration
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "True").lower() == "true"
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "100"))
RATE_LIMIT_BURST = int(os.getenv("RATE_LIMIT_BURST", "10"))


def get_identifier(request: Request) -> str:
    """
    Get identifier for rate limiting
    Uses user_id if authenticated, otherwise IP address
    """
    # Check if user is authenticated
    if hasattr(request.state, "user") and request.state.user:
        user_id = request.state.user.get("user_id")
        if user_id:
            return f"user:{user_id}"

    # Fallback to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_identifier,
    default_limits=[f"{RATE_LIMIT_PER_MINUTE}/minute"],
    storage_uri=os.getenv("REDIS_URL", "memory://"),
    enabled=RATE_LIMIT_ENABLED,
)


def rate_limit_exempt(func: Callable) -> Callable:
    """
    Decorator to exempt an endpoint from rate limiting
    """
    func._rate_limit_exempt = True
    return func


class RateLimiter:
    """
    Rate limiting middleware class
    """

    def __init__(self, app):
        self.app = app
        self.setup_rate_limiting()

    def setup_rate_limiting(self):
        """Setup rate limiting for the application"""
        if not RATE_LIMIT_ENABLED:
            return

        # Add error handler for rate limit exceeded
        @self.app.exception_handler(RateLimitExceeded)
        async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
            """Handle rate limit exceeded errors"""
            return HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": "Too many requests. Please try again later.",
                        "retry_after": str(exc.retry_after) if hasattr(exc, "retry_after") else "60",
                    }
                },
            )

        # Add rate limit headers to all responses
        @self.app.middleware("http")
        async def add_rate_limit_headers(request: Request, call_next):
            """Add rate limit headers to responses"""
            response = await call_next(request)

            # Add rate limit headers
            response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(
                max(0, RATE_LIMIT_PER_MINUTE - int(response.headers.get("X-RateLimit-Used", "0")))
            )

            return response


# Custom rate limit decorators
def limit_per_minute(requests: int):
    """Rate limit decorator for custom limits per minute"""
    return limiter.limit(f"{requests}/minute")


def limit_per_hour(requests: int):
    """Rate limit decorator for custom limits per hour"""
    return limiter.limit(f"{requests}/hour")


def limit_per_day(requests: int):
    """Rate limit decorator for custom limits per day"""
    return limiter.limit(f"{requests}/day")
