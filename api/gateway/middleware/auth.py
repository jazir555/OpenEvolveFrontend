"""
JWT Authentication Middleware
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Auth
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt, jwk
from passlib.context import CryptContext
from datetime import datetime, timedelta
from typing import Optional
import os
from dotenv import load_dotenv
import time
import logging
import httpx

load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Clerk JWT Configuration (optional)
CLERK_ISSUER = os.getenv("CLERK_ISSUER", "").strip()
CLERK_JWKS_URL = os.getenv("CLERK_JWKS_URL", "").strip()
CLERK_AUDIENCE = os.getenv("CLERK_AUDIENCE", "").strip()
CLERK_JWKS_CACHE_TTL_SECONDS = int(
    os.getenv("CLERK_JWKS_CACHE_TTL_SECONDS", "3600")
)

logger = logging.getLogger(__name__)

_clerk_jwks_cache = {
    "fetched_at": 0.0,
    "keys": [],
}

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a JWT refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        payload.setdefault("token_source", "internal")
        return payload
    except JWTError as internal_error:
        clerk_payload = _decode_clerk_token(token)
        if clerk_payload is not None:
            clerk_payload.setdefault("token_source", "clerk")
            clerk_payload.setdefault("type", "access")
            return clerk_payload
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        ) from internal_error


def _resolve_clerk_jwks_url() -> str:
    if CLERK_JWKS_URL:
        return CLERK_JWKS_URL
    if CLERK_ISSUER:
        return f"{CLERK_ISSUER.rstrip('/')}/.well-known/jwks.json"
    return ""


def _get_clerk_jwks(force_refresh: bool = False) -> list[dict]:
    jwks_url = _resolve_clerk_jwks_url()
    if not jwks_url:
        return []

    now = time.time()
    cache_age = now - _clerk_jwks_cache["fetched_at"]
    if not force_refresh and _clerk_jwks_cache["keys"] and cache_age < CLERK_JWKS_CACHE_TTL_SECONDS:
        return _clerk_jwks_cache["keys"]

    try:
        response = httpx.get(jwks_url, timeout=5.0)
        response.raise_for_status()
        data = response.json()
        keys = data.get("keys", [])
        _clerk_jwks_cache["keys"] = keys
        _clerk_jwks_cache["fetched_at"] = now
        return keys
    except Exception as exc:
        logger.warning("Failed to fetch Clerk JWKS from %s: %s", jwks_url, exc)
        return []


def _decode_clerk_token(token: str) -> Optional[dict]:
    if not (CLERK_ISSUER or CLERK_JWKS_URL):
        return None

    try:
        header = jwt.get_unverified_header(token)
    except JWTError:
        return None

    kid = header.get("kid")
    if not kid:
        return None

    keys = _get_clerk_jwks()
    key_data = next((key for key in keys if key.get("kid") == kid), None)
    if key_data is None:
        keys = _get_clerk_jwks(force_refresh=True)
        key_data = next((key for key in keys if key.get("kid") == kid), None)
        if key_data is None:
            return None

    try:
        signing_key = jwk.construct(key_data)
        decode_kwargs = {}
        if CLERK_AUDIENCE:
            decode_kwargs["audience"] = CLERK_AUDIENCE
        if CLERK_ISSUER:
            decode_kwargs["issuer"] = CLERK_ISSUER
        return jwt.decode(
            token,
            signing_key,
            algorithms=[key_data.get("alg", "RS256")],
            **decode_kwargs,
        )
    except JWTError as exc:
        logger.info("Clerk token validation failed: %s", exc)
        return None


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    """
    Get the current authenticated user from JWT token
    Dependency for protected endpoints
    """
    token = credentials.credentials
    payload = decode_token(token)

    # Check token type (Clerk tokens do not include "type")
    token_type = payload.get("type")
    if token_type and token_type != "access":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type",
        )

    # Extract user information
    user_id: str = payload.get("sub") or payload.get("user_id")
    email: Optional[str] = payload.get("email")
    if not email:
        email_addresses = payload.get("email_addresses")
        if isinstance(email_addresses, list) and email_addresses:
            first_email = email_addresses[0]
            if isinstance(first_email, dict):
                email = first_email.get("email_address")

    username: Optional[str] = (
        payload.get("username")
        or payload.get("preferred_username")
        or payload.get("name")
    )

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
        )

    return {
        "user_id": user_id,
        "email": email,
        "username": username,
    }


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[dict]:
    """
    Get the current user if authenticated, otherwise None
    For endpoints that work both authenticated and unauthenticated
    """
    if credentials is None:
        return None

    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


# Rate limit based on user
def get_rate_limit_key(user: Optional[dict] = None) -> str:
    """Get the rate limit key for a user"""
    if user and "user_id" in user:
        return f"user:{user['user_id']}"
    # Fallback to IP-based limiting (would need to be passed in)
    return "ip:unknown"


class TokenData:
    """Token data model"""
    user_id: str
    email: str
    username: str
    exp: datetime
    type: str

    @classmethod
    def from_payload(cls, payload: dict) -> "TokenData":
        """Create TokenData from JWT payload"""
        return cls(
            user_id=payload.get("sub"),
            email=payload.get("email"),
            username=payload.get("username"),
            exp=datetime.fromtimestamp(payload.get("exp")),
            type=payload.get("type"),
        )
