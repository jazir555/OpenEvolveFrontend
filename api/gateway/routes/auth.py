"""
Authentication and User Management Routes
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

from fastapi import APIRouter, Depends, HTTPException, status
from models.schemas import (
    UserRegister,
    UserLogin,
    Token,
    TokenRefresh,
    UserProfile,
    UserUpdate,
)
from middleware.auth import (
    get_current_user,
    create_access_token,
    create_refresh_token,
    get_password_hash,
    verify_password,
)
from utils.responses import success, created
from utils.errors import ConflictError, ValidationError
from datetime import timedelta
import logging

router = APIRouter(prefix="/auth", tags=["Authentication"])
logger = logging.getLogger(__name__)

# Mock user database (replace with real database in production)
users_db = {}


@router.post("/register", response_model=UserProfile, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserRegister):
    """
    Register a new user account

    Args:
        user_data: User registration data

    Returns:
        UserProfile: Created user profile

    Raises:
        ConflictError: If user already exists
    """
    # Check if user already exists
    if user_data.email in users_db:
        raise ConflictError(message="User with this email already exists")

    # Hash password
    hashed_password = get_password_hash(user_data.password)

    # Create user (in production, save to database)
    user_id = f"user_{len(users_db) + 1}"
    users_db[user_data.email] = {
        "user_id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password,
        "role": "user",
        "created_at": "2025-01-05T00:00:00Z",
    }

    logger.info(f"New user registered: {user_data.email}")

    # Return user profile (without password)
    return UserProfile(
        user_id=user_id,
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        role="user",
        created_at="2025-01-05T00:00:00Z",
    )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Authenticate user and return JWT tokens

    Args:
        credentials: User login credentials

    Returns:
        Token: Access and refresh tokens

    Raises:
        ValidationError: If credentials are invalid
    """
    # Find user
    user = users_db.get(credentials.email)
    if not user:
        raise ValidationError(message="Invalid email or password")

    # Verify password
    if not verify_password(credentials.password, user["hashed_password"]):
        raise ValidationError(message="Invalid email or password")

    # Create tokens
    access_token = create_access_token(
        data={
            "sub": user["user_id"],
            "email": user["email"],
            "username": user["username"],
        }
    )

    refresh_token = create_refresh_token(
        data={
            "sub": user["user_id"],
            "email": user["email"],
            "username": user["username"],
        }
    )

    logger.info(f"User logged in: {credentials.email}")

    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=1800,  # 30 minutes
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(token_data: TokenRefresh):
    """
    Refresh access token using refresh token

    Args:
        token_data: Refresh token

    Returns:
        Token: New access token

    Raises:
        ValidationError: If refresh token is invalid
    """
    from middleware.auth import decode_token

    try:
        # Decode refresh token
        payload = decode_token(token_data.refresh_token)

        # Verify it's a refresh token
        if payload.get("type") != "refresh":
            raise ValidationError(message="Invalid token type")

        # Create new access token
        access_token = create_access_token(
            data={
                "sub": payload["sub"],
                "email": payload["email"],
                "username": payload["username"],
            }
        )

        return Token(
            access_token=access_token,
            refresh_token=token_data.refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise ValidationError(message="Invalid refresh token")


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: dict = Depends(get_current_user)):
    """
    Logout user (invalidate refresh token)
    In production, add token to blacklist in Redis/DB
    """
    logger.info(f"User logged out: {current_user['email']}")
    # In production, add refresh token to blacklist
    return None


@router.get("/me", response_model=UserProfile)
async def get_current_user_profile(current_user: dict = Depends(get_current_user)):
    """
    Get current user profile

    Args:
        current_user: Current authenticated user

    Returns:
        UserProfile: User profile
    """
    # Get user from database
    user_data = users_db.get(current_user["email"])
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    return UserProfile(
        user_id=user_data["user_id"],
        email=user_data["email"],
        username=user_data["username"],
        full_name=user_data.get("full_name"),
        role=user_data.get("role", "user"),
        created_at=user_data.get("created_at", "2025-01-05T00:00:00Z"),
    )


@router.put("/me", response_model=UserProfile)
async def update_current_user_profile(
    user_update: UserUpdate,
    current_user: dict = Depends(get_current_user),
):
    """
    Update current user profile

    Args:
        user_update: User update data
        current_user: Current authenticated user

    Returns:
        UserProfile: Updated user profile
    """
    # Get user from database
    user_data = users_db.get(current_user["email"])
    if not user_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    # Update fields
    if user_update.full_name is not None:
        user_data["full_name"] = user_update.full_name

    if user_update.preferences is not None:
        user_data["preferences"] = user_update.preferences

    user_data["updated_at"] = "2025-01-05T00:00:00Z"

    logger.info(f"User profile updated: {current_user['email']}")

    return UserProfile(
        user_id=user_data["user_id"],
        email=user_data["email"],
        username=user_data["username"],
        full_name=user_data.get("full_name"),
        role=user_data.get("role", "user"),
        created_at=user_data.get("created_at", "2025-01-05T00:00:00Z"),
        updated_at=user_data.get("updated_at"),
        preferences=user_data.get("preferences", {}),
    )
