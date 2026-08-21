"""Models module stub."""
from __future__ import annotations

from pydantic import BaseModel
from typing import Optional, List

class EvolutionStart(BaseModel):
    """Evolution start request."""
    pass

class EvolutionStatus(BaseModel):
    """Evolution status response."""
    pass

class EvolutionListResponse(BaseModel):
    """Evolution list response."""
    pass

class UserRegister(BaseModel):
    """User registration."""
    pass

class UserLogin(BaseModel):
    """User login."""
    pass

class Token(BaseModel):
    """Token model."""
    pass

class TokenRefresh(BaseModel):
    """Token refresh."""
    pass

class UserProfile(BaseModel):
    """User profile."""
    pass

class UserUpdate(BaseModel):
    """User update."""
    pass

class schemas:
    """Schemas namespace."""
    EvolutionStart = EvolutionStart
    EvolutionStatus = EvolutionStatus
    EvolutionListResponse = EvolutionListResponse
    UserRegister = UserRegister
    UserLogin = UserLogin
    Token = Token
    TokenRefresh = TokenRefresh
    UserProfile = UserProfile
    UserUpdate = UserUpdate
