"""RESE Z3 Schema module."""
from typing import Any, Dict, List, Optional
from enum import Enum

class VerificationTier(Enum):
    """Verification tier."""
    BASIC = "basic"
    ADVANCED = "advanced"

class VerificationStatus(Enum):
    """Verification status."""
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
