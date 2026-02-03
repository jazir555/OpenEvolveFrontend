"""
Request validation utilities
"""

# **ACTUAL INTEGRATION**: Adaptive MDAP for Validators
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator
import re


def validate_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None


def validate_username(username: str) -> bool:
    """Validate username format (alphanumeric, underscores, hyphens)"""
    pattern = r'^[a-zA-Z0-9_-]{3,50}$'
    return re.match(pattern, username) is not None


def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength
    Returns (is_valid, error_message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"

    if len(password) > 100:
        return False, "Password must not exceed 100 characters"

    # Check for at least one uppercase letter
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"

    # Check for at least one lowercase letter
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"

    # Check for at least one digit
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"

    return True, None


def validate_tags(tags: List[str]) -> tuple[bool, Optional[str]]:
    """Validate tags list"""
    if not tags:
        return True, None

    if len(tags) > 20:
        return False, "Cannot have more than 20 tags"

    for tag in tags:
        if len(tag) > 50:
            return False, f"Tag '{tag}' is too long (max 50 characters)"
        if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
            return False, f"Tag '{tag}' contains invalid characters (only alphanumeric, underscore, hyphen allowed)"

    return True, None


def validate_pagination(limit: int, offset: int) -> tuple[bool, Optional[str]]:
    """Validate pagination parameters"""
    if limit < 1:
        return False, "Limit must be at least 1"
    if limit > 100:
        return False, "Limit cannot exceed 100"
    if offset < 0:
        return False, "Offset cannot be negative"

    return True, None


def validate_sort_field(field: str, allowed_fields: List[str]) -> tuple[bool, Optional[str]]:
    """Validate sort field"""
    if field not in allowed_fields:
        return False, f"Invalid sort field. Allowed fields: {', '.join(allowed_fields)}"
    return True, None


def validate_sort_order(order: str) -> tuple[bool, Optional[str]]:
    """Validate sort order"""
    if order.lower() not in ["asc", "desc"]:
        return False, "Sort order must be 'asc' or 'desc'"
    return True, None


class BaseValidator(BaseModel):
    """Base validator with common validators"""

    @validator('*')
    def strip_strings(cls, v):
        """Strip whitespace from string fields"""
        if isinstance(v, str):
            return v.strip()
        return v


def sanitize_input(text: str, max_length: Optional[int] = None) -> str:
    """
    Sanitize user input by removing potentially dangerous characters
    """
    if not text:
        return text

    # Remove null bytes
    text = text.replace('\x00', '')

    # Trim if max_length specified
    if max_length and len(text) > max_length:
        text = text[:max_length]

    return text


def validate_json_structure(data: Dict[str, Any], required_fields: List[str]) -> tuple[bool, Optional[str]]:
    """Validate that JSON data contains required fields"""
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        return False, f"Missing required fields: {', '.join(missing_fields)}"
    return True, None


def validate_file_size(size: int, max_size: int) -> tuple[bool, Optional[str]]:
    """Validate file size"""
    if size > max_size:
        return False, f"File size exceeds maximum allowed size of {max_size} bytes"
    return True, None


def validate_mime_type(mime_type: str, allowed_types: List[str]) -> tuple[bool, Optional[str]]:
    """Validate MIME type"""
    if mime_type not in allowed_types:
        return False, f"Invalid file type. Allowed types: {', '.join(allowed_types)}"
    return True, None


class RequestValidator:
    """Request validator class for complex validations"""

    @staticmethod
    def validate_evolution_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate evolution start request"""
        if "content" not in data or not data["content"].strip():
            return False, "Content is required and cannot be empty"

        if "models" not in data or not data["models"]:
            return False, "At least one model must be specified"

        for i, model in enumerate(data["models"]):
            if "provider" not in model or not model["provider"]:
                return False, f"Model {i+1}: provider is required"
            if "model" not in model or not model["model"]:
                return False, f"Model {i+1}: model name is required"
            if "api_key" not in model or not model["api_key"]:
                return False, f"Model {i+1}: api_key is required"

        # Validate parameters if present
        if "parameters" in data:
            params = data["parameters"]
            if "max_iterations" in params:
                if params["max_iterations"] < 1 or params["max_iterations"] > 1000:
                    return False, "max_iterations must be between 1 and 1000"
            if "population_size" in params:
                if params["population_size"] < 10 or params["population_size"] > 500:
                    return False, "population_size must be between 10 and 500"
            if "temperature" in params:
                if params["temperature"] < 0.0 or params["temperature"] > 2.0:
                    return False, "temperature must be between 0.0 and 2.0"

        return True, None

    @staticmethod
    def validate_adversarial_request(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Validate adversarial testing start request"""
        if "content" not in data or not data["content"].strip():
            return False, "Content is required and cannot be empty"

        if "attack_modes" not in data or not data["attack_modes"]:
            return False, "At least one attack mode must be specified"

        valid_modes = ["prompt_injection", "jailbreak", "adversarial_example", "data_poisoning"]
        for mode in data["attack_modes"]:
            if mode not in valid_modes:
                return False, f"Invalid attack mode: {mode}"

        return True, None
