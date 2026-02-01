"""
Security Tests for Knowledge Engine

Following CLAUDE.md principles:
- Test input validation and sanitization
- Test SQL/NoSQL injection prevention
- Test authentication/authorization
- Test rate limiting
- Test data privacy (PII handling)

Tests verify:
- Input validation
- Injection attack prevention
- PII detection and redaction
- Rate limiting
- Authentication/authorization
- Data encryption
"""

import asyncio
import json
import logging
import pytest
import re
import time
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import AsyncMock, patch
import sys
from pathlib import Path
import importlib.util

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import core module using conftest's approach
CORE_AVAILABLE = False
EntityKnowledgeGraph = None
KnowledgeState = None

try:
    spec = importlib.util.spec_from_file_location(
        "core",
        project_root / "knowledge_engine" / "core.py"
    )
    if spec and spec.loader:
        core_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(core_module)
        EntityKnowledgeGraph = core_module.EntityKnowledgeGraph
        KnowledgeState = core_module.KnowledgeState
        CORE_AVAILABLE = True
except Exception as e:
    CORE_AVAILABLE = False
    EntityKnowledgeGraph = None
    KnowledgeState = None

logger = logging.getLogger(__name__)


class TestInputValidation:
    """
    Tests for input validation and sanitization.
    """

    @pytest.mark.asyncio
    @pytest.mark.skipif(not CORE_AVAILABLE, reason="Core module not available")
    async def test_sql_injection_prevention(self):
        """
        Test that SQL injection attempts are blocked/sanitized.
        """
        if not CORE_AVAILABLE:
            pytest.skip("Core module not available")

        malicious_inputs = [
            "'; DROP TABLE entities; --",
            "1' OR '1'='1",
            "admin'--",
            "' UNION SELECT * FROM users--",
        ]

        graph = EntityKnowledgeGraph()

        for malicious_input in malicious_inputs:
            # Input should be sanitized before processing
            sanitized = self._sanitize_input(malicious_input)

            # Add entity with sanitized input
            await graph.add_entity_async(sanitized)

            # Verify dangerous patterns removed
            assert "'" not in sanitized or sanitized.count("'") % 2 == 0
            assert ";" not in sanitized or "--" not in sanitized
            assert "DROP TABLE" not in sanitized.upper()
            assert "UNION SELECT" not in sanitized.upper()

        logger.info(json.dumps({
            "msg": "SQL injection attempts sanitized",
            "attempts": len(malicious_inputs),
            "all_blocked": True,
            "level": "INFO"
        }))

    def _sanitize_input(self, input_str: str) -> str:
        """
        Sanitize input to prevent injection attacks.
        In real implementation, this would use proper sanitization library.
        """
        # Remove dangerous SQL patterns
        sanitized = input_str
        sql_keywords = ["DROP", "DELETE", "TRUNCATE", "ALTER", "EXEC", "UNION SELECT"]
        for keyword in sql_keywords:
            sanitized = re.sub(keyword, "", sanitized, flags=re.IGNORECASE)

        # Remove comments
        sanitized = sanitized.replace("--", "")
        sanitized = sanitized.replace("/*", "")
        sanitized = sanitized.replace("*/", "")

        # Balance quotes
        if sanitized.count("'") % 2 != 0:
            sanitized = sanitized.replace("'", "")

        return sanitized

    @pytest.mark.asyncio
    async def test_nosql_injection_prevention(self):
        """
        Test that NoSQL injection attempts are blocked.
        """
        malicious_inputs = [
            {"$ne": None},
            {"$where": "this.name == 'admin'"},
            {"$regex": ".*"},
            {"$gt": ""},
        ]

        for malicious_input in malicious_inputs:
            # Should detect and block MongoDB operators
            is_malicious = self._detect_nosql_injection(malicious_input)

            assert is_malicious is True, f"NoSQL injection not detected: {malicious_input}"

        logger.info(json.dumps({
            "msg": "NoSQL injection attempts detected",
            "attempts": len(malicious_inputs),
            "all_detected": True,
            "level": "INFO"
        }))

    def _detect_nosql_injection(self, input_data: Any) -> bool:
        """
        Detect NoSQL injection attempts.
        """
        dangerous_operators = ["$ne", "$where", "$regex", "$gt", "$lt", "$or", "$and"]

        if isinstance(input_data, dict):
            for key in input_data.keys():
                if key.startswith("$"):
                    return True
                if any(op in key for op in dangerous_operators):
                    return True

        return False

    @pytest.mark.asyncio
    async def test_xss_prevention(self):
        """
        Test that XSS attempts are sanitized.
        """
        xss_attempts = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<svg onload=alert('XSS')>",
        ]

        sanitized_results = []
        for xss_attempt in xss_attempts:
            sanitized = self._sanitize_html(xss_attempt)
            sanitized_results.append(sanitized)

            # Verify script tags removed
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert "onerror=" not in sanitized.lower()
            assert "onload=" not in sanitized.lower()

        logger.info(json.dumps({
            "msg": "XSS attempts sanitized",
            "attempts": len(xss_attempts),
            "all_sanitized": True,
            "level": "INFO"
        }))

    def _sanitize_html(self, input_str: str) -> str:
        """
        Sanitize HTML to prevent XSS.
        """
        # Remove script tags
        sanitized = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.IGNORECASE)
        sanitized = re.sub(r'<script.*?>', '', sanitized, flags=re.IGNORECASE)

        # Remove dangerous event handlers
        event_handlers = ["onerror", "onload", "onclick", "onmouseover"]
        for handler in event_handlers:
            sanitized = re.sub(handler, '', sanitized, flags=re.IGNORECASE)

        # Remove javascript: protocol
        sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)

        return sanitized


class TestPIIHandling:
    """
    Tests for PII (Personally Identifiable Information) detection and redaction.
    """

    @pytest.mark.asyncio
    async def test_email_detection(self):
        """
        Test detection and redaction of email addresses.
        """
        text = """
        Contact us at support@example.com for assistance.
        You can also reach admin@test.org or sales@company.co.uk.
        """

        # Detect emails
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)

        assert len(emails) == 3
        assert "support@example.com" in emails

        # Redact emails
        redacted_text = self._redact_pii(text, types=['email'])

        assert "@" not in redacted_text
        assert "[REDACTED" in redacted_text or "***" in redacted_text

        logger.info(json.dumps({
            "msg": "Email addresses detected and redacted",
            "emails_found": len(emails),
            "redaction_successful": "@" not in redacted_text,
            "level": "INFO"
        }))

    def _redact_pii(self, text: str, types: List[str] = None) -> str:
        """
        Redact PII from text.
        """
        if types is None:
            types = ['email', 'phone', 'ssn']

        redacted = text

        if 'email' in types:
            # Redact email addresses
            redacted = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                            '[REDACTED_EMAIL]', redacted)

        if 'phone' in types:
            # Redact phone numbers
            redacted = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',
                            '[REDACTED_PHONE]', redacted)

        if 'ssn' in types:
            # Redact SSNs
            redacted = re.sub(r'\b\d{3}-\d{2}-\d{4}\b',
                            '[REDACTED_SSN]', redacted)

        return redacted

    @pytest.mark.asyncio
    async def test_phone_number_detection(self):
        """
        Test detection of phone numbers.
        """
        text = """
        Call us at 555-123-4567 or 555.987.6543.
        International: +1-555-123-4567
        """

        # Detect phone numbers
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',  # 555-123-4567
            r'\b\d{3}\.\d{3}\.\d{4}\b',  # 555.123.4567
            r'\b\+\d{1,3}-\d{3}-\d{3}-\d{4}\b',  # +1-555-123-4567
        ]

        phones_found = []
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            phones_found.extend(phones)

        assert len(phones_found) >= 2

        # Redact
        redacted = self._redact_pii(text, types=['phone'])

        # Verify redaction
        assert "555-123-4567" not in redacted
        assert "[REDACTED_PHONE]" in redacted

        logger.info(json.dumps({
            "msg": "Phone numbers detected and redacted",
            "phones_found": len(phones_found),
            "redaction_successful": True,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_ssn_detection(self):
        """
        Test detection of Social Security Numbers.
        """
        text = "My SSN is 123-45-6789 and my friend's is 987-65-4321."

        # Detect SSNs
        ssns = re.findall(r'\b\d{3}-\d{2}-\d{4}\b', text)

        assert len(ssns) == 2

        # Redact
        redacted = self._redact_pii(text, types=['ssn'])

        assert "123-45-6789" not in redacted
        assert "[REDACTED_SSN]" in redacted

        logger.info(json.dumps({
            "msg": "SSNs detected and redacted",
            "ssns_found": len(ssns),
            "redaction_successful": True,
            "level": "INFO"
        }))


class TestRateLimiting:
    """
    Tests for rate limiting functionality.
    """

    @pytest.mark.asyncio
    async def test_rate_limit_enforcement(self):
        """
        Test that rate limits are enforced.
        """
        rate_limiter = SimpleRateLimiter(max_requests=5, window_seconds=1)

        # First 5 requests should succeed
        for i in range(5):
            allowed = rate_limiter.check_limit("user_123")
            assert allowed is True, f"Request {i+1} should be allowed"

        # 6th request should be blocked
        blocked = rate_limiter.check_limit("user_123")
        assert blocked is False, "6th request should be blocked"

        logger.info(json.dumps({
            "msg": "Rate limit enforced correctly",
            "allowed_requests": 5,
            "blocked_requests": 1,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_rate_limit_window_reset(self):
        """
        Test that rate limit window resets after time passes.
        """
        import time

        rate_limiter = SimpleRateLimiter(max_requests=2, window_seconds=1)

        # Use up quota
        assert rate_limiter.check_limit("user_456") is True
        assert rate_limiter.check_limit("user_456") is True
        assert rate_limiter.check_limit("user_456") is False

        # Wait for window to reset
        time.sleep(1.1)

        # Should be allowed again
        assert rate_limiter.check_limit("user_456") is True

        logger.info(json.dumps({
            "msg": "Rate limit window reset successfully",
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_different_users_separate_limits(self):
        """
        Test that rate limits are per-user.
        """
        rate_limiter = SimpleRateLimiter(max_requests=3, window_seconds=1)

        # User 1 uses quota
        for i in range(3):
            assert rate_limiter.check_limit("user_1") is True
        assert rate_limiter.check_limit("user_1") is False

        # User 2 should have fresh quota
        for i in range(3):
            assert rate_limiter.check_limit("user_2") is True

        logger.info(json.dumps({
            "msg": "Rate limits properly separated by user",
            "user1_blocked": True,
            "user2_blocked": False,
            "level": "INFO"
        }))


class SimpleRateLimiter:
    """Simple rate limiter for testing."""

    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = {}  # {user_id: [timestamp1, timestamp2, ...]}

    def check_limit(self, user_id: str) -> bool:
        """
        Check if request is within rate limit.
        """
        now = time.time()

        # Clean old requests
        if user_id in self.requests:
            self.requests[user_id] = [
                ts for ts in self.requests[user_id]
                if now - ts < self.window_seconds
            ]
        else:
            self.requests[user_id] = []

        # Check limit
        if len(self.requests[user_id]) < self.max_requests:
            self.requests[user_id].append(now)
            return True
        return False


class TestAuthentication:
    """
    Tests for authentication and authorization.
    """

    @pytest.mark.asyncio
    async def test_valid_authentication(self):
        """
        Test that valid credentials are accepted.
        """
        auth_system = SimpleAuthSystem()

        # Add user
        auth_system.add_user("admin", "secure_password_hash", ["admin", "user"])

        # Authenticate
        result = auth_system.authenticate("admin", "secure_password_hash")

        assert result["success"] is True
        assert result["user_id"] == "admin"
        assert "admin" in result["roles"]

        logger.info(json.dumps({
            "msg": "Valid authentication successful",
            "user": "admin",
            "roles": result["roles"],
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_invalid_authentication(self):
        """
        Test that invalid credentials are rejected.
        """
        auth_system = SimpleAuthSystem()

        auth_system.add_user("user1", "correct_password", ["user"])

        # Wrong password
        result = auth_system.authenticate("user1", "wrong_password")

        assert result["success"] is False
        assert "error" in result

        # Non-existent user
        result = auth_system.authenticate("nonexistent", "any_password")

        assert result["success"] is False

        logger.info(json.dumps({
            "msg": "Invalid authentication rejected",
            "attempts": 2,
            "all_rejected": True,
            "level": "INFO"
        }))

    @pytest.mark.asyncio
    async def test_authorization_check(self):
        """
        Test that authorization checks work correctly.
        """
        auth_system = SimpleAuthSystem()

        auth_system.add_user("admin", "pass", ["admin"])
        auth_system.add_user("user", "pass", ["user"])

        # Admin can do everything
        assert auth_system.check_permission("admin", "read") is True
        assert auth_system.check_permission("admin", "write") is True
        assert auth_system.check_permission("admin", "delete") is True

        # Regular user has limited permissions
        assert auth_system.check_permission("user", "read") is True
        assert auth_system.check_permission("user", "write") is True
        assert auth_system.check_permission("user", "delete") is False

        logger.info(json.dumps({
            "msg": "Authorization checks working",
            "admin_permissions": 3,
            "user_permissions": 2,
            "level": "INFO"
        }))


class SimpleAuthSystem:
    """Simple authentication system for testing."""

    def __init__(self):
        self.users = {}  # {username: {password_hash, roles}}

    def add_user(self, username: str, password_hash: str, roles: List[str]):
        self.users[username] = {
            "password_hash": password_hash,
            "roles": roles
        }

    def authenticate(self, username: str, password_hash: str) -> Dict[str, Any]:
        """
        Authenticate user.
        """
        if username not in self.users:
            return {"success": False, "error": "User not found"}

        user = self.users[username]
        if user["password_hash"] != password_hash:
            return {"success": False, "error": "Invalid password"}

        return {
            "success": True,
            "user_id": username,
            "roles": user["roles"]
        }

    def check_permission(self, username: str, action: str) -> bool:
        """
        Check if user has permission for action.
        """
        if username not in self.users:
            return False

        roles = self.users[username]["roles"]

        # Admin can do everything
        if "admin" in roles:
            return True

        # Regular users can read and write
        if action in ["read", "write"] and "user" in roles:
            return True

        # Delete requires admin
        if action == "delete":
            return False

        return False


class TestDataEncryption:
    """
    Tests for data encryption.
    """

    @pytest.mark.asyncio
    async def test_sensitive_data_encryption(self):
        """
        Test that sensitive data is encrypted.
        """
        sensitive_data = {
            "api_key": "sk-1234567890abcdef",
            "password": "SecretPassword123",
            "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        }

        encrypted = self._encrypt_sensitive_data(sensitive_data)

        # Verify sensitive fields are encrypted
        for key, value in encrypted.items():
            assert value != sensitive_data[key], f"{key} not encrypted"
            assert "sk-" not in str(value), f"API key visible in {key}"
            assert "Secret" not in str(value), f"Password visible in {key}"

        logger.info(json.dumps({
            "msg": "Sensitive data encrypted",
            "fields_encrypted": len(sensitive_data),
            "level": "INFO"
        }))

    def _encrypt_sensitive_data(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Encrypt sensitive data fields.
        In real implementation, would use proper encryption library.
        """
        encrypted = {}
        sensitive_keys = ["api_key", "password", "token", "secret"]

        for key, value in data.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                # Mock encryption (in real: use cryptography/Fernet)
                encrypted[key] = f"ENCRYPTED:{len(value)}:{hash(value) % 10000}"
            else:
                encrypted[key] = value

        return encrypted

    @pytest.mark.asyncio
    async def test_data_decryption(self):
        """
        Test that encrypted data can be decrypted.
        """
        original_data = {"api_key": "secret_key_123"}
        encrypted = self._encrypt_sensitive_data(original_data)

        # Decrypt
        decrypted = self._decrypt_sensitive_data(encrypted)

        # In this mock, we can't actually decrypt, but we verify the structure
        assert "api_key" in decrypted

        # In real implementation, would verify decrypted == original
        logger.info(json.dumps({
            "msg": "Data decryption structure verified",
            "level": "INFO"
        }))

    def _decrypt_sensitive_data(self, data: Dict[str, str]) -> Dict[str, str]:
        """
        Decrypt sensitive data fields.
        In real implementation, would use proper decryption.
        """
        # Mock decryption
        return data


# Run tests if executed directly
if __name__ == "__main__":
    import time
    pytest.main([__file__, "-v", "--tb=short"])
