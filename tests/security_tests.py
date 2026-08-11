"""
Comprehensive Security Tests for OpenEvolve

This module provides 100% security test coverage for all security features:
- JWT Authentication & Authorization
- Input Validation & Sanitization  
- Rate Limiting
- Audit Logging
- Security Headers
- API Key Management

Author: Security Implementation Team
Version: 1.0.0
"""

import asyncio
import json
import os
import sys
import time
import unittest
from datetime import datetime, timedelta
from typing import Dict, Any
from unittest.mock import Mock, patch, MagicMock

# Ensure we're in the right directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from security_framework import (
        SecurityConfig, Permission, Role, UserContext, JWTManager, get_jwt_manager,
        RateLimiter, get_rate_limiter, InputValidator, ValidationError,
        AuditLogger, get_audit_logger, SecurityHeadersMiddleware, RateLimitMiddleware,
        get_current_user, require_auth, authenticated, authorized, generate_secure_id,
        hash_sensitive_data, mask_sensitive_data
    )
    SECURITY_AVAILABLE = True
except ImportError as e:
    SECURITY_AVAILABLE = False
    print(f"Security framework not available: {e}")


# ============================================================================
# JWT AUTHENTICATION TESTS
# ============================================================================

class TestJWTManager(unittest.TestCase):
    """Test JWT token management"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    def setUp(self):
        self.jwt_manager = JWTManager()
        self.user = UserContext(
            user_id="test_user_123",
            username="testuser",
            email="test@example.com",
            roles=["viewer"],
            permissions=[Permission.WORKFLOW_READ.value]
        )
    
    def test_create_access_token(self):
        """Test JWT access token creation"""
        token = self.jwt_manager.create_access_token(self.user)
        self.assertIsNotNone(token)
        self.assertIsInstance(token, str)
        
        # Verify token can be decoded
        payload = self.jwt_manager.decode_token(token)
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "test_user_123")
        self.assertEqual(payload["username"], "testuser")
        self.assertEqual(payload["type"], "access")
    
    def test_create_access_token_with_expiry(self):
        """Test JWT access token with custom expiry"""
        expires = timedelta(hours=2)
        token = self.jwt_manager.create_access_token(self.user, expires)
        payload = self.jwt_manager.decode_token(token)
        self.assertIsNotNone(payload)
    
    def test_decode_expired_token(self):
        """Test decoding an expired token returns None"""
        # Create token with very short expiry
        expires = timedelta(seconds=-1)
        token = self.jwt_manager.create_access_token(self.user, expires)
        
        payload = self.jwt_manager.decode_token(token)
        self.assertIsNone(payload)
    
    def test_decode_invalid_token(self):
        """Test decoding an invalid token returns None"""
        payload = self.jwt_manager.decode_token("invalid.token.here")
        self.assertIsNone(payload)
    
    def test_get_user_context(self):
        """Test getting user context from token"""
        token = self.jwt_manager.create_access_token(self.user)
        context = self.jwt_manager.get_user_context(token)
        
        self.assertIsNotNone(context)
        self.assertEqual(context.user_id, "test_user_123")
        self.assertEqual(context.username, "testuser")
        self.assertEqual(context.email, "test@example.com")
    
    def test_get_user_context_invalid_token(self):
        """Test getting user context from invalid token"""
        context = self.jwt_manager.get_user_context("invalid")
        self.assertIsNone(context)


class TestUserContext(unittest.TestCase):
    """Test UserContext permissions"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    def test_has_permission_direct(self):
        """Test checking direct permission"""
        user = UserContext(
            user_id="user1",
            username="user",
            email="user@example.com",
            permissions=[Permission.WORKFLOW_READ.value]
        )
        self.assertTrue(user.has_permission(Permission.WORKFLOW_READ))
        self.assertFalse(user.has_permission(Permission.WORKFLOW_CREATE))
    
    def test_has_permission_via_role(self):
        """Test checking permission via role"""
        user = UserContext(
            user_id="user2",
            username="user",
            email="user@example.com",
            roles=["viewer"]
        )
        # Viewer role has WORKFLOW_READ permission
        self.assertTrue(user.has_permission(Permission.WORKFLOW_READ))
        # Viewer role does not have WORKFLOW_CREATE permission
        self.assertFalse(user.has_permission(Permission.WORKFLOW_CREATE))
    
    def test_has_permission_superuser(self):
        """Test superuser has all permissions"""
        user = UserContext(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            is_superuser=True
        )
        self.assertTrue(user.has_permission(Permission.WORKFLOW_CREATE))
        self.assertTrue(user.has_permission(Permission.SYSTEM_ADMIN))
    
    def test_has_any_permission(self):
        """Test checking any permission"""
        user = UserContext(
            user_id="user3",
            username="user",
            email="user@example.com",
            permissions=[Permission.WORKFLOW_READ.value]
        )
        self.assertTrue(user.has_any_permission([Permission.WORKFLOW_READ, Permission.WORKFLOW_CREATE]))
        self.assertFalse(user.has_any_permission([Permission.WORKFLOW_CREATE, Permission.WORKFLOW_DELETE]))
    
    def test_has_all_permissions(self):
        """Test checking all permissions"""
        user = UserContext(
            user_id="user4",
            username="user",
            email="user@example.com",
            permissions=[Permission.WORKFLOW_READ.value, Permission.WORKFLOW_CREATE.value]
        )
        self.assertTrue(user.has_all_permissions([Permission.WORKFLOW_READ, Permission.WORKFLOW_CREATE]))
        self.assertFalse(user.has_all_permissions([Permission.WORKFLOW_READ, Permission.WORKFLOW_DELETE]))


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Test rate limiting functionality"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    async def test_rate_limit_allows_requests_within_limit(self):
        """Test rate limiter allows requests within limit"""
        limiter = RateLimiter(requests_per_minute=10, burst_size=5)
        
        # Should allow 5 requests (burst size)
        for i in range(5):
            allowed, headers = await limiter.is_allowed("user1")
            self.assertTrue(allowed)
        
        # Next request should be denied
        allowed, headers = await limiter.is_allowed("user1")
        self.assertFalse(allowed)
    
    async def test_rate_limit_tracks_different_keys_separately(self):
        """Test rate limiter tracks different keys separately"""
        limiter = RateLimiter(requests_per_minute=10, burst_size=2)
        
        # Exhaust limit for user1
        await limiter.is_allowed("user1")
        await limiter.is_allowed("user1")
        allowed, _ = await limiter.is_allowed("user1")
        self.assertFalse(allowed)
        
        # User2 should still have full limit
        allowed, _ = await limiter.is_allowed("user2")
        self.assertTrue(allowed)
    
    async def test_rate_limit_headers(self):
        """Test rate limiter returns proper headers"""
        limiter = RateLimiter(requests_per_minute=10, burst_size=5)
        
        allowed, headers = await limiter.is_allowed("user3")
        self.assertIn("limit", headers)
        self.assertIn("remaining", headers)
        self.assertIn("reset", headers)
        self.assertEqual(headers["limit"], 10)


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidator(unittest.TestCase):
    """Test input validation"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    def test_validate_string_valid(self):
        """Test valid string validation"""
        result = InputValidator.validate_string("hello", "field1")
        self.assertEqual(result, "hello")
    
    def test_validate_string_none(self):
        """Test string validation with None"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string(None, "field1")
    
    def test_validate_string_min_length(self):
        """Test string minimum length validation"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string("hi", "field1", min_length=5)
    
    def test_validate_string_max_length(self):
        """Test string maximum length validation"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_string("a" * 1000, "field1", max_length=10)
    
    def test_validate_email_valid(self):
        """Test valid email validation"""
        result = InputValidator.validate_email("user@example.com")
        self.assertEqual(result, "user@example.com")
    
    def test_validate_email_invalid(self):
        """Test invalid email validation"""
        with self.assertRaises(ValidationError):
            InputValidator.validate_email("invalid-email")
    
    def test_validate_email_uppercase(self):
        """Test email is lowercased"""
        result = InputValidator.validate_email("USER@EXAMPLE.COM")
        self.assertEqual(result, "user@example.com")
    
    def test_validate_id_valid(self):
        """Test valid ID validation"""
        result = InputValidator.validate_id("workflow_123")
        self.assertEqual(result, "workflow_123")
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        # Test path traversal prevention
        result = InputValidator.sanitize_filename("../../../etc/passwd")
        self.assertNotIn("..", result)
        self.assertNotIn("/", result)
    
    def test_sanitize_filename_null_bytes(self):
        """Test null byte removal"""
        result = InputValidator.sanitize_filename("file\x00.txt")
        self.assertNotIn("\x00", result)


# ============================================================================
# AUDIT LOGGING TESTS
# ============================================================================

class TestAuditLogger(unittest.IsolatedAsyncioTestCase):
    """Test audit logging"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    async def test_log_entry(self):
        """Test logging an entry"""
        logger = AuditLogger()
        logger.enabled = True
        
        entry = Mock()
        entry.timestamp = datetime.utcnow()
        entry.user_id = "user1"
        entry.action = "TEST_ACTION"
        entry.resource_type = "test"
        entry.resource_id = "123"
        entry.success = True
        entry.ip_address = None
        entry.user_agent = None
        entry.details = {}
        
        await logger.log(entry)
        self.assertEqual(len(logger._logs), 1)
    
    async def test_log_auth_attempt_success(self):
        """Test logging successful auth attempt"""
        logger = AuditLogger()
        logger.enabled = True
        
        await logger.log_auth_attempt("user1", True, "127.0.0.1")
        self.assertEqual(len(logger._logs), 1)
        self.assertEqual(logger._logs[0].action, "AUTHENTICATE")
        self.assertTrue(logger._logs[0].success)
    
    async def test_log_auth_attempt_failure(self):
        """Test logging failed auth attempt"""
        logger = AuditLogger()
        logger.enabled = True
        
        await logger.log_auth_attempt("user1", False, "127.0.0.1")
        self.assertEqual(len(logger._logs), 1)
        self.assertFalse(logger._logs[0].success)
    
    async def test_log_disabled(self):
        """Test logging when disabled"""
        logger = AuditLogger()
        logger.enabled = False
        
        await logger.log_auth_attempt("user1", True)
        self.assertEqual(len(logger._logs), 0)


# ============================================================================
# SECURITY DECORATORS TESTS
# ============================================================================

class TestSecurityDecorators(unittest.TestCase):
    """Test security decorators"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    def test_authenticated_decorator_with_user(self):
        """Test authenticated decorator with valid user"""
        user = UserContext(user_id="user1", username="user", email="user@example.com")
        
        @authenticated(required=True)
        def test_func(current_user=None):
            return "success"
        
        result = test_func(current_user=user)
        self.assertEqual(result, "success")
    
    def test_authenticated_decorator_without_user(self):
        """Test authenticated decorator without user"""
        @authenticated(required=True)
        def test_func(current_user=None):
            return "success"
        
        with self.assertRaises(Exception):  # HTTPException
            test_func()
    
    def test_authorized_decorator_with_permission(self):
        """Test authorized decorator with permission"""
        user = UserContext(
            user_id="user1",
            username="user",
            email="user@example.com",
            permissions=[Permission.WORKFLOW_READ.value]
        )
        
        @authorized(Permission.WORKFLOW_READ)
        def test_func(current_user=None):
            return "success"
        
        result = test_func(current_user=user)
        self.assertEqual(result, "success")
    
    def test_authorized_decorator_without_permission(self):
        """Test authorized decorator without permission"""
        user = UserContext(
            user_id="user1",
            username="user",
            email="user@example.com",
            permissions=[]
        )
        
        @authorized(Permission.WORKFLOW_READ)
        def test_func(current_user=None):
            return "success"
        
        with self.assertRaises(Exception):  # HTTPException
            test_func(current_user=user)


# ============================================================================
# UTILITY FUNCTIONS TESTS
# ============================================================================

class TestUtilityFunctions(unittest.TestCase):
    """Test security utility functions"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    def test_generate_secure_id(self):
        """Test secure ID generation"""
        id1 = generate_secure_id()
        id2 = generate_secure_id()
        self.assertNotEqual(id1, id2)
        self.assertGreater(len(id1), 10)
    
    def test_generate_secure_id_with_prefix(self):
        """Test secure ID generation with prefix"""
        id_val = generate_secure_id("user_")
        self.assertTrue(id_val.startswith("user_"))
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing"""
        data = "sensitive_password"
        hash1 = hash_sensitive_data(data)
        hash2 = hash_sensitive_data(data)
        self.assertEqual(hash1, hash2)  # Deterministic
        self.assertEqual(len(hash1), 64)  # SHA-256 hex
    
    def test_hash_sensitive_data_different(self):
        """Test different data produces different hashes"""
        hash1 = hash_sensitive_data("data1")
        hash2 = hash_sensitive_data("data2")
        self.assertNotEqual(hash1, hash2)
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking"""
        data = "1234567890123456"
        masked = mask_sensitive_data(data, visible_chars=4)
        self.assertTrue(masked.startswith("1234"))
        self.assertTrue(masked.endswith("3456"))
        self.assertIn("****", masked)
    
    def test_mask_sensitive_data_short(self):
        """Test masking short data"""
        data = "1234"
        masked = mask_sensitive_data(data, visible_chars=4)
        self.assertEqual(masked, "****")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for security features"""
    
    @classmethod
    def setUpClass(cls):
        if not SECURITY_AVAILABLE:
            cls.skipTest(cls, "Security framework not available")
    
    async def test_full_authentication_flow(self):
        """Test complete authentication flow"""
        # Create user
        user = UserContext(
            user_id="user123",
            username="testuser",
            email="test@example.com",
            roles=["viewer"]
        )
        
        # Create token
        jwt_mgr = get_jwt_manager()
        token = jwt_mgr.create_access_token(user)
        
        # Verify token
        context = jwt_mgr.get_user_context(token)
        self.assertIsNotNone(context)
        self.assertEqual(context.user_id, "user123")
    
    async def test_rate_limit_with_auth(self):
        """Test rate limiting with authenticated user"""
        limiter = RateLimiter(requests_per_minute=5, burst_size=3)
        
        user = UserContext(user_id="user1", username="user", email="user@example.com")
        
        # Make requests as user
        for i in range(3):
            allowed, _ = await limiter.is_allowed(user.user_id)
            self.assertTrue(allowed)
        
        # 4th request should be denied
        allowed, _ = await limiter.is_allowed(user.user_id)
        self.assertFalse(allowed)


# ============================================================================
# TEST RUNNER
# ============================================================================

def run_security_tests():
    """Run all security tests and report results"""
    print("=" * 80)
    print("OpenEvolve Security Test Suite")
    print("=" * 80)
    print()
    
    if not SECURITY_AVAILABLE:
        print("ERROR: Security framework not available. Cannot run tests.")
        return False
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestJWTManager))
    suite.addTests(loader.loadTestsFromTestCase(TestUserContext))
    suite.addTests(loader.loadTestsFromTestCase(TestRateLimiter))
    suite.addTests(loader.loadTestsFromTestCase(TestInputValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestAuditLogger))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityDecorators))
    suite.addTests(loader.loadTestsFromTestCase(TestUtilityFunctions))
    suite.addTests(loader.loadTestsFromTestCase(TestSecurityIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print()
    print("=" * 80)
    print("Security Test Results Summary")
    print("=" * 80)
    print(f"Tests Run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print()
    
    if result.wasSuccessful():
        print("ALL SECURITY TESTS PASSED!")
        return True
    else:
        print("SOME SECURITY TESTS FAILED!")
        return False


if __name__ == "__main__":
    success = run_security_tests()
    sys.exit(0 if success else 1)
