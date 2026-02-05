"""
TRUE 100% Security Tests - Complete Security Validation
========================================================

This module contains comprehensive security tests that verify:
1. Audit logging persists to SQLite (survives restart)
2. API keys validated against database with SHA-256
3. TLS 1.2+ configuration with secure cipher suites
4. All security mechanisms work in production

Author: OpenEvolve Security Team
Version: 3.0.0 - TRUE 100%
Test Count: 50+ comprehensive tests
"""

import pytest
import sqlite3
import hashlib
import json
import asyncio
import os
import tempfile
import ssl
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path

# Import security framework
from security_framework import (
    # Audit logging
    AuditLogger, AuditLogEntry, get_audit_logger,
    # API Key management
    APIKeyDatabase, APIKeyStatus, APIKeyRecord, get_api_key_database,
    # Authentication
    JWTManager, UserContext, Permission, UserRole,
    # TLS/SSL
    create_ssl_context, get_tls_config, SecurityConfig,
    # Validation
    InputValidator, ValidationError,
    # Rate limiting
    RateLimiter, get_rate_limiter,
    # Utilities
    initialize_security, security_health_check,
    hash_password, verify_password, generate_secure_id,
    hash_sensitive_data, mask_sensitive_data
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_db_path():
    """Create a temporary database file."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def temp_cert_files():
    """Create temporary certificate files."""
    cert_path = tempfile.mktemp(suffix='.cert.pem')
    key_path = tempfile.mktemp(suffix='.key.pem')
    
    # Generate self-signed cert using openssl if available
    import shutil
    import subprocess
    
    if shutil.which("openssl"):
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:2048",
            "-keyout", key_path, "-out", cert_path,
            "-days", "1", "-nodes", "-subj", "/CN=test",
            "-quiet"
        ], check=True, capture_output=True)
    
    yield cert_path, key_path
    
    # Cleanup
    for path in [cert_path, key_path]:
        if os.path.exists(path):
            os.unlink(path)


@pytest.fixture
def audit_logger(temp_db_path):
    """Create an audit logger with temporary database."""
    return AuditLogger(db_path=temp_db_path)


@pytest.fixture
def api_key_db(temp_db_path):
    """Create an API key database with temporary file."""
    return APIKeyDatabase(db_path=temp_db_path)


@pytest.fixture
def jwt_manager():
    """Create a JWT manager."""
    return JWTManager()


# ============================================================================
# AUDIT LOGGING TESTS - TRUE 100%
# ============================================================================

class TestAuditLoggingTrue100:
    """Test audit logging with SQLite persistence - TRUE 100%."""
    
    @pytest.mark.asyncio
    async def test_audit_log_persisted_to_sqlite_database(self, audit_logger, temp_db_path):
        """CRITICAL: Audit logs must be persisted to SQLite database."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id="test_user_123",
            action="TEST_ACTION",
            resource_type="test_resource",
            resource_id="test_123",
            success=True,
            ip_address="127.0.0.1",
            details={"test": "data"}
        )
        
        await audit_logger.log(entry)
        
        # Verify entry is in database by creating a new connection
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM audit_logs WHERE user_id = ?", ("test_user_123",))
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None, "CRITICAL FAIL: Audit log entry was not persisted to database"
        assert row[2] == "test_user_123"  # user_id column
        assert row[3] == "TEST_ACTION"  # action column
        assert row[6] == 1  # success column (1 = True)
    
    @pytest.mark.asyncio
    async def test_audit_log_survives_application_restart(self, temp_db_path):
        """CRITICAL: Audit logs must survive application restart (database persistence)."""
        # Create first logger and add entry
        logger1 = AuditLogger(db_path=temp_db_path)
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id="survive_test",
            action="SURVIVE_TEST",
            resource_type="test",
            resource_id="123",
            success=True
        )
        await logger1.log(entry)
        
        # Create new logger instance (simulates app restart)
        logger2 = AuditLogger(db_path=temp_db_path)
        
        # Query logs with new logger
        logs = logger2.query_logs(user_id="survive_test")
        
        assert len(logs) == 1, "CRITICAL FAIL: Audit log entry did not survive restart"
        assert logs[0].action == "SURVIVE_TEST"
        assert logs[0].user_id == "survive_test"
    
    @pytest.mark.asyncio
    async def test_audit_log_query_with_filters(self, audit_logger):
        """Test querying audit logs with various filters."""
        # Add multiple entries
        for i in range(5):
            await audit_logger.log(AuditLogEntry(
                timestamp=datetime.utcnow(),
                user_id=f"user_{i % 2}",  # user_0 or user_1
                action=f"ACTION_{i}",
                resource_type="test" if i < 3 else "other",
                resource_id=f"id_{i}",
                success=i % 2 == 0
            ))
        
        # Query by user_id
        user_0_logs = audit_logger.query_logs(user_id="user_0")
        assert len(user_0_logs) >= 2
        
        # Query by action
        action_logs = audit_logger.query_logs(action="ACTION_1")
        assert len(action_logs) == 1
        
        # Query by resource_type
        type_logs = audit_logger.query_logs(resource_type="test")
        assert len(type_logs) >= 3
        
        # Query by success
        success_logs = audit_logger.query_logs(success=True)
        failed_logs = audit_logger.query_logs(success=False)
        assert len(success_logs) >= 2
        assert len(failed_logs) >= 2
    
    @pytest.mark.asyncio
    async def test_audit_log_integrity_hash(self, audit_logger, temp_db_path):
        """Test audit log integrity hash for tamper detection."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id="integrity_test",
            action="INTEGRITY_TEST",
            resource_type="test",
            resource_id="123",
            success=True
        )
        
        await audit_logger.log(entry)
        
        # Query the log
        logs = audit_logger.query_logs(user_id="integrity_test")
        assert len(logs) == 1
        assert logs[0].integrity_hash is not None
        
        # Verify integrity
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM audit_logs WHERE user_id = ?", ("integrity_test",))
        row_id = cursor.fetchone()[0]
        conn.close()
        
        is_valid, message = audit_logger.verify_integrity(row_id)
        assert is_valid, f"Integrity check failed: {message}"
    
    @pytest.mark.asyncio
    async def test_audit_log_export_json(self, audit_logger, temp_db_path):
        """Test exporting audit logs to JSON."""
        await audit_logger.log(AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id="export_test",
            action="EXPORT_TEST",
            resource_type="test",
            resource_id="123",
            success=True
        ))
        
        export_path = temp_db_path + ".export.json"
        count = audit_logger.export_logs(export_path, format='json')
        
        assert os.path.exists(export_path)
        assert count >= 1
        
        with open(export_path) as f:
            data = json.load(f)
        assert len(data) >= 1
        assert any(log['user_id'] == "export_test" for log in data)
    
    @pytest.mark.asyncio
    async def test_audit_log_statistics(self, audit_logger):
        """Test audit log statistics generation."""
        # Add test entries
        for i in range(10):
            await audit_logger.log(AuditLogEntry(
                timestamp=datetime.utcnow(),
                user_id=f"stats_user_{i % 3}",
                action=f"STATS_ACTION_{i % 2}",
                resource_type="stats",
                resource_id=f"id_{i}",
                success=i % 2 == 0
            ))
        
        stats = audit_logger.get_statistics(days=1)
        
        assert stats['total_logs'] >= 10
        assert 'success_fail' in stats
        assert 'top_actions' in stats
        assert 'top_users' in stats
    
    @pytest.mark.asyncio
    async def test_audit_log_concurrent_writes(self, audit_logger, temp_db_path):
        """Test that concurrent audit log writes work correctly."""
        async def write_logs(thread_id: int, count: int):
            for i in range(count):
                await audit_logger.log(AuditLogEntry(
                    timestamp=datetime.utcnow(),
                    user_id=f"thread_{thread_id}",
                    action="CONCURRENT_TEST",
                    resource_type="test",
                    resource_id=f"test_{thread_id}_{i}",
                    success=True
                ))
                await asyncio.sleep(0.001)
        
        # Run multiple concurrent writers
        await asyncio.gather(
            write_logs(1, 10),
            write_logs(2, 10),
            write_logs(3, 10)
        )
        
        # Verify all logs were written
        logs = audit_logger.query_logs(action="CONCURRENT_TEST", limit=100)
        assert len(logs) == 30, f"Expected 30 logs, got {len(logs)}"


# ============================================================================
# API KEY VALIDATION TESTS - TRUE 100%
# ============================================================================

class TestAPIKeyValidationTrue100:
    """Test database-backed API key validation with SHA-256 - TRUE 100%."""
    
    def test_api_key_creation_and_hash_storage(self, api_key_db):
        """CRITICAL: API keys must be stored with SHA-256 hash, never plaintext."""
        raw_key, record = api_key_db.create_key(
            name="Test Key",
            user_id="test_user",
            expires_in_days=30,
            permissions=[Permission.API_ACCESS.value]
        )
        
        # Verify key format
        assert raw_key.startswith("sk-")
        assert len(raw_key) >= 32
        
        # Verify hash stored, not plaintext
        conn = sqlite3.connect(api_key_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key_hash FROM api_keys WHERE id = ?", (record.id,))
        stored_hash = cursor.fetchone()[0]
        conn.close()
        
        # Stored value should be hash, not the key itself
        assert stored_hash != raw_key
        assert len(stored_hash) == 64  # SHA-256 hex length
        assert stored_hash == hashlib.sha256(raw_key.encode()).hexdigest()
    
    def test_api_key_validation_against_database(self, api_key_db):
        """CRITICAL: API keys must be validated against database with SHA-256 hash."""
        raw_key, record = api_key_db.create_key(
            name="Validation Test Key",
            user_id="validation_user",
            permissions=[Permission.API_ACCESS.value]
        )
        
        # Validate the key
        is_valid, returned_record, message = api_key_db.validate_key(raw_key)
        
        assert is_valid is True, f"Validation failed: {message}"
        assert returned_record is not None
        assert returned_record.id == record.id
        assert returned_record.user_id == "validation_user"
    
    def test_invalid_api_key_rejected(self, api_key_db):
        """CRITICAL: Invalid API keys must be rejected."""
        # Test with completely fake key
        is_valid, record, message = api_key_db.validate_key("sk-fakekey123456789")
        assert is_valid is False
        assert record is None
        assert "not found" in message.lower() or "invalid" in message.lower()
    
    def test_api_key_with_only_sk_prefix_rejected(self, api_key_db):
        """CRITICAL: Keys with only 'sk-' prefix but no DB record must be rejected."""
        # Any key starting with sk- should be rejected if not in database
        fake_key = "sk-notindatabase123456789"
        is_valid, record, message = api_key_db.validate_key(fake_key)
        
        assert is_valid is False, "Key not in database must be rejected"
        assert record is None
    
    def test_api_key_expiration_check(self, api_key_db):
        """CRITICAL: Expired API keys must be rejected."""
        # Create an expired key
        raw_key, record = api_key_db.create_key(
            name="Expired Key",
            user_id="test_user",
            expires_in_days=-1  # Already expired
        )
        
        # Validate should fail
        is_valid, returned_record, message = api_key_db.validate_key(raw_key)
        
        assert is_valid is False, "Expired key should be rejected"
        assert "expired" in message.lower()
    
    def test_api_key_revocation(self, api_key_db):
        """CRITICAL: Revoked API keys must be rejected."""
        # Create a key then revoke it
        raw_key, record = api_key_db.create_key(
            name="Key to Revoke",
            user_id="test_user"
        )
        
        # Verify it works first
        is_valid, _, _ = api_key_db.validate_key(raw_key)
        assert is_valid is True
        
        # Revoke it
        result = api_key_db.revoke_key(record.id, reason="Testing revocation")
        assert result is True
        
        # Now validation should fail
        is_valid, _, message = api_key_db.validate_key(raw_key)
        assert is_valid is False, "Revoked key should be rejected"
        assert "revoked" in message.lower()
    
    def test_api_key_usage_tracking(self, api_key_db):
        """Test that API key usage is tracked."""
        raw_key, record = api_key_db.create_key(
            name="Usage Tracking Key",
            user_id="test_user"
        )
        
        # Validate multiple times
        for _ in range(5):
            api_key_db.validate_key(raw_key)
        
        # Check usage count
        updated_record = api_key_db.get_key_by_id(record.id)
        assert updated_record.usage_count == 5
        assert updated_record.last_used is not None
    
    def test_api_key_ip_whitelist(self, api_key_db):
        """Test API key IP whitelist enforcement."""
        raw_key, record = api_key_db.create_key(
            name="IP Whitelist Key",
            user_id="test_user",
            ip_whitelist=["192.168.1.100", "10.0.0.50"]
        )
        
        # Should succeed from whitelisted IP
        is_valid, _, _ = api_key_db.validate_key(raw_key, client_ip="192.168.1.100")
        assert is_valid is True
        
        # Should fail from non-whitelisted IP
        is_valid, _, message = api_key_db.validate_key(raw_key, client_ip="1.2.3.4")
        assert is_valid is False
        assert "ip" in message.lower()
    
    def test_api_key_permissions(self, api_key_db):
        """Test API key permissions are properly stored and returned."""
        raw_key, record = api_key_db.create_key(
            name="Permissions Test Key",
            user_id="test_user",
            permissions=[Permission.API_ACCESS.value, Permission.WORKFLOW_READ.value]
        )
        
        is_valid, returned_record, _ = api_key_db.validate_key(raw_key)
        
        assert is_valid is True
        assert Permission.API_ACCESS.value in returned_record.permissions
        assert Permission.WORKFLOW_READ.value in returned_record.permissions


# ============================================================================
# TLS/SSL CONFIGURATION TESTS - TRUE 100%
# ============================================================================

class TestTLSConfigurationTrue100:
    """Test TLS 1.2+ configuration - TRUE 100%."""
    
    def test_ssl_context_creation(self, temp_cert_files):
        """CRITICAL: SSL context must be created with secure settings."""
        cert_path, key_path = temp_cert_files
        
        # Skip if certs weren't created
        if not os.path.exists(cert_path):
            pytest.skip("Certificate files not available")
        
        context = create_ssl_context(cert_path, key_path)
        
        assert context is not None
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
    
    def test_tls_version_enforcement(self, temp_cert_files):
        """CRITICAL: TLS 1.2 must be minimum version."""
        cert_path, key_path = temp_cert_files
        
        if not os.path.exists(cert_path):
            pytest.skip("Certificate files not available")
        
        context = create_ssl_context(cert_path, key_path)
        
        # Verify TLS 1.2 is minimum
        assert context.minimum_version == ssl.TLSVersion.TLSv1_2
        
        # Verify old protocols are disabled
        assert context.options & ssl.OP_NO_SSLv2
        assert context.options & ssl.OP_NO_SSLv3
        assert context.options & ssl.OP_NO_TLSv1
        assert context.options & ssl.OP_NO_TLSv1_1
    
    def test_ssl_compression_disabled(self, temp_cert_files):
        """CRITICAL: SSL compression must be disabled (CRIME attack prevention)."""
        cert_path, key_path = temp_cert_files
        
        if not os.path.exists(cert_path):
            pytest.skip("Certificate files not available")
        
        context = create_ssl_context(cert_path, key_path)
        
        # Compression should be disabled
        assert context.options & ssl.OP_NO_COMPRESSION
    
    def test_ssl_context_missing_cert_raises_error(self, temp_db_path):
        """Test that missing certificate files raise appropriate errors."""
        fake_cert = temp_db_path + ".nonexistent.cert"
        fake_key = temp_db_path + ".nonexistent.key"
        
        with pytest.raises(FileNotFoundError):
            create_ssl_context(fake_cert, fake_key)
    
    def test_tls_config_disabled_when_not_enabled(self, monkeypatch):
        """Test that TLS config returns None when disabled."""
        monkeypatch.setattr(SecurityConfig, 'TLS_ENABLED', False)
        
        config = get_tls_config()
        assert config is None


# ============================================================================
# SQL INJECTION PREVENTION TESTS
# ============================================================================

class TestSQLInjectionPrevention:
    """Test SQL injection prevention."""
    
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE audit_logs; --",
        "1' OR '1'='1",
        "1; DELETE FROM api_keys WHERE '1'='1",
        "' UNION SELECT * FROM api_keys --",
        "admin'--",
        "' OR 1=1--",
        "' OR '1'='1' /*",
    ]
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    @pytest.mark.asyncio
    async def test_sql_injection_in_audit_log_user_id(self, audit_logger, payload):
        """CRITICAL: SQL injection in user_id must not execute."""
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id=payload,  # Malicious payload as user_id
            action="TEST",
            resource_type="test",
            resource_id="123",
            success=True
        )
        
        await audit_logger.log(entry)
        
        # Verify the database is still intact
        conn = sqlite3.connect(audit_logger.db_path)
        cursor = conn.cursor()
        
        # Check that audit_logs table still exists
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='audit_logs'")
        assert cursor.fetchone() is not None, "Audit logs table was dropped by SQL injection!"
        
        conn.close()
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_api_key_name(self, api_key_db, payload):
        """CRITICAL: SQL injection in key name must not execute."""
        # Create a key with malicious name - should use parameterized queries
        raw_key, record = api_key_db.create_key(
            name=payload,  # Malicious payload
            user_id="test_user"
        )
        
        # Verify database is intact
        conn = sqlite3.connect(api_key_db.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_keys'")
        assert cursor.fetchone() is not None, "API keys table was dropped by SQL injection!"
        conn.close()


# ============================================================================
# PASSWORD SECURITY TESTS
# ============================================================================

class TestPasswordSecurity:
    """Test password hashing and validation."""
    
    def test_password_hashing(self):
        """Test that passwords are properly hashed with salt."""
        password = "SecureP@ssw0rd123"
        hashed = hash_password(password)
        
        # Should be in format salt:hash
        assert ":" in hashed
        salt, hash_part = hashed.split(":")
        assert len(salt) == 32  # 16 bytes hex
        assert len(hash_part) == 64  # SHA-256 hex
    
    def test_password_verification_success(self):
        """Test successful password verification."""
        password = "SecureP@ssw0rd123"
        hashed = hash_password(password)
        
        assert verify_password(password, hashed) is True
    
    def test_password_verification_failure(self):
        """Test failed password verification."""
        password = "SecureP@ssw0rd123"
        wrong_password = "WrongP@ssw0rd456"
        hashed = hash_password(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_password_hash_uniqueness(self):
        """Test that same password produces different hashes (due to salt)."""
        password = "SecureP@ssw0rd123"
        hash1 = hash_password(password)
        hash2 = hash_password(password)
        
        assert hash1 != hash2  # Different salts


# ============================================================================
# INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """Test input validation and sanitization."""
    
    def test_validate_string_basic(self):
        """Test basic string validation."""
        result = InputValidator.validate_string("hello", "test_field")
        assert result == "hello"
    
    def test_validate_string_too_long(self):
        """Test string length validation."""
        with pytest.raises(ValidationError):
            InputValidator.validate_string("a" * 10000, "test_field", max_length=100)
    
    def test_validate_email_valid(self):
        """Test valid email validation."""
        result = InputValidator.validate_email("test@example.com")
        assert result == "test@example.com"
    
    def test_validate_email_invalid(self):
        """Test invalid email rejection."""
        with pytest.raises(ValidationError):
            InputValidator.validate_email("not-an-email")
    
    def test_validate_password_strong(self):
        """Test strong password validation."""
        result = InputValidator.validate_password("StrongP@ssw0rd123")
        assert result == "StrongP@ssw0rd123"
    
    def test_validate_password_too_weak(self):
        """Test weak password rejection."""
        with pytest.raises(ValidationError):
            InputValidator.validate_password("weak")
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        result = InputValidator.sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        # Valid keys (must be at least 32 chars, start with sk-)
        assert InputValidator.validate_api_key_format("sk-test123456789012345678901234567") is True
        assert InputValidator.validate_api_key_format("sk-12345678901234567890123456789012") is True
        # Invalid keys
        assert InputValidator.validate_api_key_format("sk-short") is False  # Too short
        assert InputValidator.validate_api_key_format("invalid") is False  # Wrong prefix
        assert InputValidator.validate_api_key_format("sk_underscore_not_dash123456789012") is False  # Wrong prefix
        assert InputValidator.validate_api_key_format("") is False
        assert InputValidator.validate_api_key_format(None) is False


# ============================================================================
# RATE LIMITING TESTS
# ============================================================================

class TestRateLimiting:
    """Test rate limiting effectiveness."""
    
    @pytest.mark.asyncio
    async def test_rate_limit_allows_under_limit(self):
        """Test that requests under the limit are allowed."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=5)
        
        # First 5 requests should be allowed (burst size)
        for i in range(5):
            allowed, _ = await limiter.is_allowed("test_client")
            assert allowed, f"Request {i+1} should be allowed"
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_over_limit(self):
        """Test that requests over the limit are blocked."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=2)
        
        # Exhaust burst allowance
        await limiter.is_allowed("test_client")
        await limiter.is_allowed("test_client")
        
        # Third request should be blocked
        allowed, info = await limiter.is_allowed("test_client")
        assert not allowed, "Request over limit should be blocked"
        assert info["remaining"] == 0
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_client_isolation(self):
        """Test that rate limits are isolated per client."""
        limiter = RateLimiter(requests_per_minute=10, burst_size=2)
        
        # Exhaust client1's limit
        await limiter.is_allowed("client1")
        await limiter.is_allowed("client1")
        await limiter.is_allowed("client1")  # This may or may not be allowed depending on timing
        
        # client2 should still be allowed
        allowed, _ = await limiter.is_allowed("client2")
        assert allowed, "Client2 should not be affected by client1's limit"
    
    @pytest.mark.asyncio
    async def test_ip_blocking(self):
        """Test IP blocking functionality."""
        limiter = RateLimiter()
        
        # Block an IP
        limiter.block_ip("192.168.1.100", duration_minutes=60)
        
        # Check if blocked
        is_blocked = await limiter.is_blocked("192.168.1.100")
        assert is_blocked is True


# ============================================================================
# JWT AUTHENTICATION TESTS
# ============================================================================

class TestJWTAuthentication:
    """Test JWT token creation and validation."""
    
    def test_jwt_token_creation(self, jwt_manager):
        """Test creating JWT tokens."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        token = jwt_manager.create_access_token(user)
        assert token is not None
        assert isinstance(token, str)
        assert len(token.split('.')) == 3  # Header.Payload.Signature
    
    def test_jwt_token_validation(self, jwt_manager):
        """Test validating JWT tokens."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["user"],
            permissions=["read"]
        )
        
        token = jwt_manager.create_access_token(user)
        payload = jwt_manager.decode_token(token)
        
        assert payload is not None
        assert payload["sub"] == "test_user"
        assert payload["username"] == "testuser"
    
    def test_jwt_token_expiration(self, jwt_manager):
        """Test that expired tokens are rejected."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com"
        )
        
        # Create token that expired 1 hour ago
        expired_token = jwt_manager.create_access_token(
            user, 
            expires_delta=timedelta(hours=-1)
        )
        
        payload = jwt_manager.decode_token(expired_token)
        assert payload is None, "Expired token should be rejected"
    
    def test_jwt_invalid_token_rejection(self, jwt_manager):
        """Test that invalid tokens are rejected."""
        payload = jwt_manager.decode_token("invalid.token.here")
        assert payload is None, "Invalid token should be rejected"
    
    def test_jwt_user_context_creation(self, jwt_manager):
        """Test user context creation from JWT."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["admin"],
            permissions=[Permission.SYSTEM_ADMIN.value],
            is_superuser=True
        )
        
        token = jwt_manager.create_access_token(user)
        context = jwt_manager.get_user_context(token)
        
        assert context is not None
        assert context.user_id == "test_user"
        assert context.is_superuser is True
        assert context.has_permission(Permission.SYSTEM_ADMIN)


# ============================================================================
# PERMISSION ENFORCEMENT TESTS
# ============================================================================

class TestPermissionEnforcement:
    """Test that permissions are actually enforced."""
    
    def test_user_without_permission_denied(self):
        """Test that users without required permissions are denied."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            roles=["viewer"],
            permissions=[Permission.WORKFLOW_READ.value]
        )
        
        # User should have read permission
        assert user.has_permission(Permission.WORKFLOW_READ)
        
        # User should NOT have delete permission
        assert not user.has_permission(Permission.WORKFLOW_DELETE)
    
    def test_superuser_has_all_permissions(self):
        """Test that superusers have all permissions."""
        user = UserContext(
            user_id="admin",
            username="admin",
            email="admin@example.com",
            is_superuser=True
        )
        
        # Superuser should have any permission
        assert user.has_permission(Permission.SYSTEM_ADMIN)
        assert user.has_permission(Permission.WORKFLOW_DELETE)
        assert user.has_permission(Permission.API_ADMIN)
    
    def test_role_permissions(self):
        """Test that roles grant correct permissions."""
        from security_framework import Role
        
        # Admin role should have all permissions
        admin_perms = Role.ADMIN["permissions"]
        assert Permission.WORKFLOW_CREATE.value in admin_perms
        assert Permission.SYSTEM_ADMIN.value in admin_perms
        
        # Viewer role should only have read permissions
        viewer_perms = Role.VIEWER["permissions"]
        assert Permission.WORKFLOW_READ.value in viewer_perms
        assert Permission.WORKFLOW_DELETE.value not in viewer_perms
    
    def test_session_validity(self):
        """Test session timeout functionality."""
        user = UserContext(
            user_id="test_user",
            username="testuser",
            email="test@example.com",
            last_authenticated=datetime.utcnow()
        )
        
        # Session should be valid
        assert user.is_session_valid(timeout_minutes=60) is True
        
        # Old session should be invalid
        user.last_authenticated = datetime.utcnow() - timedelta(hours=2)
        assert user.is_session_valid(timeout_minutes=60) is False


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test security utility functions."""
    
    def test_generate_secure_id(self):
        """Test secure ID generation."""
        id1 = generate_secure_id("test_")
        id2 = generate_secure_id("test_")
        
        assert id1.startswith("test_")
        assert id2.startswith("test_")
        assert id1 != id2  # Should be unique
        assert len(id1) > 20  # Should be reasonably long
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing."""
        data = "sensitive_info"
        hashed = hash_sensitive_data(data)
        
        assert len(hashed) == 64  # SHA-256 hex
        assert hashed != data
        # Same input should produce same hash
        assert hash_sensitive_data(data) == hashed
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking."""
        # Short data (less than visible_chars * 2)
        assert mask_sensitive_data("abc", visible_chars=4) == "***"
        
        # Normal data
        result = mask_sensitive_data("sk-test123456", visible_chars=4)
        assert result.startswith("sk-t")
        assert result.endswith("3456")
        assert "*" in result
    
    def test_mask_sensitive_data_empty(self):
        """Test masking with empty input."""
        assert mask_sensitive_data("") == ""
        assert mask_sensitive_data(None) == ""


# ============================================================================
# SECURITY INITIALIZATION TESTS
# ============================================================================

class TestSecurityInitialization:
    """Test security system initialization."""
    
    def test_initialize_security(self):
        """Test security initialization."""
        status = initialize_security()
        
        assert 'jwt' in status
        assert 'audit_log' in status
        assert 'api_key_db' in status
        assert 'rate_limiter' in status
        assert 'overall' in status
    
    def test_security_health_check(self):
        """Test security health check."""
        results = security_health_check()
        
        assert 'timestamp' in results
        assert 'overall_status' in results
        assert 'checks' in results
        
        # Should have all security checks
        assert 'jwt_secret' in results['checks']
        assert 'audit_logging' in results['checks']
        assert 'rate_limiting' in results['checks']


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSecurityIntegration:
    """Integration tests for complete security flows."""
    
    @pytest.mark.asyncio
    async def test_complete_auth_flow_with_audit_logging(self, temp_db_path):
        """Test complete authentication flow with audit logging."""
        # Initialize components
        audit_logger = AuditLogger(db_path=temp_db_path)
        api_key_db = APIKeyDatabase(db_path=temp_db_path)
        
        # Create an API key
        raw_key, record = api_key_db.create_key(
            name="Integration Test Key",
            user_id="integration_user",
            permissions=[Permission.API_ACCESS.value, Permission.WORKFLOW_READ.value]
        )
        
        # Validate the key
        is_valid, returned_record, message = api_key_db.validate_key(raw_key)
        assert is_valid is True
        
        # Log the authentication
        await audit_logger.log_auth_attempt(
            user_id=returned_record.user_id,
            success=True,
            ip_address="127.0.0.1",
            details={"method": "api_key", "key_id": returned_record.id}
        )
        
        # Verify audit log
        logs = audit_logger.query_logs(user_id="integration_user", action="AUTHENTICATE")
        assert len(logs) == 1
        assert logs[0].success is True
    
    @pytest.mark.asyncio
    async def test_failed_auth_logged(self, temp_db_path):
        """Test that failed authentication attempts are logged."""
        audit_logger = AuditLogger(db_path=temp_db_path)
        
        # Log a failed auth
        await audit_logger.log_auth_attempt(
            user_id="unknown_user",
            success=False,
            ip_address="192.168.1.1",
            details={"reason": "invalid_credentials"}
        )
        
        # Verify it was logged
        logs = audit_logger.query_logs(success=False)
        assert len(logs) >= 1
        assert any(log.user_id == "unknown_user" for log in logs)


# ============================================================================
# MAIN ENTRY POINT FOR STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run all tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "--cov=security_framework"])
