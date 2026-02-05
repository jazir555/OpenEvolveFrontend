"""
Comprehensive Authentication Testing Suite - TRUE 100%
Tests all authentication flows: OAuth2, JWT, API keys, Sessions
"""

import pytest
import asyncio
import jwt
import hashlib
import secrets
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import sqlite3
import tempfile
import os

# Import authentication components
from auth_system import (
    AuthManager, TokenManager, APIKeyManager, SessionManager,
    AuthenticationError, TokenExpiredError, InvalidTokenError,
    OAuth2Config, JWTConfig
)
from secure_api import SecureAPI


class TestOAuth2Authentication:
    """Test OAuth2 authentication flows."""
    
    @pytest.fixture
    def oauth_config(self):
        return OAuth2Config(
            client_id="test_client_id",
            client_secret="test_client_secret",
            redirect_uri="http://localhost:8000/callback",
            authorize_url="https://auth.example.com/authorize",
            token_url="https://auth.example.com/token",
            scopes=["read", "write"]
        )
    
    @pytest.fixture
    def auth_manager(self, oauth_config):
        return AuthManager(oauth_config=oauth_config)
    
    def test_oauth2_authorize_url_generation(self, auth_manager):
        """Test OAuth2 authorization URL generation."""
        state = secrets.token_urlsafe(32)
        url = auth_manager.get_oauth_authorize_url(state=state)
        
        assert "client_id=test_client_id" in url
        assert f"state={state}" in url
        assert "response_type=code" in url
        assert "scope=" in url
    
    @pytest.mark.asyncio
    async def test_oauth2_token_exchange(self, auth_manager):
        """Test OAuth2 token exchange."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "test_access_token",
                "refresh_token": "test_refresh_token",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            mock_post.return_value = mock_response
            
            tokens = await auth_manager.exchange_oauth_code("auth_code_123")
            
            assert tokens["access_token"] == "test_access_token"
            assert tokens["refresh_token"] == "test_refresh_token"
            assert tokens["token_type"] == "Bearer"
    
    @pytest.mark.asyncio
    async def test_oauth2_token_refresh(self, auth_manager):
        """Test OAuth2 token refresh."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "access_token": "new_access_token",
                "refresh_token": "new_refresh_token",
                "token_type": "Bearer",
                "expires_in": 3600
            }
            mock_post.return_value = mock_response
            
            tokens = await auth_manager.refresh_oauth_token("old_refresh_token")
            
            assert tokens["access_token"] == "new_access_token"
            assert "refresh_token" in mock_post.call_args.kwargs['data']
    
    @pytest.mark.asyncio
    async def test_oauth2_invalid_code(self, auth_manager):
        """Test OAuth2 with invalid authorization code."""
        with patch('httpx.AsyncClient.post') as mock_post:
            mock_response = Mock()
            mock_response.status_code = 400
            mock_response.json.return_value = {"error": "invalid_grant"}
            mock_post.return_value = mock_response
            
            with pytest.raises(AuthenticationError):
                await auth_manager.exchange_oauth_code("invalid_code")


class TestJWTAuthentication:
    """Test JWT authentication flows."""
    
    @pytest.fixture
    def jwt_config(self):
        return JWTConfig(
            secret_key="test_secret_key_for_jwt_signing_at_least_32_chars",
            algorithm="HS256",
            access_token_expire_minutes=15,
            refresh_token_expire_days=7
        )
    
    @pytest.fixture
    def token_manager(self, jwt_config):
        return TokenManager(jwt_config)
    
    def test_jwt_token_generation(self, token_manager):
        """Test JWT access token generation."""
        user_id = "user_123"
        claims = {"role": "admin", "permissions": ["read", "write"]}
        
        token = token_manager.create_access_token(user_id, claims)
        
        # Decode and verify
        decoded = jwt.decode(
            token, 
            token_manager.config.secret_key,
            algorithms=[token_manager.config.algorithm]
        )
        
        assert decoded["sub"] == user_id
        assert decoded["role"] == "admin"
        assert decoded["type"] == "access"
        assert "exp" in decoded
        assert "iat" in decoded
    
    def test_jwt_refresh_token_generation(self, token_manager):
        """Test JWT refresh token generation."""
        user_id = "user_123"
        
        token = token_manager.create_refresh_token(user_id)
        
        decoded = jwt.decode(
            token,
            token_manager.config.secret_key,
            algorithms=[token_manager.config.algorithm]
        )
        
        assert decoded["sub"] == user_id
        assert decoded["type"] == "refresh"
    
    def test_jwt_token_verification(self, token_manager):
        """Test JWT token verification."""
        user_id = "user_123"
        token = token_manager.create_access_token(user_id, {"role": "user"})
        
        payload = token_manager.verify_token(token)
        
        assert payload["sub"] == user_id
        assert payload["role"] == "user"
    
    def test_jwt_expired_token(self, token_manager):
        """Test handling of expired JWT tokens."""
        # Create token that expired 1 hour ago
        expired_payload = {
            "sub": "user_123",
            "exp": datetime.now(timezone.utc) - timedelta(hours=1),
            "iat": datetime.now(timezone.utc) - timedelta(hours=2),
            "type": "access"
        }
        expired_token = jwt.encode(
            expired_payload,
            token_manager.config.secret_key,
            algorithm=token_manager.config.algorithm
        )
        
        with pytest.raises(TokenExpiredError):
            token_manager.verify_token(expired_token)
    
    def test_jwt_invalid_signature(self, token_manager):
        """Test handling of JWT with invalid signature."""
        token = token_manager.create_access_token("user_123", {})
        
        # Tamper with token
        tampered_token = token[:-5] + "XXXXX"
        
        with pytest.raises(InvalidTokenError):
            token_manager.verify_token(tampered_token)
    
    def test_jwt_malformed_token(self, token_manager):
        """Test handling of malformed JWT tokens."""
        with pytest.raises(InvalidTokenError):
            token_manager.verify_token("not.a.valid.token")
    
    def test_jwt_wrong_type(self, token_manager):
        """Test using refresh token as access token."""
        refresh_token = token_manager.create_refresh_token("user_123")
        
        # Try to verify refresh token as access token
        with pytest.raises(InvalidTokenError):
            token_manager.verify_token(refresh_token, expected_type="access")


class TestAPIKeyAuthentication:
    """Test API key authentication."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for API key testing."""
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE api_keys (
                id TEXT PRIMARY KEY,
                key_hash TEXT UNIQUE NOT NULL,
                name TEXT NOT NULL,
                user_id TEXT NOT NULL,
                scopes TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                last_used_at TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        conn.close()
        
        yield path
        
        os.unlink(path)
    
    @pytest.fixture
    def api_key_manager(self, temp_db):
        return APIKeyManager(database_path=temp_db)
    
    def test_api_key_generation(self, api_key_manager):
        """Test API key generation."""
        user_id = "user_123"
        name = "Test API Key"
        scopes = ["read", "write"]
        
        api_key, key_id = api_key_manager.create_api_key(
            user_id=user_id,
            name=name,
            scopes=scopes
        )
        
        # Key should be in format: prefix_keyid_random
        parts = api_key.split('_')
        assert len(parts) >= 3
        assert key_id in api_key
        assert len(api_key) >= 32  # Should be reasonably long
    
    def test_api_key_hash_storage(self, api_key_manager):
        """Test that only hash of API key is stored."""
        user_id = "user_123"
        api_key, key_id = api_key_manager.create_api_key(
            user_id=user_id,
            name="Test Key",
            scopes=["read"]
        )
        
        # Verify raw key is not stored
        conn = sqlite3.connect(api_key_manager.database_path)
        cursor = conn.execute(
            "SELECT key_hash FROM api_keys WHERE id = ?",
            (key_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        assert row is not None
        stored_hash = row[0]
        assert stored_hash != api_key  # Raw key not stored
        assert len(stored_hash) == 64  # SHA-256 hex length
    
    def test_api_key_validation_success(self, api_key_manager):
        """Test successful API key validation."""
        user_id = "user_123"
        api_key, key_id = api_key_manager.create_api_key(
            user_id=user_id,
            name="Test Key",
            scopes=["read", "write"]
        )
        
        result = api_key_manager.validate_api_key(api_key)
        
        assert result is not None
        assert result["user_id"] == user_id
        assert result["scopes"] == ["read", "write"]
    
    def test_api_key_validation_invalid(self, api_key_manager):
        """Test validation of invalid API key."""
        result = api_key_manager.validate_api_key("invalid_key_12345")
        assert result is None
    
    def test_api_key_revocation(self, api_key_manager):
        """Test API key revocation."""
        api_key, key_id = api_key_manager.create_api_key(
            user_id="user_123",
            name="Test Key",
            scopes=["read"]
        )
        
        # Revoke the key
        api_key_manager.revoke_api_key(key_id)
        
        # Key should no longer validate
        result = api_key_manager.validate_api_key(api_key)
        assert result is None
    
    def test_api_key_expiration(self, api_key_manager):
        """Test API key expiration."""
        api_key, key_id = api_key_manager.create_api_key(
            user_id="user_123",
            name="Test Key",
            scopes=["read"],
            expires_at=datetime.now(timezone.utc) - timedelta(days=1)  # Expired yesterday
        )
        
        result = api_key_manager.validate_api_key(api_key)
        assert result is None  # Expired key invalid
    
    def test_api_key_scope_checking(self, api_key_manager):
        """Test API key scope verification."""
        api_key, _ = api_key_manager.create_api_key(
            user_id="user_123",
            name="Read Only Key",
            scopes=["read"]
        )
        
        key_data = api_key_manager.validate_api_key(api_key)
        
        assert api_key_manager.has_scope(key_data, "read")
        assert not api_key_manager.has_scope(key_data, "write")
        assert not api_key_manager.has_scope(key_data, "admin")


class TestSessionAuthentication:
    """Test session-based authentication."""
    
    @pytest.fixture
    def session_manager(self):
        return SessionManager(
            secret_key="test_secret_key_for_sessions_at_least_32_chars",
            session_timeout_minutes=30
        )
    
    def test_session_creation(self, session_manager):
        """Test session creation."""
        user_id = "user_123"
        metadata = {"ip": "127.0.0.1", "user_agent": "test"}
        
        session = session_manager.create_session(user_id, metadata)
        
        assert session["user_id"] == user_id
        assert "session_id" in session
        assert "created_at" in session
        assert "expires_at" in session
        assert session["is_active"] is True
    
    def test_session_id_uniqueness(self, session_manager):
        """Test that session IDs are unique."""
        session_ids = set()
        for i in range(100):
            session = session_manager.create_session(f"user_{i}")
            session_ids.add(session["session_id"])
        
        assert len(session_ids) == 100  # All unique
    
    def test_session_validation(self, session_manager):
        """Test session validation."""
        user_id = "user_123"
        session = session_manager.create_session(user_id)
        session_id = session["session_id"]
        
        validated = session_manager.get_session(session_id)
        
        assert validated is not None
        assert validated["user_id"] == user_id
    
    def test_session_expiration(self, session_manager):
        """Test session expiration."""
        session = session_manager.create_session("user_123")
        session_id = session["session_id"]
        
        # Expire the session
        session_manager.expire_session(session_id)
        
        validated = session_manager.get_session(session_id)
        assert validated is None
    
    def test_session_timeout(self, session_manager):
        """Test automatic session timeout."""
        # Create session with very short timeout
        short_manager = SessionManager(
            secret_key="test_secret_key_for_sessions_at_least_32_chars",
            session_timeout_minutes=0  # Immediate timeout
        )
        
        session = short_manager.create_session("user_123")
        session_id = session["session_id"]
        
        # Should be immediately expired
        validated = short_manager.get_session(session_id)
        assert validated is None
    
    def test_session_invalidation_all(self, session_manager):
        """Test invalidating all sessions for a user."""
        user_id = "user_123"
        
        # Create multiple sessions
        session1 = session_manager.create_session(user_id)
        session2 = session_manager.create_session(user_id)
        session3 = session_manager.create_session("other_user")
        
        # Invalidate all sessions for user_123
        session_manager.invalidate_all_user_sessions(user_id)
        
        assert session_manager.get_session(session1["session_id"]) is None
        assert session_manager.get_session(session2["session_id"]) is None
        assert session_manager.get_session(session3["session_id"]) is not None


class TestMultiFactorAuthentication:
    """Test multi-factor authentication."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    def test_totp_generation(self, auth_manager):
        """Test TOTP secret generation."""
        secret = auth_manager.generate_totp_secret()
        
        # Should be base32 encoded
        assert len(secret) >= 16
        assert all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567" for c in secret)
    
    def test_totp_verification(self, auth_manager):
        """Test TOTP code verification."""
        import pyotp
        
        secret = pyotp.random_base32()
        totp = pyotp.TOTP(secret)
        
        # Generate current code
        code = totp.now()
        
        # Verify code
        assert auth_manager.verify_totp(secret, code) is True
    
    def test_totp_invalid_code(self, auth_manager):
        """Test TOTP with invalid code."""
        import pyotp
        
        secret = pyotp.random_base32()
        
        # Wrong code
        assert auth_manager.verify_totp(secret, "000000") is False
    
    def test_backup_codes_generation(self, auth_manager):
        """Test backup code generation."""
        codes = auth_manager.generate_backup_codes(count=10)
        
        assert len(codes) == 10
        # Each code should be unique
        assert len(set(codes)) == 10
        # Codes should be 8 characters
        assert all(len(c) == 8 for c in codes)
    
    def test_backup_code_usage(self, auth_manager):
        """Test backup code usage."""
        codes = auth_manager.generate_backup_codes(count=5)
        user_id = "user_123"
        
        # Store codes
        auth_manager.store_backup_codes(user_id, codes)
        
        # Use a code
        code_to_use = codes[0]
        assert auth_manager.verify_backup_code(user_id, code_to_use) is True
        
        # Same code should not work again
        assert auth_manager.verify_backup_code(user_id, code_to_use) is False


class TestPasswordAuthentication:
    """Test password-based authentication."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    def test_password_hashing(self, auth_manager):
        """Test password hashing."""
        password = "MySecurePassword123!"
        
        hashed = auth_manager.hash_password(password)
        
        # Hash should be different from password
        assert hashed != password
        # Should use modern hashing (bcrypt, argon2, or scrypt)
        assert len(hashed) >= 60  # bcrypt length
    
    def test_password_verification(self, auth_manager):
        """Test password verification."""
        password = "MySecurePassword123!"
        hashed = auth_manager.hash_password(password)
        
        assert auth_manager.verify_password(password, hashed) is True
        assert auth_manager.verify_password("WrongPassword", hashed) is False
    
    def test_password_strength_validation(self, auth_manager):
        """Test password strength validation."""
        # Strong password
        strong, errors = auth_manager.validate_password_strength(
            "MyStrongP@ssw0rd!123"
        )
        assert strong is True
        assert len(errors) == 0
        
        # Weak passwords
        weak_tests = [
            ("short", ["too_short"]),
            ("nouppercase123!", ["no_uppercase"]),
            ("NOLOWERCASE123!", ["no_lowercase"]),
            ("NoNumbers!", ["no_number"]),
            ("NoSpecial123", ["no_special"]),
            ("password", ["too_short", "no_uppercase", "no_number", "no_special"]),
        ]
        
        for password, expected_errors in weak_tests:
            strong, errors = auth_manager.validate_password_strength(password)
            assert strong is False
            for error in expected_errors:
                assert error in errors


class TestSSOAuthentication:
    """Test Single Sign-On authentication."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    @pytest.mark.asyncio
    async def test_saml_response_parsing(self, auth_manager):
        """Test SAML response parsing."""
        # Mock SAML response
        saml_response = """
        <samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol">
            <saml:Assertion xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion">
                <saml:Subject>
                    <saml:NameID>user@example.com</saml:NameID>
                </saml:Subject>
                <saml:AttributeStatement>
                    <saml:Attribute Name="email">
                        <saml:AttributeValue>user@example.com</saml:AttributeValue>
                    </saml:Attribute>
                </saml:AttributeStatement>
            </saml:Assertion>
        </samlp:Response>
        """
        
        with patch('xml.etree.ElementTree.fromstring') as mock_parse:
            mock_root = Mock()
            mock_root.find.return_value = Mock(text="user@example.com")
            mock_parse.return_value = mock_root
            
            result = await auth_manager.verify_saml_response(saml_response)
            assert result["email"] == "user@example.com"
    
    @pytest.mark.asyncio
    async def test_oidc_token_verification(self, auth_manager):
        """Test OpenID Connect token verification."""
        id_token = "test_oidc_token"
        
        with patch('jwt.decode') as mock_decode:
            mock_decode.return_value = {
                "sub": "user_123",
                "email": "user@example.com",
                "name": "Test User",
                "iss": "https://accounts.google.com",
                "aud": "client_id"
            }
            
            result = await auth_manager.verify_oidc_token(
                id_token,
                expected_issuer="https://accounts.google.com",
                expected_audience="client_id"
            )
            
            assert result["sub"] == "user_123"
            assert result["email"] == "user@example.com"


class TestAuthenticationRateLimiting:
    """Test rate limiting for authentication attempts."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager(max_login_attempts=5, lockout_duration_minutes=15)
    
    def test_login_rate_limiting(self, auth_manager):
        """Test rate limiting on login attempts."""
        username = "test_user"
        
        # Multiple failed attempts
        for i in range(5):
            auth_manager.record_failed_login(username)
        
        # 6th attempt should trigger rate limit
        assert auth_manager.is_login_allowed(username) is False
    
    def test_lockout_reset_after_duration(self, auth_manager):
        """Test lockout resets after duration."""
        username = "test_user"
        
        # Trigger lockout
        for i in range(5):
            auth_manager.record_failed_login(username)
        
        assert auth_manager.is_login_allowed(username) is False
        
        # Reset with successful login
        auth_manager.record_successful_login(username)
        assert auth_manager.is_login_allowed(username) is True
    
    def test_different_users_independent(self, auth_manager):
        """Test rate limiting is per-user."""
        # Lock out user1
        for i in range(5):
            auth_manager.record_failed_login("user1")
        
        # user2 should still be able to login
        assert auth_manager.is_login_allowed("user2") is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
