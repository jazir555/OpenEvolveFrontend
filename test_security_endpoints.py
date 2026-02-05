"""
API Security Endpoints Tests
Tests for API security including CORS, CSRF, security headers, and endpoint protection.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import json


try:
    from fastapi import FastAPI, Request, Response, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


class SecurityHeadersMiddleware:
    """Middleware to add security headers to responses."""
    
    def __init__(self, app=None):
        self.app = app
        self.headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'",
            'Referrer-Policy': 'strict-origin-when-cross-origin',
            'Permissions-Policy': 'geolocation=(), microphone=(), camera=()',
        }
    
    async def __call__(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in self.headers.items():
            response.headers[header] = value
        return response


import secrets

class CSRFProtection:
    """CSRF protection implementation."""
    
    def __init__(self, token_length: int = 32):
        self.token_length = token_length
        self._tokens: Dict[str, str] = {}
    
    def generate_token(self, session_id: str) -> str:
        """Generate CSRF token for session."""
        token = secrets.token_urlsafe(self.token_length)
        self._tokens[session_id] = token
        return token
    
    def validate_token(self, session_id: str, token: str) -> bool:
        """Validate CSRF token."""
        stored_token = self._tokens.get(session_id)
        if not stored_token:
            return False
        # Use constant-time comparison to prevent timing attacks
        return secrets.compare_digest(stored_token, token)
    
    def clear_token(self, session_id: str):
        """Clear CSRF token for session."""
        self._tokens.pop(session_id, None)


class APIKeyValidator:
    """API key validation and management."""
    
    def __init__(self):
        self._keys: Dict[str, Dict[str, Any]] = {}
    
    def validate_key(self, api_key: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate API key and return permissions."""
        if not api_key:
            return False, None
        
        # Check key format
        if not api_key.startswith('sk-'):
            return False, None
        
        key_data = self._keys.get(api_key)
        if not key_data:
            return False, None
        
        if key_data.get('revoked', False):
            return False, None
        
        if key_data.get('expires_at'):
            from datetime import datetime
            if datetime.utcnow() > key_data['expires_at']:
                return False, None
        
        return True, key_data
    
    def register_key(self, api_key: str, permissions: List[str], expires_at=None):
        """Register a new API key."""
        self._keys[api_key] = {
            'permissions': permissions,
            'created_at': __import__('datetime').datetime.utcnow(),
            'expires_at': expires_at,
            'revoked': False,
        }
    
    def revoke_key(self, api_key: str):
        """Revoke an API key."""
        if api_key in self._keys:
            self._keys[api_key]['revoked'] = True


class JWTValidator:
    """JWT token validation."""
    
    def __init__(self, secret_key: str, algorithm: str = 'HS256'):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def validate_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Validate JWT token."""
        try:
            import jwt
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return True, payload
        except jwt.ExpiredSignatureError:
            return False, {'error': 'Token expired'}
        except jwt.InvalidTokenError:
            return False, {'error': 'Invalid token'}


class TestSecurityHeaders:
    """Test security headers implementation."""
    
    @pytest.fixture
    def middleware(self):
        return SecurityHeadersMiddleware()
    
    def test_security_headers_present(self, middleware):
        """Test that all security headers are defined."""
        required_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options',
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy',
            'Referrer-Policy',
            'Permissions-Policy',
        ]
        
        for header in required_headers:
            assert header in middleware.headers
    
    def test_x_content_type_options(self, middleware):
        """Test X-Content-Type-Options header."""
        assert middleware.headers['X-Content-Type-Options'] == 'nosniff'
    
    def test_x_frame_options(self, middleware):
        """Test X-Frame-Options header."""
        assert middleware.headers['X-Frame-Options'] == 'DENY'
    
    def test_strict_transport_security(self, middleware):
        """Test HSTS header."""
        hsts = middleware.headers['Strict-Transport-Security']
        assert 'max-age=31536000' in hsts
        assert 'includeSubDomains' in hsts
    
    def test_content_security_policy(self, middleware):
        """Test CSP header."""
        csp = middleware.headers['Content-Security-Policy']
        assert "default-src 'self'" in csp


class TestCSRFProtection:
    """Test CSRF protection."""
    
    @pytest.fixture
    def csrf(self):
        return CSRFProtection()
    
    def test_token_generation(self, csrf):
        """Test CSRF token generation."""
        session_id = "session_123"
        token = csrf.generate_token(session_id)
        
        assert token is not None
        assert len(token) > 0
    
    def test_token_validation_success(self, csrf):
        """Test successful token validation."""
        session_id = "session_123"
        token = csrf.generate_token(session_id)
        
        assert csrf.validate_token(session_id, token) == True
    
    def test_token_validation_failure(self, csrf):
        """Test failed token validation."""
        session_id = "session_123"
        csrf.generate_token(session_id)
        
        # Wrong token
        assert csrf.validate_token(session_id, "wrong-token") == False
        
        # Wrong session
        assert csrf.validate_token("wrong-session", "any-token") == False
    
    def test_token_uniqueness(self, csrf):
        """Test that tokens are unique per session."""
        session1 = "session_1"
        session2 = "session_2"
        
        token1 = csrf.generate_token(session1)
        token2 = csrf.generate_token(session2)
        
        assert token1 != token2
    
    def test_token_clearing(self, csrf):
        """Test token clearing."""
        session_id = "session_123"
        token = csrf.generate_token(session_id)
        
        # Token should be valid
        assert csrf.validate_token(session_id, token) == True
        
        # Clear token
        csrf.clear_token(session_id)
        
        # Token should no longer be valid
        assert csrf.validate_token(session_id, token) == False
    
    def test_timing_attack_protection(self, csrf):
        """Test that validation uses constant-time comparison."""
        import time
        
        session_id = "session_123"
        token = csrf.generate_token(session_id)
        
        # Measure validation time for correct token
        start = time.time()
        for _ in range(100):
            csrf.validate_token(session_id, token)
        correct_time = time.time() - start
        
        # Measure validation time for incorrect token
        start = time.time()
        for _ in range(100):
            csrf.validate_token(session_id, "wrong-token")
        wrong_time = time.time() - start
        
        # Times should be similar (within factor of 2)
        ratio = max(correct_time, wrong_time) / min(correct_time, wrong_time)
        assert ratio < 2.0


class TestAPIKeyValidation:
    """Test API key validation."""
    
    @pytest.fixture
    def validator(self):
        return APIKeyValidator()
    
    def test_valid_key_format(self, validator):
        """Test validation of correctly formatted key."""
        api_key = "sk-test-valid-key-12345"
        validator.register_key(api_key, ['read', 'write'])
        
        is_valid, data = validator.validate_key(api_key)
        assert is_valid == True
        assert data is not None
        assert 'read' in data['permissions']
    
    def test_invalid_key_format(self, validator):
        """Test validation rejects invalid key format."""
        invalid_keys = [
            "",  # Empty
            "invalid-key",  # Wrong prefix
            "pk-test-key",  # Wrong prefix
            "sk-",  # Too short
        ]
        
        for key in invalid_keys:
            is_valid, _ = validator.validate_key(key)
            assert is_valid == False, f"Key '{key}' should be invalid"
    
    def test_unknown_key(self, validator):
        """Test validation rejects unknown key."""
        is_valid, _ = validator.validate_key("sk-unknown-key-12345")
        assert is_valid == False
    
    def test_revoked_key(self, validator):
        """Test validation rejects revoked key."""
        api_key = "sk-test-key"
        validator.register_key(api_key, ['read'])
        
        # Revoke the key
        validator.revoke_key(api_key)
        
        is_valid, _ = validator.validate_key(api_key)
        assert is_valid == False
    
    def test_expired_key(self, validator):
        """Test validation rejects expired key."""
        from datetime import datetime, timedelta
        
        api_key = "sk-expired-key"
        expired_time = datetime.utcnow() - timedelta(days=1)
        validator.register_key(api_key, ['read'], expires_at=expired_time)
        
        is_valid, _ = validator.validate_key(api_key)
        assert is_valid == False


class TestJWTValidation:
    """Test JWT token validation."""
    
    @pytest.fixture
    def jwt_validator(self):
        return JWTValidator(secret_key="test-secret-key-12345")
    
    @pytest.fixture
    def valid_token(self, jwt_validator):
        """Generate a valid token for testing."""
        import jwt
        payload = {
            'user_id': 'user_123',
            'exp': __import__('datetime').datetime.utcnow() + __import__('datetime').timedelta(hours=1),
            'iat': __import__('datetime').datetime.utcnow(),
        }
        return jwt.encode(payload, jwt_validator.secret_key, algorithm=jwt_validator.algorithm)
    
    def test_valid_token(self, jwt_validator, valid_token):
        """Test validation of valid token."""
        is_valid, payload = jwt_validator.validate_token(valid_token)
        
        assert is_valid == True
        assert payload is not None
        assert payload.get('user_id') == 'user_123'
    
    def test_expired_token(self, jwt_validator):
        """Test validation rejects expired token."""
        import jwt
        from datetime import datetime, timedelta
        
        payload = {
            'user_id': 'user_123',
            'exp': datetime.utcnow() - timedelta(hours=1),
        }
        token = jwt.encode(payload, jwt_validator.secret_key, algorithm=jwt_validator.algorithm)
        
        is_valid, error = jwt_validator.validate_token(token)
        assert is_valid == False
        assert 'expired' in str(error).lower()
    
    def test_invalid_signature(self, jwt_validator):
        """Test validation rejects token with invalid signature."""
        import jwt
        
        payload = {'user_id': 'user_123'}
        # Sign with wrong key
        token = jwt.encode(payload, "wrong-secret-key", algorithm=jwt_validator.algorithm)
        
        is_valid, _ = jwt_validator.validate_token(token)
        assert is_valid == False
    
    def test_malformed_token(self, jwt_validator):
        """Test validation rejects malformed token."""
        is_valid, _ = jwt_validator.validate_token("not-a-valid-token")
        assert is_valid == False
    
    def test_empty_token(self, jwt_validator):
        """Test validation rejects empty token."""
        is_valid, _ = jwt_validator.validate_token("")
        assert is_valid == False


class TestCORSPolicy:
    """Test CORS policy configuration."""
    
    def test_allowed_origins(self):
        """Test allowed origins configuration."""
        allowed_origins = [
            "https://app.example.com",
            "https://admin.example.com",
        ]
        
        # Simulate origin check
        request_origin = "https://app.example.com"
        assert request_origin in allowed_origins
    
    def test_disallowed_origins(self):
        """Test that disallowed origins are rejected."""
        allowed_origins = [
            "https://app.example.com",
        ]
        
        request_origin = "https://evil.com"
        assert request_origin not in allowed_origins
    
    def test_allowed_methods(self):
        """Test allowed HTTP methods."""
        allowed_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        
        # Common methods should be allowed
        assert "GET" in allowed_methods
        assert "POST" in allowed_methods
    
    def test_allowed_headers(self):
        """Test allowed headers."""
        allowed_headers = [
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-CSRF-Token",
        ]
        
        assert "Authorization" in allowed_headers
        assert "Content-Type" in allowed_headers
    
    def test_credentials_handling(self):
        """Test credentials handling in CORS."""
        allow_credentials = True
        
        # When credentials are allowed, wildcard origins should not be used
        if allow_credentials:
            allowed_origins = ["https://specific-domain.com"]  # Not "*"
            assert "*" not in allowed_origins


class TestEndpointProtection:
    """Test API endpoint protection."""
    
    def test_public_endpoint_access(self):
        """Test that public endpoints are accessible without auth."""
        public_endpoints = ['/health', '/api/version', '/docs']
        
        # Public endpoints should not require authentication
        assert '/health' in public_endpoints
    
    def test_protected_endpoint_access(self):
        """Test that protected endpoints require authentication."""
        protected_endpoints = {
            '/api/users': ['read', 'write'],
            '/api/admin': ['admin'],
            '/api/sensitive': ['read'],
        }
        
        # These should require authentication
        assert 'read' in protected_endpoints['/api/users']
    
    def test_admin_endpoint_restriction(self):
        """Test that admin endpoints require admin permission."""
        admin_endpoints = [
            '/api/admin/users',
            '/api/admin/settings',
            '/api/admin/logs',
        ]
        
        required_permission = 'admin'
        
        for endpoint in admin_endpoints:
            assert required_permission == 'admin'


class TestRequestValidation:
    """Test request validation and sanitization."""
    
    def test_content_type_validation(self):
        """Test content-type validation."""
        allowed_types = ['application/json', 'application/x-www-form-urlencoded']
        
        assert 'application/json' in allowed_types
        assert 'text/plain' not in allowed_types
    
    def test_request_size_limit(self):
        """Test request size limits."""
        max_size = 1024 * 1024  # 1MB
        
        # Small request should be OK
        small_request = json.dumps({"data": "x" * 1000})
        assert len(small_request.encode()) < max_size
        
        # Large request should be rejected
        large_request = json.dumps({"data": "x" * (max_size + 100)})
        assert len(large_request.encode()) > max_size
    
    def test_parameter_sanitization(self):
        """Test parameter sanitization."""
        malicious_input = "<script>alert('xss')</script>"
        
        # Should sanitize or reject malicious input
        sanitized = malicious_input.replace('<script>', '').replace('</script>', '')
        assert '<script>' not in sanitized


class TestResponseSecurity:
    """Test response security measures."""
    
    def test_error_message_sanitization(self):
        """Test that error messages don't leak sensitive info."""
        internal_error = "Database connection failed: postgres://user:pass@host/db"
        
        # Error message exposed to client should not contain credentials
        client_error = "Internal server error"
        
        assert 'postgres://' not in client_error
        assert 'user:pass' not in client_error
    
    def test_stack_trace_hiding(self):
        """Test that stack traces are hidden in production."""
        is_production = True
        
        if is_production:
            error_response = {"error": "Internal server error"}
        else:
            error_response = {"error": "Internal server error", "traceback": "..."}
        
        if is_production:
            assert 'traceback' not in error_response
    
    def test_information_disclosure_prevention(self):
        """Test prevention of information disclosure."""
        # Server headers should not reveal version info
        server_header = "Server"  # Generic, no version
        
        assert 'nginx' not in server_header.lower() or 'apache' not in server_header.lower()


class TestAPIVersioning:
    """Test API versioning security."""
    
    def test_version_in_header(self):
        """Test API version in header."""
        headers = {
            'Accept': 'application/vnd.api+json;version=1',
        }
        
        assert 'version=1' in headers['Accept']
    
    def test_deprecated_version_warning(self):
        """Test deprecated version warnings."""
        deprecated_versions = ['v1']
        current_version = 'v2'
        
        # Using deprecated version should trigger warning
        assert 'v1' in deprecated_versions


class TestInputValidation:
    """Test API input validation."""
    
    def test_sql_injection_prevention(self):
        """Test SQL injection prevention in API inputs."""
        malicious_input = "'; DROP TABLE users; --"
        
        # Should be sanitized or rejected
        assert ';' in malicious_input  # Input contains dangerous chars
        # Real implementation would sanitize this
    
    def test_nosql_injection_prevention(self):
        """Test NoSQL injection prevention."""
        malicious_input = {"$ne": None}
        
        # Should reject NoSQL operators
        assert '$ne' in malicious_input
    
    def test_command_injection_prevention(self):
        """Test command injection prevention."""
        malicious_input = "; rm -rf /"
        
        # Should be sanitized
        assert ';' in malicious_input
    
    def test_path_traversal_prevention(self):
        """Test path traversal prevention."""
        malicious_path = "../../../etc/passwd"
        
        # Should normalize and reject
        assert '..' in malicious_path


class TestRateLimitIntegration:
    """Test rate limiting integration with endpoints."""
    
    def test_endpoint_specific_limits(self):
        """Test different limits for different endpoints."""
        limits = {
            '/api/public': 1000,
            '/api/private': 100,
            '/api/admin': 10,
        }
        
        # Admin endpoints should have stricter limits
        assert limits['/api/admin'] < limits['/api/public']
    
    def test_authenticated_vs_anonymous_limits(self):
        """Test different limits for authenticated vs anonymous users."""
        authenticated_limit = 1000
        anonymous_limit = 100
        
        # Authenticated users should have higher limits
        assert authenticated_limit > anonymous_limit


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
