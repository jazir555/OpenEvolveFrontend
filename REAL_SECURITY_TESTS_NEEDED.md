# Real Security Tests Needed - Implementation Guide

This document provides examples of REAL security tests that should replace the current mock-based tests.

---

## 1. REAL SQL Injection Tests (Replace test_input_validation.py)

### Current (Fake) Test:
```python
def test_sql_injection_in_text_validation(self, validator, payload):
    result = validator._remove_script_tags(payload)  # Just removes HTML!
    assert isinstance(result, str)  # Always passes
```

### Real Test Needed:
```python
import sqlite3
import pytest
from contextlib import contextmanager

class TestRealSQLInjectionPrevention:
    """REAL SQL injection tests with actual database."""
    
    @pytest.fixture
    def real_database(self):
        """Create real SQLite database for testing."""
        conn = sqlite3.connect(':memory:')
        conn.execute('''
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                username TEXT,
                password TEXT,
                email TEXT
            )
        ''')
        conn.execute('''
            INSERT INTO users VALUES 
            (1, 'admin', 'secret123', 'admin@example.com'),
            (2, 'user', 'pass456', 'user@example.com')
        ''')
        conn.commit()
        yield conn
        conn.close()
    
    SQL_INJECTION_PAYLOADS = [
        ("' OR '1'='1", "Boolean-based blind"),
        ("' UNION SELECT * FROM users--", "Union-based"),
        ("'; DROP TABLE users; --", "Stacked queries"),
        ("' AND (SELECT * FROM (SELECT(SLEEP(5)))a)", "Time-based blind"),
        ("1' AND 1=1--", "Tautology"),
    ]
    
    @pytest.mark.parametrize("payload,technique", SQL_INJECTION_PAYLOADS)
    def test_api_uses_parameterized_queries(self, real_database, payload, technique):
        """Verify API uses parameterized queries, not string formatting."""
        from api_server import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        # Attempt injection through actual API endpoint
        response = client.post("/api/users/search", json={
            "query": payload
        })
        
        # Should either:
        # 1. Return normal results (if parameterized properly)
        # 2. Return validation error (if input validated)
        # Should NOT:
        # 1. Return all users (injection success)
        # 2. Return SQL error (reveals structure)
        # 3. Drop table (successful attack)
        
        if response.status_code == 200:
            data = response.json()
            # If parameterized, should return empty list or specific user
            # NOT all users (which would indicate successful injection)
            assert len(data) <= 1, f"Possible SQL injection! Got {len(data)} users"
    
    @pytest.mark.parametrize("payload,technique", SQL_INJECTION_PAYLOADS)
    def test_direct_sql_injection_blocked(self, real_database, payload, technique):
        """Test that direct SQL queries with injection fail."""
        
        # This tests the ACTUAL database behavior
        cursor = real_database.cursor()
        
        # Try unsafe string formatting (what NOT to do)
        try:
            unsafe_query = f"SELECT * FROM users WHERE username = '{payload}'"
            cursor.execute(unsafe_query)
            unsafe_results = cursor.fetchall()
            
            # If injection worked, we'd get more results than expected
            if len(unsafe_results) > 1:
                pytest.fail(f"SQL Injection possible with {technique}: {payload}")
        except sqlite3.Error as e:
            # Error is actually good - means injection didn't work as expected
            pass
        
        # Now test SAFE parameterized query
        safe_query = "SELECT * FROM users WHERE username = ?"
        cursor.execute(safe_query, (payload,))
        safe_results = cursor.fetchall()
        
        # Should return empty (no user with that username)
        assert len(safe_results) == 0, "Parameterized query should prevent injection"
    
    def test_sqlmap_style_union_injection(self, real_database):
        """Test against UNION-based injection attacks."""
        cursor = real_database.cursor()
        
        # Simulate sqlmap attack
        payload = "' UNION SELECT sql, null, null, null FROM sqlite_master--"
        
        cursor.execute("SELECT * FROM users WHERE username = ?", (payload,))
        results = cursor.fetchall()
        
        # Should not return schema information
        assert len(results) == 0
    
    def test_no_error_messages_leak_structure(self, real_database):
        """Verify SQL errors don't reveal database structure."""
        from api_server import app
        from fastapi.testclient import TestClient
        
        client = TestClient(app)
        
        response = client.get("/api/users?id='invalid")
        
        # Response should NOT contain SQL keywords
        response_text = response.text.lower()
        forbidden_keywords = ['sqlite', 'mysql', 'postgresql', 'table', 'column']
        
        for keyword in forbidden_keywords:
            assert keyword not in response_text, \
                f"Error message leaks database info: {keyword}"
```

---

## 2. REAL Security Headers Tests (Replace test_security_endpoints.py)

### Current (Fake) Test:
```python
class SecurityHeadersMiddleware:  # DEFINED IN TEST FILE!
    def test_security_headers_present(self, middleware):
        assert 'X-Content-Type-Options' in middleware.headers  # Just checks dict!
```

### Real Test Needed:
```python
import pytest
from fastapi.testclient import TestClient
from api_server import app  # Import ACTUAL app

class TestRealSecurityHeaders:
    """REAL security headers tests against actual server."""
    
    @pytest.fixture
    def client(self):
        """Create test client with actual app."""
        return TestClient(app)
    
    REQUIRED_SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': ['DENY', 'SAMEORIGIN'],
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=',
        'Content-Security-Policy': None,  # Just needs to exist
        'Referrer-Policy': None,
        'Permissions-Policy': None,
    }
    
    def test_all_required_headers_present(self, client):
        """Verify all security headers in actual HTTP response."""
        response = client.get("/api/health")
        
        for header in self.REQUIRED_SECURITY_HEADERS.keys():
            assert header in response.headers, \
                f"Missing security header: {header}"
    
    def test_x_content_type_options_value(self, client):
        """Verify X-Content-Type-Options header value."""
        response = client.get("/api/health")
        
        assert response.headers.get('X-Content-Type-Options') == 'nosniff'
    
    def test_x_frame_options_prevents_clickjacking(self, client):
        """Verify X-Frame-Options prevents clickjacking."""
        response = client.get("/api/health")
        
        x_frame = response.headers.get('X-Frame-Options', '').upper()
        assert x_frame in ['DENY', 'SAMEORIGIN'], \
            f"X-Frame-Options not secure: {x_frame}"
    
    def test_hsts_header_for_https(self, client):
        """Verify HSTS header for HTTPS enforcement."""
        response = client.get("/api/health")
        
        hsts = response.headers.get('Strict-Transport-Security', '')
        assert 'max-age=' in hsts
        
        # Extract max-age value
        max_age = int(hsts.split('max-age=')[1].split(';')[0])
        assert max_age >= 31536000, "HSTS max-age should be at least 1 year"
    
    def test_csp_prevents_xss(self, client):
        """Verify CSP header exists and has reasonable policy."""
        response = client.get("/api/health")
        
        csp = response.headers.get('Content-Security-Policy', '')
        assert csp, "CSP header missing"
        
        # Should have some restrictions
        assert "default-src" in csp or "script-src" in csp, \
            "CSP should restrict script sources"
    
    def test_no_server_version_header(self, client):
        """Verify Server header doesn't reveal version."""
        response = client.get("/api/health")
        
        server = response.headers.get('Server', '')
        # Should not contain version numbers
        assert not any(c.isdigit() for c in server), \
            f"Server header reveals version: {server}"
    
    def test_cors_headers_not_wildcard_with_credentials(self, client):
        """Verify CORS doesn't allow wildcard with credentials."""
        response = client.get("/api/health", headers={
            'Origin': 'https://evil.com'
        })
        
        allow_origin = response.headers.get('Access-Control-Allow-Origin', '')
        allow_creds = response.headers.get('Access-Control-Allow-Credentials', '')
        
        if allow_creds.lower() == 'true':
            assert allow_origin != '*', \
                "Cannot use wildcard origin with credentials"
    
    def test_headers_on_error_responses(self, client):
        """Verify security headers present even on error pages."""
        response = client.get("/api/nonexistent-endpoint")
        
        # Should still have security headers on 404
        assert 'X-Content-Type-Options' in response.headers
        assert 'X-Frame-Options' in response.headers
```

---

## 3. REAL Rate Limiting Tests (Replace test_rate_limiting.py)

### Current (Fake) Test:
```python
class RateLimiter:  # DEFINED IN TEST FILE!
    """Test implements its own rate limiter"""
```

### Real Test Needed:
```python
import pytest
import time
import redis
from fastapi.testclient import TestClient
from api_server import app

class TestRealRateLimiting:
    """REAL rate limiting tests against production implementation."""
    
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    @pytest.fixture
    def redis_client(self):
        """Real Redis connection for rate limit state."""
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            r.ping()
            yield r
            r.flushdb()  # Clean up after tests
        except redis.ConnectionError:
            pytest.skip("Redis not available")
    
    def test_rate_limit_blocks_excessive_requests(self, client, redis_client):
        """Verify rate limit actually blocks after limit exceeded."""
        endpoint = "/api/public/test"
        
        # Make requests up to limit (e.g., 10 requests/minute)
        responses = []
        for i in range(12):
            response = client.get(endpoint)
            responses.append(response.status_code)
        
        # First 10 should succeed
        assert all(code == 200 for code in responses[:10]), \
            "Requests within limit should succeed"
        
        # 11th+ should be rate limited
        assert responses[10] == 429, \
            "Request exceeding limit should return 429"
        assert responses[11] == 429, \
            "Request exceeding limit should return 429"
    
    def test_rate_limit_headers_present(self, client):
        """Verify rate limit headers in actual HTTP responses."""
        response = client.get("/api/public/test")
        
        assert 'X-RateLimit-Limit' in response.headers
        assert 'X-RateLimit-Remaining' in response.headers
        assert 'X-RateLimit-Reset' in response.headers
    
    def test_rate_limit_resets_after_window(self, client, redis_client):
        """Verify rate limit resets after time window."""
        endpoint = "/api/public/test"
        
        # Exhaust limit
        for _ in range(10):
            client.get(endpoint)
        
        # Should be blocked
        response = client.get(endpoint)
        assert response.status_code == 429
        
        # Wait for window to reset (e.g., 60 seconds)
        time.sleep(60)
        
        # Should work again
        response = client.get(endpoint)
        assert response.status_code == 200, \
            "Rate limit should reset after window"
    
    def test_different_endpoints_different_limits(self, client):
        """Verify different endpoints have different rate limits."""
        # Public endpoint: 100/min
        public_responses = [client.get("/api/public/test").status_code 
                          for _ in range(105)]
        
        # Admin endpoint: 10/min  
        admin_responses = [client.get("/api/admin/test").status_code 
                         for _ in range(15)]
        
        # Public should allow more
        public_blocked = sum(1 for r in public_responses if r == 429)
        admin_blocked = sum(1 for r in admin_responses if r == 429)
        
        assert public_blocked < admin_blocked, \
            "Admin endpoint should have stricter rate limit"
    
    def test_rate_limit_by_api_key(self, client, redis_client):
        """Verify rate limiting per API key."""
        api_key_1 = "sk-test-key-1"
        api_key_2 = "sk-test-key-2"
        
        # Exhaust limit for key 1
        for _ in range(100):
            client.get("/api/protected", headers={
                "X-API-Key": api_key_1
            })
        
        # Key 1 should be blocked
        response = client.get("/api/protected", headers={
            "X-API-Key": api_key_1
        })
        assert response.status_code == 429
        
        # Key 2 should still work
        response = client.get("/api/protected", headers={
            "X-API-Key": api_key_2
        })
        assert response.status_code == 200
    
    def test_rate_limit_bypass_attempts_blocked(self, client):
        """Test common rate limit bypass techniques."""
        endpoint = "/api/public/test"
        
        bypass_techniques = [
            # IP spoofing attempts
            ({"X-Forwarded-For": "1.2.3.4"}, "X-Forwarded-For spoofing"),
            ({"X-Real-IP": "1.2.3.4"}, "X-Real-IP spoofing"),
            ({"CF-Connecting-IP": "1.2.3.4"}, "CloudFlare IP spoofing"),
            # Case variation
            ({"x-api-key": "test"}, "Lowercase headers"),
        ]
        
        # Exhaust limit with real IP
        for _ in range(10):
            client.get(endpoint)
        
        # Try bypass techniques - all should fail
        for headers, technique in bypass_techniques:
            response = client.get(endpoint, headers=headers)
            assert response.status_code == 429, \
                f"Rate limit bypass worked: {technique}"
    
    def test_retry_after_header_on_429(self, client):
        """Verify Retry-After header on rate limited response."""
        endpoint = "/api/public/test"
        
        # Exhaust limit
        for _ in range(10):
            client.get(endpoint)
        
        # Get rate limited response
        response = client.get(endpoint)
        
        assert response.status_code == 429
        assert 'Retry-After' in response.headers
        
        retry_after = int(response.headers['Retry-After'])
        assert retry_after > 0, "Retry-After should indicate wait time"
```

---

## 4. REAL Audit Logging Tests (Replace test_audit_logging.py)

### Current (Fake) Test:
```python
class MockAuditLog:  # DEFINED IN TEST FILE!
    """Mock audit log for testing"""
```

### Real Test Needed:
```python
import pytest
import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

class TestRealAuditLogging:
    """REAL audit logging tests with actual file/database writes."""
    
    @pytest.fixture
    def audit_db_path(self, tmp_path):
        """Create temporary audit database."""
        return tmp_path / "audit.db"
    
    @pytest.fixture
    def audit_log_dir(self, tmp_path):
        """Create temporary log directory."""
        log_dir = tmp_path / "audit_logs"
        log_dir.mkdir()
        return log_dir
    
    def test_audit_log_written_to_database(self, audit_db_path):
        """Verify audit logs actually written to database."""
        from auth_system import AuthenticationSystem
        
        auth = AuthenticationSystem(db_path=str(audit_db_path))
        
        # Perform action that should be logged
        user = auth.create_user(
            username="testuser",
            email="test@example.com",
            password="password123"
        )
        
        # Directly query database to verify log entry
        conn = sqlite3.connect(str(audit_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT * FROM audit_logs 
            WHERE operation = 'CREATE_USER' 
            AND user_id = ?
        """, (user.id,))
        
        logs = cursor.fetchall()
        assert len(logs) >= 1, "Audit log not written to database"
        
        conn.close()
    
    def test_audit_log_written_to_file(self, audit_log_dir):
        """Verify audit logs written to actual log files."""
        from auth_system import FileAuditLogger
        
        logger = FileAuditLogger(log_dir=str(audit_log_dir))
        
        logger.log(
            action="LOGIN",
            user_id="user_123",
            success=True,
            details={"ip": "192.168.1.1"}
        )
        
        # Flush and close
        logger.close()
        
        # Find log file
        log_files = list(audit_log_dir.glob("audit_*.log"))
        assert len(log_files) >= 1, "No audit log file created"
        
        # Read and verify content
        with open(log_files[0], 'r') as f:
            content = f.read()
            assert "LOGIN" in content
            assert "user_123" in content
            assert "192.168.1.1" in content
    
    def test_audit_log_immutable_after_write(self, audit_db_path):
        """Verify audit logs cannot be modified after writing."""
        from auth_system import AuthenticationSystem
        
        auth = AuthenticationSystem(db_path=str(audit_db_path))
        
        # Create log entry
        auth.log_audit(
            user_id="user_123",
            operation="SENSITIVE_ACTION",
            success=True
        )
        
        # Try to modify log entry (simulating tampering attempt)
        conn = sqlite3.connect(str(audit_db_path))
        cursor = conn.cursor()
        
        # This should fail or be prevented
        try:
            cursor.execute("""
                UPDATE audit_logs 
                SET operation = 'MODIFIED_ACTION'
                WHERE operation = 'SENSITIVE_ACTION'
            """)
            conn.commit()
            
            # If update succeeded, verify tamper detection catches it
            cursor.execute("SELECT integrity_hash FROM audit_logs")
            rows = cursor.fetchall()
            
            for row in rows:
                integrity_hash = row[0]
                # Verify hash would fail for modified row
                assert integrity_hash is not None, \
                    "No integrity hash for tamper detection"
                    
        except sqlite3.Error as e:
            # Database-level prevention is also acceptable
            pass
        finally:
            conn.close()
    
    def test_audit_log_chain_integrity(self, audit_db_path):
        """Verify blockchain-style integrity chain in logs."""
        from auth_system import AuthenticationSystem
        
        auth = AuthenticationSystem(db_path=str(audit_db_path))
        
        # Create multiple log entries
        for i in range(5):
            auth.log_audit(
                user_id=f"user_{i}",
                operation=f"ACTION_{i}",
                success=True
            )
        
        # Verify chain integrity
        conn = sqlite3.connect(str(audit_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT log_id, previous_hash, integrity_hash 
            FROM audit_logs 
            ORDER BY timestamp
        """)
        
        logs = cursor.fetchall()
        conn.close()
        
        # Each log should reference previous
        for i in range(1, len(logs)):
            current = logs[i]
            previous = logs[i-1]
            
            assert current[1] == previous[2], \
                f"Chain broken at index {i}: previous_hash doesn't match"
    
    def test_sensitive_data_redacted_in_logs(self, audit_log_dir):
        """Verify sensitive data is redacted in audit logs."""
        from auth_system import FileAuditLogger
        
        logger = FileAuditLogger(log_dir=str(audit_log_dir))
        
        # Log action with sensitive data
        logger.log(
            action="LOGIN",
            user_id="user_123",
            details={
                "password": "secret123",  # Should be redacted
                "api_key": "sk-test-123",  # Should be redacted
                "username": "testuser"  # OK to log
            }
        )
        
        logger.close()
        
        # Read log file
        log_files = list(audit_log_dir.glob("audit_*.log"))
        with open(log_files[0], 'r') as f:
            content = f.read()
        
        # Verify sensitive data not present
        assert "secret123" not in content, "Password logged in plain text!"
        assert "sk-test-123" not in content, "API key logged in plain text!"
        assert "[REDACTED]" in content or "***" in content, \
            "Sensitive data should be replaced with redaction marker"
        
        # Non-sensitive data should be present
        assert "testuser" in content
    
    def test_failed_login_attempts_logged(self, audit_db_path):
        """Verify failed authentication attempts are logged."""
        from auth_system import AuthenticationSystem
        
        auth = AuthenticationSystem(db_path=str(audit_db_path))
        
        # Create user
        auth.create_user(
            username="testuser",
            email="test@example.com",
            password="correct_password"
        )
        
        # Attempt failed logins
        for _ in range(3):
            auth.authenticate("testuser", "wrong_password")
        
        # Query for failed attempts
        conn = sqlite3.connect(str(audit_db_path))
        cursor = conn.cursor()
        cursor.execute("""
            SELECT COUNT(*) FROM audit_logs 
            WHERE operation = 'AUTHENTICATE' AND success = 0
        """)
        
        failed_count = cursor.fetchone()[0]
        conn.close()
        
        assert failed_count >= 3, "Failed login attempts not logged"
```

---

## Summary of Required Changes

| Component | Current Approach | Required Approach |
|-----------|------------------|-------------------|
| SQL Injection | String sanitization tests | Real database with parameterized query tests |
| XSS | Regex pattern matching | Real browser rendering or bleach verification |
| Security Headers | Dict key checking | Actual HTTP response header verification |
| Rate Limiting | Test-implements-own-limiter | Production rate limiter with real requests |
| Audit Logging | Mock log class | Real file/database writes and verification |
| Encryption | Mostly real (good!) | Add key rotation and side-channel tests |

---

## Implementation Priority

### Week 1: Fix Critical Gaps
1. Add real SQL injection tests with SQLite
2. Replace security header dict tests with HTTP tests
3. Fix failing XSS tests or document limitations

### Week 2: Add Integration Tests
4. Add real rate limiting tests with FastAPI TestClient
5. Add real audit logging tests with temp files

### Week 3: Enhance Coverage
6. Add penetration test scenarios
7. Add security regression tests
8. Document all security test limitations

---

**Document Version:** 1.0  
**Created:** February 4, 2026  
**Author:** Independent Security Analysis
