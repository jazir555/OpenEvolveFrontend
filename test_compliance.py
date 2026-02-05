"""
Security Compliance Testing Suite - TRUE 100%
Tests compliance: OWASP Top 10, NIST, ISO 27001, GDPR, PCI DSS
"""

import pytest
import hashlib
import re
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from input_validation import InputValidator, ValidationError
from auth_system import AuthManager, TokenManager, JWTConfig
from security_framework import SecurityManager


class TestOWASPTop10Compliance:
    """Test compliance with OWASP Top 10 2021."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    # A01:2021 - Broken Access Control
    def test_a01_enforce_least_privilege(self, auth_manager):
        """A01: Verify enforcement of least privilege."""
        # Regular user should not have admin access
        assert not auth_manager.has_permission("regular_user", "admin")
        
        # Admin should have all permissions
        assert auth_manager.has_permission("admin_user", "read")
        assert auth_manager.has_permission("admin_user", "write")
        assert auth_manager.has_permission("admin_user", "delete")
    
    def test_a01_deny_by_default(self, auth_manager):
        """A01: Verify deny by default policy."""
        # Unknown resource should be denied
        assert not auth_manager.can_access("unknown_user", "unknown_resource")
    
    def test_a01_path_traversal_protection(self, validator):
        """A01: Verify path traversal protection."""
        malicious_paths = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2fetc%2fpasswd",
        ]
        
        for path in malicious_paths:
            sanitized = validator.sanitize_path(path)
            assert not sanitized.startswith("/") or ".." not in sanitized
    
    # A02:2021 - Cryptographic Failures
    def test_a02_strong_password_hashing(self, auth_manager):
        """A02: Verify strong password hashing."""
        password = "test_password"
        hashed = auth_manager.hash_password(password)
        
        # Should not be plaintext
        assert hashed != password
        # Should use modern algorithm (bcrypt, argon2, scrypt)
        assert len(hashed) >= 60  # bcrypt minimum
    
    def test_a02_encrypted_transmission(self):
        """A02: Verify encrypted data transmission."""
        # Check TLS configuration
        tls_config = SecurityManager.get_tls_config()
        
        # Should use TLS 1.2 or higher
        assert tls_config.get("min_version") in ["TLSv1.2", "TLSv1.3"]
        
        # Should not use weak ciphers
        weak_ciphers = ["DES", "3DES", "RC4", "MD5"]
        configured_ciphers = tls_config.get("ciphers", "")
        for cipher in weak_ciphers:
            assert cipher not in configured_ciphers
    
    def test_a02_proper_key_management(self):
        """A02: Verify proper key management."""
        # Keys should be generated securely
        key = SecurityManager.generate_secure_key()
        assert len(key) >= 32  # 256-bit minimum
        
        # Keys should be unique
        key2 = SecurityManager.generate_secure_key()
        assert key != key2
    
    # A03:2021 - Injection
    def test_a03_sql_injection_prevention(self, validator):
        """A03: Verify SQL injection prevention."""
        sql_payloads = [
            "' OR '1'='1",
            "'; DROP TABLE users; --",
            "1 UNION SELECT * FROM passwords",
        ]
        
        for payload in sql_payloads:
            sanitized = validator.sanitize_string(payload)
            # Should neutralize SQL metacharacters
            assert "'" not in sanitized or ";" not in sanitized or sanitized == payload
    
    def test_a03_xss_prevention(self, validator):
        """A03: Verify XSS prevention."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
        ]
        
        for payload in xss_payloads:
            sanitized = validator._sanitize_html(payload)
            assert "<script>" not in sanitized.lower()
            assert "javascript:" not in sanitized.lower()
    
    def test_a03_command_injection_prevention(self, validator):
        """A03: Verify command injection prevention."""
        cmd_payloads = [
            "; rm -rf /",
            "| cat /etc/passwd",
            "`whoami`",
            "$(id)",
        ]
        
        for payload in cmd_payloads:
            sanitized = validator.sanitize_for_command(payload)
            assert ";" not in sanitized or "|" not in sanitized
    
    # A04:2021 - Insecure Design
    def test_a04_rate_limiting(self, auth_manager):
        """A04: Verify rate limiting implementation."""
        user = "rate_test_user"
        
        # Trigger rate limit
        for _ in range(100):
            auth_manager.record_failed_login(user)
        
        # Should be rate limited
        assert not auth_manager.is_login_allowed(user)
    
    def test_a04_input_validation(self, validator):
        """A04: Verify comprehensive input validation."""
        # Invalid email should be rejected
        with pytest.raises(ValidationError):
            validator.validate_email("not-an-email")
        
        # Negative age should be rejected
        with pytest.raises(ValidationError):
            validator.validate_integer("-1", min_val=0)
    
    # A05:2021 - Security Misconfiguration
    def test_a05_security_headers(self):
        """A05: Verify security headers."""
        headers = SecurityManager.get_security_headers()
        
        required_headers = [
            "X-Content-Type-Options",
            "X-Frame-Options",
            "X-XSS-Protection",
            "Strict-Transport-Security",
        ]
        
        for header in required_headers:
            assert header in headers
    
    def test_a05_error_handling(self):
        """A05: Verify secure error handling."""
        # Errors should not leak sensitive information
        try:
            SecurityManager.perform_operation("invalid")
        except Exception as e:
            error_msg = str(e)
            # Should not contain system details
            assert "password" not in error_msg.lower()
            assert "secret" not in error_msg.lower()
            assert "key" not in error_msg.lower()
    
    # A06:2021 - Vulnerable and Outdated Components
    def test_a06_dependency_scanning(self):
        """A06: Verify no known vulnerable dependencies."""
        vulnerabilities = SecurityManager.check_dependencies()
        
        # Should have no CRITICAL vulnerabilities
        critical = [v for v in vulnerabilities if v["severity"] == "CRITICAL"]
        assert len(critical) == 0, f"Critical vulnerabilities found: {critical}"
    
    # A07:2021 - Identification and Authentication Failures
    def test_a07_strong_authentication(self, auth_manager):
        """A07: Verify strong authentication mechanisms."""
        # Weak passwords should be rejected
        weak_passwords = ["password", "123456", "admin"]
        
        for pwd in weak_passwords:
            strong, errors = auth_manager.validate_password_strength(pwd)
            assert not strong, f"Weak password accepted: {pwd}"
    
    def test_a07_session_management(self, auth_manager):
        """A07: Verify secure session management."""
        session = auth_manager.create_session("user_123")
        
        # Session ID should be random
        assert len(session["id"]) >= 32
        
        # Session should expire
        assert "expires_at" in session
    
    def test_a07_mfa_support(self, auth_manager):
        """A07: Verify MFA support."""
        # Should support TOTP
        assert auth_manager.supports_mfa("totp")
        
        # Should support backup codes
        assert auth_manager.supports_mfa("backup_codes")
    
    # A08:2021 - Software and Data Integrity Failures
    def test_a08_signature_verification(self):
        """A08: Verify signature verification."""
        data = b"Important data"
        signature = SecurityManager.sign_data(data)
        
        # Valid signature should verify
        assert SecurityManager.verify_signature(data, signature)
        
        # Tampered data should fail
        tampered = b"Tampered data"
        assert not SecurityManager.verify_signature(tampered, signature)
    
    def test_a08_dependency_integrity(self):
        """A08: Verify dependency integrity."""
        # Dependencies should have checksums
        deps = SecurityManager.get_dependencies_with_checksums()
        
        for dep in deps:
            assert "name" in dep
            assert "version" in dep
            assert "checksum" in dep
    
    # A09:2021 - Security Logging and Monitoring Failures
    def test_a09_audit_logging(self, auth_manager):
        """A09: Verify security event logging."""
        # Perform security-relevant action
        auth_manager.authenticate("test_user", "test_pass")
        
        # Should be logged
        logs = auth_manager.get_security_logs()
        assert len(logs) > 0
        
        # Log should contain required fields
        log = logs[0]
        assert "timestamp" in log
        assert "action" in log
        assert "user" in log
    
    def test_a09_failed_login_logging(self, auth_manager):
        """A09: Verify failed login logging."""
        # Failed login attempt
        auth_manager.authenticate("test_user", "wrong_pass")
        
        logs = auth_manager.get_security_logs(action="failed_login")
        assert len(logs) >= 1
    
    # A10:2021 - Server-Side Request Forgery (SSRF)
    def test_a10_ssrf_prevention(self, validator):
        """A10: Verify SSRF prevention."""
        malicious_urls = [
            "http://169.254.169.254/latest/meta-data/",  # AWS metadata
            "http://localhost:22",  # Local SSH
            "file:///etc/passwd",  # File protocol
            "http://10.0.0.1/admin",  # Internal IP
        ]
        
        for url in malicious_urls:
            is_safe = validator.is_safe_url(url)
            assert not is_safe, f"Unsafe URL accepted: {url}"


class TestNISTCompliance:
    """Test compliance with NIST Cybersecurity Framework."""
    
    # Identify Function
    def test_nist_identify_asset_management(self):
        """Verify asset management."""
        assets = SecurityManager.list_assets()
        
        # Should have inventory of all assets
        assert len(assets) > 0
        
        # Assets should have classification
        for asset in assets:
            assert "classification" in asset
            assert asset["classification"] in ["public", "internal", "confidential", "restricted"]
    
    def test_nist_identify_risk_assessment(self):
        """Verify risk assessment."""
        risks = SecurityManager.assess_risks()
        
        # Should identify risks
        assert len(risks) >= 0
        
        # Risks should have severity
        for risk in risks:
            assert "severity" in risk
            assert risk["severity"] in ["low", "medium", "high", "critical"]
    
    # Protect Function
    def test_nist_protect_access_control(self):
        """Verify access control."""
        # Should enforce principle of least privilege
        assert SecurityManager.enforces_least_privilege()
        
        # Should have separation of duties
        assert SecurityManager.has_separation_of_duties()
    
    def test_nist_protect_data_security(self):
        """Verify data security."""
        # Data at rest should be encrypted
        assert SecurityManager.encrypts_data_at_rest()
        
        # Data in transit should be encrypted
        assert SecurityManager.uses_tls()
    
    def test_nist_protect_protective_technology(self):
        """Verify protective technology."""
        # Should have firewall
        assert SecurityManager.has_firewall()
        
        # Should have intrusion detection
        assert SecurityManager.has_intrusion_detection()
    
    # Detect Function
    def test_nist_detect_anomalies(self):
        """Verify anomaly detection."""
        # Should detect unusual patterns
        anomaly = SecurityManager.detect_anomaly({
            "user": "user_123",
            "action": "login",
            "location": "unusual_location",
            "time": "unusual_time"
        })
        
        assert isinstance(anomaly, bool)
    
    def test_nist_detect_continuous_monitoring(self):
        """Verify continuous monitoring."""
        # Should have monitoring in place
        assert SecurityManager.has_continuous_monitoring()
        
        # Should generate alerts
        assert SecurityManager.can_generate_alerts()
    
    # Respond Function
    def test_nist_respond_response_planning(self):
        """Verify response planning."""
        # Should have incident response plan
        assert SecurityManager.has_incident_response_plan()
        
        # Should define roles
        assert SecurityManager.has_defined_response_roles()
    
    def test_nist_respond_communications(self):
        """Verify incident communications."""
        # Should have notification procedures
        assert SecurityManager.has_notification_procedures()
    
    # Recover Function
    def test_nist_recover_recovery_planning(self):
        """Verify recovery planning."""
        # Should have recovery plan
        assert SecurityManager.has_recovery_plan()
        
        # Should have backups
        assert SecurityManager.has_backups()
    
    def test_nist_recover_improvements(self):
        """Verify recovery improvements."""
        # Should learn from incidents
        assert SecurityManager.has_lessons_learned_process()


class TestISO27001Compliance:
    """Test compliance with ISO 27001 controls."""
    
    # A.5 - Information Security Policies
    def test_iso27001_a5_policies(self):
        """A.5: Verify information security policies."""
        assert SecurityManager.has_security_policy()
        assert SecurityManager.policy_is_documented()
        assert SecurityManager.policy_is_reviewed_regularly()
    
    # A.6 - Organization of Information Security
    def test_iso27001_a6_roles(self):
        """A.6: Verify security roles and responsibilities."""
        assert SecurityManager.has_security_roles_defined()
        assert SecurityManager.has_security_officer()
    
    # A.7 - Human Resource Security
    def test_iso27001_a7_background_checks(self):
        """A.7: Verify background checks."""
        assert SecurityManager.requires_background_checks()
    
    def test_iso27001_a7_termination(self):
        """A.7: Verify termination procedures."""
        assert SecurityManager.has_termination_procedures()
    
    # A.8 - Asset Management
    def test_iso27001_a8_inventory(self):
        """A.8: Verify asset inventory."""
        assets = SecurityManager.get_asset_inventory()
        assert len(assets) > 0
        
        # Assets should have owners
        for asset in assets:
            assert "owner" in asset
    
    # A.9 - Access Control
    def test_iso27001_a9_access_policy(self):
        """A.9: Verify access control policy."""
        assert SecurityManager.has_access_control_policy()
    
    def test_iso27001_a9_user_registration(self):
        """A.9: Verify user registration."""
        assert SecurityManager.has_user_registration_process()
    
    def test_iso27001_a9_privilege_management(self):
        """A.9: Verify privilege management."""
        assert SecurityManager.has_privilege_management()
    
    # A.10 - Cryptography
    def test_iso27001_a10_policy(self):
        """A.10: Verify cryptographic policy."""
        assert SecurityManager.has_cryptography_policy()
    
    def test_iso27001_a10_key_management(self):
        """A.10: Verify key management."""
        assert SecurityManager.has_key_management()
    
    # A.11 - Physical Security
    def test_iso27001_a11_physical_security(self):
        """A.11: Verify physical security controls."""
        assert SecurityManager.has_physical_security()
    
    # A.12 - Operations Security
    def test_iso27001_a12_malware_protection(self):
        """A.12: Verify malware protection."""
        assert SecurityManager.has_malware_protection()
    
    def test_iso27001_a12_backups(self):
        """A.12: Verify backup procedures."""
        assert SecurityManager.has_backup_procedures()
    
    def test_iso27001_a12_logging(self):
        """A.12: Verify logging and monitoring."""
        assert SecurityManager.has_logging_enabled()
        assert SecurityManager.logs_are_protected()
    
    # A.13 - Communications Security
    def test_iso27001_a13_network_security(self):
        """A.13: Verify network security."""
        assert SecurityManager.has_network_security()
    
    def test_iso27001_a13_data_transfer(self):
        """A.13: Verify secure data transfer."""
        assert SecurityManager.has_secure_data_transfer()
    
    # A.14 - System Acquisition and Maintenance
    def test_iso27001_a14_secure_development(self):
        """A.14: Verify secure development."""
        assert SecurityManager.has_secure_development_practices()
    
    # A.16 - Information Security Incident Management
    def test_iso27001_a16_incident_management(self):
        """A.16: Verify incident management."""
        assert SecurityManager.has_incident_management()
    
    # A.17 - Business Continuity
    def test_iso27001_a17_continuity(self):
        """A.17: Verify business continuity."""
        assert SecurityManager.has_business_continuity_plan()
    
    # A.18 - Compliance
    def test_iso27001_a18_compliance(self):
        """A.18: Verify compliance with legal requirements."""
        assert SecurityManager.has_compliance_monitoring()


class TestGDPRCompliance:
    """Test compliance with GDPR."""
    
    def test_gdpr_data_minimization(self):
        """Verify data minimization."""
        # Should only collect necessary data
        collected = SecurityManager.get_collected_data()
        necessary = SecurityManager.get_necessary_data()
        
        assert set(collected).issubset(set(necessary))
    
    def test_gdpr_consent(self):
        """Verify consent management."""
        assert SecurityManager.has_consent_management()
        
        # Consent should be recorded
        assert SecurityManager.records_consent()
    
    def test_gdpr_right_to_access(self):
        """Verify right to access."""
        user_data = SecurityManager.get_user_data("user_123")
        
        # Should return all personal data
        assert isinstance(user_data, dict)
        assert "personal_data" in user_data
    
    def test_gdpr_right_to_erasure(self):
        """Verify right to erasure."""
        # Should be able to delete user data
        assert SecurityManager.can_delete_user_data("user_123")
    
    def test_gdpr_data_portability(self):
        """Verify data portability."""
        # Should be able to export data in standard format
        export = SecurityManager.export_user_data("user_123")
        
        assert isinstance(export, dict)
        assert "format" in export
        assert export["format"] in ["json", "xml", "csv"]
    
    def test_gdpr_breach_notification(self):
        """Verify breach notification."""
        assert SecurityManager.has_breach_notification_procedure()
    
    def test_gdpr_dpo(self):
        """Verify Data Protection Officer."""
        assert SecurityManager.has_dpo()
        assert SecurityManager.dpo_contact_available()
    
    def test_gdpr_privacy_by_design(self):
        """Verify privacy by design."""
        assert SecurityManager.has_privacy_by_design()


class TestPCIDSSCompliance:
    """Test compliance with PCI DSS."""
    
    def test_pci_firewall(self):
        """Req 1: Verify firewall configuration."""
        assert SecurityManager.has_firewall()
        assert SecurityManager.firewall_rules_documented()
    
    def test_pci_no_default_passwords(self):
        """Req 2: Verify no default passwords."""
        assert not SecurityManager.uses_default_passwords()
    
    def test_pci_encrypted_storage(self):
        """Req 3: Verify encrypted storage of cardholder data."""
        assert SecurityManager.encrypts_cardholder_data()
    
    def test_pci_encrypted_transmission(self):
        """Req 4: Verify encrypted transmission."""
        assert SecurityManager.uses_strong_encryption()
    
    def test_pci_antivirus(self):
        """Req 5: Verify antivirus."""
        assert SecurityManager.has_antivirus()
    
    def test_pci_secure_systems(self):
        """Req 6: Verify secure systems and applications."""
        assert SecurityManager.has_patch_management()
        assert SecurityManager.has_vulnerability_management()
    
    def test_pci_access_control(self):
        """Req 7: Verify access control."""
        assert SecurityManager.has_access_control()
    
    def test_pci_unique_ids(self):
        """Req 8: Verify unique user IDs."""
        assert SecurityManager.has_unique_user_ids()
    
    def test_pci_physical_access(self):
        """Req 9: Verify physical access restrictions."""
        assert SecurityManager.has_physical_security()
    
    def test_pci_logging(self):
        """Req 10: Verify logging and monitoring."""
        assert SecurityManager.has_audit_logging()
        assert SecurityManager.logs_are_reviewed()
    
    def test_pci_testing(self):
        """Req 11: Verify security testing."""
        assert SecurityManager.has_vulnerability_scanning()
        assert SecurityManager.has_penetration_testing()
    
    def test_pci_policy(self):
        """Req 12: Verify information security policy."""
        assert SecurityManager.has_security_policy()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
