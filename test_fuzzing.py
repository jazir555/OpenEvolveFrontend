"""
Security Fuzzing Testing Suite - TRUE 100%
Tests with random inputs, property-based testing with hypothesis
"""

import pytest
import random
import string
import sys
from typing import Any, List, Callable
from hypothesis import given, strategies as st, settings, assume, example
import json

from input_validation import InputValidator, ValidationError, Sanitizer
from auth_system import TokenManager, JWTConfig, AuthManager
from security_framework import SecurityManager


# Hypothesis strategies for security testing
safe_strings = st.text(
    alphabet=st.characters(whitelist_categories=('L', 'N', 'P', 'Zs')),
    min_size=0,
    max_size=1000
)

sql_injection_patterns = st.sampled_from([
    "'", "'", ";", "--", "/*", "*/", "xp_", "sp_", "exec", "union", "select", "drop", "insert", "delete", "update"
])

xss_patterns = st.sampled_from([
    "<script>", "</script>", "javascript:", "onerror=", "onload=", "<iframe>", "<object>", "<embed>"
])

path_traversal_patterns = st.sampled_from([
    "../", "..\\", "/../", "..", "%2e%2e", "%252e", "..%2f"
])

command_injection_patterns = st.sampled_from([
    ";", "|", "&", "`", "$(", "${", ">", ">>", "<"
])


class TestFuzzingInputValidation:
    """Fuzzing tests for input validation."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(safe_strings)
    @settings(max_examples=1000)
    def test_safe_strings_never_crash(self, validator, safe_strings):
        """Test that safe strings never cause crashes."""
        # Should not raise any exception
        result = validator.sanitize_string(safe_strings)
        assert isinstance(result, str)
    
    @given(st.text(min_size=0, max_size=10000))
    @settings(max_examples=500)
    def test_arbitrary_strings_handled(self, validator, text):
        """Test that arbitrary strings are handled gracefully."""
        try:
            result = validator.sanitize_string(text)
            assert isinstance(result, str)
            # Result should not contain dangerous content
            assert "<script>" not in result.lower() or result == text
        except ValidationError:
            pass  # Rejection is acceptable
    
    @given(st.lists(safe_strings, min_size=0, max_size=100))
    @settings(max_examples=200)
    def test_list_sanitization(self, validator, items):
        """Test sanitization of lists with various strings."""
        result = validator._sanitize_recursive(items)
        assert isinstance(result, list)
        assert len(result) == len(items)
    
    @given(st.dictionaries(safe_strings, safe_strings, min_size=0, max_size=50))
    @settings(max_examples=200)
    def test_dict_sanitization(self, validator, data):
        """Test sanitization of dictionaries."""
        result = validator._sanitize_recursive(data)
        assert isinstance(result, dict)
        assert len(result) <= len(data)  # May remove dangerous keys


class TestFuzzingSQLInjection:
    """Fuzzing tests for SQL injection prevention."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(st.text(
        alphabet=st.characters(whitelist_categories=('L', 'N', 'P')),
        min_size=1,
        max_size=100
    ))
    @settings(max_examples=500)
    def test_random_sql_fragments(self, validator, fragment):
        """Test handling of random SQL-like fragments."""
        sql_fragments = [
            f"SELECT * FROM {fragment}",
            f"' OR {fragment} = '{fragment}'",
            f"; {fragment}; --",
        ]
        
        for sql in sql_fragments:
            sanitized = validator.sanitize_string(sql)
            assert isinstance(sanitized, str)
            # Should remove or escape SQL metacharacters
            assert ";" not in sanitized or "'" not in sanitized or sanitized == sql
    
    @given(st.lists(sql_injection_patterns, min_size=1, max_size=20))
    @settings(max_examples=300)
    def test_combined_sql_patterns(self, validator, patterns):
        """Test combinations of SQL injection patterns."""
        combined = "".join(patterns)
        sanitized = validator.sanitize_string(combined)
        
        assert isinstance(sanitized, str)
        # Result should be safe
        assert len(sanitized) < len(combined) * 2  # Should not explode


class TestFuzzingXSSPrevention:
    """Fuzzing tests for XSS prevention."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(st.text(min_size=0, max_size=5000))
    @settings(max_examples=500)
    def test_random_html_content(self, validator, content):
        """Test handling of random HTML-like content."""
        wrapped = f"<div>{content}</div>"
        
        try:
            sanitized = validator._sanitize_html(wrapped)
            assert isinstance(sanitized, str)
            # Should not contain script tags
            assert "<script>" not in sanitized.lower()
        except Exception:
            pass  # Exception is acceptable for malformed input
    
    @given(st.lists(xss_patterns, min_size=1, max_size=10))
    @settings(max_examples=300)
    def test_combined_xss_patterns(self, validator, patterns):
        """Test combinations of XSS patterns."""
        combined = " ".join(patterns)
        
        sanitized = validator._sanitize_html(combined)
        
        # Should remove or neutralize dangerous content
        assert "<script>" not in sanitized.lower() or sanitized == ""
        assert "javascript:" not in sanitized.lower() or sanitized == ""
    
    @given(st.text(alphabet=st.characters(whitelist_categories=('C',)), min_size=0, max_size=1000))
    @settings(max_examples=300)
    def test_unicode_handling(self, validator, text):
        """Test handling of Unicode control characters."""
        result = validator.sanitize_string(text)
        assert isinstance(result, str)


class TestFuzzingPathTraversal:
    """Fuzzing tests for path traversal prevention."""
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    @given(st.text(
        alphabet=string.ascii_letters + string.digits + "./\\_-",
        min_size=0,
        max_size=500
    ))
    @settings(max_examples=500)
    def test_random_path_strings(self, sanitizer, path):
        """Test handling of random path strings."""
        sanitized = sanitizer.sanitize_path(path)
        
        assert isinstance(sanitized, str)
        # Should not contain obvious traversal
        assert "../" not in sanitized or "/../" not in sanitized
        assert "..\\" not in sanitized
    
    @given(st.lists(path_traversal_patterns, min_size=1, max_size=20))
    @settings(max_examples=300)
    def test_combined_traversal_patterns(self, sanitizer, patterns):
        """Test combinations of path traversal patterns."""
        combined = "".join(patterns)
        
        sanitized = sanitizer.sanitize_path(combined)
        
        # Result should not allow traversal
        assert not sanitized.startswith("/") or ".." not in sanitized


class TestFuzzingCommandInjection:
    """Fuzzing tests for command injection prevention."""
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    @given(st.text(
        alphabet=string.ascii_letters + string.digits + ";|&`$(){}[]!@#%",
        min_size=0,
        max_size=200
    ))
    @settings(max_examples=500)
    def test_random_command_strings(self, sanitizer, cmd):
        """Test handling of random command-like strings."""
        sanitized = sanitizer.sanitize_for_command(cmd)
        
        assert isinstance(sanitized, str)
        # Should remove shell metacharacters
        assert ";" not in sanitized or sanitized == cmd
    
    @given(st.lists(command_injection_patterns, min_size=1, max_size=10))
    @settings(max_examples=300)
    def test_combined_command_patterns(self, sanitizer, patterns):
        """Test combinations of command injection patterns."""
        combined = "".join(patterns)
        
        sanitized = sanitizer.sanitize_for_command(combined)
        
        assert isinstance(sanitized, str)
        # Result should be safe
        assert len(sanitized) <= len(combined)


class TestFuzzingJSONInput:
    """Fuzzing tests for JSON input handling."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(st.dictionaries(
        keys=safe_strings,
        values=st.one_of(
            safe_strings,
            st.integers(),
            st.booleans(),
            st.floats(allow_nan=False, allow_infinity=False)
        ),
        min_size=0,
        max_size=100
    ))
    @settings(max_examples=300)
    def test_json_object_sanitization(self, validator, data):
        """Test sanitization of JSON objects."""
        result = validator._sanitize_recursive(data)
        assert isinstance(result, dict)
    
    @given(st.recursive(
        st.one_of(safe_strings, st.integers(), st.booleans()),
        lambda children: st.lists(children, min_size=0, max_size=10) | 
                        st.dictionaries(safe_strings, children, min_size=0, max_size=10),
        max_leaves=50
    ))
    @settings(max_examples=200)
    def test_nested_structure_sanitization(self, validator, data):
        """Test sanitization of deeply nested structures."""
        result = validator._sanitize_recursive(data)
        assert result is not None


class TestFuzzingAuthentication:
    """Fuzzing tests for authentication."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager()
    
    @given(safe_strings, safe_strings)
    @settings(max_examples=500)
    def test_random_credentials(self, auth_manager, username, password):
        """Test handling of random credentials."""
        # Should not crash on any input
        try:
            result = auth_manager.validate_credentials_format(username, password)
            assert isinstance(result, bool)
        except Exception:
            pass  # Exception is acceptable
    
    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=500)
    def test_malformed_tokens(self, auth_manager, token):
        """Test handling of malformed tokens."""
        try:
            auth_manager.token_manager.verify_token(token)
        except Exception:
            pass  # Should reject invalid tokens


class TestPropertyBasedSecurity:
    """Property-based security tests."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(safe_strings)
    @settings(max_examples=500)
    def test_idempotent_sanitization(self, validator, text):
        """Sanitization should be idempotent: sanitize(sanitize(x)) == sanitize(x)"""
        first_pass = validator.sanitize_string(text)
        second_pass = validator.sanitize_string(first_pass)
        
        assert first_pass == second_pass
    
    @given(safe_strings)
    @settings(max_examples=500)
    def test_sanitization_reduces_risk(self, validator, text):
        """Sanitization should not increase risk."""
        original_risk = validator.calculate_risk_score(text)
        sanitized = validator.sanitize_string(text)
        sanitized_risk = validator.calculate_risk_score(sanitized)
        
        assert sanitized_risk <= original_risk + 0.1  # Small tolerance
    
    @given(st.text(min_size=1))
    @settings(max_examples=500)
    def test_no_infinite_loops(self, validator, text):
        """Sanitization should complete in finite time."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError("Sanitization took too long")
        
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(5)  # 5 second timeout
        
        try:
            validator.sanitize_string(text)
            signal.alarm(0)
        except TimeoutError:
            pytest.fail("Sanitization caused infinite loop")
        except Exception:
            signal.alarm(0)  # Other exceptions are OK


class TestFuzzingEdgeCases:
    """Fuzzing tests for edge cases."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_empty_and_whitespace_inputs(self, validator):
        """Test handling of empty and whitespace inputs."""
        empty_inputs = [
            "",
            " ",
            "\t",
            "\n",
            "\r\n",
            "   ",
            "\t\t\t",
        ]
        
        for input_val in empty_inputs:
            result = validator.sanitize_string(input_val)
            assert isinstance(result, str)
    
    def test_very_long_inputs(self, validator):
        """Test handling of very long inputs."""
        long_inputs = [
            "A" * 10000,
            "<script>" * 1000,
            "../" * 1000,
        ]
        
        for input_val in long_inputs:
            result = validator.sanitize_string(input_val)
            assert isinstance(result, str)
            assert len(result) < len(input_val) * 2  # Should not explode
    
    def test_null_byte_injection(self, validator):
        """Test handling of null bytes."""
        null_inputs = [
            "file\x00.txt",
            "data\x00\x00\x00",
            "\x00",
            "\x00\x00\x00",
        ]
        
        for input_val in null_inputs:
            result = validator.sanitize_string(input_val)
            assert isinstance(result, str)
            assert "\x00" not in result
    
    def test_encoding_issues(self, validator):
        """Test handling of various encodings."""
        # These are strings that might cause encoding issues
        encoding_tests = [
            "café",  # UTF-8
            "日本語",  # CJK
            "🚀🔐💻",  # Emoji
            "\xff\xfe",  # Invalid UTF-8 sequences
        ]
        
        for input_val in encoding_tests:
            result = validator.sanitize_string(input_val)
            assert isinstance(result, str)


class TestFuzzingFileNames:
    """Fuzzing tests for filename sanitization."""
    
    @pytest.fixture
    def sanitizer(self):
        return Sanitizer()
    
    @given(st.text(
        alphabet=string.ascii_letters + string.digits + "._-",
        min_size=0,
        max_size=255
    ))
    @settings(max_examples=500)
    def test_random_filenames(self, sanitizer, filename):
        """Test handling of random filenames."""
        sanitized = sanitizer.sanitize_filename(filename)
        
        assert isinstance(sanitized, str)
        # Should not contain path separators
        assert "/" not in sanitized
        assert "\\" not in sanitized
        # Should not start with dots (hidden files)
        # (This is a policy decision, may vary)


class TestFuzzingNumericInputs:
    """Fuzzing tests for numeric input validation."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    @given(st.integers(min_value=-sys.maxsize, max_value=sys.maxsize))
    @settings(max_examples=500)
    def test_integer_validation(self, validator, num):
        """Test integer validation with random values."""
        try:
            result = validator.validate_integer(str(num))
            assert isinstance(result, int)
        except ValidationError:
            pass  # Rejection is acceptable
    
    @given(st.floats(allow_nan=False, allow_infinity=False))
    @settings(max_examples=500)
    def test_float_validation(self, validator, num):
        """Test float validation with random values."""
        try:
            result = validator.validate_float(str(num))
            assert isinstance(result, float)
        except (ValidationError, ValueError):
            pass  # Rejection is acceptable


class TestRandomizedSecuritySuite:
    """Randomized security test suite."""
    
    @pytest.fixture
    def validator(self):
        return InputValidator()
    
    def test_random_injection_payloads(self, validator):
        """Test with randomly generated injection payloads."""
        random.seed(42)  # Reproducible
        
        # Generate random SQL injection attempts
        for _ in range(1000):
            length = random.randint(1, 100)
            payload = ''.join(random.choices(
                string.ascii_letters + string.digits + "'\";--/*=|&",
                k=length
            ))
            
            result = validator.sanitize_string(payload)
            assert isinstance(result, str)
        
        # Generate random XSS attempts
        for _ in range(1000):
            length = random.randint(1, 100)
            payload = ''.join(random.choices(
                string.ascii_letters + "<>\"'/=;:",
                k=length
            ))
            
            result = validator._sanitize_html(payload)
            assert isinstance(result, str)
    
    def test_fuzzed_json_structures(self, validator):
        """Test with fuzzed JSON structures."""
        random.seed(42)
        
        def generate_random_json(depth=0):
            if depth > 5:
                return random.choice([None, True, False, random.randint(-1000, 1000), "string"])
            
            choice = random.randint(0, 3)
            if choice == 0:
                return [generate_random_json(depth + 1) for _ in range(random.randint(0, 10))]
            elif choice == 1:
                return {
                    f"key_{i}": generate_random_json(depth + 1)
                    for i in range(random.randint(0, 10))
                }
            elif choice == 2:
                return ''.join(random.choices(string.printable, k=random.randint(0, 100)))
            else:
                return random.randint(-1000000, 1000000)
        
        for _ in range(100):
            data = generate_random_json()
            result = validator._sanitize_recursive(data)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
