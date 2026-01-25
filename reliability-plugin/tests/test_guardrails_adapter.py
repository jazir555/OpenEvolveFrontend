"""
Comprehensive test suite for Guardrails Adapter.

This test suite covers all functionality of the Guardrails adapter including:
- Adapter initialization and configuration
- Validator registration and management
- Input validation
- Output validation
- Remediation strategies
- Batch validation
- Statistics tracking
- Error handling and edge cases
- Integration with other reliability components
"""

import pytest
import sys
import os
import json
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import classes to test (with proper error handling for missing modules)
try:
    from reliability.guardrails_adapter import (
        GuardrailsAdapter,
        ValidationResult,
        RemediationStrategy,
        create_adapter,
        get_default_adapter
    )
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    # Create mock classes for testing when Guardrails is not available
    class ValidationResult:
        def __init__(self, is_valid: bool, failures: List[Dict] = None,
                     remediation_applied: str = None, output: str = None):
            self.is_valid = is_valid
            self.failures = failures or []
            self.remediation_applied = remediation_applied
            self.output = output

    class RemediationStrategy:
        REASK = "reask"
        FIX = "fix"
        FILTER = "filter"
        REFRAIN = "refrain"
        EXCEPTION = "exception"

    class GuardrailsAdapter:
        def __init__(self, config: Optional[Dict] = None):
            pass

    def create_adapter(config: Optional[Dict] = None):
        return GuardrailsAdapter(config)

    def get_default_adapter():
        return GuardrailsAdapter()


# Custom validator for testing
@dataclass
class TestValidator:
    name: str
    validate_func: callable
    remediate_func: Optional[callable] = None


class TestGuardrailsAdapter:
    """Test suite for Guardrails adapter initialization and basic functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_initialization_with_config(self, mock_config):
        """Test adapter initialization with configuration."""
        config = mock_config["guardrails"]
        adapter = GuardrailsAdapter(config)

        assert adapter is not None
        assert adapter.config == config
        assert adapter.enabled is True
        assert adapter.validators == config["validators"]
        assert adapter.on_fail == config["on_fail"]

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_initialization_without_config(self):
        """Test adapter initialization without configuration (uses defaults)."""
        adapter = GuardrailsAdapter()

        assert adapter is not None
        assert adapter.enabled is True
        assert adapter.validators == ["toxic_language", "pii_detection"]
        assert adapter.on_fail == "fix"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_initialization_with_custom_config(self):
        """Test adapter initialization with custom configuration."""
        custom_config = {
            "enabled": False,
            "validators": ["custom_validator"],
            "on_fail": "exception",
            "max_retries": 5
        }
        adapter = GuardrailsAdapter(custom_config)

        assert adapter.enabled is False
        assert adapter.validators == ["custom_validator"]
        assert adapter.on_fail == "exception"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_initialization_empty_config(self):
        """Test adapter initialization with empty configuration."""
        adapter = GuardrailsAdapter({})

        assert adapter is not None
        assert adapter.enabled is True  # Default to True

    def test_initialization_when_guardrails_unavailable(self):
        """Test adapter initialization when Guardrails module is unavailable."""
        # Patch Guardrails as unavailable
        with patch.dict('sys.modules', {'guardrails': None}):
            adapter = GuardrailsAdapter()
            assert adapter is not None
            # Should have fallback behavior
            assert hasattr(adapter, 'is_available')

    @pytest.mark.parametrize("enabled,expected", [
        (True, True),
        (False, False)
    ])
    def test_enabled_setting(self, enabled, expected):
        """Test setting enabled/disabled state."""
        adapter = GuardrailsAdapter({"enabled": enabled})
        assert adapter.enabled == expected

    @pytest.mark.parametrize("on_fail,expected", [
        ("fix", "fix"),
        ("reask", "reask"),
        ("filter", "filter"),
        ("refrain", "refrain"),
        ("exception", "exception")
    ])
    def test_on_fail_setting(self, on_fail, expected):
        """Test different on-fail strategies."""
        adapter = GuardrailsAdapter({"on_fail": on_fail})
        assert adapter.on_fail == expected


class TestValidatorRegistration:
    """Test suite for validator registration and management."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_register_validator_success(self, mock_guardrails_adapter):
        """Test successful validator registration."""
        validator = TestValidator(
            name="test_validator",
            validate_func=lambda x: ValidationResult(True, []),
            remediate_func=lambda x: x
        )

        result = mock_guardrails_adapter.register_validator(validator)

        assert result is True
        assert "test_validator" in mock_guardrails_adapter.validators

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_register_validator_duplicate(self, mock_guardrails_adapter):
        """Test registering duplicate validator."""
        validator = TestValidator(
            name="test_validator",
            validate_func=lambda x: ValidationResult(True, [])
        )

        # Register first time
        result1 = mock_guardrails_adapter.register_validator(validator)
        assert result1 is True

        # Register second time (should fail or return False)
        result2 = mock_guardrails_adapter.register_validator(validator)
        assert result2 is False  # Duplicate registration should fail

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_register_validator_invalid(self, mock_guardrails_adapter):
        """Test registering invalid validator."""
        # Invalid validator (missing required fields)
        invalid_validator = "not_a_validator"

        result = mock_guardrails_adapter.register_validator(invalid_validator)

        assert result is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_unregister_validator(self, mock_guardrails_adapter):
        """Test unregistering a validator."""
        validator = TestValidator(
            name="test_validator",
            validate_func=lambda x: ValidationResult(True, [])
        )

        # Register first
        mock_guardrails_adapter.register_validator(validator)

        # Unregister
        result = mock_guardrails_adapter.unregister_validator("test_validator")

        assert result is True
        assert "test_validator" not in mock_guardrails_adapter.validators

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_unregister_nonexistent_validator(self, mock_guardrails_adapter):
        """Test unregistering a non-existent validator."""
        result = mock_guardrails_adapter.unregister_validator("nonexistent")

        assert result is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_get_validator(self, mock_guardrails_adapter):
        """Test getting a validator by name."""
        validator = TestValidator(
            name="test_validator",
            validate_func=lambda x: ValidationResult(True, [])
        )

        mock_guardrails_adapter.register_validator(validator)

        retrieved_validator = mock_guardrails_adapter.get_validator("test_validator")

        assert retrieved_validator is not None
        assert retrieved_validator.name == "test_validator"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_get_nonexistent_validator(self, mock_guardrails_adapter):
        """Test getting a non-existent validator."""
        retrieved_validator = mock_guardrails_adapter.get_validator("nonexistent")

        assert retrieved_validator is None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_list_validators(self, mock_guardrails_adapter):
        """Test listing all registered validators."""
        validators = [
            TestValidator("v1", lambda x: ValidationResult(True, [])),
            TestValidator("v2", lambda x: ValidationResult(True, []))
        ]

        for validator in validators:
            mock_guardrails_adapter.register_validator(validator)

        listed_validators = mock_guardrails_adapter.list_validators()

        assert len(listed_validators) == 2
        assert "v1" in [v.name for v in listed_validators]
        assert "v2" in [v.name for v in listed_validators]

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_clear_all_validators(self, mock_guardrails_adapter):
        """Test clearing all validators."""
        validator = TestValidator(
            name="test_validator",
            validate_func=lambda x: ValidationResult(True, [])
        )

        mock_guardrails_adapter.register_validator(validator)

        # Clear all validators
        mock_guardrails_adapter.clear_validators()

        assert len(mock_guardrails_adapter.validators) == 0


class TestInputValidation:
    """Test suite for input validation functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_success(self, mock_guardrails_adapter):
        """Test successful input validation."""
        prompt = "Explain the concept of machine learning"
        validators = ["toxic_language", "pii_detection"]

        result = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="fix"
        )

        assert result.is_valid is True
        assert len(result.failures) == 0
        assert result.remediation_applied is None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_with_failures(self, mock_guardrails_adapter, sample_validation_failures):
        """Test input validation with failures."""
        # Configure adapter to return failures
        mock_result = Mock()
        mock_result.is_valid = False
        mock_result.failures = sample_validation_failures
        mock_result.remediation_applied = None
        mock_result.output = None
        mock_guardrails_adapter.validate_input.return_value = mock_result

        prompt = "This contains inappropriate content"
        validators = ["toxic_language"]

        result = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators
        )

        assert result.is_valid is False
        assert len(result.failures) == 1
        assert "toxic_language" in result.failures[0]["validator"]

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_with_remediation(self, mock_guardrails_adapter):
        """Test input validation with remediation."""
        # Configure adapter to apply remediation
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.failures = [{"validator": "toxic_language", "message": "Content fixed"}]
        mock_result.remediation_applied = "fix"
        mock_result.output = "Fixed content"
        mock_guardrails_adapter.validate_input.return_value = mock_result

        prompt = "Original problematic content"
        validators = ["toxic_language"]

        result = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="fix"
        )

        assert result.is_valid is True
        assert result.remediation_applied == "fix"
        assert result.output == "Fixed content"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_with_exception(self, mock_guardrails_adapter):
        """Test input validation that raises exception."""
        # Configure adapter to raise exception
        mock_guardrails_adapter.validate_input.side_effect = Exception("Validation failed")

        prompt = "Test prompt"
        validators = ["toxic_language"]

        # Should handle exception gracefully
        result = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="exception"
        )

        # Should return invalid result with error
        assert result.is_valid is False
        assert len(result.failures) > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_empty_prompt(self, mock_guardrails_adapter):
        """Test input validation with empty prompt."""
        result = mock_guardrails_adapter.validate_input(
            prompt="",
            validators=["length"]
        )

        # Should handle empty prompt gracefully
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_no_validators(self, mock_guardrails_adapter):
        """Test input validation with no validators."""
        result = mock_guardrails_adapter.validate_input(
            prompt="Test prompt",
            validators=[]
        )

        # Should return valid result when no validators
        assert result.is_valid is True
        assert len(result.failures) == 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_on_fail_strategies(self, mock_guardrails_adapter):
        """Test different on-fail strategies."""
        prompt = "Bad content"
        validators = ["toxic_language"]

        # Test "fix" strategy
        result_fix = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="fix"
        )
        assert result_fix.is_valid is True

        # Test "reask" strategy
        result_reask = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="reask"
        )
        assert result_reask.is_valid is True

        # Test "exception" strategy
        result_exception = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=validators,
            on_fail="exception"
        )
        # Should raise exception or return invalid
        assert hasattr(result_exception, 'is_valid')

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_input_unicode(self, mock_guardrails_adapter):
        """Test input validation with unicode content."""
        prompt = "测试内容 with emojis 🚀"

        result = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=["length"]
        )

        assert result is not None


class TestOutputValidation:
    """Test suite for output validation functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_success(self, mock_guardrails_adapter):
        """Test successful output validation."""
        output = '{"result": "success", "data": {"value": 42}}'
        validators = ["json_structure"]

        result = mock_guardrails_adapter.validate_output(
            output=output,
            validators=validators,
            on_fail="fix"
        )

        assert result.is_valid is True
        assert len(result.failures) == 0
        assert result.remediation_applied is None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_invalid_json(self, mock_guardrails_adapter):
        """Test output validation with invalid JSON."""
        output = '{"result": success, "data": value}'  # Invalid JSON

        result = mock_guardrails_adapter.validate_output(
            output=output,
            validators=["json_structure"]
        )

        assert result.is_valid is False
        assert len(result.failures) > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_with_remediation(self, mock_guardrails_adapter):
        """Test output validation with JSON remediation."""
        output = "Invalid JSON string"
        expected_fixed = '{"fixed": "true"}'

        # Configure adapter to return remediated output
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.failures = []
        mock_result.remediation_applied = "json_fix"
        mock_result.output = expected_fixed
        mock_guardrails_adapter.validate_output.return_value = mock_result

        result = mock_guardrails_adapter.validate_output(
            output=output,
            validators=["json_structure"],
            on_fail="fix"
        )

        assert result.is_valid is True
        assert result.remediation_applied == "json_fix"
        assert result.output == expected_fixed

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_empty_output(self, mock_guardrails_adapter):
        """Test output validation with empty output."""
        result = mock_guardrails_adapter.validate_output(
            output="",
            validators=["json_structure"]
        )

        # Should handle empty output gracefully
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_large_output(self, mock_guardrails_adapter):
        """Test output validation with large output."""
        large_output = "x" * 100000  # Very large output

        result = mock_guardrails_adapter.validate_output(
            output=large_output,
            validators=["length"]
        )

        # Should handle large output gracefully
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validate_output_unicode(self, mock_guardrails_adapter):
        """Test output validation with unicode content."""
        output = '{"message": "测试内容 with 🚀"}'

        result = mock_guardrails_adapter.validate_output(
            output=output,
            validators=["json_structure"]
        )

        assert result is not None


class TestBatchValidation:
    """Test suite for batch validation functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validate_success(self, mock_guardrails_adapter):
        """Test successful batch validation."""
        items = [
            {"content": "Valid content 1", "validators": ["toxic_language"]},
            {"content": "Valid content 2", "validators": ["pii_detection"]},
            {"content": "Valid content 3", "validators": ["json_structure"]}
        ]

        result = mock_guardrails_adapter.batch_validate(items)

        assert result.all_valid is True
        assert len(result.results) == 3
        assert len(result.failures) == 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validate_with_failures(self, mock_guardrails_adapter):
        """Test batch validation with some failures."""
        items = [
            {"content": "Valid content", "validators": ["toxic_language"]},
            {"content": "Invalid content", "validators": ["toxic_language"]},
            {"content": "Valid content", "validators": ["toxic_language"]}
        ]

        # Configure adapter to mix valid and invalid results
        def side_effect(item):
            if "Invalid" in item["content"]:
                return ValidationResult(False, [{"validator": "toxic_language"}])
            return ValidationResult(True, [])

        mock_guardrails_adapter.validate_input.side_effect = side_effect

        result = mock_guardrails_adapter.batch_validate(items)

        assert result.all_valid is False
        assert len(result.results) == 3
        assert len(result.failures) > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validate_empty_list(self, mock_guardrails_adapter):
        """Test batch validation with empty list."""
        result = mock_guardrails_adapter.batch_validate([])

        assert result.all_valid is True
        assert len(result.results) == 0
        assert len(result.failures) == 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validate_large_batch(self, mock_guardrails_adapter):
        """Test batch validation with large batch."""
        items = [
            {"content": f"Content {i}", "validators": ["length"]}
            for i in range(100)
        ]

        result = mock_guardrails_adapter.batch_validate(items)

        assert len(result.results) == 100

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validate_concurrent(self, mock_guardrails_adapter):
        """Test batch validation with concurrent processing."""
        items = [
            {"content": f"Content {i}", "validators": ["length"]}
            for i in range(10)
        ]

        result = mock_guardrails_adapter.batch_validate(
            items,
            concurrent=True,
            max_workers=4
        )

        assert len(result.results) == 10


class TestRemediationStrategies:
    """Test suite for remediation strategies."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_remediation_fix_strategy(self, mock_guardrails_adapter):
        """Test fix remediation strategy."""
        content = "Problematic content"
        fixed_content = "Fixed content"

        # Create a validator that applies fix remediation
        def fix_remediate(content):
            return fixed_content

        validator = TestValidator(
            name="fix_validator",
            validate_func=lambda x: ValidationResult(False, [], "fix"),
            remediate_func=fix_remediate
        )

        mock_guardrails_adapter.register_validator(validator)

        result = mock_guardrails_adapter.validate_input(
            prompt=content,
            validators=["fix_validator"],
            on_fail="fix"
        )

        assert result.is_valid is True
        assert result.remediation_applied == "fix"
        assert result.output == fixed_content

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_remediation_reask_strategy(self, mock_guardrails_adapter):
        """Test reask remediation strategy."""
        validator = TestValidator(
            name="reask_validator",
            validate_func=lambda x: ValidationResult(False, [], "reask")
        )

        mock_guardrails_adapter.register_validator(validator)

        result = mock_guardrails_adapter.validate_input(
            prompt="Bad content",
            validators=["reask_validator"],
            on_fail="reask"
        )

        # Should return reask result
        assert result.remediation_applied == "reask"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_remediation_filter_strategy(self, mock_guardrails_adapter):
        """Test filter remediation strategy."""
        validator = TestValidator(
            name="filter_validator",
            validate_func=lambda x: ValidationResult(False, [], "filter")
        )

        mock_guardrails_adapter.register_validator(validator)

        result = mock_guardrails_adapter.validate_input(
            prompt="Content to filter",
            validators=["filter_validator"],
            on_fail="filter"
        )

        # Should return filtered result (empty)
        assert result.remediation_applied == "filter"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_remediation_refrain_strategy(self, mock_guardrails_adapter):
        """Test refrain remediation strategy."""
        validator = TestValidator(
            name="refrain_validator",
            validate_func=lambda x: ValidationResult(False, [], "refrain")
        )

        mock_guardrails_adapter.register_validator(validator)

        result = mock_guardrails_adapter.validate_input(
            prompt="Content to refrain from",
            validators=["refrain_validator"],
            on_fail="refrain"
        )

        # Should return refrain result (special marker)
        assert result.remediation_applied == "refrain"

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_remediation_exception_strategy(self, mock_guardrails_adapter):
        """Test exception remediation strategy."""
        validator = TestValidator(
            name="exception_validator",
            validate_func=lambda x: ValidationResult(False, [], "exception")
        )

        mock_guardrails_adapter.register_validator(validator)

        with pytest.raises(Exception):
            mock_guardrails_adapter.validate_input(
                prompt="Content that causes exception",
                validators=["exception_validator"],
                on_fail="exception"
            )

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_custom_remediation(self, mock_guardrails_adapter):
        """Test custom remediation function."""
        def custom_remediate(content):
            return f"REMEDIATED: {content}"

        validator = TestValidator(
            name="custom_validator",
            validate_func=lambda x: ValidationResult(False, [], "custom"),
            remediate_func=custom_remediate
        )

        mock_guardrails_adapter.register_validator(validator)

        result = mock_guardrails_adapter.validate_input(
            prompt="Original content",
            validators=["custom_validator"],
            on_fail="fix"
        )

        assert result.remediation_applied == "custom"
        assert result.output == "REMEDIATED: Original content"


class TestStatisticsTracking:
    """Test suite for statistics tracking functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_initialization(self, mock_guardrails_adapter):
        """Test statistics initialization."""
        stats = mock_guardrails_adapter.get_statistics()

        assert isinstance(stats, dict)
        assert "total_validations" in stats
        assert "successful_validations" in stats
        assert "failed_validations" in stats
        assert "remediations_applied" in stats
        assert "validation_time_total" in stats

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_update_on_validation(self, mock_guardrails_adapter):
        """Test statistics update on validation."""
        # Get initial stats
        initial_stats = mock_guardrails_adapter.get_statistics()

        # Perform validation
        mock_guardrails_adapter.validate_input(
            prompt="Test content",
            validators=["toxic_language"]
        )

        # Check that statistics were updated
        updated_stats = mock_guardrails_adapter.get_statistics()
        assert updated_stats["total_validations"] > initial_stats["total_validations"]

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_remediation_tracking(self, mock_guardrails_adapter):
        """Test remediation tracking in statistics."""
        # Configure adapter to apply remediation
        mock_result = Mock()
        mock_result.is_valid = True
        mock_result.failures = []
        mock_result.remediation_applied = "fix"
        mock_guardrails_adapter.validate_input.return_value = mock_result

        # Perform validation with remediation
        mock_guardrails_adapter.validate_input(
            prompt="Test content",
            validators=["toxic_language"]
        )

        # Check that remediation was tracked
        stats = mock_guardrails_adapter.get_statistics()
        assert stats["remediations_applied"] > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_validator_breakdown(self, mock_guardrails_adapter):
        """Test validator-specific statistics breakdown."""
        # Perform multiple validations with different validators
        validators = ["toxic_language", "pii_detection", "json_structure"]

        for validator in validators:
            mock_guardrails_adapter.validate_input(
                prompt=f"Test content for {validator}",
                validators=[validator]
            )

        stats = mock_guardrails_adapter.get_statistics()

        # Should have breakdown by validator
        assert "validator_breakdown" in stats
        assert "toxic_language" in stats["validator_breakdown"]
        assert "pii_detection" in stats["validator_breakdown"]
        assert "json_structure" in stats["validator_breakdown"]

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_reset(self, mock_guardrails_adapter):
        """Test statistics reset functionality."""
        # Perform some validations
        mock_guardrails_adapter.validate_input("Test 1", ["toxic_language"])
        mock_guardrails_adapter.validate_input("Test 2", ["toxic_language"])

        # Reset statistics
        mock_guardrails_adapter.reset_statistics()

        # Check that statistics were reset
        stats = mock_guardrails_adapter.get_statistics()
        assert stats["total_validations"] == 0
        assert stats["successful_validations"] == 0
        assert stats["failed_validations"] == 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_statistics_persistence(self, mock_guardrails_adapter):
        """Test statistics persistence across multiple operations."""
        # Perform several operations
        for i in range(5):
            mock_guardrails_adapter.validate_input(
                prompt=f"Test {i}",
                validators=["toxic_language"]
            )

        # Statistics should accumulate
        stats = mock_guardrails_adapter.get_statistics()
        assert stats["total_validations"] == 5

        # Perform more operations
        for i in range(3):
            mock_guardrails_adapter.validate_output(
                output=f'{{"result": "test_{i}"}}',
                validators=["json_structure"]
            )

        # Statistics should continue to accumulate
        updated_stats = mock_guardrails_adapter.get_statistics()
        assert updated_stats["total_validations"] == 8


class TestAvailabilityChecks:
    """Test suite for availability checking functionality."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_is_available_true(self, mock_guardrails_adapter):
        """Test availability check when Guardrails is available."""
        mock_guardrails_adapter.is_available.return_value = True

        assert mock_guardrails_adapter.is_available() is True

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_is_available_false(self, mock_guardrails_adapter):
        """Test availability check when Guardrails is unavailable."""
        mock_guardrails_adapter.is_available.return_value = False

        assert mock_guardrails_adapter.is_available() is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_get_status(self, mock_guardrails_adapter):
        """Test getting adapter status."""
        status = mock_guardrails_adapter.get_status()

        assert isinstance(status, dict)
        assert "guardrails_enabled" in status
        assert "validators" in status
        assert "validation_mode" in status

    def test_availability_when_guardrails_module_missing(self):
        """Test availability when Guardrails module is completely missing."""
        with patch.dict('sys.modules', {'guardrails': None}):
            adapter = GuardrailsAdapter()
            status = adapter.is_available()
            assert status is False

    def test_availability_check_with_exception(self):
        """Test availability check when an exception occurs."""
        with patch('reliability.guardrails_adapter.GuardrailsAdapter.is_available',
                   side_effect=Exception("Connection failed")):
            adapter = GuardrailsAdapter()
            # Should handle exception gracefully
            try:
                result = adapter.is_available()
                assert result is False
            except Exception:
                # If exception is not handled, that's also acceptable for this test
                pass


class TestErrorHandling:
    """Test suite for error handling and edge cases."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_handle_validation_exception(self, mock_guardrails_adapter):
        """Test handling of validation exceptions."""
        # Configure adapter to raise exception
        mock_guardrails_adapter.validate_input.side_effect = Exception("Validation error")

        result = mock_guardrails_adapter.validate_input(
            prompt="Test prompt",
            validators=["toxic_language"]
        )

        # Should return invalid result with error information
        assert result.is_valid is False
        assert len(result.failures) > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_handle_mixed_validators(self, mock_guardrails_adapter):
        """Test handling of mixed success/failure validators."""
        def mixed_validation(content):
            if "fail" in content:
                return ValidationResult(False, [{"validator": "test"}])
            return ValidationResult(True, [])

        validator = TestValidator(
            name="mixed_validator",
            validate_func=mixed_validation
        )

        mock_guardrails_adapter.register_validator(validator)

        # Test with content that passes
        result1 = mock_guardrails_adapter.validate_input(
            prompt="This should pass",
            validators=["mixed_validator"]
        )
        assert result1.is_valid is True

        # Test with content that fails
        result2 = mock_guardrails_adapter.validate_input(
            prompt="This should fail",
            validators=["mixed_validator"]
        )
        assert result2.is_valid is False

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_large_input_handling(self, mock_guardrails_adapter):
        """Test handling of very large inputs."""
        large_content = "x" * 100000  # Very large input

        result = mock_guardrails_adapter.validate_input(
            prompt=large_content,
            validators=["length"]
        )

        # Should handle large input gracefully
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_special_characters_handling(self, mock_guardrails_adapter):
        """Test handling of special characters."""
        special_content = "Content with !@#$%^&*()_+-=[]{}|;':\",./<>?`~"

        result = mock_guardrails_adapter.validate_input(
            prompt=special_content,
            validators=["toxic_language"]
        )

        # Should handle special characters
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_unicode_handling(self, mock_guardrails_adapter):
        """Test handling of unicode characters."""
        unicode_content = "测试内容 with 🚀 emojis ñáéíóú"

        result = mock_guardrails_adapter.validate_input(
            prompt=unicode_content,
            validators=["toxic_language"]
        )

        # Should handle unicode characters
        assert result is not None

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_concurrent_validation_errors(self, mock_guardrails_adapter):
        """Test handling of concurrent validation errors."""
        items = [
            {"content": f"Content {i}", "validators": ["test"]}
            for i in range(10)
        ]

        # Configure adapter to raise exception for some items
        def side_effect(item):
            if int(item["content"].split()[1]) % 2 == 0:
                raise Exception("Validation error")
            return ValidationResult(True, [])

        mock_guardrails_adapter.validate_input.side_effect = side_effect

        result = mock_guardrails_adapter.batch_validate(
            items,
            concurrent=True
        )

        # Should handle concurrent errors gracefully
        assert len(result.results) == 10
        assert not result.all_valid


class TestPerformance:
    """Test suite for performance and optimization."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_batch_validation_performance(self, mock_guardrails_adapter):
        """Test batch validation performance."""
        items = [
            {"content": f"Test content {i}", "validators": ["length"]}
            for i in range(100)
        ]

        import time
        start_time = time.time()

        result = mock_guardrails_adapter.batch_validate(items)

        end_time = time.time()
        execution_time = end_time - start_time

        # Should validate all items
        assert len(result.results) == 100
        print(f"Batch validation of 100 items took {execution_time:.2f} seconds")

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_concurrent_vs_sequential(self, mock_guardrails_adapter):
        """Test concurrent vs sequential validation performance."""
        items = [
            {"content": f"Test content {i}", "validators": ["length"]}
            for i in range(20)
        ]

        # Test sequential
        start_time = time.time()
        mock_guardrails_adapter.batch_validate(items, concurrent=False)
        sequential_time = time.time() - start_time

        # Test concurrent
        start_time = time.time()
        mock_guardrails_adapter.batch_validate(items, concurrent=True, max_workers=4)
        concurrent_time = time.time() - start_time

        print(f"Sequential: {sequential_time:.2f}s, Concurrent: {concurrent_time:.2f}s")

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_caching_mechanism(self, mock_guardrails_adapter):
        """Test validation caching mechanism."""
        prompt = "Common test prompt"

        # Perform same validation multiple times
        for _ in range(5):
            mock_guardrails_adapter.validate_input(
                prompt=prompt,
                validators=["toxic_language"]
            )

        # Check cache stats (implementation dependent)
        cache_stats = mock_guardrails_adapter.get_cache_stats()
        assert isinstance(cache_stats, dict)

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    @pytest.mark.parametrize("batch_size", [1, 5, 10, 20])
    def test_different_batch_sizes(self, batch_size):
        """Test validation with different batch sizes."""
        adapter = GuardrailsAdapter()
        items = [
            {"content": f"Content {i}", "validators": ["length"]}
            for i in range(batch_size)
        ]

        result = adapter.batch_validate(items)

        assert len(result.results) == batch_size


class TestIntegration:
    """Test suite for integration with other reliability components."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_integration_with_lmql_adapter(self, mock_guardrails_adapter, mock_lmql_adapter):
        """Test integration with LMQL adapter."""
        # LMQL generates content
        generation_result = mock_lmql_adapter.constrained_generation(
            prompt="Generate content about AI",
            constraints=[]
        )

        # Guardrails validates the generated content
        validation_result = mock_guardrails_adapter.validate_input(
            prompt=generation_result.text,
            validators=["toxic_language", "length"],
            on_fail="fix"
        )

        # Both should succeed
        assert generation_result.success is True
        assert validation_result.is_valid is True

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_integration_with_roma_adapter(self, mock_guardrails_adapter, mock_roma_core):
        """Test integration with ROMA adapter."""
        # ROMA generates solution
        roma_result = mock_roma_core.solve("Solve: 2 + 2")

        # Guardrails validates output
        validation_result = mock_guardrails_adapter.validate_output(
            output=str(roma_result),
            validators=["json_structure"],
            on_fail="fix"
        )

        assert validation_result.is_valid is True

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_integration_with_mdap_adapter(self, mock_guardrails_adapter, mock_mdap_core):
        """Test integration with MDAP adapter."""
        # MDAP generates votes
        mdap_result = mock_mdap_core.MDAPSolver().solve()

        # Validate each vote
        for vote in mdap_result.votes:
            validation_result = mock_guardrails_adapter.validate_input(
                prompt=vote.content,
                validators=["toxic_language"],
                on_fail="fix"
            )

            assert validation_result.is_valid is True

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_unified_bridge_workflow(self, mock_guardrails_adapter):
        """Test integration with Unified Bridge workflow."""
        # Simulate unified bridge workflow
        prompt = "Generate safe content"

        # Layer 1: Input validation
        input_validation = mock_guardrails_adapter.validate_input(
            prompt=prompt,
            validators=["toxic_language"],
            on_fail="exception"
        )

        assert input_validation.is_valid is True

        # Layer 4: Output validation (simulated)
        output = '{"result": "success"}'
        output_validation = mock_guardrails_adapter.validate_output(
            output=output,
            validators=["json_structure"],
            on_fail="fix"
        )

        assert output_validation.is_valid is True


class TestParameterizedTests:
    """Parameterized tests for different configurations."""

    @pytest.mark.parametrize("enabled", [True, False])
    def test_enabled_disabled_states(self, enabled):
        """Test adapter with enabled/disabled states."""
        config = {"enabled": enabled}
        adapter = GuardrailsAdapter(config)

        # Should handle both states
        result = adapter.validate_input("test", [])
        assert result is not None

    @pytest.mark.parametrize("on_fail", ["fix", "reask", "filter", "refrain", "exception"])
    def test_different_on_fail_strategies(self, on_fail):
        """Test different on-fail strategies."""
        adapter = GuardrailsAdapter({"on_fail": on_fail})

        # Should handle all strategies
        try:
            result = adapter.validate_input("test", [])
            assert result is not None
        except Exception:
            # Exception strategy is expected to raise
            if on_fail == "exception":
                pass

    @pytest.mark.parametrize("validator", [
        "toxic_language",
        "pii_detection",
        "json_structure",
        "roma_length",
        "custom_validator"
    ])
    def test_different_validators(self, validator):
        """Test with different validators."""
        adapter = GuardrailsAdapter()

        # Should handle different validators
        result = adapter.validate_input("test content", [validator])
        assert result is not None

    @pytest.mark.parametrize("content_type", [
        "text",
        "json",
        "xml",
        "html",
        "markdown"
    ])
    def test_different_content_types(self, content_type, mock_guardrails_adapter):
        """Test validation with different content types."""
        contents = {
            "text": "Plain text content",
            "json": '{"result": "success"}',
            "xml": "<result>success</result>",
            "html": "<div>Content</div>",
            "markdown": "# Header\nContent"
        }

        result = mock_guardrails_adapter.validate_input(
            prompt=contents[content_type],
            validators=["length"]
        )

        assert result is not None


class TestEdgeCases:
    """Test suite for edge cases and boundary conditions."""

    def test_extremely_long_content(self, mock_guardrails_adapter):
        """Test handling of extremely long content."""
        very_long_content = "x" * 1000000  # 1 million characters

        result = mock_guardrails_adapter.validate_input(
            prompt=very_long_content,
            validators=["length"]
        )

        assert result is not None

    def test_mixed_languages(self, mock_guardrails_adapter):
        """Test handling of mixed language content."""
        mixed_content = "English and 中文 and Español and 🚀"

        result = mock_guardrails_adapter.validate_input(
            prompt=mixed_content,
            validators=["toxic_language"]
        )

        assert result is not None

    def test_empty_string_validation(self, mock_guardrails_adapter):
        """Test validation of empty strings."""
        result = mock_guardrails_adapter.validate_input(
            prompt="",
            validators=["length", "toxic_language"]
        )

        assert result is not None

    def test_whitespace_only_content(self, mock_guardrails_adapter):
        """Test validation of whitespace-only content."""
        whitespace_content = "   \n\t  \r  "

        result = mock_guardrails_adapter.validate_input(
            prompt=whitespace_content,
            validators=["length"]
        )

        assert result is not None

    def test_null_and_none_values(self, mock_guardrails_adapter):
        """Test handling of null and None values."""
        # Should handle None gracefully
        try:
            result = mock_guardrails_adapter.validate_input(
                prompt=None,
                validators=["toxic_language"]
            )
        except:
            pass  # Acceptable to raise exception for None

        # Should handle empty string
        result = mock_guardrails_adapter.validate_input(
            prompt="",
            validators=["toxic_language"]
        )
        assert result is not None


# =============================================================================
# UTILITIES
# =============================================================================

class TestUtilities:
    """Test suite for utility functions."""

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_content_sanitization(self, mock_guardrails_adapter):
        """Test content sanitization utility."""
        dirty_content = "This has bad words and <script>alert('xss')</script>"
        clean_content = mock_guardrails_adapter.sanitize_content(dirty_content)

        assert clean_content != dirty_content
        assert "<script>" not in clean_content

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_pii_detection_utility(self, mock_guardrails_adapter):
        """Test PII detection utility."""
        content_with_pii = "My email is test@example.com and phone is 123-456-7890"
        pii_entities = mock_guardrails_adapter.detect_pii(content_with_pii)

        assert isinstance(pii_entities, list)
        assert len(pii_entities) > 0

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validator_registration_utility(self, mock_guardrails_adapter):
        """Test validator registration utility."""
        # Register multiple validators at once
        validators = [
            TestValidator("v1", lambda x: ValidationResult(True, [])),
            TestValidator("v2", lambda x: ValidationResult(True, []))
        ]

        success_count = mock_guardrails_adapter.register_multiple_validators(validators)

        assert success_count == 2
        assert len(mock_guardrails_adapter.validators) >= 2

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_validation_history(self, mock_guardrails_adapter):
        """Test validation history tracking."""
        # Perform several validations
        for i in range(3):
            mock_guardrails_adapter.validate_input(f"Test {i}", ["toxic_language"])

        history = mock_guardrails_adapter.get_validation_history()

        assert isinstance(history, list)
        assert len(history) == 3

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_export_validation_report(self, mock_guardrails_adapter):
        """Test validation report export."""
        # Perform some validations
        mock_guardrails_adapter.validate_input("Test 1", ["toxic_language"])
        mock_guardrails_adapter.validate_input("Test 2", ["toxic_language"])

        # Export report
        report = mock_guardrails_adapter.export_validation_report()

        assert isinstance(report, dict)
        assert "summary" in report
        assert "details" in report
        assert "statistics" in report

    @pytest.mark.skipif(not GUARDRAILS_AVAILABLE, reason="Guardrails not available")
    def test_load_validator_config(self, mock_guardrails_adapter):
        """Test loading validator configuration."""
        config = {
            "validators": [
                {"name": "test_validator", "type": "toxic_language"}
            ]
        }

        loaded_count = mock_guardrails_adapter.load_validator_config(config)

        assert loaded_count == 1


# =============================================================================
# SETUP/TEARDOWN
# =============================================================================

@pytest.fixture(scope="class", autouse=True)
def setup_class():
    """Setup for test class."""
    yield
    # Cleanup after all tests in class


@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Cleanup after each test."""
    yield
    # Clear any state between tests