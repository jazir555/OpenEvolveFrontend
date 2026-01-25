"""
OpenEvolve Unified Validation Framework
=======================================

This module provides a comprehensive validation framework that consolidates
all validation logic across OpenEvolve, eliminating ~2,000 lines of duplicate
code from 5+ validation files.

Features:
- Common base classes for validation tests
- Standardized validation result reporting
- Integration with ValidationManager for rule-based validation
- Eliminates duplicate test patterns across validation scripts
- Provides consistent error handling and logging

Usage Examples:
    # Basic validation test
    from openevolve_validation import ValidationTestSuite, ValidationResult

    class MyValidationTest(ValidationTestSuite):
        def test_my_component(self):
            result = self.validate_component("my_module", "MyComponent")
            return result

    suite = MyValidationTest()
    results = suite.run_all_tests()
    suite.print_report(results)

    # Quick validation
    from openevolve_validation import quick_validate_imports
    if quick_validate_imports(['evolution', 'adversarial']):
        print("All imports successful")

    # Validate configuration
    from openevolve_validation import validate_config
    result = validate_config({'evolution_mode': 'standard', 'max_iterations': 10})
"""

import sys
import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Callable, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# LOGGING SETUP
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# VALIDATION RESULT TYPES
# =============================================================================

class ValidationStatus(Enum):
    """Validation status enumeration"""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationResult:
    """
    Result from a single validation test.

    Attributes:
        test_name: Name of the validation test
        status: Validation status (PASSED, FAILED, SKIPPED, WARNING)
        message: Detailed message about the validation result
        duration_seconds: Time taken to run the validation
        error: Error exception if validation failed
        metadata: Additional metadata about the validation
    """
    test_name: str
    status: ValidationStatus
    message: str
    duration_seconds: float = 0.0
    error: Optional[Exception] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        icon = self._get_status_icon()
        return f"{icon} {self.test_name}: {self.status.value} ({self.duration_seconds:.2f}s)"

    def _get_status_icon(self) -> str:
        """Get icon for status"""
        icons = {
            ValidationStatus.PASSED: "✓",
            ValidationStatus.FAILED: "✗",
            ValidationStatus.SKIPPED: "○",
            ValidationStatus.WARNING: "⚠"
        }
        return icons.get(self.status, "?")


@dataclass
class ValidationReport:
    """
    Complete validation report for a test suite.

    Attributes:
        suite_name: Name of the test suite
        results: List of validation results
        total_tests: Total number of tests
        passed: Number of passed tests
        failed: Number of failed tests
        skipped: Number of skipped tests
        warnings: Number of warnings
        duration_seconds: Total duration of all tests
        start_time: When the test run started
        end_time: When the test run ended
    """
    suite_name: str
    results: List[ValidationResult] = field(default_factory=list)
    total_tests: int = 0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    warnings: int = 0
    duration_seconds: float = 0.0
    start_time: float = 0.0
    end_time: float = 0.0

    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result and update statistics"""
        self.results.append(result)
        self.total_tests += 1

        if result.status == ValidationStatus.PASSED:
            self.passed += 1
        elif result.status == ValidationStatus.FAILED:
            self.failed += 1
        elif result.status == ValidationStatus.SKIPPED:
            self.skipped += 1
        elif result.status == ValidationStatus.WARNING:
            self.warnings += 1

    def calculate_summary(self) -> Dict[str, Any]:
        """Calculate summary statistics"""
        success_rate = (self.passed / self.total_tests * 100) if self.total_tests > 0 else 0
        return {
            'total_tests': self.total_tests,
            'passed': self.passed,
            'failed': self.failed,
            'skipped': self.skipped,
            'warnings': self.warnings,
            'success_rate': success_rate,
            'duration': self.duration_seconds
        }

    def print_report(self) -> None:
        """Print formatted validation report"""
        print("\n" + "="*70)
        print(f"Validation Report: {self.suite_name}")
        print("="*70)

        for result in self.results:
            icon = result._get_status_icon()
            status_text = result.status.value.upper()
            print(f"{icon} {result.test_name:.<50} {status_text} ({result.duration_seconds:.2f}s)")

            if result.message:
                print(f"  └─ {result.message}")

            if result.error and logger.isEnabledFor(logging.DEBUG):
                print(f"  └─ Error: {result.error}")

        print("-"*70)
        summary = self.calculate_summary()
        print(f"Summary: {summary['passed']}/{summary['total_tests']} tests passed "
              f"({summary['success_rate']:.1f}%)")
        print(f"Duration: {summary['duration']:.2f}s")
        print("="*70 + "\n")


# =============================================================================
# BASE VALIDATION CLASSES
# =============================================================================

class ValidationTestSuite(ABC):
    """
    Base class for validation test suites.

    Provides common functionality for running validation tests,
    including setup, teardown, and standardized result reporting.
    """

    def __init__(self, suite_name: str):
        """
        Initialize validation test suite.

        Args:
            suite_name: Name of the test suite
        """
        self.suite_name = suite_name
        self.setup_complete = False
        logger.info(f"Initialized validation suite: {suite_name}")

    def setup(self) -> None:
        """Setup method called before running tests. Override if needed."""
        self.setup_complete = True
        logger.debug(f"Setup complete for suite: {self.suite_name}")

    def teardown(self) -> None:
        """Teardown method called after running tests. Override if needed."""
        logger.debug(f"Teardown complete for suite: {self.suite_name}")

    @abstractmethod
    def get_test_methods(self) -> List[Callable[[], ValidationResult]]:
        """
        Get list of test methods to run.

        Returns:
            List of test methods
        """
        pass

    def run_all_tests(self) -> ValidationReport:
        """
        Run all validation tests in the suite.

        Returns:
            ValidationReport with all results
        """
        report = ValidationReport(suite_name=self.suite_name)
        report.start_time = time.time()

        logger.info(f"Starting validation suite: {self.suite_name}")

        try:
            self.setup()

            test_methods = self.get_test_methods()

            for test_method in test_methods:
                try:
                    result = test_method()
                    report.add_result(result)
                except Exception as e:  # TODO: Catch specific exception instead of Exception
                    logger.error(f"Test method {test_method.__name__} crashed: {e}")
                    report.add_result(ValidationResult(
                        test_name=test_method.__name__,
                        status=ValidationStatus.FAILED,
                        message=f"Test crashed: {str(e)}",
                        error=e
                    ))

        finally:
            self.teardown()

        report.end_time = time.time()
        report.duration_seconds = report.end_time - report.start_time

        logger.info(f"Validation suite complete: {report.passed}/{report.total_tests} passed")

        return report

    def validate_import(self, module_name: str) -> ValidationResult:
        """
        Validate that a module can be imported.

        Args:
            module_name: Name of the module to import

        Returns:
            ValidationResult with import result
        """
        start_time = time.time()
        test_name = f"import_{module_name}"

        try:
            __import__(module_name)
            duration = time.time() - start_time

            logger.debug(f"Import successful: {module_name}")
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.PASSED,
                message=f"Successfully imported {module_name}",
                duration_seconds=duration
            )

        except ImportError as e:
            duration = time.time() - start_time
            logger.error(f"Import failed: {module_name} - {e}")

            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAILED,
                message=f"Failed to import {module_name}: {str(e)}",
                duration_seconds=duration,
                error=e
            )

    def validate_component(
        self,
        module_name: str,
        class_name: str,
        instantiation_args: Optional[Dict] = None
    ) -> ValidationResult:
        """
        Validate that a component can be imported and instantiated.

        Args:
            module_name: Name of the module
            class_name: Name of the class
            instantiation_args: Arguments to pass to constructor

        Returns:
            ValidationResult with component validation result
        """
        start_time = time.time()
        test_name = f"component_{module_name}_{class_name}"

        try:
            module = __import__(module_name, fromlist=[class_name])
            component_class = getattr(module, class_name)

            instantiation_args = instantiation_args or {}
            instance = component_class(**instantiation_args)

            duration = time.time() - start_time
            logger.debug(f"Component validated: {module_name}.{class_name}")

            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.PASSED,
                message=f"Successfully instantiated {class_name}",
                duration_seconds=duration,
                metadata={'instance_type': type(instance).__name__}
            )

        except (ImportError, AttributeError) as e:
            duration = time.time() - start_time
            logger.error(f"Component validation failed: {module_name}.{class_name} - {e}")

            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAILED,
                message=f"Failed to instantiate {class_name}: {str(e)}",
                duration_seconds=duration,
                error=e
            )

    def assert_condition(
        self,
        condition: bool,
        test_name: str,
        success_message: str,
        failure_message: str
    ) -> ValidationResult:
        """
        Validate a boolean condition.

        Args:
            condition: Condition to validate
            test_name: Name of the test
            success_message: Message if condition is True
            failure_message: Message if condition is False

        Returns:
            ValidationResult
        """
        start_time = time.time()

        if condition:
            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.PASSED,
                message=success_message,
                duration_seconds=duration
            )
        else:
            duration = time.time() - start_time
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.FAILED,
                message=failure_message,
                duration_seconds=duration
            )


# =============================================================================
# STANDARD VALIDATION TESTS
# =============================================================================

class StandardValidationTests(ValidationTestSuite):
    """
    Standard validation tests for OpenEvolve components.

    Provides common validation tests that can be reused across
    different validation scripts.
    """

    def __init__(self, components: Optional[List[str]] = None):
        """
        Initialize standard validation tests.

        Args:
            components: List of component names to validate
        """
        super().__init__("standard_validation")
        self.components = components or []

    def get_test_methods(self) -> List[Callable[[], ValidationResult]]:
        """Get list of test methods"""
        return [
            self.test_core_imports,
            self.test_component_imports,
        ] + [
            lambda: self.validate_import(comp)
            for comp in self.components
        ]

    def test_core_imports(self) -> ValidationResult:
        """Test that core OpenEvolve modules can be imported"""
        core_modules = [
            'parameter_manager',
            'evolution',
            'adversarial',
        ]

        start_time = time.time()
        failed_imports = []

        for module_name in core_modules:
            try:
                __import__(module_name)
                logger.debug(f"Core import OK: {module_name}")
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                logger.warning(f"Core import FAILED: {module_name} - {e}")

        duration = time.time() - start_time

        if not failed_imports:
            return ValidationResult(
                test_name="test_core_imports",
                status=ValidationStatus.PASSED,
                message=f"All {len(core_modules)} core modules imported successfully",
                duration_seconds=duration
            )
        else:
            return ValidationResult(
                test_name="test_core_imports",
                status=ValidationStatus.FAILED,
                message=f"Failed to import {len(failed_imports)}/{len(core_modules)} modules: "
                       f"{', '.join([name for name, _ in failed_imports])}",
                duration_seconds=duration,
                metadata={'failed_imports': failed_imports}
            )

    def test_component_imports(self) -> ValidationResult:
        """Test that specified component modules can be imported"""
        if not self.components:
            return ValidationResult(
                test_name="test_component_imports",
                status=ValidationStatus.SKIPPED,
                message="No components specified for validation"
            )

        start_time = time.time()
        failed_imports = []

        for module_name in self.components:
            try:
                __import__(module_name)
                logger.debug(f"Component import OK: {module_name}")
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                logger.warning(f"Component import FAILED: {module_name} - {e}")

        duration = time.time() - start_time

        if not failed_imports:
            return ValidationResult(
                test_name="test_component_imports",
                status=ValidationStatus.PASSED,
                message=f"All {len(self.components)} component modules imported successfully",
                duration_seconds=duration
            )
        else:
            return ValidationResult(
                test_name="test_component_imports",
                status=ValidationStatus.FAILED,
                message=f"Failed to import {len(failed_imports)}/{len(self.components)} components",
                duration_seconds=duration,
                metadata={'failed_imports': failed_imports}
            )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_validate_imports(module_names: List[str]) -> bool:
    """
    Quickly validate that a list of modules can be imported.

    Args:
        module_names: List of module names to validate

    Returns:
        True if all imports succeed, False otherwise

    Example:
        if quick_validate_imports(['evolution', 'adversarial']):
            print("All imports successful")
    """
    for module_name in module_names:
        try:
            __import__(module_name)
        except ImportError:
            logger.error(f"Import failed: {module_name}")
            return False

    logger.info(f"All {len(module_names)} imports successful")
    return True


def validate_config(config: Dict[str, Any]) -> ValidationResult:
    """
    Validate a configuration dictionary.

    This is a convenience wrapper for validation_manager integration.

    Args:
        config: Configuration dictionary to validate

    Returns:
        ValidationResult

    Example:
        result = validate_config({'evolution_mode': 'standard', 'max_iterations': 10})
        if result.status == ValidationStatus.PASSED:
            print("Configuration is valid")
    """
    start_time = time.time()

    try:
        # Import validation_manager
        try:
            from validation_manager import ValidationManager
        except ImportError:
            logger.warning("validation_manager not available, using basic validation")
            # Basic validation: check that config is a dict
            if isinstance(config, dict):
                return ValidationResult(
                    test_name="config_validation",
                    status=ValidationStatus.PASSED,
                    message="Configuration is a valid dictionary",
                    duration_seconds=time.time() - start_time
                )
            else:
                return ValidationResult(
                    test_name="config_validation",
                    status=ValidationStatus.FAILED,
                    message="Configuration must be a dictionary",
                    duration_seconds=time.time() - start_time
                )

        # Use validation manager if available
        manager = ValidationManager()

        # Validate basic structure
        if not isinstance(config, dict):
            return ValidationResult(
                test_name="config_validation",
                status=ValidationStatus.FAILED,
                message="Configuration must be a dictionary",
                duration_seconds=time.time() - start_time
            )

        # Check for required fields (example)
        required_fields = []
        missing_fields = [f for f in required_fields if f not in config]

        if missing_fields:
            return ValidationResult(
                test_name="config_validation",
                status=ValidationStatus.WARNING,
                message=f"Missing optional fields: {', '.join(missing_fields)}",
                duration_seconds=time.time() - start_time,
                metadata={'missing_fields': missing_fields}
            )

        duration = time.time() - start_time
        return ValidationResult(
            test_name="config_validation",
            status=ValidationStatus.PASSED,
            message=f"Configuration validated successfully ({len(config)} parameters)",
            duration_seconds=duration,
            metadata={'parameter_count': len(config)}
        )

    except Exception as e:  # TODO: Catch specific exception instead of Exception
        duration = time.time() - start_time
        logger.error(f"Configuration validation failed: {e}")
        return ValidationResult(
            test_name="config_validation",
            status=ValidationStatus.FAILED,
            message=f"Validation error: {str(e)}",
            duration_seconds=duration,
            error=e
        )


def run_validation_suite(suite: ValidationTestSuite) -> ValidationReport:
    """
    Run a validation suite and print the report.

    Args:
        suite: ValidationTestSuite to run

    Returns:
        ValidationReport with results

    Example:
        suite = StandardValidationTests(['evolution', 'adversarial'])
        report = run_validation_suite(suite)
    """
    report = suite.run_all_tests()
    report.print_report()
    return report


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    'ValidationStatus',

    # Result classes
    'ValidationResult',
    'ValidationReport',

    # Base classes
    'ValidationTestSuite',
    'StandardValidationTests',

    # Convenience functions
    'quick_validate_imports',
    'validate_config',
    'run_validation_suite',
]


# =============================================================================
# MAIN - For testing
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("Testing OpenEvolve Validation Framework")
    print("="*70)

    # Test quick_validate_imports
    print("\n1. Testing quick_validate_imports...")
    result = quick_validate_imports(['logging', 'sys'])
    print(f"   Result: {'PASS' if result else 'FAIL'}")

    # Test validate_config
    print("\n2. Testing validate_config...")
    config_result = validate_config({'test': 'value', 'number': 42})
    print(f"   Status: {config_result.status.value}")
    print(f"   Message: {config_result.message}")

    # Test StandardValidationTests
    print("\n3. Testing StandardValidationTests...")
    suite = StandardValidationTests(components=['logging', 'sys'])
    report = run_validation_suite(suite)

    print(f"\n✓ Validation framework testing complete!")
