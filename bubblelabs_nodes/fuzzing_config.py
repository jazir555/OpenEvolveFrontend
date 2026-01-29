"""
Fuzzing Configuration and Validation

Provides configuration management, validation, and environment
variable support for the fuzzing system.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import os
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class FuzzingInputType(Enum):
    """Supported fuzzing input types"""
    AUTO = "auto"
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    NULL = "null"
    EDGE_CASE = "edge_case"
    MALFORMED = "malformed"


@dataclass
class FuzzingConfig:
    """Complete fuzzing configuration"""

    # Enable/disable fuzzing
    enabled: bool = False

    # Iteration settings
    max_iterations: int = 1000
    min_iterations: int = 100

    # Timeout settings
    timeout_seconds: int = 30
    min_timeout_seconds: int = 1
    max_timeout_seconds: int = 300

    # Concurrency settings
    max_concurrent: int = 4
    min_concurrent: int = 1
    max_concurrent_limit: int = 100

    # Corpus settings
    corpus_size: int = 100
    min_corpus_size: int = 0
    max_corpus_size: int = 10000

    # Input generation
    input_types: List[str] = None
    random_seed: Optional[int] = None

    # Analysis settings
    crash_analysis_enabled: bool = True
    vulnerability_detection_enabled: bool = True
    pattern_recognition_enabled: bool = True

    # Reporting
    generate_reports: bool = True
    report_format: str = "text"  # text, markdown, json
    save_corpus: bool = False
    corpus_path: Optional[str] = None

    def __post_init__(self):
        if self.input_types is None:
            self.input_types = [FuzzingInputType.AUTO.value]

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate fuzzing configuration.

        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []

        # Validate enabled flag
        if not isinstance(self.enabled, bool):
            errors.append("enabled must be a boolean")

        # Validate iterations
        if self.max_iterations < self.min_iterations:
            errors.append(
                f"max_iterations ({self.max_iterations}) must be >= "
                f"min_iterations ({self.min_iterations})"
            )

        if self.max_iterations < 1:
            errors.append("max_iterations must be at least 1")

        if self.max_iterations > 1000000:
            errors.append("max_iterations must be <= 1,000,000")

        # Validate timeout
        if self.timeout_seconds < self.min_timeout_seconds:
            errors.append(
                f"timeout_seconds ({self.timeout_seconds}) must be >= "
                f"min_timeout_seconds ({self.min_timeout_seconds})"
            )

        if self.timeout_seconds > self.max_timeout_seconds:
            errors.append(
                f"timeout_seconds ({self.timeout_seconds}) must be <= "
                f"max_timeout_seconds ({self.max_timeout_seconds})"
            )

        # Validate concurrency
        if self.max_concurrent < self.min_concurrent:
            errors.append(
                f"max_concurrent ({self.max_concurrent}) must be >= "
                f"min_concurrent ({self.min_concurrent})"
            )

        if self.max_concurrent > self.max_concurrent_limit:
            errors.append(
                f"max_concurrent ({self.max_concurrent}) must be <= "
                f"max_concurrent_limit ({self.max_concurrent_limit})"
            )

        # Validate corpus
        if self.corpus_size < self.min_corpus_size:
            errors.append(
                f"corpus_size ({self.corpus_size}) must be >= "
                f"min_corpus_size ({self.min_corpus_size})"
            )

        if self.corpus_size > self.max_corpus_size:
            errors.append(
                f"corpus_size ({self.corpus_size}) must be <= "
                f"max_corpus_size ({self.max_corpus_size})"
            )

        # Validate input types
        valid_types = {t.value for t in FuzzingInputType}
        for input_type in self.input_types:
            if input_type not in valid_types:
                errors.append(
                    f"Invalid input_type: {input_type}. "
                    f"Valid types: {', '.join(valid_types)}"
                )

        # Validate report format
        valid_formats = ['text', 'markdown', 'json']
        if self.report_format not in valid_formats:
            errors.append(
                f"Invalid report_format: {self.report_format}. "
                f"Valid formats: {', '.join(valid_formats)}"
            )

        # Validate corpus path if saving corpus
        if self.save_corpus and not self.corpus_path:
            errors.append("corpus_path is required when save_corpus is True")

        return (len(errors) == 0, errors)

    @classmethod
    def from_environment(cls) -> 'FuzzingConfig':
        """
        Create configuration from environment variables.

        Environment Variables:
            FUZZING_ENABLED: Enable/disable fuzzing (default: false)
            FUZZ_ITERATIONS: Max iterations (default: 1000)
            FUZZ_TIMEOUT: Timeout in seconds (default: 30)
            FUZZ_MAX_CONCURRENT: Max concurrent executions (default: 4)
            FUZZ_CORPUS_SIZE: Corpus size limit (default: 100)
            FUZZ_INPUT_TYPES: Comma-separated input types (default: auto)
            FUZZ_CRASH_ANALYSIS: Enable crash analysis (default: true)
            FUZZ_REPORT_FORMAT: Report format (default: text)
            FUZZ_SAVE_CORPUS: Save corpus to file (default: false)
            FUZZ_CORPUS_PATH: Path to save corpus (optional)
        """
        return cls(
            enabled=os.getenv('FUZZING_ENABLED', 'false').lower() == 'true',
            max_iterations=int(os.getenv('FUZZ_ITERATIONS', '1000')),
            timeout_seconds=int(os.getenv('FUZZ_TIMEOUT', '30')),
            max_concurrent=int(os.getenv('FUZZ_MAX_CONCURRENT', '4')),
            corpus_size=int(os.getenv('FUZZ_CORPUS_SIZE', '100')),
            input_types=os.getenv('FUZZ_INPUT_TYPES', 'auto').split(','),
            crash_analysis_enabled=os.getenv('FUZZ_CRASH_ANALYSIS', 'true').lower() == 'true',
            report_format=os.getenv('FUZZ_REPORT_FORMAT', 'text'),
            save_corpus=os.getenv('FUZZ_SAVE_CORPUS', 'false').lower() == 'true',
            corpus_path=os.getenv('FUZZ_CORPUS_PATH'),
        )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'FuzzingConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            FuzzingConfig instance
        """
        # Filter out None values
        filtered = {k: v for k, v in config_dict.items() if v is not None}
        return cls(**filtered)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            'enabled': self.enabled,
            'max_iterations': self.max_iterations,
            'min_iterations': self.min_iterations,
            'timeout_seconds': self.timeout_seconds,
            'min_timeout_seconds': self.min_timeout_seconds,
            'max_timeout_seconds': self.max_timeout_seconds,
            'max_concurrent': self.max_concurrent,
            'min_concurrent': self.min_concurrent,
            'max_concurrent_limit': self.max_concurrent_limit,
            'corpus_size': self.corpus_size,
            'min_corpus_size': self.min_corpus_size,
            'max_corpus_size': self.max_corpus_size,
            'input_types': self.input_types,
            'random_seed': self.random_seed,
            'crash_analysis_enabled': self.crash_analysis_enabled,
            'vulnerability_detection_enabled': self.vulnerability_detection_enabled,
            'pattern_recognition_enabled': self.pattern_recognition_enabled,
            'generate_reports': self.generate_reports,
            'report_format': self.report_format,
            'save_corpus': self.save_corpus,
            'corpus_path': self.corpus_path,
        }

    def merge_with(self, other: 'FuzzingConfig') -> 'FuzzingConfig':
        """
        Merge this configuration with another, with other taking precedence.

        Args:
            other: Configuration to merge with

        Returns:
            Merged configuration
        """
        merged_dict = self.to_dict()
        other_dict = other.to_dict()

        # Update with non-None values from other
        for key, value in other_dict.items():
            if value is not None:
                merged_dict[key] = value

        return FuzzingConfig.from_dict(merged_dict)


class FuzzingConfigValidator:
    """
    Validates fuzzing configuration and provides helpful error messages.
    """

    @staticmethod
    def validate_iterations(iterations: int) -> Tuple[bool, Optional[str]]:
        """Validate iteration count"""
        if iterations < 1:
            return (False, "Iterations must be at least 1")
        if iterations > 1000000:
            return (False, "Iterations must be <= 1,000,000")
        return (True, None)

    @staticmethod
    def validate_timeout(timeout: int) -> Tuple[bool, Optional[str]]:
        """Validate timeout"""
        if timeout < 1:
            return (False, "Timeout must be at least 1 second")
        if timeout > 3600:
            return (False, "Timeout must be <= 3600 seconds (1 hour)")
        return (True, None)

    @staticmethod
    def validate_concurrency(concurrency: int) -> Tuple[bool, Optional[str]]:
        """Validate concurrency level"""
        if concurrency < 1:
            return (False, "Concurrency must be at least 1")
        if concurrency > 1000:
            return (False, "Concurrency must be <= 1000")
        return (True, None)

    @staticmethod
    def validate_corpus_size(size: int) -> Tuple[bool, Optional[str]]:
        """Validate corpus size"""
        if size < 0:
            return (False, "Corpus size must be non-negative")
        if size > 100000:
            return (False, "Corpus size must be <= 100,000")
        return (True, None)

    @staticmethod
    def validate_input_types(input_types: List[str]) -> Tuple[bool, Optional[str]]:
        """Validate input types"""
        valid_types = {t.value for t in FuzzingInputType}

        for input_type in input_types:
            if input_type not in valid_types:
                return (
                    False,
                    f"Invalid input type: {input_type}. "
                    f"Valid types: {', '.join(sorted(valid_types))}"
                )

        return (True, None)

    @staticmethod
    def validate_report_format(format: str) -> Tuple[bool, Optional[str]]:
        """Validate report format"""
        valid_formats = ['text', 'markdown', 'json']

        if format not in valid_formats:
            return (
                False,
                f"Invalid report format: {format}. "
                f"Valid formats: {', '.join(valid_formats)}"
            )

        return (True, None)

    @classmethod
    def validate_config(cls, config: FuzzingConfig) -> Tuple[bool, List[str]]:
        """
        Validate complete fuzzing configuration.

        Args:
            config: FuzzingConfig to validate

        Returns:
            Tuple of (is_valid, error_messages)
        """
        return config.validate()


def load_fuzzing_config(
    config_dict: Optional[Dict[str, Any]] = None,
    use_environment: bool = True
) -> FuzzingConfig:
    """
    Load fuzzing configuration from dictionary and/or environment.

    Args:
        config_dict: Optional configuration dictionary
        use_environment: Whether to use environment variables

    Returns:
        FuzzingConfig instance
    """
    # Start with environment config
    if use_environment:
        config = FuzzingConfig.from_environment()
    else:
        config = FuzzingConfig()

    # Merge with provided config
    if config_dict:
        config = config.merge_with(FuzzingConfig.from_dict(config_dict))

    # Validate
    is_valid, errors = config.validate()

    if not is_valid:
        error_msg = "Invalid fuzzing configuration:\n" + "\n".join(f"  - {e}" for e in errors)
        logger.error(error_msg)
        raise ValueError(error_msg)

    return config


def get_default_fuzzing_config() -> FuzzingConfig:
    """Get default fuzzing configuration"""
    return FuzzingConfig()


def get_development_fuzzing_config() -> FuzzingConfig:
    """Get development-friendly fuzzing configuration (faster, fewer iterations)"""
    return FuzzingConfig(
        enabled=True,
        max_iterations=100,
        timeout_seconds=5,
        max_concurrent=2,
        corpus_size=10,
        crash_analysis_enabled=True,
        generate_reports=True,
        report_format='text',
    )


def get_production_fuzzing_config() -> FuzzingConfig:
    """Get production fuzzing configuration (thorough, more iterations)"""
    return FuzzingConfig(
        enabled=True,
        max_iterations=10000,
        timeout_seconds=60,
        max_concurrent=8,
        corpus_size=1000,
        crash_analysis_enabled=True,
        vulnerability_detection_enabled=True,
        pattern_recognition_enabled=True,
        generate_reports=True,
        report_format='json',
        save_corpus=True,
        corpus_path='./fuzzing_corpus',
    )


# Convenience function for quick validation
def validate_fuzzing_settings(
    iterations: int = None,
    timeout: int = None,
    concurrency: int = None,
    corpus_size: int = None
) -> Tuple[bool, List[str]]:
    """
    Quickly validate fuzzing settings.

    Args:
        iterations: Number of iterations
        timeout: Timeout in seconds
        concurrency: Max concurrent executions
        corpus_size: Corpus size limit

    Returns:
        Tuple of (is_valid, error_messages)
    """
    errors = []

    if iterations is not None:
        valid, msg = FuzzingConfigValidator.validate_iterations(iterations)
        if not valid:
            errors.append(msg)

    if timeout is not None:
        valid, msg = FuzzingConfigValidator.validate_timeout(timeout)
        if not valid:
            errors.append(msg)

    if concurrency is not None:
        valid, msg = FuzzingConfigValidator.validate_concurrency(concurrency)
        if not valid:
            errors.append(msg)

    if corpus_size is not None:
        valid, msg = FuzzingConfigValidator.validate_corpus_size(corpus_size)
        if not valid:
            errors.append(msg)

    return (len(errors) == 0, errors)
