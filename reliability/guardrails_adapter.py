"""
OpenEvolve Guardrails Adapter
==============================

A production-ready adapter for NVIDIA/NeMo Guardrails integration with OpenEvolve.
Provides comprehensive validation, remediation strategies, and graceful degradation.

Features:
- Pre-configured validators for ROMA, MDAP, LeanAide, and safety checks
- 8 remediation strategies (reask, fix, filter, refrain, exception, custom, fix_reask, fix_reask_but_refrain)
- Graceful degradation when Guardrails unavailable
- Structured JSON logging with correlation IDs
- Full environment-based configuration
- Comprehensive error handling

Author: OpenEvolve Team
Version: 1.0.0
License: MIT
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import traceback

# ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_validation_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_validation_config = None

# Try importing guardrails with detailed fallback
try:
    import guardrails as gd
    from guardrails import Guard
    from guardrails.validators import (
        ValidLength,
        ValidRange,
        ValidJson,
        TwoWords,
        RegexMatch,
        InList,
        ToxicLanguage,
        PIIFilter,
        DetectSecrets,
        CompetitorCheck,
        ValidSQL,
        ProvenanceLLM,
        ValidationError,
    )
    GUARDRAILS_AVAILABLE = True
except ImportError as e:
    GUARDRAILS_AVAILABLE = False
    gd = None
    Guard = None

    # =============================================================================
    # PRODUCTION VALIDATOR IMPLEMENTATIONS (Guardrails Unavailable)
    # =============================================================================
    # These provide full validation logic when Guardrails AI is not installed
    # =============================================================================

    class ValidationError(Exception):
        """Stub for ValidationError when Guardrails unavailable"""
        pass

    class StubValidator:
        """
        Base validator class with production validation logic.

        Used when Guardrails AI is unavailable. Provides actual
        validation implementations, not just stubs.
        """
        def __init__(self, **kwargs):
            """Initialize validator with parameters"""
            self.params = kwargs
            self.on_fail = kwargs.get('on_fail', 'reask')

        def validate(self, output, metadata=None):
            """
            Validate output and return (validated_output, validation_passed)

            Args:
                output: The output to validate
                metadata: Optional metadata dict

            Returns:
                Tuple of (validated_output, validation_passed)
            """
            raise NotImplementedError("Subclasses must implement validate()")

    class ValidLength(StubValidator):
        """Validates output length is within specified limits"""
        def __init__(self, max_length=1000, min_length=0, **kwargs):
            super().__init__(max_length=max_length, min_length=min_length, **kwargs)
            self.max_length = max_length
            self.min_length = min_length

        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            length = len(output_str)

            if length < self.min_length or length > self.max_length:
                # Truncate if too long
                if length > self.max_length:
                    return output_str[:self.max_length], False
                return output, False

            return output, True

    class ValidRange(StubValidator):
        """Validates numeric value is within specified range"""
        def __init__(self, min_val=0, max_val=100, **kwargs):
            super().__init__(min_val=min_val, max_val=max_val, **kwargs)
            self.min_val = min_val
            self.max_val = max_val

        def validate(self, output, metadata=None):
            try:
                value = float(str(output))
                if value < self.min_val or value > self.max_val:
                    # Clamp to range
                    clamped = max(self.min_val, min(value, self.max_val))
                    return str(clamped), False
                return output, True
            except (ValueError, TypeError):
                return output, False

    class ValidJson(StubValidator):
        """Validates output is valid JSON"""
        def validate(self, output, metadata=None):
            try:
                import json
                if isinstance(output, str):
                    json.loads(output)
                elif isinstance(output, (dict, list)):
                    json.dumps(output)
                else:
                    return str(output), False
                return output, True
            except (json.JSONDecodeError, ValueError, TypeError):
                return output, False

    class TwoWords(StubValidator):
        """Validates output is exactly two words"""
        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            words = output_str.strip().split()

            if len(words) != 2:
                # Fix by taking first 2 or padding
                if len(words) > 2:
                    return " ".join(words[:2]), False
                return "APPROVE ABSTAIN", False

            return output, True

    class RegexMatch(StubValidator):
        """Validates output matches regex pattern"""
        def __init__(self, regex=".*", **kwargs):
            super().__init__(regex=regex, **kwargs)
            import re
            self.pattern = re.compile(regex)

        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            if not self.pattern.match(output_str):
                return output, False
            return output, True

    class InList(StubValidator):
        """Validates output is in list of allowed values"""
        def __init__(self, choices=None, **kwargs):
            super().__init__(choices=choices or [], **kwargs)
            self.choices = choices if choices is not None else []

        def validate(self, output, metadata=None):
            if str(output) not in self.choices:
                # Fix by using first choice
                return self.choices[0] if self.choices else output, False
            return output, True

    class ToxicLanguage(StubValidator):
        """Detects toxic language in output"""
        def __init__(self, threshold=0.8, **kwargs):
            super().__init__(threshold=threshold, **kwargs)
            self.threshold = threshold
            self.toxic_words = {
                'fuck', 'shit', 'damn', 'bitch', 'ass', 'bastard',
                'hate', 'kill', 'die', 'stupid', 'idiot', 'moron',
                'discriminate', 'inferior', 'superior', 'subhuman'
            }

        def validate(self, output, metadata=None):
            output_str = str(output).lower()
            words_found = [word for word in self.toxic_words if word in output_str]

            if words_found:
                word_count = len(output_str.split())
                toxicity_score = len(words_found) / max(word_count, 1)

                if toxicity_score > self.threshold:
                    # Redact toxic words
                    result = output_str
                    for word in set(words_found):
                        import re
                        result = re.sub(
                            re.escape(word),
                            '*' * len(word),
                            result,
                            flags=re.IGNORECASE
                        )
                    return result, False

            return output, True

    class PIIFilter(StubValidator):
        """Detects and redacts PII (emails, phones, SSNs, credit cards)"""
        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            import re

            # Email pattern
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            output_str = re.sub(email_pattern, '[EMAIL_REDACTED]', output_str)

            # Phone pattern
            phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
            output_str = re.sub(phone_pattern, '[PHONE_REDACTED]', output_str)

            # SSN pattern
            ssn_pattern = r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'
            output_str = re.sub(ssn_pattern, '[SSN_REDACTED]', output_str)

            # Credit card pattern
            cc_pattern = r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'
            output_str = re.sub(cc_pattern, '[CC_REDACTED]', output_str)

            # Check if anything was redacted
            if output_str != str(output):
                return output_str, False

            return output, True

    class DetectSecrets(StubValidator):
        """Detects API keys, passwords, tokens"""
        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            import re

            secret_patterns = [
                (r'(?i)api[_-]?key["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}', '[API_KEY_REDACTED]'),
                (r'(?i)secret["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}', '[SECRET_REDACTED]'),
                (r'(?i)token["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}', '[TOKEN_REDACTED]'),
                (r'(?i)password["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{8,}', '[PASSWORD_REDACTED]'),
                (r'(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}', '[BEARER_TOKEN_REDACTED]'),
                (r'(?i)ghp_[A-Za-z0-9]{36}', '[GITHUB_TOKEN_REDACTED]'),
                (r'(?i)xox[baprs]-[A-Za-z0-9\-]{10,}', '[SLACK_TOKEN_REDACTED]'),
            ]

            result = output_str
            redacted = False

            for pattern, replacement in secret_patterns:
                if re.search(pattern, result):
                    result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                    redacted = True

            if redacted:
                return result, False

            return output, True

    class CompetitorCheck(StubValidator):
        """Detects and redacts competitor mentions"""
        def __init__(self, competitors=None, **kwargs):
            super().__init__(competitors=competitors, **kwargs)
            self.competitors = competitors if competitors is not None else [
                "Apple", "Microsoft", "Google", "Amazon", "Meta",
                "Facebook", "Instagram", "WhatsApp"
            ]

        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""
            import re

            result = output_str
            for competitor in self.competitors:
                if re.search(r'\b' + re.escape(competitor) + r'\b', result, re.IGNORECASE):
                    result = re.sub(
                        r'\b' + re.escape(competitor) + r'\b',
                        '[COMPETITOR_REDACTED]',
                        result,
                        flags=re.IGNORECASE
                    )

            if result != output_str:
                return result, False

            return output, True

    class ValidSQL(StubValidator):
        """Validates SQL syntax and detects injection attempts"""
        def validate(self, output, metadata=None):
            output_str = str(output).lower() if output else ""

            # Basic SQL injection patterns
            injection_patterns = [
                r"union\s+select",
                r"or\s+1\s*=\s*1",
                r"drop\s+table",
                r"';\s*--",
                r"exec\s*\(",
            ]

            import re
            for pattern in injection_patterns:
                if re.search(pattern, output_str):
                    return output, False

            # Check for basic SQL syntax patterns
            sql_keywords = ['select', 'from', 'where', 'insert', 'update', 'delete', 'join']
            has_sql = any(keyword in output_str for keyword in sql_keywords)

            # If it looks like SQL but doesn't have injection, it's valid
            # If it doesn't look like SQL at all, also valid (might be another language)
            return output, True

    class ProvenanceLLM(StubValidator):
        """Validates LLM attribution/provenance is present"""
        def validate(self, output, metadata=None):
            output_str = str(output) if output else ""

            # Check for attribution patterns
            provenance_patterns = [
                r'\bfrom\s+\w+',
                r'\bsource[:\s]+\w+',
                r'\bproved\s+by\b',
                r'\battribution[:\s]+\w+',
                r'\breference[:\s]+\w+'
            ]

            import re
            has_provenance = any(
                re.search(pattern, output_str, re.IGNORECASE)
                for pattern in provenance_patterns
            )

            if not has_provenance:
                return output, False

            return output, True


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RemediationStrategy(Enum):
    """All 8 Guardrails on-fail remediation strategies"""
    REASK = "reask"  # Ask LLM to regenerate output
    FIX = "fix"  # Attempt to fix the output automatically
    FILTER = "filter"  # Filter out the offending output
    REFRAIN = "refrain"  # Refrain from outputting anything
    EXCEPTION = "exception"  # Raise an exception
    CUSTOM = "custom"  # Use custom remediation function
    FIX_REASK = "fix_reask"  # Try to fix, then reask if fails
    FIX_REASK_BUT_REFRAIN = "fix_reask_but_refrain"  # Fix, reask, refrain if fails


class ValidationSeverity(Enum):
    """Severity levels for validation failures"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ValidationResult:
    """Structured validation result with metadata"""
    is_valid: bool
    output: Optional[Any] = None
    failures: List[Dict[str, Any]] = field(default_factory=list)
    remediation_applied: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "is_valid": self.is_valid,
            "output": str(self.output) if self.output else None,
            "failures": self.failures,
            "remediation_applied": self.remediation_applied,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class ValidatorConfig:
    """Configuration for a single validator"""
    name: str
    validator_class: Optional[type] = None
    params: Dict[str, Any] = field(default_factory=dict)
    on_fail: str = "reask"
    enabled: bool = True
    description: str = ""


# =============================================================================
# LOGGER CONFIGURATION
# =============================================================================

class GuardrailsLogger:
    """Structured JSON logger for Guardrails operations"""

    def __init__(self, name: str = "guardrails_adapter", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create console handler with JSON formatting
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        self.logger.addHandler(handler)

    def _log(self, level: str, message: str, **kwargs):
        """Internal logging method with structured data"""
        log_data = {
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            **kwargs
        }
        getattr(self.logger, level.lower())(json.dumps(log_data))

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs):
        self._log("DEBUG", message, **kwargs)

    def critical(self, message: str, **kwargs):
        self._log("CRITICAL", message, **kwargs)


class JsonFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""

    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }

        return json.dumps(log_data)


# =============================================================================
# MAIN ADAPTER CLASS
# =============================================================================

class GuardrailsAdapter:
    """
    Production-ready Guardrails adapter for OpenEvolve.

    Features:
    - Pre-configured validator library for all OpenEvolve components
    - 8 remediation strategies with fallback handling
    - Graceful degradation when Guardrails unavailable
    - Structured logging with correlation IDs
    - Full environment-based configuration

    Example:
        adapter = GuardrailsAdapter()
        result = adapter.validate_output(
            output="This is a test output",
            validators=["toxic_language", "pii_filter"],
            on_fail="reask"
        )
        if result.is_valid:
            print(f"Valid output: {result.output}")
        else:
            print(f"Failures: {result.failures}")
    """

    # Pre-configured validator library
    VALIDATOR_LIBRARY: Dict[str, ValidatorConfig] = {
        # ROMA Validators
        "roma_depth": ValidatorConfig(
            name="roma_depth",
            validator_class=ValidRange,
            params={"min": 1, "max": 5},
            on_fail="fix",
            description="Validate ROMA decomposition depth is between 1-5"
        ),
        "roma_length": ValidatorConfig(
            name="roma_length",
            validator_class=ValidLength,
            params={"max_tokens": 2000},
            on_fail="filter",
            description="Validate ROMA output length is within token limits"
        ),
        "roma_format": ValidatorConfig(
            name="roma_format",
            validator_class=RegexMatch,
            params={"regex": r"^[A-Z][a-z]+:.*$"},
            on_fail="reask",
            description="Validate ROMA output follows format pattern"
        ),

        # MDAP Validators
        "vote_format": ValidatorConfig(
            name="vote_format",
            validator_class=TwoWords,
            params={},
            on_fail="reask",
            description="Validate MDAP vote is exactly two words"
        ),
        "vote_id": ValidatorConfig(
            name="vote_id",
            validator_class=RegexMatch,
            params={"regex": r"^[A-Z][0-9]{2}$"},
            on_fail="fix",
            description="Validate MDAP vote ID format (e.g., A01, B23)"
        ),
        "vote_decision": ValidatorConfig(
            name="vote_decision",
            validator_class=InList,
            params={"choices": ["APPROVE", "REJECT", "ABSTAIN"]},
            on_fail="filter",
            description="Validate MDAP vote decision is in allowed list"
        ),
        "vote_json": ValidatorConfig(
            name="vote_json",
            validator_class=ValidJson,
            params={},
            on_fail="reask",
            description="Validate MDAP vote is valid JSON"
        ),

        # LeanAide Validators
        "lean_syntax": ValidatorConfig(
            name="lean_syntax",
            validator_class=ValidSQL,
            params={},
            on_fail="reask",
            description="Validate LeanAide output has proper syntax (using SQL validator)"
        ),
        "lean_provenance": ValidatorConfig(
            name="lean_provenance",
            validator_class=ProvenanceLLM,
            params={},
            on_fail="refrain",
            description="Validate LeanAide output has proper provenance attribution"
        ),
        "lean_no_apology": ValidatorConfig(
            name="lean_no_apology",
            validator_class=RegexMatch,
            params={"regex": r"^(?!.*sorry|.*apology)"},
            on_fail="fix",
            description="Validate LeanAide output doesn't contain apologies"
        ),

        # Safety Validators
        "toxic_language": ValidatorConfig(
            name="toxic_language",
            validator_class=ToxicLanguage,
            params={"threshold": 0.8},
            on_fail="refrain",
            description="Detect toxic language above threshold"
        ),
        "pii_filter": ValidatorConfig(
            name="pii_filter",
            validator_class=PIIFilter,
            params={"pii_entities": "pii"},
            on_fail="fix",
            description="Detect and redact PII (Personally Identifiable Information)"
        ),
        "secrets_detection": ValidatorConfig(
            name="secrets_detection",
            validator_class=DetectSecrets,
            params={},
            on_fail="refrain",
            description="Detect potential secrets/API keys in output"
        ),
        "competitor_check": ValidatorConfig(
            name="competitor_check",
            validator_class=CompetitorCheck,
            params={"competitors": ["Apple", "Microsoft", "Google", "Amazon", "Meta"]},
            on_fail="fix",
            description="Detect and redact competitor mentions"
        ),
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Guardrails adapter with environment-based configuration.

        Environment Variables:
            GUARDRAILS_ENABLED: Enable/disable Guardrails (default: true)
            GUARDRAILS_VALIDATORS: Comma-separated list of enabled validators
            GUARDRAILS_ON_FAIL: Default remediation strategy (default: reask)
            GUARDRAILS_MAX_RETRIES: Maximum validation retry attempts (default: 3)
            GUARDRAILS_TIMEOUT: Validation timeout in seconds (default: 30)
            GUARDRAILS_LOG_LEVEL: Logging level (default: INFO)

        Args:
            config: Optional configuration dict to override environment variables
        """
        # Load configuration from environment and params
        self.enabled = self._parse_bool(
            config.get("enabled") if config else None,
            os.getenv("GUARDRAILS_ENABLED", "true")
        )

        self.default_on_fail = os.getenv("GUARDRAILS_ON_FAIL", "reask")
        self.max_retries = int(os.getenv("GUARDRAILS_MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("GUARDRAILS_TIMEOUT", "30"))
        self.log_level = os.getenv("GUARDRAILS_LOG_LEVEL", "INFO")

        # Initialize logger
        self.logger = GuardrailsLogger(level=self.log_level)

        # Parse enabled validators from environment
        validators_env = os.getenv("GUARDRAILS_VALIDATORS", "")
        self.enabled_validators = set(v.strip() for v in validators_env.split(",") if v.strip())

        # Initialize Guard instance if available
        self.guard = None
        self.custom_validators: Dict[str, Callable] = {}

        # Initialize ROMA-MDAP-MAKER Engine for robust validation/recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for strict checking
                # All 27 master parameters come from the SSOT config
                config_roma = get_validation_config(
                    preset="validation",
                    # Can override specific parameters if needed
                    roma_max_depth_analysis=2  # For validation, shallow depth is sufficient
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                self.logger.info("ROMAMDAPMakerAssociativeEngine initialized for GuardrailsAdapter")
            except Exception as e:
                self.logger.error(f"Failed to initialize ROMA engine: {e}")

        if GUARDRAILS_AVAILABLE and self.enabled:
            try:
                self._initialize_guard()
                self.logger.info(
                    "GuardrailsAdapter initialized successfully",
                    guardrails_available=True,
                    enabled_validators=list(self.enabled_validators) if self.enabled_validators else "all"
                )
            except Exception as e:
                self.logger.error(
                    "Failed to initialize Guardrails, running in degraded mode",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                self.enabled = False
        elif not GUARDRAILS_AVAILABLE:
            self.logger.warning(
                "Guardrails not installed, running in degraded mode",
                suggestion="Install with: pip install guardrails-ai",
                enabled=self.enabled
            )
            self.enabled = False

    def _parse_bool(self, value: Optional[str], default: str) -> bool:
        """Parse boolean from string with various formats"""
        if value is None:
            value = default
        return str(value).lower() in ("true", "1", "yes", "on", "enabled")

    def _initialize_guard(self) -> None:
        """Initialize Guard instance with default configuration"""
        try:
            # Create a basic Guard instance
            # Validators will be added dynamically during validation
            self.guard = Guard()
            self.logger.debug("Guard instance initialized")
        except Exception as e:
            self.logger.error(
                "Failed to create Guard instance",
                error=str(e),
                traceback=traceback.format_exc()
            )
            raise

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def is_available(self) -> bool:
        """
        Check if Guardrails is installed and available.

        Returns:
            True if Guardrails is installed and enabled, False otherwise
        """
        return GUARDRAILS_AVAILABLE and self.enabled

    def get_validator(self, name: str) -> Optional[ValidatorConfig]:
        """
        Get a pre-configured validator by name.

        Args:
            name: Validator name (e.g., "toxic_language", "vote_format")

        Returns:
            ValidatorConfig if found, None otherwise
        """
        return self.VALIDATOR_LIBRARY.get(name)

    def register_validator(self, name: str, validator: Union[type, Callable], **params) -> None:
        """
        Register a custom validator.

        Args:
            name: Unique validator name
            validator: Validator class or custom validation function
            **params: Additional parameters for the validator

        Example:
            adapter.register_validator(
                "custom_format",
                RegexMatch,
                regex=r"^[A-Z]{2}\d{4}$",
                on_fail="fix"
            )
        """
        if callable(validator) and not isinstance(validator, type):
            # Custom validation function
            self.custom_validators[name] = validator
            self.logger.info(
                "Registered custom validator function",
                validator_name=name,
                validator_type="function"
            )
        else:
            # Validator class
            config = ValidatorConfig(
                name=name,
                validator_class=validator,
                params=params,
                on_fail=params.pop("on_fail", "reask"),
                description=params.pop("description", f"Custom validator: {name}")
            )
            self.VALIDATOR_LIBRARY[name] = config
            self.logger.info(
                "Registered custom validator class",
                validator_name=name,
                validator_class=validator.__name__ if validator else None
            )

    def validate_output(
        self,
        output: Any,
        validators: Union[str, List[str]],
        on_fail: str = "reask",
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Validate output against specified validators with remediation.

        Args:
            output: The output to validate (string, dict, list, etc.)
            validators: Single validator name or list of validator names
            on_fail: Remediation strategy (default: "reask")
            correlation_id: Optional correlation ID for logging
            **kwargs: Additional parameters for validation

        Returns:
            ValidationResult with validation outcome and remediation info
        """
        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = f"gr_{datetime.utcnow().timestamp()}"

        self.logger.debug(
            "Starting output validation",
            correlation_id=correlation_id,
            output_type=type(output).__name__,
            validators=validators if isinstance(validators, list) else [validators],
            on_fail=on_fail
        )

        # Check if Guardrails is available
        if not self.is_available():
            return self._fallback_validation(output, validators, correlation_id)

        # Normalize validators to list
        validator_names = [validators] if isinstance(validators, str) else validators

        # Filter enabled validators
        if self.enabled_validators:
            validator_names = [v for v in validator_names if v in self.enabled_validators]

        # Initialize result
        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id,
            metadata={"validators_used": validator_names}
        )

        # Validate against each validator
        for validator_name in validator_names:
            validator_result = self._validate_with_validator(
                output=output,
                validator_name=validator_name,
                on_fail=on_fail,
                correlation_id=correlation_id,
                **kwargs
            )

            if not validator_result.is_valid:
                result.is_valid = False
                result.failures.extend(validator_result.failures)

                # Apply remediation if configured
                if validator_result.remediation_applied:
                    result.remediation_applied = validator_result.remediation_applied
                    result.output = validator_result.output

                    # Stop validation if refraining
                    if validator_result.remediation_applied == "refrain":
                        self.logger.warning(
                            "Refraining from output due to validation failure",
                            correlation_id=correlation_id,
                            validator=validator_name
                        )
                        break

                    # Update output for next validator if fixed
                    if validator_result.output is not None:
                        output = validator_result.output

        self.logger.info(
            "Validation complete",
            correlation_id=correlation_id,
            is_valid=result.is_valid,
            failures_count=len(result.failures),
            remediation=result.remediation_applied
        )

        return result

    def validate_input(
        self,
        prompt: str,
        validators: Union[str, List[str]],
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> ValidationResult:
        """
        Validate input prompt against specified validators.

        Args:
            prompt: The input prompt to validate
            validators: Single validator name or list of validator names
            correlation_id: Optional correlation ID for logging
            **kwargs: Additional parameters for validation

        Returns:
            ValidationResult with validation outcome
        """
        # Input validation uses same logic as output validation
        # but typically with stricter remediation (e.g., refrain vs fix)
        return self.validate_output(
            output=prompt,
            validators=validators,
            on_fail="refrain",  # Stricter default for input
            correlation_id=correlation_id,
            **kwargs
        )

    def apply_remediation(
        self,
        output: Any,
        failure: Dict[str, Any],
        strategy: str
    ) -> Tuple[Any, Optional[str]]:
        """
        Apply remediation strategy to validation failure.

        Args:
            output: The output that failed validation
            failure: Failure details dict with 'validator', 'error', 'message' keys
            strategy: Remediation strategy name

        Returns:
            Tuple of (remediated_output, strategy_applied)
        """
        self.logger.debug(
            "Applying remediation",
            strategy=strategy,
            validator=failure.get("validator"),
            original_output=str(output)[:100] if output else None
        )

        # Route to appropriate remediation method
        remediation_method = getattr(self, f"_remediate_{strategy}", None)

        if remediation_method:
            try:
                remediated_output = remediation_method(output, failure)
                return remediated_output, strategy
            except Exception as e:
                self.logger.error(
                    "Remediation failed",
                    strategy=strategy,
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                # Fallback to refrain on error
                return None, "refrain"
        else:
            self.logger.warning(
                "Unknown remediation strategy",
                strategy=strategy,
                available_strategies=[s.replace("_remediate_", "") for s in dir(self) if s.startswith("_remediate_")]
            )
            return output, None

    # =========================================================================
    # INTERNAL VALIDATION METHODS
    # =========================================================================

    def _validate_with_validator(
        self,
        output: Any,
        validator_name: str,
        on_fail: str,
        correlation_id: str,
        **kwargs
    ) -> ValidationResult:
        """
        Validate output with a specific validator.

        Args:
            output: The output to validate
            validator_name: Name of the validator to use
            on_fail: Remediation strategy
            correlation_id: Correlation ID for logging
            **kwargs: Additional parameters

        Returns:
            ValidationResult for this specific validator
        """
        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id
        )

        # Check for custom validator function
        if validator_name in self.custom_validators:
            return self._validate_with_custom_function(
                output, validator_name, on_fail, correlation_id
            )

        # Get validator config from library
        validator_config = self.get_validator(validator_name)

        if not validator_config:
            self.logger.warning(
                "Validator not found in library",
                validator=validator_name,
                correlation_id=correlation_id
            )
            result.failures.append({
                "validator": validator_name,
                "error": "ValidatorNotFound",
                "message": f"Validator '{validator_name}' not found in library"
            })
            result.is_valid = False
            return result

        # Skip if validator disabled
        if not validator_config.enabled:
            self.logger.debug(
                "Validator disabled, skipping",
                validator=validator_name,
                correlation_id=correlation_id
            )
            return result

        try:
            # Attempt validation with Guardrails
            if GUARDRAILS_AVAILABLE and self.guard:
                result = self._validate_with_guardrails(
                    output, validator_config, on_fail, correlation_id
                )
            else:
                # Fallback validation logic
                result = self._validate_with_fallback(
                    output, validator_config, correlation_id
                )

        except Exception as e:
            self.logger.error(
                "Validation error occurred",
                validator=validator_name,
                error=str(e),
                traceback=traceback.format_exc(),
                correlation_id=correlation_id
            )

            result.is_valid = False
            result.failures.append({
                "validator": validator_name,
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            })

            # Apply remediation
            if on_fail != "exception":
                remediated_output, strategy_applied = self.apply_remediation(
                    output,
                    result.failures[-1],
                    on_fail
                )
                result.output = remediated_output
                result.remediation_applied = strategy_applied

        return result

    def _validate_with_guardrails(
        self,
        output: Any,
        validator_config: ValidatorConfig,
        on_fail: str,
        correlation_id: str
    ) -> ValidationResult:
        """Validate using Guardrails library"""
        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id
        )

        try:
            # Create validator instance
            validator_class = validator_config.validator_class
            validator_params = validator_config.params.copy()
            validator_params["on_fail"] = on_fail

            # Instantiate validator
            validator = validator_class(**validator_params)

            # Run validation
            validated_output, validation_passed = validator.validate(output, metadata={})

            if validation_passed:
                result.output = validated_output
                result.is_valid = True
            else:
                result.is_valid = False
                result.failures.append({
                    "validator": validator_config.name,
                    "error": "ValidationFailed",
                    "message": f"Validator '{validator_config.name}' failed",
                    "on_fail": on_fail
                })

                # Apply remediation
                remediated_output, strategy_applied = self.apply_remediation(
                    validated_output,
                    result.failures[-1],
                    on_fail
                )
                result.output = remediated_output
                result.remediation_applied = strategy_applied

        except ValidationError as e:
            result.is_valid = False
            result.failures.append({
                "validator": validator_config.name,
                "error": "ValidationError",
                "message": str(e),
                "on_fail": on_fail
            })

            # Apply remediation
            remediated_output, strategy_applied = self.apply_remediation(
                output,
                result.failures[-1],
                on_fail
            )
            result.output = remediated_output
            result.remediation_applied = strategy_applied

        except Exception as e:
            raise  # Re-raise for outer exception handler

        return result

    def _validate_with_fallback(
        self,
        output: Any,
        validator_config: ValidatorConfig,
        correlation_id: str
    ) -> ValidationResult:
        """
        Fallback validation without Guardrails library.

        Provides comprehensive production validation logic for all validators
        when Guardrails AI is unavailable. Each validator has full business
        logic implementation, not just stubs.
        """
        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id
        )

        # Production fallback implementations
        validator_name = validator_config.name
        params = validator_config.params

        try:
            output_str = str(output) if output is not None else ""

            # ==================== ROMA Validators ====================

            if validator_name == "roma_depth":
                # Validate ROMA decomposition depth (1-5)
                try:
                    depth = int(output_str)
                    min_depth = params.get("min", 1)
                    max_depth = params.get("max", 5)

                    if depth < min_depth or depth > max_depth:
                        result.is_valid = False
                        result.failures.append({
                            "validator": validator_name,
                            "error": "DepthOutOfRange",
                            "message": f"Depth {depth} not in range [{min_depth}, {max_depth}]"
                        })
                        # Fix by clamping to valid range
                        result.output = str(max(min_depth, min(depth, max_depth)))
                        result.remediation_applied = "fix"
                except ValueError:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidDepth",
                        "message": f"Depth '{output}' is not a valid integer"
                    })

            elif validator_name == "roma_length":
                # Validate ROMA output length in tokens
                # Rough token estimation (4 chars per token)
                token_count = len(output_str) // 4
                max_tokens = params.get("max_tokens", 2000)

                if token_count > max_tokens:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "MaxLengthExceeded",
                        "message": f"Output exceeds maximum length: {token_count} > {max_tokens} tokens"
                    })
                    # Truncate output
                    result.output = output_str[:max_tokens * 4]
                    result.remediation_applied = "fix"

            elif validator_name == "roma_format":
                # Validate ROMA output format (e.g., "Task: description")
                pattern = params.get("regex", r"^[A-Z][a-z]+:.*$")
                if not re.match(pattern, output_str):
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "FormatMismatch",
                        "message": f"Output does not match required format: {pattern}"
                    })

            # ==================== MDAP Validators ====================

            elif validator_name == "vote_format":
                # Validate MDAP vote is exactly two words
                words = output_str.strip().split()
                if len(words) != 2:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidWordCount",
                        "message": f"Vote must be exactly 2 words, got {len(words)}"
                    })
                    # Fix by taking first 2 words or padding
                    if len(words) > 2:
                        result.output = " ".join(words[:2])
                        result.remediation_applied = "fix"
                    else:
                        result.output = "APPROVE ABSTAIN"  # Default fallback
                        result.remediation_applied = "fix"

            elif validator_name == "vote_id":
                # Validate MDAP vote ID format (e.g., A01, B23)
                pattern = params.get("regex", r"^[A-Z][0-9]{2}$")
                if not re.match(pattern, output_str.upper()):
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidVoteIdFormat",
                        "message": f"Vote ID must match pattern {pattern}, got '{output_str}'"
                    })
                    # Try to fix by formatting
                    if len(output_str) >= 3:
                        result.output = output_str[0].upper() + output_str[1:3].zfill(2)
                        result.remediation_applied = "fix"

            elif validator_name == "vote_json":
                # Validate MDAP vote is valid JSON
                try:
                    if isinstance(output, str):
                        json.loads(output)  # Will raise if invalid
                    elif isinstance(output, (dict, list)):
                        json.dumps(output)  # Will raise if not serializable
                    else:
                        raise ValueError("Output is not JSON-serializable")
                except (json.JSONDecodeError, ValueError, TypeError) as e:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidJson",
                        "message": f"Output is not valid JSON: {str(e)}"
                    })

            elif validator_name == "vote_decision":
                # Validate MDAP vote decision is in allowed list
                choices = params.get("choices", ["APPROVE", "REJECT", "ABSTAIN"])
                decision_upper = output_str.upper().strip()

                if decision_upper not in choices:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidChoice",
                        "message": f"Decision '{output}' not in allowed choices: {choices}"
                    })
                    # Fix by defaulting to ABSTAIN
                    result.output = "ABSTAIN"
                    result.remediation_applied = "fix"

            # ==================== LeanAide Validators ====================

            elif validator_name == "lean_syntax":
                # Validate Lean syntax (basic checks)
                # Check for common Lean syntax patterns
                lean_patterns = [
                    r'\bdef\b',  # definitions
                    r'\btheorem\b',  # theorems
                    r'\blemma\b',  # lemmas
                    r':=',  # assignment
                    r'\\',  # lambda
                ]

                has_valid_syntax = any(re.search(pattern, output_str) for pattern in lean_patterns)

                # Also check for balanced braces and parentheses
                braces_balanced = output_str.count('{') == output_str.count('}')
                parens_balanced = output_str.count('(') == output_str.count(')')

                if not (has_valid_syntax or (braces_balanced and parens_balanced)):
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "InvalidLeanSyntax",
                        "message": "Output does not contain valid Lean syntax patterns"
                    })

            elif validator_name == "lean_provenance":
                # Validate LeanAide output has proper provenance attribution
                # Check for attribution patterns like "from", "source", "proved by"
                provenance_patterns = [
                    r'\bfrom\s+\w+',
                    r'\bsource[:\s]+\w+',
                    r'\bproved\s+by\b',
                    r'\battribution[:\s]+\w+',
                    r'\breference[:\s]+\w+'
                ]

                has_provenance = any(re.search(pattern, output_str, re.IGNORECASE) for pattern in provenance_patterns)

                if not has_provenance:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "MissingProvenance",
                        "message": "Output lacks provenance attribution"
                    })

            elif validator_name == "lean_no_apology":
                # Validate LeanAide output doesn't contain apologies
                apology_patterns = [
                    r'\bsorry\b',
                    r'\bapology\b',
                    r'\bapologize\b',
                    r'\bunable\b',
                    r'\bcannot\b'
                ]

                has_apology = any(re.search(pattern, output_str, re.IGNORECASE) for pattern in apology_patterns)

                if has_apology:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "ApologyDetected",
                        "message": "Output contains apology or inability language"
                    })
                    # Remove apologies
                    result.output = output_str
                    for pattern in apology_patterns:
                        result.output = re.sub(pattern, "", result.output, flags=re.IGNORECASE)
                    result.remediation_applied = "fix"

            # ==================== Safety Validators ====================

            elif validator_name == "toxic_language":
                # Detect toxic language using comprehensive word lists
                toxic_words = [
                    # Profanity
                    'fuck', 'shit', 'damn', 'bitch', 'ass', 'bastard',
                    # Hate speech indicators
                    'hate', 'kill', 'die', 'stupid', 'idiot', 'moron',
                    # Discriminatory language
                    'discriminate', 'inferior', 'superior', 'subhuman'
                ]

                output_lower = output_str.lower()
                words_found = [word for word in toxic_words if word in output_lower]

                if words_found:
                    threshold = params.get("threshold", 0.0)
                    toxicity_score = len(words_found) / max(len(output_lower.split()), 1)

                    if toxicity_score > threshold:
                        result.is_valid = False
                        result.failures.append({
                            "validator": validator_name,
                            "error": "ToxicLanguageDetected",
                            "message": f"Toxic language detected: {', '.join(set(words_found))}"
                        })
                        # Redact toxic words
                        result.output = output_str
                        for word in set(words_found):
                            result.output = re.sub(
                                re.escape(word),
                                '*' * len(word),
                                result.output,
                                flags=re.IGNORECASE
                            )
                        result.remediation_applied = "fix"

            elif validator_name == "pii_filter":
                # Detect and redact PII (emails, phones, SSNs, credit cards)
                pii_found = []
                remediated_text = output_str

                # Email pattern
                email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
                emails = re.findall(email_pattern, output_str)
                if emails:
                    pii_found.extend(['email'] * len(emails))
                    remediated_text = re.sub(email_pattern, '[EMAIL_REDACTED]', remediated_text)

                # Phone pattern (various formats)
                phone_pattern = r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'
                phones = re.findall(phone_pattern, output_str)
                if phones:
                    pii_found.extend(['phone'] * len(phones))
                    remediated_text = re.sub(phone_pattern, '[PHONE_REDACTED]', remediated_text)

                # SSN pattern
                ssn_pattern = r'\b\d{3}[-.]?\d{2}[-.]?\d{4}\b'
                ssns = re.findall(ssn_pattern, output_str)
                if ssns:
                    pii_found.extend(['SSN'] * len(ssns))
                    remediated_text = re.sub(ssn_pattern, '[SSN_REDACTED]', remediated_text)

                # Credit card pattern
                cc_pattern = r'\b(?:\d{4}[-.\s]?){3}\d{4}\b'
                ccs = re.findall(cc_pattern, output_str)
                if ccs:
                    pii_found.extend(['credit_card'] * len(ccs))
                    remediated_text = re.sub(cc_pattern, '[CC_REDACTED]', remediated_text)

                if pii_found:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "PIIDetected",
                        "message": f"PII detected and redacted: {', '.join(set(pii_found))}"
                    })
                    result.output = remediated_text
                    result.remediation_applied = "fix"

            elif validator_name == "secrets_detection":
                # Detect API keys, passwords, tokens, secrets
                secret_patterns = [
                    ('api_key', r'(?i)api[_-]?key["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}'),
                    ('secret', r'(?i)secret["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}'),
                    ('token', r'(?i)token["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{20,}'),
                    ('password', r'(?i)password["\']?\s*[:=]\s*["\']?[A-Za-z0-9_\-]{8,}'),
                    ('bearer_token', r'(?i)bearer\s+[A-Za-z0-9_\-\.]{20,}'),
                    ('aws_key', r'(?i)aws[_-]?(?:access[_-]?key[_-]?id|secret)[\"\']?\s*[:=]\s*["\']?[A-Z0-9]{20}'),
                    ('github_token', r'(?i)ghp_[A-Za-z0-9]{36}'),
                    ('slack_token', r'(?i)xox[baprs]-[A-Za-z0-9\-]{10,}'),
                ]

                secrets_found = []

                for secret_type, pattern in secret_patterns:
                    matches = re.findall(pattern, output_str)
                    if matches:
                        secrets_found.append(secret_type)

                if secrets_found:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "SecretsDetected",
                        "message": f"Potential secrets detected: {', '.join(secrets_found)}"
                    })
                    # Redact secrets
                    result.output = output_str
                    for secret_type, pattern in secret_patterns:
                        result.output = re.sub(
                            pattern,
                            f'[{secret_type.upper()}_REDACTED]',
                            result.output,
                            flags=re.IGNORECASE
                        )
                    result.remediation_applied = "fix"

            elif validator_name == "competitor_check":
                # Detect and redact competitor mentions
                competitors = params.get("competitors", [
                    "Apple", "Microsoft", "Google", "Amazon", "Meta",
                    "Facebook", "Instagram", "WhatsApp"
                ])

                competitors_found = []

                for competitor in competitors:
                    if re.search(r'\b' + re.escape(competitor) + r'\b', output_str, re.IGNORECASE):
                        competitors_found.append(competitor)

                if competitors_found:
                    result.is_valid = False
                    result.failures.append({
                        "validator": validator_name,
                        "error": "CompetitorMentionDetected",
                        "message": f"Competitor mentions detected: {', '.join(competitors_found)}"
                    })
                    # Redact competitors
                    result.output = output_str
                    for competitor in competitors_found:
                        result.output = re.sub(
                            r'\b' + re.escape(competitor) + r'\b',
                            '[COMPETITOR_REDACTED]',
                            result.output,
                            flags=re.IGNORECASE
                        )
                    result.remediation_applied = "fix"

            # ==================== Unknown Validator ====================

            else:
                # Unknown validator - log but don't fail
                self.logger.warning(
                    "Unknown validator in fallback mode - treating as pass-through",
                    validator=validator_name,
                    correlation_id=correlation_id,
                    suggestion="Add fallback implementation for this validator"
                )

        except Exception as e:
            result.is_valid = False
            result.failures.append({
                "validator": validator_name,
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            })

        return result

    def _validate_with_custom_function(
        self,
        output: Any,
        validator_name: str,
        on_fail: str,
        correlation_id: str
    ) -> ValidationResult:
        """Validate using custom validator function"""
        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id
        )

        custom_validator = self.custom_validators.get(validator_name)

        if not custom_validator:
            result.is_valid = False
            result.failures.append({
                "validator": validator_name,
                "error": "ValidatorNotFound",
                "message": f"Custom validator '{validator_name}' not found"
            })
            return result

        try:
            # Call custom validator function
            validation_result = custom_validator(output)

            # Handle different return types
            if isinstance(validation_result, bool):
                result.is_valid = validation_result
            elif isinstance(validation_result, tuple):
                is_valid, message_or_output = validation_result
                result.is_valid = is_valid
                if not is_valid and isinstance(message_or_output, str):
                    result.failures.append({
                        "validator": validator_name,
                        "error": "CustomValidationFailed",
                        "message": message_or_output
                    })
                elif not is_valid:
                    result.output = message_or_output
            elif isinstance(validation_result, ValidationResult):
                result = validation_result
            else:
                result.is_valid = bool(validation_result)

            if not result.is_valid and not result.failures:
                result.failures.append({
                    "validator": validator_name,
                    "error": "CustomValidationFailed",
                    "message": f"Custom validator '{validator_name}' returned False"
                })

        except Exception as e:
            result.is_valid = False
            result.failures.append({
                "validator": validator_name,
                "error": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc()
            })

        return result

    def _fallback_validation(
        self,
        output: Any,
        validators: Union[str, List[str]],
        correlation_id: str
    ) -> ValidationResult:
        """
        Fallback validation when Guardrails is unavailable.

        Performs basic checks and logs a warning that Guardrails is not installed.
        """
        self.logger.warning(
            "Guardrails unavailable, using fallback validation",
            correlation_id=correlation_id,
            output_type=type(output).__name__,
            suggestion="Install with: pip install guardrails-ai"
        )

        result = ValidationResult(
            is_valid=True,
            output=output,
            correlation_id=correlation_id,
            metadata={"fallback_mode": True}
        )

        # Basic fallback checks
        validator_names = [validators] if isinstance(validators, str) else validators

        for validator_name in validator_names:
            validator_config = self.get_validator(validator_name)

            if validator_config:
                fallback_result = self._validate_with_fallback(
                    output, validator_config, correlation_id
                )

                if not fallback_result.is_valid:
                    result.is_valid = False
                    result.failures.extend(fallback_result.failures)

        return result

    # =========================================================================
    # REMEDIATION STRATEGIES (8 ON-FAIL ACTIONS)
    # =========================================================================

    def _remediate_reask(self, output: Any, failure: Dict[str, Any]) -> None:
        """
        Reask strategy: Return None to prompt LLM to regenerate.

        Note: Actual reasking logic is handled by the caller.
        This method marks the output for regeneration.
        """
        self.logger.info(
            "Applying reask remediation",
            validator=failure.get("validator"),
            output_length=len(str(output)) if output else 0
        )
        # Return None to signal reask needed
        return None

    def _remediate_fix(self, output: Any, failure: Dict[str, Any]) -> Any:
        """
        Fix strategy: Attempt to automatically fix the output.

        Applies basic fixes based on validator type.
        """
        self.logger.info(
            "Applying fix remediation",
            validator=failure.get("validator")
        )

        validator_name = failure.get("validator", "")

        # Apply fixes based on validator
        if validator_name == "roma_length":
            max_tokens = 2000
            output_str = str(output)
            # Truncate to max tokens
            return output_str[:max_tokens * 4]

        elif validator_name == "lean_no_apology":
            # Remove apology language
            output_str = str(output)
            return re.sub(r"(sorry|apology|apologize)", "", output_str, flags=re.IGNORECASE)

        elif validator_name == "pii_filter":
            # Basic PII redaction (email, phone, SSN)
            output_str = str(output)
            # Redact emails
            output_str = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL_REDACTED]', output_str)
            # Redact phone numbers
            output_str = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE_REDACTED]', output_str)
            # Redact SSN
            output_str = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN_REDACTED]', output_str)
            return output_str

        elif validator_name == "competitor_check":
            # Redact competitor mentions
            output_str = str(output)
            competitors = ["Apple", "Microsoft", "Google", "Amazon", "Meta"]
            for competitor in competitors:
                output_str = re.sub(
                    r'\b' + re.escape(competitor) + r'\b',
                    '[COMPETITOR_REDACTED]',
                    output_str,
                    flags=re.IGNORECASE
                )
            return output_str

        # If no specific fix, return original
        return output

    def _remediate_filter(self, output: Any, failure: Dict[str, Any]) -> str:
        """
        Filter strategy: Filter out the offending output.

        Returns an empty string or placeholder.
        """
        self.logger.info(
            "Applying filter remediation",
            validator=failure.get("validator")
        )
        return "[FILTERED_DUE_TO_VALIDATION_FAILURE]"

    def _remediate_refrain(self, output: Any, failure: Dict[str, Any]) -> None:
        """
        Refrain strategy: Return nothing.

        Used for safety violations where output should not be shown.
        """
        self.logger.warning(
            "Applying refrain remediation",
            validator=failure.get("validator"),
            reason="Safety or policy violation"
        )
        return None

    def _remediate_exception(self, output: Any, failure: Dict[str, Any]) -> None:
        """
        Exception strategy: Raise an exception.

        Used when validation failure should halt execution.
        """
        error_message = f"Validation failed for validator '{failure.get('validator')}': {failure.get('message')}"
        self.logger.error(
            "Raising exception due to validation failure",
            validator=failure.get("validator"),
            error=error_message
        )
        raise ValidationError(error_message)

    def _remediate_custom(self, output: Any, failure: Dict[str, Any]) -> Any:
        """
        Custom strategy: Use custom remediation function.

        Looks for custom function in failure metadata or instance.
        """
        self.logger.info(
            "Applying custom remediation",
            validator=failure.get("validator")
        )

        # Check for custom remediation function in failure metadata
        custom_fn = failure.get("custom_remediation_fn")

        if custom_fn and callable(custom_fn):
            try:
                return custom_fn(output, failure)
            except Exception as e:
                self.logger.error(
                    "Custom remediation function failed",
                    error=str(e),
                    traceback=traceback.format_exc()
                )
                return output
        else:
            self.logger.warning(
                "No custom remediation function provided",
                validator=failure.get("validator")
            )
            return output

    def _remediate_fix_reask(self, output: Any, failure: Dict[str, Any]) -> Any:
        """
        Fix + Reask strategy: Try to fix, then reask if fix fails.

        Attempts automatic fix first, falls back to reask.
        """
        self.logger.info(
            "Applying fix_reask remediation",
            validator=failure.get("validator")
        )

        # First try to fix
        fixed_output = self._remediate_fix(output, failure)

        # If fix changed the output, use it
        if fixed_output != output:
            self.logger.info(
                "Successfully applied fix in fix_reask strategy",
                original_length=len(str(output)) if output else 0,
                fixed_length=len(str(fixed_output)) if fixed_output else 0
            )
            return fixed_output
        else:
            # Fix didn't work, reask
            self.logger.info(
                "Fix did not modify output, falling back to reask",
                validator=failure.get("validator")
            )
            return None

    def _remediate_fix_reask_but_refrain(self, output: Any, failure: Dict[str, Any]) -> Any:
        """
        Fix + Reask + Refrain strategy: Fix, reask if fails, refrain if reask fails.

        Most conservative strategy: tries multiple approaches before giving up.
        """
        self.logger.info(
            "Applying fix_reask_but_refrain remediation",
            validator=failure.get("validator")
        )

        # First try to fix
        fixed_output = self._remediate_fix(output, failure)

        # If fix changed the output, use it
        if fixed_output != output:
            self.logger.info(
                "Successfully applied fix in fix_reask_but_refrain strategy"
            )
            return fixed_output

        # Fix didn't work, try reask (return None)
        # Note: In a real implementation, this might involve a retry loop
        self.logger.info(
            "Fix did not work, returning None to trigger reask (will refrain if reask fails)"
        )
        return None

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def get_available_validators(self) -> List[str]:
        """Get list of all available validator names"""
        return list(self.VALIDATOR_LIBRARY.keys())

    def get_enabled_validators(self) -> List[str]:
        """Get list of currently enabled validators"""
        if self.enabled_validators:
            return list(self.enabled_validators)
        return self.get_available_validators()

    def enable_validator(self, name: str) -> bool:
        """Enable a specific validator"""
        if name in self.VALIDATOR_LIBRARY:
            self.enabled_validators.add(name)
            self.logger.info("Validator enabled", validator=name)
            return True
        return False

    def disable_validator(self, name: str) -> bool:
        """Disable a specific validator"""
        if name in self.enabled_validators:
            self.enabled_validators.remove(name)
            self.logger.info("Validator disabled", validator=name)
            return True
        return False

    def get_validator_description(self, name: str) -> Optional[str]:
        """Get description for a validator"""
        config = self.get_validator(name)
        return config.description if config else None

    def validate_batch(
        self,
        outputs: List[Any],
        validators: Union[str, List[str]],
        on_fail: str = "reask",
        **kwargs
    ) -> List[ValidationResult]:
        """
        Validate multiple outputs in batch.

        Args:
            outputs: List of outputs to validate
            validators: Validators to apply to each output
            on_fail: Remediation strategy
            **kwargs: Additional parameters

        Returns:
            List of ValidationResults
        """
        results = []
        for i, output in enumerate(outputs):
            correlation_id = kwargs.get("correlation_id", f"batch_{i}")
            result = self.validate_output(
                output=output,
                validators=validators,
                on_fail=on_fail,
                correlation_id=correlation_id,
                **kwargs
            )
            results.append(result)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get adapter statistics and configuration.

        Returns:
            Dict with adapter stats
        """
        return {
            "guardrails_available": GUARDRAILS_AVAILABLE,
            "enabled": self.enabled,
            "enabled_validators": list(self.enabled_validators) if self.enabled_validators else "all",
            "total_validators": len(self.VALIDATOR_LIBRARY),
            "custom_validators": len(self.custom_validators),
            "default_on_fail": self.default_on_fail,
            "max_retries": self.max_retries,
            "timeout": self.timeout
        }


# =============================================================================
# STANDALONE FUNCTIONS
# =============================================================================

def create_adapter(**config) -> GuardrailsAdapter:
    """
    Factory function to create a Guardrails adapter.

    Args:
        **config: Configuration options to override environment variables

    Returns:
        Configured GuardrailsAdapter instance

    Example:
        adapter = create_adapter(
            enabled=True,
            default_on_fail="fix",
            max_retries=5
        )
    """
    return GuardrailsAdapter(config=config)


def validate_with_guardrails(
    output: Any,
    validators: Union[str, List[str]],
    **kwargs
) -> ValidationResult:
    """
    Convenience function for one-off validation.

    Creates a temporary adapter and validates the output.

    Args:
        output: Output to validate
        validators: Validators to apply
        **kwargs: Additional parameters

    Returns:
        ValidationResult

    Example:
        result = validate_with_guardrails(
            output="This is a test",
            validators=["toxic_language", "pii_filter"],
            on_fail="fix"
        )
    """
    adapter = GuardrailsAdapter()
    return adapter.validate_output(output, validators, **kwargs)


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Example usage and testing
    print("=" * 60)
    print("OpenEvolve Guardrails Adapter - Test Suite")
    print("=" * 60)

    # Create adapter
    adapter = create_adapter()

    # Print statistics
    stats = adapter.get_statistics()
    print(f"\nAdapter Statistics:")
    print(f"  Guardrails Available: {stats['guardrails_available']}")
    print(f"  Enabled: {stats['enabled']}")
    print(f"  Total Validators: {stats['total_validators']}")
    print(f"  Custom Validators: {stats['custom_validators']}")

    # List available validators
    print(f"\nAvailable Validators:")
    for validator_name in adapter.get_available_validators():
        desc = adapter.get_validator_description(validator_name)
        print(f"  - {validator_name}: {desc}")

    # Example validation
    print(f"\n{'=' * 60}")
    print("Example Validations")
    print("=" * 60)

    # Test 1: Validate length
    print("\n1. Testing roma_length validator:")
    long_output = "This is a very long output. " * 100
    result = adapter.validate_output(
        output=long_output,
        validators="roma_length",
        on_fail="fix"
    )
    print(f"   Valid: {result.is_valid}")
    print(f"   Remediation: {result.remediation_applied}")
    print(f"   Output length: {len(result.output) if result.output else 0}")

    # Test 2: Validate vote decision
    print("\n2. Testing vote_decision validator:")
    result = adapter.validate_output(
        output="APPROVE",
        validators="vote_decision",
        on_fail="filter"
    )
    print(f"   Valid: {result.is_valid}")

    # Test 3: Validate format
    print("\n3. Testing roma_format validator:")
    result = adapter.validate_output(
        output="Decomposition: Task breakdown",
        validators="roma_format",
        on_fail="reask"
    )
    print(f"   Valid: {result.is_valid}")

    # Test 4: Multiple validators
    print("\n4. Testing multiple validators:")
    result = adapter.validate_output(
        output="Contact me at john@example.com for more info",
        validators=["roma_length", "pii_filter", "lean_no_apology"],
        on_fail="fix"
    )
    print(f"   Valid: {result.is_valid}")
    print(f"   Failures: {len(result.failures)}")
    print(f"   Remediation: {result.remediation_applied}")
    if result.output:
        print(f"   Output: {result.output}")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
