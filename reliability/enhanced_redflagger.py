"""
Enhanced Red Flagging System with LMQL and Guardrails Integration.

This module provides a multi-layered red flagging system:
1. Pre-generation: LMQL constraints prevent flagged content
2. Post-generation: Guardrails validators check output
3. Custom rules: Domain-specific validation

Air Gap Principle: This wrapper does NOT modify MDAP core.
"""

import sys
import os
import logging
import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability.lmql_adapter import LMQLAdapter, get_default_adapter, Constraint, ConstraintType
from reliability.guardrails_adapter import GuardrailsAdapter, create_adapter
from reliability.config import get_config

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

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class RedFlagSeverity(str, Enum):
    """Severity levels for red flags"""
    CRITICAL = "critical"  # Security threats, malicious content
    HIGH = "high"         # Violates core requirements
    MEDIUM = "medium"     # Quality issues, edge cases
    LOW = "low"          # Minor issues, warnings


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class RedFlag:
    """Individual red flag with detailed information"""
    category: str
    message: str
    severity: RedFlagSeverity
    validator: Optional[str] = None
    remediation: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        return asdict(self)

    def __str__(self) -> str:
        """String representation"""
        return f"[{self.severity.value.upper()}] {self.category}: {self.message}"


@dataclass
class EnhancedRedFlagRules:
    """Comprehensive rules for enhanced red flagging"""
    # Basic rules (from original RedFlagRules)
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    blocked_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.2
    require_schema_match: bool = True

    # LMQL pre-generation constraints
    enable_lmql_constraints: bool = True
    lmql_max_retries: int = 3

    # Guardrails validators
    guardrails_validators: List[str] = field(default_factory=lambda: [
        "toxic_language",
        "pii_filter",
        "secrets_detection",
        "malicious_patterns"
    ])

    # Custom rules
    required_keywords: List[str] = field(default_factory=list)
    forbidden_keywords: List[str] = field(default_factory=list)
    required_format: Optional[str] = None  # e.g., "json", "markdown"

    # Severity thresholds
    toxicity_threshold: float = 0.8
    pii_detection_strict: bool = True
    confidence_threshold: float = 0.5

    # Advanced
    enable_semantic_validation: bool = False
    semantic_similarity_threshold: float = 0.7
    enable_temporal_consistency: bool = False


@dataclass
class RedFlagStatistics:
    """Statistics tracking for red flagging operations"""
    total_checks: int = 0
    pre_generation_preventions: int = 0  # LMQL prevented bad output
    post_generation_flags: int = 0       # Guardrails caught issues
    remediated_outputs: int = 0
    rejected_outputs: int = 0
    lmql_available: bool = False
    guardrails_available: bool = False
    critical_flags: int = 0
    high_flags: int = 0
    medium_flags: int = 0
    low_flags: int = 0
    start_time: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with computed metrics"""
        from dataclasses import asdict
        stats = asdict(self)

        # Add computed metrics
        if stats["total_checks"] > 0:
            stats["flag_rate"] = (
                stats["critical_flags"] + stats["high_flags"] +
                stats["medium_flags"] + stats["low_flags"]
            ) / stats["total_checks"]

            stats["prevention_rate"] = (
                stats["pre_generation_preventions"] / stats["total_checks"]
            )

            stats["remediation_rate"] = (
                stats["remediated_outputs"] / stats["total_checks"]
            )

        return stats


# =============================================================================
# MAIN ENHANCED RED FLAGGER CLASS
# =============================================================================

class EnhancedRedFlagger:
    """
    Enhanced red flagging with LMQL pre-generation and Guardrails validation.

    This provides a multi-layered approach:
    1. LMQL Constraints: Prevent flagged content during generation
    2. Guardrails Validators: Validate output after generation
    3. Custom Rules: Domain-specific validation

    Example:
        flagger = EnhancedRedFlagger(rules=EnhancedRedFlagRules())

        # Pre-generation: Get LMQL constraints
        constraints = flagger.get_lmql_constraints()

        # Generate with constraints
        result = lmql_adapter.constrained_generation(prompt, constraints)

        # Post-generation: Validate output
        flags = flagger.check_for_red_flags(output, schema)
    """

    def __init__(
        self,
        rules: Optional[EnhancedRedFlagRules] = None,
        lmql_adapter: Optional[LMQLAdapter] = None,
        guardrails_adapter: Optional[GuardrailsAdapter] = None,
        config: Optional[Any] = None
    ):
        """
        Initialize EnhancedRedFlagger.

        Args:
            rules: Red flagging rules configuration
            lmql_adapter: Optional LMQL adapter instance
            guardrails_adapter: Optional Guardrails adapter instance
            config: Optional reliability configuration
        """
        self.rules = rules or EnhancedRedFlagRules()
        self.lmql_adapter = lmql_adapter or get_default_adapter()
        self.guardrails_adapter = guardrails_adapter or create_adapter()
        self.config = config or get_config()

        # Initialize ROMA-MDAP-MAKER Engine for robust validation
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset (optimized for high-reliability checking)
                # All 27 master parameters come from the SSOT config
                config_roma = get_validation_config(
                    preset="validation",
                    # Can override specific parameters if needed
                    roma_max_depth_analysis=2  # For validation, shallow depth is sufficient
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for EnhancedRedFlagger")
            except Exception as e:
                logger.warning(f"Failed to initialize ROMA engine: {e}")

        # Statistics tracking
        self.stats = RedFlagStatistics(
            lmql_available=self.lmql_adapter.is_available(),
            guardrails_available=self.guardrails_adapter.is_available()
        )

        # Validator registry for custom validators
        self._validator_registry = self._build_validator_registry()

        logger.info(
            f"Enhanced RedFlagger initialized: "
            f"LMQL={self.stats.lmql_available}, "
            f"Guardrails={self.stats.guardrails_available}, "
            f"rules={len([f for f in [self.rules.max_tokens, self.rules.max_characters] if f])}"
        )

    def get_lmql_constraints(self, context: Optional[Dict[str, Any]] = None) -> List[Constraint]:
        """
        Get LMQL constraints for pre-generation red flag prevention.

        These constraints prevent flagged content from being generated
        in the first place, providing 70-90% cost reduction.

        Args:
            context: Optional context for constraint generation

        Returns:
            List of LMQL constraints
        """
        if not self.rules.enable_lmql_constraints or not self.lmql_adapter.is_available():
            logger.debug("LMQL constraints disabled or unavailable")
            return []

        constraints = []

        # 1. Length constraint (prevent excessive output)
        if self.rules.max_tokens:
            constraints.append(Constraint(
                name="max_tokens",
                type=ConstraintType.LENGTH,
                max_length=self.rules.max_tokens,
                description=f"Prevent excessive token generation (> {self.rules.max_tokens})"
            ))

        if self.rules.max_characters:
            constraints.append(Constraint(
                name="max_characters",
                type=ConstraintType.LENGTH,
                max_length=self.rules.max_characters,
                description=f"Prevent excessive character length (> {self.rules.max_characters})"
            ))

        # 2. Format constraint (require specific output format)
        if self.rules.required_format:
            if self.rules.required_format == "json":
                constraints.append(Constraint(
                    name="json_format",
                    type=ConstraintType.JSON_SCHEMA,
                    schema={"type": "object"},
                    description="Require valid JSON output"
                ))

        # 3. Confidence constraint (prevent low-confidence outputs)
        if self.rules.confidence_threshold:
            constraints.append(Constraint(
                name="confidence_threshold",
                type=ConstraintType.NUMERICAL,
                min_value=self.rules.confidence_threshold,
                max_value=1.0,
                description=f"Require confidence >= {self.rules.confidence_threshold}"
            ))

        # 4. Forbidden keywords (prevent specific terms)
        if self.rules.forbidden_keywords:
            constraints.append(Constraint(
                name="forbidden_keywords",
                type=ConstraintType.CUSTOM,
                validator=lambda x: all(
                    kw.lower() not in str(x).lower()
                    for kw in self.rules.forbidden_keywords
                ),
                description=f"Forbid: {', '.join(self.rules.forbidden_keywords[:5])}"
            ))

        # 5. Required keywords (ensure specific terms present)
        if self.rules.required_keywords:
            constraints.append(Constraint(
                name="required_keywords",
                type=ConstraintType.CUSTOM,
                validator=lambda x: all(
                    kw.lower() in str(x).lower()
                    for kw in self.rules.required_keywords
                ),
                description=f"Require: {', '.join(self.rules.required_keywords[:5])}"
            ))

        logger.debug(f"Generated {len(constraints)} LMQL constraints for pre-generation prevention")
        return constraints

    def check_for_red_flags(
        self,
        raw_text: str,
        candidate: Any,
        schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, List[RedFlag]]:
        """
        Check for red flags using multi-layered validation.

        Args:
            raw_text: Raw text output from LLM
            candidate: Parsed candidate (dict, list, etc.)
            schema: Optional schema for validation
            context: Optional context for validation

        Returns:
            Tuple of (is_flagged, list of RedFlag objects)
        """
        flags: List[RedFlag] = []
        self.stats.total_checks += 1

        # Layer 1: Basic validation (fast checks)
        basic_flags = self._basic_validation(raw_text, candidate, schema)
        flags.extend(basic_flags)

        # Layer 2: Pattern matching (regex-based)
        pattern_flags = self._pattern_validation(raw_text)
        flags.extend(pattern_flags)

        # Layer 3: Guardrails validation (comprehensive checks)
        if self.guardrails_adapter.is_available():
            guardrails_flags = self._guardrails_validation(raw_text, candidate)
            flags.extend(guardrails_flags)

        # Layer 4: Schema validation (if schema provided)
        if schema is not None and self.rules.require_schema_match:
            schema_flags = self._schema_validation(candidate, schema)
            flags.extend(schema_flags)

        # Layer 5: Custom rules
        custom_flags = self._custom_validation(raw_text, candidate, context)
        flags.extend(custom_flags)

        # Update statistics
        self._update_statistics(flags)

        return len(flags) > 0, flags

    def _basic_validation(
        self,
        raw_text: str,
        candidate: Any,
        schema: Optional[Dict[str, Any]]
    ) -> List[RedFlag]:
        """Basic validation checks"""
        flags = []

        # Empty check
        if raw_text is None or raw_text.strip() == "":
            flags.append(RedFlag(
                category="empty_response",
                message="Response is empty",
                severity=RedFlagSeverity.CRITICAL
            ))
            return flags  # Early return

        # Length checks
        if self.rules.max_characters and len(raw_text) > self.rules.max_characters:
            flags.append(RedFlag(
                category="response_too_long",
                message=f"Response exceeds {self.rules.max_characters} characters",
                severity=RedFlagSeverity.MEDIUM,
                remediation="truncate_output"
            ))

        if self.rules.max_tokens:
            token_count = _approx_token_count(raw_text)
            if token_count > self.rules.max_tokens:
                flags.append(RedFlag(
                    category="token_limit_exceeded",
                    message=f"Response exceeds {self.rules.max_tokens} tokens (approx {token_count})",
                    severity=RedFlagSeverity.HIGH,
                    remediation="truncate_output"
                ))

        # Confidence check
        confidence = candidate_confidence(candidate)
        if confidence < self.rules.min_confidence:
            flags.append(RedFlag(
                category="low_confidence",
                message=f"Confidence {confidence:.2f} below threshold {self.rules.min_confidence}",
                severity=RedFlagSeverity.MEDIUM,
                remediation="regenerate"
            ))

        return flags

    def _pattern_validation(self, raw_text: str) -> List[RedFlag]:
        """Pattern-based validation"""
        flags = []

        # Blocked patterns
        for pattern in self.rules.blocked_patterns:
            try:
                if re.search(pattern, raw_text, re.IGNORECASE):
                    flags.append(RedFlag(
                        category="blocked_pattern",
                        message=f"Blocked pattern detected: {pattern[:50]}",
                        severity=RedFlagSeverity.HIGH,
                        remediation="reject"
                    ))
            except re.error as e:
                logger.warning(f"Invalid regex pattern: {pattern}, error: {e}")

        # Forbidden keywords
        for keyword in self.rules.forbidden_keywords:
            if keyword.lower() in raw_text.lower():
                flags.append(RedFlag(
                    category="forbidden_keyword",
                    message=f"Forbidden keyword detected: {keyword}",
                    severity=RedFlagSeverity.HIGH,
                    remediation="reject"
                ))

        # Required keywords
        if self.rules.required_keywords:
            missing = [
                kw for kw in self.rules.required_keywords
                if kw.lower() not in raw_text.lower()
            ]
            if missing:
                flags.append(RedFlag(
                    category="missing_required_keyword",
                    message=f"Missing required keywords: {', '.join(missing)}",
                    severity=RedFlagSeverity.MEDIUM,
                    remediation="regenerate"
                ))

        return flags

    def _guardrails_validation(
        self,
        raw_text: str,
        candidate: Any
    ) -> List[RedFlag]:
        """Guardrails validation"""
        flags = []

        try:
            # Convert candidate to string
            candidate_str = json.dumps(candidate) if isinstance(candidate, (dict, list)) else raw_text

            # Run Guardrails validators
            validation_result = self.guardrails_adapter.validate_output(
                output=candidate_str,
                validators=self.rules.guardrails_validators,
                on_fail="refrain"
            )

            self.stats.post_generation_flags += 1

            if not validation_result.is_valid:
                # Convert Guardrails failures to RedFlags
                for failure in validation_result.failures:
                    # Determine severity based on validator
                    severity = self._determine_severity_from_validator(failure)

                    flag = RedFlag(
                        category="guardrails_validation",
                        message=failure.get('message', str(failure)),
                        severity=severity,
                        validator=failure.get('validator', 'unknown'),
                        remediation="refrain" if severity == RedFlagSeverity.CRITICAL else "fix",
                        metadata={
                            "validation_result": validation_result.to_dict(),
                            "failure_details": failure
                        }
                    )
                    flags.append(flag)

                # Track remediation
                if validation_result.remediation_applied == "refrain":
                    self.stats.rejected_outputs += 1
                elif validation_result.remediation_applied:
                    self.stats.remediated_outputs += 1

        except Exception as e:
            logger.warning(f"Guardrails validation failed: {e}")
            flags.append(RedFlag(
                category="guardrails_error",
                message=f"Guardrails validation error: {str(e)}",
                severity=RedFlagSeverity.LOW,
                metadata={"error_type": type(e).__name__}
            ))

        return flags

    def _schema_validation(
        self,
        candidate: Any,
        schema: Dict[str, Any]
    ) -> List[RedFlag]:
        """Schema validation"""
        flags = []

        try:
            is_valid, errors = validate_schema(candidate, schema)
            if not is_valid:
                for error in errors:
                    flags.append(RedFlag(
                        category="schema_violation",
                        message=error,
                        severity=RedFlagSeverity.HIGH,
                        remediation="regenerate",
                        metadata={"schema": schema}
                    ))
        except Exception as e:
            logger.warning(f"Schema validation error: {e}")
            flags.append(RedFlag(
                category="schema_validation_error",
                message=f"Schema validation error: {str(e)}",
                severity=RedFlagSeverity.MEDIUM
            ))

        return flags

    def _custom_validation(
        self,
        raw_text: str,
        candidate: Any,
        context: Optional[Dict[str, Any]]
    ) -> List[RedFlag]:
        """Custom validation rules"""
        flags = []

        # Format validation
        if self.rules.required_format:
            if self.rules.required_format == "json":
                try:
                    json.loads(raw_text)
                except json.JSONDecodeError as e:
                    flags.append(RedFlag(
                        category="invalid_format",
                        message=f"Output is not valid {self.rules.required_format.upper()}: {str(e)}",
                        severity=RedFlagSeverity.HIGH,
                        remediation="regenerate"
                    ))

        # Additional custom validation logic can be added here
        # Example: Semantic validation if enabled
        if self.rules.enable_semantic_validation:
            semantic_flags = self._semantic_validation(raw_text, candidate, context)
            flags.extend(semantic_flags)

        return flags

    def _semantic_validation(
        self,
        raw_text: str,
        candidate: Any,
        context: Optional[Dict[str, Any]]
    ) -> List[RedFlag]:
        """
        Semantic validation using embeddings/similarity checks.

        Performs semantic consistency validation between output and
        reference/context when provided.

        Args:
            raw_text: Raw text output from LLM
            candidate: Parsed candidate (dict, list, etc.)
            context: Optional context for validation (may contain reference_text,
                    task_goal, previous_outputs, etc.)

        Returns:
            List of semantic validation red flags
        """
        flags = []

        # Skip if no context provided
        if not context:
            return flags

        # Check 1: Semantic consistency with reference (if available)
        if "reference_text" in context:
            reference = context["reference_text"]

            # Basic semantic check: keyword overlap
            ref_words = set(reference.lower().split())
            output_words = set(raw_text.lower().split())

            # Calculate overlap
            if ref_words:
                overlap = len(ref_words & output_words) / len(ref_words)

                # Require at least 30% semantic overlap
                if overlap < 0.3:
                    flags.append(RedFlag(
                        category="semantic_drift",
                        message=f"Output has low semantic overlap ({overlap:.1%}) with reference",
                        severity=RedFlagSeverity.MEDIUM,
                        validator="semantic_validation",
                        remediation="regenerate_with_reference"
                    ))

        # Check 2: Consistency with task goal (if provided)
        if "task_goal" in context:
            goal = context["task_goal"]
            goal_words = set(goal.lower().split())
            output_words = set(raw_text.lower().split())

            # Check if output addresses the goal
            if goal_words:
                relevance = len(goal_words & output_words) / len(goal_words)

                if relevance < 0.5:
                    flags.append(RedFlag(
                        category="goal_irrelevance",
                        message=f"Output has low relevance ({relevance:.1%}) to task goal",
                        severity=RedFlagSeverity.HIGH,
                        validator="semantic_validation",
                        remediation="regenerate_with_goal_focus"
                    ))

        # Check 3: Temporal consistency (if enabled)
        if self.rules.enable_temporal_consistency and "previous_outputs" in context:
            previous = context["previous_outputs"]

            if previous:
                # Check for contradictions with previous outputs
                # This is a basic implementation - for full semantic validation
                # you would use embeddings and cosine similarity
                prev_text = " ".join(previous[-3:])  # Last 3 outputs

                # Simple heuristic: check if current output contradicts previous
                contradict_patterns = [
                    ("however", "but"), ("although", "despite"),
                    ("on the contrary", "conversely")
                ]

                for pattern in contradict_patterns:
                    if pattern in prev_text.lower() and pattern in raw_text.lower():
                        flags.append(RedFlag(
                            category="temporal_inconsistency",
                            message=f"Contradiction detected with previous output: {pattern}",
                            severity=RedFlagSeverity.MEDIUM,
                            validator="semantic_validation",
                            remediation="check_consistency"
                        ))

        return flags

    def _determine_severity_from_validator(self, failure: Dict[str, Any]) -> RedFlagSeverity:
        """Determine red flag severity from validator type"""
        validator = failure.get('validator', '')
        message = failure.get('message', '').lower()

        # Critical validators
        if 'secrets_detection' in validator or 'malicious_patterns' in validator:
            return RedFlagSeverity.CRITICAL

        # High severity
        if 'toxic_language' in validator or 'injection' in message:
            return RedFlagSeverity.HIGH

        # Medium severity
        if 'pii_filter' in validator or 'schema' in validator:
            return RedFlagSeverity.MEDIUM

        # Low severity
        return RedFlagSeverity.LOW

    def _update_statistics(self, flags: List[RedFlag]):
        """Update statistics based on flags"""
        for flag in flags:
            if flag.severity == RedFlagSeverity.CRITICAL:
                self.stats.critical_flags += 1
            elif flag.severity == RedFlagSeverity.HIGH:
                self.stats.high_flags += 1
            elif flag.severity == RedFlagSeverity.MEDIUM:
                self.stats.medium_flags += 1
            else:
                self.stats.low_flags += 1

    def _build_validator_registry(self) -> Dict[str, callable]:
        """
        Build registry of custom validators.

        Returns:
            Dictionary mapping validator names to validation functions
        """
        registry = {}

        # Example: Add custom validator for specific use cases
        def code_injection_validator(text: str) -> Tuple[bool, Optional[str]]:
            """Check for potential code injection patterns"""
            injection_patterns = [
                r'__import__',
                r'eval\s*\(',
                r'exec\s*\(',
                r'os\.system',
                r'subprocess'
            ]
            for pattern in injection_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return False, f"Potential code injection pattern: {pattern}"
            return True, None

        registry["code_injection"] = code_injection_validator

        return registry

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get red flagging statistics.

        Returns:
            Dictionary with statistics and computed metrics
        """
        return self.stats.to_dict()

    def reset_statistics(self):
        """Reset statistics tracking"""
        old_lmql_available = self.stats.lmql_available
        old_guardrails_available = self.stats.guardrails_available

        self.stats = RedFlagStatistics(
            lmql_available=old_lmql_available,
            guardrails_available=old_guardrails_available
        )

        logger.info("Statistics reset")

    def register_custom_validator(self, name: str, validator: callable):
        """
        Register a custom validation function.

        Args:
            name: Unique validator name
            validator: Validation function that returns (bool, Optional[str])
                     where bool is validity and str is error message if invalid

        Example:
            def my_validator(text: str) -> Tuple[bool, Optional[str]]:
                if "bad" in text.lower():
                    return False, "Contains bad word"
                return True, None

            flagger.register_custom_validator("my_validator", my_validator)
        """
        self._validator_registry[name] = validator
        logger.info(f"Registered custom validator: {name}")

    def validate_with_custom_validator(
        self,
        text: str,
        validator_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate text using a registered custom validator.

        Args:
            text: Text to validate
            validator_name: Name of registered validator

        Returns:
            Tuple of (is_valid, error_message)
        """
        validator = self._validator_registry.get(validator_name)

        if not validator:
            return False, f"Validator '{validator_name}' not found in registry"

        try:
            return validator(text)
        except Exception as e:
            logger.error(f"Custom validator '{validator_name}' failed: {e}")
            return False, f"Validator error: {str(e)}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _approx_token_count(text: str) -> int:
    """
    Approximate token count (rough estimate: 4 chars per token).

    Args:
        text: Input text

    Returns:
        Approximate token count
    """
    return len(text) // 4


def validate_schema(candidate: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate candidate against schema.

    Args:
        candidate: Candidate to validate
        schema: Schema to validate against

    Returns:
        Tuple of (is_valid, list of error messages)
    """
    # Import here to avoid circular dependency
    try:
        from mdap_engine import validate_schema as mdap_validate_schema
        return mdap_validate_schema(candidate, schema)
    except ImportError:
        # Basic validation
        errors = []
        schema_type = schema.get("type")

        if schema_type == "object":
            if not isinstance(candidate, dict):
                return False, ["Candidate is not an object"]

            required = schema.get("required", [])
            for key in required:
                if key not in candidate:
                    errors.append(f"Missing required key: {key}")

        elif schema_type == "array":
            if not isinstance(candidate, list):
                errors.append("Candidate is not an array")

        elif schema_type == "string":
            if not isinstance(candidate, str):
                errors.append("Candidate is not a string")

        elif schema_type == "number":
            if not isinstance(candidate, (int, float)):
                errors.append("Candidate is not a number")

        elif schema_type == "boolean":
            if not isinstance(candidate, bool):
                errors.append("Candidate is not a boolean")

        return len(errors) == 0, errors


def candidate_confidence(candidate: Any, default: float = 0.5) -> float:
    """
    Extract confidence from candidate.

    Args:
        candidate: Candidate object
        default: Default confidence if not found

    Returns:
        Confidence score (0.0 - 1.0)
    """
    if isinstance(candidate, dict):
        return float(candidate.get("confidence", default))
    return default


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1)
def create_enhanced_redflagger(
    rules: Optional[EnhancedRedFlagRules] = None,
    config: Optional[Any] = None
) -> EnhancedRedFlagger:
    """
    Create enhanced red flagger with default configuration.

    Args:
        rules: Optional red flagging rules
        config: Optional reliability configuration

    Returns:
        Configured EnhancedRedFlagger instance
    """
    return EnhancedRedFlagger(rules=rules, config=config)


def quick_validate(
    text: str,
    rules: Optional[EnhancedRedFlagRules] = None
) -> Tuple[bool, List[RedFlag]]:
    """
    Quick validation without initializing flagger.

    Args:
        text: Text to validate
        rules: Optional red flagging rules

    Returns:
        Tuple of (is_valid, list of RedFlag objects)
    """
    flagger = create_enhanced_redflagger(rules=rules)
    return flagger.check_for_red_flags(text, text)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "EnhancedRedFlagger",
    "EnhancedRedFlagRules",
    "RedFlag",
    "RedFlagSeverity",
    "RedFlagStatistics",
    "create_enhanced_redflagger",
    "quick_validate",
    "validate_schema",
    "candidate_confidence",
]


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Example usage
    print("=" * 60)
    print("Enhanced Red Flagging System - Test Suite")
    print("=" * 60)

    # Create flagger
    flagger = EnhancedRedFlagger()

    # Test 1: Basic validation
    print("\n1. Testing basic validation:")
    test_text = "This is a test response"
    is_flagged, flags = flagger.check_for_red_flags(test_text, test_text)
    print(f"   Flagged: {is_flagged}")
    print(f"   Flags: {len(flags)}")

    # Test 2: Length validation
    print("\n2. Testing length validation:")
    long_text = "This is a very long response. " * 100
    is_flagged, flags = flagger.check_for_red_flags(long_text, long_text)
    print(f"   Flagged: {is_flagged}")
    for flag in flags:
        print(f"   - {flag}")

    # Test 3: LMQL constraints
    print("\n3. Testing LMQL constraints:")
    constraints = flagger.get_lmql_constraints()
    print(f"   Constraints generated: {len(constraints)}")
    for constraint in constraints:
        print(f"   - {constraint.name}: {constraint.description}")

    # Test 4: Statistics
    print("\n4. Testing statistics:")
    stats = flagger.get_statistics()
    print(f"   Total checks: {stats['total_checks']}")
    print(f"   Flag rate: {stats.get('flag_rate', 'N/A')}")

    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60)
