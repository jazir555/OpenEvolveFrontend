"""
LMQL Adapter for OpenEvolve
============================

Production-ready adapter for Language Model Query Language (LMQL) integration.
Provides constrained generation capabilities with graceful degradation.

Author: OpenEvolve Reliability Layer
Version: 1.0.0
"""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field
from functools import lru_cache
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

# Configure structured JSON logging
logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class LMQLConfig:
    """Configuration for LMQL adapter from environment variables."""

    enabled: bool = True
    model: str = "openai/gpt-4"
    decoding: str = "argmax"  # Options: argmax, sample, beam
    cache_enabled: bool = True
    timeout: int = 30
    max_retries: int = 3
    temperature: float = 0.7
    max_tokens: int = 2048

    @classmethod
    def from_env(cls) -> 'LMQLConfig':
        """Load configuration from environment variables."""
        return cls(
            enabled=os.getenv("LMQL_ENABLED", "true").lower() == "true",
            model=os.getenv("LMQL_MODEL", "openai/gpt-4"),
            decoding=os.getenv("LMQL_DECODING", "argmax"),
            cache_enabled=os.getenv("LMQL_CACHE", "true").lower() == "true",
            timeout=int(os.getenv("LMQL_TIMEOUT", "30")),
            max_retries=int(os.getenv("LMQL_MAX_RETRIES", "3")),
            temperature=float(os.getenv("LMQL_TEMPERATURE", "0.7")),
            max_tokens=int(os.getenv("LMQL_MAX_TOKENS", "2048"))
        )

    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []

        if self.decoding not in ["argmax", "sample", "beam"]:
            errors.append(f"Invalid decoding method: {self.decoding}")

        if self.timeout < 1 or self.timeout > 300:
            errors.append(f"Timeout must be 1-300 seconds, got: {self.timeout}")

        if self.max_retries < 0 or self.max_retries > 10:
            errors.append(f"Max retries must be 0-10, got: {self.max_retries}")

        if self.temperature < 0.0 or self.temperature > 2.0:
            errors.append(f"Temperature must be 0.0-2.0, got: {self.temperature}")

        if self.max_tokens < 1 or self.max_tokens > 128000:
            errors.append(f"Max tokens must be 1-128000, got: {self.max_tokens}")

        if errors:
            logger.error({
                "event": "config_validation_failed",
                "errors": errors,
                "config": self.__dict__
            })
            return False

        return True


# =============================================================================
# CONSTRAINT DEFINITIONS
# =============================================================================

class ConstraintType(Enum):
    """Types of constraints supported."""
    REGEX = "regex"
    LENGTH = "length"
    JSON_SCHEMA = "json_schema"
    CUSTOM = "custom"
    FROM_LIST = "from_list"
    NUMERICAL = "numerical"


@dataclass
class Constraint:
    """Definition of a generation constraint."""
    name: str
    type: ConstraintType
    pattern: Optional[str] = None  # For regex
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    schema: Optional[Dict] = None  # For JSON schema
    validator: Optional[Callable[[str], bool]] = None  # Custom validator
    allowed_values: Optional[List[Any]] = None  # For from_list
    min_value: Optional[float] = None  # For numerical
    max_value: Optional[float] = None  # For numerical
    description: str = ""

    def to_lmql_syntax(self) -> str:
        """Convert constraint to LMQL syntax."""
        if self.type == ConstraintType.REGEX and self.pattern:
            return f'STR matches r"{self.pattern}"'

        elif self.type == ConstraintType.LENGTH:
            parts = []
            if self.min_length:
                parts.append(f"len(STR) >= {self.min_length}")
            if self.max_length:
                parts.append(f"len(STR) <= {self.max_length}")
            return " and ".join(parts)

        elif self.type == ConstraintType.FROM_LIST and self.allowed_values:
            values_str = ", ".join(repr(v) for v in self.allowed_values)
            return f"STR in [{values_str}]"

        elif self.type == ConstraintType.NUMERICAL:
            parts = []
            if self.min_value is not None:
                parts.append(f"float(STR) >= {self.min_value}")
            if self.max_value is not None:
                parts.append(f"float(STR) <= {self.max_value}")
            return " and ".join(parts)

        elif self.type == ConstraintType.CUSTOM and self.validator:
            # Custom constraints are handled in post-processing
            return f"/* Custom validation: {self.description} */"

        return ""  # JSON schema handled separately


# =============================================================================
# RESULT TYPES
# =============================================================================

@dataclass
class GenerationResult:
    """Result from constrained generation."""
    success: bool
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    error_type: Optional[str] = None
    fallback_used: bool = False
    constraint_violations: List[str] = field(default_factory=list)
    correlation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "text": self.text,
            "metadata": self.metadata,
            "error": self.error,
            "error_type": self.error_type,
            "fallback_used": self.fallback_used,
            "constraint_violations": self.constraint_violations,
            "correlation_id": self.correlation_id
        }


# =============================================================================
# MAIN ADAPTER CLASS
# =============================================================================

class LMQLAdapter:
    """
    Production adapter for LMQL with graceful degradation.

    This adapter provides constrained generation capabilities using LMQL
    with automatic fallback to standard generation when LMQL is unavailable.
    """

    def __init__(self, config: Optional[LMQLConfig] = None):
        """
        Initialize the LMQL adapter.

        Args:
            config: Optional configuration. Loads from env if not provided.
        """
        self.config = config or LMQLConfig.from_env()
        self.config.validate()

        # Try importing LMQL
        self.lmql_available = self._check_lmql_availability()
        self.lmql_module = None

        if self.lmql_available:
            try:
                import lmql
                self.lmql_module = lmql
                logger.info({
                    "event": "lmql_loaded",
                    "version": getattr(lmql, "__version__", "unknown")
                })
            except ImportError as e:
                self.lmql_available = False
                logger.warning({
                    "event": "lmql_import_failed",
                    "error": str(e),
                    "message": "LMQL not available, will use fallback generation"
                })

        # Constraint template library
        self._constraint_templates = self._build_constraint_templates()

        # Cache for repeated patterns
        self._cache: Dict[str, Any] = {}

        # Initialize ROMA-MDAP-MAKER Engine for robust constrained generation/recomposition
        self.roma_engine = None
        if ROMA_MDAP_MAKER_AVAILABLE:
            try:
                # Use SSOT validation preset for high-reliability checking
                # All 27 master parameters come from the SSOT config
                config_roma = get_validation_config(
                    preset="validation",
                    # Can override specific parameters if needed
                    roma_max_depth_analysis=2  # For validation, shallow depth is sufficient
                )
                self.roma_engine = ROMAMDAPMakerAssociativeEngine(config_roma)
                logger.info("ROMAMDAPMakerAssociativeEngine initialized for LMQLAdapter")
            except Exception as e:
                logger.error(f"Failed to initialize ROMA engine: {e}")

        logger.info({
            "event": "adapter_initialized",
            "lmql_available": self.lmql_available,
            "config": self.config.__dict__
        })

    def _check_lmql_availability(self) -> bool:
        """
        Check if LMQL is installed and available.

        Returns:
            True if LMQL is available, False otherwise.
        """
        try:
            import lmql
            return True
        except ImportError:
            return False

    def is_available(self) -> bool:
        """
        Check if LMQL is available for use.

        Returns:
            True if LMQL is installed and enabled, False otherwise.
        """
        return self.lmql_available and self.config.enabled

    # =========================================================================
    # CONSTRAINT TEMPLATE LIBRARY
    # =========================================================================

    def _build_constraint_templates(self) -> Dict[str, Callable[..., Constraint]]:
        """
        Build the library of pre-built constraint templates.

        Returns:
            Dictionary of constraint template functions.
        """
        return {
            "roma_depth": self.roma_depth_constraint,
            "vote_id": self.vote_id_constraint,
            "lean_theorem": self.lean_theorem_constraint,
            "json_output": self.json_output_constraint,
            "length": self.length_constraint,
            "mdap_score": self.mdap_score_constraint,
            "roma_step": self.roma_step_constraint,
            "lean_tactic": self.lean_tactic_constraint,
        }

    @staticmethod
    def roma_depth_constraint(max_depth: int = 5) -> Constraint:
        """
        Constraint for ROMA decomposition depth.

        Ensures depth values are between 1 and max_depth.

        Args:
            max_depth: Maximum allowed depth level.

        Returns:
            Numerical constraint for depth values.
        """
        return Constraint(
            name="roma_depth",
            type=ConstraintType.NUMERICAL,
            min_value=1,
            max_value=max_depth,
            min_length=1,
            max_length=2,
            description=f"ROMA decomposition depth (1-{max_depth})"
        )

    @staticmethod
    def vote_id_constraint() -> Constraint:
        """
        Constraint for MDAP vote ID format.

        Format: Single uppercase letter followed by two digits (e.g., "A01", "B99").

        Returns:
            Regex constraint for vote IDs.
        """
        return Constraint(
            name="vote_id",
            type=ConstraintType.REGEX,
            pattern=r"^[A-Z][0-9]{2}$",
            description="MDAP vote ID (e.g., A01, B99)"
        )

    @staticmethod
    def lean_theorem_constraint() -> Constraint:
        """
        Constraint for Lean theorem format.

        Format: theorem_name : proposition
        Example: "add_comm : ∀ a b, a + b = b + a"

        Returns:
            Regex constraint for Lean theorems.
        """
        return Constraint(
            name="lean_theorem",
            type=ConstraintType.REGEX,
            pattern=r"^[a-z_][a-z0-9_']*\s*:\s*.*$",
            description="Lean theorem declaration (name : proposition)"
        )

    @staticmethod
    def json_output_constraint() -> Constraint:
        """
        Constraint for valid JSON output.

        Ensures output is parseable as valid JSON.

        Returns:
            Custom constraint for JSON validation.
        """
        def json_validator(text: str) -> bool:
            try:
                json.loads(text)
                return True
            except (json.JSONDecodeError, ValueError):
                return False

        return Constraint(
            name="json_output",
            type=ConstraintType.CUSTOM,
            validator=json_validator,
            description="Valid JSON output"
        )

    @staticmethod
    def length_constraint(max_tokens: int = 2048, max_chars: Optional[int] = None) -> Constraint:
        """
        Constraint for output length.

        Args:
            max_tokens: Maximum number of tokens.
            max_chars: Maximum number of characters (optional).

        Returns:
            Length constraint.
        """
        return Constraint(
            name="length",
            type=ConstraintType.LENGTH,
            max_length=max_chars,
            description=f"Length constraint (max {max_tokens} tokens)"
        )

    @staticmethod
    def mdap_score_constraint() -> Constraint:
        """
        Constraint for MDAP confidence scores.

        Scores must be between 0.0 and 1.0.

        Returns:
            Numerical constraint for confidence scores.
        """
        return Constraint(
            name="mdap_score",
            type=ConstraintType.NUMERICAL,
            min_value=0.0,
            max_value=1.0,
            description="MDAP confidence score (0.0-1.0)"
        )

    @staticmethod
    def roma_step_constraint() -> Constraint:
        """
        Constraint for ROMA step format.

        Format: STEP_1, STEP_2, etc. or step_1, step_2

        Returns:
            Regex constraint for step identifiers.
        """
        return Constraint(
            name="roma_step",
            type=ConstraintType.REGEX,
            pattern=r"^[A-Z][A-Z_]*_\d+$",
            description="ROMA step identifier (e.g., STEP_1, ACTION_2)"
        )

    @staticmethod
    def lean_tactic_constraint() -> Constraint:
        """
        Constraint for Lean tactic format.

        Format: tactic_name arg1 arg2 ...
        Example: "rw [add_comm]", "apply H1", "simp"

        Returns:
            Regex constraint for Lean tactics.
        """
        return Constraint(
            name="lean_tactic",
            type=ConstraintType.REGEX,
            pattern=r"^[a-z_][a-z0-9_']*(\s+\[[^\]]+\])?(\s+[^\s]+)*$",
            description="Lean tactic invocation"
        )

    def get_constraint_template(self, name: str, **kwargs) -> Optional[Constraint]:
        """
        Get a pre-built constraint template by name.

        Args:
            name: Name of the constraint template.
            **kwargs: Arguments to pass to the constraint constructor.

        Returns:
            Constraint object or None if not found.
        """
        template_func = self._constraint_templates.get(name)
        if template_func:
            try:
                return template_func(**kwargs)
            except Exception as e:
                logger.error({
                    "event": "constraint_template_failed",
                    "constraint": name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                })
                return None

        logger.warning({
            "event": "constraint_not_found",
            "constraint": name,
            "available": list(self._constraint_templates.keys())
        })
        return None

    # =========================================================================
    # VALIDATION
    # =========================================================================

    def validate_constraints(self, constraints: List[Constraint]) -> Dict[str, Any]:
        """
        Validate a list of constraint definitions.

        Args:
            constraints: List of constraints to validate.

        Returns:
            Validation result with valid flag and any errors.
        """
        errors = []
        warnings = []

        for constraint in constraints:
            # Check regex pattern validity
            if constraint.type == ConstraintType.REGEX and constraint.pattern:
                try:
                    re.compile(constraint.pattern)
                except re.error as e:
                    errors.append(f"Invalid regex in {constraint.name}: {e}")

            # Check numerical bounds
            if constraint.type == ConstraintType.NUMERICAL:
                if constraint.min_value is not None and constraint.max_value is not None:
                    if constraint.min_value > constraint.max_value:
                        errors.append(
                            f"Invalid bounds in {constraint.name}: "
                            f"min ({constraint.min_value}) > max ({constraint.max_value})"
                        )

            # Check length bounds
            if constraint.type == ConstraintType.LENGTH:
                if constraint.min_length and constraint.max_length:
                    if constraint.min_length > constraint.max_length:
                        errors.append(
                            f"Invalid length bounds in {constraint.name}: "
                            f"min ({constraint.min_length}) > max ({constraint.max_length})"
                        )

            # Check JSON schema validity
            if constraint.type == ConstraintType.JSON_SCHEMA and constraint.schema:
                try:
                    json.dumps(constraint.schema)
                except (TypeError, ValueError) as e:
                    errors.append(f"Invalid JSON schema in {constraint.name}: {e}")

            # Warning for custom validators
            if constraint.type == ConstraintType.CUSTOM and not constraint.validator:
                warnings.append(f"{constraint.name} has no custom validator function")

        result = {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "constraint_count": len(constraints)
        }

        if errors:
            logger.error({
                "event": "constraint_validation_failed",
                "result": result
            })

        return result

    # =========================================================================
    # GENERATION METHODS
    # =========================================================================

    def constrained_generation(
        self,
        prompt: str,
        constraints: List[Constraint],
        decoding: Optional[str] = None,
        **kwargs
    ) -> GenerationResult:
        """
        Generate text with constraints using LMQL.

        Args:
            prompt: Input prompt for generation.
            constraints: List of constraints to apply.
            decoding: Decoding strategy (argmax, sample, beam).
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with generated text or error.
        """
        correlation_id = f"gen_{datetime.utcnow().timestamp()}"

        # Validate constraints first
        validation = self.validate_constraints(constraints)
        if not validation["valid"]:
            return GenerationResult(
                success=False,
                error="Constraint validation failed",
                error_type="ValidationError",
                correlation_id=correlation_id
            )

        # Use LMQL if available
        if self.is_available():
            try:
                return self._lmql_generate(
                    prompt=prompt,
                    constraints=constraints,
                    decoding=decoding or self.config.decoding,
                    correlation_id=correlation_id,
                    **kwargs
                )
            except Exception as e:
                logger.error({
                    "event": "lmql_generation_failed",
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "correlation_id": correlation_id
                })
                # Fall through to fallback generation

        # Fallback to standard generation
        return self._fallback_generation(
            prompt=prompt,
            constraints=constraints,
            correlation_id=correlation_id,
            **kwargs
        )

    def _lmql_generate(
        self,
        prompt: str,
        constraints: List[Constraint],
        decoding: str,
        correlation_id: str,
        **kwargs
    ) -> GenerationResult:
        """
        Generate using LMQL with constraints.

        Args:
            prompt: Input prompt.
            constraints: List of constraints.
            decoding: Decoding strategy.
            correlation_id: Request correlation ID.
            **kwargs: Additional parameters.

        Returns:
            GenerationResult.
        """
        try:
            # Build LMQL query string
            query = self._build_lmql_query(prompt, constraints)

            logger.info({
                "event": "lmql_generation_start",
                "correlation_id": correlation_id,
                "decoding": decoding,
                "constraint_count": len(constraints)
            })

            # Execute LMQL query
            result = self.lmql_module.query(
                query,
                model=self.config.model,
                decoder=decoding,
                timeout=self.config.timeout,
                **kwargs
            )

            # Extract generated text
            if hasattr(result, 'variables'):
                text = result.variables.get('ANSWER', '')
            elif isinstance(result, dict):
                text = result.get('ANSWER', '')
            else:
                text = str(result)

            # Post-process validation
            violations = self._validate_output(text, constraints)

            success = len(violations) == 0

            logger.info({
                "event": "lmql_generation_complete",
                "correlation_id": correlation_id,
                "success": success,
                "violations": violations
            })

            return GenerationResult(
                success=success,
                text=text,
                metadata={
                    "method": "lmql",
                    "decoding": decoding,
                    "model": self.config.model
                },
                constraint_violations=violations,
                correlation_id=correlation_id
            )

        except Exception as e:
            logger.error({
                "event": "lmql_execution_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })

            # Re-raise to trigger fallback
            raise

    def _build_lmql_query(self, prompt: str, constraints: List[Constraint]) -> str:
        """
        Build an LMQL query string from prompt and constraints.

        Args:
            prompt: Input prompt.
            constraints: List of constraints.

        Returns:
            LMQL query string.
        """
        # Start with the prompt
        query_parts = [f'"""{prompt}"""']

        # Build constraint clause
        constraint_clauses = []
        for constraint in constraints:
            lmql_syntax = constraint.to_lmql_syntax()
            if lmql_syntax and not lmql_syntax.startswith("/*"):
                constraint_clauses.append(lmql_syntax)

        # Combine constraints
        if constraint_clauses:
            where_clause = " and ".join(constraint_clauses)
            query_parts.append(f"WHERE {where_clause}")

        # Add answer variable
        query_parts.append("ANSWER")

        # Join into final query
        return "\n".join(query_parts)

    def _fallback_generation(
        self,
        prompt: str,
        constraints: List[Constraint],
        correlation_id: str,
        **kwargs
    ) -> GenerationResult:
        """
        Fallback generation without LMQL.

        Uses standard generation with post-processing validation.

        Args:
            prompt: Input prompt.
            constraints: List of constraints.
            correlation_id: Request correlation ID.
            **kwargs: Additional parameters.

        Returns:
            GenerationResult (possibly with constraint violations).
        """
        logger.info({
            "event": "fallback_generation_start",
            "correlation_id": correlation_id,
            "reason": "lmql_unavailable"
        })

        try:
            # Import OpenAI client
            try:
                from openai import OpenAI
                client = OpenAI()
            except ImportError:
                logger.error({
                    "event": "openai_not_available",
                    "correlation_id": correlation_id
                })
                return GenerationResult(
                    success=False,
                    error="OpenAI client not available",
                    error_type="ImportError",
                    fallback_used=True,
                    correlation_id=correlation_id
                )

            # Prepare prompt with constraint hints
            enhanced_prompt = self._enhance_prompt_with_constraints(prompt, constraints)

            # Generate
            response = client.chat.completions.create(
                model=self.config.model.split("/")[-1],  # Extract model name
                messages=[{"role": "user", "content": enhanced_prompt}],
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                timeout=self.config.timeout
            )

            text = response.choices[0].message.content

            # Validate output
            violations = self._validate_output(text, constraints)
            success = len(violations) == 0

            logger.info({
                "event": "fallback_generation_complete",
                "correlation_id": correlation_id,
                "success": success,
                "violations": violations
            })

            return GenerationResult(
                success=success,
                text=text,
                metadata={
                    "method": "fallback",
                    "model": self.config.model
                },
                constraint_violations=violations,
                fallback_used=True,
                correlation_id=correlation_id
            )

        except Exception as e:
            logger.error({
                "event": "fallback_generation_error",
                "error": str(e),
                "traceback": traceback.format_exc(),
                "correlation_id": correlation_id
            })

            return GenerationResult(
                success=False,
                error=str(e),
                error_type=type(e).__name__,
                fallback_used=True,
                correlation_id=correlation_id
            )

    def _enhance_prompt_with_constraints(
        self,
        prompt: str,
        constraints: List[Constraint]
    ) -> str:
        """
        Enhance prompt with constraint descriptions.

        Args:
            prompt: Original prompt.
            constraints: List of constraints.

        Returns:
            Enhanced prompt with constraint hints.
        """
        constraint_descriptions = []

        for constraint in constraints:
            if constraint.description:
                constraint_descriptions.append(f"- {constraint.description}")

            if constraint.type == ConstraintType.REGEX and constraint.pattern:
                constraint_descriptions.append(f"- Must match pattern: {constraint.pattern}")

            if constraint.type == ConstraintType.FROM_LIST and constraint.allowed_values:
                values = ", ".join(str(v) for v in constraint.allowed_values)
                constraint_descriptions.append(f"- Must be one of: {values}")

            if constraint.type == ConstraintType.LENGTH:
                if constraint.min_length:
                    constraint_descriptions.append(f"- Minimum length: {constraint.min_length}")
                if constraint.max_length:
                    constraint_descriptions.append(f"- Maximum length: {constraint.max_length}")

        if constraint_descriptions:
            enhanced = f"{prompt}\n\nConstraints:\n" + "\n".join(constraint_descriptions)
            return enhanced

        return prompt

    def _validate_output(
        self,
        text: str,
        constraints: List[Constraint]
    ) -> List[str]:
        """
        Validate output against constraints.

        Args:
            text: Generated text to validate.
            constraints: List of constraints.

        Returns:
            List of constraint violation messages.
        """
        violations = []

        for constraint in constraints:
            # Regex validation
            if constraint.type == ConstraintType.REGEX and constraint.pattern:
                if not re.match(constraint.pattern, text.strip()):
                    violations.append(
                        f"Failed regex constraint '{constraint.name}': "
                        f"does not match pattern {constraint.pattern}"
                    )

            # Length validation
            if constraint.type == ConstraintType.LENGTH:
                if constraint.min_length and len(text) < constraint.min_length:
                    violations.append(
                        f"Failed length constraint '{constraint.name}': "
                        f"{len(text)} < {constraint.min_length}"
                    )
                if constraint.max_length and len(text) > constraint.max_length:
                    violations.append(
                        f"Failed length constraint '{constraint.name}': "
                        f"{len(text)} > {constraint.max_length}"
                    )

            # Numerical validation
            if constraint.type == ConstraintType.NUMERICAL:
                try:
                    value = float(text.strip())
                    if constraint.min_value is not None and value < constraint.min_value:
                        violations.append(
                            f"Failed numerical constraint '{constraint.name}': "
                            f"{value} < {constraint.min_value}"
                        )
                    if constraint.max_value is not None and value > constraint.max_value:
                        violations.append(
                            f"Failed numerical constraint '{constraint.name}': "
                            f"{value} > {constraint.max_value}"
                        )
                except ValueError:
                    violations.append(
                        f"Failed numerical constraint '{constraint.name}': "
                        f"'{text}' is not a valid number"
                    )

            # From list validation
            if constraint.type == ConstraintType.FROM_LIST and constraint.allowed_values:
                if text.strip() not in [str(v) for v in constraint.allowed_values]:
                    violations.append(
                        f"Failed list constraint '{constraint.name}': "
                        f"'{text}' not in allowed values"
                    )

            # Custom validator
            if constraint.type == ConstraintType.CUSTOM and constraint.validator:
                try:
                    if not constraint.validator(text):
                        violations.append(
                            f"Failed custom constraint '{constraint.name}': "
                            f"{constraint.description}"
                        )
                except Exception as e:
                    violations.append(
                        f"Error in custom constraint '{constraint.name}': {e}"
                    )

            # JSON schema validation
            if constraint.type == ConstraintType.JSON_SCHEMA and constraint.schema:
                try:
                    data = json.loads(text)
                    # Basic schema validation (could use jsonschema library for full validation)
                    if "$schema" in constraint.schema or "type" in constraint.schema:
                        violations.append(
                            f"JSON schema validation for '{constraint.name}' "
                            f"requires jsonschema library for full validation"
                        )
                except json.JSONDecodeError:
                    violations.append(
                        f"Failed JSON constraint '{constraint.name}': "
                        f"output is not valid JSON"
                    )

        return violations

    # =========================================================================
    # STRUCTURED GENERATION
    # =========================================================================

    def structured_generation(
        self,
        schema: Dict[str, Any],
        prompt: str,
        **kwargs
    ) -> GenerationResult:
        """
        Generate structured output matching a JSON schema.

        Args:
            schema: JSON schema for output structure.
            prompt: Input prompt.
            **kwargs: Additional generation parameters.

        Returns:
            GenerationResult with structured output.
        """
        correlation_id = f"struct_{datetime.utcnow().timestamp()}"

        # Create JSON constraint
        json_constraint = Constraint(
            name="structured_output",
            type=ConstraintType.JSON_SCHEMA,
            schema=schema,
            description="Structured JSON output"
        )

        # Add schema requirements to prompt
        schema_description = json.dumps(schema, indent=2)
        enhanced_prompt = (
            f"{prompt}\n\n"
            f"Please provide output in the following JSON format:\n"
            f"```json\n{schema_description}\n```"
        )

        logger.info({
            "event": "structured_generation_start",
            "correlation_id": correlation_id
        })

        # Generate with JSON constraint
        result = self.constrained_generation(
            prompt=enhanced_prompt,
            constraints=[json_constraint, self.json_output_constraint()],
            **kwargs
        )

        # Parse and validate JSON structure
        if result.success and result.text:
            try:
                parsed = json.loads(result.text)
                result.metadata["parsed_json"] = parsed
            except json.JSONDecodeError as e:
                result.success = False
                result.error = f"Failed to parse JSON output: {e}"
                result.error_type = "JSONDecodeError"

        return result

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    @lru_cache(maxsize=128)
    def list_constraints(self) -> Dict[str, str]:
        """
        List available constraint templates with descriptions.

        Returns:
            Dictionary mapping constraint names to descriptions.
        """
        return {
            "roma_depth": "ROMA decomposition depth constraint (1-5 levels)",
            "vote_id": "MDAP vote ID format (e.g., A01, B99)",
            "lean_theorem": "Lean theorem declaration format",
            "json_output": "Valid JSON output constraint",
            "length": "Output length constraint",
            "mdap_score": "MDAP confidence score (0.0-1.0)",
            "roma_step": "ROMA step identifier format",
            "lean_tactic": "Lean tactic invocation format"
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get adapter status and configuration.

        Returns:
            Status dictionary with configuration and availability info.
        """
        return {
            "lmql_available": self.is_available(),
            "config": self.config.__dict__,
            "constraint_templates": list(self._constraint_templates.keys()),
            "cache_size": len(self._cache),
            "version": "1.0.0"
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

@lru_cache(maxsize=1)
def get_default_adapter() -> LMQLAdapter:
    """
    Get or create the default LMQL adapter instance.

    Returns:
        Cached LMQLAdapter instance.
    """
    return LMQLAdapter()


def generate_with_constraints(
    prompt: str,
    constraints: List[Union[Constraint, Dict[str, Any]]],
    **kwargs
) -> GenerationResult:
    """
    Convenience function for constrained generation.

    Args:
        prompt: Input prompt.
        constraints: List of constraints or constraint definitions.
        **kwargs: Additional generation parameters.

    Returns:
        GenerationResult.
    """
    adapter = get_default_adapter()

    # Convert dict constraints to Constraint objects
    constraint_objects = []
    for c in constraints:
        if isinstance(c, dict):
            constraint_objects.append(Constraint(**c))
        else:
            constraint_objects.append(c)

    return adapter.constrained_generation(prompt, constraint_objects, **kwargs)


def generate_structured(
    schema: Dict[str, Any],
    prompt: str,
    **kwargs
) -> GenerationResult:
    """
    Convenience function for structured generation.

    Args:
        schema: JSON schema for output.
        prompt: Input prompt.
        **kwargs: Additional generation parameters.

    Returns:
        GenerationResult with structured output.
    """
    adapter = get_default_adapter()
    return adapter.structured_generation(schema, prompt, **kwargs)


# =============================================================================
# MAIN EXPORTS
# =============================================================================

__all__ = [
    "LMQLAdapter",
    "LMQLConfig",
    "Constraint",
    "ConstraintType",
    "GenerationResult",
    "get_default_adapter",
    "generate_with_constraints",
    "generate_structured"
]
