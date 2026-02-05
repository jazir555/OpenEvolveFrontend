"""
RESE Framework - Configuration Validator

This module enforces the "Law of Configuration Explicitness":
- All required variables must be present
- All values must pass validation
- Application crashes if validation fails
"""

import os
import sys
import re
from typing import Any, Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class ValidationError(Exception):
    """Configuration validation failed."""
    pass


class ValidationLevel(Enum):
    """Validation strictness levels."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class VariableSpec:
    """Specification for a configuration variable."""
    name: str
    required: bool
    type: type
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    allowed_values: Optional[List[Any]] = None
    pattern: Optional[str] = None
    description: str = ""


class ConfigValidator:
    """Validates RESE configuration environment variables."""

    # Variable specifications following .env.example structure
    VARIABLE_SPECS: Dict[str, VariableSpec] = {
        # General Configuration
        "RESE_ENV": VariableSpec(
            name="RESE_ENV",
            required=True,
            type=str,
            allowed_values=["development", "staging", "production"],
            description="Environment mode"
        ),
        "RESE_LOG_LEVEL": VariableSpec(
            name="RESE_LOG_LEVEL",
            required=True,
            type=str,
            allowed_values=["DEBUG", "INFO", "WARN", "ERROR", "CRITICAL"],
            description="Logging level"
        ),
        "RESE_CORRELATION_ID": VariableSpec(
            name="RESE_CORRELATION_ID",
            required=True,
            type=str,
            description="Correlation ID for request tracing"
        ),

        # Phase I Variables
        "PHASE1_TIMEOUT_MS": VariableSpec(
            name="PHASE1_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=1000,
            max_value=300000,
            description="Phase I timeout in milliseconds"
        ),
        "PHASE1_MAX_ASSUMPTIONS": VariableSpec(
            name="PHASE1_MAX_ASSUMPTIONS",
            required=True,
            type=int,
            min_value=10,
            max_value=1000,
            description="Maximum assumptions to extract"
        ),
        "PHASE1_MIN_ASSUMPTION_CONFIDENCE": VariableSpec(
            name="PHASE1_MIN_ASSUMPTION_CONFIDENCE",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence for assumptions"
        ),
        "PHASE1_CIRCUIT_BREAKER_THRESHOLD": VariableSpec(
            name="PHASE1_CIRCUIT_BREAKER_THRESHOLD",
            required=True,
            type=int,
            min_value=1,
            max_value=100,
            description="Circuit breaker threshold"
        ),
        "PHASE1_ENABLE_TACIT_MINING": VariableSpec(
            name="PHASE1_ENABLE_TACIT_MINING",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable tacit assumption mining"
        ),
        "PHASE1_ENABLE_RED_TEAM": VariableSpec(
            name="PHASE1_ENABLE_RED_TEAM",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable red team mode"
        ),
        "PHASE1_ENABLE_LEAN4_INTEGRATION": VariableSpec(
            name="PHASE1_ENABLE_LEAN4_INTEGRATION",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable Lean4 integration"
        ),

        # Phase II Variables
        "PHASE2_TIMEOUT_MS": VariableSpec(
            name="PHASE2_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=1000,
            max_value=600000,
            description="Phase II timeout in milliseconds"
        ),
        "PHASE2_IMECH_THRESHOLD": VariableSpec(
            name="PHASE2_IMECH_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Isomorphism threshold"
        ),
        "PHASE2_MAX_TARGET_DOMAINS": VariableSpec(
            name="PHASE2_MAX_TARGET_DOMAINS",
            required=True,
            type=int,
            min_value=1,
            max_value=50,
            description="Maximum target domains"
        ),
        "PHASE2_PATTERN_THRESHOLD": VariableSpec(
            name="PHASE2_PATTERN_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Pattern matching threshold"
        ),
        "PHASE2_MAX_MAPPINGS": VariableSpec(
            name="PHASE2_MAX_MAPPINGS",
            required=True,
            type=int,
            min_value=1,
            max_value=1000,
            description="Maximum isomorphic mappings"
        ),
        "PHASE2_ENABLE_CONSTRAINT_INVERSION": VariableSpec(
            name="PHASE2_ENABLE_CONSTRAINT_INVERSION",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable constraint inversion"
        ),
        "PHASE2_SEARCH_DEPTH": VariableSpec(
            name="PHASE2_SEARCH_DEPTH",
            required=True,
            type=int,
            min_value=1,
            max_value=20,
            description="Search depth"
        ),

        # Phase III Variables
        "PHASE3_TIMEOUT_MS": VariableSpec(
            name="PHASE3_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=1000,
            max_value=3600000,
            description="Phase III timeout in milliseconds"
        ),
        "PHASE3_ITERATIONS": VariableSpec(
            name="PHASE3_ITERATIONS",
            required=True,
            type=int,
            min_value=100,
            max_value=10000000,
            description="MCTS iterations"
        ),
        "PHASE3_UCB1_C": VariableSpec(
            name="PHASE3_UCB1_C",
            required=True,
            type=float,
            min_value=0.0,
            max_value=10.0,
            description="UCB1 exploration constant"
        ),
        "PHASE3_CONVERGENCE_THRESHOLD": VariableSpec(
            name="PHASE3_CONVERGENCE_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Convergence threshold"
        ),
        "PHASE3_ACI_WINDOW": VariableSpec(
            name="PHASE3_ACI_WINDOW",
            required=True,
            type=int,
            min_value=10,
            max_value=10000,
            description="ACI window size"
        ),
        "PHASE3_SIG_THRESHOLD": VariableSpec(
            name="PHASE3_SIG_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Significance threshold"
        ),
        "PHASE3_PARALLEL_WORKERS": VariableSpec(
            name="PHASE3_PARALLEL_WORKERS",
            required=True,
            type=int,
            min_value=1,
            max_value=64,
            description="Parallel workers"
        ),

        # Phase IV Variables
        "PHASE4_TIMEOUT_MS": VariableSpec(
            name="PHASE4_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=1000,
            max_value=300000,
            description="Phase IV timeout in milliseconds"
        ),
        "PHASE4_BEAM_WIDTH": VariableSpec(
            name="PHASE4_BEAM_WIDTH",
            required=True,
            type=int,
            min_value=1,
            max_value=100,
            description="Beam width"
        ),
        "PHASE4_VALIDATION_LEVEL": VariableSpec(
            name="PHASE4_VALIDATION_LEVEL",
            required=True,
            type=int,
            min_value=0,
            max_value=3,
            description="Validation level"
        ),
        "PHASE4_INTEGRATION_STRATEGY": VariableSpec(
            name="PHASE4_INTEGRATION_STRATEGY",
            required=True,
            type=str,
            allowed_values=["conservative", "balanced", "aggressive"],
            description="Integration strategy"
        ),
        "PHASE4_MIN_CONFIDENCE_THRESHOLD": VariableSpec(
            name="PHASE4_MIN_CONFIDENCE_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Minimum confidence threshold"
        ),

        # LLTL Variables
        "LLTL_ENCODING_DIM": VariableSpec(
            name="LLTL_ENCODING_DIM",
            required=True,
            type=int,
            min_value=64,
            max_value=4096,
            description="LLTL encoding dimension"
        ),
        "LLTL_DEFAULT_LOSS_TYPE": VariableSpec(
            name="LLTL_DEFAULT_LOSS_TYPE",
            required=True,
            type=str,
            allowed_values=["cross_entropy", "mse", "hinge"],
            description="Default loss type"
        ),
        "LLTL_CONTRADICTION_THRESHOLD": VariableSpec(
            name="LLTL_CONTRADICTION_THRESHOLD",
            required=True,
            type=float,
            min_value=0.0,
            max_value=1.0,
            description="Contradiction threshold"
        ),
        "LLTL_TIMEOUT_MS": VariableSpec(
            name="LLTL_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=100,
            max_value=60000,
            description="LLTL timeout in milliseconds"
        ),

        # External Services
        "OPENAI_API_KEY": VariableSpec(
            name="OPENAI_API_KEY",
            required=True,
            type=str,
            pattern=r"^sk-",
            description="OpenAI API key"
        ),
        "OPENAI_MODEL": VariableSpec(
            name="OPENAI_MODEL",
            required=True,
            type=str,
            description="OpenAI model"
        ),
        "REDIS_URL": VariableSpec(
            name="REDIS_URL",
            required=True,
            type=str,
            pattern=r"^redis://",
            description="Redis URL"
        ),
        "REDIS_KEY_TTL": VariableSpec(
            name="REDIS_KEY_TTL",
            required=True,
            type=int,
            min_value=60,
            max_value=604800,
            description="Redis key TTL in seconds"
        ),

        # Telemetry
        "ENABLE_METRICS": VariableSpec(
            name="ENABLE_METRICS",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable metrics"
        ),
        "METRICS_PORT": VariableSpec(
            name="METRICS_PORT",
            required=False,
            type=int,
            min_value=1024,
            max_value=65535,
            description="Metrics port"
        ),
        "ENABLE_TRACING": VariableSpec(
            name="ENABLE_TRACING",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable tracing"
        ),

        # Failure Handling
        "ENABLE_CIRCUIT_BREAKERS": VariableSpec(
            name="ENABLE_CIRCUIT_BREAKERS",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable circuit breakers"
        ),
        "CIRCUIT_BREAKER_RESET_TIMEOUT_MS": VariableSpec(
            name="CIRCUIT_BREAKER_RESET_TIMEOUT_MS",
            required=True,
            type=int,
            min_value=1000,
            max_value=3600000,
            description="Circuit breaker reset timeout"
        ),
        "ENABLE_RETRY": VariableSpec(
            name="ENABLE_RETRY",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable retry"
        ),
        "MAX_RETRY_ATTEMPTS": VariableSpec(
            name="MAX_RETRY_ATTEMPTS",
            required=True,
            type=int,
            min_value=1,
            max_value=10,
            description="Maximum retry attempts"
        ),
        "RETRY_BASE_DELAY_MS": VariableSpec(
            name="RETRY_BASE_DELAY_MS",
            required=True,
            type=int,
            min_value=100,
            max_value=10000,
            description="Retry base delay"
        ),
        "ENABLE_DLQ": VariableSpec(
            name="ENABLE_DLQ",
            required=True,
            type=str,
            allowed_values=["true", "false"],
            description="Enable dead letter queue"
        ),

        # Advanced
        "MAX_MEMORY_MB": VariableSpec(
            name="MAX_MEMORY_MB",
            required=True,
            type=int,
            min_value=128,
            max_value=65536,
            description="Maximum memory usage"
        ),
    }

    def __init__(self, env_dict: Optional[Dict[str, str]] = None):
        """
        Initialize validator.

        Args:
            env_dict: Environment variables dict (defaults to os.environ)
        """
        self.env = env_dict or os.environ
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def validate_all(self) -> bool:
        """
        Validate all configuration variables.

        Returns:
            True if validation passes

        Raises:
            ValidationError: If validation fails
        """
        self.errors = []
        self.warnings = []

        # Check each variable specification
        for var_name, spec in self.VARIABLE_SPECS.items():
            self._validate_variable(var_name, spec)

        # Check for conditional requirements
        self._validate_conditional_requirements()

        # Report results
        if self.errors:
            error_msg = "Configuration validation failed:\n\n"
            error_msg += "\n".join(f"  - {err}" for err in self.errors)
            if self.warnings:
                error_msg += "\n\nWarnings:\n"
                error_msg += "\n".join(f"  - {warn}" for warn in self.warnings)
            raise ValidationError(error_msg)

        if self.warnings:
            print("Configuration warnings:", file=sys.stderr)
            for warning in self.warnings:
                print(f"  - {warning}", file=sys.stderr)

        return True

    def _validate_variable(self, var_name: str, spec: VariableSpec) -> None:
        """Validate a single configuration variable."""
        value = self.env.get(var_name)

        # Check presence
        if spec.required and not value:
            self.errors.append(
                f"Missing required variable: {var_name}\n"
                f"  Description: {spec.description}"
            )
            return

        if not value:
            return  # Optional variable not set

        # Type conversion and validation
        try:
            if spec.type == int:
                value = int(value)
            elif spec.type == float:
                value = float(value)
            elif spec.type == bool:
                value = value.lower() in ["true", "1", "yes"]
            elif spec.type == str:
                value = str(value)
        except ValueError:
            self.errors.append(
                f"Invalid type for {var_name}: expected {spec.type.__name__}, got '{value}'"
            )
            return

        # Range validation
        if spec.min_value is not None and value < spec.min_value:
            self.errors.append(
                f"{var_name}={value} is below minimum of {spec.min_value}"
            )

        if spec.max_value is not None and value > spec.max_value:
            self.errors.append(
                f"{var_name}={value} exceeds maximum of {spec.max_value}"
            )

        # Allowed values validation
        if spec.allowed_values is not None:
            # Handle boolean strings
            check_value = value.lower() if isinstance(value, str) else value
            if check_value not in spec.allowed_values:
                self.errors.append(
                    f"{var_name}='{value}' is not one of {spec.allowed_values}"
                )

        # Pattern validation
        if spec.pattern and isinstance(value, str):
            if not re.match(spec.pattern, value):
                self.errors.append(
                    f"{var_name}='{value}' does not match pattern '{spec.pattern}'"
                )

    def _validate_conditional_requirements(self) -> None:
        """Validate conditional requirements (e.g., if feature X is enabled)."""
        # Lean4 requires exec path
        if self.env.get("PHASE1_ENABLE_LEAN4_INTEGRATION", "false").lower() == "true":
            lean_path = self.env.get("LEAN4_EXEC_PATH")
            if not lean_path:
                self.errors.append(
                    "PHASE1_ENABLE_LEAN4_INTEGRATION=true requires LEAN4_EXEC_PATH to be set"
                )
            elif not os.path.exists(lean_path):
                self.errors.append(
                    f"LEAN4_EXEC_PATH={lean_path} does not exist or is not accessible"
                )

        # Metrics require port
        if self.env.get("ENABLE_METRICS", "false").lower() == "true":
            if not self.env.get("METRICS_PORT"):
                self.errors.append(
                    "ENABLE_METRICS=true requires METRICS_PORT to be set"
                )

        # Tracing requires endpoint
        if self.env.get("ENABLE_TRACING", "false").lower() == "true":
            if not self.env.get("JAEGER_ENDPOINT"):
                self.warnings.append(
                    "ENABLE_TRACING=true but JAEGER_ENDPOINT not set (tracing may not work)"
                )


def main():
    """Main entry point for command-line validation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate RESE framework configuration"
    )
    parser.add_argument(
        "--env-file",
        type=str,
        help="Path to .env file (defaults to .env in current directory)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed validation results"
    )

    args = parser.parse_args()

    # Load .env file if specified
    env_dict = dict(os.environ)
    if args.env_file:
        try:
            with open(args.env_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, value = line.split("=", 1)
                        env_dict[key.strip()] = value.strip()
        except FileNotFoundError:
            print(f"Error: .env file not found: {args.env_file}", file=sys.stderr)
            sys.exit(1)

    # Run validation
    validator = ConfigValidator(env_dict)

    try:
        if args.verbose:
            print("Validating configuration...", file=sys.stderr)

        validator.validate_all()

        print("✅ Configuration validation passed!", file=sys.stderr)
        if args.verbose:
            print(f"\nValidated {len(validator.VARIABLE_SPECS)} configuration variables")

        sys.exit(0)

    except ValidationError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
