"""
Configuration Validation Module for Knowledge Engine

This module provides comprehensive validation of all required and optional
environment variables used throughout the Knowledge Engine.

Following the ZERO TRUST principle: All required configuration must be
present at startup or the system will fail fast with a clear error message.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ConfigVariable:
    """
    Represents a single configuration variable.

    Attributes:
        name: Environment variable name
        category: Configuration category (e.g., "Database", "Cloud Storage")
        required: Whether this variable is required or optional
        default_value: Default value if not set (None for required vars)
        description: Human-readable description
        example: Example value
    """
    name: str
    category: str
    required: bool = True
    default_value: Optional[str] = None
    description: str = ""
    example: str = ""

    def validate(self) -> Tuple[bool, Optional[str]]:
        """
        Validate this configuration variable.

        Returns:
            Tuple of (is_valid, error_message)
        """
        value = os.getenv(self.name)

        if self.required:
            if not value:
                return False, f"Required environment variable '{self.name}' is not set"

            # Check for empty string defaults (bad pattern)
            if value == "" and self.default_value == "":
                return False, f"Environment variable '{self.name}' is set to empty string"

        return True, None


@dataclass
class ValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    missing_required: List[str] = field(default_factory=list)
    present_optional: Dict[str, str] = field(default_factory=dict)

    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
        self.is_valid = False

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'is_valid': self.is_valid,
            'errors': self.errors,
            'warnings': self.warnings,
            'missing_required': self.missing_required,
            'present_optional': self.present_optional,
            'total_issues': len(self.errors) + len(self.warnings)
        }


class ConfigValidator:
    """
    Comprehensive configuration validator for Knowledge Engine.

    Validates all required and optional environment variables across all
    integrated components and subsystems.
    """

    # Core Knowledge Graph Variables
    CORE_GRAPH: List[ConfigVariable] = [
        ConfigVariable(
            name="GRAPHITI_URI",
            category="Core Knowledge Graph",
            required=True,
            description="URI for Graphiti/Neo4j knowledge graph",
            example="bolt://localhost:7687"
        ),
        ConfigVariable(
            name="GRAPHITI_USER",
            category="Core Knowledge Graph",
            required=True,
            default_value="neo4j",
            description="Username for knowledge graph database",
            example="neo4j"
        ),
        ConfigVariable(
            name="GRAPHITI_PASSWORD",
            category="Core Knowledge Graph",
            required=True,
            description="Password for knowledge graph database",
            example="your_secure_password"
        ),
        ConfigVariable(
            name="NEO4J_URI",
            category="Core Knowledge Graph",
            required=False,
            description="Alternative Neo4j URI (if different from GRAPHITI_URI)",
            example="bolt://localhost:7687"
        ),
        ConfigVariable(
            name="NEO4J_USER",
            category="Core Knowledge Graph",
            required=False,
            default_value="neo4j",
            description="Alternative Neo4j user",
            example="neo4j"
        ),
        ConfigVariable(
            name="NEO4J_PASSWORD",
            category="Core Knowledge Graph",
            required=False,
            description="Alternative Neo4j password",
            example="your_secure_password"
        ),
    ]

    # LLM Configuration
    LLM_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="OPENAI_API_KEY",
            category="LLM Providers",
            required=False,  # Optional - system can work with other providers or without LLM
            description="OpenAI API key for LLM operations",
            example="sk-proj-..."
        ),
        ConfigVariable(
            name="ANTHROPIC_API_KEY",
            category="LLM Providers",
            required=False,
            description="Anthropic API key for Claude models",
            example="sk-ant-..."
        ),
        ConfigVariable(
            name="LLM_API_KEY",
            category="LLM Providers",
            required=False,
            description="Generic LLM API key (if not provider-specific)",
            example="your_api_key"
        ),
        ConfigVariable(
            name="LLM_API_BASE",
            category="LLM Providers",
            required=False,
            default_value="https://api.openai.com/v1",
            description="Base URL for LLM API",
            example="https://api.openai.com/v1"
        ),
        ConfigVariable(
            name="LLM_DEFAULT_MODEL",
            category="LLM Providers",
            required=False,
            default_value="gpt-4o",
            description="Default LLM model to use",
            example="gpt-4o"
        ),
        ConfigVariable(
            name="LLM_TEMPERATURE",
            category="LLM Providers",
            required=False,
            default_value="0.1",
            description="Default temperature for LLM generation",
            example="0.1"
        ),
        ConfigVariable(
            name="LLM_MAX_TOKENS",
            category="LLM Providers",
            required=False,
            default_value="2000",
            description="Default max tokens for LLM generation",
            example="2000"
        ),
        ConfigVariable(
            name="LLM_TIMEOUT",
            category="LLM Providers",
            required=False,
            default_value="120",
            description="LLM request timeout in seconds",
            example="120"
        ),
        ConfigVariable(
            name="LLM_MAX_RETRIES",
            category="LLM Providers",
            required=False,
            default_value="3",
            description="Maximum retries for LLM requests",
            example="3"
        ),
    ]

    # KGGen Integration
    KGGEN_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="KGGEN_ENTITY_MODEL",
            category="KGGen Integration",
            required=False,
            default_value="gpt-4o",
            description="Model for entity extraction",
            example="gpt-4o"
        ),
        ConfigVariable(
            name="KGGEN_RELATION_MODEL",
            category="KGGen Integration",
            required=False,
            default_value="gpt-4o",
            description="Model for relation extraction",
            example="gpt-4o"
        ),
        ConfigVariable(
            name="KGGEN_TIMEOUT_MS",
            category="KGGen Integration",
            required=False,
            default_value="30000",
            description="Timeout for KGGen operations (milliseconds)",
            example="30000"
        ),
        ConfigVariable(
            name="KGGEN_CHUNK_SIZE",
            category="KGGen Integration",
            required=False,
            default_value="5000",
            description="Chunk size for text processing",
            example="5000"
        ),
    ]

    # OneKE Integration
    ONEKE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="ONEKE_MODEL_NAME",
            category="OneKE Integration",
            required=False,
            default_value="oneke/OneKE-13B",
            description="OneKE model to use",
            example="oneke/OneKE-13B"
        ),
        ConfigVariable(
            name="ONEKE_DEVICE",
            category="OneKE Integration",
            required=False,
            default_value="cuda",
            description="Device for OneKE inference (cuda/cpu)",
            example="cuda"
        ),
        ConfigVariable(
            name="ONEKE_TIMEOUT_MS",
            category="OneKE Integration",
            required=False,
            default_value="60000",
            description="Timeout for OneKE operations (milliseconds)",
            example="60000"
        ),
        ConfigVariable(
            name="ONEKE_TASK_TIMEOUT",
            category="OneKE Integration",
            required=False,
            default_value="300",
            description="Task timeout for OneKE (seconds)",
            example="300"
        ),
        ConfigVariable(
            name="ONEKE_MAX_RETRIES",
            category="OneKE Integration",
            required=False,
            default_value="3",
            description="Maximum retries for OneKE tasks",
            example="3"
        ),
    ]

    # Qdrant Vector Store
    QDRANT_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="QDRANT_HOST",
            category="Qdrant Vector Store",
            required=False,
            default_value="localhost",
            description="Qdrant host address",
            example="localhost"
        ),
        ConfigVariable(
            name="QDRANT_PORT",
            category="Qdrant Vector Store",
            required=False,
            default_value="6333",
            description="Qdrant port",
            example="6333"
        ),
    ]

    # PostgreSQL Database
    POSTGRESQL_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="POSTGRESQL_URI",
            category="PostgreSQL Database",
            required=False,
            description="PostgreSQL connection URI",
            example="postgresql://user:pass@localhost:5432/openevolve"
        ),
        ConfigVariable(
            name="DB_HOST",
            category="PostgreSQL Database",
            required=False,
            default_value="localhost",
            description="Database host",
            example="localhost"
        ),
        ConfigVariable(
            name="DB_PORT",
            category="PostgreSQL Database",
            required=False,
            default_value="5432",
            description="Database port",
            example="5432"
        ),
        ConfigVariable(
            name="DB_USERNAME",
            category="PostgreSQL Database",
            required=False,
            default_value="openevolve",
            description="Database username",
            example="openevolve"
        ),
        ConfigVariable(
            name="DB_PASSWORD",
            category="PostgreSQL Database",
            required=False,
            description="Database password (REQUIRED in production)",
            example="your_secure_password"
        ),
        ConfigVariable(
            name="DB_NAME",
            category="PostgreSQL Database",
            required=False,
            default_value="openevolve_kg",
            description="Database name",
            example="openevolve_kg"
        ),
    ]

    # Redis Cache
    REDIS_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="REDIS_HOST",
            category="Redis Cache",
            required=False,
            default_value="localhost",
            description="Redis host address",
            example="localhost"
        ),
        ConfigVariable(
            name="REDIS_PORT",
            category="Redis Cache",
            required=False,
            default_value="6379",
            description="Redis port",
            example="6379"
        ),
    ]

    # Elasticsearch
    ELASTICSEARCH_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="ELASTICSEARCH_HOSTS",
            category="Elasticsearch",
            required=False,
            default_value="http://localhost:9200",
            description="Elasticsearch hosts (comma-separated)",
            example="http://localhost:9200"
        ),
        ConfigVariable(
            name="ELASTICSEARCH_API_KEY",
            category="Elasticsearch",
            required=False,
            description="Elasticsearch API key",
            example="your_elasticsearch_api_key"
        ),
        ConfigVariable(
            name="ELASTICSEARCH_INDEX_PREFIX",
            category="Elasticsearch",
            required=False,
            default_value="openevolve",
            description="Prefix for Elasticsearch indices",
            example="openevolve"
        ),
    ]

    # Cloud Storage (AWS S3)
    AWS_STORAGE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="AWS_ACCESS_KEY_ID",
            category="AWS S3 Storage",
            required=False,
            description="AWS access key ID (REQUIRED if using S3 storage)",
            example="AKIAIOSFODNN7EXAMPLE"
        ),
        ConfigVariable(
            name="AWS_SECRET_ACCESS_KEY",
            category="AWS S3 Storage",
            required=False,
            description="AWS secret access key (REQUIRED if using S3 storage)",
            example="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        ),
        ConfigVariable(
            name="AWS_REGION",
            category="AWS S3 Storage",
            required=False,
            default_value="us-east-1",
            description="AWS region",
            example="us-east-1"
        ),
        ConfigVariable(
            name="AWS_ENDPOINT_URL",
            category="AWS S3 Storage",
            required=False,
            description="Custom endpoint URL (for MinIO compatibility)",
            example="http://localhost:9000"
        ),
    ]

    # Google Cloud Storage
    GCS_STORAGE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="GCS_PROJECT_ID",
            category="Google Cloud Storage",
            required=False,
            description="GCP project ID (REQUIRED if using GCS)",
            example="my-gcp-project"
        ),
        ConfigVariable(
            name="GOOGLE_APPLICATION_CREDENTIALS",
            category="Google Cloud Storage",
            required=False,
            description="Path to GCP service account credentials JSON",
            example="/path/to/credentials.json"
        ),
        ConfigVariable(
            name="GCS_CREDENTIALS_JSON",
            category="Google Cloud Storage",
            required=False,
            description="Raw GCP credentials JSON string",
            example='{"type": "service_account", ...}'
        ),
    ]

    # Azure Blob Storage
    AZURE_STORAGE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="AZURE_STORAGE_ACCOUNT",
            category="Azure Blob Storage",
            required=False,
            description="Azure storage account name (REQUIRED if using Azure)",
            example="mystorageaccount"
        ),
        ConfigVariable(
            name="AZURE_STORAGE_KEY",
            category="Azure Blob Storage",
            required=False,
            description="Azure storage key",
            example="your_storage_key"
        ),
        ConfigVariable(
            name="AZURE_STORAGE_CONNECTION_STRING",
            category="Azure Blob Storage",
            required=False,
            description="Azure storage connection string",
            example="DefaultEndpointsProtocol=https;AccountName=..."
        ),
        ConfigVariable(
            name="AZURE_STORAGE_SAS_TOKEN",
            category="Azure Blob Storage",
            required=False,
            description="Azure SAS token",
            example="your_sas_token"
        ),
    ]

    # SFTP Storage
    SFTP_STORAGE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="SFTP_HOST",
            category="SFTP Storage",
            required=False,
            description="SFTP host (REQUIRED if using SFTP)",
            example="sftp.example.com"
        ),
        ConfigVariable(
            name="SFTP_PORT",
            category="SFTP Storage",
            required=False,
            default_value="22",
            description="SFTP port",
            example="22"
        ),
        ConfigVariable(
            name="SFTP_USERNAME",
            category="SFTP Storage",
            required=False,
            description="SFTP username",
            example="sftp_user"
        ),
        ConfigVariable(
            name="SFTP_PASSWORD",
            category="SFTP Storage",
            required=False,
            description="SFTP password",
            example="sftp_password"
        ),
        ConfigVariable(
            name="SFTP_PRIVATE_KEY_PATH",
            category="SFTP Storage",
            required=False,
            description="Path to SFTP private key",
            example="/path/to/private_key"
        ),
        ConfigVariable(
            name="SFTP_KEY_PASSPHRASE",
            category="SFTP Storage",
            required=False,
            description="Passphrase for SFTP private key",
            example="key_passphrase"
        ),
    ]

    # Math Knowledge Integration
    MATH_KNOWLEDGE_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="MATH_KNOWLEDGE_DB_URL",
            category="Math Knowledge",
            required=False,
            default_value="sqlite:///math_knowledge.db",
            description="Math knowledge database URL",
            example="sqlite:///math_knowledge.db"
        ),
        ConfigVariable(
            name="MATH_KNOWLEDGE_Z3_TIMEOUT_MS",
            category="Math Knowledge",
            required=False,
            default_value="30000",
            description="Z3 solver timeout (milliseconds)",
            example="30000"
        ),
        ConfigVariable(
            name="MATH_KNOWLEDGE_Z3_MEMORY_MB",
            category="Math Knowledge",
            required=False,
            default_value="4096",
            description="Z3 solver memory limit (MB)",
            example="4096"
        ),
    ]

    # Server Configuration
    SERVER_CONFIG: List[ConfigVariable] = [
        ConfigVariable(
            name="SERVER_HOST",
            category="Server",
            required=False,
            default_value="0.0.0.0",
            description="Server host address",
            example="0.0.0.0"
        ),
        ConfigVariable(
            name="SERVER_PORT",
            category="Server",
            required=False,
            default_value="8000",
            description="Server port",
            example="8000"
        ),
        ConfigVariable(
            name="LOG_LEVEL",
            category="Server",
            required=False,
            default_value="INFO",
            description="Logging level",
            example="INFO"
        ),
    ]

    @classmethod
    def get_all_configs(cls) -> List[ConfigVariable]:
        """Get all configuration variables across all categories."""
        all_configs = []
        for attr_name in dir(cls):
            if attr_name.endswith('_CONFIG') and attr_name.isupper():
                all_configs.extend(getattr(cls, attr_name))
        return all_configs

    @classmethod
    def get_required_configs(cls) -> List[ConfigVariable]:
        """Get all required configuration variables."""
        return [cfg for cfg in cls.get_all_configs() if cfg.required]

    @classmethod
    def get_optional_configs(cls) -> List[ConfigVariable]:
        """Get all optional configuration variables."""
        return [cfg for cfg in cls.get_all_configs() if not cfg.required]

    def __init__(self):
        """Initialize the configuration validator."""
        self.result = ValidationResult(is_valid=True)

    def validate(self, strict: bool = False) -> ValidationResult:
        """
        Validate all configuration variables.

        Args:
            strict: If True, fail on warnings as well as errors

        Returns:
            ValidationResult with validation status
        """
        self.result = ValidationResult(is_valid=True)

        # Validate all configuration variables
        for config_var in self.get_all_configs():
            is_valid, error_msg = config_var.validate()

            if not is_valid:
                self.result.add_error(error_msg)
                self.result.missing_required.append(config_var.name)
            elif config_var.required and os.getenv(config_var.name):
                # Track present required variables
                self.result.present_optional[config_var.name] = os.getenv(config_var.name, "")
            elif not config_var.required and os.getenv(config_var.name):
                # Track present optional variables
                self.result.present_optional[config_var.name] = os.getenv(config_var.name, "")

        # Check for common misconfigurations
        self._check_common_misconfigurations()

        # Log results
        self._log_results()

        # Update is_valid based on errors
        self.result.is_valid = len(self.result.errors) == 0

        # If strict mode, also consider warnings
        if strict and self.result.warnings:
            self.result.is_valid = False

        return self.result

    def _check_common_misconfigurations(self):
        """Check for common configuration mistakes."""
        # Check for empty API keys
        api_key_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "LLM_API_KEY",
            "AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"
        ]
        for var in api_key_vars:
            value = os.getenv(var)
            if value and len(value.strip()) < 10:
                self.result.add_warning(
                    f"Environment variable '{var}' appears to have an invalid value (too short)"
                )

        # Check for default passwords
        password_vars = ["GRAPHITI_PASSWORD", "NEO4J_PASSWORD", "DB_PASSWORD"]
        for var in password_vars:
            value = os.getenv(var)
            if value and value.lower() in ["password", "pass", "123456", "admin"]:
                self.result.add_warning(
                    f"Environment variable '{var}' is using a default/weak password"
                )

        # Check for localhost in production
        if os.getenv("ENVIRONMENT", "development").lower() == "production":
            localhost_warnings = []
            if os.getenv("DB_HOST") == "localhost":
                localhost_warnings.append("DB_HOST")
            if os.getenv("REDIS_HOST") == "localhost":
                localhost_warnings.append("REDIS_HOST")
            if os.getenv("QDRANT_HOST") == "localhost":
                localhost_warnings.append("QDRANT_HOST")

            if localhost_warnings:
                self.result.add_warning(
                    f"Production environment detected but the following variables are set to localhost: "
                    f"{', '.join(localhost_warnings)}"
                )

    def _log_results(self):
        """Log validation results."""
        if self.result.is_valid:
            logger.info({
                "msg": "Configuration validation passed",
                "required_vars_present": len(self.result.present_optional),
                "warnings": len(self.result.warnings),
            })
        else:
            logger.error({
                "msg": "Configuration validation failed",
                "errors": self.result.errors,
                "warnings": self.result.warnings,
                "missing_required_count": len(self.result.missing_required)
            })

        # Log warnings
        for warning in self.result.warnings:
            logger.warning({"msg": warning})

    def print_report(self):
        """Print a human-readable validation report."""
        print("\n" + "="*80)
        print("KNOWLEDGE ENGINE CONFIGURATION VALIDATION REPORT")
        print("="*80 + "\n")

        if self.result.is_valid:
            print("[OK] Configuration validation PASSED\n")
        else:
            print("[FAIL] Configuration validation FAILED\n")

        # Print errors
        if self.result.errors:
            print("ERRORS:")
            print("-" * 80)
            for error in self.result.errors:
                print(f"  [FAIL] {error}")
            print()

        # Print warnings
        if self.result.warnings:
            print("WARNINGS:")
            print("-" * 80)
            for warning in self.result.warnings:
                print(f"  [WARN] {warning}")
            print()

        # Print present configuration
        if self.result.present_optional:
            print("CONFIGURED VARIABLES:")
            print("-" * 80)
            # Group by category
            by_category: Dict[str, List[str]] = {}
            all_configs = self.get_all_configs()
            config_map = {cfg.name: cfg for cfg in all_configs}

            for var_name, var_value in self.result.present_optional.items():
                if var_name in config_map:
                    category = config_map[var_name].category
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append(f"  {var_name}: {self._mask_sensitive(var_name, var_value)}")

            for category, vars_list in sorted(by_category.items()):
                print(f"\n  {category}:")
                for var_str in vars_list:
                    print(var_str)
            print()

        print("="*80)
        print()

    def _mask_sensitive(self, var_name: str, value: str) -> str:
        """Mask sensitive values in output."""
        sensitive_keywords = ["PASSWORD", "SECRET", "KEY", "TOKEN", "CREDENTIALS"]
        if any(keyword in var_name.upper() for keyword in sensitive_keywords):
            if len(value) > 8:
                return f"{value[:4]}...{value[-4:]}"
            else:
                return "***"
        return value


def validate_config(strict: bool = False, silent: bool = False) -> ValidationResult:
    """
    Validate all Knowledge Engine configuration.

    This function should be called at application startup to ensure
    all required configuration is present. Follows the FAIL FAST principle.

    Args:
        strict: If True, fail on warnings as well as errors
        silent: If True, don't print the validation report

    Returns:
        ValidationResult with validation status

    Raises:
        ConfigError: If validation fails and not in silent mode

    Example:
        >>> from knowledge_engine.config_validation import validate_config
        >>> try:
        ...     result = validate_config()
        ...     print("Configuration is valid!")
        ... except ConfigError as e:
        ...     print(f"Configuration error: {e}")
    """
    validator = ConfigValidator()
    result = validator.validate(strict=strict)

    if not silent:
        validator.print_report()

    if not result.is_valid:
        error_msg = (
            f"Configuration validation failed with {len(result.errors)} error(s)\n"
            f"Missing required variables: {', '.join(result.missing_required)}\n"
            "Please set these environment variables before starting the Knowledge Engine."
        )
        raise ConfigError(error_msg)

    return result


def get_config_template() -> str:
    """
    Generate a configuration template file.

    Returns:
        String content of a .env template file
    """
    lines = [
        "# Knowledge Engine Configuration Template",
        "#",
        "# Copy this file to .env and fill in the required values",
        "#",
        "# IMPORTANT: Never commit .env to version control!",
        "",
    ]

    all_configs = ConfigValidator.get_all_configs()
    by_category: Dict[str, List[ConfigVariable]] = {}

    for cfg in all_configs:
        if cfg.category not in by_category:
            by_category[cfg.category] = []
        by_category[cfg.category].append(cfg)

    for category in sorted(by_category.keys()):
        lines.append(f"# {category}")
        for cfg in by_category[category]:
            if cfg.required:
                lines.append(f"# REQUIRED: {cfg.description}")
                lines.append(f"{cfg.name}={cfg.example}")
            else:
                default = cfg.default_value or "unset"
                lines.append(f"# Optional: {cfg.description}")
                lines.append(f"# {cfg.name}={default}")
        lines.append("")

    return "\n".join(lines)


# Export main validation function
__all__ = [
    'ConfigError',
    'ConfigVariable',
    'ValidationResult',
    'ConfigValidator',
    'validate_config',
    'get_config_template',
]
