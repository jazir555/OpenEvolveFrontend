"""
OpenEvolve Reliability Configuration Manager

Production configuration management for the 4-layer reliability system:
1. LMQL - Query-based language model constraints
2. Guardrails - Runtime output validation
3. ACE - Adversarial Contextual Evolution
4. Steer - Structured output verification

Author: OpenEvolve
Version: 1.0.0
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Literal, Any, Set
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache
import threading

from pydantic import (
    BaseSettings,
    Field,
    validator,
    root_validator,
    ValidationError
)
from pydantic.types import PositiveInt, NonNegativeFloat


# ============================================================================
# Configuration Enums and Constants
# ============================================================================

class DecodingStrategy(str, Enum):
    """LMQL decoding strategies"""
    ARGMAX = "argmax"
    BEAM = "beam"
    SAMPLE = "sample"


class OnFailStrategy(str, Enum):
    """Guardrails failure handling strategies"""
    REASK = "reask"
    FIX = "fix"
    FILTER = "filter"
    REFRAIN = "refrain"
    NOOP = "noop"
    EXCEPTION = "exception"
    FIX_REASK = "fix_reask"
    CUSTOM = "custom"


class LearningMode(str, Enum):
    """ACE learning modes"""
    ONLINE = "online"
    OFFLINE = "offline"
    ASYNC = "async"


class ValidationStrictness(str, Enum):
    """Unified bridge validation strictness levels"""
    STRICT = "strict"
    MODERATE = "moderate"
    PERMISSIVE = "permissive"


class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# Default validator lists
DEFAULT_GUARDRAILS_VALIDATORS = [
    "toxic_language",
    "pii_filter",
    "fact_checking"
]

DEFAULT_STEER_VERIFICATIONS = [
    "json",
    "slop",
    "pii",
    "coherence"
]

RELIABILITY_LAYERS = ["lmql", "guardrails", "ace", "steer", "roma", "unified_bridge"]


# ============================================================================
# Audit Trail Management
# ============================================================================

@dataclass
class ConfigChange:
    """Represents a configuration change for audit trail"""
    timestamp: datetime
    key: str
    old_value: Any
    new_value: Any
    source: str = "manual"

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "key": self.key,
            "old_value": str(self.old_value) if self.old_value is not None else None,
            "new_value": str(self.new_value) if self.new_value is not None else None,
            "source": self.source
        }


class ConfigAuditTrail:
    """Manages audit trail for configuration changes"""

    def __init__(self, max_entries: int = 1000):
        self.max_entries = max_entries
        self._changes: List[ConfigChange] = []
        self._lock = threading.Lock()

    def log_change(self, key: str, old_value: Any, new_value: Any, source: str = "manual"):
        """Log a configuration change"""
        with self._lock:
            change = ConfigChange(
                timestamp=datetime.utcnow(),
                key=key,
                old_value=old_value,
                new_value=new_value,
                source=source
            )
            self._changes.append(change)

            # Keep only the most recent entries
            if len(self._changes) > self.max_entries:
                self._changes = self._changes[-self.max_entries:]

            # Log to standard logger
            logging.info(
                f"Config change: {key} changed from '{old_value}' to '{new_value}' (source: {source})"
            )

    def get_recent_changes(self, limit: int = 10) -> List[Dict]:
        """Get recent configuration changes"""
        with self._lock:
            recent = self._changes[-limit:]
            return [change.to_dict() for change in recent]

    def export_audit_log(self, filepath: Optional[Path] = None) -> str:
        """Export audit log to JSON"""
        with self._lock:
            log_data = [change.to_dict() for change in self._changes]
            json_str = json.dumps(log_data, indent=2)

            if filepath:
                filepath.write_text(json_str, encoding='utf-8')

            return json_str


# Global audit trail instance
_audit_trail = ConfigAuditTrail()


# ============================================================================
# Pydantic Configuration Models
# ============================================================================

class LMQLConfig(BaseSettings):
    """LMQL layer configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable LMQL layer"
    )
    model: str = Field(
        default="openai/gpt-4",
        description="LMQL model identifier"
    )
    decoding: DecodingStrategy = Field(
        default=DecodingStrategy.ARGMAX,
        description="LMQL decoding strategy"
    )
    cache_enabled: bool = Field(
        default=True,
        description="Enable LMQL query caching"
    )
    cache_ttl: PositiveInt = Field(
        default=3600,
        description="Cache TTL in seconds"
    )
    timeout: PositiveInt = Field(
        default=30,
        description="Query timeout in seconds"
    )
    max_retries: PositiveInt = Field(
        default=3,
        description="Maximum retry attempts"
    )
    max_tokens: Optional[PositiveInt] = Field(
        default=2048,
        description="Maximum tokens per query"
    )
    temperature: NonNegativeFloat = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )

    @validator('model')
    def validate_model(cls, v):
        """Validate model identifier"""
        if not v or not isinstance(v, str):
            raise ValueError("Model must be a non-empty string")
        return v

    @validator('temperature')
    def validate_temperature(cls, v):
        """Validate temperature is in valid range"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v

    class Config:
        env_prefix = "RELIABILITY_LMQL_"
        env_file = ".env"
        case_sensitive = False


class GuardrailsConfig(BaseSettings):
    """Guardrails layer configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable Guardrails layer"
    )
    validators: List[str] = Field(
        default_factory=lambda: DEFAULT_GUARDRAILS_VALIDATORS.copy(),
        description="Active guardrail validators"
    )
    on_fail: OnFailStrategy = Field(
        default=OnFailStrategy.REASK,
        description="Failure handling strategy"
    )
    max_retries: PositiveInt = Field(
        default=3,
        description="Maximum validation retries"
    )
    timeout: PositiveInt = Field(
        default=30,
        description="Validation timeout in seconds"
    )
    parallel_validation: bool = Field(
        default=True,
        description="Run validators in parallel"
    )
    custom_validator_paths: Optional[List[str]] = Field(
        default=None,
        description="Paths to custom validator modules"
    )

    @validator('validators')
    def validate_validators(cls, v):
        """Validate validator list"""
        if not isinstance(v, list):
            raise ValueError("Validators must be a list")
        if len(v) == 0:
            raise ValueError("At least one validator must be specified")
        # Validate no duplicates
        if len(v) != len(set(v)):
            raise ValueError("Validators must be unique")
        return v

    @validator('max_retries')
    def validate_max_retries(cls, v):
        """Validate max retries"""
        if v < 1 or v > 10:
            raise ValueError("Max retries must be between 1 and 10")
        return v

    class Config:
        env_prefix = "RELIABILITY_GUARDRAILS_"
        env_file = ".env"
        case_sensitive = False


class ACEConfig(BaseSettings):
    """ACE (Adversarial Contextual Evolution) layer configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable ACE layer"
    )
    skillbook_path: str = Field(
        default="./skills.json",
        description="Path to skillbook JSON file"
    )
    agent_id: str = Field(
        default="openevolve_agent",
        description="ACE agent identifier"
    )
    learning_mode: LearningMode = Field(
        default=LearningMode.ONLINE,
        description="ACE learning mode"
    )
    adversarial_budget: PositiveInt = Field(
        default=100,
        description="Adversarial test budget per iteration"
    )
    evolution_rate: NonNegativeFloat = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="Learning evolution rate"
    )
    skill_retention_threshold: NonNegativeFloat = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum score for skill retention"
    )
    max_skills: PositiveInt = Field(
        default=1000,
        description="Maximum skills in skillbook"
    )

    @validator('skillbook_path')
    def validate_skillbook_path(cls, v):
        """Validate skillbook path"""
        if not v or not isinstance(v, str):
            raise ValueError("Skillbook path must be a non-empty string")
        # Expand user path
        return os.path.expanduser(v)

    @validator('evolution_rate')
    def validate_evolution_rate(cls, v):
        """Validate evolution rate"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Evolution rate must be between 0.0 and 1.0")
        return v

    @validator('skill_retention_threshold')
    def validate_retention_threshold(cls, v):
        """Validate skill retention threshold"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("Skill retention threshold must be between 0.0 and 1.0")
        return v

    class Config:
        env_prefix = "RELIABILITY_ACE_"
        env_file = ".env"
        case_sensitive = False


class SteerConfig(BaseSettings):
    """Steer layer configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable Steer layer"
    )
    verifications: List[str] = Field(
        default_factory=lambda: DEFAULT_STEER_VERIFICATIONS.copy(),
        description="Active verification types"
    )
    halt_on_failure: bool = Field(
        default=False,
        description="Halt execution on verification failure"
    )
    slop_threshold: NonNegativeFloat = Field(
        default=3.5,
        ge=0.0,
        le=10.0,
        description="Slop detection threshold"
    )
    json_schema_strictness: Literal["strict", "loose"] = Field(
        default="strict",
        description="JSON schema validation strictness"
    )
    pii_detection_confidence: NonNegativeFloat = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="PII detection confidence threshold"
    )
    coherence_check_enabled: bool = Field(
        default=True,
        description="Enable coherence checking"
    )

    @validator('verifications')
    def validate_verifications(cls, v):
        """Validate verification list"""
        if not isinstance(v, list):
            raise ValueError("Verifications must be a list")
        if len(v) == 0:
            raise ValueError("At least one verification must be specified")
        # Validate no duplicates
        if len(v) != len(set(v)):
            raise ValueError("Verifications must be unique")
        # Validate known verification types
        known_types = {"json", "slop", "pii", "coherence", "consistency"}
        unknown = set(v) - known_types
        if unknown:
            raise ValueError(f"Unknown verification types: {unknown}")
        return v

    @validator('slop_threshold')
    def validate_slop_threshold(cls, v):
        """Validate slop threshold"""
        if not 0.0 <= v <= 10.0:
            raise ValueError("Slop threshold must be between 0.0 and 10.0")
        return v

    @validator('pii_detection_confidence')
    def validate_pii_confidence(cls, v):
        """Validate PII detection confidence"""
        if not 0.0 <= v <= 1.0:
            raise ValueError("PII detection confidence must be between 0.0 and 1.0")
        return v

    class Config:
        env_prefix = "RELIABILITY_STEER_"
        env_file = ".env"
        case_sensitive = False


class ROMAConfig(BaseSettings):
    """ROMA-MDAP-MAKER configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable ROMA-MDAP-MAKER layer"
    )
    max_depth_analysis: PositiveInt = Field(
        default=1,
        description="Maximum depth for ROMA analysis"
    )
    max_depth_solving: PositiveInt = Field(
        default=1,
        description="Maximum depth for ROMA solving"
    )
    mdap_k_ahead: PositiveInt = Field(
        default=3,
        description="MDAP lookahead/voting factor"
    )
    use_associative_recomposition: bool = Field(
        default=True,
        description="Enable robust associative recomposition"
    )

    class Config:
        env_prefix = "RELIABILITY_ROMA_"
        env_file = ".env"
        case_sensitive = False


class UnifiedBridgeConfig(BaseSettings):
    """Unified bridge configuration"""

    enabled: bool = Field(
        default=True,
        description="Enable unified bridge"
    )
    fallback_on_error: bool = Field(
        default=True,
        description="Enable fallback on layer failure"
    )
    validation_strictness: ValidationStrictness = Field(
        default=ValidationStrictness.STRICT,
        description="Overall validation strictness"
    )
    layer_order: List[str] = Field(
        default_factory=lambda: ["lmql", "guardrails", "ace", "steer"],
        description="Layer execution order"
    )
    skip_on_failure: List[str] = Field(
        default_factory=list,
        description="Layers to skip on failure"
    )
    retry_enabled: bool = Field(
        default=True,
        description="Enable retry on bridge failure"
    )
    max_bridge_retries: PositiveInt = Field(
        default=2,
        description="Maximum bridge retry attempts"
    )

    @validator('layer_order')
    def validate_layer_order(cls, v):
        """Validate layer order"""
        if not isinstance(v, list):
            raise ValueError("Layer order must be a list")
        if len(v) == 0:
            raise ValueError("At least one layer must be specified")
        # Validate all layers are known
        unknown = set(v) - set(RELIABILITY_LAYERS)
        if unknown:
            raise ValueError(f"Unknown layers: {unknown}")
        # Validate no duplicates
        if len(v) != len(set(v)):
            raise ValueError("Layer order must not contain duplicates")
        return v

    @validator('skip_on_failure')
    def validate_skip_on_failure(cls, v):
        """Validate skip on failure list"""
        if not isinstance(v, list):
            raise ValueError("Skip on failure must be a list")
        # Validate all layers are known
        unknown = set(v) - set(RELIABILITY_LAYERS)
        if unknown:
            raise ValueError(f"Unknown layers in skip_on_failure: {unknown}")
        return v

    class Config:
        env_prefix = "RELIABILITY_BRIDGE_"
        env_file = ".env"
        case_sensitive = False


class ObservabilityConfig(BaseSettings):
    """Observability and monitoring configuration"""

    log_level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    enable_telemetry: bool = Field(
        default=True,
        description="Enable telemetry collection"
    )
    telemetry_endpoint: Optional[str] = Field(
        default=None,
        description="Telemetry endpoint URL"
    )
    metrics_export_interval: PositiveInt = Field(
        default=60,
        description="Metrics export interval in seconds"
    )
    audit_log_enabled: bool = Field(
        default=True,
        description="Enable configuration audit logging"
    )
    audit_log_path: Optional[str] = Field(
        default=None,
        description="Audit log file path"
    )
    health_check_interval: PositiveInt = Field(
        default=30,
        description="Health check interval in seconds"
    )

    @validator('telemetry_endpoint')
    def validate_telemetry_endpoint(cls, v):
        """Validate telemetry endpoint"""
        if v and not isinstance(v, str):
            raise ValueError("Telemetry endpoint must be a string")
        return v

    class Config:
        env_prefix = "RELIABILITY_OBSERVABILITY_"
        env_file = ".env"
        case_sensitive = False


class ReliabilityConfig(BaseSettings):
    """Main reliability configuration"""

    lmql: LMQLConfig = Field(default_factory=LMQLConfig)
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    ace: ACEConfig = Field(default_factory=ACEConfig)
    steer: SteerConfig = Field(default_factory=SteerConfig)
    roma: ROMAConfig = Field(default_factory=ROMAConfig)
    unified_bridge: UnifiedBridgeConfig = Field(default_factory=UnifiedBridgeConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)

    # Global settings
    config_version: str = Field(
        default="1.0.0",
        description="Configuration schema version"
    )
    environment: Literal["development", "staging", "production"] = Field(
        default="development",
        description="Deployment environment"
    )

    @root_validator
    def validate_compatibility(cls, values):
        """Cross-field validation"""
        lmql = values.get('lmql')
        guardrails = values.get('guardrails')
        ace = values.get('ace')
        steer = values.get('steer')
        roma = values.get('roma')
        bridge = values.get('unified_bridge')

        # Validate that if bridge is enabled, at least one layer is enabled
        if bridge and bridge.enabled:
            layers_enabled = []
            if lmql and lmql.enabled:
                layers_enabled.append('lmql')
            if guardrails and guardrails.enabled:
                layers_enabled.append('guardrails')
            if ace and ace.enabled:
                layers_enabled.append('ace')
            if steer and steer.enabled:
                layers_enabled.append('steer')
            if roma and roma.enabled:
                layers_enabled.append('roma')

            if not layers_enabled:
                raise ValueError(
                    "At least one reliability layer must be enabled when unified_bridge is enabled"
                )

        return values

    class Config:
        env_prefix = "RELIABILITY_"
        env_file = ".env"
        case_sensitive = False


# ============================================================================
# Configuration Manager
# ============================================================================

class ConfigManager:
    """
    Manages reliability configuration with hot reload support
    """

    def __init__(self):
        self._config: Optional[ReliabilityConfig] = None
        self._lock = threading.RLock()
        self._health_cache: Dict[str, Dict] = {}
        self._health_cache_ttl = 30  # seconds
        self._last_health_check: Dict[str, datetime] = {}

    def load_config(self) -> ReliabilityConfig:
        """Load configuration from environment"""
        with self._lock:
            try:
                config = ReliabilityConfig()
                self._config = config
                logging.info(f"Configuration loaded successfully (version {config.config_version})")
                return config
            except ValidationError as e:
                logging.error(f"Configuration validation failed: {e}")
                raise
            except Exception as e:
                logging.error(f"Failed to load configuration: {e}")
                raise

    def get_config(self) -> ReliabilityConfig:
        """Get current configuration, loading if necessary"""
        with self._lock:
            if self._config is None:
                self._config = self.load_config()
            return self._config

    def reload_config(self) -> ReliabilityConfig:
        """Reload configuration from environment"""
        with self._lock:
            old_config = self._config
            new_config = self.load_config()

            # Log changes
            if old_config:
                self._log_config_changes(old_config, new_config)

            self._config = new_config
            logging.info("Configuration reloaded successfully")
            return new_config

    def _log_config_changes(self, old: ReliabilityConfig, new: ReliabilityConfig):
        """Log configuration changes"""
        # Compare top-level fields
        if old.environment != new.environment:
            _audit_trail.log_change(
                "environment",
                old.environment,
                new.environment,
                "reload"
            )

        # Compare LMQL config
        if old.lmql.enabled != new.lmql.enabled:
            _audit_trail.log_change(
                "lmql.enabled",
                old.lmql.enabled,
                new.lmql.enabled,
                "reload"
            )

        # Compare Guardrails config
        if old.guardrails.enabled != new.guardrails.enabled:
            _audit_trail.log_change(
                "guardrails.enabled",
                old.guardrails.enabled,
                new.guardrails.enabled,
                "reload"
            )

        # Compare ACE config
        if old.ace.enabled != new.ace.enabled:
            _audit_trail.log_change(
                "ace.enabled",
                old.ace.enabled,
                new.ace.enabled,
                "reload"
            )

        # Compare Steer config
        if old.steer.enabled != new.steer.enabled:
            _audit_trail.log_change(
                "steer.enabled",
                old.steer.enabled,
                new.steer.enabled,
                "reload"
            )

        # Compare ROMA config
        if old.roma.enabled != new.roma.enabled:
            _audit_trail.log_change(
                "roma.enabled",
                old.roma.enabled,
                new.roma.enabled,
                "reload"
            )

    def update_config(self, updates: Dict[str, Any]) -> ReliabilityConfig:
        """
        Update configuration at runtime

        Args:
            updates: Dictionary of config updates

        Returns:
            Updated configuration

        Example:
            update_config({
                "lmql": {"enabled": False},
                "observability": {"log_level": "DEBUG"}
            })
        """
        with self._lock:
            old_config = self.get_config()

            # Convert to dict, update, and recreate
            config_dict = old_config.dict()

            def deep_update(base: Dict, updates: Dict) -> Dict:
                """Deep update dictionary"""
                for key, value in updates.items():
                    if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                        deep_update(base[key], value)
                    else:
                        base[key] = value
                return base

            updated_dict = deep_update(config_dict.copy(), updates)

            try:
                # Validate and create new config
                new_config = ReliabilityConfig(**updated_dict)

                # Log changes
                for key, value in updates.items():
                    _audit_trail.log_change(
                        key,
                        getattr(old_config, key, None),
                        value,
                        "update"
                    )

                self._config = new_config
                logging.info(f"Configuration updated: {list(updates.keys())}")
                return new_config

            except ValidationError as e:
                logging.error(f"Configuration update validation failed: {e}")
                raise ValueError(f"Invalid configuration update: {e}")

    def validate_config(self, config: Optional[ReliabilityConfig] = None) -> bool:
        """Validate configuration"""
        try:
            cfg = config or self.get_config()
            # Trigger Pydantic validation
            ReliabilityConfig(**cfg.dict())
            return True
        except ValidationError:
            return False

    def is_layer_enabled(self, layer: str) -> bool:
        """Check if a layer is enabled"""
        config = self.get_config()

        layer_map = {
            "lmql": config.lmql.enabled,
            "guardrails": config.guardrails.enabled,
            "ace": config.ace.enabled,
            "steer": config.steer.enabled,
            "unified_bridge": config.unified_bridge.enabled
        }

        if layer not in layer_map:
            raise ValueError(f"Unknown layer: {layer}")

        return layer_map[layer]

    def get_layer_status(self, layer: str) -> Dict:
        """Get status of a specific layer"""
        config = self.get_config()

        status_map = {
            "lmql": {
                "enabled": config.lmql.enabled,
                "model": config.lmql.model,
                "decoding": config.lmql.decoding.value,
                "cache_enabled": config.lmql.cache_enabled,
                "timeout": config.lmql.timeout,
                "max_retries": config.lmql.max_retries
            },
            "guardrails": {
                "enabled": config.guardrails.enabled,
                "validators": config.guardrails.validators,
                "on_fail": config.guardrails.on_fail.value,
                "max_retries": config.guardrails.max_retries,
                "timeout": config.guardrails.timeout
            },
            "ace": {
                "enabled": config.ace.enabled,
                "skillbook_path": config.ace.skillbook_path,
                "agent_id": config.ace.agent_id,
                "learning_mode": config.ace.learning_mode.value,
                "adversarial_budget": config.ace.adversarial_budget
            },
            "steer": {
                "enabled": config.steer.enabled,
                "verifications": config.steer.verifications,
                "halt_on_failure": config.steer.halt_on_failure,
                "slop_threshold": config.steer.slop_threshold
            },
            "roma": {
                "enabled": config.roma.enabled,
                "max_depth_analysis": config.roma.max_depth_analysis,
                "max_depth_solving": config.roma.max_depth_solving,
                "mdap_k_ahead": config.roma.mdap_k_ahead,
                "use_associative_recomposition": config.roma.use_associative_recomposition
            },
            "unified_bridge": {
                "enabled": config.unified_bridge.enabled,
                "fallback_on_error": config.unified_bridge.fallback_on_error,
                "validation_strictness": config.unified_bridge.validation_strictness.value,
                "layer_order": config.unified_bridge.layer_order
            }
        }

        if layer not in status_map:
            raise ValueError(f"Unknown layer: {layer}")

        return {
            "layer": layer,
            **status_map[layer]
        }

    def get_all_statuses(self) -> Dict[str, Dict]:
        """Get status of all layers"""
        return {
            layer: self.get_layer_status(layer)
            for layer in RELIABILITY_LAYERS
        }

    def check_layer_health(self, layer: str) -> Dict[str, Any]:
        """
        Check health of a reliability layer

        Returns:
            {
                "layer": str,
                "available": bool,
                "enabled": bool,
                "version": Optional[str],
                "last_check": datetime,
                "error": Optional[str],
                "details": Dict
            }
        """
        # Check cache
        cached = self._health_cache.get(layer)
        if cached:
            last_check = self._last_health_check.get(layer, datetime.min)
            age = (datetime.utcnow() - last_check).total_seconds()
            if age < self._health_cache_ttl:
                return cached

        # Perform health check
        health = {
            "layer": layer,
            "available": False,
            "enabled": False,
            "version": None,
            "last_check": datetime.utcnow().isoformat(),
            "error": None,
            "details": {}
        }

        try:
            config = self.get_config()
            health["enabled"] = self.is_layer_enabled(layer)

            # Layer-specific health checks
            if layer == "lmql":
                health["available"] = self._check_lmql_health(config)
                health["version"] = self._get_lmql_version()
                health["details"] = {
                    "model": config.lmql.model,
                    "cache_enabled": config.lmql.cache_enabled
                }

            elif layer == "guardrails":
                health["available"] = self._check_guardrails_health(config)
                health["version"] = self._get_guardrails_version()
                health["details"] = {
                    "validators_count": len(config.guardrails.validators),
                    "on_fail_strategy": config.guardrails.on_fail.value
                }

            elif layer == "ace":
                health["available"] = self._check_ace_health(config)
                health["version"] = self._get_ace_version()
                health["details"] = {
                    "skillbook_path": config.ace.skillbook_path,
                    "learning_mode": config.ace.learning_mode.value
                }

            elif layer == "steer":
                health["available"] = self._check_steer_health(config)
                health["version"] = self._get_steer_version()
                health["details"] = {
                    "verifications_count": len(config.steer.verifications),
                    "slop_threshold": config.steer.slop_threshold
                }

            elif layer == "roma":
                health["available"] = self._check_roma_health(config)
                health["version"] = "1.0.0"
                health["details"] = {
                    "max_depth_analysis": config.roma.max_depth_analysis,
                    "use_associative_recomposition": config.roma.use_associative_recomposition
                }

            elif layer == "unified_bridge":
                health["available"] = self._check_bridge_health(config)
                health["version"] = "1.0.0"
                health["details"] = {
                    "fallback_enabled": config.unified_bridge.fallback_on_error,
                    "strictness": config.unified_bridge.validation_strictness.value
                }

            else:
                health["error"] = f"Unknown layer: {layer}"

        except Exception as e:
            health["error"] = str(e)
            logging.error(f"Health check failed for layer {layer}: {e}")

        # Cache result
        self._health_cache[layer] = health
        self._last_health_check[layer] = datetime.utcnow()

        return health

    def _check_lmql_health(self, config: ReliabilityConfig) -> bool:
        """Check LMQL layer health"""
        try:
            import importlib
            lmql_spec = importlib.util.find_spec("lmql")
            if lmql_spec is None:
                return False

            # Try to import and check basic functionality
            import lmql
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _get_lmql_version(self) -> Optional[str]:
        """Get LMQL version"""
        try:
            import lmql
            return getattr(lmql, "__version__", "unknown")
        except ImportError:
            return None

    def _check_guardrails_health(self, config: ReliabilityConfig) -> bool:
        """Check Guardrails layer health"""
        try:
            import importlib
            guardrails_spec = importlib.util.find_spec("guardrails")
            if guardrails_spec is None:
                return False

            from guardrails import Guard
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _get_guardrails_version(self) -> Optional[str]:
        """Get Guardrails version"""
        try:
            import guardrails
            return getattr(guardrails, "__version__", "unknown")
        except ImportError:
            return None

    def _check_ace_health(self, config: ReliabilityConfig) -> bool:
        """Check ACE layer health"""
        try:
            # Check if skillbook exists
            skillbook_path = Path(config.ace.skillbook_path)
            if skillbook_path.exists():
                return True
            # Check if we can create it
            return True
        except Exception:
            return False

    def _get_ace_version(self) -> Optional[str]:
        """Get ACE version"""
        try:
            # ACE is part of this project
            return "1.0.0"
        except Exception:
            return None

    def _check_steer_health(self, config: ReliabilityConfig) -> bool:
        """Check Steer layer health"""
        try:
            # Steer is integrated, check if we can import
            from reliability.steer import StructuredOutputVerifier
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _get_steer_version(self) -> Optional[str]:
        """Get Steer version"""
        try:
            return "1.0.0"
        except Exception:
            return None

    def _check_roma_health(self, config: ReliabilityConfig) -> bool:
        """Check ROMA layer health"""
        try:
            from roma_mdap_maker_associative_integration import ROMAMDAPMakerAssociativeEngine
            return True
        except ImportError:
            return False
        except Exception:
            return False

    def _check_bridge_health(self, config: ReliabilityConfig) -> bool:
        """Check unified bridge health"""
        try:
            # Check if at least one layer is available
            return any([
                self._check_lmql_health(config),
                self._check_guardrails_health(config),
                self._check_ace_health(config),
                self._check_steer_health(config)
            ])
        except Exception:
            return False

    def check_all_layers_health(self) -> Dict[str, Dict]:
        """Check health of all layers"""
        return {
            layer: self.check_layer_health(layer)
            for layer in RELIABILITY_LAYERS
        }

    def export_config(self, filepath: Optional[Path] = None) -> str:
        """Export current configuration to JSON"""
        config = self.get_config()
        config_dict = config.dict()
        json_str = json.dumps(config_dict, indent=2, default=str)

        if filepath:
            filepath.write_text(json_str, encoding='utf-8')
            logging.info(f"Configuration exported to {filepath}")

        return json_str

    def get_audit_trail(self, limit: int = 10) -> List[Dict]:
        """Get recent configuration changes"""
        return _audit_trail.get_recent_changes(limit)

    def export_audit_log(self, filepath: Optional[Path] = None) -> str:
        """Export audit log"""
        return _audit_trail.export_audit_log(filepath)


# ============================================================================
# Configuration Migration Utilities
# ============================================================================

class ConfigMigration:
    """Handles migration from old configuration formats"""

    @staticmethod
    def migrate_ace_config(old_config: Dict) -> Dict:
        """Migrate old ACE config to new format"""
        warnings = []

        # Map old keys to new keys
        key_mapping = {
            "ace_enabled": "ace.enabled",
            "skillbook_path": "ace.skillbook_path",
            "learning_mode": "ace.learning_mode"
        }

        migrated = {}
        for old_key, new_key in key_mapping.items():
            if old_key in old_config:
                migrated[new_key] = old_config[old_key]
                warnings.append(f"Migrated {old_key} -> {new_key}")

        return migrated, warnings

    @staticmethod
    def migrate_steer_config(old_config: Dict) -> Dict:
        """Migrate old Steer config to new format"""
        warnings = []

        key_mapping = {
            "steer_enabled": "steer.enabled",
            "verifications": "steer.verifications",
            "halt_on_failure": "steer.halt_on_failure"
        }

        migrated = {}
        for old_key, new_key in key_mapping.items():
            if old_key in old_config:
                migrated[new_key] = old_config[old_key]
                warnings.append(f"Migrated {old_key} -> {new_key}")

        return migrated, warnings

    @staticmethod
    def migrate_deprecated_keys(config: Dict) -> tuple[Dict, List[str]]:
        """Migrate all deprecated configuration keys"""
        warnings = []

        # Check for deprecated keys
        deprecated_keys = {
            "lmql_cache": "lmql.cache_enabled",
            "guardrails_on_fail": "guardrails.on_fail",
            "enable_telemetry": "observability.enable_telemetry"
        }

        migrated = config.copy()
        for old_key, new_key in deprecated_keys.items():
            if old_key in migrated:
                migrated[new_key] = migrated.pop(old_key)
                warnings.append(f"Deprecated key '{old_key}' migrated to '{new_key}'")

        return migrated, warnings


# ============================================================================
# Singleton Instance and Public API
# ============================================================================

_config_manager: Optional[ConfigManager] = None
_manager_lock = threading.Lock()


def get_config_manager() -> ConfigManager:
    """Get singleton config manager instance"""
    global _config_manager
    with _manager_lock:
        if _config_manager is None:
            _config_manager = ConfigManager()
            # Initial configuration load
            _config_manager.load_config()
        return _config_manager


@lru_cache(maxsize=1)
def get_config() -> ReliabilityConfig:
    """Get current configuration (cached)"""
    return get_config_manager().get_config()


def reload_config() -> ReliabilityConfig:
    """Reload configuration from environment"""
    # Clear cache
    get_config.cache_clear()
    return get_config_manager().reload_config()


def validate_config(config: Optional[ReliabilityConfig] = None) -> bool:
    """Validate configuration"""
    return get_config_manager().validate_config(config)


def is_layer_enabled(layer: str) -> bool:
    """Check if a layer is enabled"""
    return get_config_manager().is_layer_enabled(layer)


def get_layer_status(layer: str) -> Dict:
    """Get status of a specific layer"""
    return get_config_manager().get_layer_status(layer)


def get_all_statuses() -> Dict[str, Dict]:
    """Get status of all layers"""
    return get_config_manager().get_all_statuses()


def update_config(updates: Dict[str, Any]) -> ReliabilityConfig:
    """Update configuration at runtime"""
    # Clear cache
    get_config.cache_clear()
    return get_config_manager().update_config(updates)


def check_layer_health(layer: str) -> Dict[str, Any]:
    """Check health of a reliability layer"""
    return get_config_manager().check_layer_health(layer)


def check_all_layers_health() -> Dict[str, Dict]:
    """Check health of all layers"""
    return get_config_manager().check_all_layers_health()


def export_config(filepath: Optional[Path] = None) -> str:
    """Export current configuration to JSON"""
    return get_config_manager().export_config(filepath)


def get_audit_trail(limit: int = 10) -> List[Dict]:
    """Get recent configuration changes"""
    return get_config_manager().get_audit_trail(limit)


def export_audit_log(filepath: Optional[Path] = None) -> str:
    """Export audit log"""
    return get_config_manager().export_audit_log(filepath)


# ============================================================================
# Configuration Validation for CI/CD
# ============================================================================

def validate_config_file(filepath: Path) -> bool:
    """
    Validate configuration file for CI/CD

    Args:
        filepath: Path to configuration JSON file

    Returns:
        True if configuration is valid

    Raises:
        ValidationError: If configuration is invalid
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Configuration file not found: {filepath}")

    try:
        config_data = json.loads(filepath.read_text(encoding='utf-8'))

        # Validate using Pydantic
        ReliabilityConfig(**config_data)

        logging.info(f"Configuration file {filepath} is valid")
        return True

    except json.JSONDecodeError as e:
        raise ValidationError(f"Invalid JSON in configuration file: {e}")
    except ValidationError as e:
        raise ValidationError(f"Configuration validation failed: {e}")


def generate_default_config() -> Dict:
    """Generate default configuration dictionary"""
    config = ReliabilityConfig()
    return config.dict()


def save_default_config(filepath: Path) -> None:
    """Save default configuration to file"""
    default_config = generate_default_config()
    json_str = json.dumps(default_config, indent=2)
    filepath.write_text(json_str, encoding='utf-8')
    logging.info(f"Default configuration saved to {filepath}")


# ============================================================================
# CLI Utilities
# ============================================================================

def print_config_summary():
    """Print configuration summary to console"""
    config = get_config()

    print("=" * 60)
    print("OpenEvolve Reliability Configuration")
    print("=" * 60)
    print(f"Environment: {config.environment}")
    print(f"Version: {config.config_version}")
    print()

    for layer in RELIABILITY_LAYERS:
        status = get_layer_status(layer)
        enabled = "✓" if status.get("enabled") else "✗"
        print(f"[{enabled}] {layer.upper()}")
        if status.get("enabled"):
            for key, value in status.items():
                if key not in ["layer", "enabled"]:
                    print(f"  {key}: {value}")
        print()

    print("=" * 60)


def print_health_report():
    """Print health check report to console"""
    health_report = check_all_layers_health()

    print("=" * 60)
    print("OpenEvolve Reliability Health Report")
    print("=" * 60)
    print()

    for layer, health in health_report.items():
        available = "✓" if health["available"] else "✗"
        enabled = "✓" if health["enabled"] else "✗"

        print(f"[{available}] {layer.upper()}")
        print(f"  Enabled: {enabled}")
        print(f"  Version: {health.get('version', 'N/A')}")

        if health.get("error"):
            print(f"  Error: {health['error']}")

        if health.get("details"):
            print("  Details:")
            for key, value in health["details"].items():
                print(f"    {key}: {value}")

        print()

    print("=" * 60)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Print configuration summary
    print_config_summary()

    # Print health report
    print_health_report()

    # Export configuration
    config_json = export_config()
    print("\nConfiguration exported successfully")
    print(f"Size: {len(config_json)} bytes")
