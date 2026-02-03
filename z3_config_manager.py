"""
Z3 Integration Configuration Manager

Unified configuration management for the entire Z3-LeanAIDE-OpenEvolve-BubbleLabs integration.

Features:
- YAML/JSON configuration loading
- Environment variable substitution
- Configuration validation
- Runtime configuration updates
- Configuration versioning
- Secrets management

Author: OpenEvolve
Created: 2026-01-31
"""


import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Try to import YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logger.warning("PyYAML not available - YAML configuration disabled")


# =============================================================================
# Configuration Data Classes
# =============================================================================

@dataclass
class Z3Config:
    """Z3 solver configuration."""
    enabled: bool = True
    timeout: float = 60.0
    memory_limit_mb: int = 8192
    num_threads: int = 4
    proof_generation: bool = True
    tactics: Dict[str, str] = field(default_factory=dict)
    portfolio: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.tactics:
            self.tactics = {
                "default": "default",
                "quantifier": "qe",
                "arithmetic": "qfnra",
                "bitvector": "qfbv",
                "arrays": "qfauflia"
            }
        if not self.portfolio:
            self.portfolio = {
                "enabled": True,
                "strategies": ["default", "smt", "qflia", "qfnra"],
                "parallel": True,
                "max_workers": 4
            }


@dataclass
class LeanAideConfig:
    """LeanAIDE configuration."""
    enabled: bool = True
    host: str = "localhost"
    port: int = 7654
    timeout: float = 300.0
    max_retries: int = 3
    autoformalization: Dict[str, Any] = field(default_factory=dict)
    translation: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.autoformalization:
            self.autoformalization = {
                "enabled": True,
                "auto_detect_math": True,
                "confidence_threshold": 0.7,
                "require_formal_proof": False,
                "store_proofs": True
            }
        if not self.translation:
            self.translation = {
                "enable_smt_to_lean": True,
                "enable_lean_to_smt": True,
                "validate_translations": True
            }


@dataclass
class BridgeConfig:
    """Z3-LeanAIDE bridge configuration."""
    enabled: bool = True
    default_strategy: str = "adaptive"
    enable_cross_validation: bool = True
    confidence_threshold: float = 0.7
    use_z3_for_constraints: bool = True
    use_lean_for_theorems: bool = True
    use_parallel_for_critical: bool = True


@dataclass
class CacheConfig:
    """Caching configuration."""
    enabled: bool = True
    max_size: int = 10000
    default_ttl: float = 7200
    policy: str = "lru"
    persistent_storage: bool = True
    db_path: str = "./data/z3_cache.db"
    compression: bool = False
    checksum_verification: bool = True
    distributed: bool = False
    redis_host: Optional[str] = None
    redis_port: int = 6379
    redis_db: int = 0


@dataclass
class MonitoringConfig:
    """Performance monitoring configuration."""
    enabled: bool = True
    collection_interval: float = 10.0
    history_window: int = 3600
    thresholds: Dict[str, float] = field(default_factory=dict)
    alerts: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.thresholds:
            self.thresholds = {
                "solve_time_warning": 30.0,
                "solve_time_critical": 60.0,
                "error_rate_warning": 0.1,
                "error_rate_critical": 0.25,
                "memory_warning_mb": 1024.0,
                "memory_critical_mb": 2048.0,
                "queue_depth_warning": 10.0,
                "queue_depth_critical": 50.0
            }
        if not self.alerts:
            self.alerts = {
                "log_to_console": True,
                "log_to_file": True,
                "log_file": "./logs/alerts.log",
                "webhook_url": None
            }


@dataclass
class KnowledgeConfig:
    """Knowledge extraction configuration."""
    enabled: bool = True
    pattern_extraction: Dict[str, Any] = field(default_factory=dict)
    strategy_learning: Dict[str, Any] = field(default_factory=dict)
    storage: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.pattern_extraction:
            self.pattern_extraction = {
                "min_proof_length": 2,
                "max_pattern_length": 5,
                "min_confidence": 0.5
            }
        if not self.strategy_learning:
            self.strategy_learning = {
                "enabled": True,
                "min_successes": 3,
                "update_threshold": 0.1
            }
        if not self.storage:
            self.storage = {
                "db_path": "./data/knowledge.db",
                "auto_save": True,
                "save_interval": 300
            }


@dataclass
class SecurityConfig:
    """Security configuration."""
    api_key_required: bool = False
    api_keys: List[str] = field(default_factory=list)
    rate_limiting: Dict[str, Any] = field(default_factory=dict)
    cors: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.rate_limiting:
            self.rate_limiting = {
                "enabled": True,
                "requests_per_minute": 60,
                "burst_size": 10
            }
        if not self.cors:
            self.cors = {
                "enabled": True,
                "allowed_origins": ["http://localhost:3000", "http://localhost:8080"]
            }


@dataclass
class ServerConfig:
    """API server configuration."""
    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8765
    http: Dict[str, Any] = field(default_factory=dict)
    websocket: Dict[str, Any] = field(default_factory=dict)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    def __post_init__(self):
        if not self.http:
            self.http = {
                "max_request_size": 10485760,
                "timeout": 300.0,
                "keep_alive": True,
                "compression": True
            }
        if not self.websocket:
            self.websocket = {
                "enabled": True,
                "ping_interval": 20.0,
                "ping_timeout": 10.0,
                "max_connections": 1000
            }


@dataclass
class BubbleLabsConfig:
    """BubbleLabs integration configuration."""
    enabled: bool = True
    nodes: Dict[str, Any] = field(default_factory=dict)
    visualization: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.nodes:
            self.nodes = {
                "auto_register": True,
                "enable_advanced_visualizations": True,
                "refresh_interval": 1000
            }
        if not self.visualization:
            self.visualization = {
                "constraint_graphs": True,
                "proof_trees": True,
                "optimization_landscapes": True,
                "real_time_progress": True,
                "max_points_3d": 10000,
                "export_formats": ["json", "csv", "svg", "png"]
            }


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Dict[str, Any] = field(default_factory=dict)
    console: Dict[str, Any] = field(default_factory=dict)
    structured: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.file:
            self.file = {
                "enabled": True,
                "path": "./logs/z3_integration.log",
                "max_size_mb": 100,
                "backup_count": 5
            }
        if not self.console:
            self.console = {
                "enabled": True,
                "colorize": True
            }
        if not self.structured:
            self.structured = {
                "enabled": False,
                "path": "./logs/z3_integration.jsonl"
            }


@dataclass
class DatabaseConfig:
    """Database configuration."""
    type: str = "sqlite"
    sqlite: Dict[str, str] = field(default_factory=dict)
    postgresql: Dict[str, Any] = field(default_factory=dict)
    pool: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.sqlite:
            self.sqlite = {"path": "./data/z3_integration.db"}
        if not self.postgresql:
            self.postgresql = {
                "host": "localhost",
                "port": 5432,
                "database": "z3_integration",
                "username": "z3_user",
                "password": "${DB_PASSWORD}"
            }
        if not self.pool:
            self.pool = {
                "min_connections": 1,
                "max_connections": 10,
                "connection_timeout": 30.0
            }


@dataclass
class FeaturesConfig:
    """Feature flags configuration."""
    constraint_solving: bool = True
    optimization: bool = True
    theorem_proving: bool = True
    smt_translation: bool = True
    incremental_solving: bool = True
    portfolio_solving: bool = True
    proof_extraction: bool = True
    mcp_tools: bool = True
    crewai_agents: bool = True
    knowledge_extraction: bool = True
    experimental: Dict[str, bool] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.experimental:
            self.experimental = {
                "distributed_solving": False,
                "gpu_acceleration": False,
                "quantum_optimization": False
            }


@dataclass
class IntegrationConfig:
    """Complete integration configuration."""
    z3: Z3Config = field(default_factory=Z3Config)
    leanaide: LeanAideConfig = field(default_factory=LeanAideConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    knowledge: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    bubblelabs: BubbleLabsConfig = field(default_factory=BubbleLabsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    features: FeaturesConfig = field(default_factory=FeaturesConfig)
    development: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.development:
            self.development = {
                "debug": False,
                "reload": False,
                "profiling": {
                    "enabled": False,
                    "type": "cprofile",
                    "output_path": "./profiles/"
                },
                "testing": {
                    "mock_z3": False,
                    "mock_leanaide": False,
                    "test_timeout": 60.0
                }
            }


# =============================================================================
# Configuration Manager
# =============================================================================

class ConfigManager:
    """
    Centralized configuration management.
    
    Features:
    - Load from YAML/JSON
    - Environment variable substitution
    - Validation
    - Runtime updates
    - Secrets management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._find_config_file()
        self.config: IntegrationConfig = IntegrationConfig()
        self._lock = threading.RLock()
        self._validators: List[callable] = []
        
        # Load initial configuration
        if self.config_path:
            self.load()
    
    def _find_config_file(self) -> Optional[str]:
        """Find configuration file in standard locations."""
        search_paths = [
            "./z3_config.yaml",
            "./z3_config.json",
            "./config/z3_config.yaml",
            "./config/z3_config.json",
            "~/.config/z3_integration/config.yaml",
            "/etc/z3_integration/config.yaml"
        ]
        
        for path in search_paths:
            expanded_path = Path(path).expanduser()
            if expanded_path.exists():
                return str(expanded_path)
        
        return None
    
    def load(self, path: Optional[str] = None):
        """Load configuration from file."""
        path = path or self.config_path
        
        if not path:
            logger.warning("No configuration file specified, using defaults")
            return
        
        path = Path(path)
        
        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return
        
        try:
            content = path.read_text()
            
            # Substitute environment variables
            content = self._substitute_env_vars(content)
            
            # Parse based on extension
            if path.suffix in ['.yaml', '.yml']:
                if not YAML_AVAILABLE:
                    raise RuntimeError("PyYAML required for YAML config files")
                data = yaml.safe_load(content)
            elif path.suffix == '.json':
                data = json.loads(content)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")
            
            # Convert to dataclass
            self.config = self._dict_to_config(data)
            
            logger.info(f"Loaded configuration from {path}")
        
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute environment variables in content."""
        pattern = r'\$\{(\w+)(?::([^}]*))?\}'
        
        def replace(match):
            var_name = match.group(1)
            default_value = match.group(2)
            
            value = os.getenv(var_name, default_value)
            
            if value is None:
                logger.warning(f"Environment variable not set: {var_name}")
                return match.group(0)
            
            return value
        
        return re.sub(pattern, replace, content)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> IntegrationConfig:
        """Convert dictionary to configuration dataclass."""
        # Build configuration from nested dictionaries
        config = IntegrationConfig()
        
        if 'z3' in data:
            config.z3 = Z3Config(**data['z3'])
        if 'leanaide' in data:
            config.leanaide = LeanAideConfig(**data['leanaide'])
        if 'bridge' in data:
            config.bridge = BridgeConfig(**data['bridge'])
        if 'cache' in data:
            config.cache = CacheConfig(**data['cache'])
        if 'monitoring' in data:
            config.monitoring = MonitoringConfig(**data['monitoring'])
        if 'knowledge' in data:
            config.knowledge = KnowledgeConfig(**data['knowledge'])
        if 'server' in data:
            server_data = data['server']
            if 'security' in server_data:
                server_data['security'] = SecurityConfig(**server_data['security'])
            config.server = ServerConfig(**server_data)
        if 'bubblelabs' in data:
            config.bubblelabs = BubbleLabsConfig(**data['bubblelabs'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        if 'database' in data:
            config.database = DatabaseConfig(**data['database'])
        if 'features' in data:
            config.features = FeaturesConfig(**data['features'])
        if 'development' in data:
            config.development = data['development']
        
        return config
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file."""
        path = path or self.config_path
        
        if not path:
            raise ValueError("No configuration path specified")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        data = asdict(self.config)
        
        # Write based on extension
        if path.suffix in ['.yaml', '.yml']:
            if not YAML_AVAILABLE:
                raise RuntimeError("PyYAML required for YAML config files")
            content = yaml.dump(data, default_flow_style=False, sort_keys=False)
        elif path.suffix == '.json':
            content = json.dumps(data, indent=2)
        else:
            raise ValueError(f"Unsupported config format: {path.suffix}")
        
        path.write_text(content)
        logger.info(f"Saved configuration to {path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key path (e.g., 'z3.timeout')."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if hasattr(value, k):
                value = getattr(value, k)
            elif isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value by key path."""
        with self._lock:
            keys = key.split('.')
            target = self.config
            
            for k in keys[:-1]:
                if hasattr(target, k):
                    target = getattr(target, k)
                elif isinstance(target, dict) and k in target:
                    target = target[k]
                else:
                    raise KeyError(f"Invalid config path: {key}")
            
            if hasattr(target, keys[-1]):
                setattr(target, keys[-1], value)
            elif isinstance(target, dict):
                target[keys[-1]] = value
            else:
                raise KeyError(f"Invalid config path: {key}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Validate Z3 config
        if self.config.z3.timeout <= 0:
            errors.append("z3.timeout must be positive")
        
        # Validate server config
        if self.config.server.port < 1 or self.config.server.port > 65535:
            errors.append("server.port must be between 1 and 65535")
        
        # Run custom validators
        for validator in self._validators:
            try:
                result = validator(self.config)
                if result:
                    errors.extend(result if isinstance(result, list) else [result])
            except Exception as e:
                errors.append(f"Validator error: {e}")
        
        return errors
    
    def register_validator(self, validator: callable):
        """Register custom configuration validator."""
        self._validators.append(validator)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self.config)
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-style access."""
        return self.get(key)
    
    def __setitem__(self, key: str, value: Any):
        """Allow dict-style setting."""
        self.set(key, value)


# =============================================================================
# Global Instance
# =============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def load_config(path: Optional[str] = None) -> IntegrationConfig:
    """Load configuration from file."""
    return get_config_manager(path).config


# =============================================================================
# Example Usage
# =============================================================================

def example_config_usage():
    """Example: Configuration usage."""
    # Create default config
    manager = ConfigManager()
    
    # Access values
    print(f"Z3 timeout: {manager.get('z3.timeout')}")
    print(f"Server port: {manager.get('server.port')}")
    
    # Set values
    manager.set('z3.timeout', 120.0)
    
    # Validate
    errors = manager.validate()
    if errors:
        print(f"Validation errors: {errors}")
    
    # Convert to dict
    config_dict = manager.to_dict()
    print(f"Config keys: {list(config_dict.keys())}")


if __name__ == "__main__":
    print("Z3 Configuration Manager")
    print("=" * 50)
    example_config_usage()
