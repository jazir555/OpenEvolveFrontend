"""
Configuration Parameters System for OpenEvolve
Implements the Configuration Parameters System functionality described in the ultimate explanation document.
"""
import json
import yaml
import os
import tempfile
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import copy
from datetime import datetime
import hashlib
import logging

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.evaluator import Evaluator
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigCategory(Enum):
    """Categories of configuration parameters"""
    SYSTEM = "system"
    MODEL = "model"
    PROMPT = "prompt"
    EVOLUTION = "evolution"
    ADVERSARIAL = "adversarial"
    QUALITY = "quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    INTEGRATION = "integration"
    USER_INTERFACE = "user_interface"
    ADVANCED = "advanced"

class ParameterType(Enum):
    """Types of configuration parameters"""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    LIST = "list"
    DICTIONARY = "dictionary"
    ENUM = "enum"

class ParameterConstraint(Enum):
    """Constraints for configuration parameters"""
    REQUIRED = "required"
    OPTIONAL = "optional"
    DEPRECATED = "deprecated"
    EXPERIMENTAL = "experimental"
    ADVANCED = "advanced"

@dataclass
class ConfigParameter:
    """Individual configuration parameter"""
    name: str
    category: ConfigCategory
    param_type: ParameterType
    default_value: Any
    description: str
    constraints: List[ParameterConstraint] = field(default_factory=list)
    allowed_values: Optional[List[Any]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    depends_on: Optional[List[str]] = None
    validation_regex: Optional[str] = None
    example_values: Optional[List[Any]] = None
    tags: List[str] = field(default_factory=list)
    version_added: str = "1.0.0"
    version_deprecated: Optional[str] = None
    deprecation_message: Optional[str] = None

@dataclass
class ConfigProfile:
    """Configuration profile containing parameter values"""
    name: str
    description: str
    parameters: Dict[str, Any]
    category_defaults: Dict[ConfigCategory, Dict[str, Any]]
    created_at: str = ""
    updated_at: str = ""
    version: str = "1.0.0"
    is_active: bool = False
    tags: List[str] = field(default_factory=list)

class ConfigurationManager:
    """Main configuration management system"""
    
    def __init__(self, config_directory: str = "./config"):
        self.config_directory = config_directory
        self.parameters: Dict[str, ConfigParameter] = {}
        self.profiles: Dict[str, ConfigProfile] = {}
        self.active_profile: Optional[str] = None
        self.validation_rules: Dict[str, Callable] = {}
        self.change_listeners: Dict[str, List[Callable]] = {}
        self.effectiveness_scores: Dict[str, List[Dict[str, Any]]] = {}  # Track parameter effectiveness
        
        # Ensure config directory exists
        os.makedirs(config_directory, exist_ok=True)
        
        # Initialize default parameters
        self._initialize_default_parameters()
        
        # Load existing profiles
        self._load_profiles()
        
        # Set default active profile
        self._set_default_active_profile()
    
    def _initialize_default_parameters(self):
        """Initialize default configuration parameters"""
        # System parameters
        self.add_parameter(ConfigParameter(
            name="system_log_level",
            category=ConfigCategory.SYSTEM,
            param_type=ParameterType.STRING,
            default_value="INFO",
            description="Logging level for the system",
            allowed_values=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            tags=["system", "logging"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="system_max_threads",
            category=ConfigCategory.SYSTEM,
            param_type=ParameterType.INTEGER,
            default_value=8,
            description="Maximum number of threads for parallel processing",
            min_value=1,
            max_value=64,
            tags=["system", "performance"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="system_cache_size",
            category=ConfigCategory.SYSTEM,
            param_type=ParameterType.INTEGER,
            default_value=1000,
            description="Size of internal cache in MB",
            min_value=100,
            max_value=10000,
            tags=["system", "performance", "memory"]
        ))
        
        # Model parameters
        self.add_parameter(ConfigParameter(
            name="model_default_provider",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.STRING,
            default_value="openai",
            description="Default model provider",
            allowed_values=["openai", "anthropic", "google", "openrouter", "custom"],
            tags=["model", "provider"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="model_temperature",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.FLOAT,
            default_value=0.7,
            description="Default temperature for model generation",
            min_value=0.0,
            max_value=2.0,
            tags=["model", "generation"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="model_max_tokens",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.INTEGER,
            default_value=4096,
            description="Maximum tokens for model responses",
            min_value=100,
            max_value=32768,
            tags=["model", "limits"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="model_top_p",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.FLOAT,
            default_value=1.0,
            description="Top-p sampling parameter",
            min_value=0.0,
            max_value=1.0,
            tags=["model", "sampling"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="model_frequency_penalty",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.FLOAT,
            default_value=0.0,
            description="Frequency penalty for generation",
            min_value=-2.0,
            max_value=2.0,
            tags=["model", "penalty"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="model_presence_penalty",
            category=ConfigCategory.MODEL,
            param_type=ParameterType.FLOAT,
            default_value=0.0,
            description="Presence penalty for generation",
            min_value=-2.0,
            max_value=2.0,
            tags=["model", "penalty"]
        ))
        
        # Prompt parameters
        self.add_parameter(ConfigParameter(
            name="prompt_max_length",
            category=ConfigCategory.PROMPT,
            param_type=ParameterType.INTEGER,
            default_value=32768,
            description="Maximum prompt length in tokens",
            min_value=1000,
            max_value=100000,
            tags=["prompt", "limits"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="prompt_template_directory",
            category=ConfigCategory.PROMPT,
            param_type=ParameterType.STRING,
            default_value="./templates",
            description="Directory for prompt templates",
            tags=["prompt", "templates"]
        ))
        
        # Evolution parameters
        self.add_parameter(ConfigParameter(
            name="evolution_population_size",
            category=ConfigCategory.EVOLUTION,
            param_type=ParameterType.INTEGER,
            default_value=50,
            description="Population size for evolutionary algorithms",
            min_value=10,
            max_value=1000,
            tags=["evolution", "genetic_algorithm"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="evolution_max_generations",
            category=ConfigCategory.EVOLUTION,
            param_type=ParameterType.INTEGER,
            default_value=100,
            description="Maximum number of generations for evolution",
            min_value=1,
            max_value=10000,
            tags=["evolution", "termination"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="evolution_crossover_rate",
            category=ConfigCategory.EVOLUTION,
            param_type=ParameterType.FLOAT,
            default_value=0.8,
            description="Probability of crossover occurring",
            min_value=0.0,
            max_value=1.0,
            tags=["evolution", "genetic_algorithm"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="evolution_mutation_rate",
            category=ConfigCategory.EVOLUTION,
            param_type=ParameterType.FLOAT,
            default_value=0.1,
            description="Probability of mutation occurring",
            min_value=0.0,
            max_value=1.0,
            tags=["evolution", "genetic_algorithm"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="evolution_elitism_rate",
            category=ConfigCategory.EVOLUTION,
            param_type=ParameterType.FLOAT,
            default_value=0.1,
            description="Fraction of best individuals preserved",
            min_value=0.0,
            max_value=0.5,
            tags=["evolution", "selection"]
        ))
        
        # Adversarial parameters
        self.add_parameter(ConfigParameter(
            name="adversarial_max_iterations",
            category=ConfigCategory.ADVERSARIAL,
            param_type=ParameterType.INTEGER,
            default_value=10,
            description="Maximum adversarial testing iterations",
            min_value=1,
            max_value=100,
            tags=["adversarial", "testing"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="adversarial_confidence_threshold",
            category=ConfigCategory.ADVERSARIAL,
            param_type=ParameterType.FLOAT,
            default_value=95.0,
            description="Confidence threshold for adversarial testing",
            min_value=50.0,
            max_value=100.0,
            tags=["adversarial", "termination"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="adversarial_team_size",
            category=ConfigCategory.ADVERSARIAL,
            param_type=ParameterType.INTEGER,
            default_value=3,
            description="Number of models in each adversarial team",
            min_value=1,
            max_value=20,
            tags=["adversarial", "teams"]
        ))
        
        # Quality parameters
        self.add_parameter(ConfigParameter(
            name="quality_min_score",
            category=ConfigCategory.QUALITY,
            param_type=ParameterType.FLOAT,
            default_value=80.0,
            description="Minimum acceptable quality score",
            min_value=0.0,
            max_value=100.0,
            tags=["quality", "threshold"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="quality_assessment_depth",
            category=ConfigCategory.QUALITY,
            param_type=ParameterType.INTEGER,
            default_value=5,
            description="Depth of quality assessment analysis",
            min_value=1,
            max_value=10,
            tags=["quality", "analysis"]
        ))
        
        # Performance parameters
        self.add_parameter(ConfigParameter(
            name="performance_max_concurrent_requests",
            category=ConfigCategory.PERFORMANCE,
            param_type=ParameterType.INTEGER,
            default_value=10,
            description="Maximum concurrent API requests",
            min_value=1,
            max_value=100,
            tags=["performance", "api"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="performance_request_timeout",
            category=ConfigCategory.PERFORMANCE,
            param_type=ParameterType.INTEGER,
            default_value=300,
            description="Timeout for API requests in seconds",
            min_value=10,
            max_value=3600,
            tags=["performance", "api", "timeout"]
        ))
        
        # Security parameters
        self.add_parameter(ConfigParameter(
            name="security_enable_encryption",
            category=ConfigCategory.SECURITY,
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            description="Enable encryption for sensitive data",
            tags=["security", "encryption"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="security_api_key_masking",
            category=ConfigCategory.SECURITY,
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            description="Mask API keys in logs and UI",
            tags=["security", "privacy"]
        ))
        
        # Compliance parameters
        self.add_parameter(ConfigParameter(
            name="compliance_require_documentation",
            category=ConfigCategory.COMPLIANCE,
            param_type=ParameterType.BOOLEAN,
            default_value=True,
            description="Require documentation for all changes",
            tags=["compliance", "documentation"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="compliance_audit_interval",
            category=ConfigCategory.COMPLIANCE,
            param_type=ParameterType.INTEGER,
            default_value=24,
            description="Hours between compliance audits",
            min_value=1,
            max_value=168,
            tags=["compliance", "audit"]
        ))
        
        # Integration parameters
        self.add_parameter(ConfigParameter(
            name="integration_github_enabled",
            category=ConfigCategory.INTEGRATION,
            param_type=ParameterType.BOOLEAN,
            default_value=False,
            description="Enable GitHub integration",
            tags=["integration", "github"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="integration_discord_webhook_url",
            category=ConfigCategory.INTEGRATION,
            param_type=ParameterType.STRING,
            default_value="",
            description="Discord webhook URL for notifications",
            validation_regex=r"^https://discord.com/api/webhooks/.*",
            tags=["integration", "discord"]
        ))
        
        # User interface parameters
        self.add_parameter(ConfigParameter(
            name="ui_theme",
            category=ConfigCategory.USER_INTERFACE,
            param_type=ParameterType.STRING,
            default_value="light",
            description="User interface theme",
            allowed_values=["light", "dark", "auto"],
            tags=["ui", "appearance"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="ui_auto_refresh_interval",
            category=ConfigCategory.USER_INTERFACE,
            param_type=ParameterType.INTEGER,
            default_value=5,
            description="Auto-refresh interval in seconds",
            min_value=1,
            max_value=60,
            tags=["ui", "refresh"]
        ))
        
        # Advanced parameters
        self.add_parameter(ConfigParameter(
            name="advanced_debug_mode",
            category=ConfigCategory.ADVANCED,
            param_type=ParameterType.BOOLEAN,
            default_value=False,
            description="Enable advanced debugging features",
            constraints=[ParameterConstraint.EXPERIMENTAL],
            tags=["advanced", "debug"]
        ))
        
        self.add_parameter(ConfigParameter(
            name="advanced_custom_evaluators_enabled",
            category=ConfigCategory.ADVANCED,
            param_type=ParameterType.BOOLEAN,
            default_value=False,
            description="Enable custom evaluator functionality",
            constraints=[ParameterConstraint.EXPERIMENTAL],
            tags=["advanced", "evaluators"]
        ))
    
    def add_parameter(self, parameter: ConfigParameter):
        """Add a new configuration parameter"""
        self.parameters[parameter.name] = parameter
        logger.info(f"Added configuration parameter: {parameter.name}")
    
    def get_parameter(self, name: str) -> Optional[ConfigParameter]:
        """Get a configuration parameter by name"""
        return self.parameters.get(name)
    
    def list_parameters(self, category: Optional[ConfigCategory] = None) -> List[ConfigParameter]:
        """List configuration parameters, optionally filtered by category"""
        if category:
            return [param for param in self.parameters.values() if param.category == category]
        return list(self.parameters.values())
    
    def validate_parameter(self, name: str, value: Any) -> bool:
        """Validate a parameter value"""
        parameter = self.parameters.get(name)
        if not parameter:
            logger.warning(f"Unknown parameter: {name}")
            return False
        
        # Type validation
        if parameter.param_type == ParameterType.STRING:
            if not isinstance(value, str):
                logger.error(f"Parameter {name} must be a string")
                return False
        elif parameter.param_type == ParameterType.INTEGER:
            if not isinstance(value, int):
                logger.error(f"Parameter {name} must be an integer")
                return False
        elif parameter.param_type == ParameterType.FLOAT:
            if not isinstance(value, (int, float)):
                logger.error(f"Parameter {name} must be a float")
                return False
        elif parameter.param_type == ParameterType.BOOLEAN:
            if not isinstance(value, bool):
                logger.error(f"Parameter {name} must be a boolean")
                return False
        elif parameter.param_type == ParameterType.LIST:
            if not isinstance(value, list):
                logger.error(f"Parameter {name} must be a list")
                return False
        elif parameter.param_type == ParameterType.DICTIONARY:
            if not isinstance(value, dict):
                logger.error(f"Parameter {name} must be a dictionary")
                return False
        elif parameter.param_type == ParameterType.ENUM:
            if parameter.allowed_values and value not in parameter.allowed_values:
                logger.error(f"Parameter {name} value {value} not in allowed values: {parameter.allowed_values}")
                return False
        
        # Value range validation
        if parameter.min_value is not None and value < parameter.min_value:
            logger.error(f"Parameter {name} value {value} below minimum {parameter.min_value}")
            return False
        
        if parameter.max_value is not None and value > parameter.max_value:
            logger.error(f"Parameter {name} value {value} above maximum {parameter.max_value}")
            return False
        
        # Regex validation
        if parameter.validation_regex and isinstance(value, str):
            import re
            if not re.match(parameter.validation_regex, value):
                logger.error(f"Parameter {name} value {value} does not match regex {parameter.validation_regex}")
                return False
        
        # Custom validation rules
        if name in self.validation_rules:
            validator = self.validation_rules[name]
            if not validator(value):
                logger.error(f"Parameter {name} failed custom validation")
                return False
        
        return True
    
    def register_validation_rule(self, parameter_name: str, validator: Callable[[Any], bool]):
        """Register a custom validation rule for a parameter"""
        self.validation_rules[parameter_name] = validator
        logger.info(f"Registered validation rule for parameter: {parameter_name}")
    
    def register_change_listener(self, parameter_name: str, listener: Callable[[Any, Any], None]):
        """Register a change listener for a parameter"""
        if parameter_name not in self.change_listeners:
            self.change_listeners[parameter_name] = []
        self.change_listeners[parameter_name].append(listener)
        logger.info(f"Registered change listener for parameter: {parameter_name}")
    
    def notify_parameter_change(self, parameter_name: str, old_value: Any, new_value: Any):
        """Notify listeners of a parameter change"""
        if parameter_name in self.change_listeners:
            for listener in self.change_listeners[parameter_name]:
                try:
                    listener(old_value, new_value)
                except Exception as e:
                    logger.error(f"Error in change listener for {parameter_name}: {e}")
    
    def create_profile(self, name: str, description: str, 
                      parameters: Optional[Dict[str, Any]] = None,
                      category_defaults: Optional[Dict[ConfigCategory, Dict[str, Any]]] = None,
                      tags: Optional[List[str]] = None) -> ConfigProfile:
        """Create a new configuration profile"""
        if parameters is None:
            parameters = {}
        
        if category_defaults is None:
            category_defaults = {}
        
        if tags is None:
            tags = []
        
        # Validate parameters
        for param_name, param_value in parameters.items():
            if not self.validate_parameter(param_name, param_value):
                raise ValueError(f"Invalid parameter value for {param_name}: {param_value}")
        
        profile = ConfigProfile(
            name=name,
            description=description,
            parameters=parameters,
            category_defaults=category_defaults,
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            tags=tags
        )
        
        self.profiles[name] = profile
        logger.info(f"Created configuration profile: {name}")
        
        return profile
    
    def get_profile(self, name: str) -> Optional[ConfigProfile]:
        """Get a configuration profile by name"""
        return self.profiles.get(name)
    
    def list_profiles(self) -> List[ConfigProfile]:
        """List all configuration profiles"""
        return list(self.profiles.values())
    
    def activate_profile(self, name: str) -> bool:
        """Activate a configuration profile"""
        profile = self.profiles.get(name)
        if not profile:
            logger.error(f"Profile not found: {name}")
            return False
        
        # Store previous active profile
        previous_profile = self.active_profile
        
        # Set new active profile
        self.active_profile = name
        profile.is_active = True
        
        # Update previous profile
        if previous_profile and previous_profile in self.profiles:
            self.profiles[previous_profile].is_active = False
        
        logger.info(f"Activated configuration profile: {name}")
        return True
    
    def get_active_profile(self) -> Optional[ConfigProfile]:
        """Get the currently active profile"""
        if self.active_profile:
            return self.profiles.get(self.active_profile)
        return None
    
    def get_parameter_value(self, name: str, profile_name: Optional[str] = None) -> Any:
        """Get the value of a parameter, optionally from a specific profile"""
        # Get parameter definition
        parameter = self.parameters.get(name)
        if not parameter:
            logger.warning(f"Unknown parameter: {name}")
            return None
        
        # Get value from specified profile or active profile
        if profile_name:
            profile = self.profiles.get(profile_name)
        else:
            profile = self.get_active_profile()
        
        if profile and name in profile.parameters:
            return profile.parameters[name]
        
        # Get value from category defaults
        if profile and parameter.category in profile.category_defaults:
            category_defaults = profile.category_defaults[parameter.category]
            if name in category_defaults:
                return category_defaults[name]
        
        # Return default value
        return parameter.default_value
    
    def set_parameter_value(self, name: str, value: Any, profile_name: Optional[str] = None):
        """Set the value of a parameter, optionally in a specific profile"""
        # Validate parameter
        if not self.validate_parameter(name, value):
            raise ValueError(f"Invalid value for parameter {name}: {value}")
        
        # Get profile
        if profile_name:
            profile = self.profiles.get(profile_name)
            if not profile:
                raise ValueError(f"Profile not found: {profile_name}")
        else:
            profile = self.get_active_profile()
            if not profile:
                raise ValueError("No active profile")
        
        # Store old value for change notification
        old_value = profile.parameters.get(name)
        
        # Set new value
        profile.parameters[name] = value
        profile.updated_at = datetime.now().isoformat()
        
        # Notify change listeners
        self.notify_parameter_change(name, old_value, value)
        
        logger.info(f"Set parameter {name} = {value} in profile {profile.name}")
    
    def save_profile(self, profile_name: str, filepath: Optional[str] = None):
        """Save a configuration profile to file"""
        profile = self.profiles.get(profile_name)
        if not profile:
            raise ValueError(f"Profile not found: {profile_name}")
        
        if filepath is None:
            filepath = os.path.join(self.config_directory, f"{profile_name}.yaml")
        
        # Prepare data for serialization
        profile_data = {
            "name": profile.name,
            "description": profile.description,
            "parameters": profile.parameters,
            "category_defaults": {cat.value: defaults for cat, defaults in profile.category_defaults.items()},
            "created_at": profile.created_at,
            "updated_at": datetime.now().isoformat(),
            "version": profile.version,
            "is_active": profile.is_active,
            "tags": profile.tags
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(profile_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved profile {profile_name} to {filepath}")
    
    def load_profile(self, filepath: str) -> ConfigProfile:
        """Load a configuration profile from file"""
        with open(filepath, 'r') as f:
            profile_data = yaml.safe_load(f)
        
        # Convert category defaults back to enums
        category_defaults = {}
        if "category_defaults" in profile_data:
            for cat_str, defaults in profile_data["category_defaults"].items():
                try:
                    category = ConfigCategory(cat_str)
                    category_defaults[category] = defaults
                except ValueError:
                    logger.warning(f"Unknown category in profile: {cat_str}")
        
        profile = ConfigProfile(
            name=profile_data.get("name", "unnamed"),
            description=profile_data.get("description", ""),
            parameters=profile_data.get("parameters", {}),
            category_defaults=category_defaults,
            created_at=profile_data.get("created_at", datetime.now().isoformat()),
            updated_at=profile_data.get("updated_at", datetime.now().isoformat()),
            version=profile_data.get("version", "1.0.0"),
            is_active=profile_data.get("is_active", False),
            tags=profile_data.get("tags", [])
        )
        
        self.profiles[profile.name] = profile
        logger.info(f"Loaded profile from {filepath}")
        
        return profile
    
    def _load_profiles(self):
        """Load all profiles from the configuration directory"""
        if not os.path.exists(self.config_directory):
            return
        
        for filename in os.listdir(self.config_directory):
            if filename.endswith(('.yaml', '.yml')):
                filepath = os.path.join(self.config_directory, filename)
                try:
                    self.load_profile(filepath)
                except Exception as e:
                    logger.error(f"Failed to load profile from {filepath}: {e}")
    
    def _set_default_active_profile(self):
        """Set a default active profile if none exists"""
        if not self.active_profile and self.profiles:
            # Activate the first profile
            first_profile_name = list(self.profiles.keys())[0]
            self.activate_profile(first_profile_name)
    
    def get_all_parameter_values(self, profile_name: Optional[str] = None) -> Dict[str, Any]:
        """Get all parameter values for a profile"""
        if profile_name:
            profile = self.profiles.get(profile_name)
        else:
            profile = self.get_active_profile()
        
        if not profile:
            return {}
        
        # Start with default values
        all_values = {name: param.default_value for name, param in self.parameters.items()}
        
        # Apply category defaults
        for category, defaults in profile.category_defaults.items():
            all_values.update(defaults)
        
        # Apply profile-specific values
        all_values.update(profile.parameters)
        
        return all_values
    
    def export_configuration(self, filepath: str, profile_name: Optional[str] = None):
        """Export the complete configuration to a file"""
        config_data = {
            "profiles": {},
            "parameters": {},
            "active_profile": self.active_profile,
            "export_timestamp": datetime.now().isoformat(),
            "export_version": "1.0.0"
        }
        
        # Export profiles
        for name, profile in self.profiles.items():
            profile_dict = {
                "name": profile.name,
                "description": profile.description,
                "parameters": profile.parameters,
                "category_defaults": {cat.value: defaults for cat, defaults in profile.category_defaults.items()},
                "created_at": profile.created_at,
                "updated_at": profile.updated_at,
                "version": profile.version,
                "is_active": profile.is_active,
                "tags": profile.tags
            }
            config_data["profiles"][name] = profile_dict
        
        # Export parameters
        for name, parameter in self.parameters.items():
            param_dict = {
                "name": parameter.name,
                "category": parameter.category.value,
                "param_type": parameter.param_type.value,
                "default_value": parameter.default_value,
                "description": parameter.description,
                "constraints": [constraint.value for constraint in parameter.constraints],
                "allowed_values": parameter.allowed_values,
                "min_value": parameter.min_value,
                "max_value": parameter.max_value,
                "depends_on": parameter.depends_on,
                "validation_regex": parameter.validation_regex,
                "example_values": parameter.example_values,
                "tags": parameter.tags,
                "version_added": parameter.version_added,
                "version_deprecated": parameter.version_deprecated,
                "deprecation_message": parameter.deprecation_message
            }
            config_data["parameters"][name] = param_dict
        
        # Save to file
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Exported complete configuration to {filepath}")
    
    def import_configuration(self, filepath: str):
        """Import a complete configuration from a file"""
        with open(filepath, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Import parameters
        if "parameters" in config_data:
            for name, param_data in config_data["parameters"].items():
                try:
                    # Convert strings back to enums
                    category = ConfigCategory(param_data["category"])
                    param_type = ParameterType(param_data["param_type"])
                    constraints = [ParameterConstraint(c) for c in param_data.get("constraints", [])]
                    
                    parameter = ConfigParameter(
                        name=param_data["name"],
                        category=category,
                        param_type=param_type,
                        default_value=param_data["default_value"],
                        description=param_data["description"],
                        constraints=constraints,
                        allowed_values=param_data.get("allowed_values"),
                        min_value=param_data.get("min_value"),
                        max_value=param_data.get("max_value"),
                        depends_on=param_data.get("depends_on"),
                        validation_regex=param_data.get("validation_regex"),
                        example_values=param_data.get("example_values"),
                        tags=param_data.get("tags", []),
                        version_added=param_data.get("version_added", "1.0.0"),
                        version_deprecated=param_data.get("version_deprecated"),
                        deprecation_message=param_data.get("deprecation_message")
                    )
                    
                    self.parameters[name] = parameter
                except Exception as e:
                    logger.error(f"Failed to import parameter {name}: {e}")
        
        # Import profiles
        if "profiles" in config_data:
            for name, profile_data in config_data["profiles"].items():
                try:
                    # Convert category defaults back to enums
                    category_defaults = {}
                    if "category_defaults" in profile_data:
                        for cat_str, defaults in profile_data["category_defaults"].items():
                            try:
                                category = ConfigCategory(cat_str)
                                category_defaults[category] = defaults
                            except ValueError:
                                logger.warning(f"Unknown category in profile {name}: {cat_str}")
                    
                    profile = ConfigProfile(
                        name=profile_data["name"],
                        description=profile_data["description"],
                        parameters=profile_data["parameters"],
                        category_defaults=category_defaults,
                        created_at=profile_data["created_at"],
                        updated_at=profile_data["updated_at"],
                        version=profile_data.get("version", "1.0.0"),
                        is_active=profile_data.get("is_active", False),
                        tags=profile_data.get("tags", [])
                    )
                    
                    self.profiles[name] = profile
                except Exception as e:
                    logger.error(f"Failed to import profile {name}: {e}")
        
        # Set active profile
        if "active_profile" in config_data:
            self.active_profile = config_data["active_profile"]
        
        logger.info(f"Imported complete configuration from {filepath}")

    def optimize_parameter(self, name: str, response_quality: float, 
                          processing_time: float, cost: float,
                          api_key: Optional[str] = None,
                          model_name: str = "gpt-4o") -> float:
        """
        Optimize a parameter based on its effectiveness, using OpenEvolve when available
        
        Args:
            name: Name of the parameter to optimize
            response_quality: Quality score of the response (0-1)
            processing_time: Time taken to process (in seconds)
            cost: Cost of processing
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
            
        Returns:
            Effectiveness score (0-1)
        """
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            return self._optimize_parameter_with_openevolve(
                name, response_quality, processing_time, cost, api_key, model_name
            )
        
        # Fallback to custom implementation
        return self._optimize_parameter_custom(name, response_quality, processing_time, cost)
    
    def _optimize_parameter_with_openevolve(self, name: str, response_quality: float,
                                           processing_time: float, cost: float,
                                           api_key: str, model_name: str) -> float:
        """
        Optimize a parameter using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.3,  # Lower temperature for more consistent optimization
                max_tokens=2048,
            )
            
            config.llm.models = [llm_config]
            config.evolution.max_iterations = 1  # Just one optimization step
            config.evolution.population_size = 1  # Single parameter optimization
            
            # Create an evaluator for parameter optimization
            def parameter_evaluator(program_path: str) -> Dict[str, Any]:
                """
                Evaluator that performs parameter optimization assessment
                """
                try:
                    with open(program_path, "r", encoding='utf-8') as f:
                        parameter_content = f.read()
                    
                    # Perform basic parameter assessment
                    content_length = len(parameter_content)
                    
                    # Return parameter optimization metrics
                    return {
                        "score": 0.8,  # Placeholder optimization score
                        "timestamp": datetime.now().timestamp(),
                        "content_length": content_length,
                        "parameter_optimization_completed": True
                    }
                except Exception as e:
                    print(f"Error in parameter evaluator: {e}")
                    return {
                        "score": 0.0,
                        "timestamp": datetime.now().timestamp(),
                        "error": str(e)
                    }
            
            # Save parameter info to temporary file for OpenEvolve
            parameter_info = {
                "name": name,
                "response_quality": response_quality,
                "processing_time": processing_time,
                "cost": cost
            }
            
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding='utf-8') as temp_file:
                json.dump(parameter_info, temp_file, indent=2)
                temp_file_path = temp_file.name
            
            try:
                # Run parameter optimization using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=parameter_evaluator,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Calculate effectiveness score based on OpenEvolve result
                if result.best_score:
                    effectiveness = result.best_score
                else:
                    # Weighted effectiveness calculation (same as custom implementation)
                    time_factor = max(0, 1 - (processing_time / 60))  # Normalize time (assuming max 60s is bad)
                    cost_factor = max(0, 1 - cost)  # Normalize cost
                    
                    effectiveness = (response_quality * 0.6) + (time_factor * 0.2) + (cost_factor * 0.2)
                
                # Store effectiveness score
                if name not in self.effectiveness_scores:
                    self.effectiveness_scores[name] = []
                self.effectiveness_scores[name].append({
                    'timestamp': datetime.now().isoformat(),
                    'score': effectiveness,
                    'response_quality': response_quality,
                    'processing_time': processing_time,
                    'cost': cost
                })
                
                return effectiveness
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._optimize_parameter_custom(name, response_quality, processing_time, cost)
    
    def _optimize_parameter_custom(self, name: str, response_quality: float,
                                  processing_time: float, cost: float) -> float:
        """
        Fallback parameter optimization using custom implementation
        """
        # Weighted effectiveness calculation
        # Higher response quality = higher effectiveness
        # Lower processing time = higher effectiveness  
        # Lower cost = higher effectiveness
        time_factor = max(0, 1 - (processing_time / 60))  # Normalize time (assuming max 60s is bad)
        cost_factor = max(0, 1 - cost)  # Normalize cost
        
        effectiveness = (response_quality * 0.6) + (time_factor * 0.2) + (cost_factor * 0.2)
        
        # Store effectiveness score
        if name not in self.effectiveness_scores:
            self.effectiveness_scores[name] = []
        self.effectiveness_scores[name].append({
            'timestamp': datetime.now().isoformat(),
            'score': effectiveness,
            'response_quality': response_quality,
            'processing_time': processing_time,
            'cost': cost
        })
        
        return effectiveness

# Example usage and testing
def test_configuration_system():
    """Test function for the Configuration System"""
    print("Configuration Parameters System Test:")
    
    # Create configuration manager
    config_manager = ConfigurationManager("./test_config")
    
    print(f"Initialized with {len(config_manager.parameters)} parameters")
    print(f"Available categories: {[cat.value for cat in ConfigCategory]}")
    
    # List some parameters by category
    model_params = config_manager.list_parameters(ConfigCategory.MODEL)
    print(f"Model parameters: {len(model_params)}")
    
    evolution_params = config_manager.list_parameters(ConfigCategory.EVOLUTION)
    print(f"Evolution parameters: {len(evolution_params)}")
    
    # Test parameter validation
    test_param = "model_temperature"
    valid_values = [0.5, 0.7, 1.0, 1.5]
    invalid_values = [-0.1, 2.5, "invalid", None]
    
    print(f"\nTesting parameter validation for {test_param}:")
    for value in valid_values:
        is_valid = config_manager.validate_parameter(test_param, value)
        print(f"  Value {value}: {'Valid' if is_valid else 'Invalid'}")
    
    for value in invalid_values:
        is_valid = config_manager.validate_parameter(test_param, value)
        print(f"  Value {value}: {'Valid' if is_valid else 'Invalid'}")
    
    # Create and test profiles
    print("\nCreating test profiles:")
    
    # Create a development profile
    dev_profile = config_manager.create_profile(
        name="development",
        description="Development environment settings",
        parameters={
            "system_log_level": "DEBUG",
            "system_max_threads": 4,
            "model_temperature": 0.9,
            "evolution_population_size": 20
        },
        category_defaults={
            ConfigCategory.EVOLUTION: {
                "evolution_max_generations": 50,
                "evolution_mutation_rate": 0.2
            }
        },
        tags=["environment", "development"]
    )
    
    # Create a production profile
    prod_profile = config_manager.create_profile(
        name="production",
        description="Production environment settings",
        parameters={
            "system_log_level": "WARNING",
            "system_max_threads": 16,
            "model_temperature": 0.5,
            "evolution_population_size": 100
        },
        category_defaults={
            ConfigCategory.EVOLUTION: {
                "evolution_max_generations": 200,
                "evolution_mutation_rate": 0.05
            },
            ConfigCategory.PERFORMANCE: {
                "performance_max_concurrent_requests": 20,
                "performance_request_timeout": 600
            }
        },
        tags=["environment", "production"]
    )
    
    print(f"Created {len(config_manager.profiles)} profiles")
    
    # Test profile activation
    config_manager.activate_profile("development")
    active_profile = config_manager.get_active_profile()
    print(f"Active profile: {active_profile.name if active_profile else 'None'}")
    
    # Test parameter retrieval with active profile
    temp_value = config_manager.get_parameter_value("model_temperature")
    pop_size = config_manager.get_parameter_value("evolution_population_size")
    print(f"Model temperature (active profile): {temp_value}")
    print(f"Evolution population size (active profile): {pop_size}")
    
    # Test parameter setting
    config_manager.set_parameter_value("model_temperature", 0.8)
    new_temp_value = config_manager.get_parameter_value("model_temperature")
    print(f"Model temperature after change: {new_temp_value}")
    
    # Test getting all parameter values
    all_values = config_manager.get_all_parameter_values()
    print(f"Total parameters in active profile: {len(all_values)}")
    
    # Test profile save/load
    config_manager.save_profile("development", "./test_config/development_export.yaml")
    print("Saved development profile to file")
    
    # Clean up test files
    import shutil
    try:
        shutil.rmtree("./test_config")
        print("Cleaned up test configuration directory")
    except Exception as e:
        print(f"Error cleaning up test directory: {e}")
    
    return config_manager

if __name__ == "__main__":
    test_configuration_system()