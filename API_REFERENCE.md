# API Reference - OpenEvolve Unified Configuration System

Complete API documentation for the OpenEvolve unified configuration system, including UnifiedConfiguration, adapters, and centralized imports.

**Table of Contents:**
- [UnifiedConfiguration](#unifiedconfiguration)
- [Factory Functions](#factory-functions)
- [EvolutionAdapter](#evolutionadapter)
- [AdversarialAdapter](#adversarialadapter)
- [Import System](#import-system)
- [API Classes](#api-classes)
- [Result Types](#result-types)
- [Exceptions](#exceptions)

---

## UnifiedConfiguration

The core configuration class that serves as single source of truth for all 272 OpenEvolve parameters.

### Class: `UnifiedConfiguration`

```python
class UnifiedConfiguration:
    """Single configuration class for ALL OpenEvolve modules"""

    def __init__(
        self,
        parameters: Dict[str, Any],
        manager: Optional['ParameterManager'] = None,
        validate: bool = True
    )
```

**Parameters:**
- `parameters` (Dict[str, Any]): Dictionary of parameter values
- `manager` (Optional[ParameterManager]): ParameterManager instance for validation (creates new if None)
- `validate` (bool): Whether to validate parameters (default: True)

**Raises:**
- `ConfigurationValidationError`: If validation fails and validate=True

**Example:**
```python
from unified_configuration import UnifiedConfiguration

config = UnifiedConfiguration({
    'max_iterations': 20,
    'temperature': 0.7
})
```

---

### Core Properties

#### `parameters`

Get all configuration parameters as dictionary.

```python
@property
def parameters(self) -> Dict[str, Any]
```

**Returns:** Dictionary of all parameters with defaults applied

**Example:**
```python
config = create_unified_config({'max_iterations': 20})
params = config.parameters
print(params)  # {'max_iterations': 20, 'temperature': 0.7, ...}
```

---

### Convenience Properties

#### `evolution_mode`

Get evolution mode parameter.

```python
@property
def evolution_mode(self) -> str
```

**Returns:** Evolution mode (default: 'standard')

**Example:**
```python
config = create_unified_config({'evolution_mode': 'adversarial'})
print(config.evolution_mode)  # 'adversarial'
```

#### `max_iterations`

Get max iterations parameter.

```python
@property
def max_iterations(self) -> int
```

**Returns:** Maximum iterations (default: 10)

#### `population_size`

Get population size parameter.

```python
@property
def population_size(self) -> int
```

**Returns:** Population size (default: 20)

#### `temperature`

Get LLM temperature parameter.

```python
@property
def temperature(self) -> float
```

**Returns:** Temperature (default: 0.7)

#### `max_tokens`

Get max tokens parameter.

```python
@property
def max_tokens(self) -> int
```

**Returns:** Max tokens (default: 2048)

#### `seed`

Get random seed parameter.

```python
@property
def seed(self) -> Optional[int]
```

**Returns:** Random seed or None

#### `api_key`

Get API key parameter.

```python
@property
def api_key(self) -> str
```

**Returns:** API key (default: '')

#### `api_base`

Get API base URL parameter.

```python
@property
def api_base(self) -> str
```

**Returns:** API base URL (default: 'https://api.openai.com/v1')

#### `model_id`

Get model ID parameter.

```python
@property
def model_id(self) -> str
```

**Returns:** Model ID (default: 'gpt-4')

#### `adversarial_rounds`

Get adversarial rounds parameter.

```python
@property
def adversarial_rounds(self) -> int
```

**Returns:** Adversarial rounds (default: 5)

#### `attack_strength`

Get attack strength parameter.

```python
@property
def attack_strength(self) -> float
```

**Returns:** Attack strength (default: 0.5)

#### `defense_strategy`

Get defense strategy parameter.

```python
@property
def defense_strategy(self) -> str
```

**Returns:** Defense strategy (default: 'reactive')

---

### Dynamic Parameter Access

#### `get()`

Get any parameter by name.

```python
def get(self, name: str, default: Any = None) -> Any
```

**Parameters:**
- `name` (str): Parameter name
- `default` (Any): Default value if parameter not found

**Returns:** Parameter value or default

**Example:**
```python
config = create_unified_config({'max_iterations': 20})
iterations = config.get('max_iterations', 10)
```

#### `get_category_params()`

Get all parameters for a specific category.

```python
def get_category_params(self, category: str) -> Dict[str, Any]
```

**Parameters:**
- `category` (str): Category name (e.g., 'core_evolution', 'adversarial')

**Returns:** Dictionary of parameters in that category

**Example:**
```python
config = create_unified_config()
core_params = config.get_category_params('core_evolution')
```

#### `set()`

Set a parameter value.

```python
def set(self, name: str, value: Any, validate: bool = False) -> None
```

**Parameters:**
- `name` (str): Parameter name
- `value` (Any): New value
- `validate` (bool): Whether to validate the new value

**Raises:** `ConfigurationValidationError` if validation fails

**Example:**
```python
config = create_unified_config()
config.set('max_iterations', 20, validate=True)
```

#### `update()`

Update multiple parameters at once.

```python
def update(self, parameters: Dict[str, Any], validate: bool = True) -> None
```

**Parameters:**
- `parameters` (Dict[str, Any]): Dictionary of parameters to update
- `validate` (bool): Whether to validate the updated configuration

**Raises:** `ConfigurationValidationError` if validation fails

**Example:**
```python
config = create_unified_config()
config.update({
    'max_iterations': 20,
    'temperature': 0.8
}, validate=True)
```

---

### Parameter Merging

#### `merge()`

Merge this configuration with other parameter dictionaries.

```python
def merge(self, *others: Dict[str, Any], validate: bool = True) -> UnifiedConfiguration
```

**Parameters:**
- `*others` (Dict[str, Any]): Variable number of parameter dictionaries to merge
- `validate` (bool): Whether to validate the merged result

**Returns:** New UnifiedConfiguration with merged parameters

**Raises:** `ConfigurationValidationError` if validation fails

**Example:**
```python
config1 = create_unified_config({'max_iterations': 10})
config2 = config1.merge({'temperature': 0.8}, {'population_size': 30})
```

---

### Conversion Methods

#### `to_evolution_config()`

Convert to EvolutionConfiguration for evolution module.

```python
def to_evolution_config(self) -> EvolutionConfiguration
```

**Returns:** EvolutionConfiguration instance with all parameters

**Example:**
```python
config = create_unified_config({'max_iterations': 20})
evo_config = config.to_evolution_config()
```

#### `to_adversarial_config()`

Convert to AdversarialConfiguration for adversarial module.

```python
def to_adversarial_config(self) -> AdversarialConfiguration
```

**Returns:** AdversarialConfiguration instance with all parameters

**Example:**
```python
config = create_unified_config({'adversarial_rounds': 5})
adv_config = config.to_adversarial_config()
```

#### `to_dict()`

Export configuration as dictionary.

```python
def to_dict(self) -> Dict[str, Any]
```

**Returns:** Complete parameter dictionary

**Example:**
```python
config = create_unified_config({'max_iterations': 20})
params = config.to_dict()
```

#### `validate()`

Validate the current configuration.

```python
def validate(self) -> ValidationResult
```

**Returns:** ValidationResult with validation status

**Example:**
```python
config = create_unified_config()
validation = config.validate()
if validation.valid:
    print("Valid!")
else:
    print(f"Errors: {validation.errors}")
```

---

### Utility Methods

#### `__repr__()`

String representation showing key parameters.

```python
def __repr__(self) -> str
```

**Example:**
```python
config = create_unified_config({'max_iterations': 20})
print(config)  # UnifiedConfiguration(mode=standard, iterations=20, temp=0.7, 272 params total)
```

#### `__len__()`

Return number of parameters.

```python
def __len__(self) -> int
```

**Example:**
```python
config = create_unified_config()
print(len(config))  # 272
```

#### `__contains__()`

Check if parameter exists.

```python
def __contains__(self, name: str) -> bool
```

**Example:**
```python
config = create_unified_config()
print('max_iterations' in config)  # True
```

#### `__getitem__()`

Allow dict-style access: `config['temperature']`

```python
def __getitem__(self, name: str) -> Any
```

**Example:**
```python
config = create_unified_config()
temp = config['temperature']
```

#### `__setitem__()`

Allow dict-style setting: `config['temperature'] = 0.8`

```python
def __setitem__(self, name: str, value: Any) -> None
```

**Example:**
```python
config = create_unified_config()
config['temperature'] = 0.8
```

---

## Factory Functions

### `create_unified_config()`

Factory function to create UnifiedConfiguration with defaults.

```python
def create_unified_config(
    parameters: Optional[Dict[str, Any]] = None,
    manager: Optional['ParameterManager'] = None,
    validate: bool = True
) -> UnifiedConfiguration
```

**Parameters:**
- `parameters` (Optional[Dict[str, Any]]): Initial parameters (uses all defaults if None)
- `manager` (Optional[ParameterManager]): ParameterManager instance
- `validate` (bool): Whether to validate parameters (default: True)

**Returns:** UnifiedConfiguration instance

**Example:**
```python
from unified_configuration import create_unified_config

# All defaults
config = create_unified_config()

# Custom parameters
config = create_unified_config({
    'max_iterations': 20,
    'temperature': 0.8
})
```

---

### `merge_configs()`

Merge multiple configuration dictionaries into UnifiedConfiguration.

```python
def merge_configs(
    *configs: Dict[str, Any],
    manager: Optional['ParameterManager'] = None
) -> UnifiedConfiguration
```

**Parameters:**
- `*configs` (Dict[str, Any]): Variable number of config dicts to merge
- `manager` (Optional[ParameterManager]): ParameterManager for validation

**Returns:** UnifiedConfiguration with merged parameters

**Note:** Later configs override earlier ones (last one wins)

**Example:**
```python
from unified_configuration import merge_configs

config = merge_configs(
    {'max_iterations': 10},
    {'temperature': 0.7},
    {'population_size': 30}
)
```

---

### `load_unified_config_from_file()`

Load UnifiedConfiguration from JSON file.

```python
def load_unified_config_from_file(
    filepath: str,
    manager: Optional['ParameterManager'] = None
) -> UnifiedConfiguration
```

**Parameters:**
- `filepath` (str): Path to JSON configuration file
- `manager` (Optional[ParameterManager]): ParameterManager instance

**Returns:** UnifiedConfiguration instance

**Raises:**
- `FileNotFoundError`: If file doesn't exist
- `json.JSONDecodeError`: If file is not valid JSON
- `ConfigurationValidationError`: If config is invalid

**Example:**
```python
from unified_configuration import load_unified_config_from_file

config = load_unified_config_from_file('config.json')
```

---

### `save_unified_config_to_file()`

Save UnifiedConfiguration to JSON file.

```python
def save_unified_config_to_file(
    config: UnifiedConfiguration,
    filepath: str,
    pretty: bool = True
) -> None
```

**Parameters:**
- `config` (UnifiedConfiguration): Configuration to save
- `filepath` (str): Path to save configuration
- `pretty` (bool): Whether to format JSON prettily (default: True)

**Example:**
```python
from unified_configuration import save_unified_config_to_file

save_unified_config_to_file(config, 'config.json')
```

---

### `create_standard_evolution_config()`

Create UnifiedConfiguration with standard evolution presets.

```python
def create_standard_evolution_config(
    max_iterations: int = 10,
    population_size: int = 20,
    temperature: float = 0.7,
    **kwargs
) -> UnifiedConfiguration
```

**Parameters:**
- `max_iterations` (int): Maximum iterations (default: 10)
- `population_size` (int): Population size (default: 20)
- `temperature` (float): LLM temperature (default: 0.7)
- `**kwargs`: Additional parameters to override

**Returns:** UnifiedConfiguration for standard evolution

**Example:**
```python
from unified_configuration import create_standard_evolution_config

config = create_standard_evolution_config(
    max_iterations=20,
    temperature=0.8
)
```

---

### `create_adversarial_testing_config()`

Create UnifiedConfiguration with adversarial testing presets.

```python
def create_adversarial_testing_config(
    adversarial_rounds: int = 5,
    attack_strength: float = 0.5,
    defense_strategy: str = 'reactive',
    **kwargs
) -> UnifiedConfiguration
```

**Parameters:**
- `adversarial_rounds` (int): Number of adversarial rounds (default: 5)
- `attack_strength` (float): Strength of attacks 0.0-1.0 (default: 0.5)
- `defense_strategy` (str): Defense strategy to use (default: 'reactive')
- `**kwargs`: Additional parameters to override

**Returns:** UnifiedConfiguration for adversarial testing

**Example:**
```python
from unified_configuration import create_adversarial_testing_config

config = create_adversarial_testing_config(
    adversarial_rounds=10,
    attack_strength=0.7
)
```

---

### `create_quality_diversity_config()`

Create UnifiedConfiguration with quality diversity presets.

```python
def create_quality_diversity_config(
    archive_size: int = 100,
    feature_bins: int = 10,
    diversity_weight: float = 0.5,
    **kwargs
) -> UnifiedConfiguration
```

**Parameters:**
- `archive_size` (int): Size of archive for MAP-Elites (default: 100)
- `feature_bins` (int): Number of bins per feature dimension (default: 10)
- `diversity_weight` (float): Weight of diversity vs quality 0.0-1.0 (default: 0.5)
- `**kwargs`: Additional parameters to override

**Returns:** UnifiedConfiguration for quality diversity evolution

**Example:**
```python
from unified_configuration import create_quality_diversity_config

config = create_quality_diversity_config(
    archive_size=200,
    diversity_weight=0.7
)
```

---

## EvolutionAdapter

Adapter for evolution module using unified configuration.

### Class: `EvolutionAdapter`

```python
class EvolutionAdapter:
    """Adapter for evolution module using unified configuration"""

    def __init__(
        self,
        config: UnifiedConfiguration,
        evaluator: Optional[Callable] = None,
        status_callback: Optional[Callable[[str], None]] = None
    )
```

**Parameters:**
- `config` (UnifiedConfiguration): Configuration with all evolution parameters
- `evaluator` (Optional[Callable]): Custom evaluator function
- `status_callback` (Optional[Callable]): Callback for status updates during execution

**Example:**
```python
from evolution_adapter import EvolutionAdapter, create_unified_config

config = create_unified_config({'max_iterations': 20})
adapter = EvolutionAdapter(config)
```

---

### Methods

#### `run_evolution()`

Run evolution with the configured parameters.

```python
def run_evolution(
    self,
    initial_content: str,
    **kwargs
) -> EvolutionResult
```

**Parameters:**
- `initial_content` (str): Initial content to evolve
- `**kwargs`: Additional parameters to override config

**Returns:** EvolutionResult with execution results

**Example:**
```python
adapter = create_evolution_adapter(max_iterations=20)
result = adapter.run_evolution("Initial content")
if result.success:
    print(f"Final fitness: {result.final_fitness}")
```

---

### Factory Function

#### `create_evolution_adapter()`

Convenience factory to create EvolutionAdapter with parameters.

```python
def create_evolution_adapter(**kwargs) -> EvolutionAdapter
```

**Parameters:**
- `**kwargs`: Any configuration parameters (272 parameters available)

**Returns:** Configured EvolutionAdapter instance

**Example:**
```python
from evolution_adapter import create_evolution_adapter

adapter = create_evolution_adapter(
    max_iterations=20,
    temperature=0.7,
    population_size=30
)
result = adapter.run_evolution(content)
```

---

## AdversarialAdapter

Adapter for adversarial testing module using unified configuration.

### Class: `AdversarialAdapter`

```python
class AdversarialAdapter:
    """Adapter for adversarial testing module using unified configuration"""

    def __init__(
        self,
        config: UnifiedConfiguration,
        status_callback: Optional[Callable[[str], None]] = None
    )
```

**Parameters:**
- `config` (UnifiedConfiguration): Configuration with all adversarial parameters
- `status_callback` (Optional[Callable]): Callback for status updates

**Example:**
```python
from adversarial_adapter import AdversarialAdapter, create_unified_config

config = create_unified_config({'adversarial_rounds': 5})
adapter = AdversarialAdapter(config)
```

---

### Methods

#### `run_adversarial_testing()`

Run adversarial testing with the configured parameters.

```python
def run_adversarial_testing(
    self,
    content: str,
    **kwargs
) -> AdversarialResult
```

**Parameters:**
- `content` (str): Content to test
- `**kwargs`: Additional parameters to override config

**Returns:** AdversarialResult with testing results

**Example:**
```python
adapter = create_adversarial_adapter(adversarial_rounds=5)
result = adapter.run_adversarial_testing(content)
if result.success:
    print(f"Vulnerabilities found: {result.vulnerabilities_found}")
```

---

### Factory Function

#### `create_adversarial_adapter()`

Convenience factory to create AdversarialAdapter with parameters.

```python
def create_adversarial_adapter(**kwargs) -> AdversarialAdapter
```

**Parameters:**
- `**kwargs`: Any configuration parameters (272 parameters available)

**Returns:** Configured AdversarialAdapter instance

**Example:**
```python
from adversarial_adapter import create_adversarial_adapter

adapter = create_adversarial_adapter(
    adversarial_rounds=5,
    attack_strength=0.7
)
result = adapter.run_adversarial_testing(content)
```

---

## Import System

Centralized import management from `openevolve_imports.py`.

### Availability Flags

Boolean flags indicating module availability:

```python
EVOLUTION_AVAILABLE: bool
ADVERSARIAL_AVAILABLE: bool
PARAMETER_MANAGER_AVAILABLE: bool
KNOWLEDGE_ENGINE_AVAILABLE: bool
LEANAIDE_AVAILABLE: bool
CREWAI_AVAILABLE: bool
OPENEREVOLVE_AVAILABLE: bool
DECOMPOSITION_AVAILABLE: bool
MAKER_ENGINE_AVAILABLE: bool
MDAP_ENGINE_AVAILABLE: bool
INVENTION_PLANNER_AVAILABLE: bool
EVALUATOR_TEAM_AVAILABLE: bool
BLUE_TEAM_AVAILABLE: bool
RED_TEAM_AVAILABLE: bool
VISUALIZATION_AVAILABLE: bool
SESSION_UTILS_AVAILABLE: bool
```

**Example:**
```python
from openevolve_imports import EVOLUTION_AVAILABLE, ADVERSARIAL_AVAILABLE

if EVOLUTION_AVAILABLE:
    print("Evolution is available")

if ADVERSARIAL_AVAILABLE:
    print("Adversarial is available")
```

---

### Convenience Functions

#### `get_available_modules()`

Get dictionary of all available modules and their status.

```python
def get_available_modules() -> Dict[str, bool]
```

**Returns:** Dict mapping module names to availability status

**Example:**
```python
from openevolve_imports import get_available_modules

available = get_available_modules()
for module, status in available.items():
    print(f"{module}: {status}")
```

#### `require_evolution()`

Get evolution module, raise error if not available.

```python
def require_evolution() -> Any
```

**Returns:** The evolution module

**Raises:** ImportError if evolution module is not available

**Example:**
```python
from openevolve_imports import require_evolution

try:
    evolution = require_evolution()
    result = evolution.run_evolution_loop(...)
except ImportError as e:
    print(f"Evolution not available: {e}")
```

#### `require_adversarial()`

Get adversarial module, raise error if not available.

```python
def require_adversarial() -> Any
```

**Returns:** The adversarial module

**Raises:** ImportError if adversarial module is not available

#### `require_parameter_manager()`

Get parameter manager module, raise error if not available.

```python
def require_parameter_manager() -> Any
```

**Returns:** The parameter manager module

**Raises:** ImportError if parameter manager is not available

#### `safe_import_evolution()`

Safely get evolution module, return None if not available.

```python
def safe_import_evolution() -> Optional[Any]
```

**Returns:** The evolution module or None

**Example:**
```python
from openevolve_imports import safe_import_evolution

evolution = safe_import_evolution()
if evolution:
    result = evolution.run_evolution_loop(...)
```

#### `safe_import_adversarial()`

Safely get adversarial module, return None if not available.

```python
def safe_import_adversarial() -> Optional[Any]
```

**Returns:** The adversarial module or None

#### `safe_import_parameter_manager()`

Safely get parameter manager module, return None if not available.

```python
def safe_import_parameter_manager() -> Optional[Any]
```

**Returns:** The parameter manager module or None

#### `print_import_status()`

Print the availability status of all OpenEvolve modules.

```python
def print_import_status() -> None
```

**Example:**
```python
from openevolve_imports import print_import_status

print_import_status()
# Output:
# ============================================================
#   OpenEvolve Module Import Status
# ============================================================
#   evolution............................ ✓ Available
#   adversarial.......................... ✓ Available
#   parameter_manager.................... ✓ Available
#   ...
```

---

## API Classes

### `EvolutionAPI`

Wrapper for evolution module functionality.

#### Methods

##### `is_available()`

Check if evolution module is available.

```python
@staticmethod
def is_available() -> bool
```

**Returns:** True if available, False otherwise

##### `run_evolution_loop()`

Run evolution loop.

```python
@staticmethod
def run_evolution_loop(*args, **kwargs)
```

**Raises:** ImportError if evolution module is not available

##### `get_evolution_config()`

Get evolution configuration.

```python
@staticmethod
def get_evolution_config(*args, **kwargs)
```

**Raises:** ImportError if evolution module is not available

---

### `AdversarialAPI`

Wrapper for adversarial module functionality.

#### Methods

##### `is_available()`

Check if adversarial module is available.

```python
@staticmethod
def is_available() -> bool
```

**Returns:** True if available, False otherwise

##### `run_comprehensive_adversarial_testing()`

Run comprehensive adversarial testing.

```python
@staticmethod
def run_comprehensive_adversarial_testing(*args, **kwargs)
```

**Raises:** ImportError if adversarial module is not available

##### `get_adversarial_config()`

Get adversarial configuration.

```python
@staticmethod
def get_adversarial_config(*args, **kwargs)
```

**Raises:** ImportError if adversarial module is not available

---

### `ParameterAPI`

Wrapper for parameter manager functionality.

#### Methods

##### `is_available()`

Check if parameter manager is available.

```python
@staticmethod
def is_available() -> bool
```

**Returns:** True if available, False otherwise

##### `get_parameter_manager()`

Get parameter manager instance.

```python
@staticmethod
def get_parameter_manager(*args, **kwargs)
```

**Raises:** ImportError if parameter manager is not available

---

### `KnowledgeAPI`

Wrapper for knowledge engine functionality.

#### Methods

##### `is_available()`

Check if knowledge engine is available.

```python
@staticmethod
def is_available() -> bool
```

**Returns:** True if available, False otherwise

##### `query_knowledge_base()`

Query knowledge base.

```python
@staticmethod
def query_knowledge_base(*args, **kwargs)
```

**Raises:** ImportError if knowledge engine is not available

---

## Result Types

### `EvolutionResult`

Result from evolution execution.

```python
@dataclass
class EvolutionResult:
    success: bool
    final_content: str
    original_content: str
    iterations_completed: int = 0
    best_fitness: float = 0.0
    final_fitness: float = 0.0
    improvement_ratio: float = 0.0
    convergence_iteration: Optional[int] = None
    total_evaluations: int = 0
    duration_seconds: float = 0.0
    evolution_mode: str = "standard"
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
```

**Attributes:**
- `success` (bool): Whether evolution completed successfully
- `final_content` (str): The evolved content
- `original_content` (str): The original input content
- `iterations_completed` (int): Number of iterations actually completed
- `best_fitness` (float): Best fitness score achieved
- `final_fitness` (float): Fitness of final content
- `improvement_ratio` (float): Ratio of improvement
- `convergence_iteration` (Optional[int]): Iteration where convergence occurred
- `total_evaluations` (int): Total number of evaluations performed
- `duration_seconds` (float): Total execution time in seconds
- `evolution_mode` (str): Mode of evolution used
- `metrics` (Dict[str, Any]): Additional metrics from evolution
- `error` (Optional[str]): Error message if execution failed

---

### `AdversarialResult`

Result from adversarial testing execution.

```python
@dataclass
class AdversarialResult:
    success: bool
    final_content: str
    original_content: str
    total_rounds: int = 0
    vulnerabilities_found: int = 0
    fixes_applied: int = 0
    robustness_score: float = 0.0
    attack_success_rate: float = 0.0
    defense_success_rate: float = 0.0
    consensus_score: float = 0.0
    improvement_ratio: float = 0.0
    duration_seconds: float = 0.0
    team_results: Dict[str, Any] = field(default_factory=dict)
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    fixes: List[Dict[str, Any]] = field(default_factory=list)
    rounds: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
```

**Attributes:**
- `success` (bool): Whether testing completed successfully
- `final_content` (str): The hardened/defended content
- `original_content` (str): The original input content
- `total_rounds` (int): Number of adversarial rounds completed
- `vulnerabilities_found` (int): Total vulnerabilities identified
- `fixes_applied` (int): Total fixes applied
- `robustness_score` (float): Final robustness score (0.0-1.0)
- `attack_success_rate` (float): Rate of successful attacks (0.0-1.0)
- `defense_success_rate` (float): Rate of successful defenses (0.0-1.0)
- `consensus_score` (float): Evaluator team consensus score (0.0-1.0)
- `improvement_ratio` (float): Ratio of content improvement
- `duration_seconds` (float): Total execution time
- `team_results` (Dict[str, Any]): Detailed results from each team
- `vulnerabilities` (List[Dict[str, Any]]): List of all vulnerabilities found
- `fixes` (List[Dict[str, Any]]): List of all fixes applied
- `rounds` (List[Dict[str, Any]]): Detailed per-round results
- `error` (Optional[str]): Error message if execution failed

---

### `ValidationResult`

Result from configuration validation.

```python
class ValidationResult:
    def __init__(
        self,
        valid: bool = True,
        errors: List[str] = None,
        warnings: List[str] = None
    ):
        self.valid = valid
        self.errors = errors or []
        self.warnings = warnings or []
```

**Attributes:**
- `valid` (bool): Whether validation passed
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of validation warnings

---

## Exceptions

### `UnifiedConfigurationError`

Base error for UnifiedConfiguration operations.

```python
class UnifiedConfigurationError(Exception):
    """Base error for UnifiedConfiguration operations"""
    pass
```

---

### `ConfigurationValidationError`

Raised when configuration validation fails.

```python
class ConfigurationValidationError(UnifiedConfigurationError):
    """Raised when configuration validation fails"""

    def __init__(self, errors: List[str], warnings: List[str] = None):
        self.errors = errors
        self.warnings = warnings or []
        super().__init__(f"Configuration validation failed with {len(errors)} errors")
```

**Attributes:**
- `errors` (List[str]): List of validation errors
- `warnings` (List[str]): List of validation warnings

**Example:**
```python
from unified_configuration import create_unified_config, ConfigurationValidationError

try:
    config = create_unified_config({'max_iterations': -1})
except ConfigurationValidationError as e:
    print(f"Validation failed with {len(e.errors)} errors:")
    for error in e.errors:
        print(f"  - {error}")
```

---

**Last Updated:** 2025-01-03
**Version:** 1.0.0
