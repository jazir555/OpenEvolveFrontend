"""
Type Safety and Type Stubs for Enhanced Adversarial Testing System

This module provides:
- Marker file for PEP 561 type checking
- Comprehensive type hints
- Type stubs for better IDE support
- Generic type aliases
- Type guards and validation
- Runtime type checking utilities

Author: OpenEvolve Type Safety Team
Created: 2025-01-07
Version: 1.0.0
"""

from __future__ import annotations

# PEP 561 marker file - indicates this package supports type checking
__all__ = [
    # Type aliases
    "Content",
    "ContentType",
    "Theorem",
    "Context",
    "AttackResult",
    "DefenseResult",
    "EvaluationResult",
    "TestResult",
    "ConfigDict",

    # Type constructors
    "is_attack_result",
    "is_defense_result",
    "is_evaluation_result",
    "validate_type",
    "check_type",
    "assert_type",

    # Generic types
    "JSONValue",
    "JSONObject",
    "JSONArray",
]

from typing import (
    Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, TypeVar, Union,
    Literal, TypedDict, Protocol, runtime_checkable, Generic, TypeGuard,
    get_type_hints, get_origin, get_args
)
from enum import Enum
from dataclasses import dataclass
from typing_extensions import NotRequired, Required


# =============================================================================
# TYPE ALIASES
# =============================================================================

# Basic types
Content = str
ContentType = Literal[
    "code_python",
    "code_javascript",
    "code_typescript",
    "document_general",
    "document_legal",
    "document_medical",
    "api_spec",
    "database_schema",
    "config_file"
]
Theorem = str
Context = Dict[str, Any]

# Numeric types
Confidence = float  # 0.0 to 1.0
Severity = float    # 0.0 to 1.0
Robustness = float  # 0.0 to 1.0
Effectiveness = float  # 0.0 to 1.0
ProgressPercent = float  # 0.0 to 100.0

# Result types
class AttackResult(TypedDict):
    """Type definition for attack result"""
    success: bool
    severity: Severity
    description: str
    weak_point: str
    confidence: Confidence
    attack_type: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]

class DefenseResult(TypedDict):
    """Type definition for defense result"""
    attack_blocked: bool
    effectiveness: Effectiveness
    improved_proof: str
    description: str
    confidence: Confidence
    defense_type: NotRequired[str]
    metadata: NotRequired[Dict[str, Any]]

class EvaluationResult(TypedDict):
    """Type definition for evaluation result"""
    score: float
    metrics: Dict[str, float]
    issues: List[str]
    recommendations: List[str]

class ExplanationResult(TypedDict):
    """Type definition for explanation result"""
    decision_type: str
    reasoning: str
    context: Context
    confidence: Confidence
    alternatives_considered: NotRequired[List[str]]

class TestResult(TypedDict):
    """Type definition for test result"""
    success: bool
    final_robustness: Robustness
    duration: float
    iterations: int
    attacks: List[AttackResult]
    defenses: List[DefenseResult]
    explanations: List[ExplanationResult]
    adaptations: List[Dict[str, Any]]
    learning_insights: Dict[str, Any]

# Configuration types
ConfigDict = Dict[str, Any]
PluginConfig = Dict[str, Any]

# JSON types
JSONPrimitive = Union[str, int, float, bool, None]
JSONObject = Dict[str, "JSONValue"]
JSONArray = List["JSONValue"]
JSONValue = Union[JSONPrimitive, JSONObject, JSONArray]

# Async types
AsyncAttackFunc = Callable[[Content, ContentType, Theorem, Context], Awaitable[AttackResult]]
AsyncDefenseFunc = Callable[[Content, AttackResult, Theorem, Context], Awaitable[DefenseResult]]
AsyncEvalFunc = Callable[[Content, ContentType, Theorem, Context], Awaitable[EvaluationResult]]

# Callback types
ProgressCallback = Callable[[ProgressPercent, str, int, int], None]
ErrorCallback = Callable[[Exception, Context], None]


# =============================================================================
# GENERIC TYPES
# =============================================================================

T = TypeVar('T')
T_co = TypeVar('T_co', covariant=True)
T_contra = TypeVar('T_contra', contravariant=True)


# =============================================================================
# TYPE GUARDS
# =============================================================================

def is_attack_result(value: Any) -> TypeGuard[AttackResult]:
    """Type guard for AttackResult"""
    required_keys = {"success", "severity", "description", "weak_point", "confidence"}
    return (
        isinstance(value, dict)
        and all(key in value for key in required_keys)
        and isinstance(value["success"], bool)
        and isinstance(value["severity"], (int, float))
        and isinstance(value["description"], str)
        and isinstance(value["weak_point"], str)
        and isinstance(value["confidence"], (int, float))
    )


def is_defense_result(value: Any) -> TypeGuard[DefenseResult]:
    """Type guard for DefenseResult"""
    required_keys = {"attack_blocked", "effectiveness", "improved_proof", "description", "confidence"}
    return (
        isinstance(value, dict)
        and all(key in value for key in required_keys)
        and isinstance(value["attack_blocked"], bool)
        and isinstance(value["effectiveness"], (int, float))
        and isinstance(value["improved_proof"], str)
        and isinstance(value["description"], str)
        and isinstance(value["confidence"], (int, float))
    )


def is_evaluation_result(value: Any) -> TypeGuard[EvaluationResult]:
    """Type guard for EvaluationResult"""
    required_keys = {"score", "metrics", "issues", "recommendations"}
    return (
        isinstance(value, dict)
        and all(key in value for key in required_keys)
        and isinstance(value["score"], (int, float))
        and isinstance(value["metrics"], dict)
        and isinstance(value["issues"], list)
        and isinstance(value["recommendations"], list)
    )


def is_test_result(value: Any) -> TypeGuard[TestResult]:
    """Type guard for TestResult"""
    required_keys = {
        "success", "final_robustness", "duration", "iterations",
        "attacks", "defenses", "explanations", "adaptations", "learning_insights"
    }
    return (
        isinstance(value, dict)
        and all(key in value for key in required_keys)
    )


# =============================================================================
# TYPE VALIDATION
# =============================================================================

class TypeError_(Exception):
    """Type validation error"""

    def __init__(self, message: str, value: Any = None, expected_type: Type = None):
        self.message = message
        self.value = value
        self.expected_type = expected_type
        super().__init__(self.message)


def validate_type(value: Any, expected_type: Type, type_name: str = "value") -> Any:
    """
    Validate that a value matches the expected type

    Args:
        value: Value to validate
        expected_type: Expected type
        type_name: Name for error messages

    Returns:
        The value if valid

    Raises:
        TypeError_: If type doesn't match
    """
    # Handle None for Optional types
    if value is None:
        return None

    # Get origin type for generic types
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Handle Union types (including Optional)
    if origin is Union:
        # Try each type in the union
        for arg in args:
            try:
                return validate_type(value, arg, type_name)
            except TypeError_:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Continuing after error", exc_info=True)
                continue
        raise TypeError_(
            f"{type_name} must be one of {args}, got {type(value).__name__}",
            value=value,
            expected_type=expected_type
        )

    # Handle Literal types
    if origin is Literal:
        if value not in args:
            raise TypeError_(
                f"{type_name} must be one of {args}, got {value!r}",
                value=value,
                expected_type=expected_type
            )
        return value

    # Handle List types
    if origin is list:
        if not isinstance(value, list):
            raise TypeError_(
                f"{type_name} must be a list, got {type(value).__name__}",
                value=value,
                expected_type=expected_type
            )
        if args:
            # Validate element types
            element_type = args[0]
            return [validate_type(item, element_type, f"{type_name}[{i}]") for i, item in enumerate(value)]
        return value

    # Handle Dict types
    if origin is dict:
        if not isinstance(value, dict):
            raise TypeError_(
                f"{type_name} must be a dict, got {type(value).__name__}",
                value=value,
                expected_type=expected_type
            )
        if args and len(args) == 2:
            # Validate key and value types
            key_type, value_type = args
            return {
                validate_type(k, key_type, f"{type_name} key"): validate_type(v, value_type, f"{type_name} value")
                for k, v in value.items()
            }
        return value

    # Handle basic types
    if not isinstance(value, expected_type):
        raise TypeError_(
            f"{type_name} must be {expected_type.__name__}, got {type(value).__name__}",
            value=value,
            expected_type=expected_type
        )

    return value


def check_type(value: Any, expected_type: Type) -> bool:
    """
    Check if a value matches the expected type (returns bool)

    Args:
        value: Value to check
        expected_type: Expected type

    Returns:
        True if type matches, False otherwise
    """
    try:
        validate_type(value, expected_type)
        return True
    except TypeError_:
        return False


def assert_type(value: Any, expected_type: Type, type_name: str = "value") -> None:
    """
    Assert that a value matches the expected type

    Args:
        value: Value to check
        expected_type: Expected type
        type_name: Name for error messages

    Raises:
        TypeError_: If type doesn't match
    """
    validate_type(value, expected_type, type_name)


# =============================================================================
# RUNTIME TYPE CHECKING DECORATORS
# =============================================================================

def typed(func: Callable[..., T]) -> Callable[..., T]:
    """
    Decorator to add runtime type checking to a function

    Uses type hints to validate arguments and return value

    Example:
        @typed
        def process_attack(content: str, severity: float) -> AttackResult:
            # Function implementation
            return {"success": True, ...}
    """
    import functools

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints
        type_hints = get_type_hints(func)

        # Validate arguments
        sig = func.__signature__  # type: ignore
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                validate_type(param_value, expected_type, param_name)

        # Call function
        result = func(*args, **kwargs)

        # Validate return type
        if "return" in type_hints:
            validate_type(result, type_hints["return"], "return value")

        return result

    return wrapper


def async_typed(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """
    Decorator to add runtime type checking to an async function

    Example:
        @async_typed
        async def generate_attack(content: str) -> AttackResult:
            # Async implementation
            return {"success": True, ...}
    """
    import functools

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # Get type hints
        type_hints = get_type_hints(func)

        # Validate arguments
        import inspect
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        for param_name, param_value in bound_args.arguments.items():
            if param_name in type_hints:
                expected_type = type_hints[param_name]
                validate_type(param_value, expected_type, param_name)

        # Call function
        result = await func(*args, **kwargs)

        # Validate return type
        if "return" in type_hints:
            validate_type(result, type_hints["return"], "return value")

        return result

    return wrapper


# =============================================================================
# PROTOCOLS (STRUCTURAL TYPING)
# =============================================================================

@runtime_checkable
class AttackGenerator(Protocol):
    """Protocol for attack generators"""

    async def generate_attack(
        self,
        content: Content,
        content_type: ContentType,
        theorem: Theorem,
        context: Context
    ) -> AttackResult:
        """Generate an attack"""
        ...


@runtime_checkable
class DefenseGenerator(Protocol):
    """Protocol for defense generators"""

    async def generate_defense(
        self,
        content: Content,
        attack: AttackResult,
        theorem: Theorem,
        context: Context
    ) -> DefenseResult:
        """Generate a defense"""
        ...


@runtime_checkable
class ContentEvaluator(Protocol):
    """Protocol for content evaluators"""

    async def evaluate(
        self,
        content: Content,
        content_type: ContentType,
        theorem: Theorem,
        context: Context
    ) -> EvaluationResult:
        """Evaluate content"""
        ...


# =============================================================================
# TYPED DICTS FOR COMMON STRUCTURES
# =============================================================================

class MetricData(TypedDict):
    """Type for metric data"""
    name: str
    value: float
    timestamp: str
    metadata: NotRequired[Dict[str, Any]]

class LearningExperience(TypedDict):
    """Type for learning experience"""
    attack: AttackResult
    defense: DefenseResult
    outcome: Dict[str, Any]
    timestamp: NotRequired[str]

class PluginMetadata(TypedDict):
    """Type for plugin metadata"""
    plugin_id: str
    plugin_type: str
    name: str
    version: str
    author: str
    description: str
    enabled: NotRequired[bool]
    dependencies: NotRequired[List[str]]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_optional_type(type_hint: Type) -> Optional[Type]:
    """
    Get the wrapped type from an Optional type

    Example:
        get_optional_type(Optional[int]) -> int
    """
    origin = get_origin(type_hint)
    if origin is Union:
        args = get_args(type_hint)
        if type(None) in args:
            # Return the non-None type
            for arg in args:
                if arg is not type(None):
                    return arg
    return None


def is_optional_type(type_hint: Type) -> bool:
    """Check if a type hint is Optional"""
    return get_optional_type(type_hint) is not None


def extract_union_types(type_hint: Type) -> List[Type]:
    """
    Extract all types from a Union

    Example:
        extract_union_types(Union[int, str, None]) -> [int, str, None]
    """
    origin = get_origin(type_hint)
    if origin is Union:
        return list(get_args(type_hint))
    return [type_hint]


# =============================================================================
# DEMO / TYPE CHECKING TESTS
# =============================================================================

if __name__ == "__main__":
    print("Type Safety and Validation System")
    print("=" * 60)

    # Demo 1: Type validation
    print("\n1. Type Validation")
    print("-" * 40)

    # Valid type
    try:
        result = validate_type({"success": True, "severity": 0.8, "description": "test", "weak_point": "test", "confidence": 0.9}, AttackResult, "attack_result")
        print("✓ Valid AttackResult")
    except TypeError_ as e:
        print(f"✗ {e.message}")

    # Invalid type
    try:
        result = validate_type("not a dict", AttackResult, "attack_result")
        print("✗ Should have failed!")
    except TypeError_ as e:
        print(f"✓ Caught type error: {e.message}")

    # Demo 2: Type guards
    print("\n2. Type Guards")
    print("-" * 40)

    valid_attack = {
        "success": True,
        "severity": 0.8,
        "description": "Test",
        "weak_point": "Test",
        "confidence": 0.9
    }

    invalid_attack = {"not": "an attack"}

    print(f"Valid attack: {is_attack_result(valid_attack)}")
    print(f"Invalid attack: {is_attack_result(invalid_attack)}")

    # Demo 3: Type checking decorator
    print("\n3. Type Checking Decorator")
    print("-" * 40)

    @typed
    def process_attack(severity: Severity, description: str) -> AttackResult:
        return {
            "success": True,
            "severity": severity,
            "description": description,
            "weak_point": "test",
            "confidence": 0.9
        }

    try:
        result = process_attack(0.8, "test")
        print("✓ Function executed with correct types")
    except TypeError_ as e:
        print(f"✗ {e.message}")

    try:
        result = process_attack("not a float", "test")
        print("✗ Should have failed!")
    except TypeError_ as e:
        print(f"✓ Caught type error in decorator: {e.message}")

    # Demo 4: Protocol checking
    print("\n4. Protocol Checking")
    print("-" * 40)

    @runtime_checkable
    class MyGenerator(Protocol):
        async def generate_attack(self, content: str) -> AttackResult:
            ...

    class ConcreteGenerator:
        async def generate_attack(self, content: str) -> AttackResult:
            return {
                "success": True,
                "severity": 0.5,
                "description": "test",
                "weak_point": "test",
                "confidence": 0.9
            }

    generator = ConcreteGenerator()
    print(f"Implements protocol: {isinstance(generator, MyGenerator)}")

    # Demo 5: Optional type extraction
    print("\n5. Type Utilities")
    print("-" * 40)

    from typing import Optional

    print(f"Optional[int] wrapper: {get_optional_type(Optional[int])}")
    print(f"Is Optional: {is_optional_type(Optional[str])}")
    print(f"Union types: {extract_union_types(Union[int, str, None])}")

    print("\n" + "=" * 60)
    print("Type safety demo complete!")
    print("\nNote: Use 'mypy <filename>' for static type checking")
