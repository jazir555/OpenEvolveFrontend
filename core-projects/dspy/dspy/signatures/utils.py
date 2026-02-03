
# **ACTUAL INTEGRATION**: Adaptive MDAP for Utils
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    from adaptive_mdap.core.types import SubProblem
    ADAPTIVE_MDAP_AVAILABLE = True
except ImportError:
    ADAPTIVE_MDAP_AVAILABLE = False
    TaskComplexityClassifier = None
    AdaptiveMDAPAllocator = None
    SubProblem = None

from typing import Literal

from pydantic.fields import FieldInfo


def get_dspy_field_type(field: FieldInfo) -> Literal["input", "output"]:
    field_type = field.json_schema_extra.get("__dspy_field_type")
    if field_type is None:
        raise ValueError(f"Field {field} does not have a __dspy_field_type")
    return field_type
