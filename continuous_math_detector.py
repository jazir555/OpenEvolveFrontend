"""
Continuous Math Detector
Identifies mathematical content and continuous math problems
"""


from enum import Enum


class ScientificDomain(Enum):
    """Scientific domains for classification."""
    PHYSICS = "physics"
    MATHEMATICS = "mathematics"
    ENGINEERING = "engineering"
    CHEMISTRY = "chemistry"
    BIOLOGY = "biology"
    COMPUTER_SCIENCE = "computer_science"
    GENERAL = "general"


class MathType(Enum):
    """Types of mathematical expressions."""
    ALGEBRAIC = "algebraic"
    CALCULUS = "calculus"
    STATISTICAL = "statistical"
    GEOMETRIC = "geometric"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class ProblemType(Enum):
    """Types of problems."""
    OPTIMIZATION = "optimization"
    EQUATION_SOLVING = "equation_solving"
    PROOF = "proof"
    MODELING = "modeling"
    SIMULATION = "simulation"
    ANALYSIS = "analysis"


def detect_math_type(text: str) -> MathType:
    """Detect the type of mathematics in text."""
    # Simple heuristic detection
    text_lower = text.lower()
    if "integral" in text_lower or "derivative" in text_lower:
        return MathType.CALCULUS
    elif "probability" in text_lower or "statistics" in text_lower:
        return MathType.STATISTICAL
    elif "equation" in text_lower:
        return MathType.ALGEBRAIC
    else:
        return MathType.GENERAL


def detect_domain(text: str) -> ScientificDomain:
    """Detect the scientific domain of text."""
    text_lower = text.lower()
    if "physics" in text_lower:
        return ScientificDomain.PHYSICS
    elif "chemistry" in text_lower:
        return ScientificDomain.CHEMISTRY
    elif "biology" in text_lower:
        return ScientificDomain.BIOLOGY
    else:
        return ScientificDomain.GENERAL
