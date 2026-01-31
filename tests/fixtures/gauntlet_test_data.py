"""
Test fixtures and data for enhanced gauntlet system testing

Provides realistic test solutions, configurations, and scenarios
for comprehensive validation of the 3-round gauntlet pipeline.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class TestSolution:
    """Test solution with expected outcomes"""
    solution_id: str
    code: str
    problem_type: str
    domain: str
    expected_round1_score: float
    expected_round2_score: Optional[float] = None
    expected_round3_score: Optional[float] = None
    expected_final_score: Optional[float] = None
    should_pass_round1: bool = True
    should_pass_round2: bool = True
    should_pass_round3: bool = True
    expected_termination_round: Optional[int] = None
    artifacts: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


@dataclass
class GauntletTestConfig:
    """Test configuration for gauntlet runs"""
    config_id: str
    round1_threshold: float
    round2_threshold: float
    round3_threshold: float
    round1_weight: float = 0.2
    round2_weight: float = 0.3
    round3_weight: float = 0.5
    enable_early_termination: bool = True
    max_timeout_seconds: int = 600
    enable_artifact_fusion: bool = True


# Sample solutions with simple code to avoid syntax issues

# Perfect solution
PERFECT_CODE_1 = """
def perfect_solution(data):
    '''Excellent implementation with proper error handling'''
    if not data:
        return 0

    result = 0
    for item in data:
        result += item * item

    return result
"""

# Poor solution
POOR_CODE_1 = """
def bad():
    return 1
"""

# Moderate solution
MODERATE_CODE_1 = """
def moderate_solution(data):
    result = 0
    for item in data:
        result += item
    return result
"""

# Good solution
GOOD_CODE_1 = """
def good_solution(data):
    '''Good implementation with some error handling'''
    if not data:
        return 0

    result = 0
    for item in data:
        result += item

    return result
"""

# ============================================================================
# PERFECT SOLUTIONS (Pass all rounds)
# ============================================================================

PERFECT_SOLUTIONS = [
    TestSolution(
        solution_id="perfect_basic_001",
        code=PERFECT_CODE_1,
        problem_type="basic",
        domain="general",
        expected_round1_score=0.95,
        expected_round2_score=0.92,
        expected_round3_score=0.90,
        expected_final_score=0.917,
        should_pass_round1=True,
        should_pass_round2=True,
        should_pass_round3=True,
        expected_termination_round=None,
        artifacts=["analysis", "validation"],
        strengths=["Proper error handling", "Well documented", "Efficient"]
    )
]


# ============================================================================
# POOR SOLUTIONS (Fail Round 1)
# ============================================================================

POOR_SOLUTIONS = [
    TestSolution(
        solution_id="poor_simple_001",
        code=POOR_CODE_1,
        problem_type="simple",
        domain="general",
        expected_round1_score=0.20,
        expected_round2_score=None,
        expected_round3_score=None,
        expected_final_score=0.20,
        should_pass_round1=False,
        should_pass_round2=False,
        should_pass_round3=False,
        expected_termination_round=1,
        weaknesses=["Too simple", "No error handling", "Not documented"]
    )
]


# ============================================================================
# MODERATE SOLUTIONS (Pass R1, Fail R2)
# ============================================================================

MODERATE_SOLUTIONS = [
    TestSolution(
        solution_id="moderate_basic_001",
        code=MODERATE_CODE_1,
        problem_type="basic",
        domain="general",
        expected_round1_score=0.65,
        expected_round2_score=0.45,
        expected_round3_score=None,
        expected_final_score=0.525,
        should_pass_round1=True,
        should_pass_round2=False,
        should_pass_round3=False,
        expected_termination_round=2,
        strengths=["Working code"],
        weaknesses=["No error handling", "Basic implementation"]
    )
]


# ============================================================================
# GOOD SOLUTIONS (Pass R1, R2, Fail R3)
# ============================================================================

GOOD_SOLUTIONS = [
    TestSolution(
        solution_id="good_basic_001",
        code=GOOD_CODE_1,
        problem_type="basic",
        domain="general",
        expected_round1_score=0.82,
        expected_round2_score=0.78,
        expected_round3_score=0.65,
        expected_final_score=0.727,
        should_pass_round1=True,
        should_pass_round2=True,
        should_pass_round3=False,
        expected_termination_round=3,
        strengths=["Error handling", "Clear code"],
        weaknesses=["Could be more robust"]
    )
]


# ============================================================================
# EDGE CASE SOLUTIONS
# ============================================================================

EDGE_CASE_SOLUTIONS = [
    TestSolution(
        solution_id="edge_timeout_001",
        code="import time\ntime.sleep(100)\nreturn 'done'",
        problem_type="timeout",
        domain="general",
        expected_round1_score=0.5,
        expected_round2_score=None,
        expected_round3_score=None,
        expected_final_score=0.5,
        should_pass_round1=False,
        should_pass_round2=False,
        should_pass_round3=False,
        expected_termination_round=1,
        weaknesses=["Timeout", "Too slow"]
    )
]


# ============================================================================
# CONFIGURATIONS
# ============================================================================

STRICT_CONFIG = GauntletTestConfig(
    config_id="strict",
    round1_threshold=0.7,
    round2_threshold=0.8,
    round3_threshold=0.9,
    enable_early_termination=True
)

LENIENT_CONFIG = GauntletTestConfig(
    config_id="lenient",
    round1_threshold=0.3,
    round2_threshold=0.5,
    round3_threshold=0.6,
    enable_early_termination=True
)

BALANCED_CONFIG = GauntletTestConfig(
    config_id="balanced",
    round1_threshold=0.5,
    round2_threshold=0.7,
    round3_threshold=0.8,
    enable_early_termination=True
)

NO_EARLY_TERMINATION_CONFIG = GauntletTestConfig(
    config_id="no_early_term",
    round1_threshold=0.5,
    round2_threshold=0.7,
    round3_threshold=0.8,
    enable_early_termination=False
)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_all_solutions() -> List[TestSolution]:
    """Get all test solutions"""
    return (
        PERFECT_SOLUTIONS +
        POOR_SOLUTIONS +
        MODERATE_SOLUTIONS +
        GOOD_SOLUTIONS +
        EDGE_CASE_SOLUTIONS
    )


def get_solutions_by_category(category: str) -> List[TestSolution]:
    """Get solutions by category"""
    categories = {
        'perfect': PERFECT_SOLUTIONS,
        'poor': POOR_SOLUTIONS,
        'moderate': MODERATE_SOLUTIONS,
        'good': GOOD_SOLUTIONS,
        'edge': EDGE_CASE_SOLUTIONS
    }
    return categories.get(category, [])


def get_solution_by_id(solution_id: str) -> Optional[TestSolution]:
    """Get specific solution by ID"""
    for solution in get_all_solutions():
        if solution.solution_id == solution_id:
            return solution
    return None


if __name__ == '__main__':
    # Demo: print all solutions
    print("Test Solutions Available:")
    print("=" * 60)

    for category in ['perfect', 'poor', 'moderate', 'good', 'edge']:
        solutions = get_solutions_by_category(category)
        print(f"\n{category.upper()} ({len(solutions)} solutions):")
        for sol in solutions:
            print(f"  - {sol.solution_id}")
            print(f"    Expected R1: {sol.expected_round1_score}")
            if sol.expected_termination_round:
                print(f"    Termination: Round {sol.expected_termination_round}")

    print("\n" + "=" * 60)
    print(f"Total solutions: {len(get_all_solutions())}")
