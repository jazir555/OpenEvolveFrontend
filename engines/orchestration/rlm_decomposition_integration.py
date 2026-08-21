"""
RLM-Decomposition Integration

RLM powers the Blue/Red/Gold team execution in the decomposition workflow.

Blue Team: RLM executes code to solve sub-problems
Red Team: RLM adversarially tests solutions with code
Gold Team: RLM verifies with test code execution
"""
from __future__ import annotations


import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

try:
    from rlm_core_integration import RLMExecutionEngine, RLMIntegrationConfig
    RLM_CORE_AVAILABLE = True
except ImportError:
    RLM_CORE_AVAILABLE = False


# ============================================================================
# RLM BLUE TEAM EXECUTOR
# ============================================================================

class RLMBlueTeamExecutor:
    """
    Blue Team powered by RLM.
    
    RLM solves sub-problems by:
    1. Writing solution code
    2. Executing in REPL
    3. Iterating until working
    4. Spawning sub-LMs if needed
    """
    
    def __init__(self, config: Optional['RLMIntegrationConfig'] = None):
        if not RLM_CORE_AVAILABLE:
            raise ImportError("RLM core integration not available. Install rlm_core_integration.")
        self.config = config or RLMIntegrationConfig()
        self.rlm = RLMExecutionEngine(self.config)
    
    def solve(
        self,
        subproblem: str,
        context: Dict[str, Any],
        attempt: int = 1
    ) -> 'BlueTeamResult':
        """
        Solve sub-problem using RLM.
        
        RLM can execute code and spawn sub-calls automatically.
        """
        solve_prompt = f"""
        Solve this sub-problem. Write and execute code as needed.
        If too complex, use recursive sub-calls.
        
        Sub-problem: {subproblem}
        Context: {json.dumps(context)}
        Attempt: {attempt}
        """
        
        result = self.rlm.execute(task=solve_prompt)
        
        return BlueTeamResult(
            solution=result.solution,
            code_executed=self._extract_code(result),
            iterations=len(result.iterations),
            confidence=self._estimate_confidence(result)
        )
    
    def _extract_code(self, result: Any) -> List[str]:
        """Extract code blocks from RLM result."""
        code_blocks = []
        if hasattr(result, 'code_blocks'):
            code_blocks = result.code_blocks
        elif hasattr(result, 'iterations'):
            for iteration in result.iterations:
                if hasattr(iteration, 'code'):
                    code_blocks.append(iteration.code)
        return code_blocks
    
    def _estimate_confidence(self, result: Any) -> float:
        """Estimate confidence from RLM result."""
        if hasattr(result, 'confidence'):
            return result.confidence
        elif hasattr(result, 'success') and result.success:
            return 0.9
        return 0.5


# ============================================================================
# RLM RED TEAM ANALYZER
# ============================================================================

class RLMRedTeamAnalyzer:
    """
    Red Team powered by RLM.
    
    RLM adversarially tests by:
    1. Writing attack/test code
    2. Executing against solution
    3. Finding edge cases
    """
    
    def __init__(self, config: Optional['RLMIntegrationConfig'] = None):
        if not RLM_CORE_AVAILABLE:
            raise ImportError("RLM core integration not available. Install rlm_core_integration.")
        self.config = config or RLMIntegrationConfig()
        self.rlm = RLMExecutionEngine(self.config)
    
    def analyze(
        self,
        solution: str,
        attack_vectors: List[str]
    ) -> 'RedTeamResult':
        """
        Adversarial analysis using RLM.
        
        RLM writes test code to find vulnerabilities.
        """
        attack_prompt = f"""
        Attack this solution. Write and run test code to find:
        - Edge cases
        - Input validation issues
        - Security vulnerabilities
        - Performance problems
        
        Solution: {solution}
        Attack vectors: {attack_vectors}
        """
        
        result = self.rlm.execute(task=attack_prompt)
        
        return RedTeamResult(
            vulnerabilities_found=self._extract_vulnerabilities(result),
            test_cases_run=self._count_tests(result),
            attack_success=self._has_failures(result),
            recommendations=result.solution
        )
    
    def _extract_vulnerabilities(self, result: Any) -> List[str]:
        """Extract found vulnerabilities from RLM result."""
        vulnerabilities = []
        if hasattr(result, 'vulnerabilities'):
            vulnerabilities = result.vulnerabilities
        elif hasattr(result, 'issues_found'):
            vulnerabilities = result.issues_found
        return vulnerabilities
    
    def _count_tests(self, result: Any) -> int:
        """Count number of test cases executed."""
        if hasattr(result, 'tests_run'):
            return result.tests_run
        elif hasattr(result, 'iterations'):
            return len(result.iterations)
        return 0
    
    def _has_failures(self, result: Any) -> bool:
        """Check if attacks found failures."""
        if hasattr(result, 'has_failures'):
            return result.has_failures
        elif hasattr(result, 'vulnerabilities'):
            return len(result.vulnerabilities) > 0
        return False


# ============================================================================
# RLM GOLD TEAM VERIFIER
# ============================================================================

class RLMGoldTeamVerifier:
    """
    Gold Team powered by RLM.
    
    RLM verifies by:
    1. Writing verification tests
    2. Executing against criteria
    3. Checking all requirements
    """
    
    def __init__(self, config: Optional['RLMIntegrationConfig'] = None):
        if not RLM_CORE_AVAILABLE:
            raise ImportError("RLM core integration not available. Install rlm_core_integration.")
        self.config = config or RLMIntegrationConfig()
        self.rlm = RLMExecutionEngine(self.config)
    
    def verify(
        self,
        solution: str,
        criteria: List[str]
    ) -> 'GoldTeamResult':
        """
        Verify solution using RLM.
        
        RLM writes and runs verification code.
        """
        verify_prompt = f"""
        Verify this solution meets all criteria.
        Write and run comprehensive tests.
        Report pass/fail for each criterion.
        
        Solution: {solution}
        Criteria: {criteria}
        """
        
        result = self.rlm.execute(task=verify_prompt)
        
        return GoldTeamResult(
            passed="PASS" in result.solution.upper(),
            criteria_met=self._extract_criteria_status(result),
            test_coverage=self._estimate_coverage(result),
            verification_report=result.solution
        )
    
    def _extract_criteria_status(self, result: Any) -> Dict[str, bool]:
        """Extract criteria pass/fail status from result."""
        criteria_status = {}
        if hasattr(result, 'criteria_results'):
            criteria_status = result.criteria_results
        return criteria_status
    
    def _estimate_coverage(self, result: Any) -> float:
        """Estimate test coverage from RLM result."""
        if hasattr(result, 'coverage'):
            return result.coverage
        elif hasattr(result, 'tests_passed') and hasattr(result, 'tests_total'):
            if result.tests_total > 0:
                return result.tests_passed / result.tests_total
        return 0.0


# ============================================================================
# RLM GAUNTLET RUNNER
# ============================================================================

class RLMGauntletRunner:
    """
    3-round gauntlet using RLM for all teams.
    """
    
    def __init__(self, config: Optional['RLMIntegrationConfig'] = None):
        self.blue = RLMBlueTeamExecutor(config)
        self.red = RLMRedTeamAnalyzer(config)
        self.gold = RLMGoldTeamVerifier(config)
    
    def run(
        self,
        subproblem: str,
        max_attempts: int = 3
    ) -> 'GauntletResult':
        """
        Run full gauntlet (Blue -> Red -> Gold).
        
        If Red finds issues, Blue retries with feedback.
        """
        for attempt in range(1, max_attempts + 1):
            # Round 1: Blue solves
            blue_result = self.blue.solve(subproblem, {}, attempt)
            
            # Round 2: Red attacks
            red_result = self.red.analyze(
                blue_result.solution,
                ["edge_cases", "security", "performance"]
            )
            
            # If Red finds issues and we have attempts left, retry
            if red_result.attack_success and attempt < max_attempts:
                feedback = f"Red Team found issues: {red_result.vulnerabilities_found}"
                subproblem = f"{subproblem}\n\nPrevious issues to fix: {feedback}"
                continue
            
            # Round 3: Gold verifies
            gold_result = self.gold.verify(
                blue_result.solution,
                ["correctness", "completeness", "robustness"]
            )
            
            return GauntletResult(
                solution=blue_result.solution,
                blue_score=blue_result.confidence,
                red_score=1.0 - len(red_result.vulnerabilities_found) * 0.1,
                gold_score=gold_result.test_coverage,
                passed=gold_result.passed,
                attempts=attempt
            )
        
        # If we exhaust all attempts without passing, return last attempt's results
        return GauntletResult(
            solution=blue_result.solution,
            blue_score=blue_result.confidence,
            red_score=1.0 - len(red_result.vulnerabilities_found) * 0.1,
            gold_score=0.0,
            passed=False,
            attempts=max_attempts
        )


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class BlueTeamResult:
    solution: str
    code_executed: List[str]
    iterations: int
    confidence: float


@dataclass
class RedTeamResult:
    vulnerabilities_found: List[str]
    test_cases_run: int
    attack_success: bool
    recommendations: str


@dataclass
class GoldTeamResult:
    passed: bool
    criteria_met: Dict[str, bool]
    test_coverage: float
    verification_report: str


@dataclass
class GauntletResult:
    solution: str
    blue_score: float
    red_score: float
    gold_score: float
    passed: bool
    attempts: int


# ============================================================================
# FACTORY
# ============================================================================

def create_rlm_gauntlet_runner(**kwargs) -> RLMGauntletRunner:
    """Create RLM-powered gauntlet runner."""
    if not RLM_CORE_AVAILABLE:
        raise ImportError(
            "RLM core integration not available. "
            "Please install rlm_core_integration to use this feature."
        )
    from rlm_core_integration import RLMIntegrationConfig
    config = RLMIntegrationConfig(**kwargs)
    return RLMGauntletRunner(config)


# ============================================================================
# BACKWARD COMPATIBILITY / FALLBACK
# ============================================================================

class MockRLMExecutionEngine:
    """
    Mock RLM execution engine for when RLM core is not available.
    Provides a fallback interface that raises informative errors.
    """
    
    def __init__(self, config: Any = None):
        self.config = config
    
    def execute(self, task: str) -> Any:
        """Raise informative error about missing RLM core."""
        raise ImportError(
            "RLM core integration is not available. "
            "To use RLM-powered decomposition:\n"
            "1. Install the rlm_core_integration package\n"
            "2. Ensure RLMExecutionEngine and RLMIntegrationConfig are importable\n"
            "3. Or use the standard decomposition_engine without RLM"
        )


# Use mock if RLM core not available
if not RLM_CORE_AVAILABLE:
    RLMExecutionEngine = MockRLMExecutionEngine  # type: ignore
    RLMIntegrationConfig = dict  # type: ignore
