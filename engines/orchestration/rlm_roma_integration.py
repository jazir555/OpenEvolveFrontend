"""
RLM-ROMA Integration

Replaces ROMA's standard solver with RLM's recursive execution engine.

ROMA handles:
- Problem decomposition (what to solve)
- Result aggregation (combining sub-solutions)

RLM handles:
- Recursive execution (how to solve)
- Code execution in REPL
- Sub-LM spawning for sub-problems

Author: Claude Code
Date: 2026-02-04
Version: 1.0.0
"""
from __future__ import annotations


import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# RLM imports (recursive language model execution engine)
try:
    from rlm_core_integration import (
        RLMExecutionEngine,
        RLMIntegrationConfig,
        RLMExecutionResult,
        create_rlm_engine,
    )
    RLM_CORE_AVAILABLE = True
except ImportError:
    RLM_CORE_AVAILABLE = False
    RLMExecutionEngine = None  # type: ignore
    RLMIntegrationConfig = None  # type: ignore
    RLMExecutionResult = None  # type: ignore
    create_rlm_engine = None  # type: ignore
    logger.warning("RLM core integration not available. RLM features will be disabled.")

# ROMA imports
try:
    from roma_openevolve_integration import ROMAOpenEvolveAdapter
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    ROMAOpenEvolveAdapter = None  # type: ignore
    logger.warning("ROMA integration not available. ROMA features will be disabled.")


# ============================================================================
# RESULT TYPES
# ============================================================================

@dataclass
class ROMASolution:
    """ROMA solution result with RLM execution trace."""
    answer: str
    sub_solutions: List[Any] = field(default_factory=list)
    execution_trace: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ROMAAnalysis:
    """ROMA problem analysis result."""
    structure: str
    subproblems: List[str] = field(default_factory=list)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    complexity_estimate: Optional[str] = None
    recommended_strategy: Optional[str] = None


@dataclass
class ROMACritique:
    """ROMA solution critique result."""
    critique: str
    issues: List[str] = field(default_factory=list)
    confidence: float = 0.8
    severity: str = "medium"  # low, medium, high, critical


@dataclass
class ROMAVerification:
    """ROMA solution verification result."""
    passed: bool = False
    verification_details: str = ""
    test_results: List[Any] = field(default_factory=list)
    coverage: float = 0.0


# ============================================================================
# RLM AS ROMA SOLVER
# ============================================================================

class RLMROMASolver:
    """
    RLM as ROMA's recursive solver.
    
    Traditional ROMA solver: Direct LLM calls
    RLM solver: Recursive LM with code execution and sub-LM spawning
    
    This enables ROMA to:
    1. Execute code during solving
    2. Spawn child LMs for sub-problems automatically
    3. Iterate until solution converges
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize RLM ROMA Solver.
        
        Args:
            config: RLMIntegrationConfig instance or None for defaults
        """
        if not RLM_CORE_AVAILABLE:
            raise ImportError(
                "RLM core integration is required but not available. "
                "Please install rlm_core_integration module."
            )
        
        self.config = config or RLMIntegrationConfig()
        self.rlm_engine = RLMExecutionEngine(self.config)
        logger.info("RLM ROMA Solver initialized with model: %s", 
                   getattr(self.config, 'model_name', 'default'))
    
    def solve(
        self,
        problem: str,
        decomposition: Optional[List[str]] = None
    ) -> ROMASolution:
        """
        Solve problem using RLM.
        
        If decomposition provided, RLM solves each sub-problem.
        If not, RLM may decompose recursively itself.
        
        Args:
            problem: The main problem to solve
            decomposition: Optional list of sub-problems
            
        Returns:
            ROMASolution with answer and execution trace
        """
        logger.info("RLM solving problem: %s", problem[:100] + "..." if len(problem) > 100 else problem)
        
        if decomposition:
            # Solve each sub-problem via RLM
            logger.debug("Solving %d sub-problems", len(decomposition))
            sub_solutions = []
            execution_traces = []
            
            for i, subproblem in enumerate(decomposition):
                logger.debug("Solving sub-problem %d/%d", i + 1, len(decomposition))
                result = self.rlm_engine.solve_subproblem(
                    subproblem=subproblem,
                    parent_context={"parent_problem": problem, "index": i}
                )
                sub_solutions.append(result)
                if hasattr(result, 'execution_trace'):
                    execution_traces.append(result.execution_trace)
            
            # Aggregate solutions
            aggregated = self._aggregate_solutions(sub_solutions)
            aggregated.execution_trace = {
                "sub_problem_count": len(decomposition),
                "sub_traces": execution_traces
            }
            return aggregated
        else:
            # Let RLM handle everything recursively
            logger.debug("Letting RLM handle full recursive solving")
            result = self.rlm_engine.execute(task=problem)
            return ROMASolution(
                answer=result.solution if hasattr(result, 'solution') else str(result),
                sub_solutions=[],
                execution_trace=result.execution_trace if hasattr(result, 'execution_trace') else {}
            )
    
    def analyze(self, problem: str) -> ROMAAnalysis:
        """
        Analyze problem structure using RLM.
        
        RLM can write code to analyze the problem,
        not just reason about it textually.
        
        Args:
            problem: The problem to analyze
            
        Returns:
            ROMAAnalysis with structure and subproblems
        """
        logger.info("RLM analyzing problem structure")
        
        analysis_prompt = f"""
        Analyze this problem and identify:
        1. Key components/sub-problems
        2. Dependencies between components
        3. Estimated complexity
        
        Problem: {problem}
        
        Write code if needed to explore the problem space.
        Return your analysis in a structured format.
        """
        
        result = self.rlm_engine.execute(task=analysis_prompt)
        solution_text = result.solution if hasattr(result, 'solution') else str(result)
        
        return ROMAAnalysis(
            structure=solution_text,
            subproblems=self._extract_subproblems(result),
            dependencies=[],
            complexity_estimate=self._extract_complexity(solution_text)
        )
    
    def critique(self, solution: ROMASolution) -> ROMACritique:
        """
        Critique solution using RLM.
        
        RLM can test the solution with code execution.
        
        Args:
            solution: The solution to critique
            
        Returns:
            ROMACritique with issues and confidence
        """
        logger.info("RLM critiquing solution")
        
        critique_prompt = f"""
        Critique this solution. Test it with code if possible.
        Identify:
        1. Potential errors
        2. Edge cases not handled
        3. Performance issues
        4. Security vulnerabilities
        
        Solution: {solution.answer}
        
        Provide a detailed critique with severity ratings (low/medium/high/critical).
        """
        
        result = self.rlm_engine.execute(task=critique_prompt)
        critique_text = result.solution if hasattr(result, 'solution') else str(result)
        
        return ROMACritique(
            critique=critique_text,
            issues=self._extract_issues(result),
            confidence=self._calculate_confidence(critique_text),
            severity=self._extract_severity(critique_text)
        )
    
    def verify(self, solution: ROMASolution, criteria: List[str]) -> ROMAVerification:
        """
        Verify solution against criteria using RLM.
        
        RLM can write test code to verify.
        
        Args:
            solution: The solution to verify
            criteria: List of criteria to verify against
            
        Returns:
            ROMAVerification with pass/fail status
        """
        logger.info("RLM verifying solution against %d criteria", len(criteria))
        
        criteria_text = "\n".join(f"- {c}" for c in criteria)
        verify_prompt = f"""
        Verify this solution meets all criteria.
        Write and run test code if possible.
        
        Solution: {solution.answer}
        
        Criteria:
        {criteria_text}
        
        For each criterion, indicate PASS or FAIL with explanation.
        Provide an overall PASS/FAIL verdict.
        """
        
        result = self.rlm_engine.execute(task=verify_prompt)
        verification_text = result.solution if hasattr(result, 'solution') else str(result)
        
        # Determine if passed based on content
        passed = "PASS" in verification_text.upper() and "FAIL" not in verification_text.upper().split("PASS")[0]
        
        return ROMAVerification(
            passed=passed,
            verification_details=verification_text,
            test_results=[],
            coverage=self._estimate_coverage(verification_text)
        )
    
    def _aggregate_solutions(self, sub_solutions: List[Any]) -> ROMASolution:
        """Aggregate multiple sub-solutions into a single solution."""
        if not sub_solutions:
            return ROMASolution(answer="", sub_solutions=[])
        
        # Combine answers
        answers = []
        for sol in sub_solutions:
            if hasattr(sol, 'solution'):
                answers.append(sol.solution)
            elif hasattr(sol, 'answer'):
                answers.append(sol.answer)
            else:
                answers.append(str(sol))
        
        combined = "\n\n".join(f"Part {i+1}:\n{ans}" for i, ans in enumerate(answers))
        
        return ROMASolution(
            answer=combined,
            sub_solutions=sub_solutions
        )
    
    def _extract_subproblems(self, result: Any) -> List[str]:
        """Extract subproblems from RLM result."""
        # This would parse the RLM output to identify subproblems
        # For now, return empty list - can be enhanced with NLP parsing
        return []
    
    def _extract_issues(self, result: Any) -> List[str]:
        """Extract issues from RLM critique result."""
        # This would parse the RLM output to identify issues
        # For now, return empty list - can be enhanced with NLP parsing
        return []
    
    def _extract_complexity(self, text: str) -> Optional[str]:
        """Extract complexity estimate from analysis text."""
        text_lower = text.lower()
        if "complex" in text_lower or "difficult" in text_lower:
            return "high"
        elif "moderate" in text_lower or "medium" in text_lower:
            return "medium"
        elif "simple" in text_lower or "easy" in text_lower:
            return "low"
        return None
    
    def _extract_severity(self, text: str) -> str:
        """Extract severity level from critique text."""
        text_lower = text.lower()
        if "critical" in text_lower:
            return "critical"
        elif "high" in text_lower:
            return "high"
        elif "low" in text_lower:
            return "low"
        return "medium"
    
    def _calculate_confidence(self, text: str) -> float:
        """Calculate confidence score from critique text."""
        # Simple heuristic - more detailed critiques with specific issues = higher confidence
        text_lower = text.lower()
        confidence = 0.8
        
        if "certain" in text_lower or "definitely" in text_lower:
            confidence = 0.9
        elif "uncertain" in text_lower or "unclear" in text_lower:
            confidence = 0.6
        
        return confidence
    
    def _estimate_coverage(self, text: str) -> float:
        """Estimate test coverage from verification text."""
        # Simple heuristic - more detailed verification = higher coverage
        if len(text) > 500:
            return 0.8
        elif len(text) > 200:
            return 0.6
        return 0.4


# ============================================================================
# RLM-ROMA ADAPTER
# ============================================================================

class RLMROMAAdapter:
    """
    Adapter integrating RLM into ROMA's workflow.
    
    Maintains ROMA's phase interface but uses RLM internally.
    """
    
    def __init__(self, rlm_config: Optional[Any] = None):
        """
        Initialize RLM-ROMA Adapter.
        
        Args:
            rlm_config: RLM configuration or None for defaults
        """
        self.solver = RLMROMASolver(rlm_config)
        logger.info("RLM-ROMA Adapter initialized")
    
    def phase_1_setup(self, problem: str) -> ROMAAnalysis:
        """
        Phase 1: Setup and analyze with RLM.
        
        Args:
            problem: The problem to analyze
            
        Returns:
            ROMAAnalysis with problem structure
        """
        logger.info("Phase 1: Setup and analysis")
        return self.solver.analyze(problem)
    
    def phase_2_solve(self, problem: str, decomposition: List[str]) -> ROMASolution:
        """
        Phase 2: Solve with RLM.
        
        Args:
            problem: The main problem
            decomposition: List of sub-problems to solve
            
        Returns:
            ROMASolution with aggregated results
        """
        logger.info("Phase 2: Solving with %d sub-problems", len(decomposition))
        return self.solver.solve(problem, decomposition)
    
    def phase_3_critique(self, solution: ROMASolution) -> ROMACritique:
        """
        Phase 3: Critique with RLM.
        
        Args:
            solution: The solution to critique
            
        Returns:
            ROMACritique with identified issues
        """
        logger.info("Phase 3: Critique")
        return self.solver.critique(solution)
    
    def phase_4_verify(self, solution: ROMASolution, criteria: List[str]) -> ROMAVerification:
        """
        Phase 4: Verify with RLM.
        
        Args:
            solution: The solution to verify
            criteria: List of verification criteria
            
        Returns:
            ROMAVerification with pass/fail status
        """
        logger.info("Phase 4: Verification against %d criteria", len(criteria))
        return self.solver.verify(solution, criteria)
    
    def phase_5_reassemble(self, solutions: List[ROMASolution]) -> ROMASolution:
        """
        Phase 5: Reassemble solutions.
        
        Uses RLM to intelligently aggregate partial solutions.
        
        Args:
            solutions: List of partial solutions
            
        Returns:
            ROMASolution with reassembled answer
        """
        logger.info("Phase 5: Reassembling %d solutions", len(solutions))
        
        if not solutions:
            return ROMASolution(answer="", sub_solutions=[])
        
        if len(solutions) == 1:
            return solutions[0]
        
        # Use RLM to intelligently aggregate
        answers = [s.answer for s in solutions]
        agg_prompt = f"""
        Combine these partial solutions into a coherent, unified solution:
        
        {json.dumps(answers, indent=2)}
        
        Ensure the final solution:
        1. Integrates insights from all partial solutions
        2. Maintains logical flow and coherence
        3. Resolves any contradictions
        4. Presents a complete answer to the original problem
        """
        
        result = self.solver.rlm_engine.execute(task=agg_prompt)
        aggregated_answer = result.solution if hasattr(result, 'solution') else str(result)
        
        return ROMASolution(
            answer=aggregated_answer,
            sub_solutions=solutions,
            execution_trace={"reassembled_from": len(solutions)}
        )
    
    def run_full_workflow(
        self,
        problem: str,
        criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run complete ROMA workflow with RLM solver.
        
        Args:
            problem: The problem to solve
            criteria: Optional verification criteria
            
        Returns:
            Dictionary with all phase results
        """
        logger.info("Running full RLM-ROMA workflow")
        
        # Phase 1: Analysis
        analysis = self.phase_1_setup(problem)
        
        # Phase 2: Solve
        solution = self.phase_2_solve(problem, analysis.subproblems)
        
        # Phase 3: Critique
        critique = self.phase_3_critique(solution)
        
        # Phase 4: Verify (if criteria provided)
        verification = None
        if criteria:
            verification = self.phase_4_verify(solution, criteria)
        
        return {
            "analysis": analysis,
            "solution": solution,
            "critique": critique,
            "verification": verification,
            "success": verification.passed if verification else True
        }


# ============================================================================
# FACTORY
# ============================================================================

def create_rlm_roma_solver(
    model: str = "gpt-4o",
    max_depth: int = 3,
    temperature: float = 0.7
) -> RLMROMASolver:
    """
    Create RLM-powered ROMA solver.
    
    Args:
        model: Model name to use (e.g., "gpt-4o", "claude-3-opus")
        max_depth: Maximum recursion depth for RLM
        temperature: Sampling temperature
        
    Returns:
        Configured RLMROMASolver instance
        
    Raises:
        ImportError: If RLM core integration is not available
    """
    if not RLM_CORE_AVAILABLE:
        raise ImportError(
            "RLM core integration is required but not available. "
            "Please install rlm_core_integration module."
        )
    
    config = RLMIntegrationConfig(
        model_name=model,
        max_depth=max_depth,
        temperature=temperature
    )
    return RLMROMASolver(config)


def create_rlm_roma_adapter(
    model: str = "gpt-4o",
    max_depth: int = 3,
    temperature: float = 0.7
) -> RLMROMAAdapter:
    """
    Create RLM-ROMA adapter.
    
    Args:
        model: Model name to use
        max_depth: Maximum recursion depth
        temperature: Sampling temperature
        
    Returns:
        Configured RLMROMAAdapter instance
    """
    config = RLMIntegrationConfig(
        model_name=model,
        max_depth=max_depth,
        temperature=temperature
    )
    return RLMROMAAdapter(config)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def solve_with_rlm_roma(
    problem: str,
    model: str = "gpt-4o",
    max_depth: int = 3
) -> ROMASolution:
    """
    Convenience function to solve a problem with RLM-ROMA.
    
    Args:
        problem: The problem to solve
        model: Model to use
        max_depth: Maximum recursion depth
        
    Returns:
        ROMASolution with answer
    """
    solver = create_rlm_roma_solver(model=model, max_depth=max_depth)
    return solver.solve(problem)


def analyze_with_rlm_roma(
    problem: str,
    model: str = "gpt-4o"
) -> ROMAAnalysis:
    """
    Convenience function to analyze a problem with RLM-ROMA.
    
    Args:
        problem: The problem to analyze
        model: Model to use
        
    Returns:
        ROMAAnalysis with problem structure
    """
    solver = create_rlm_roma_solver(model=model, max_depth=2)
    return solver.analyze(problem)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    if not RLM_CORE_AVAILABLE:
        print("RLM core integration not available. Please install rlm_core_integration.")
        exit(1)
    
    # Example problem
    example_problem = "Calculate the first 10 Fibonacci numbers and analyze their properties"
    
    print(f"Testing RLM-ROMA integration with problem: {example_problem}")
    print("-" * 60)
    
    try:
        # Create solver
        solver = create_rlm_roma_solver(model="gpt-4o", max_depth=2)
        
        # Analyze
        print("\n1. ANALYSIS:")
        analysis = solver.analyze(example_problem)
        print(f"Structure: {analysis.structure[:200]}...")
        print(f"Subproblems: {analysis.subproblems}")
        
        # Solve
        print("\n2. SOLUTION:")
        solution = solver.solve(example_problem, analysis.subproblems)
        print(f"Answer: {solution.answer[:300]}...")
        
        # Critique
        print("\n3. CRITIQUE:")
        critique = solver.critique(solution)
        print(f"Critique: {critique.critique[:200]}...")
        print(f"Issues: {critique.issues}")
        print(f"Severity: {critique.severity}")
        
        print("\nRLM-ROMA integration test complete!")
        
    except Exception as e:
        logger.error("Error during test: %s", e)
        raise
