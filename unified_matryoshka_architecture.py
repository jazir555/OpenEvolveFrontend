"""
Unified Matryoshka Architecture

Positions Matryoshka as a generalized Recursive Language Model (RLM) execution engine
that integrates with:
- ROMA: Recursive decomposition and solving
- Decomposition Workflow: Blue/Red/Gold team execution
- MDAP/MAKER: Voting consensus and error correction
- Unified Memory: Context rot prevention

Matryoshka is NOT just for documents - it's a universal execution engine
for any problem space that can be represented symbolically and explored iteratively.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, Tuple

# ============================================================================
# OPTIONAL IMPORTS - Only import if available
# ============================================================================

# Core Matryoshka
try:
    from matryoshka_core import MatryoshkaExecutionEngine, MatryoshkaExecutionConfig
    MATRYOSHKA_EXECUTION_AVAILABLE = True
except ImportError:
    MATRYOSHKA_EXECUTION_AVAILABLE = False
    
    # Stub classes for type hints
    @dataclass
    class MatryoshkaExecutionConfig:
        max_iterations: int = 10
        exploration_depth: int = 3
        verbose: bool = False
    
    class MatryoshkaExecutionEngine:
        def __init__(self, config): pass
        def execute(self, task, space, context): pass

# ROMA Integration
try:
    from roma_matryoshka_adapter import ROMAMatryoshkaAdapter, ROMAMatryoshkaConfig
    MATRYOSHKA_ROMA_AVAILABLE = True
except ImportError:
    MATRYOSHKA_ROMA_AVAILABLE = False
    
    @dataclass
    class ROMAMatryoshkaConfig:
        enable_analysis: bool = True
        enable_critique: bool = True
    
    class ROMAMatryoshkaAdapter:
        def __init__(self, config): pass
        def phase_1_analyze_problem(self, problem, context): pass
        def phase_2_solve_subproblem(self, subproblem, context): pass
        def phase_3_critique_solution(self, solution, context): pass
        def phase_4_verify_solution(self, solution, context): pass

# Decomposition/Team Execution
try:
    from matryoshka_team_executor import MatryoshkaTeamExecutor
    MATRYOSHKA_TEAM_AVAILABLE = True
except ImportError:
    MATRYOSHKA_TEAM_AVAILABLE = False
    
    class MatryoshkaTeamExecutor:
        def __init__(self, config): pass
        def blue_team_solve(self, subproblem, context): pass
        def red_team_critique(self, solution, context): pass
        def gold_team_verify(self, solution, criteria, context): pass

# MDAP/MAKER Integration
try:
    from matryoshka_mdap_integration import MatryoshkaVotingExplorer, MDAPMatryoshkaConfig
    MATRYOSHKA_MDAP_AVAILABLE = True
except ImportError:
    MATRYOSHKA_MDAP_AVAILABLE = False
    
    @dataclass
    class MDAPMatryoshkaConfig:
        voting_threshold: float = 0.7
        max_voting_rounds: int = 5
    
    class MatryoshkaVotingExplorer:
        def __init__(self, config): pass
        def explore_with_voting(self, task, strategies, k_ahead): pass

# Memory Integration
try:
    from matryoshka_memory_bridge import MatryoshkaMemoryBridge
    MATRYOSHKA_MEMORY_AVAILABLE = True
except ImportError:
    MATRYOSHKA_MEMORY_AVAILABLE = False
    
    class MatryoshkaMemoryBridge:
        def __init__(self, storage_path): pass

# Problem Space Types
try:
    from problem_space import ProblemSpace, SubProblem
except ImportError:
    class ProblemSpace:
        name: str = "abstract"
        dimensions: List[str] = field(default_factory=list)
    
    class SubProblem:
        id: str = ""
        description: str = ""
        success_criteria: List[str] = field(default_factory=list)

# Context Types
try:
    from roma_types import ROMAContext
except ImportError:
    ROMAContext = Dict[str, Any]

try:
    from team_types import TeamContext
except ImportError:
    TeamContext = Dict[str, Any]

# Result Types
try:
    from execution_types import (
        ExecutionResult, ProblemAnalysisResult, SubProblemSolution,
        CritiqueResult, VerificationResult, BlueTeamResult, RedTeamResult,
        GoldTeamResult, GauntletResult, GauntletConfig, VotedExplorationResult,
        SolverResult, DecompositionResult, ExplorationStrategy
    )
except ImportError:
    ExecutionResult = Dict[str, Any]
    ProblemAnalysisResult = Dict[str, Any]
    SubProblemSolution = Dict[str, Any]
    CritiqueResult = Dict[str, Any]
    VerificationResult = Dict[str, Any]
    BlueTeamResult = Dict[str, Any]
    RedTeamResult = Dict[str, Any]
    GoldTeamResult = Dict[str, Any]
    GauntletResult = Dict[str, Any]
    GauntletConfig = Dict[str, Any]
    VotedExplorationResult = Dict[str, Any]
    SolverResult = Dict[str, Any]
    DecompositionResult = Dict[str, Any]
    ExplorationStrategy = Dict[str, Any]


# ============================================================================
# UNIFIED CONFIGURATION
# ============================================================================

@dataclass
class UnifiedMatryoshkaConfig:
    """
    Master configuration for Matryoshka across all systems.
    
    All features are optional - configure what you need.
    """
    # Core Matryoshka execution
    execution_config: MatryoshkaExecutionConfig = field(
        default_factory=MatryoshkaExecutionConfig
    )
    
    # ROMA integration
    roma_config: Optional[ROMAMatryoshkaConfig] = None
    enable_roma_integration: bool = True
    
    # Decomposition workflow integration
    enable_decomposition_integration: bool = True
    decomposition_team_role: str = "all"  # "blue", "red", "gold", "all"
    
    # MDAP/MAKER integration
    enable_mdap_integration: bool = True
    mdap_voting_config: Optional[MDAPMatryoshkaConfig] = None
    
    # Unified memory integration
    enable_memory_integration: bool = True
    memory_storage_path: Optional[str] = None
    
    # Global settings
    auto_detect_problem_space: bool = True
    fallback_to_standard: bool = True
    verbose_logging: bool = False


# ============================================================================
# UNIFIED EXECUTION INTERFACE
# ============================================================================

class UnifiedMatryoshkaExecutor:
    """
    Single unified interface for Matryoshka across all systems.
    
    Usage:
        executor = UnifiedMatryoshkaExecutor(config)
        
        # For ROMA workflows
        result = executor.execute_for_roma(problem, phase="solve")
        
        # For Decomposition workflows  
        result = executor.execute_for_decomposition(subproblem, team="blue")
        
        # For MDAP/MAKER workflows
        result = executor.execute_for_mdap(candidates, voting_round=1)
        
        # Direct execution (any problem space)
        result = executor.execute(task, problem_space)
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.config = config or UnifiedMatryoshkaConfig()
        
        # Initialize all subsystems (if available)
        self._init_execution_engine()
        self._init_roma_integration()
        self._init_decomposition_integration()
        self._init_mdap_integration()
        self._init_memory_integration()
    
    def _init_execution_engine(self):
        """Initialize core Matryoshka execution engine."""
        if MATRYOSHKA_EXECUTION_AVAILABLE:
            self.execution_engine = MatryoshkaExecutionEngine(
                self.config.execution_config
            )
        else:
            self.execution_engine = None
    
    def _init_roma_integration(self):
        """Initialize ROMA integration."""
        if self.config.enable_roma_integration and MATRYOSHKA_ROMA_AVAILABLE:
            roma_config = self.config.roma_config or ROMAMatryoshkaConfig()
            self.roma_adapter = ROMAMatryoshkaAdapter(roma_config)
        else:
            self.roma_adapter = None
    
    def _init_decomposition_integration(self):
        """Initialize Decomposition integration."""
        if self.config.enable_decomposition_integration and MATRYOSHKA_TEAM_AVAILABLE:
            self.team_executor = MatryoshkaTeamExecutor(
                self.config.execution_config
            )
        else:
            self.team_executor = None
    
    def _init_mdap_integration(self):
        """Initialize MDAP integration."""
        if self.config.enable_mdap_integration and MATRYOSHKA_MDAP_AVAILABLE:
            self.mdap_config = self.config.mdap_voting_config or MDAPMatryoshkaConfig()
        else:
            self.mdap_config = None
    
    def _init_memory_integration(self):
        """Initialize Memory integration."""
        if self.config.enable_memory_integration and MATRYOSHKA_MEMORY_AVAILABLE:
            self.memory_bridge = MatryoshkaMemoryBridge(
                self.config.memory_storage_path or "./matryoshka_memory"
            )
        else:
            self.memory_bridge = None
    
    # ========================================================================
    # PUBLIC API - Unified Interface
    # ========================================================================
    
    def execute(
        self,
        task: str,
        problem_space: Optional[ProblemSpace] = None,
        context: Optional[Dict] = None
    ) -> ExecutionResult:
        """
        Execute task directly using Matryoshka.
        
        Lowest-level interface - use for any problem space.
        """
        if self.execution_engine is None:
            raise RuntimeError("Matryoshka execution engine not available")
        
        space = problem_space or self._auto_detect_space(task, context)
        return self.execution_engine.execute(task, space, context)
    
    def execute_for_roma(
        self,
        problem: str,
        phase: str = "solve",
        subproblem: Optional[SubProblem] = None,
        context: Optional[ROMAContext] = None
    ) -> Union[ProblemAnalysisResult, SubProblemSolution, CritiqueResult, VerificationResult]:
        """
        Execute within ROMA workflow.
        
        Args:
            problem: The problem to work on
            phase: Which ROMA phase ("analyze", "solve", "critique", "verify")
            subproblem: Specific sub-problem (for solve/critique/verify phases)
            context: ROMA context
            
        Returns:
            Phase-appropriate result
        """
        if self.roma_adapter is None:
            raise RuntimeError("ROMA integration not available")
        
        if phase == "analyze":
            return self.roma_adapter.phase_1_analyze_problem(problem, context)
        elif phase == "solve" and subproblem:
            return self.roma_adapter.phase_2_solve_subproblem(subproblem, context)
        elif phase == "critique":
            solution = subproblem or (context.get("solution") if context else None)
            return self.roma_adapter.phase_3_critique_solution(solution, [])
        elif phase == "verify":
            solution = subproblem or (context.get("solution") if context else None)
            return self.roma_adapter.phase_4_verify_solution(solution, [])
        else:
            raise ValueError(f"Unknown phase: {phase}")
    
    def execute_for_decomposition(
        self,
        subproblem: SubProblem,
        team: str = "blue",
        context: Optional[TeamContext] = None
    ) -> Union[BlueTeamResult, RedTeamResult, GoldTeamResult]:
        """
        Execute within Decomposition workflow.
        
        Args:
            subproblem: The sub-problem to work on
            team: Which team ("blue", "red", "gold")
            context: Team context
            
        Returns:
            Team-appropriate result
        """
        if self.team_executor is None:
            raise RuntimeError("Decomposition integration not available")
        
        if team == "blue":
            return self.team_executor.blue_team_solve(subproblem, context)
        elif team == "red":
            solution = context.get("solution") if context else None
            return self.team_executor.red_team_critique(solution, context)
        elif team == "gold":
            solution = context.get("solution") if context else None
            criteria = subproblem.success_criteria if hasattr(subproblem, 'success_criteria') else []
            return self.team_executor.gold_team_verify(solution, criteria, context)
        else:
            raise ValueError(f"Unknown team: {team}")
    
    def execute_gauntlet(
        self,
        subproblem: SubProblem,
        config: Optional[GauntletConfig] = None
    ) -> GauntletResult:
        """
        Execute full 3-round gauntlet (Blue → Red → Gold).
        
        Convenience method for decomposition workflow.
        """
        if self.team_executor is None:
            raise RuntimeError("Decomposition integration not available")
        
        # Import and use gauntlet runner
        try:
            from matryoshka_gauntlet_runner import MatryoshkaGauntletRunner
            gauntlet = MatryoshkaGauntletRunner(self.team_executor)
            return gauntlet.run_gauntlet(subproblem, config or GauntletConfig())
        except ImportError:
            # Fallback: manual gauntlet execution
            return self._run_manual_gauntlet(subproblem, config or GauntletConfig())
    
    def _run_manual_gauntlet(self, subproblem: SubProblem, config: GauntletConfig) -> GauntletResult:
        """Manual gauntlet execution when runner not available."""
        # Blue team
        blue_result = self.execute_for_decomposition(subproblem, team="blue")
        
        # Red team critique
        context = {"solution": blue_result.get("solution") if isinstance(blue_result, dict) else blue_result}
        red_result = self.execute_for_decomposition(subproblem, team="red", context=context)
        
        # Gold team verify
        gold_result = self.execute_for_decomposition(subproblem, team="gold", context=context)
        
        passed = (
            gold_result.get("passed", False) if isinstance(gold_result, dict) else True
        )
        
        return GauntletResult(
            blue_result=blue_result,
            red_result=red_result,
            gold_result=gold_result,
            passed=passed
        ) if isinstance(GauntletResult, type) else {
            "blue_result": blue_result,
            "red_result": red_result,
            "gold_result": gold_result,
            "passed": passed
        }
    
    def solve_with_voting(
        self,
        task: str,
        strategies: List[ExplorationStrategy],
        k_ahead: int = 3
    ) -> VotedExplorationResult:
        """
        Solve with MAKER-style voting on exploration strategies.
        
        Multiple Matryoshka exploration strategies compete,
        first-to-ahead-by-k wins.
        """
        if not MATRYOSHKA_MDAP_AVAILABLE:
            raise RuntimeError("MDAP integration not available")
        
        explorer = MatryoshkaVotingExplorer(self.config.execution_config)
        return explorer.explore_with_voting(task, strategies, k_ahead)
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _auto_detect_space(self, task: str, context: Optional[Dict]) -> ProblemSpace:
        """Auto-detect problem space from task description."""
        if not self.config.auto_detect_problem_space:
            return ProblemSpace(name="abstract") if isinstance(ProblemSpace, type) else {"name": "abstract"}
        
        # Simple heuristic-based detection
        task_lower = task.lower()
        
        if any(kw in task_lower for kw in ["document", "text", "summarize", "extract", "parse"]):
            return ProblemSpace(name="document") if isinstance(ProblemSpace, type) else {"name": "document"}
        elif any(kw in task_lower for kw in ["code", "function", "class", "bug", "refactor"]):
            return ProblemSpace(name="code") if isinstance(ProblemSpace, type) else {"name": "code"}
        elif any(kw in task_lower for kw in ["data", "table", "csv", "analyze"]):
            return ProblemSpace(name="data") if isinstance(ProblemSpace, type) else {"name": "data"}
        else:
            return ProblemSpace(name="abstract") if isinstance(ProblemSpace, type) else {"name": "abstract"}
    
    @property
    def capabilities(self) -> Dict[str, bool]:
        """Check available capabilities."""
        return {
            "core_execution": self.execution_engine is not None,
            "roma_integration": self.roma_adapter is not None,
            "decomposition_integration": self.team_executor is not None,
            "mdap_integration": self.mdap_config is not None,
            "memory_integration": self.memory_bridge is not None,
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status of all subsystems."""
        return {
            "execution_engine": type(self.execution_engine).__name__ if self.execution_engine else None,
            "roma_adapter": type(self.roma_adapter).__name__ if self.roma_adapter else None,
            "team_executor": type(self.team_executor).__name__ if self.team_executor else None,
            "memory_bridge": type(self.memory_bridge).__name__ if self.memory_bridge else None,
            "capabilities": self.capabilities,
        }


# ============================================================================
# HIGH-LEVEL WORKFLOW INTERFACES
# ============================================================================

class MatryoshkaROMAWorkflow:
    """
    High-level ROMA workflow using Matryoshka.
    
    Complete end-to-end ROMA solving with Matryoshka.
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.executor = UnifiedMatryoshkaExecutor(config)
    
    def solve(self, problem: str) -> SolverResult:
        """Complete ROMA solve workflow."""
        # Phase 1: Analyze
        analysis = self.executor.execute_for_roma(problem, phase="analyze")
        
        # Get subproblems from analysis
        subproblems = analysis.get("subproblems", []) if isinstance(analysis, dict) else getattr(analysis, 'subproblems', [])
        findings = analysis.get("findings", []) if isinstance(analysis, dict) else getattr(analysis, 'findings', [])
        
        # Phase 2: Solve sub-problems
        solutions = []
        for subproblem in subproblems:
            solution = self.executor.execute_for_roma(
                problem, phase="solve", subproblem=subproblem
            )
            solutions.append(solution)
        
        # Phase 3 & 4: Critique and Verify
        verified = []
        for solution in solutions:
            critique = self.executor.execute_for_roma(
                problem, phase="critique", context={"solution": solution}
            )
            critique_score = critique.get("critique_score", 0) if isinstance(critique, dict) else getattr(critique, 'critique_score', 0)
            if critique_score > 0.7:  # Passes critique
                verify = self.executor.execute_for_roma(
                    problem, phase="verify", context={"solution": solution}
                )
                passed = verify.get("passed", False) if isinstance(verify, dict) else getattr(verify, 'passed', False)
                if passed:
                    verified.append(solution)
        
        # Aggregate
        return SolverResult(
            solutions=verified,
            findings=findings,
            success=len(verified) > 0
        ) if isinstance(SolverResult, type) else {
            "solutions": verified,
            "findings": findings,
            "success": len(verified) > 0
        }


class MatryoshkaDecompositionWorkflow:
    """
    High-level Decomposition workflow using Matryoshka.
    
    Complete end-to-end decomposition with Blue/Red/Gold teams.
    """
    
    def __init__(self, config: Optional[UnifiedMatryoshkaConfig] = None):
        self.executor = UnifiedMatryoshkaExecutor(config)
    
    def solve(self, problem) -> DecompositionResult:
        """Complete decomposition workflow."""
        # Decompose
        subproblems = self._decompose(problem)
        
        # Solve each with gauntlet
        solutions = []
        for subproblem in subproblems:
            gauntlet_result = self.executor.execute_gauntlet(subproblem)
            passed = gauntlet_result.get("passed", False) if isinstance(gauntlet_result, dict) else getattr(gauntlet_result, 'passed', False)
            if passed:
                solution = gauntlet_result.get("solution", gauntlet_result) if isinstance(gauntlet_result, dict) else getattr(gauntlet_result, 'solution', gauntlet_result)
                solutions.append(solution)
        
        # Aggregate
        return DecompositionResult(
            problem=problem,
            solutions=solutions,
            success=len(solutions) == len(subproblems)
        ) if isinstance(DecompositionResult, type) else {
            "problem": problem,
            "solutions": solutions,
            "success": len(solutions) == len(subproblems)
        }
    
    def _decompose(self, problem):
        """Decompose problem into subproblems."""
        # Try to use decomposition engine
        try:
            from decomposition_engine import DecompositionEngine
            engine = DecompositionEngine()
            result = engine.decompose(problem)
            return result.get("subproblems", result) if isinstance(result, dict) else getattr(result, 'subproblems', [problem])
        except ImportError:
            # Fallback: treat as single subproblem
            return [problem]


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_unified_executor(
    enable_roma: bool = True,
    enable_decomposition: bool = True,
    enable_mdap: bool = True,
    enable_memory: bool = True,
    storage_path: Optional[str] = None
) -> UnifiedMatryoshkaExecutor:
    """
    Create unified executor with specified integrations.
    
    Args:
        enable_roma: Enable ROMA integration
        enable_decomposition: Enable Decomposition workflow integration
        enable_mdap: Enable MDAP/MAKER integration
        enable_memory: Enable Unified Memory integration
        storage_path: Path for memory storage
        
    Returns:
        Configured UnifiedMatryoshkaExecutor
    """
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=enable_roma,
        enable_decomposition_integration=enable_decomposition,
        enable_mdap_integration=enable_mdap,
        enable_memory_integration=enable_memory,
        memory_storage_path=storage_path
    )
    return UnifiedMatryoshkaExecutor(config)


def create_roma_workflow(
    storage_path: Optional[str] = None
) -> MatryoshkaROMAWorkflow:
    """Create ROMA workflow with Matryoshka."""
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=True,
        enable_decomposition_integration=False,
        enable_mdap_integration=False,
        memory_storage_path=storage_path
    )
    return MatryoshkaROMAWorkflow(config)


def create_decomposition_workflow(
    storage_path: Optional[str] = None
) -> MatryoshkaDecompositionWorkflow:
    """Create Decomposition workflow with Matryoshka."""
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=False,
        enable_decomposition_integration=True,
        enable_mdap_integration=False,
        memory_storage_path=storage_path
    )
    return MatryoshkaDecompositionWorkflow(config)


def create_full_stack_workflow(
    storage_path: str = "./matryoshka_memory"
) -> Tuple[MatryoshkaROMAWorkflow, MatryoshkaDecompositionWorkflow]:
    """
    Create both ROMA and Decomposition workflows with shared memory.
    
    Enables cross-workflow learning.
    """
    config = UnifiedMatryoshkaConfig(
        enable_roma_integration=True,
        enable_decomposition_integration=True,
        enable_mdap_integration=True,
        enable_memory_integration=True,
        memory_storage_path=storage_path
    )
    return (
        MatryoshkaROMAWorkflow(config),
        MatryoshkaDecompositionWorkflow(config)
    )


# ============================================================================
# ARCHITECTURE DOCUMENTATION
# ============================================================================

ARCHITECTURE_OVERVIEW = """
╔══════════════════════════════════════════════════════════════════════════════╗
║           UNIFIED MATRYOSHKA ARCHITECTURE                                     ║
║     Recursive Language Model (RLM) as Generalized Execution Engine           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │                    UNIFIED MATRYOSHKA EXECUTOR                       │     ║
║  │                      (Single Interface)                              │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                    │                                          ║
║          ┌─────────────────────────┼─────────────────────────┐                ║
║          │                         │                         │                ║
║          ▼                         ▼                         ▼                ║
║  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐         ║
║  │     ROMA      │        │ Decomposition │        │  MDAP/MAKER   │         ║
║  │  Integration  │        │   Workflow    │        │  Integration  │         ║
║  └───────────────┘        └───────────────┘        └───────────────┘         ║
║          │                         │                         │                ║
║          └─────────────────────────┼─────────────────────────┘                ║
║                                    │                                          ║
║                                    ▼                                          ║
║  ┌─────────────────────────────────────────────────────────────────────┐     ║
║  │              MATRYOSHKA EXECUTION ENGINE                             │     ║
║  │                                                                     │     ║
║  │  • Iterative symbolic execution via Nucleus/Lattice                 │     ║
║  │  • Problem space abstraction (Document, Code, Data, Abstract)       │     ║
║  │  • State management across iterations                               │     ║
║  │  • Finding and failure tracking                                     │     ║
║  └─────────────────────────────────────────────────────────────────────┘     ║
║                                    │                                          ║
║          ┌─────────────────────────┼─────────────────────────┐                ║
║          │                         │                         │                ║
║          ▼                         ▼                         ▼                ║
║  ┌───────────────┐        ┌───────────────┐        ┌───────────────┐         ║
║  │   4-Layer     │        │  Always-True  │        │    Hybrid     │         ║
║  │    Indexes    │        │     State     │        │   Retrieval   │         ║
║  │               │        │               │        │               │         ║
║  │ • Hierarchical│        │ • Never drops │        │ • 4 strategies│         ║
║  │ • Graph       │        │ • Continuous  │        │ • Top-N limit │         ║
║  │ • Hash        │        │ • Source of   │        │ • ~5KB limit  │         ║
║  │ • Semantic    │        │   truth       │        │               │         ║
║  └───────────────┘        └───────────────┘        └───────────────┘         ║
║                                                                               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Matryoshka is NOT just for documents - it's a universal execution engine
for any problem space that can be represented symbolically.

Key Design Principles:

1. UNIFIED INTERFACE
   - Single executor for all workflows
   - Automatic capability detection
   - Graceful degradation when components unavailable

2. INTEGRATION LAYERS
   - ROMA: Recursive decomposition phases (analyze/solve/critique/verify)
   - Decomposition: Blue/Red/Gold team gauntlet
   - MDAP/MAKER: Voting-based strategy selection

3. EXECUTION ENGINE
   - Problem-space agnostic
   - Iterative exploration
   - Symbolic representation

4. MEMORY SYSTEM
   - 4-layer indexing (hierarchical, graph, hash, semantic)
   - Always-true state preservation
   - Context rot prevention
"""

# Print architecture on module import for documentation
if __name__ == "__main__":
    print(ARCHITECTURE_OVERVIEW)
