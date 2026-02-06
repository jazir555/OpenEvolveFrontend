"""
Universal Problem Solver - End-to-End Decomposition/Recomposition System

This module provides a complete problem-solving pipeline:
    1. Problem Analysis - Understand domain, constraints, success criteria
    2. Decomposition - Break into atomic sub-problems
    3. Sub-Problem Solving - Solve each sub-problem independently
    4. Reassembly - Combine solutions with conflict resolution
    5. Validation - Verify final solution

The system is generic and works for any industry:
    - Software Engineering
    - Finance/Trading
    - Scientific Research
    - Healthcare
    - Manufacturing
    - Business Strategy
    - Legal/Compliance
    - And more...

Usage:
    >>> from universal_problem_solver import UniversalProblemSolver
    >>> 
    >>> solver = UniversalProblemSolver()
    >>> 
    >>> # Solve any problem
    >>> result = solver.solve(
    ...     problem_statement="Build a real-time trading risk system",
    ...     domain="finance",
    ...     constraints=["regulatory_compliance", "sub_millisecond_latency"]
    ... )
    >>> 
    >>> print(result.final_solution.assembled_content)
    >>> print(f"Quality: {result.quality_score}")
"""

import logging
import json
import dataclasses
import uuid
from typing import Dict, List, Any, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import decomposition engine
from universal_decomposition_engine import (
    UniversalDecompositionEngine,
    ProblemDomain,
    DecompositionStrategy,
    DecompositionPlan,
    ProblemDefinition,
    SubProblem,
    SubProblemType,
    SubProblemStatus,
    ComplexityScore,
    SuccessCriterion,
    Constraint,
    FinanceDomainExtension
)

# Import recomposition engine
from universal_recomposition_engine import (
    UniversalRecompositionEngine,
    AssemblyStrategy,
    IntegratedSolution,
    SubProblemSolution,
    QualityMetrics
)

from team_manager import TeamManager
from gauntlet_manager import GauntletManager
from utils.entanglement_utils import (
    normalize_entanglement_matrix,
    build_symbolic_entanglement_matrix,
    serialize_entanglement_matrix,
)

# Optional Blue Team + Enhanced Recomposition wiring
try:
    from blue_team_solver_engine import SubProblemSolver as BlueTeamSubProblemSolver
    BLUE_TEAM_AVAILABLE = True
except Exception:
    BLUE_TEAM_AVAILABLE = False
    BlueTeamSubProblemSolver = None  # type: ignore

try:
    from enhanced_recomposition_engine import (
        EnhancedRecompositionEngine,
        SubProblemSolution as EnhancedSubProblemSolution,
        IntegratedSolution as EnhancedIntegratedSolution,
        RecompositionConfig as EnhancedRecompositionConfig,
    )
    ENHANCED_RECOMPOSITION_AVAILABLE = True
except ImportError:
    ENHANCED_RECOMPOSITION_AVAILABLE = False
    EnhancedRecompositionEngine = None  # type: ignore
    EnhancedSubProblemSolution = None  # type: ignore
    EnhancedIntegratedSolution = None  # type: ignore
    EnhancedRecompositionConfig = None  # type: ignore

# Optional Web3 audit ingestion tools
try:
    from decomposition_mcp_tools import (
        web3_ingest_contract_audit_stack,
        get_mcp_tool_inventory,
    )
    WEB3_INGESTION_AVAILABLE = True
except Exception:
    WEB3_INGESTION_AVAILABLE = False
    web3_ingest_contract_audit_stack = None  # type: ignore
    get_mcp_tool_inventory = None  # type: ignore

# Optional ROMA integration
try:
    from roma_openevolve_integration import create_roma_adapter, ROMAOpenEvolveConfig
    ROMA_AVAILABLE = True
except ImportError:
    ROMA_AVAILABLE = False
    create_roma_adapter = None  # type: ignore
    ROMAOpenEvolveConfig = None  # type: ignore

# Optional CAV-NLP integration for enhanced problem solving
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver, ConstraintFormalizer
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    EnhancedZ3Solver = None  # type: ignore
    ConstraintFormalizer = None  # type: ignore
    UnifiedMathService = None  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SolutionStep:
    """Record of a solution step"""
    step_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "pending"  # pending, in_progress, completed, failed
    details: Dict[str, Any] = field(default_factory=dict)
    
    def duration_seconds(self) -> float:
        """Get duration in seconds"""
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


@dataclass
class GauntletOutcome:
    """Normalized gauntlet execution outcome."""
    stage: str
    gauntlet_name: Optional[str]
    team_name: Optional[str]
    team_role: Optional[str]
    is_approved: bool
    report_summary: str
    report_object: Optional[Any] = None
    targeted_feedback: List[str] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)
    skipped: bool = False
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "gauntlet_name": self.gauntlet_name,
            "team_name": self.team_name,
            "team_role": self.team_role,
            "is_approved": self.is_approved,
            "report_summary": self.report_summary,
            "targeted_feedback": list(self.targeted_feedback),
            "logs": list(self.logs),
            "skipped": self.skipped,
            "error": self.error,
        }


@dataclass
class GauntletBundle:
    """Resolved gauntlet + team bundle for pipeline execution."""
    solver_generation_gauntlet: Optional[Any] = None
    sub_problem_red_gauntlet: Optional[Any] = None
    sub_problem_gold_gauntlet: Optional[Any] = None
    final_red_gauntlet: Optional[Any] = None
    final_gold_gauntlet: Optional[Any] = None
    solver_generation_team: Optional[Any] = None
    sub_problem_red_team: Optional[Any] = None
    sub_problem_gold_team: Optional[Any] = None
    final_red_team: Optional[Any] = None
    final_gold_team: Optional[Any] = None

    def has_any(self) -> bool:
        return any([
            self.solver_generation_gauntlet,
            self.sub_problem_red_gauntlet,
            self.sub_problem_gold_gauntlet,
            self.final_red_gauntlet,
            self.final_gold_gauntlet,
        ])

    def has_subproblem_any(self) -> bool:
        return any([
            self.solver_generation_gauntlet,
            self.sub_problem_red_gauntlet,
            self.sub_problem_gold_gauntlet,
        ])


@dataclass
class SolverResult:
    """Complete result from problem solving"""
    problem_id: str
    problem_title: str
    problem_statement: str
    domain: str
    
    # Decomposition results
    decomposition_plan: DecompositionPlan
    
    # Solving results
    sub_problem_solutions: Dict[str, Any]
    solving_steps: List[SolutionStep]
    
    # Reassembly results
    final_solution: Any
    
    # Overall metrics
    quality_score: float
    total_duration_seconds: float
    conflicts_detected: int
    conflicts_resolved: int
    
    # Metadata
    execution_log: List[str] = field(default_factory=list)
    gauntlet_results: Dict[str, Any] = field(default_factory=dict)
    gauntlet_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    # CAV-NLP verification results (optional)
    verification: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        assembly_strategy = getattr(self.final_solution, "assembly_strategy", None)
        if hasattr(assembly_strategy, "value"):
            assembly_strategy = assembly_strategy.value

        return {
            'problem_id': self.problem_id,
            'problem_title': self.problem_title,
            'domain': self.domain,
            'quality_score': self.quality_score,
            'total_duration_seconds': self.total_duration_seconds,
            'conflicts_detected': self.conflicts_detected,
            'conflicts_resolved': self.conflicts_resolved,
            'num_sub_problems': len(self.decomposition_plan.sub_problems),
            'num_solutions': len(self.sub_problem_solutions),
            'assembly_strategy': assembly_strategy,
            'gauntlet_summary': self.gauntlet_summary,
            'created_at': self.created_at.isoformat()
        }
    
    def summary(self) -> str:
        """Get human-readable summary"""
        assembly_strategy = getattr(self.final_solution, "assembly_strategy", "unknown")
        if hasattr(assembly_strategy, "value"):
            assembly_strategy = assembly_strategy.value

        lines = [
            f"Problem: {self.problem_title}",
            f"Domain: {self.domain}",
            f"",
            f"Decomposition: {len(self.decomposition_plan.sub_problems)} sub-problems",
            f"Solutions Generated: {len(self.sub_problem_solutions)}",
            f"Assembly Strategy: {assembly_strategy}",
            f"",
            f"Quality Score: {self.quality_score:.2f}/1.0",
            f"Conflicts: {self.conflicts_resolved} resolved, {self.conflicts_detected} remaining",
            f"Total Time: {self.total_duration_seconds:.2f} seconds",
            f"",
            f"Final Solution Length: {len(self.final_solution.assembled_content)} characters"
        ]
        if self.gauntlet_summary:
            gauntlet_runs = self.gauntlet_summary.get("total_runs", 0)
            gauntlet_failed = self.gauntlet_summary.get("failed", 0)
            gauntlet_skipped = self.gauntlet_summary.get("skipped", 0)
            lines.extend([
                "",
                f"Gauntlets: {gauntlet_runs} run, {gauntlet_failed} failed, {gauntlet_skipped} skipped",
            ])
        return "\n".join(lines)


# ============================================================================
# SUB-PROBLEM SOLVERS
# ============================================================================

class SubProblemSolver:
    """
    Solves individual sub-problems.
    
    Can be extended with different solving strategies:
    - Template-based solving
    - LLM-based solving
    - Rule-based solving
    - Hybrid approaches
    """
    
    # Domain-specific solution templates
    TEMPLATES = {
        'software': {
            'implementation': """
## Implementation: {title}

### Overview
{description}

### Approach
1. Design the core components
2. Implement primary functionality
3. Add error handling
4. Write tests
5. Document the solution

### Key Considerations
- Maintainability
- Performance
- Security
- Testing coverage

### Success Criteria
{success_criteria}

[Implementation code would go here]
""",
            'design': """
## Design: {title}

### Problem
{description}

### Design Decisions
1. Architecture pattern selection
2. Component boundaries
3. Interface definitions
4. Data flow

### Diagram
[Architecture diagram would go here]

### Trade-offs Considered
{constraints}
"""
        },
        'finance': {
            'implementation': """
## Financial System Component: {title}

### Business Context
{description}

### Implementation Requirements
1. Regulatory compliance: {constraints}
2. Risk controls
3. Audit logging
4. Performance requirements

### Key Components
- Input validation
- Business logic
- Risk checks
- Reporting

### Testing Requirements
- Unit tests
- Integration tests
- Regulatory scenario tests
""",
            'risk_analysis': """
## Risk Analysis: {title}

### Scope
{description}

### Risk Factors Identified
1. Market risk
2. Credit risk
3. Operational risk
4. Regulatory risk

### Mitigation Strategies
[Risk mitigation details]

### Monitoring Requirements
- Real-time alerts
- Periodic reports
- Stress testing
"""
        },
        'scientific': {
            'research': """
## Research Component: {title}

### Research Question
{description}

### Methodology
1. Literature review
2. Hypothesis formation
3. Experimental design
4. Data collection
5. Analysis
6. Validation

### Expected Outcomes
{success_criteria}

### Reproducibility Requirements
- Data availability
- Code sharing
- Method documentation
"""
        }
    }
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def solve(
        self,
        sub_problem: SubProblem,
        parent_problem: ProblemDefinition,
        context: Optional[Dict[str, Any]] = None
    ) -> SubProblemSolution:
        """
        Solve a single sub-problem.
        
        In a production system, this would:
        - Use LLM to generate solutions
        - Apply domain-specific reasoning
        - Validate against success criteria
        - Run tests if applicable
        """
        start_time = datetime.now()
        
        self.logger.info(f"Solving sub-problem: {sub_problem.title}")
        
        # Select template based on domain and type
        domain = parent_problem.domain
        if isinstance(domain, ProblemDomain):
            domain = domain.value
        
        template = self._select_template(domain, sub_problem.type)
        
        # Generate solution content
        content = self._generate_solution_content(
            template,
            sub_problem,
            parent_problem
        )

        entangled_with = []
        entanglement_symbols = []
        if context:
            entangled_with = context.get("entangled_with", []) or []
            entanglement_symbols = context.get("entanglement_symbols", []) or []
        if entangled_with:
            symbols_text = ", ".join(entanglement_symbols) if entanglement_symbols else "n/a"
            content = (
                f"{content}\n\n### Entanglement Context\n"
                f"- Entangled with: {', '.join(entangled_with)}\n"
                f"- Shared symbols: {symbols_text}\n"
                f"- Coordination note: keep shared interfaces consistent with entangled peers\n"
            )
        
        # Estimate quality based on completeness
        quality = self._estimate_quality(content, sub_problem)
        
        # Create solution
        solution = SubProblemSolution(
            sub_problem_id=sub_problem.id,
            solution_content=content,
            quality_score=quality,
            verification_status="pending",
            metadata={
                'title': sub_problem.title,
                'domain': domain,
                'type': sub_problem.type if isinstance(sub_problem.type, str) else sub_problem.type.value,
                'solving_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'template_used': template is not None,
                'entangled_with': entangled_with,
                'entanglement_symbols': entanglement_symbols,
            }
        )
        
        return solution
    
    def _select_template(self, domain: str, problem_type) -> Optional[str]:
        """Select appropriate template"""
        type_str = problem_type if isinstance(problem_type, str) else problem_type.value
        
        domain_templates = self.TEMPLATES.get(domain, self.TEMPLATES.get('software', {}))
        
        # Map problem types to templates
        template_key = 'implementation'
        if 'design' in type_str.lower():
            template_key = 'design'
        elif 'research' in type_str.lower():
            template_key = 'research'
        elif 'risk' in type_str.lower():
            template_key = 'risk_analysis'
        
        return domain_templates.get(template_key)
    
    def _generate_solution_content(
        self,
        template: Optional[str],
        sub_problem: SubProblem,
        parent_problem: ProblemDefinition
    ) -> str:
        """Generate solution content"""
        
        if template:
            # Format template with sub-problem details
            success_criteria_text = "\n".join(
                f"- {sc.description}" 
                for sc in sub_problem.success_criteria[:3]
            ) or "- Complete implementation"
            
            constraints_text = "\n".join(
                f"- {c.description}" 
                for c in parent_problem.constraints[:3]
            ) or "- Meet all requirements"
            
            content = template.format(
                title=sub_problem.title,
                description=sub_problem.description,
                success_criteria=success_criteria_text,
                constraints=constraints_text
            )
        else:
            # Generate without template
            content = f"""
## Solution: {sub_problem.title}

### Description
{sub_problem.description}

### Approach
This sub-problem addresses the {sub_problem.title} component.

### Implementation
[Detailed implementation would be provided here based on the specific requirements]

### Verification
Success criteria:
"""
            for sc in sub_problem.success_criteria:
                content += f"\n- {sc.description}"
        
        return content.strip()
    
    def _estimate_quality(self, content: str, sub_problem: SubProblem) -> float:
        """Estimate solution quality"""
        # Simple heuristic based on content completeness
        score = 0.7  # Base score
        
        # Increase for content length
        if len(content) > 500:
            score += 0.1
        if len(content) > 1000:
            score += 0.05
        
        # Increase for structure (has headers)
        if '##' in content:
            score += 0.05
        
        # Decrease for complexity mismatch
        complexity = sub_problem.complexity_score.overall_complexity
        if complexity > 7 and len(content) < 300:
            score -= 0.1
        
        return min(0.95, max(0.5, score))


# ============================================================================
# MAIN UNIVERSAL PROBLEM SOLVER
# ============================================================================

class UniversalProblemSolver:
    """
    Universal problem solver implementing the complete decomposition/recomposition workflow.
    
    The workflow:
        1. Analyze problem and determine domain
        2. Decompose into sub-problems
        3. Solve each sub-problem independently
        4. Detect and resolve conflicts
        5. Reassemble into integrated solution
        6. Validate final solution
    
    Usage:
        >>> solver = UniversalProblemSolver()
        >>> 
        >>> # Software problem
        >>> result = solver.solve(
        ...     problem_statement="Build a REST API for user management",
        ...     domain=ProblemDomain.SOFTWARE,
        ...     constraints=["oauth2", "rate_limiting"]
        ... )
        >>> 
        >>> # Finance problem
        >>> result = solver.solve(
        ...     problem_statement="Implement trading risk controls",
        ...     domain=ProblemDomain.FINANCE,
        ...     constraints=["mifid_compliance", "real_time"]
        ... )
        >>> 
        >>> print(result.summary())
    """
    
    def __init__(
        self,
        llm_client: Optional[Any] = None,
        decomposition_strategy: DecompositionStrategy = DecompositionStrategy.HYBRID,
        assembly_strategy: AssemblyStrategy = AssemblyStrategy.ADAPTIVE,
        blue_team_config: Optional[Dict[str, Any]] = None,
        gauntlet_config: Optional[Dict[str, Any]] = None,
        enable_gauntlets: bool = True,
        max_gauntlet_refinement_loops: int = 1,
        enable_roma: bool = False,
        use_roma_mdap_maker: bool = False,
        roma_config: Optional[Dict[str, Any]] = None,
        team_manager: Optional[TeamManager] = None,
        gauntlet_manager: Optional[GauntletManager] = None,
    ):
        self.llm_client = llm_client
        self.decomposition_strategy = decomposition_strategy
        self.assembly_strategy = assembly_strategy
        self.blue_team_config = blue_team_config or {}
        self.gauntlet_config = gauntlet_config or {}
        self.enable_gauntlets = enable_gauntlets
        self.max_gauntlet_refinement_loops = max_gauntlet_refinement_loops
        self.enable_roma = enable_roma
        self.use_roma_mdap_maker = use_roma_mdap_maker
        self.roma_config = roma_config or {}
        self.team_manager = team_manager or TeamManager()
        self.gauntlet_manager = gauntlet_manager or GauntletManager()
        
        # Initialize components
        self.decomposition_engine = UniversalDecompositionEngine(llm_client)
        self.recomposition_engine = UniversalRecompositionEngine(llm_client)
        self.sub_problem_solver = SubProblemSolver(llm_client)
        self.blue_team_solver = None
        if BLUE_TEAM_AVAILABLE:
            try:
                self.blue_team_solver = BlueTeamSubProblemSolver(self.blue_team_config)
            except Exception:
                self.blue_team_solver = None

        self.roma_adapter = None
        if self.enable_roma and ROMA_AVAILABLE and create_roma_adapter is not None:
            try:
                self.roma_adapter = create_roma_adapter(
                    enable_roma=True,
                    use_mdap_maker=self.use_roma_mdap_maker,
                    **self.roma_config,
                )
            except Exception:
                self.roma_adapter = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.solution_history: List[SolverResult] = []
    
    def solve(
        self,
        problem_statement: str,
        title: Optional[str] = None,
        domain: Union[ProblemDomain, str] = ProblemDomain.GENERIC,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        max_subproblems: int = 15,
        solve_subproblems: bool = True,
        detect_conflicts: bool = True,
        resolve_conflicts: bool = True,
        use_blue_team_solver: bool = False,
        use_roma_solver: Optional[bool] = None,
        run_gauntlets: Optional[bool] = None,
        gauntlet_config: Optional[Dict[str, Any]] = None,
        max_gauntlet_refinement_loops: Optional[int] = None,
        enable_web3_ingestion: bool = True,
        web3_project_path: str = ".",
        web3_run_fuzzing: bool = True,
    ) -> SolverResult:
        """
        Solve a problem end-to-end.
        
        Args:
            problem_statement: The problem to solve
            title: Optional title
            domain: Problem domain (or auto-detect if not specified)
            constraints: List of constraints
            success_criteria: List of success criteria
            max_subproblems: Maximum sub-problems to create
            solve_subproblems: Whether to solve sub-problems (vs just decompose)
            detect_conflicts: Whether to detect conflicts during assembly
            resolve_conflicts: Whether to attempt conflict resolution
            use_blue_team_solver: Use Blue Team solver (Z3/Lean) with enhanced recomposition
            use_roma_solver: Use ROMA phase-2 solving when available
            run_gauntlets: Whether to run Blue/Red/Gold gauntlet pipeline
            gauntlet_config: Optional gauntlet configuration overrides
            max_gauntlet_refinement_loops: Override max gauntlet refinement loops
            enable_web3_ingestion: Run Slither/Forge ingestion when domain is Web3
            web3_project_path: Contract project root for Web3 ingestion
            web3_run_fuzzing: Include Forge fuzzing as part of ingestion
            
        Returns:
            SolverResult with complete solution and metadata
        """
        overall_start = datetime.now()
        execution_log = []
        steps: List[SolutionStep] = []
        
        self.logger.info(f"Starting problem solving: {title or problem_statement[:50]}...")

        if use_blue_team_solver and not self.blue_team_solver:
            self.logger.warning("Blue Team solver unavailable; falling back to standard solver")
            use_blue_team_solver = False

        if use_roma_solver is None:
            use_roma_solver = self.enable_roma
        if use_roma_solver and not self.roma_adapter:
            self.logger.warning("ROMA solver unavailable; falling back to standard solver")
            use_roma_solver = False
        if use_blue_team_solver and use_roma_solver:
            use_roma_solver = False

        if run_gauntlets is None:
            run_gauntlets = self.enable_gauntlets

        max_refinement_loops = (
            self.max_gauntlet_refinement_loops
            if max_gauntlet_refinement_loops is None
            else max_gauntlet_refinement_loops
        )

        gauntlet_bundle: Optional[GauntletBundle] = None
        gauntlet_results: Dict[str, Any] = {"sub_problems": {}, "final": {}}
        gauntlet_summary: Dict[str, Any] = {}
        domain_artifacts: Dict[str, Any] = {}
        
        # ============================================================================
        # STEP 1: Domain Detection (if not specified)
        # ============================================================================
        step_domain = SolutionStep("domain_detection", datetime.now())
        
        if isinstance(domain, str):
            domain = self._parse_domain(domain)
        
        # Auto-detect if generic
        if domain == ProblemDomain.GENERIC:
            domain = self._detect_domain(problem_statement)
            self.logger.info(f"Auto-detected domain: {domain.value}")
        
        step_domain.end_time = datetime.now()
        step_domain.status = "completed"
        step_domain.details = {'detected_domain': domain.value}
        steps.append(step_domain)
        execution_log.append(f"Domain: {domain.value}")

        # ============================================================================
        # STEP 1B: Optional Web3 Ingestion
        # ============================================================================
        if domain == ProblemDomain.WEB3 and enable_web3_ingestion:
            step_web3 = SolutionStep("web3_ingestion", datetime.now())
            if WEB3_INGESTION_AVAILABLE and web3_ingest_contract_audit_stack:
                try:
                    ingestion = web3_ingest_contract_audit_stack(
                        project_path=web3_project_path,
                        run_fuzzing=web3_run_fuzzing,
                    )
                    domain_artifacts["web3_ingestion"] = ingestion
                    if isinstance(ingestion, dict):
                        if ingestion.get("entanglement_matrix"):
                            domain_artifacts["entanglement_matrix"] = ingestion["entanglement_matrix"]
                        if ingestion.get("contracts"):
                            domain_artifacts["contracts"] = ingestion["contracts"]
                    step_web3.status = "completed"
                    step_web3.details = {
                        "success": bool(ingestion.get("success")) if isinstance(ingestion, dict) else False,
                        "contracts": len(ingestion.get("contracts", [])) if isinstance(ingestion, dict) else 0,
                        "fuzzing_enabled": web3_run_fuzzing,
                    }
                    execution_log.append(
                        f"Web3 ingestion: {step_web3.details.get('contracts', 0)} contracts analyzed"
                    )
                except Exception as exc:
                    step_web3.status = "failed"
                    step_web3.details = {"error": str(exc)}
                    execution_log.append(f"Web3 ingestion failed: {exc}")
            else:
                step_web3.status = "skipped"
                step_web3.details = {
                    "reason": "Web3 ingestion tools unavailable",
                    "inventory_available": WEB3_INGESTION_AVAILABLE,
                }
                execution_log.append("Web3 ingestion skipped: tools unavailable")
            step_web3.end_time = datetime.now()
            steps.append(step_web3)
        
        # ============================================================================
        # STEP 2: Decomposition
        # ============================================================================
        step_decomp = SolutionStep("decomposition", datetime.now())

        plan = None
        if self.enable_roma and self.roma_adapter and self.roma_adapter.is_decomposition_available():
            problem_def = self.decomposition_engine._create_problem_definition(
                problem_statement=problem_statement,
                title=title,
                domain=domain,
                constraints=constraints or [],
                success_criteria=success_criteria or [],
                domain_artifacts=domain_artifacts,
            )
            try:
                roma_result = self.roma_adapter.setup_and_decompose_problem(
                    problem_statement=problem_statement,
                    problem_type=problem_def.metadata.get("problem_type")
                    if isinstance(problem_def.metadata, dict)
                    else None,
                    domain=domain.value if hasattr(domain, "value") else str(domain),
                )
                plan = self._plan_from_roma_result(problem_def, roma_result)
            except Exception as exc:
                self.logger.warning("ROMA decomposition failed; falling back: %s", exc)
                plan = None

        if plan is None:
            plan = self.decomposition_engine.decompose(
                problem_statement=problem_statement,
                title=title,
                domain=domain,
                constraints=constraints or [],
                success_criteria=success_criteria or [],
                strategy=self.decomposition_strategy,
                max_subproblems=max_subproblems,
                domain_artifacts=domain_artifacts,
            )
        
        # Apply domain-specific enhancements if not already applied
        if not plan.metadata.get("domain_extensions_applied"):
            plan = self.decomposition_engine._apply_domain_extensions(plan)
            if plan.metadata.get("domain_extensions_applied"):
                plan.dependency_graph = self.decomposition_engine._build_dependency_graph(plan.sub_problems)
                plan.execution_order = self.decomposition_engine._calculate_execution_order(
                    plan.sub_problems, plan.dependency_graph
                )
                plan.parallel_groups = self.decomposition_engine._identify_parallel_groups(
                    plan.sub_problems, plan.dependency_graph
                )
                plan.quality_score = self.decomposition_engine._calculate_quality_score(
                    plan.original_problem, plan.sub_problems, plan.dependency_graph
                )
                try:
                    matrix, symbols_by_id = build_symbolic_entanglement_matrix(
                        plan.sub_problems,
                        allowed_ids=[sp.id for sp in plan.sub_problems],
                        enforce_symmetry=True,
                        strict=False,
                    )
                    serialized = serialize_entanglement_matrix(matrix)
                    plan.metadata["entanglement_matrix"] = serialized
                    plan.analyzed_context.setdefault("entanglement_matrix", serialized)
                    for sp in plan.sub_problems:
                        entangled_with = serialized.get(sp.id, [])
                        sp.metadata["entangled_with"] = entangled_with
                        if entangled_with and "entanglement_source" not in sp.metadata:
                            sp.metadata["entanglement_source"] = "symbolic_overlap"
                        if sp.id in symbols_by_id:
                            sp.metadata["entanglement_symbols"] = sorted(symbols_by_id.get(sp.id, set()))
                except Exception as exc:
                    self.logger.warning("Failed to rebuild entanglement matrix: %s", exc)

        if domain_artifacts:
            plan.metadata.setdefault("domain_artifacts", {})
            if isinstance(plan.metadata["domain_artifacts"], dict):
                plan.metadata["domain_artifacts"].update(domain_artifacts)
            if domain == ProblemDomain.WEB3:
                plan.metadata.setdefault("web3", {})
                if isinstance(plan.metadata["web3"], dict):
                    plan.metadata["web3"]["ingestion"] = domain_artifacts.get("web3_ingestion", {})
        
        step_decomp.end_time = datetime.now()
        step_decomp.status = "completed"
        step_decomp.details = {
            'num_subproblems': len(plan.sub_problems),
            'strategy': plan.strategy_used.value if hasattr(plan.strategy_used, 'value') else str(plan.strategy_used),
            'quality_score': plan.quality_score
        }
        steps.append(step_decomp)
        execution_log.append(f"Decomposed into {len(plan.sub_problems)} sub-problems")

        if run_gauntlets:
            gauntlet_bundle = self._resolve_gauntlet_bundle(gauntlet_config)
            if not gauntlet_bundle.has_any() and not (self.roma_adapter and self.enable_roma):
                run_gauntlets = False
                execution_log.append("Gauntlets: no configured gauntlets found; skipping gauntlet pipeline")
        
        # ============================================================================
        # STEP 3: Sub-Problem Solving
        # ============================================================================
        sub_solutions: Dict[str, SubProblemSolution] = {}
        entanglement_matrix: Dict[str, List[str]] = {}
        if hasattr(plan, "metadata") and isinstance(plan.metadata, dict):
            entanglement_matrix = plan.metadata.get("entanglement_matrix", {}) or {}
        
        if solve_subproblems:
            step_solving = SolutionStep("subproblem_solving", datetime.now())

            if use_blue_team_solver and self.blue_team_solver:
                sub_solutions = self._solve_with_blue_team(
                    plan=plan,
                    entanglement_matrix=entanglement_matrix,
                )
            elif use_roma_solver:
                sub_solutions = self._solve_with_roma(
                    plan=plan,
                    entanglement_matrix=entanglement_matrix,
                )
            else:
                for sp in plan.sub_problems:
                    self.logger.info(f"Solving: {sp.title}")
                    entangled_with = []
                    entanglement_symbols = []
                    if hasattr(sp, "metadata") and isinstance(sp.metadata, dict):
                        entangled_with = sp.metadata.get("entangled_with", []) or []
                        entanglement_symbols = sp.metadata.get("entanglement_symbols", []) or []
                    context = {
                        "entanglement_matrix": entanglement_matrix,
                        "entangled_with": entangled_with,
                        "entangled_components": entangled_with,
                        "entanglement_symbols": entanglement_symbols,
                    }
                    solution = self.sub_problem_solver.solve(
                        sub_problem=sp,
                        parent_problem=plan.original_problem,
                        context=context
                    )
                    sub_solutions[sp.id] = solution
            
            step_solving.end_time = datetime.now()
            step_solving.status = "completed"
            step_solving.details = {
                'num_solved': len(sub_solutions),
                'avg_quality': sum(s.quality_score for s in sub_solutions.values()) / len(sub_solutions) if sub_solutions else 0
            }
            steps.append(step_solving)
            execution_log.append(f"Solved {len(sub_solutions)} sub-problems")

            roma_stage_results: Dict[str, Dict[str, GauntletOutcome]] = {}
            roma_failed_ids: Set[str] = set()
            if run_gauntlets and self.roma_adapter and self.enable_roma:
                roma_stage_results, roma_failed_ids = self._run_roma_subproblem_checks(
                    plan=plan,
                    sub_solutions=sub_solutions,
                    entanglement_matrix=entanglement_matrix,
                )

            if run_gauntlets and (
                (gauntlet_bundle and gauntlet_bundle.has_subproblem_any()) or roma_stage_results
            ):
                step_gauntlet = SolutionStep("subproblem_gauntlets", datetime.now())
                sub_gauntlet_results: Dict[str, Dict[str, GauntletOutcome]] = {}
                failed_ids: Set[str] = set()
                if gauntlet_bundle and gauntlet_bundle.has_subproblem_any():
                    sub_gauntlet_results, failed_ids = self._run_subproblem_gauntlets(
                        plan=plan,
                        sub_solutions=sub_solutions,
                        entanglement_matrix=entanglement_matrix,
                        gauntlet_bundle=gauntlet_bundle,
                    )
                if roma_stage_results:
                    self._merge_gauntlet_results(sub_gauntlet_results, roma_stage_results)
                gauntlet_results["sub_problems"] = sub_gauntlet_results
                failed_ids = set(failed_ids) | set(roma_failed_ids)

                # Optional self-healing loop for sub-problem failures
                remaining_failures = set(failed_ids)
                previous_failures = set()
                refinement_round = 0
                while (
                    remaining_failures
                    and max_refinement_loops > 0
                    and refinement_round < max_refinement_loops
                    and remaining_failures != previous_failures
                ):
                    refinement_round += 1
                    previous_failures = set(remaining_failures)
                    expanded_targets = self._expand_entangled_ids(
                        remaining_failures, entanglement_matrix
                    )
                    execution_log.append(
                        f"Gauntlet refinement {refinement_round}: re-solving {len(expanded_targets)} sub-problems"
                    )
                    updated_solutions = self._solve_subproblem_subset(
                        plan=plan,
                        sub_solutions=sub_solutions,
                        target_ids=expanded_targets,
                        entanglement_matrix=entanglement_matrix,
                        use_blue_team_solver=use_blue_team_solver,
                        use_roma_solver=use_roma_solver,
                    )
                    sub_solutions.update(updated_solutions)
                    refreshed_results, remaining_failures = self._run_subproblem_gauntlets(
                        plan=plan,
                        sub_solutions=sub_solutions,
                        entanglement_matrix=entanglement_matrix,
                        gauntlet_bundle=gauntlet_bundle,
                        target_ids=expanded_targets,
                    )
                    self._merge_gauntlet_results(
                        gauntlet_results["sub_problems"],
                        refreshed_results,
                    )

                step_gauntlet.end_time = datetime.now()
                step_gauntlet.status = "completed"
                step_gauntlet.details = {
                    "gauntlets_run": self._count_gauntlet_outcomes(gauntlet_results["sub_problems"]),
                    "failed_sub_problems": len(remaining_failures),
                }
                steps.append(step_gauntlet)
                execution_log.append(
                    f"Gauntlets (sub-problems): {step_gauntlet.details['gauntlets_run']} runs, "
                    f"{step_gauntlet.details['failed_sub_problems']} unresolved failures"
                )
        
        # ============================================================================
        # STEP 4: Reassembly
        # ============================================================================
        step_assembly = SolutionStep("reassembly", datetime.now())
        
        if sub_solutions:
            if use_blue_team_solver and ENHANCED_RECOMPOSITION_AVAILABLE:
                final_solution = self._recompose_with_enhanced_engine(
                    plan=plan,
                    sub_solutions=sub_solutions,
                    entanglement_matrix=entanglement_matrix,
                )
            else:
                final_solution = self.recomposition_engine.assemble(
                    plan=plan,
                    sub_solutions=sub_solutions,
                    strategy=self.assembly_strategy,
                    detect_conflicts=detect_conflicts,
                    resolve_conflicts=resolve_conflicts
                )
            final_solution = self._maybe_apply_roma_reassembly(
                plan=plan,
                sub_solutions=sub_solutions,
                final_solution=final_solution,
                entanglement_matrix=entanglement_matrix,
            )
        else:
            # Create empty solution
            from universal_recomposition_engine import IntegratedSolution, QualityMetrics
            final_solution = IntegratedSolution(
                solution_id="empty",
                problem_id=plan.original_problem.id,
                decomposition_plan_id=plan.id,
                assembled_content="# No solutions generated\n",
                assembly_strategy=self.assembly_strategy.value,
                sub_solutions={},
                quality_metrics=QualityMetrics(0, 0, 0, 0, 0),
                conflicts_detected=[],
                conflicts_resolved=[]
            )

        if (
            run_gauntlets
            and gauntlet_bundle
            and (
                gauntlet_bundle.final_red_gauntlet
                or gauntlet_bundle.final_gold_gauntlet
                or (self.roma_adapter and self.enable_roma)
            )
            and sub_solutions
        ):
            step_final_gauntlet = SolutionStep("final_gauntlets", datetime.now())
            final_failures: Set[str] = set()
            previous_failures: Set[str] = set()
            refinement_round = 0

            while True:
                final_results, final_failures = self._run_final_gauntlets(
                    plan=plan,
                    final_solution=final_solution,
                    sub_solutions=sub_solutions,
                    entanglement_matrix=entanglement_matrix,
                    gauntlet_bundle=gauntlet_bundle,
                )
                if self.roma_adapter and self.enable_roma:
                    roma_outcome, roma_failed = self._run_roma_final_checks(
                        plan=plan,
                        final_solution=final_solution,
                    )
                    if roma_outcome:
                        final_results["roma_final"] = roma_outcome
                        if roma_failed:
                            final_failures.update([sp.id for sp in plan.sub_problems])
                gauntlet_results["final"] = final_results

                if not final_failures or max_refinement_loops <= 0:
                    break
                if refinement_round >= max_refinement_loops or final_failures == previous_failures:
                    break

                refinement_round += 1
                previous_failures = set(final_failures)
                expanded_targets = self._expand_entangled_ids(
                    final_failures, entanglement_matrix
                )
                execution_log.append(
                    f"Final gauntlet refinement {refinement_round}: re-solving {len(expanded_targets)} sub-problems"
                )
                updated_solutions = self._solve_subproblem_subset(
                    plan=plan,
                    sub_solutions=sub_solutions,
                    target_ids=expanded_targets,
                    entanglement_matrix=entanglement_matrix,
                    use_blue_team_solver=use_blue_team_solver,
                    use_roma_solver=use_roma_solver,
                )
                sub_solutions.update(updated_solutions)
                if use_blue_team_solver and ENHANCED_RECOMPOSITION_AVAILABLE:
                    final_solution = self._recompose_with_enhanced_engine(
                        plan=plan,
                        sub_solutions=sub_solutions,
                        entanglement_matrix=entanglement_matrix,
                    )
                else:
                    final_solution = self.recomposition_engine.assemble(
                        plan=plan,
                        sub_solutions=sub_solutions,
                        strategy=self.assembly_strategy,
                        detect_conflicts=detect_conflicts,
                        resolve_conflicts=resolve_conflicts,
                    )
                final_solution = self._maybe_apply_roma_reassembly(
                    plan=plan,
                    sub_solutions=sub_solutions,
                    final_solution=final_solution,
                    entanglement_matrix=entanglement_matrix,
                )

            step_final_gauntlet.end_time = datetime.now()
            step_final_gauntlet.status = "completed"
            step_final_gauntlet.details = {
                "gauntlets_run": self._count_gauntlet_outcomes(gauntlet_results["final"]),
                "failed_final_checks": len(final_failures),
            }
            steps.append(step_final_gauntlet)
            execution_log.append(
                f"Gauntlets (final): {step_final_gauntlet.details['gauntlets_run']} runs, "
                f"{step_final_gauntlet.details['failed_final_checks']} unresolved failures"
            )
        
        step_assembly.end_time = datetime.now()
        step_assembly.status = "completed"
        assembly_strategy = getattr(final_solution, "assembly_strategy", None)
        if hasattr(assembly_strategy, "value"):
            assembly_strategy = assembly_strategy.value
        step_assembly.details = {
            'assembly_strategy': assembly_strategy,
            'quality_score': self._extract_quality_score(final_solution),
            'conflicts_detected': self._extract_conflict_counts(final_solution)[0],
            'conflicts_resolved': self._extract_conflict_counts(final_solution)[1]
        }
        steps.append(step_assembly)
        execution_log.append(f"Assembled solution with strategy: {final_solution.assembly_strategy}")
        
        # ============================================================================
        # STEP 5: Finalize Result
        # ============================================================================
        total_duration = (datetime.now() - overall_start).total_seconds()

        if run_gauntlets:
            gauntlet_summary = self._summarize_gauntlet_results(gauntlet_results)
        else:
            gauntlet_summary = {}
        if gauntlet_summary:
            execution_log.append(
                "Gauntlets summary: "
                f"{gauntlet_summary.get('total_runs', 0)} runs, "
                f"{gauntlet_summary.get('failed', 0)} failed, "
                f"{gauntlet_summary.get('skipped', 0)} skipped"
            )

        adjusted_quality = self._apply_gauntlet_quality_adjustment(
            self._extract_quality_score(final_solution),
            gauntlet_summary,
        )
        if hasattr(final_solution, "metadata") and isinstance(final_solution.metadata, dict):
            final_solution.metadata["gauntlet_summary"] = gauntlet_summary
        
        result = SolverResult(
            problem_id=plan.original_problem.id,
            problem_title=plan.original_problem.title,
            problem_statement=problem_statement,
            domain=domain.value if isinstance(domain, ProblemDomain) else domain,
            decomposition_plan=plan,
            sub_problem_solutions=sub_solutions,
            solving_steps=steps,
            final_solution=final_solution,
            quality_score=adjusted_quality,
            total_duration_seconds=total_duration,
            conflicts_detected=self._extract_conflict_counts(final_solution)[0],
            conflicts_resolved=self._extract_conflict_counts(final_solution)[1],
            execution_log=execution_log,
            gauntlet_results=gauntlet_results,
            gauntlet_summary=gauntlet_summary,
        )
        
        self.solution_history.append(result)
        
        self.logger.info(f"Problem solving complete: quality={result.quality_score:.2f}, time={total_duration:.2f}s")
        
        return result

    def _solve_with_blue_team(
        self,
        plan: DecompositionPlan,
        entanglement_matrix: Dict[str, List[str]],
        target_ids: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """Solve sub-problems using Blue Team solver and map to enhanced solution objects."""
        if not self.blue_team_solver:
            return {}
        solutions: Dict[str, Any] = {}
        for sp in plan.sub_problems:
            if target_ids and sp.id not in target_ids:
                continue
            entangled_with = []
            entanglement_symbols = []
            if hasattr(sp, "metadata") and isinstance(sp.metadata, dict):
                entangled_with = sp.metadata.get("entangled_with", []) or []
                entanglement_symbols = sp.metadata.get("entanglement_symbols", []) or []
            if not entangled_with:
                entangled_with = list(entanglement_matrix.get(sp.id, []) or [])

            result = self.blue_team_solver.solve_sub_problem(
                sub_problem_id=sp.id,
                description=f"{sp.title}\n\n{sp.description}",
                dependencies=list(sp.dependencies or []),
                complexity_score=int(round(sp.complexity_score.overall_complexity)),
                priority=getattr(sp, "priority", 5),
                context={
                    "entanglement_matrix": entanglement_matrix,
                    "entangled_with": entangled_with,
                    "entangled_components": entangled_with,
                    "entanglement_symbols": entanglement_symbols,
                    "domain": plan.original_problem.domain.value,
                },
                requirements=[],
                constraints=[c.description for c in plan.original_problem.constraints],
                success_criteria=[sc.description for sc in sp.success_criteria],
                metadata={
                    "entangled_with": entangled_with,
                    "entanglement_symbols": entanglement_symbols,
                    "entanglement_source": sp.metadata.get("entanglement_source")
                    if isinstance(sp.metadata, dict)
                    else None,
                },
            )

            if ENHANCED_RECOMPOSITION_AVAILABLE:
                quality_score = result.quality_metrics.overall_score
                metadata = dict(result.metadata or {})
                metadata.update({
                    "solver_status": result.status.value if hasattr(result.status, "value") else str(result.status),
                    "strategy_used": result.strategy_used.value if hasattr(result.strategy_used, "value") else str(result.strategy_used),
                    "entangled_with": entangled_with,
                    "entanglement_symbols": entanglement_symbols,
                })
                solutions[sp.id] = EnhancedSubProblemSolution(
                    sub_problem_id=sp.id,
                    solution_content=result.solution,
                    quality_score=quality_score,
                    metadata=metadata,
                )
            else:
                solutions[sp.id] = SubProblemSolution(
                    sub_problem_id=sp.id,
                    solution_content=result.solution,
                    quality_score=result.quality_metrics.overall_score,
                    metadata=result.metadata or {},
                )

        return solutions

    def _solve_with_roma(
        self,
        plan: DecompositionPlan,
        entanglement_matrix: Dict[str, List[str]],
        target_ids: Optional[Set[str]] = None,
    ) -> Dict[str, SubProblemSolution]:
        """Solve sub-problems using ROMA phase-2 solver."""
        if not self.roma_adapter:
            return {}

        payload: List[Dict[str, Any]] = []
        for sp in plan.sub_problems:
            if target_ids and sp.id not in target_ids:
                continue
            metadata = dict(sp.metadata or {})
            if "entangled_with" not in metadata:
                metadata["entangled_with"] = entanglement_matrix.get(sp.id, [])
            if "entanglement_symbols" in sp.metadata:
                metadata["entanglement_symbols"] = sp.metadata.get("entanglement_symbols")
            payload.append(
                {
                    "id": sp.id,
                    "title": sp.title,
                    "description": sp.description,
                    "dependencies": list(sp.dependencies or []),
                    "type": sp.type.value if hasattr(sp.type, "value") else str(sp.type),
                    "metadata": metadata,
                }
            )

        result = self.roma_adapter.solve_sub_problems(payload)
        solutions_list = result.get("solutions", []) if isinstance(result, dict) else []

        solutions: Dict[str, SubProblemSolution] = {}
        for sol in solutions_list:
            if not isinstance(sol, dict):
                continue
            sol_id = sol.get("id") or sol.get("sub_problem_id") or sol.get("title")
            if not sol_id:
                sol_id = f"roma_sol_{len(solutions) + 1}"
            content = sol.get("solution") or sol.get("solution_content") or ""
            quality = sol.get("quality_score") or sol.get("confidence") or 0.7
            metadata = sol.get("metadata", {}) if isinstance(sol.get("metadata"), dict) else {}
            for key in ("entangled_with", "entanglement_symbols", "entanglement_source"):
                if key in sol:
                    metadata[key] = sol.get(key)
            solutions[sol_id] = SubProblemSolution(
                sub_problem_id=sol_id,
                solution_content=str(content),
                quality_score=float(quality),
                metadata=metadata,
            )

        # Fill missing sub-problems with standard solver
        missing = [sp for sp in plan.sub_problems if sp.id not in solutions]
        for sp in missing:
            if target_ids and sp.id not in target_ids:
                continue
            context = {
                "entanglement_matrix": entanglement_matrix,
                "entangled_with": entanglement_matrix.get(sp.id, []),
                "entangled_components": entanglement_matrix.get(sp.id, []),
                "entanglement_symbols": sp.metadata.get("entanglement_symbols", []),
            }
            fallback_solution = self.sub_problem_solver.solve(
                sub_problem=sp,
                parent_problem=plan.original_problem,
                context=context,
            )
            solutions[sp.id] = fallback_solution

        return solutions

    def _plan_from_roma_result(
        self,
        problem_def: ProblemDefinition,
        roma_result: Dict[str, Any],
    ) -> Optional[DecompositionPlan]:
        """Convert ROMA decomposition output into a Universal DecompositionPlan."""
        sub_payload = roma_result.get("sub_problems") or []
        if not isinstance(sub_payload, list) or not sub_payload:
            return None

        def _infer_type(title: str, description: str) -> SubProblemType:
            text = f"{title} {description}".lower()
            if "test" in text or "qa" in text:
                return SubProblemType.TESTING
            if "design" in text or "architecture" in text:
                return SubProblemType.DESIGN
            if "analy" in text or "analysis" in text:
                return SubProblemType.ANALYSIS
            if "research" in text or "investigate" in text:
                return SubProblemType.RESEARCH
            if "document" in text or "docs" in text:
                return SubProblemType.DOCUMENTATION
            if "integrat" in text or "interface" in text:
                return SubProblemType.INTEGRATION
            if "validate" in text or "verify" in text:
                return SubProblemType.VALIDATION
            return SubProblemType.IMPLEMENTATION

        sub_problems: List[SubProblem] = []
        for item in sub_payload:
            if not isinstance(item, dict):
                continue
            sp_id = item.get("id") or item.get("sub_problem_id") or f"roma_sp_{uuid.uuid4().hex[:8]}"
            title = item.get("title") or item.get("name") or sp_id
            description = item.get("description") or item.get("detail") or title
            dependencies = item.get("dependencies") or item.get("depends_on") or []
            if isinstance(dependencies, str):
                dependencies = [dependencies]
            complexity_value = (
                item.get("complexity_score")
                or item.get("complexity")
                or item.get("estimated_complexity")
                or problem_def.complexity_score.overall_complexity
            )
            try:
                complexity_value = float(complexity_value)
            except (TypeError, ValueError):
                complexity_value = problem_def.complexity_score.overall_complexity
            complexity_score = ComplexityScore(
                cognitive_complexity=min(10.0, complexity_value),
                computational_complexity=min(10.0, complexity_value),
                domain_complexity=min(10.0, complexity_value),
                integration_complexity=min(10.0, complexity_value),
                overall_complexity=min(10.0, complexity_value),
                explanation="ROMA-derived complexity estimate",
            )
            type_value = item.get("type")
            sp_type = _infer_type(title, description)
            if isinstance(type_value, str):
                try:
                    sp_type = SubProblemType(type_value.lower())
                except ValueError:
                    pass

            success_criteria = []
            for sc in item.get("success_criteria", []) or []:
                if isinstance(sc, str):
                    success_criteria.append(
                        SuccessCriterion(
                            id=f"roma_sc_{uuid.uuid4().hex[:6]}",
                            description=sc,
                            metric="completion",
                            threshold=0.9,
                        )
                    )

            metadata = dict(item.get("metadata") or {})
            for key in ("entangled_with", "entanglement_symbols", "entanglement_source"):
                if key in item:
                    metadata[key] = item.get(key)
            metadata.setdefault("roma_source", True)

            sub_problems.append(
                SubProblem(
                    id=str(sp_id),
                    parent_id=problem_def.id,
                    title=str(title),
                    description=str(description),
                    type=sp_type,
                    complexity_score=complexity_score,
                    dependencies=list(dependencies) if isinstance(dependencies, list) else [],
                    success_criteria=success_criteria,
                    estimated_effort_hours=max(1.0, complexity_value * 3),
                    priority=int(item.get("priority", 5) or 5),
                    status=SubProblemStatus.PENDING,
                    metadata=metadata,
                )
            )

        if not sub_problems:
            return None

        dependency_graph = self.decomposition_engine._build_dependency_graph(sub_problems)
        execution_order = self.decomposition_engine._calculate_execution_order(
            sub_problems, dependency_graph
        )
        parallel_groups = self.decomposition_engine._identify_parallel_groups(
            sub_problems, dependency_graph
        )
        quality_score = self.decomposition_engine._calculate_quality_score(
            problem_def, sub_problems, dependency_graph
        )

        plan = DecompositionPlan(
            id=f"roma_plan_{uuid.uuid4().hex[:10]}",
            original_problem=problem_def,
            sub_problems=sub_problems,
            strategy_used=self.decomposition_strategy,
            dependency_graph=dependency_graph,
            execution_order=execution_order,
            parallel_groups=parallel_groups,
            quality_score=quality_score,
            metadata={},
            analyzed_context={},
        )
        if isinstance(roma_result, dict):
            plan.metadata["roma_result"] = roma_result
        entanglement_matrix = roma_result.get("entanglement_matrix", {}) if isinstance(roma_result, dict) else {}
        if entanglement_matrix:
            normalized = normalize_entanglement_matrix(
                entanglement_matrix,
                allowed_ids=[sp.id for sp in sub_problems],
                enforce_symmetry=True,
                strict=False,
            )
            serialized = {key: sorted(list(val)) for key, val in normalized.items()}
            plan.metadata["entanglement_matrix"] = serialized
            plan.analyzed_context["entanglement_matrix"] = serialized
            for sp in sub_problems:
                entangled_with = serialized.get(sp.id, [])
                if entangled_with:
                    sp.metadata.setdefault("entangled_with", entangled_with)
                    sp.metadata.setdefault("entanglement_source", "roma")

        return plan

    def _recompose_with_enhanced_engine(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        entanglement_matrix: Dict[str, List[str]],
    ) -> Any:
        """Recompose using EnhancedRecompositionEngine with entanglement context."""
        recomposition_config = None
        if (
            plan.original_problem.domain == ProblemDomain.WEB3
            and EnhancedRecompositionConfig is not None
        ):
            recomposition_config = EnhancedRecompositionConfig(
                enable_defi_gauntlet=True,
                defi_max_attack_vectors=4,
                defi_symbolic_timeout_seconds=12.0,
            )
        engine = EnhancedRecompositionEngine(config=recomposition_config)
        if plan.original_problem.domain == ProblemDomain.WEB3:
            web3_meta = plan.metadata.get("web3", {}) if isinstance(plan.metadata, dict) else {}
            if isinstance(web3_meta, dict):
                attack_vectors = []
                ingestion = web3_meta.get("ingestion", {})
                if isinstance(ingestion, dict):
                    slither = ingestion.get("slither", {})
                    if isinstance(slither, dict):
                        for finding in slither.get("findings", [])[:3]:
                            if isinstance(finding, dict):
                                attack_vectors.append(
                                    {
                                        "id": finding.get("check", "slither_finding"),
                                        "name": finding.get("check", "Slither Finding"),
                                        "goal": finding.get("description", ""),
                                        "predicate": finding.get("impact", ""),
                                    }
                                )
                engine.configure_defi_gauntlet(
                    additional_vectors=attack_vectors or None,
                    symbolic_timeout_seconds=12.0,
                )
        dependency_graph = plan.dependency_graph
        return engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=plan.original_problem.id,
            decomposition_plan_id=plan.id,
            dependency_graph=dependency_graph,
            entanglement_matrix=entanglement_matrix or None,
        )

    def _build_roma_solution_payload(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        entanglement_matrix: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for sp_id, sol in sub_solutions.items():
            content = getattr(sol, "solution_content", None) or getattr(sol, "solution", "")
            metadata = sol.metadata if hasattr(sol, "metadata") else {}
            payload.append(
                {
                    "id": sp_id,
                    "solution": content,
                    "dependencies": plan.dependency_graph.get(sp_id, []),
                    "metadata": metadata if isinstance(metadata, dict) else {},
                    "entangled_with": entanglement_matrix.get(sp_id, []),
                }
            )
        return payload

    @staticmethod
    def _roma_outcome_from_item(
        item: Dict[str, Any],
        stage: str,
        default_approved: bool = True,
    ) -> GauntletOutcome:
        approved = item.get("approved")
        if approved is None:
            approved = item.get("is_approved")
        score = item.get("score") or item.get("overall_score") or item.get("confidence")
        if approved is None and score is not None:
            try:
                approved = float(score) >= 0.6
            except (TypeError, ValueError):
                approved = default_approved
        if approved is None:
            approved = default_approved

        summary = (
            item.get("summary")
            or item.get("critique")
            or item.get("verification")
            or item.get("message")
            or ""
        )
        findings = item.get("findings") or item.get("issues") or []
        targeted_feedback = []
        if isinstance(findings, list):
            for finding in findings:
                if isinstance(finding, dict):
                    detail = finding.get("finding") or finding.get("issue") or finding.get("detail")
                    if detail:
                        targeted_feedback.append(str(detail))
                elif isinstance(finding, str):
                    targeted_feedback.append(finding)

        return GauntletOutcome(
            stage=stage,
            gauntlet_name=f"roma_{stage}",
            team_name="ROMA",
            team_role="ROMA",
            is_approved=bool(approved),
            report_summary=str(summary),
            report_object=item,
            targeted_feedback=targeted_feedback,
        )

    def _run_roma_subproblem_checks(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        entanglement_matrix: Dict[str, List[str]],
    ) -> Tuple[Dict[str, Dict[str, GauntletOutcome]], Set[str]]:
        if not self.roma_adapter or not self.roma_adapter.is_available():
            return {}, set()

        payload = self._build_roma_solution_payload(plan, sub_solutions, entanglement_matrix)
        outcomes: Dict[str, Dict[str, GauntletOutcome]] = {sp_id: {} for sp_id in sub_solutions}
        failed_ids: Set[str] = set()

        critique_result = self.roma_adapter.critique_solutions(
            solutions=payload,
            problem_statement=plan.original_problem.description,
        )
        critiques = critique_result.get("critiques") if isinstance(critique_result, dict) else None
        if isinstance(critiques, list) and critiques:
            for item in critiques:
                if not isinstance(item, dict):
                    continue
                sol_id = item.get("solution_id") or item.get("id") or item.get("sub_problem_id")
                if not sol_id:
                    continue
                outcome = self._roma_outcome_from_item(item, "roma_critique")
                outcomes.setdefault(sol_id, {})["roma_critique"] = outcome
                if not outcome.is_approved:
                    failed_ids.add(sol_id)
        else:
            for sp_id in sub_solutions:
                outcome = self._roma_outcome_from_item(
                    critique_result if isinstance(critique_result, dict) else {},
                    "roma_critique",
                    default_approved=True,
                )
                outcomes.setdefault(sp_id, {})["roma_critique"] = outcome
                if not outcome.is_approved:
                    failed_ids.add(sp_id)

        verify_result = self.roma_adapter.verify_solutions(
            solutions=payload,
            requirements=[sc.description for sc in plan.original_problem.success_criteria],
            problem_statement=plan.original_problem.description,
        )
        verifications = verify_result.get("verifications") if isinstance(verify_result, dict) else None
        if verifications is None and isinstance(verify_result, dict):
            verifications = verify_result.get("results") or verify_result.get("verification_results")
        if isinstance(verifications, list) and verifications:
            for item in verifications:
                if not isinstance(item, dict):
                    continue
                sol_id = item.get("solution_id") or item.get("id") or item.get("sub_problem_id")
                if not sol_id:
                    continue
                outcome = self._roma_outcome_from_item(item, "roma_verify")
                outcomes.setdefault(sol_id, {})["roma_verify"] = outcome
                if not outcome.is_approved:
                    failed_ids.add(sol_id)
        else:
            for sp_id in sub_solutions:
                outcome = self._roma_outcome_from_item(
                    verify_result if isinstance(verify_result, dict) else {},
                    "roma_verify",
                    default_approved=True,
                )
                outcomes.setdefault(sp_id, {})["roma_verify"] = outcome
                if not outcome.is_approved:
                    failed_ids.add(sp_id)

        return outcomes, failed_ids

    def _run_roma_final_checks(
        self,
        plan: DecompositionPlan,
        final_solution: Any,
    ) -> Tuple[Optional[GauntletOutcome], bool]:
        if not self.roma_adapter or not self.roma_adapter.is_available():
            return None, False
        content = getattr(final_solution, "assembled_content", None) or ""
        result = self.roma_adapter.final_validation(
            final_solution=content,
            problem_statement=plan.original_problem.description,
        )
        outcome = self._roma_outcome_from_item(
            result if isinstance(result, dict) else {},
            "roma_final_validation",
        )
        return outcome, not outcome.is_approved

    def _maybe_apply_roma_reassembly(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        final_solution: Any,
        entanglement_matrix: Dict[str, List[str]],
    ) -> Any:
        if not self.roma_adapter or not self.roma_adapter.is_available():
            return final_solution
        if self.assembly_strategy not in (
            AssemblyStrategy.ROMA_DETERMINISTIC,
            AssemblyStrategy.ROMA_CREATIVE,
        ):
            return final_solution

        payload = self._build_roma_solution_payload(plan, sub_solutions, entanglement_matrix)
        roma_result = self.roma_adapter.reassemble_solutions(
            solutions=payload,
            problem_statement=plan.original_problem.description,
        )
        if not isinstance(roma_result, dict):
            return final_solution
        roma_content = roma_result.get("final_solution") or roma_result.get("reassembled_solution")
        if not roma_content:
            return final_solution
        if hasattr(final_solution, "assembled_content"):
            final_solution.assembled_content = roma_content
        metadata = getattr(final_solution, "metadata", None)
        if isinstance(metadata, dict):
            metadata["roma_reassembly"] = {
                "roma_used": roma_result.get("roma_used", False),
                "roma_type": roma_result.get("roma_type"),
                "message": roma_result.get("message"),
            }
        return final_solution

    def _resolve_gauntlet_bundle(self, config_override: Optional[Dict[str, Any]] = None) -> GauntletBundle:
        """Resolve gauntlet definitions and teams from config or managers."""
        config: Dict[str, Any] = {}
        if self.gauntlet_config:
            config.update(self.gauntlet_config)
        if config_override:
            config.update(config_override)

        team_overrides = config.get("team_overrides", {}) if isinstance(config, dict) else {}

        bundle = GauntletBundle()
        bundle.solver_generation_gauntlet = self._resolve_gauntlet_definition(
            config.get("solver_generation_gauntlet") or config.get("solver_generation")
        )
        bundle.sub_problem_red_gauntlet = self._resolve_gauntlet_definition(
            config.get("sub_problem_red_gauntlet") or config.get("sub_red_gauntlet")
        )
        bundle.sub_problem_gold_gauntlet = self._resolve_gauntlet_definition(
            config.get("sub_problem_gold_gauntlet") or config.get("sub_gold_gauntlet")
        )
        bundle.final_red_gauntlet = self._resolve_gauntlet_definition(
            config.get("final_red_gauntlet") or config.get("final_red")
        )
        bundle.final_gold_gauntlet = self._resolve_gauntlet_definition(
            config.get("final_gold_gauntlet") or config.get("final_gold")
        )

        if not bundle.solver_generation_gauntlet:
            bundle.solver_generation_gauntlet = self._select_gauntlet_by_role("Blue", "solver")
        if not bundle.sub_problem_red_gauntlet:
            bundle.sub_problem_red_gauntlet = self._select_gauntlet_by_role("Red", "sub")
        if not bundle.sub_problem_gold_gauntlet:
            bundle.sub_problem_gold_gauntlet = self._select_gauntlet_by_role("Gold", "sub")
        if not bundle.final_red_gauntlet:
            bundle.final_red_gauntlet = self._select_gauntlet_by_role("Red", "final")
        if not bundle.final_gold_gauntlet:
            bundle.final_gold_gauntlet = self._select_gauntlet_by_role("Gold", "final")

        bundle.solver_generation_team = self._resolve_team_for_gauntlet(
            bundle.solver_generation_gauntlet,
            team_overrides.get("solver_generation_team") or team_overrides.get("solver_generation"),
            role_hint="Blue",
        )
        bundle.sub_problem_red_team = self._resolve_team_for_gauntlet(
            bundle.sub_problem_red_gauntlet,
            team_overrides.get("sub_problem_red_team") or team_overrides.get("sub_red_team"),
            role_hint="Red",
        )
        bundle.sub_problem_gold_team = self._resolve_team_for_gauntlet(
            bundle.sub_problem_gold_gauntlet,
            team_overrides.get("sub_problem_gold_team") or team_overrides.get("sub_gold_team"),
            role_hint="Gold",
        )
        bundle.final_red_team = self._resolve_team_for_gauntlet(
            bundle.final_red_gauntlet,
            team_overrides.get("final_red_team"),
            role_hint="Red",
        )
        bundle.final_gold_team = self._resolve_team_for_gauntlet(
            bundle.final_gold_gauntlet,
            team_overrides.get("final_gold_team"),
            role_hint="Gold",
        )

        return bundle

    def _resolve_gauntlet_definition(self, value: Any) -> Optional[Any]:
        """Resolve gauntlet definition from name, object, or dict."""
        if value is None:
            return None
        if hasattr(value, "rounds") and hasattr(value, "name"):
            return value
        if isinstance(value, str):
            return self.gauntlet_manager.get_gauntlet(value)
        if isinstance(value, dict) and value.get("name"):
            try:
                from openevolve_structures import GauntletDefinition, GauntletRoundRule
                rounds_data = value.get("rounds", [])
                rounds = [
                    r if isinstance(r, GauntletRoundRule) else GauntletRoundRule(**r)
                    for r in rounds_data
                ]
                return GauntletDefinition(
                    name=value["name"],
                    team_name=value.get("team_name", ""),
                    rounds=rounds,
                    description=value.get("description"),
                    attack_modes=value.get("attack_modes", []),
                    generation_mode=value.get("generation_mode", "single_candidate"),
                    gauntlet_type=value.get("gauntlet_type", "standard"),
                    gauntlet_config=value.get("gauntlet_config"),
                )
            except Exception:
                return None
        return None

    def _resolve_team_for_gauntlet(
        self,
        gauntlet_def: Optional[Any],
        override: Optional[Any],
        role_hint: Optional[str] = None,
    ) -> Optional[Any]:
        """Resolve team for a gauntlet using overrides or manager lookups."""
        if override is not None:
            if hasattr(override, "members") and hasattr(override, "role"):
                return override
            if isinstance(override, str):
                team = self.team_manager.get_team(override)
                if team:
                    return team

        if gauntlet_def and getattr(gauntlet_def, "team_name", None):
            team = self.team_manager.get_team(gauntlet_def.team_name)
            if team:
                return team

        if role_hint:
            teams = self.team_manager.get_teams_by_role(role_hint)
            if teams:
                return teams[0]

        return None

    def _select_gauntlet_by_role(self, role: str, stage_hint: str) -> Optional[Any]:
        """Pick a gauntlet by team role with name-based ranking."""
        candidates = []
        for gauntlet in self.gauntlet_manager.get_all_gauntlets():
            team = self.team_manager.get_team(gauntlet.team_name)
            if not team or team.role.lower() != role.lower():
                continue
            name_lower = gauntlet.name.lower()
            score = 0
            if stage_hint in name_lower:
                score += 2
            if role.lower() in name_lower:
                score += 1
            candidates.append((score, gauntlet))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
        return candidates[0][1]

    def _run_subproblem_gauntlets(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        entanglement_matrix: Dict[str, List[str]],
        gauntlet_bundle: GauntletBundle,
        target_ids: Optional[Set[str]] = None,
    ) -> Tuple[Dict[str, Dict[str, GauntletOutcome]], Set[str]]:
        """Run Blue/Red/Gold gauntlets for sub-problem solutions."""
        results: Dict[str, Dict[str, GauntletOutcome]] = {}
        failed_ids: Set[str] = set()

        for sp in plan.sub_problems:
            if target_ids and sp.id not in target_ids:
                continue
            solution = sub_solutions.get(sp.id)
            if not solution:
                continue

            entangled_with = []
            entanglement_symbols = []
            if hasattr(sp, "metadata") and isinstance(sp.metadata, dict):
                entangled_with = sp.metadata.get("entangled_with", []) or []
                entanglement_symbols = sp.metadata.get("entanglement_symbols", []) or []

            context = {
                "solution_id": sp.id,
                "sub_problem": sp,
                "problem_statement": plan.original_problem.description,
                "domain": plan.original_problem.domain.value if hasattr(plan.original_problem.domain, "value") else str(plan.original_problem.domain),
                "constraints": [c.description for c in plan.original_problem.constraints],
                "success_criteria": [sc.description for sc in sp.success_criteria],
                "dependency_graph": plan.dependency_graph,
                "entanglement_matrix": entanglement_matrix,
                "entangled_with": entangled_with,
                "entanglement_symbols": entanglement_symbols,
                "domain_artifacts": plan.metadata.get("domain_artifacts", {}) if isinstance(plan.metadata, dict) else {},
            }
            if plan.original_problem.domain == ProblemDomain.WEB3:
                context["attack_vectors"] = [
                    "flash_loan_attack",
                    "reentrancy_attack",
                    "symbolic_execution_probe",
                    "oracle_manipulation",
                ]

            stage_results: Dict[str, GauntletOutcome] = {}

            if gauntlet_bundle.solver_generation_gauntlet and gauntlet_bundle.solver_generation_team:
                outcome = self._execute_gauntlet_stage(
                    "blue",
                    solution.solution_content,
                    gauntlet_bundle.solver_generation_gauntlet,
                    gauntlet_bundle.solver_generation_team,
                    context,
                )
                stage_results["blue"] = outcome
                if not outcome.is_approved and not outcome.skipped:
                    failed_ids.update(outcome.targeted_feedback or [sp.id])

            if gauntlet_bundle.sub_problem_red_gauntlet and gauntlet_bundle.sub_problem_red_team:
                outcome = self._execute_gauntlet_stage(
                    "red",
                    solution.solution_content,
                    gauntlet_bundle.sub_problem_red_gauntlet,
                    gauntlet_bundle.sub_problem_red_team,
                    context,
                )
                stage_results["red"] = outcome
                if not outcome.is_approved and not outcome.skipped:
                    failed_ids.update(outcome.targeted_feedback or [sp.id])

            if gauntlet_bundle.sub_problem_gold_gauntlet and gauntlet_bundle.sub_problem_gold_team:
                outcome = self._execute_gauntlet_stage(
                    "gold",
                    solution.solution_content,
                    gauntlet_bundle.sub_problem_gold_gauntlet,
                    gauntlet_bundle.sub_problem_gold_team,
                    context,
                )
                stage_results["gold"] = outcome
                if not outcome.is_approved and not outcome.skipped:
                    failed_ids.update(outcome.targeted_feedback or [sp.id])

            if stage_results:
                results[sp.id] = stage_results
                self._attach_gauntlet_metadata(solution, stage_results)

        return results, failed_ids

    def _run_final_gauntlets(
        self,
        plan: DecompositionPlan,
        final_solution: Any,
        sub_solutions: Dict[str, Any],
        entanglement_matrix: Dict[str, List[str]],
        gauntlet_bundle: GauntletBundle,
    ) -> Tuple[Dict[str, GauntletOutcome], Set[str]]:
        """Run final Red/Gold gauntlets on assembled solution."""
        results: Dict[str, GauntletOutcome] = {}
        failed_ids: Set[str] = set()

        context = {
            "solution_id": plan.original_problem.id,
            "final_solution": final_solution,
            "problem_statement": plan.original_problem.description,
            "domain": plan.original_problem.domain.value if hasattr(plan.original_problem.domain, "value") else str(plan.original_problem.domain),
            "constraints": [c.description for c in plan.original_problem.constraints],
            "success_criteria": [sc.description for sc in plan.original_problem.success_criteria],
            "dependency_graph": plan.dependency_graph,
            "entanglement_matrix": entanglement_matrix,
            "sub_problem_solutions": {
                sp_id: getattr(sol, "solution_content", str(sol))
                for sp_id, sol in sub_solutions.items()
            },
            "domain_artifacts": plan.metadata.get("domain_artifacts", {}) if isinstance(plan.metadata, dict) else {},
        }
        if plan.original_problem.domain == ProblemDomain.WEB3:
            context["attack_vectors"] = [
                "flash_loan_attack",
                "reentrancy_attack",
                "symbolic_execution_probe",
                "oracle_manipulation",
            ]

        if gauntlet_bundle.final_red_gauntlet and gauntlet_bundle.final_red_team:
            outcome = self._execute_gauntlet_stage(
                "final_red",
                final_solution.assembled_content,
                gauntlet_bundle.final_red_gauntlet,
                gauntlet_bundle.final_red_team,
                context,
            )
            results["red"] = outcome
            if not outcome.is_approved and not outcome.skipped:
                failed_ids.update(outcome.targeted_feedback)

        if gauntlet_bundle.final_gold_gauntlet and gauntlet_bundle.final_gold_team:
            outcome = self._execute_gauntlet_stage(
                "final_gold",
                final_solution.assembled_content,
                gauntlet_bundle.final_gold_gauntlet,
                gauntlet_bundle.final_gold_team,
                context,
            )
            results["gold"] = outcome
            if not outcome.is_approved and not outcome.skipped:
                failed_ids.update(outcome.targeted_feedback)

        return results, failed_ids

    def _execute_gauntlet_stage(
        self,
        stage: str,
        solution_content: str,
        gauntlet_def: Any,
        team: Any,
        context: Dict[str, Any],
    ) -> GauntletOutcome:
        """Execute a gauntlet stage using the headless pipeline with offline fallback."""
        if not gauntlet_def or not team:
            return GauntletOutcome(
                stage=stage,
                gauntlet_name=getattr(gauntlet_def, "name", None),
                team_name=getattr(team, "name", None),
                team_role=getattr(team, "role", None),
                is_approved=True,
                report_summary="Gauntlet skipped (missing configuration)",
                skipped=True,
            )

        if not getattr(team, "members", None):
            return GauntletOutcome(
                stage=stage,
                gauntlet_name=getattr(gauntlet_def, "name", None),
                team_name=getattr(team, "name", None),
                team_role=getattr(team, "role", None),
                is_approved=False,
                report_summary="Gauntlet skipped (no team members configured)",
                skipped=True,
            )

        try:
            import os
            import sys
            module_dir = os.path.abspath(os.path.dirname(__file__))
            if module_dir not in sys.path:
                sys.path.insert(0, module_dir)
            from workflow_engine import run_gauntlet_headless, parse_targeted_feedback
            result = run_gauntlet_headless(
                solution_content=solution_content,
                gauntlet_def=gauntlet_def,
                team=team,
                context=context,
            )
            report_obj = result.get("report_object") or result.get("critique_report") or result.get("verification_report")
            targeted_feedback = []
            if report_obj:
                targeted_feedback = parse_targeted_feedback(report_obj)
            return GauntletOutcome(
                stage=stage,
                gauntlet_name=getattr(gauntlet_def, "name", None),
                team_name=getattr(team, "name", None),
                team_role=getattr(team, "role", None),
                is_approved=bool(result.get("is_approved", False)),
                report_summary=result.get("report_summary", ""),
                report_object=report_obj,
                targeted_feedback=targeted_feedback,
                logs=result.get("logs", []) or [],
            )
        except Exception as exc:
            return self._execute_gauntlet_offline(
                stage=stage,
                solution_content=solution_content,
                gauntlet_def=gauntlet_def,
                team=team,
                context=context,
                error=exc,
            )

    def _execute_gauntlet_offline(
        self,
        stage: str,
        solution_content: str,
        gauntlet_def: Any,
        team: Any,
        context: Dict[str, Any],
        error: Exception,
    ) -> GauntletOutcome:
        """Fallback gauntlet execution without external LLM calls."""
        min_score = self._gauntlet_min_score(gauntlet_def)
        team_role = getattr(team, "role", None)

        if team_role == "Red":
            from red_team import RedTeam
            from workflow_structures import CritiqueReport

            red_team = RedTeam()
            assessment = red_team.assess_content(
                solution_content,
                content_type="general",
                attack_modes=getattr(gauntlet_def, "attack_modes", None),
            )

            findings = []
            for finding in assessment.findings:
                findings.append({
                    "title": finding.title,
                    "description": finding.description,
                    "severity": finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity),
                    "category": finding.category.value if hasattr(finding.category, "value") else str(finding.category),
                    "location": finding.location,
                    "confidence": finding.confidence,
                    "suggested_fix": finding.suggested_fix,
                })

            severity_scores: Dict[str, float] = {}
            for finding in findings:
                severity_scores[finding["severity"]] = severity_scores.get(finding["severity"], 0.0) + 1.0

            has_critical = any(finding["severity"] == "critical" for finding in findings)
            is_approved = assessment.confidence_score >= min_score and not has_critical
            summary = assessment.assessment_summary

            report = CritiqueReport(
                solution_attempt_id=context.get("solution_id", "unknown"),
                gauntlet_name=getattr(gauntlet_def, "name", "offline_red_gauntlet"),
                is_approved=is_approved,
                reports_by_judge=[{
                    "member": "offline_red_team",
                    "score": assessment.confidence_score,
                    "justification": summary,
                    "targeted_feedback": [context.get("solution_id")] if not is_approved else [],
                }],
                summary=summary,
                overall_score=assessment.confidence_score,
                flaw_severity_scores=severity_scores,
                identified_flaws=findings,
                suggested_improvements=[
                    finding.get("suggested_fix")
                    for finding in findings
                    if finding.get("suggested_fix")
                ],
            )

            return GauntletOutcome(
                stage=stage,
                gauntlet_name=getattr(gauntlet_def, "name", None),
                team_name=getattr(team, "name", None),
                team_role=team_role,
                is_approved=is_approved,
                report_summary=summary,
                report_object=report,
                targeted_feedback=[context.get("solution_id")] if not is_approved else [],
                logs=[f"Gauntlet fallback used: {error}"],
                error=str(error),
            )

        from quality_assessment import QualityAssessmentEngine
        from workflow_structures import VerificationReport

        assessor = QualityAssessmentEngine()
        assessment = assessor.assess_quality(
            solution_content,
            content_type="general",
            custom_requirements={
                "constraints": context.get("constraints", []),
                "success_criteria": context.get("success_criteria", []),
            },
        )
        raw_score = assessment.composite_score
        score = raw_score / 100.0 if raw_score > 1.0 else raw_score
        is_approved = score >= min_score
        dimension_scores = {
            dim.value if hasattr(dim, "value") else str(dim): val / 100.0 if val > 1.0 else val
            for dim, val in assessment.scores.items()
        }

        report = VerificationReport(
            solution_attempt_id=context.get("solution_id", "unknown"),
            gauntlet_name=getattr(gauntlet_def, "name", "offline_gold_gauntlet"),
            is_approved=is_approved,
            reports_by_judge=[{
                "member": "offline_quality_assessor",
                "score": score,
                "justification": "Offline quality assessment",
                "targeted_feedback": [context.get("solution_id")] if not is_approved else [],
            }],
            average_score=score,
            score_variance=0.0,
            summary="Offline quality assessment",
            dimension_scores=dimension_scores,
            criteria_met=[rec for rec in assessment.recommendations if "improve" not in rec.lower()],
            criteria_not_met=[rec for rec in assessment.recommendations if "improve" in rec.lower()],
        )

        return GauntletOutcome(
            stage=stage,
            gauntlet_name=getattr(gauntlet_def, "name", None),
            team_name=getattr(team, "name", None),
            team_role=team_role,
            is_approved=is_approved,
            report_summary="Offline quality assessment",
            report_object=report,
            targeted_feedback=[context.get("solution_id")] if not is_approved else [],
            logs=[f"Gauntlet fallback used: {error}"],
            error=str(error),
        )

    def _gauntlet_min_score(self, gauntlet_def: Any) -> float:
        """Determine minimum approval score from gauntlet definition."""
        try:
            rounds = getattr(gauntlet_def, "rounds", []) or []
            if rounds:
                return float(getattr(rounds[0], "min_overall_confidence", 0.7))
        except Exception:
            pass
        return 0.7

    def _attach_gauntlet_metadata(
        self,
        solution: Any,
        stage_results: Dict[str, GauntletOutcome],
    ) -> None:
        """Attach gauntlet outcome summaries to a solution metadata dict."""
        metadata = getattr(solution, "metadata", None)
        if metadata is None or not isinstance(metadata, dict):
            return
        metadata.setdefault("gauntlet_results", {})
        for stage, outcome in stage_results.items():
            metadata["gauntlet_results"][stage] = outcome.to_dict()
            if not outcome.is_approved and outcome.targeted_feedback:
                metadata.setdefault("gauntlet_feedback", []).extend(outcome.targeted_feedback)

    def _expand_entangled_ids(
        self,
        ids: Set[str],
        entanglement_matrix: Dict[str, List[str]],
    ) -> Set[str]:
        """Expand IDs with their entangled partners."""
        expanded = set(ids)
        if not entanglement_matrix:
            return expanded
        normalized = normalize_entanglement_matrix(entanglement_matrix)
        for sp_id in list(expanded):
            expanded.update(normalized.get(sp_id, set()))
        return expanded

    def _solve_subproblem_subset(
        self,
        plan: DecompositionPlan,
        sub_solutions: Dict[str, Any],
        target_ids: Set[str],
        entanglement_matrix: Dict[str, List[str]],
        use_blue_team_solver: bool,
        use_roma_solver: bool,
    ) -> Dict[str, Any]:
        """Solve a subset of sub-problems, optionally using Blue Team solver."""
        if not target_ids:
            return {}
        if use_blue_team_solver and self.blue_team_solver:
            return self._solve_with_blue_team(
                plan=plan,
                entanglement_matrix=entanglement_matrix,
                target_ids=target_ids,
            )
        if use_roma_solver:
            return self._solve_with_roma(
                plan=plan,
                entanglement_matrix=entanglement_matrix,
                target_ids=target_ids,
            )

        updated: Dict[str, Any] = {}
        for sp in plan.sub_problems:
            if sp.id not in target_ids:
                continue
            entangled_with = []
            entanglement_symbols = []
            if hasattr(sp, "metadata") and isinstance(sp.metadata, dict):
                entangled_with = sp.metadata.get("entangled_with", []) or []
                entanglement_symbols = sp.metadata.get("entanglement_symbols", []) or []
            context = {
                "entanglement_matrix": entanglement_matrix,
                "entangled_with": entangled_with,
                "entangled_components": entangled_with,
                "entanglement_symbols": entanglement_symbols,
            }
            existing = sub_solutions.get(sp.id)
            if existing and isinstance(getattr(existing, "metadata", None), dict):
                feedback = existing.metadata.get("gauntlet_feedback")
                if feedback:
                    context["gauntlet_feedback"] = feedback
            updated[sp.id] = self.sub_problem_solver.solve(
                sub_problem=sp,
                parent_problem=plan.original_problem,
                context=context,
            )
        return updated

    def _merge_gauntlet_results(
        self,
        base: Dict[str, Dict[str, GauntletOutcome]],
        updates: Dict[str, Dict[str, GauntletOutcome]],
    ) -> None:
        for sp_id, stage_results in updates.items():
            base.setdefault(sp_id, {}).update(stage_results)

    def _count_gauntlet_outcomes(self, results: Any) -> int:
        """Count gauntlet outcomes in nested result structures."""
        if not results:
            return 0
        if isinstance(results, GauntletOutcome):
            return 1
        if isinstance(results, dict):
            count = 0
            for value in results.values():
                if isinstance(value, GauntletOutcome):
                    count += 1
                elif isinstance(value, dict):
                    count += len(value)
            return count
        return 0

    def _summarize_gauntlet_results(self, gauntlet_results: Dict[str, Any]) -> Dict[str, Any]:
        total_runs = 0
        failed = 0
        skipped = 0
        failed_final = 0

        sub_results = gauntlet_results.get("sub_problems", {})
        for stage_results in sub_results.values():
            for outcome in stage_results.values():
                total_runs += 0 if outcome.skipped else 1
                skipped += 1 if outcome.skipped else 0
                if not outcome.skipped and not outcome.is_approved:
                    failed += 1

        final_results = gauntlet_results.get("final", {})
        for outcome in final_results.values():
            total_runs += 0 if outcome.skipped else 1
            skipped += 1 if outcome.skipped else 0
            if not outcome.skipped and not outcome.is_approved:
                failed += 1
                failed_final += 1

        return {
            "total_runs": total_runs,
            "failed": failed,
            "skipped": skipped,
            "failed_final": failed_final,
        }

    def _apply_gauntlet_quality_adjustment(
        self,
        base_quality: float,
        gauntlet_summary: Dict[str, Any],
    ) -> float:
        if not gauntlet_summary:
            return base_quality
        total_runs = gauntlet_summary.get("total_runs", 0)
        failed = gauntlet_summary.get("failed", 0)
        failed_final = gauntlet_summary.get("failed_final", 0)
        if total_runs <= 0:
            return base_quality
        fail_ratio = failed / total_runs if total_runs else 0.0
        adjusted = base_quality * max(0.7, 1.0 - (0.3 * fail_ratio))
        if failed_final:
            adjusted *= 0.8
        return max(0.0, min(1.0, adjusted))

    @staticmethod
    def _extract_quality_score(final_solution: Any) -> float:
        if hasattr(final_solution, "quality_metrics") and hasattr(final_solution.quality_metrics, "overall_score"):
            return final_solution.quality_metrics.overall_score
        if isinstance(getattr(final_solution, "quality_metrics", None), dict):
            return final_solution.quality_metrics.get("overall_score", 0.0)
        return 0.0

    @staticmethod
    def _extract_conflict_counts(final_solution: Any) -> Tuple[int, int]:
        detected = getattr(final_solution, "conflicts_detected", []) or []
        resolved = getattr(final_solution, "conflicts_resolved", []) or []
        return len(detected), len(resolved)
    
    def _parse_domain(self, domain_str: str) -> ProblemDomain:
        """Parse domain from string"""
        domain_map = {
            'software': ProblemDomain.SOFTWARE,
            'finance': ProblemDomain.FINANCE,
            'financial': ProblemDomain.FINANCE,
            'trading': ProblemDomain.FINANCE,
            'web3': ProblemDomain.WEB3,
            'defi': ProblemDomain.WEB3,
            'smart_contract': ProblemDomain.WEB3,
            'smart-contract': ProblemDomain.WEB3,
            'solidity': ProblemDomain.WEB3,
            'evm': ProblemDomain.WEB3,
            'onchain': ProblemDomain.WEB3,
            'on-chain': ProblemDomain.WEB3,
            'rust_contract': ProblemDomain.WEB3,
            'scientific': ProblemDomain.SCIENTIFIC,
            'research': ProblemDomain.SCIENTIFIC,
            'healthcare': ProblemDomain.HEALTHCARE,
            'medical': ProblemDomain.HEALTHCARE,
            'manufacturing': ProblemDomain.MANUFACTURING,
            'legal': ProblemDomain.LEGAL,
            'business': ProblemDomain.BUSINESS,
            'education': ProblemDomain.EDUCATION,
            'generic': ProblemDomain.GENERIC
        }
        return domain_map.get(domain_str.lower(), ProblemDomain.GENERIC)
    
    def _detect_domain(self, problem_statement: str) -> ProblemDomain:
        """Auto-detect domain from problem statement"""
        lower = problem_statement.lower()

        # Web3 indicators
        web3_terms = [
            'web3', 'defi', 'smart contract', 'solidity', 'evm', 'onchain',
            'on-chain', 'slither', 'foundry', 'forge', 'hardhat', 'reentrancy',
            'flash loan', 'oracle manipulation', 'amm', 'liquidity pool', 'vault',
            'bridge', 'rust contract', 'anchor',
        ]
        if any(term in lower for term in web3_terms):
            return ProblemDomain.WEB3
        
        # Finance indicators
        finance_terms = ['trading', 'risk', 'portfolio', 'market data', 'compliance', 
                        'regulatory', 'mifid', 'settlement', 'clearing', 'derivative',
                        'equity', 'fixed income', 'fx', 'credit risk', 'var']
        if any(term in lower for term in finance_terms):
            return ProblemDomain.FINANCE
        
        # Software indicators
        software_terms = ['api', 'microservice', 'database', 'frontend', 'backend',
                         'authentication', 'authorization', 'cloud', 'deployment',
                         'kubernetes', 'docker', 'server', 'client']
        if any(term in lower for term in software_terms):
            return ProblemDomain.SOFTWARE
        
        # Scientific indicators
        science_terms = ['experiment', 'hypothesis', 'data analysis', 'research',
                        'statistical', 'machine learning', 'algorithm', 'simulation']
        if any(term in lower for term in science_terms):
            return ProblemDomain.SCIENTIFIC
        
        # Healthcare indicators
        health_terms = ['patient', 'clinical', 'diagnosis', 'treatment', 'medical',
                       'health record', 'ehr', 'hipaa']
        if any(term in lower for term in health_terms):
            return ProblemDomain.HEALTHCARE
        
        return ProblemDomain.GENERIC
    
    def get_solution_history(self) -> List[SolverResult]:
        """Get history of all solutions"""
        return self.solution_history.copy()
    
    # ============================================================================
    # CAV-NLP ENHANCED PROBLEM SOLVING METHODS
    # ============================================================================
    
    async def solve_natural_language(
        self,
        problem: str,
        title: Optional[str] = None,
        domain: Union[ProblemDomain, str] = ProblemDomain.GENERIC,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        **kwargs
    ) -> SolverResult:
        """
        Solve a problem stated in natural language using CAV-NLP.
        
        This method uses the CAV-NLP (Computer-Assisted Verification with NLP)
        integration to:
        1. Formalize natural language problem descriptions into structured constraints
        2. Solve the formalized problem using the standard decomposition/recomposition pipeline
        3. Optionally verify the solution with formal methods
        
        Args:
            problem: Natural language description of the problem
            title: Optional title for the problem
            domain: Problem domain (or auto-detect if not specified)
            constraints: List of constraints
            success_criteria: List of success criteria
            **kwargs: Additional arguments passed to solve()
            
        Returns:
            SolverResult with complete solution and CAV-NLP metadata
            
        Raises:
            ValueError: If CAV-NLP is not available
            
        Example:
            >>> solver = UniversalProblemSolver()
            >>> result = await solver.solve_natural_language(
            ...     "Build a trading system that handles 1000 orders per second "
            ...     "with sub-millisecond latency and MiFID II compliance",
            ...     title="High-Frequency Trading System",
            ...     domain=ProblemDomain.FINANCE
            ... )
        """
        if not CAV_NLP_AVAILABLE:
            raise ValueError(
                "CAV-NLP required for natural language problem solving. "
                "Install the openevolve CAV-NLP integration package."
            )
        
        self.logger.info(f"Starting CAV-NLP natural language solving: {title or problem[:50]}...")
        
        # Step 1: Formalize the natural language problem
        step_formalize = SolutionStep("cav_nlp_formalization", datetime.now())
        
        try:
            service = UnifiedMathService()
            formalized = await service.formalize(problem)
            
            step_formalize.end_time = datetime.now()
            step_formalize.status = "completed"
            step_formalize.details = {
                'formalization_success': True,
                'code_length': len(formalized.code) if hasattr(formalized, 'code') else 0,
                'constraints_extracted': len(formalized.constraints) if hasattr(formalized, 'constraints') else 0
            }
            self.logger.info("CAV-NLP formalization completed successfully")
        except Exception as e:
            step_formalize.end_time = datetime.now()
            step_formalize.status = "failed"
            step_formalize.details = {'error': str(e)}
            self.logger.error(f"CAV-NLP formalization failed: {e}")
            raise ValueError(f"Failed to formalize problem: {e}")
        
        # Step 2: Solve the formalized problem using standard pipeline
        formalized_code = formalized.code if hasattr(formalized, 'code') else str(formalized)
        
        # Merge formalized constraints with user-provided constraints
        merged_constraints = list(constraints) if constraints else []
        if hasattr(formalized, 'constraints') and formalized.constraints:
            merged_constraints.extend(formalized.constraints)
        
        # Extract formalized success criteria if available
        merged_success_criteria = list(success_criteria) if success_criteria else []
        if hasattr(formalized, 'success_criteria') and formalized.success_criteria:
            merged_success_criteria.extend(formalized.success_criteria)
        
        result = self.solve(
            problem_statement=formalized_code,
            title=title or (formalized.title if hasattr(formalized, 'title') else None),
            domain=domain,
            constraints=merged_constraints,
            success_criteria=merged_success_criteria,
            **kwargs
        )
        
        # Add CAV-NLP metadata to result
        result.solving_steps.insert(0, step_formalize)
        if hasattr(result, 'metadata') and isinstance(result.metadata, dict):
            result.metadata['cav_nlp'] = {
                'original_problem': problem,
                'formalized_code': formalized_code,
                'formalization_metadata': getattr(formalized, 'metadata', {})
            }
        
        return result
    
    async def solve_hybrid(
        self,
        problem: str,
        title: Optional[str] = None,
        domain: Union[ProblemDomain, str] = ProblemDomain.GENERIC,
        constraints: Optional[List[str]] = None,
        success_criteria: Optional[List[str]] = None,
        verify_with_lean: bool = True,
        **kwargs
    ) -> SolverResult:
        """
        Solve with hybrid Z3 + Lean verification.
        
        This method combines the standard decomposition/recomposition pipeline
        with formal verification using Z3 and Lean theorem prover. It:
        1. Solves the problem using the standard pipeline
        2. Extracts constraints from the solution
        3. Verifies the solution using Z3 SMT solver
        4. Optionally verifies with Lean 4 theorem prover (if available)
        
        Args:
            problem: Problem statement (can be natural language or formal)
            title: Optional title for the problem
            domain: Problem domain (or auto-detect if not specified)
            constraints: List of constraints
            success_criteria: List of success criteria
            verify_with_lean: Whether to use Lean verification (if available)
            **kwargs: Additional arguments passed to solve()
            
        Returns:
            SolverResult with complete solution and verification results
            
        Example:
            >>> solver = UniversalProblemSolver()
            >>> result = await solver.solve_hybrid(
            ...     "Design a distributed consensus protocol with safety guarantees",
            ...     title="Consensus Protocol Design",
            ...     verify_with_lean=True
            ... )
            >>> print(f"Z3 verified: {result.verification.get('z3_verified', False)}")
            >>> print(f"Lean verified: {result.verification.get('lean_verified', False)}")
        """
        self.logger.info(f"Starting hybrid Z3+Lean solving: {title or problem[:50]}...")
        
        # Step 1: Solve using standard pipeline
        solution = self.solve(
            problem_statement=problem,
            title=title,
            domain=domain,
            constraints=constraints,
            success_criteria=success_criteria,
            **kwargs
        )
        
        verification_results: Dict[str, Any] = {
            'z3_verified': False,
            'lean_verified': False,
            'verification_performed': False,
            'errors': []
        }
        
        # Step 2: Apply CAV-NLP verification if available
        if CAV_NLP_AVAILABLE and solution.success:
            step_verify = SolutionStep("cav_nlp_verification", datetime.now())
            
            try:
                solver = EnhancedZ3Solver()
                
                # Extract constraints from solution
                solution_constraints = []
                if hasattr(solution, 'constraints') and solution.constraints:
                    solution_constraints.extend(solution.constraints)
                if hasattr(solution.final_solution, 'constraints'):
                    solution_constraints.extend(solution.final_solution.constraints)
                
                # Perform Z3 verification
                z3_result = await solver.verify(solution_constraints)
                verification_results['z3_verified'] = getattr(z3_result, 'verified', False)
                verification_results['z3_result'] = z3_result
                
                # Optionally verify with Lean
                if verify_with_lean:
                    try:
                        lean_result = await solver.verify_with_lean(solution_constraints)
                        verification_results['lean_verified'] = getattr(lean_result, 'verified', False)
                        verification_results['lean_result'] = lean_result
                    except Exception as lean_error:
                        verification_results['lean_error'] = str(lean_error)
                        self.logger.warning(f"Lean verification failed: {lean_error}")
                
                verification_results['verification_performed'] = True
                step_verify.end_time = datetime.now()
                step_verify.status = "completed"
                step_verify.details = {
                    'z3_verified': verification_results['z3_verified'],
                    'lean_verified': verification_results['lean_verified']
                }
                self.logger.info(
                    f"CAV-NLP verification completed: Z3={verification_results['z3_verified']}, "
                    f"Lean={verification_results['lean_verified']}"
                )
                
            except Exception as e:
                verification_results['errors'].append(str(e))
                step_verify.end_time = datetime.now()
                step_verify.status = "failed"
                step_verify.details = {'error': str(e)}
                self.logger.error(f"CAV-NLP verification failed: {e}")
            
            solution.solving_steps.append(step_verify)
        elif not CAV_NLP_AVAILABLE:
            self.logger.warning("CAV-NLP not available, skipping verification")
            verification_results['errors'].append("CAV-NLP not available")
        
        # Attach verification results to solution
        solution.verification = verification_results
        if hasattr(solution, 'metadata') and isinstance(solution.metadata, dict):
            solution.metadata['hybrid_verification'] = verification_results
        
        return solution
    
    def is_cav_nlp_available(self) -> bool:
        """
        Check if CAV-NLP integration is available.
        
        Returns:
            True if CAV-NLP can be used, False otherwise
        """
        return CAV_NLP_AVAILABLE


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'SolutionStep',
    'GauntletOutcome',
    'GauntletBundle',
    'SolverResult',
    
    # Solvers
    'SubProblemSolver',
    'UniversalProblemSolver',
]


# ============================================================================
# MAIN EXECUTION (EXAMPLES)
# ============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("Universal Problem Solver - End-to-End Examples")
    print("=" * 80)
    
    # Initialize solver
    solver = UniversalProblemSolver()
    
    # Example 1: Software Problem
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Software Engineering - Authentication Microservice")
    print("=" * 80)
    
    software_problem = """
    Build a microservice-based authentication system with the following requirements:
    - OAuth2 and JWT token support
    - Role-based access control (RBAC)
    - Integration with LDAP for enterprise users
    - Rate limiting to prevent brute force attacks
    - Comprehensive audit logging
    - Support for 10,000 concurrent users
    - Response time under 100ms
    """
    
    result = solver.solve(
        problem_statement=software_problem,
        title="Authentication Microservice",
        domain=ProblemDomain.SOFTWARE,
        constraints=["OAuth2 support", "LDAP integration", "sub-100ms response"],
        success_criteria=["10K concurrent users", "99.9% uptime"]
    )
    
    print("\n" + result.summary())
    print("\n" + "-" * 80)
    print("SAMPLE OF FINAL SOLUTION:")
    print("-" * 80)
    print(result.final_solution.assembled_content[:1000])
    print("...")
    
    # Example 2: Finance Problem
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Finance - Trading Risk Management System")
    print("=" * 80)
    
    finance_problem = """
    Implement a real-time trading risk management system that:
    - Monitors position limits and exposure across all asset classes
    - Calculates Value at Risk (VaR) using Monte Carlo simulation
    - Generates MiFID II compliant regulatory reports
    - Provides real-time alerts when risk thresholds are breached
    - Maintains sub-millisecond latency for high-frequency trading
    - Includes comprehensive audit trails for compliance
    """
    
    result = solver.solve(
        problem_statement=finance_problem,
        title="Trading Risk Management System",
        domain=ProblemDomain.FINANCE,
        constraints=["MiFID II compliance", "sub-millisecond latency", "real-time processing"],
        success_criteria=["VaR accuracy > 99%", "99.99% uptime"]
    )
    
    print("\n" + result.summary())
    
    # Example 3: Scientific Problem
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Scientific - Genomic Analysis Pipeline")
    print("=" * 80)
    
    science_problem = """
    Develop a machine learning pipeline for genomic sequence analysis:
    - Process large-scale genomic datasets (100GB+)
    - Implement multiple classification algorithms (Random Forest, SVM, Neural Networks)
    - Perform cross-validation and hyperparameter tuning
    - Generate interpretable reports for biologists
    - Ensure reproducibility with version control
    - Maintain HIPAA compliance for patient data
    """
    
    result = solver.solve(
        problem_statement=science_problem,
        title="Genomic ML Pipeline",
        domain=ProblemDomain.SCIENTIFIC,
        constraints=["HIPAA compliance", "reproducibility", "100GB+ data"],
        success_criteria=["classification accuracy > 90%", "cross-validation support"]
    )
    
    print("\n" + result.summary())
    
    # Example 4: Auto-detect Domain
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Auto-Domain Detection")
    print("=" * 80)
    
    auto_problem = """
    Create a portfolio optimization system that uses modern portfolio theory
    to balance risk and return across equity and fixed income assets.
    Must handle real-time market data and rebalance positions automatically.
    """
    
    result = solver.solve(
        problem_statement=auto_problem,
        title="Portfolio Optimization System"
        # No domain specified - will auto-detect
    )
    
    print("\n" + result.summary())
    print(f"\nAuto-detected domain: {result.domain}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED")
    print("=" * 80)
    print(f"\nTotal problems solved: {len(solver.get_solution_history())}")
    print("\nThe Universal Problem Solver demonstrates:")
    print("  - Domain-agnostic decomposition")
    print("  - Automatic domain detection")
    print("  - Sub-problem generation and solving")
    print("  - Solution assembly with conflict detection")
    print("  - Quality scoring and validation")
    print("  - Support for Software, Finance, Scientific, and other domains")
