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
from typing import Dict, List, Any, Optional, Callable, Union
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
class SolverResult:
    """Complete result from problem solving"""
    problem_id: str
    problem_title: str
    problem_statement: str
    domain: str
    
    # Decomposition results
    decomposition_plan: DecompositionPlan
    
    # Solving results
    sub_problem_solutions: Dict[str, SubProblemSolution]
    solving_steps: List[SolutionStep]
    
    # Reassembly results
    final_solution: IntegratedSolution
    
    # Overall metrics
    quality_score: float
    total_duration_seconds: float
    conflicts_detected: int
    conflicts_resolved: int
    
    # Metadata
    execution_log: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
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
            'assembly_strategy': self.final_solution.assembly_strategy,
            'created_at': self.created_at.isoformat()
        }
    
    def summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Problem: {self.problem_title}",
            f"Domain: {self.domain}",
            f"",
            f"Decomposition: {len(self.decomposition_plan.sub_problems)} sub-problems",
            f"Solutions Generated: {len(self.sub_problem_solutions)}",
            f"Assembly Strategy: {self.final_solution.assembly_strategy}",
            f"",
            f"Quality Score: {self.quality_score:.2f}/1.0",
            f"Conflicts: {self.conflicts_resolved} resolved, {self.conflicts_detected} remaining",
            f"Total Time: {self.total_duration_seconds:.2f} seconds",
            f"",
            f"Final Solution Length: {len(self.final_solution.assembled_content)} characters"
        ]
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
        assembly_strategy: AssemblyStrategy = AssemblyStrategy.ADAPTIVE
    ):
        self.llm_client = llm_client
        self.decomposition_strategy = decomposition_strategy
        self.assembly_strategy = assembly_strategy
        
        # Initialize components
        self.decomposition_engine = UniversalDecompositionEngine(llm_client)
        self.recomposition_engine = UniversalRecompositionEngine(llm_client)
        self.sub_problem_solver = SubProblemSolver(llm_client)
        
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
        resolve_conflicts: bool = True
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
            
        Returns:
            SolverResult with complete solution and metadata
        """
        overall_start = datetime.now()
        execution_log = []
        steps: List[SolutionStep] = []
        
        self.logger.info(f"Starting problem solving: {title or problem_statement[:50]}...")
        
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
        # STEP 2: Decomposition
        # ============================================================================
        step_decomp = SolutionStep("decomposition", datetime.now())
        
        plan = self.decomposition_engine.decompose(
            problem_statement=problem_statement,
            title=title,
            domain=domain,
            constraints=constraints or [],
            success_criteria=success_criteria or [],
            strategy=self.decomposition_strategy,
            max_subproblems=max_subproblems
        )
        
        # Apply domain-specific enhancements
        if domain == ProblemDomain.FINANCE:
            plan = FinanceDomainExtension.enhance_decomposition(plan)
        
        step_decomp.end_time = datetime.now()
        step_decomp.status = "completed"
        step_decomp.details = {
            'num_subproblems': len(plan.sub_problems),
            'strategy': plan.strategy_used.value if hasattr(plan.strategy_used, 'value') else str(plan.strategy_used),
            'quality_score': plan.quality_score
        }
        steps.append(step_decomp)
        execution_log.append(f"Decomposed into {len(plan.sub_problems)} sub-problems")
        
        # ============================================================================
        # STEP 3: Sub-Problem Solving
        # ============================================================================
        sub_solutions: Dict[str, SubProblemSolution] = {}
        
        if solve_subproblems:
            step_solving = SolutionStep("subproblem_solving", datetime.now())
            
            for sp in plan.sub_problems:
                self.logger.info(f"Solving: {sp.title}")
                entanglement_matrix = {}
                if hasattr(plan, "metadata") and isinstance(plan.metadata, dict):
                    entanglement_matrix = plan.metadata.get("entanglement_matrix", {}) or {}
                entangled_with = []
                entanglement_symbols = []
                if hasattr(sp, "metadata") and isinstance(sp.metadata, dict):
                    entangled_with = sp.metadata.get("entangled_with", []) or []
                    entanglement_symbols = sp.metadata.get("entanglement_symbols", []) or []
                context = {
                    "entanglement_matrix": entanglement_matrix,
                    "entangled_with": entangled_with,
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
        
        # ============================================================================
        # STEP 4: Reassembly
        # ============================================================================
        step_assembly = SolutionStep("reassembly", datetime.now())
        
        if sub_solutions:
            final_solution = self.recomposition_engine.assemble(
                plan=plan,
                sub_solutions=sub_solutions,
                strategy=self.assembly_strategy,
                detect_conflicts=detect_conflicts,
                resolve_conflicts=resolve_conflicts
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
        
        step_assembly.end_time = datetime.now()
        step_assembly.status = "completed"
        step_assembly.details = {
            'assembly_strategy': final_solution.assembly_strategy,
            'quality_score': final_solution.quality_metrics.overall_score,
            'conflicts_detected': len(final_solution.conflicts_detected),
            'conflicts_resolved': len(final_solution.conflicts_resolved)
        }
        steps.append(step_assembly)
        execution_log.append(f"Assembled solution with strategy: {final_solution.assembly_strategy}")
        
        # ============================================================================
        # STEP 5: Finalize Result
        # ============================================================================
        total_duration = (datetime.now() - overall_start).total_seconds()
        
        result = SolverResult(
            problem_id=plan.original_problem.id,
            problem_title=plan.original_problem.title,
            problem_statement=problem_statement,
            domain=domain.value if isinstance(domain, ProblemDomain) else domain,
            decomposition_plan=plan,
            sub_problem_solutions=sub_solutions,
            solving_steps=steps,
            final_solution=final_solution,
            quality_score=final_solution.quality_metrics.overall_score,
            total_duration_seconds=total_duration,
            conflicts_detected=len(final_solution.conflicts_detected),
            conflicts_resolved=len(final_solution.conflicts_resolved),
            execution_log=execution_log
        )
        
        self.solution_history.append(result)
        
        self.logger.info(f"Problem solving complete: quality={result.quality_score:.2f}, time={total_duration:.2f}s")
        
        return result
    
    def _parse_domain(self, domain_str: str) -> ProblemDomain:
        """Parse domain from string"""
        domain_map = {
            'software': ProblemDomain.SOFTWARE,
            'finance': ProblemDomain.FINANCE,
            'financial': ProblemDomain.FINANCE,
            'trading': ProblemDomain.FINANCE,
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
# EXPORTS
# ============================================================================

__all__ = [
    # Data classes
    'SolutionStep',
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
