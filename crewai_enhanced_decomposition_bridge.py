"""
CrewAI Bridge for Enhanced Decomposition/Recomposition

This module integrates the enhanced decomposition/recomposition systems with CrewAI,
enabling agent-based delegation of sub-problems and collaborative solution assembly.

Features:
- Automatic agent assignment based on sub-problem type
- Crew formation for parallel sub-problem solving
- Agent coordination during recomposition
- Result aggregation and quality validation
"""

from __future__ import annotations

import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

# Import enhanced systems
from enhanced_decomposition_engine import (
    EnhancedDecompositionEngine,
    ProblemDefinition,
    DecompositionPlan,
    SubProblem,
    SubProblemType,
    ProblemDomain,
    create_problem_definition
)

from enhanced_recomposition_engine import (
    EnhancedRecompositionEngine,
    SubProblemSolution,
    IntegratedSolution,
    AssemblyStrategy
)

from openevolve_enhanced_decomposition_integration import (
    OpenEvolveIntegratedPipeline,
    OpenEvolveSolutionSolver,
    EvolutionConfig
)

# Configure logging
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class AgentProfile:
    """Profile for a CrewAI agent."""
    agent_id: str
    name: str
    role: str
    expertise: List[str]
    capabilities: List[str]
    max_concurrent_tasks: int = 3
    
    def can_handle(self, sub_problem: SubProblem) -> Tuple[bool, float]:
        """Check if agent can handle sub-problem."""
        score = 0.0
        
        # Check expertise match
        problem_type = sub_problem.type.value
        if any(problem_type in cap for cap in self.capabilities):
            score += 0.5
        
        # Check role match
        type_role_map = {
            'implementation': 'developer',
            'design': 'architect',
            'research': 'researcher',
            'analysis': 'analyst',
            'testing': 'qa_engineer'
        }
        
        expected_role = type_role_map.get(problem_type, 'generalist')
        if expected_role in self.role.lower():
            score += 0.3
        
        # Check complexity capability
        if sub_problem.complexity_score.overall_complexity < 7:
            score += 0.2
        
        return score > 0.5, score


@dataclass
class CrewAssignment:
    """Assignment of sub-problem to crew/agent."""
    assignment_id: str
    sub_problem_id: str
    agent_id: str
    status: str  # pending, in_progress, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[SubProblemSolution] = None
    error: Optional[str] = None


# ============================================================================
# AGENT POOL
# ============================================================================

class AgentPool:
    """Pool of available CrewAI agents."""
    
    def __init__(self):
        self.agents: Dict[str, AgentProfile] = {}
        self.assignments: Dict[str, CrewAssignment] = {}
        self.logger = logging.getLogger(__name__)
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent profiles."""
        default_agents = [
            AgentProfile(
                agent_id="agent_dev_1",
                name="Senior Developer",
                role="senior_developer",
                expertise=["python", "javascript", "system_design"],
                capabilities=["implementation", "coding", "development"]
            ),
            AgentProfile(
                agent_id="agent_arch_1",
                name="System Architect",
                role="architect",
                expertise=["architecture", "microservices", "cloud"],
                capabilities=["design", "architecture", "planning"]
            ),
            AgentProfile(
                agent_id="agent_researcher_1",
                name="Technical Researcher",
                role="researcher",
                expertise=[["research", "analysis", "documentation"]],
                capabilities=["research", "investigation", "exploration"]
            ),
            AgentProfile(
                agent_id="agent_qa_1",
                name="QA Engineer",
                role="qa_engineer",
                expertise=["testing", "quality_assurance", "automation"],
                capabilities=["testing", "validation", "verification"]
            ),
            AgentProfile(
                agent_id="agent_devops_1",
                name="DevOps Engineer",
                role="devops_engineer",
                expertise=["deployment", "ci_cd", "infrastructure"],
                capabilities=["deployment", "integration", "configuration"]
            ),
        ]
        
        for agent in default_agents:
            self.agents[agent.agent_id] = agent
    
    def register_agent(self, agent: AgentProfile):
        """Register a new agent."""
        self.agents[agent.agent_id] = agent
        self.logger.info(f"Registered agent: {agent.name} ({agent.agent_id})")
    
    def find_best_agent(self, sub_problem: SubProblem) -> Optional[AgentProfile]:
        """Find best agent for sub-problem."""
        best_agent = None
        best_score = 0.0
        
        for agent in self.agents.values():
            can_handle, score = agent.can_handle(sub_problem)
            if can_handle and score > best_score:
                best_score = score
                best_agent = agent
        
        return best_agent
    
    def get_agent_assignments(self, agent_id: str) -> List[CrewAssignment]:
        """Get all assignments for an agent."""
        return [
            a for a in self.assignments.values()
            if a.agent_id == agent_id
        ]
    
    def create_assignment(
        self,
        sub_problem: SubProblem,
        agent_id: str
    ) -> CrewAssignment:
        """Create new assignment."""
        assignment = CrewAssignment(
            assignment_id=f"assign_{uuid.uuid4().hex[:8]}",
            sub_problem_id=sub_problem.id,
            agent_id=agent_id,
            status="pending"
        )
        
        self.assignments[assignment.assignment_id] = assignment
        return assignment


# ============================================================================
# CREWAI DECOMPOSITION BRIDGE
# ============================================================================

class CrewAIDecompositionBridge:
    """
    Bridge between enhanced decomposition and CrewAI.
    
    This bridge enables:
    - Automatic agent assignment for sub-problems
    - Crew formation for parallel execution
    - Agent coordination during solution assembly
    """
    
    def __init__(
        self,
        decomposition_engine: Optional[EnhancedDecompositionEngine] = None,
        agent_pool: Optional[AgentPool] = None
    ):
        self.decomposition_engine = decomposition_engine or EnhancedDecompositionEngine()
        self.agent_pool = agent_pool or AgentPool()
        self.solver = OpenEvolveSolutionSolver()
        self.logger = logging.getLogger(__name__)
    
    def decompose_and_assign(
        self,
        problem: ProblemDefinition,
        strategy: Optional[str] = None
    ) -> Tuple[DecompositionPlan, Dict[str, str]]:
        """
        Decompose problem and assign agents to sub-problems.
        
        Args:
            problem: Problem to decompose
            strategy: Decomposition strategy
            
        Returns:
            Tuple of (decomposition_plan, sub_problem_to_agent_mapping)
        """
        # Decompose problem
        from enhanced_decomposition_engine import DecompositionStrategy
        
        strategy_enum = None
        if strategy:
            try:
                strategy_enum = DecompositionStrategy(strategy)
            except ValueError:
                pass
        
        plan = self.decomposition_engine.decompose(
            problem,
            strategy=strategy_enum
        )
        
        # Assign agents to sub-problems
        assignments = {}
        for sub_problem in plan.sub_problems:
            agent = self.agent_pool.find_best_agent(sub_problem)
            if agent:
                assignments[sub_problem.id] = agent.agent_id
                self.agent_pool.create_assignment(sub_problem, agent.agent_id)
                self.logger.info(
                    f"Assigned {agent.name} to {sub_problem.title}"
                )
            else:
                self.logger.warning(
                    f"No suitable agent found for {sub_problem.title}"
                )
        
        return plan, assignments
    
    def execute_with_crew(
        self,
        problem: ProblemDefinition,
        use_evolution: bool = True
    ) -> Dict[str, Any]:
        """
        Execute full pipeline with CrewAI agent delegation.
        
        Args:
            problem: Problem to solve
            use_evolution: Whether to use OpenEvolve evolution
            
        Returns:
            Execution results
        """
        start_time = datetime.now()
        
        # Decompose and assign
        plan, assignments = self.decompose_and_assign(problem)
        
        # Create crew formations based on dependencies
        crew_formations = self._create_crew_formations(
            plan,
            assignments
        )
        
        # Execute sub-problems with crews
        solutions = {}
        
        if use_evolution:
            # Use OpenEvolve for solution generation
            pipeline = OpenEvolveIntegratedPipeline()
            
            for formation in crew_formations:
                formation_solutions = self._execute_formation(
                    formation,
                    pipeline
                )
                solutions.update(formation_solutions)
        else:
            # Use basic solver
            for sub_problem in plan.sub_problems:
                solution = self.solver.solve(sub_problem)
                solutions[sub_problem.id] = solution
        
        # Assemble solutions
        recomposition_engine = EnhancedRecompositionEngine()
        integrated = recomposition_engine.assemble(
            sub_solutions=solutions,
            problem_id=problem.id,
            decomposition_plan_id=plan.id,
            dependency_graph=plan.dependency_graph
        )
        
        end_time = datetime.now()
        
        return {
            'success': True,
            'problem_id': problem.id,
            'decomposition_plan': plan,
            'assignments': assignments,
            'crew_formations': len(crew_formations),
            'solutions': solutions,
            'integrated_solution': integrated,
            'execution_time': (end_time - start_time).total_seconds(),
            'quality': integrated.quality_metrics.overall_score if integrated.quality_metrics else 0
        }
    
    def _create_crew_formations(
        self,
        plan: DecompositionPlan,
        assignments: Dict[str, str]
    ) -> List[List[SubProblem]]:
        """Create crew formations based on parallel groups."""
        formations = []
        
        # Use parallel groups from decomposition plan
        for group in plan.parallel_groups:
            formation = [
                sp for sp in plan.sub_problems
                if sp.id in group and sp.id in assignments
            ]
            if formation:
                formations.append(formation)
        
        return formations
    
    def _execute_formation(
        self,
        formation: List[SubProblem],
        pipeline: OpenEvolveIntegratedPipeline
    ) -> Dict[str, SubProblemSolution]:
        """Execute a crew formation."""
        solutions = {}
        
        for sub_problem in formation:
            # Create mini-problem for evolution
            mini_problem = create_problem_definition(
                title=sub_problem.title,
                description=sub_problem.description,
                complexity=sub_problem.complexity_score.overall_complexity
            )
            
            # Evolve solution
            result = pipeline.execute(mini_problem)
            
            if result.sub_solutions:
                # Use first solution
                solution = list(result.sub_solutions.values())[0]
                solutions[sub_problem.id] = solution
            else:
                # Fallback
                solutions[sub_problem.id] = self.solver.solve(sub_problem)
        
        return solutions


# ============================================================================
# CREWAI RECOMPOSITION COORDINATOR
# ============================================================================

class CrewAIRecompositionCoordinator:
    """
    Coordinates agent collaboration during recomposition.
    
    Manages:
    - Conflict resolution between agent solutions
    - Solution quality validation
    - Final assembly approval
    """
    
    def __init__(self, recomposition_engine: Optional[EnhancedRecompositionEngine] = None):
        self.recomposition_engine = recomposition_engine or EnhancedRecompositionEngine()
        self.logger = logging.getLogger(__name__)
    
    def coordinate_assembly(
        self,
        sub_solutions: Dict[str, SubProblemSolution],
        plan: DecompositionPlan,
        approval_agents: Optional[List[str]] = None
    ) -> IntegratedSolution:
        """
        Coordinate solution assembly with agent approval.
        
        Args:
            sub_solutions: Sub-problem solutions
            plan: Decomposition plan
            approval_agents: Agents required to approve assembly
            
        Returns:
            Integrated solution
        """
        # Perform assembly
        solution = self.recomposition_engine.assemble(
            sub_solutions=sub_solutions,
            problem_id=plan.original_problem.id,
            decomposition_plan_id=plan.id,
            dependency_graph=plan.dependency_graph
        )
        
        # Check if approval needed
        if approval_agents and solution.quality_metrics:
            if solution.quality_metrics.overall_score < 0.8:
                self.logger.info("Solution quality below threshold, requesting agent review")
                # In real implementation, would notify approval agents
        
        return solution
    
    def resolve_conflicts_with_agents(
        self,
        conflicts: List[Any],
        agent_votes: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Resolve conflicts using agent voting/consensus.
        
        Args:
            conflicts: List of conflicts to resolve
            agent_votes: Agent votes on resolution strategies
            
        Returns:
            Resolution decisions
        """
        resolutions = {}
        
        for conflict in conflicts:
            # Count votes
            vote_counts = {}
            for agent_id, vote in agent_votes.items():
                if conflict.conflict_id in vote:
                    strategy = vote[conflict.conflict_id]
                    vote_counts[strategy] = vote_counts.get(strategy, 0) + 1
            
            # Select winning strategy
            if vote_counts:
                winning_strategy = max(vote_counts, key=vote_counts.get)
                resolutions[conflict.conflict_id] = winning_strategy
            else:
                # Default resolution
                resolutions[conflict.conflict_id] = "manual_review"
        
        return resolutions


# ============================================================================
# UNIFIED CREWAI FACADE
# ============================================================================

class CrewAIDecompositionFacade:
    """
    Unified facade for CrewAI + Enhanced Decomposition integration.
    
    Provides a simple interface for the complete workflow:
    1. Problem decomposition
    2. Agent assignment
    3. Parallel execution
    4. Solution assembly
    """
    
    def __init__(self):
        self.bridge = CrewAIDecompositionBridge()
        self.coordinator = CrewAIRecompositionCoordinator()
        self.logger = logging.getLogger(__name__)
    
    def solve(
        self,
        title: str,
        description: str,
        domain: str = "software",
        complexity: Optional[float] = None,
        strategy: Optional[str] = None,
        use_crewai: bool = True,
        use_evolution: bool = True
    ) -> Dict[str, Any]:
        """
        Solve problem using CrewAI agents and enhanced decomposition.
        
        Args:
            title: Problem title
            description: Problem description
            domain: Problem domain
            complexity: Complexity estimate
            strategy: Decomposition strategy
            use_crewai: Whether to use CrewAI agents
            use_evolution: Whether to use OpenEvolve evolution
            
        Returns:
            Solution results
        """
        # Create problem
        domain_enum = ProblemDomain(domain) if domain in [d.value for d in ProblemDomain] else ProblemDomain.GENERIC
        
        problem = create_problem_definition(
            title=title,
            description=description,
            domain=domain_enum,
            complexity=complexity
        )
        
        self.logger.info(f"Solving problem: {title}")
        
        if use_crewai:
            # Use CrewAI delegation
            result = self.bridge.execute_with_crew(
                problem,
                use_evolution=use_evolution
            )
        else:
            # Use pipeline directly
            pipeline = OpenEvolveIntegratedPipeline()
            pipeline_result = pipeline.execute(problem)
            
            result = {
                'success': pipeline_result.is_successful(),
                'problem_id': problem.id,
                'decomposition_plan': pipeline_result.decomposition_plan,
                'solutions': pipeline_result.sub_solutions,
                'integrated_solution': pipeline_result.integrated_solution,
                'quality': pipeline_result.overall_quality
            }
        
        return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CrewAI Enhanced Decomposition Bridge Demo")
    print("=" * 70)
    
    # Create facade
    facade = CrewAIDecompositionFacade()
    
    # Define problem
    result = facade.solve(
        title="Build API Gateway",
        description="""
        Design and implement an API gateway system with:
        - Request routing and load balancing
        - Authentication and authorization
        - Rate limiting and throttling
        - Request/response transformation
        - Caching layer
        - Monitoring and logging
        """,
        domain="software",
        complexity=7.5,
        strategy="hybrid",
        use_crewai=True,
        use_evolution=True
    )
    
    print("\nResults:")
    print("-" * 70)
    print(f"Success: {result['success']}")
    print(f"Quality: {result.get('quality', 0):.2f}")
    
    if 'assignments' in result:
        print(f"\nAgent Assignments:")
        for sp_id, agent_id in result['assignments'].items():
            print(f"  {sp_id} -> {agent_id}")
    
    if 'integrated_solution' in result and result['integrated_solution']:
        sol = result['integrated_solution']
        print(f"\nIntegrated Solution:")
        print(f"  Conflicts: {len(sol.conflicts_detected)} detected")
        print(f"  Resolved: {len(sol.conflicts_resolved)} resolved")
        if sol.quality_metrics:
            print(f"  Quality: {sol.quality_metrics.overall_score:.2f}")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
