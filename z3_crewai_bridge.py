"""
Z3-CrewAI Bridge for Agent Workflows

Integrates Z3 capabilities with CrewAI agent framework, enabling:
- Z3-powered agents for constraint solving
- Multi-agent theorem proving workflows
- Agent-based optimization
- Collaborative proof construction
- Agent-mediated translation between Z3 and Lean

Agent Types:
- Z3SolverAgent: Handles constraint satisfaction
- Z3OptimizerAgent: Handles optimization problems
- Z3TheoremProverAgent: Handles theorem proving
- Z3TranslatorAgent: Handles translations
- Z3VerifierAgent: Cross-validates results

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3SolverResult, Z3ResultStatus,
        get_z3_solver_engine, get_z3_theorem_prover
    )
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    logger.warning("Z3 integration not available")

try:
    from z3prover_advanced import (
        Z3AdvancedSolver, OptimizationObjective, PortfolioResult,
        get_z3_advanced_solver
    )
    Z3_ADVANCED_AVAILABLE = True
except ImportError:
    Z3_ADVANCED_AVAILABLE = False

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, CombinedVerificationResult,
        VerificationStrategy, get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

class AgentRole(Enum):
    """Roles for Z3 agents."""
    SOLVER = "solver"
    OPTIMIZER = "optimizer"
    PROVER = "prover"
    TRANSLATOR = "translator"
    VERIFIER = "verifier"
    COORDINATOR = "coordinator"


@dataclass
class AgentTask:
    """Task for Z3 agent."""
    task_id: str
    role: AgentRole
    problem: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 1
    timeout: float = 60.0
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role.value,
            "problem": self.problem[:100] + "..." if len(self.problem) > 100 else self.problem,
            "priority": self.priority,
            "timeout": self.timeout,
            "dependencies": self.dependencies
        }


@dataclass
class AgentResult:
    """Result from Z3 agent."""
    task_id: str
    success: bool
    role: AgentRole
    result_data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0
    confidence: float = 0.0
    agent_id: str = ""
    subtasks: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "success": self.success,
            "role": self.role.value,
            "execution_time": self.execution_time,
            "confidence": self.confidence,
            "agent_id": self.agent_id,
            "errors": self.errors
        }


@dataclass
class CollaborationSession:
    """Session for multi-agent collaboration."""
    session_id: str
    problem: str
    agents_involved: List[str] = field(default_factory=list)
    tasks: List[AgentTask] = field(default_factory=list)
    results: List[AgentResult] = field(default_factory=list)
    consensus_reached: bool = False
    final_solution: Optional[Dict[str, Any]] = None
    created_at: float = field(default_factory=lambda: asyncio.get_event_loop().time() if asyncio.get_event_loop().is_running() else 0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "agents_count": len(self.agents_involved),
            "tasks_count": len(self.tasks),
            "results_count": len(self.results),
            "consensus_reached": self.consensus_reached,
            "final_solution": self.final_solution
        }


# =============================================================================
# Base Z3 Agent
# =============================================================================

class Z3BaseAgent:
    """Base class for Z3-powered agents."""
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        config: Optional[Z3Config] = None
    ):
        self.agent_id = agent_id
        self.role = role
        self.config = config or Z3Config()
        self.task_history: List[AgentTask] = []
        self.result_history: List[AgentResult] = []
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a task."""
        raise NotImplementedError
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities."""
        return [self.role.value]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "capabilities": self.get_capabilities(),
            "tasks_completed": len(self.task_history)
        }


# =============================================================================
# Specialized Z3 Agents
# =============================================================================

class Z3SolverAgent(Z3BaseAgent):
    """Agent for constraint solving."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.SOLVER, config)
        self.solver = get_z3_solver_engine(config) if Z3_AVAILABLE else None
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute constraint solving task."""
        import time
        start = time.time()
        
        if not self.solver:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Z3 solver not available"],
                execution_time=time.time() - start
            )
        
        try:
            # Parse problem
            problem = task.problem
            
            # Check if SMT-LIB
            if '(assert' in problem or '(declare' in problem:
                result = self.solver.solve_smtlib(problem)
            else:
                # Extract variables and constraints from natural language
                variables = task.parameters.get('variables', [])
                constraints = task.parameters.get('constraints', [])
                
                z3_vars = [
                    Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()])
                    for v in variables
                ]
                z3_constraints = [
                    Z3Constraint(c, Z3ConstraintType.INTEGER)
                    for c in constraints
                ]
                
                result = self.solver.solve_constraints(z3_vars, z3_constraints)
            
            execution_time = time.time() - start
            
            return AgentResult(
                task_id=task.task_id,
                success=result.is_sat(),
                role=self.role,
                result_data={
                    "status": result.status.value,
                    "model": result.model.assignments if result.model else None,
                    "satisfiable": result.is_sat()
                },
                execution_time=execution_time,
                confidence=0.95 if result.is_sat() else 0.5,
                agent_id=self.agent_id
            )
        
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(e)],
                execution_time=time.time() - start,
                agent_id=self.agent_id
            )
    
    def get_capabilities(self) -> List[str]:
        return ["constraint_solving", "smt_solving", "sat_solving"]


class Z3OptimizerAgent(Z3BaseAgent):
    """Agent for optimization problems."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.OPTIMIZER, config)
        self.solver = get_z3_advanced_solver(config) if Z3_ADVANCED_AVAILABLE else None
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute optimization task."""
        import time
        start = time.time()
        
        if not self.solver:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Z3 advanced solver not available"]
            )
        
        try:
            variables = [
                Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()])
                for v in task.parameters.get('variables', [])
            ]
            
            constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in task.parameters.get('constraints', [])
            ]
            
            objective = task.parameters.get('objective', {})
            obj_expr = objective.get('expression', 'x')
            obj_type = OptimizationObjective.MINIMIZE if objective.get('direction') == 'minimize' else OptimizationObjective.MAXIMIZE
            
            result = self.solver.optimize(variables, constraints, [(obj_expr, obj_type)])
            
            return AgentResult(
                task_id=task.task_id,
                success=result.success,
                role=self.role,
                result_data={
                    "optimal_value": result.optimal_value,
                    "model": result.optimal_model.assignments if result.optimal_model else None
                },
                execution_time=time.time() - start,
                confidence=0.9 if result.success else 0.0,
                agent_id=self.agent_id
            )
        
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(e)],
                execution_time=time.time() - start,
                agent_id=self.agent_id
            )
    
    def get_capabilities(self) -> List[str]:
        return ["optimization", "linear_programming", "integer_programming"]


class Z3TheoremProverAgent(Z3BaseAgent):
    """Agent for theorem proving."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.PROVER, config)
        self.prover = get_z3_theorem_prover(config) if Z3_AVAILABLE else None
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute theorem proving task."""
        import time
        start = time.time()
        
        if not self.prover:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Z3 prover not available"]
            )
        
        try:
            theorem = task.problem
            assumptions = task.parameters.get('assumptions', [])
            
            result = self.prover.prove_theorem(theorem, assumptions)
            
            return AgentResult(
                task_id=task.task_id,
                success=result.proven,
                role=self.role,
                result_data={
                    "proven": result.proven,
                    "tactic_used": result.tactic_used,
                    "counterexample": result.counterexample,
                    "proof": result.proof[:500] if result.proof else None
                },
                execution_time=time.time() - start,
                confidence=0.95 if result.proven else 0.3,
                agent_id=self.agent_id
            )
        
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(e)],
                execution_time=time.time() - start,
                agent_id=self.agent_id
            )
    
    def get_capabilities(self) -> List[str]:
        return ["theorem_proving", "smt_proving", "proof_generation"]


class Z3TranslatorAgent(Z3BaseAgent):
    """Agent for translating between formats."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.TRANSLATOR, config)
        self.bridge = get_z3_leanaide_bridge_sync() if Z3_LEANAIDE_AVAILABLE else None
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute translation task."""
        import time
        start = time.time()
        
        if not self.bridge:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Z3-LeanAIDE bridge not available"]
            )
        
        try:
            direction = task.parameters.get('direction', 'smt_to_lean')
            content = task.problem
            
            if direction == 'smt_to_lean':
                result = await self.bridge.translate_smt_to_lean(content)
            else:
                result = await self.bridge.translate_lean_to_smt(content)
            
            return AgentResult(
                task_id=task.task_id,
                success=result.success,
                role=self.role,
                result_data={
                    "translation": result.translation,
                    "metadata": result.metadata
                },
                execution_time=time.time() - start,
                confidence=0.8 if result.success else 0.0,
                agent_id=self.agent_id
            )
        
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(e)],
                execution_time=time.time() - start,
                agent_id=self.agent_id
            )
    
    def get_capabilities(self) -> List[str]:
        return ["smt_to_lean_translation", "lean_to_smt_translation"]


class Z3VerifierAgent(Z3BaseAgent):
    """Agent for cross-verification."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.VERIFIER, config)
        self.bridge = get_z3_leanaide_bridge_sync() if Z3_LEANAIDE_AVAILABLE else None
    
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute verification task."""
        import time
        start = time.time()
        
        if not self.bridge:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Z3-LeanAIDE bridge not available"]
            )
        
        try:
            problem = task.problem
            strategy_str = task.parameters.get('strategy', 'parallel')
            strategy = VerificationStrategy[strategy_str.upper()]
            
            entanglement_context = task.parameters.get("entanglement_context")
            if not entanglement_context:
                entanglement_context = {}
                for key in ("entanglement_constraints", "entangled_with", "entangled_constraints"):
                    if key in task.parameters:
                        entanglement_context[key] = task.parameters.get(key)
                if not entanglement_context:
                    entanglement_context = None

            result = await self.bridge.verify_with_both(
                problem,
                strategy,
                entanglement_context=entanglement_context
            )
            
            return AgentResult(
                task_id=task.task_id,
                success=result.success,
                role=self.role,
                result_data={
                    "agreement": result.agreement,
                    "confidence_score": result.confidence_score,
                    "z3_result": result.z3_result.status.value if result.z3_result else None,
                    "recommendation": result.recommendation
                },
                execution_time=time.time() - start,
                confidence=result.confidence_score,
                agent_id=self.agent_id
            )
        
        except Exception as e:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(e)],
                execution_time=time.time() - start,
                agent_id=self.agent_id
            )
    
    def get_capabilities(self) -> List[str]:
        return ["cross_verification", "z3_verification", "lean_verification"]


# =============================================================================
# Agent Coordinator
# =============================================================================

class Z3AgentCoordinator:
    """
    Coordinates multiple Z3 agents for complex tasks.
    
    Implements:
    - Task decomposition
    - Agent selection
    - Result aggregation
    - Consensus building
    """
    
    def __init__(self):
        self.agents: Dict[str, Z3BaseAgent] = {}
        self.sessions: Dict[str, CollaborationSession] = {}
        self._executor = ThreadPoolExecutor(max_workers=8)
    
    def register_agent(self, agent: Z3BaseAgent):
        """Register an agent."""
        self.agents[agent.agent_id] = agent
        logger.info(f"Registered agent: {agent.agent_id} ({agent.role.value})")
    
    def create_solver_agent(self, agent_id: str) -> Z3SolverAgent:
        """Create and register a solver agent."""
        agent = Z3SolverAgent(agent_id)
        self.register_agent(agent)
        return agent
    
    def create_optimizer_agent(self, agent_id: str) -> Z3OptimizerAgent:
        """Create and register an optimizer agent."""
        agent = Z3OptimizerAgent(agent_id)
        self.register_agent(agent)
        return agent
    
    def create_prover_agent(self, agent_id: str) -> Z3TheoremProverAgent:
        """Create and register a prover agent."""
        agent = Z3TheoremProverAgent(agent_id)
        self.register_agent(agent)
        return agent
    
    def create_translator_agent(self, agent_id: str) -> Z3TranslatorAgent:
        """Create and register a translator agent."""
        agent = Z3TranslatorAgent(agent_id)
        self.register_agent(agent)
        return agent
    
    def create_verifier_agent(self, agent_id: str) -> Z3VerifierAgent:
        """Create and register a verifier agent."""
        agent = Z3VerifierAgent(agent_id)
        self.register_agent(agent)
        return agent
    
    def select_agents_for_task(self, task: AgentTask) -> List[Z3BaseAgent]:
        """Select appropriate agents for a task."""
        selected = []
        
        for agent in self.agents.values():
            if agent.role == task.role:
                selected.append(agent)
        
        # Sort by capability match
        return selected
    
    async def execute_single(
        self,
        task: AgentTask,
        agent_id: Optional[str] = None
    ) -> AgentResult:
        """Execute task with a single agent."""
        if agent_id:
            agent = self.agents.get(agent_id)
            if not agent:
                return AgentResult(
                    task_id=task.task_id,
                    success=False,
                    role=task.role,
                    errors=[f"Agent not found: {agent_id}"]
                )
            return await agent.execute(task)
        
        # Auto-select agent
        candidates = self.select_agents_for_task(task)
        if not candidates:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=task.role,
                errors=[f"No agent available for role: {task.role.value}"]
            )
        
        return await candidates[0].execute(task)
    
    async def execute_collaborative(
        self,
        session_id: str,
        problem: str,
        strategy: str = "parallel"
    ) -> CollaborationSession:
        """
        Execute collaborative problem solving.
        
        Args:
            session_id: Unique session ID
            problem: Problem to solve
            strategy: "parallel", "sequential", "hierarchical"
        """
        session = CollaborationSession(
            session_id=session_id,
            problem=problem
        )
        self.sessions[session_id] = session
        
        # Create tasks for different approaches
        tasks = [
            AgentTask(f"{session_id}_solve", AgentRole.SOLVER, problem),
            AgentTask(f"{session_id}_prove", AgentRole.PROVER, problem),
        ]
        
        if Z3_LEANAIDE_AVAILABLE:
            tasks.append(AgentTask(
                f"{session_id}_verify",
                AgentRole.VERIFIER,
                problem,
                parameters={"strategy": "parallel"}
            ))
        
        session.tasks = tasks
        
        # Execute based on strategy
        if strategy == "parallel":
            results = await asyncio.gather(*[
                self.execute_single(task) for task in tasks
            ])
        else:
            results = []
            for task in tasks:
                result = await self.execute_single(task)
                results.append(result)
        
        session.results = results
        session.agents_involved = list(self.agents.keys())
        
        # Determine consensus
        successful = [r for r in results if r.success]
        session.consensus_reached = len(successful) >= len(results) / 2
        
        if successful:
            # Select best result
            best = max(successful, key=lambda r: r.confidence)
            session.final_solution = {
                "role": best.role.value,
                "confidence": best.confidence,
                "data": best.result_data
            }
        
        return session
    
    def get_session(self, session_id: str) -> Optional[CollaborationSession]:
        """Get collaboration session."""
        return self.sessions.get(session_id)


# =============================================================================
# Global Coordinator
# =============================================================================

_coordinator: Optional[Z3AgentCoordinator] = None


def get_z3_agent_coordinator() -> Z3AgentCoordinator:
    """Get global agent coordinator."""
    global _coordinator
    if _coordinator is None:
        _coordinator = Z3AgentCoordinator()
    return _coordinator


# =============================================================================
# Example Usage
# =============================================================================

async def example_single_agent():
    """Example: Single agent execution."""
    coordinator = get_z3_agent_coordinator()
    
    # Create and register agents
    solver = coordinator.create_solver_agent("solver_1")
    
    # Create task
    task = AgentTask(
        task_id="task_1",
        role=AgentRole.SOLVER,
        problem="""
        (set-logic LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (> x 0))
        (assert (< x 10))
        (assert (= y (+ x 5)))
        (check-sat)
        """
    )
    
    # Execute
    result = await coordinator.execute_single(task)
    
    print("Single Agent Result:")
    print(json.dumps(result.to_dict(), indent=2))
    
    return result


async def example_collaborative():
    """Example: Collaborative problem solving."""
    coordinator = get_z3_agent_coordinator()
    
    # Create agents
    coordinator.create_solver_agent("solver_1")
    coordinator.create_prover_agent("prover_1")
    
    if Z3_LEANAIDE_AVAILABLE:
        coordinator.create_verifier_agent("verifier_1")
    
    # Collaborative session
    session = await coordinator.execute_collaborative(
        session_id="collab_1",
        problem="""
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (< x 100))
        (check-sat)
        """,
        strategy="parallel"
    )
    
    print("\nCollaborative Session:")
    print(json.dumps(session.to_dict(), indent=2))
    
    for result in session.results:
        print(f"\n  {result.role.value}: {result.success} (confidence: {result.confidence:.2f})")
    
    return session


async def main():
    """Run examples."""
    print("Z3-CrewAI Bridge")
    print("=" * 50)
    
    await example_single_agent()
    print("\n" + "=" * 50)
    await example_collaborative()


if __name__ == "__main__":
    asyncio.run(main())
