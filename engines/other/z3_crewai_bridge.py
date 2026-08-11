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
from web3_formal_evidence import (
    build_web3_formal_evidence,
    verify_web3_lean_proof_async,
)

# Configure logging
logger = logging.getLogger(__name__)

# Import Z3 integration
try:
    from z3prover_integration import (
        Z3SolverEngine, Z3TheoremProver, Z3Variable, Z3Constraint,
        Z3ConstraintType, Z3Config, Z3SolverResult, Z3ResultStatus,
        get_z3_solver_engine, get_z3_theorem_prover, translate_solidity_assignment_to_z3,
        verify_solidity_invariant_translation, solve_smart_contract_exploit_witness
    )
    Z3_AVAILABLE = True
    WEB3_FORMAL_AVAILABLE = (
        translate_solidity_assignment_to_z3 is not None
        and solve_smart_contract_exploit_witness is not None
    )
except ImportError:
    Z3_AVAILABLE = False
    WEB3_FORMAL_AVAILABLE = False
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    solve_smart_contract_exploit_witness = None
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

# Import CAV-NLP integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False


def _get_web3_formal_capabilities() -> Dict[str, bool]:
    """Return capability flags for Web3 formal audit operations."""
    return {
        "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
        "invariant_translation_verification": verify_solidity_invariant_translation is not None,
        "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
        "composite_exploit_verification": (
            translate_solidity_assignment_to_z3 is not None
            and solve_smart_contract_exploit_witness is not None
        ),
    }


def _get_web3_formal_tools() -> List[str]:
    """Return normalized Web3 formal tool identifiers exposed by CrewAI bridge."""
    caps = _get_web3_formal_capabilities()
    tools: List[str] = []
    if caps["solidity_invariant_translation"]:
        tools.append("z3_translate_solidity_invariant")
    if caps["symbolic_exploit_witness"]:
        tools.append("z3_solve_smart_contract_exploit_witness")
    if caps["composite_exploit_verification"]:
        tools.append("z3_web3_audit_exploit_verification")
    return sorted(set(tools))


def _is_web3_formal_available() -> bool:
    """Infer whether any Web3 formal capability is currently available."""
    caps = _get_web3_formal_capabilities()
    return bool(_get_web3_formal_tools()) or any(bool(v) for v in caps.values()) or bool(
        WEB3_FORMAL_AVAILABLE
    )


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
    WEB3_AUDITOR = "web3_auditor"
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
        try:
            # Log the task execution
            self.logger.info(f"Executing task: {task.description[:100]}...")
            
            # Create a crew based on the task requirements
            from crewai import Agent, Task, Crew
            import asyncio
            
            # Create agent with the role and tools
            agent = Agent(
                role=self.role.value,
                goal=f"Complete the assigned task: {task.description}",
                backstory=f"As an expert {self.role.value}, you have extensive experience in this domain.",
                verbose=True,
                allow_delegation=False,
            )
            
            # Create the task for the agent
            crew_task = Task(
                description=task.description,
                expected_output="A detailed response to the task with supporting evidence and reasoning.",
                agent=agent
            )
            
            # Create and run the crew
            crew = Crew(
                agents=[agent],
                tasks=[crew_task],
                verbose=2
            )
            
            # Execute the crew
            result = crew.kickoff()
            
            # Create and return the result
            agent_result = AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result=result,
                success=True,
                execution_time=0.0,  # Placeholder - would need timing mechanism
                timestamp=datetime.now()
            )
            
            # Add to history
            self.task_history.append(task)
            self.result_history.append(agent_result)
            
            return agent_result
            
        except Exception as e:
            self.logger.error(f"Task execution failed: {e}")
            return AgentResult(
                task_id=task.id,
                agent_id=self.agent_id,
                result=f"Execution failed: {str(e)}",
                success=False,
                execution_time=0.0,
                timestamp=datetime.now()
            )
    
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
    """Agent for translating between formats with CAV-NLP support."""
    
    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.TRANSLATOR, config)
        self.bridge = get_z3_leanaide_bridge_sync() if Z3_LEANAIDE_AVAILABLE else None
        # Initialize CAV-NLP components
        self.use_cav_nlp = config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE if config else CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            self.enhanced_solver = EnhancedZ3Solver()
            self.math_service = UnifiedMathService()
            logger.info("CAV-NLP enhancement enabled for Z3TranslatorAgent")
    
    def formalize_for_crewai(self, natural_language: str) -> str:
        """Formalize NL for CrewAI agents using CAV-NLP.
        
        Args:
            natural_language: Natural language problem statement
            
        Returns:
            Formalized code representation
        """
        if not self.use_cav_nlp:
            logger.warning("CAV-NLP not available, returning original text")
            return natural_language
        
        result = self.math_service.formalize(natural_language)
        return result.code
    
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


class Z3Web3AuditAgent(Z3BaseAgent):
    """Agent for smart-contract invariant translation and exploit witness checks."""

    def __init__(self, agent_id: str, config: Optional[Z3Config] = None):
        super().__init__(agent_id, AgentRole.WEB3_AUDITOR, config)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute Web3 audit task."""
        import time

        start = time.time()
        if not _is_web3_formal_available():
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=["Web3 formal tools unavailable"],
                execution_time=time.time() - start,
                agent_id=self.agent_id,
            )

        try:
            action = str(task.parameters.get("action", "full_audit")).strip().lower()
            translate_actions = {
                "translate",
                "translate_invariant",
                "full_audit",
                "audit_exploit_verification",
            }
            witness_actions = {
                "witness",
                "exploit_witness",
                "full_audit",
                "audit_exploit_verification",
            }
            supported_actions = translate_actions | witness_actions
            if action not in supported_actions:
                return AgentResult(
                    task_id=task.task_id,
                    success=False,
                    role=self.role,
                    errors=[f"Unsupported Web3 audit action: {action}"],
                    execution_time=time.time() - start,
                    agent_id=self.agent_id,
                )

            formal_capabilities = _get_web3_formal_capabilities()
            verify_translation = bool(task.parameters.get("verify_translation", True))
            required_capabilities: List[str] = []
            if action in translate_actions:
                required_capabilities.append("solidity_invariant_translation")
                if verify_translation:
                    required_capabilities.append("invariant_translation_verification")
            if action in witness_actions:
                required_capabilities.append("symbolic_exploit_witness")
            if action in {"full_audit", "audit_exploit_verification"}:
                required_capabilities.append("composite_exploit_verification")
            missing_capabilities = sorted(
                {cap for cap in required_capabilities if not formal_capabilities.get(cap, False)}
            )
            if missing_capabilities:
                return AgentResult(
                    task_id=task.task_id,
                    success=False,
                    role=self.role,
                    errors=[
                        "Missing Web3 formal capabilities: "
                        + ", ".join(missing_capabilities)
                    ],
                    execution_time=time.time() - start,
                    agent_id=self.agent_id,
                )

            payload: Dict[str, Any] = {}

            if action in translate_actions:
                statement = (
                    task.parameters.get("statement")
                    or task.problem
                    or "balance[msg.sender] -= amount;"
                )
                translation = translate_solidity_assignment_to_z3(
                    statement=statement,
                    non_negative_target=bool(task.parameters.get("non_negative_target", True)),
                    max_withdraw_expr=task.parameters.get("max_withdraw_expr"),
                )
                payload["translation"] = translation
                if verify_translation and verify_solidity_invariant_translation is not None:
                    payload["verification"] = verify_solidity_invariant_translation(
                        translation=translation,
                        assume_non_negative_amount=bool(
                            task.parameters.get("assume_non_negative_amount", True)
                        ),
                    )

            if action in witness_actions:
                witness = solve_smart_contract_exploit_witness(
                    additional_constraints=task.parameters.get("additional_constraints"),
                    timeout=float(task.parameters.get("timeout", task.timeout)),
                )
                payload["exploit_witness"] = witness

            success = True
            if "verification" in payload and isinstance(payload["verification"], dict):
                proven = payload["verification"].get("proven")
                if proven is False:
                    success = True
            if "exploit_witness" in payload and isinstance(payload["exploit_witness"], dict):
                success = success and bool(payload["exploit_witness"].get("satisfiable", True))

            if action in {"full_audit", "audit_exploit_verification"}:
                verification = payload.get("verification")
                witness = payload.get("exploit_witness", {})
                lean_proof_verification = await verify_web3_lean_proof_async(
                    payload.get("translation"),
                    use_real_lean=True,
                )
                verified_exploit = bool(witness.get("satisfiable", False))
                if isinstance(verification, dict):
                    verified_exploit = verified_exploit and bool(verification.get("proven", False))
                payload["lean_proof_verification"] = lean_proof_verification
                payload["formal_evidence"] = build_web3_formal_evidence(
                    verification,
                    witness,
                    lean_proof_verification,
                )
                payload["verified_exploit"] = verified_exploit

            return AgentResult(
                task_id=task.task_id,
                success=success,
                role=self.role,
                result_data=payload,
                execution_time=time.time() - start,
                confidence=0.9 if success else 0.6,
                agent_id=self.agent_id,
            )
        except Exception as exc:
            return AgentResult(
                task_id=task.task_id,
                success=False,
                role=self.role,
                errors=[str(exc)],
                execution_time=time.time() - start,
                agent_id=self.agent_id,
            )

    def get_capabilities(self) -> List[str]:
        return [
            "solidity_invariant_translation",
            "invariant_verification",
            "symbolic_exploit_witness",
            "smart_contract_audit",
            "composite_exploit_verification",
        ]


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

    def create_web3_audit_agent(self, agent_id: str) -> Z3Web3AuditAgent:
        """Create and register a Web3 audit agent."""
        agent = Z3Web3AuditAgent(agent_id)
        self.register_agent(agent)
        return agent

    def create_agent(self, agent_id: str, role: str) -> Z3BaseAgent:
        """Compatibility factory for role-based agent creation."""
        normalized_role = str(role or "").strip().lower()
        if normalized_role in {"solver", AgentRole.SOLVER.value}:
            return self.create_solver_agent(agent_id)
        if normalized_role in {"optimizer", AgentRole.OPTIMIZER.value}:
            return self.create_optimizer_agent(agent_id)
        if normalized_role in {"prover", AgentRole.PROVER.value}:
            return self.create_prover_agent(agent_id)
        if normalized_role in {"translator", AgentRole.TRANSLATOR.value}:
            return self.create_translator_agent(agent_id)
        if normalized_role in {"verifier", AgentRole.VERIFIER.value}:
            return self.create_verifier_agent(agent_id)
        if normalized_role in {
            "web3_auditor",
            "web3-auditor",
            "web3_audit",
            "web3_audit_agent",
            AgentRole.WEB3_AUDITOR.value,
        }:
            return self.create_web3_audit_agent(agent_id)
        raise ValueError(f"Unknown agent role: {role}")

    def get_status(self) -> Dict[str, Any]:
        """Return coordinator status including Web3 formal tool inventory."""
        role_counts: Dict[str, int] = {}
        for agent in self.agents.values():
            role_counts[agent.role.value] = role_counts.get(agent.role.value, 0) + 1
        formal_capabilities = _get_web3_formal_capabilities()
        web3_formal_tools = _get_web3_formal_tools()
        inferred_formal_available = bool(web3_formal_tools) or any(
            bool(v) for v in formal_capabilities.values()
        )
        return {
            "z3_available": Z3_AVAILABLE,
            "web3_formal_available": inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE),
            "web3_formal_verification_available": (
                inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE)
            ),
            "formal_capabilities": formal_capabilities,
            "web3_formal_tools": web3_formal_tools,
            "audit_exploit_verification_available": bool(
                formal_capabilities.get("composite_exploit_verification")
            ),
            "registered_agents": len(self.agents),
            "active_sessions": len(self.sessions),
            "role_counts": role_counts,
        }
    
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

        if self._is_web3_problem(problem) and any(
            agent.role == AgentRole.WEB3_AUDITOR for agent in self.agents.values()
        ):
            tasks.append(
                AgentTask(
                    f"{session_id}_web3_audit",
                    AgentRole.WEB3_AUDITOR,
                    problem,
                    parameters={"action": "full_audit"},
                )
            )
        
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

    @staticmethod
    def _is_web3_problem(problem: str) -> bool:
        """Heuristic detector for smart-contract audit prompts."""
        text = (problem or "").lower()
        keywords = [
            "web3", "defi", "smart contract", "solidity", "evm", "reentrancy",
            "flash loan", "oracle", "vault", "exploit", "bug bounty", "invariant",
        ]
        return any(keyword in text for keyword in keywords)
    
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


def get_web3_formal_status() -> Dict[str, Any]:
    """Get normalized Web3 formal capability status for CrewAI bridge."""
    formal_capabilities = _get_web3_formal_capabilities()
    web3_formal_tools = _get_web3_formal_tools()
    inferred_formal_available = bool(web3_formal_tools) or any(
        bool(v) for v in formal_capabilities.values()
    )
    return {
        "available": inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE),
        "web3_formal_available": inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE),
        "web3_formal_verification_available": (
            inferred_formal_available or bool(WEB3_FORMAL_AVAILABLE)
        ),
        "web3_formal_tools": web3_formal_tools,
        "formal_capabilities": formal_capabilities,
        "audit_exploit_verification_available": bool(
            formal_capabilities.get("composite_exploit_verification")
        ),
    }


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
