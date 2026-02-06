"""
Z3-LeanAIDE BubbleLabs UI Integration

This module provides BubbleLabs UI components and workflow nodes for the
Z3-LeanAIDE-OpenEvolve integration, enabling:
- Visual workflow nodes for Z3 and LeanAIDE operations
- Real-time constraint solving visualization
- Theorem proving progress tracking
- Proof tree visualization
- Cross-verification results display

UI Components:
- Z3ConstraintSolverNode: Interactive constraint solving
- Z3TheoremProverNode: Theorem proving with step display
- LeanAIDEProverNode: Lean4 proof visualization
- CrossVerificationNode: Combined verification display
- ProofComparisonView: Side-by-side Z3/Lean results

Integration Points:
- bubblelabs_integration.py (workflow management)
- bubblelabs_leanaide_integration.py (existing LeanAIDE nodes)
- z3_leanaide_openevolve_integration.py (core logic)

Author: OpenEvolve
Created: 2026-01-31
"""


import asyncio
import json
import logging
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logger = logging.getLogger(__name__)

# Import core integrations
try:
    from z3prover_integration import (
        Z3SolverResult, Z3TheoremResult, Z3ResultStatus,
        Z3Variable, Z3Constraint, get_z3_solver_engine, get_z3_theorem_prover,
        translate_solidity_assignment_to_z3, verify_solidity_invariant_translation,
        solve_smart_contract_exploit_witness
    )
    Z3_AVAILABLE = True
    WEB3_FORMAL_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False
    WEB3_FORMAL_AVAILABLE = False
    translate_solidity_assignment_to_z3 = None
    verify_solidity_invariant_translation = None
    solve_smart_contract_exploit_witness = None
    logger.warning("Z3 integration not available")

# CAV-NLP imports
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
    logger.info("CAV-NLP integration available for BubbleLabs UI")
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available")

try:
    from z3_leanaide_bridge import (
        Z3LeanAideBridge, CombinedVerificationResult,
        VerificationStrategy, get_z3_leanaide_bridge_sync
    )
    Z3_LEANAIDE_AVAILABLE = True
except ImportError:
    Z3_LEANAIDE_AVAILABLE = False
    logger.warning("Z3-LeanAIDE bridge not available")

try:
    from z3_leanaide_openevolve_integration import (
        Z3LeanAideOpenEvolveIntegration,
        ProblemCategory,
        IntegratedSolution,
        get_z3_leanaide_openevolve_integration
    )
    FULL_INTEGRATION_AVAILABLE = True
except ImportError:
    FULL_INTEGRATION_AVAILABLE = False
    logger.warning("Full Z3-LeanAIDE-OpenEvolve integration not available")

try:
    from bubblelabs_leanaide_integration import (
        LeanAideIntegrationBridge, LeanAideTaskType,
        MCTSNodeVisualization, MCTSTreeVisualization,
        Lean4ProofStep, Lean4ProofVisualization,
        get_leanaide_bridge
    )
    BUBBLELABS_LEANAIDE_AVAILABLE = True
except ImportError:
    BUBBLELABS_LEANAIDE_AVAILABLE = False
    logger.warning("BubbleLabs-LeanAIDE integration not available")


# =============================================================================
# Data Classes for UI Visualization
# =============================================================================

class NodeStatus(Enum):
    """Status of a workflow node."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    CANCELLED = "cancelled"


@dataclass
class Z3ConstraintVisualization:
    """Visualization data for Z3 constraints."""
    constraint_id: str
    expression: str
    status: str  # "satisfied", "violated", "unknown"
    description: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Z3VariableAssignment:
    """Variable assignment visualization."""
    variable_name: str
    value: Any
    type: str
    bounds: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "variable_name": self.variable_name,
            "value": str(self.value),
            "type": self.type,
            "bounds": self.bounds
        }


@dataclass
class Z3SolverNodeState:
    """State for Z3 solver workflow node."""
    node_id: str
    status: NodeStatus
    problem_text: str = ""
    variables: List[Z3VariableAssignment] = field(default_factory=list)
    constraints: List[Z3ConstraintVisualization] = field(default_factory=list)
    result_status: Optional[str] = None
    execution_time: float = 0.0
    solution_found: bool = False
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "problem_text": self.problem_text,
            "variables": [v.to_dict() for v in self.variables],
            "constraints": [c.to_dict() for c in self.constraints],
            "result_status": self.result_status,
            "execution_time": self.execution_time,
            "solution_found": self.solution_found,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


@dataclass
class Z3ProofStep:
    """Single step in Z3 proof."""
    step_number: int
    tactic: str
    result: str
    goals_before: List[str] = field(default_factory=list)
    goals_after: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Z3TheoremProverNodeState:
    """State for Z3 theorem prover workflow node."""
    node_id: str
    status: NodeStatus
    theorem_statement: str = ""
    proof_steps: List[Z3ProofStep] = field(default_factory=list)
    proven: bool = False
    counterexample: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    tactic_used: Optional[str] = None
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "theorem_statement": self.theorem_statement,
            "proof_steps": [s.to_dict() for s in self.proof_steps],
            "proven": self.proven,
            "counterexample": self.counterexample,
            "execution_time": self.execution_time,
            "tactic_used": self.tactic_used,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


@dataclass
class CrossVerificationNodeState:
    """State for cross-verification node."""
    node_id: str
    status: NodeStatus
    problem_text: str = ""
    strategy_used: str = "adaptive"
    z3_status: Optional[str] = None
    lean_status: Optional[str] = None
    agreement: bool = False
    confidence_score: float = 0.0
    recommendation: str = ""
    z3_details: Optional[Dict[str, Any]] = None
    lean_details: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0
    error_message: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "problem_text": self.problem_text,
            "strategy_used": self.strategy_used,
            "z3_status": self.z3_status,
            "lean_status": self.lean_status,
            "agreement": self.agreement,
            "confidence_score": self.confidence_score,
            "recommendation": self.recommendation,
            "z3_details": self.z3_details,
            "lean_details": self.lean_details,
            "execution_time": self.execution_time,
            "error_message": self.error_message,
            "timestamp": self.timestamp
        }


@dataclass
class ProblemClassificationNodeState:
    """State for problem classification node."""
    node_id: str
    status: NodeStatus
    problem_text: str = ""
    classification: Optional[str] = None
    confidence: float = 0.0
    recommended_solver: str = ""
    alternative_solver: Optional[str] = None
    reasoning: str = ""
    suggested_strategy: str = "adaptive"
    execution_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "problem_text": self.problem_text,
            "classification": self.classification,
            "confidence": self.confidence,
            "recommended_solver": self.recommended_solver,
            "alternative_solver": self.alternative_solver,
            "reasoning": self.reasoning,
            "suggested_strategy": self.suggested_strategy,
            "execution_time": self.execution_time,
            "timestamp": self.timestamp
        }


# =============================================================================
# Z3 BubbleLabs UI Manager
# =============================================================================

class Z3BubbleLabsUIManager:
    """
    Manages Z3-specific UI components for BubbleLabs.
    
    Provides:
    - Node state management
    - Real-time updates
    - Visualization data generation
    - Event handling
    - CAV-NLP enhanced UI components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._solver_states: Dict[str, Z3SolverNodeState] = {}
        self._prover_states: Dict[str, Z3TheoremProverNodeState] = {}
        self._cross_verify_states: Dict[str, CrossVerificationNodeState] = {}
        self._classification_states: Dict[str, ProblemClassificationNodeState] = {}
        
        self._executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize core components
        self.z3_engine = get_z3_solver_engine() if Z3_AVAILABLE else None
        self.z3_prover = get_z3_theorem_prover() if Z3_AVAILABLE else None
        self.z3_bridge = get_z3_leanaide_bridge_sync() if Z3_LEANAIDE_AVAILABLE else None
        self.full_integration = get_z3_leanaide_openevolve_integration() if FULL_INTEGRATION_AVAILABLE else None
        self.leanaide_bridge = get_leanaide_bridge() if BUBBLELABS_LEANAIDE_AVAILABLE else None
        
        # CAV-NLP integration
        self.use_cav_nlp = self.config.get("use_cav_nlp", True) and CAV_NLP_AVAILABLE
        if self.use_cav_nlp:
            try:
                self.enhanced_solver = EnhancedZ3Solver()
                self.math_service = UnifiedMathService()
                logger.info("CAV-NLP UI components initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP components: {e}")
                self.use_cav_nlp = False
                self.enhanced_solver = None
                self.math_service = None
    
    def get_status(self) -> Dict[str, Any]:
        """Get UI manager status."""
        formal_capabilities = {
            "solidity_invariant_translation": translate_solidity_assignment_to_z3 is not None,
            "invariant_translation_verification": verify_solidity_invariant_translation is not None,
            "symbolic_exploit_witness": solve_smart_contract_exploit_witness is not None,
            "composite_exploit_verification": (
                translate_solidity_assignment_to_z3 is not None
                and solve_smart_contract_exploit_witness is not None
            ),
        }
        web3_formal_tools: List[str] = []
        if formal_capabilities["solidity_invariant_translation"]:
            web3_formal_tools.append("z3_translate_solidity_invariant")
        if formal_capabilities["symbolic_exploit_witness"]:
            web3_formal_tools.append("z3_solve_smart_contract_exploit_witness")
        if formal_capabilities["composite_exploit_verification"]:
            web3_formal_tools.append("z3_web3_audit_exploit_verification")

        return {
            "z3_available": Z3_AVAILABLE and self.z3_engine is not None,
            "web3_formal_available": WEB3_FORMAL_AVAILABLE,
            "web3_formal_tools": web3_formal_tools,
            "formal_capabilities": formal_capabilities,
            "z3_leanaide_available": Z3_LEANAIDE_AVAILABLE and self.z3_bridge is not None,
            "full_integration_available": FULL_INTEGRATION_AVAILABLE and self.full_integration is not None,
            "bubblelabs_leanaide_available": BUBBLELABS_LEANAIDE_AVAILABLE and self.leanaide_bridge is not None,
            "active_solver_nodes": len(self._solver_states),
            "active_prover_nodes": len(self._prover_states),
            "active_cross_verify_nodes": len(self._cross_verify_states),
            "active_classification_nodes": len(self._classification_states)
        }
    
    # =====================================================================
    # Problem Classification Node
    # =====================================================================
    
    async def create_classification_node(
        self,
        problem_text: str,
        node_id: Optional[str] = None
    ) -> ProblemClassificationNodeState:
        """Create and execute a problem classification node."""
        node_id = node_id or f"classify_{int(time.time())}_{hashlib.md5(problem_text.encode()).hexdigest()[:8]}"
        
        state = ProblemClassificationNodeState(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            problem_text=problem_text
        )
        self._classification_states[node_id] = state
        
        try:
            start_time = time.time()
            
            if self.full_integration:
                classification = self.full_integration.classifier.classify(problem_text)
                
                state.classification = classification.category.value
                state.confidence = classification.confidence
                state.recommended_solver = classification.recommended_solver
                state.alternative_solver = classification.alternative_solver
                state.reasoning = classification.reasoning
                state.suggested_strategy = classification.suggested_strategy.value
            else:
                # Fallback classification
                state.classification = "standard"
                state.confidence = 0.5
                state.recommended_solver = "standard"
                state.reasoning = "Integration not available - using default"
            
            state.execution_time = time.time() - start_time
            state.status = NodeStatus.SUCCESS
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            state.status = NodeStatus.ERROR
            state.error_message = str(e)
        
        return state
    
    def get_classification_node(self, node_id: str) -> Optional[ProblemClassificationNodeState]:
        """Get classification node state."""
        return self._classification_states.get(node_id)
    
    # =====================================================================
    # Z3 Constraint Solver Node
    # =====================================================================
    
    async def create_solver_node(
        self,
        problem_text: str,
        variables: Optional[List[Dict[str, Any]]] = None,
        constraints: Optional[List[str]] = None,
        node_id: Optional[str] = None
    ) -> Z3SolverNodeState:
        """Create and execute a Z3 constraint solver node."""
        node_id = node_id or f"z3_solver_{int(time.time())}"
        
        state = Z3SolverNodeState(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            problem_text=problem_text
        )
        self._solver_states[node_id] = state
        
        try:
            if not self.z3_engine:
                state.status = NodeStatus.ERROR
                state.error_message = "Z3 engine not available"
                return state
            
            start_time = time.time()
            
            # Parse variables and constraints from input or extract from text
            z3_vars = self._parse_variables(variables) if variables else []
            z3_constraints = self._parse_constraints(constraints) if constraints else []
            
            # Execute solver
            result = self.z3_engine.solve_constraints(z3_vars, z3_constraints)
            
            # Update state with results
            state.execution_time = time.time() - start_time
            state.result_status = result.status.value
            
            if result.model:
                state.solution_found = True
                state.variables = [
                    Z3VariableAssignment(
                        variable_name=name,
                        value=value,
                        type=type(value).__name__
                    )
                    for name, value in result.model.assignments.items()
                ]
            
            if result.is_sat():
                state.status = NodeStatus.SUCCESS
            elif result.is_unsat():
                state.status = NodeStatus.WARNING
                state.error_message = "Problem is unsatisfiable"
            else:
                state.status = NodeStatus.ERROR
                state.error_message = result.reason or "Unknown error"
            
        except Exception as e:
            logger.error(f"Z3 solving failed: {e}")
            state.status = NodeStatus.ERROR
            state.error_message = str(e)
        
        return state
    
    def _parse_variables(self, variables: List[Dict[str, Any]]) -> List[Z3Variable]:
        """Parse variable definitions."""
        z3_vars = []
        for v in variables:
            var_type_str = v.get('type', 'INTEGER').upper()
            var = Z3Variable(
                name=v['name'],
                var_type=Z3ConstraintType[var_type_str],
                bounds=v.get('bounds'),
                bit_width=v.get('bit_width')
            )
            z3_vars.append(var)
        return z3_vars
    
    def _parse_constraints(self, constraints: List[str]) -> List[Z3Constraint]:
        """Parse constraint expressions."""
        z3_constraints = []
        for c in constraints:
            constraint = Z3Constraint(
                expression=c,
                constraint_type=Z3ConstraintType.INTEGER
            )
            z3_constraints.append(constraint)
        return z3_constraints
    
    def get_solver_node(self, node_id: str) -> Optional[Z3SolverNodeState]:
        """Get solver node state."""
        return self._solver_states.get(node_id)
    
    # =====================================================================
    # Z3 Theorem Prover Node
    # =====================================================================
    
    async def create_theorem_prover_node(
        self,
        theorem_statement: str,
        smtlib_format: bool = False,
        node_id: Optional[str] = None
    ) -> Z3TheoremProverNodeState:
        """Create and execute a Z3 theorem prover node."""
        node_id = node_id or f"z3_prover_{int(time.time())}"
        
        state = Z3TheoremProverNodeState(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            theorem_statement=theorem_statement
        )
        self._prover_states[node_id] = state
        
        try:
            if not self.z3_prover:
                state.status = NodeStatus.ERROR
                state.error_message = "Z3 prover not available"
                return state
            
            start_time = time.time()
            
            # Execute prover
            result = self.z3_prover.prove_theorem(theorem_statement)
            
            state.execution_time = time.time() - start_time
            state.proven = result.proven
            state.tactic_used = result.tactic_used
            
            if result.counterexample:
                state.counterexample = result.counterexample
            
            if result.proven:
                state.status = NodeStatus.SUCCESS
                # Add proof steps if available
                if result.proof:
                    state.proof_steps.append(Z3ProofStep(
                        step_number=1,
                        tactic="smt",
                        result="proven",
                        goals_after=[]
                    ))
            else:
                state.status = NodeStatus.WARNING
                if result.counterexample:
                    state.error_message = "Counterexample found"
                else:
                    state.error_message = "Could not prove theorem"
            
        except Exception as e:
            logger.error(f"Z3 theorem proving failed: {e}")
            state.status = NodeStatus.ERROR
            state.error_message = str(e)
        
        return state
    
    def get_theorem_prover_node(self, node_id: str) -> Optional[Z3TheoremProverNodeState]:
        """Get theorem prover node state."""
        return self._prover_states.get(node_id)
    
    # =====================================================================
    # Cross-Verification Node
    # =====================================================================
    
    async def create_cross_verification_node(
        self,
        problem_text: str,
        strategy: str = "adaptive",
        node_id: Optional[str] = None,
        entanglement_context: Optional[Dict[str, Any]] = None
    ) -> CrossVerificationNodeState:
        """Create and execute a cross-verification node."""
        node_id = node_id or f"cross_verify_{int(time.time())}"
        
        state = CrossVerificationNodeState(
            node_id=node_id,
            status=NodeStatus.RUNNING,
            problem_text=problem_text,
            strategy_used=strategy
        )
        self._cross_verify_states[node_id] = state
        
        try:
            if not self.z3_bridge:
                state.status = NodeStatus.ERROR
                state.error_message = "Z3-LeanAIDE bridge not available"
                return state
            
            start_time = time.time()
            
            # Determine strategy
            strategy_enum = VerificationStrategy.ADAPTIVE
            try:
                strategy_enum = VerificationStrategy(strategy)
            except ValueError:
                pass
            
            # Execute cross-verification
            result = await self.z3_bridge.verify_with_both(
                problem_text,
                strategy_enum,
                entanglement_context=entanglement_context
            )
            
            state.execution_time = time.time() - start_time
            state.agreement = result.agreement
            state.confidence_score = result.confidence_score
            state.recommendation = result.recommendation
            
            # Extract Z3 status
            if result.z3_result:
                if hasattr(result.z3_result, 'status'):
                    state.z3_status = result.z3_result.status.value
                elif hasattr(result.z3_result, 'proven'):
                    state.z3_status = "proven" if result.z3_result.proven else "failed"
                state.z3_details = result.z3_result.to_dict() if hasattr(result.z3_result, 'to_dict') else {}
            
            # Extract Lean status
            if result.lean_result:
                if hasattr(result.lean_result, 'success'):
                    state.lean_status = "success" if result.lean_result.success else "failed"
                else:
                    state.lean_status = "success" if result.lean_result.get('success') else "failed"
                state.lean_details = result.lean_result.to_dict() if hasattr(result.lean_result, 'to_dict') else result.lean_result
            
            if result.success:
                state.status = NodeStatus.SUCCESS
            else:
                state.status = NodeStatus.WARNING
            
        except Exception as e:
            logger.error(f"Cross-verification failed: {e}")
            state.status = NodeStatus.ERROR
            state.error_message = str(e)
        
        return state
    
    def get_cross_verification_node(self, node_id: str) -> Optional[CrossVerificationNodeState]:
        """Get cross-verification node state."""
        return self._cross_verify_states.get(node_id)
    
    # =====================================================================
    # Workflow Node Definitions for BubbleLabs
    # =====================================================================
    
    def get_node_definitions(self) -> List[Dict[str, Any]]:
        """Get BubbleLabs node type definitions."""
        return [
            {
                "type": "z3_problem_classifier",
                "category": "z3_leanaide",
                "name": "Problem Classifier",
                "description": "Classifies problem for Z3 or LeanAIDE",
                "inputs": ["problem_text"],
                "outputs": ["classification", "recommended_solver"],
                "icon": "🔍",
                "color": "#6366f1"
            },
            {
                "type": "z3_constraint_solver",
                "category": "z3_leanaide",
                "name": "Z3 Constraint Solver",
                "description": "Solves constraint satisfaction problems",
                "inputs": ["problem_text", "variables", "constraints"],
                "outputs": ["solution", "assignments", "status"],
                "icon": "🔧",
                "color": "#8b5cf6"
            },
            {
                "type": "z3_theorem_prover",
                "category": "z3_leanaide",
                "name": "Z3 Theorem Prover",
                "description": "Proves theorems using Z3 SMT solver",
                "inputs": ["theorem_statement", "smtlib_format"],
                "outputs": ["proven", "proof_steps", "counterexample"],
                "icon": "📐",
                "color": "#a855f7"
            },
            {
                "type": "z3_leanaide_cross_verify",
                "category": "z3_leanaide",
                "name": "Cross Verification",
                "description": "Verifies using both Z3 and LeanAIDE",
                "inputs": ["problem_text", "strategy"],
                "outputs": ["verified", "agreement", "confidence", "recommendation"],
                "icon": "[OK][OK]",
                "color": "#ec4899"
            },
            {
                "type": "z3_smt_solver",
                "category": "z3_leanaide",
                "name": "SMT-LIB Solver",
                "description": "Solves SMT-LIB format problems",
                "inputs": ["smtlib_content"],
                "outputs": ["result", "model", "proof"],
                "icon": "⚙️",
                "color": "#06b6d4"
            },
            {
                "type": "z3_web3_invariant_translate",
                "category": "z3_web3",
                "name": "Web3 Invariant Translation",
                "description": "Translate Solidity state updates to Z3/Lean invariants",
                "inputs": ["statement", "non_negative_target", "max_withdraw_expr", "verify_translation"],
                "outputs": ["translation", "verification"],
                "icon": "⛓️",
                "color": "#0ea5e9"
            },
            {
                "type": "z3_web3_exploit_witness",
                "category": "z3_web3",
                "name": "Web3 Exploit Witness",
                "description": "Solve symbolic exploit-witness predicates for smart contracts",
                "inputs": ["additional_constraints", "timeout_seconds"],
                "outputs": ["result"],
                "icon": "🛡️",
                "color": "#ef4444"
            },
            {
                "type": "z3_web3_audit_exploit_verification",
                "category": "z3_web3",
                "name": "Web3 Exploit Verification",
                "description": "Run combined invariant translation and symbolic exploit-witness verification",
                "inputs": [
                    "statement",
                    "non_negative_target",
                    "max_withdraw_expr",
                    "verify_translation",
                    "additional_constraints",
                    "timeout_seconds",
                ],
                "outputs": ["translation", "verification", "result", "verified_exploit"],
                "icon": "🔍",
                "color": "#dc2626"
            }
        ]
    
    # =====================================================================
    # Event Handlers for BubbleLabs
    # =====================================================================
    
    async def handle_node_execution(
        self,
        node_type: str,
        node_id: str,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle node execution request from BubbleLabs.
        
        Args:
            node_type: Type of node to execute
            node_id: Unique node ID
            inputs: Node input parameters
            
        Returns:
            Execution results
        """
        try:
            if node_type == "z3_problem_classifier":
                state = await self.create_classification_node(
                    inputs.get("problem_text", ""),
                    node_id
                )
                return state.to_dict()
            
            elif node_type == "z3_constraint_solver":
                state = await self.create_solver_node(
                    problem_text=inputs.get("problem_text", ""),
                    variables=inputs.get("variables"),
                    constraints=inputs.get("constraints"),
                    node_id=node_id
                )
                return state.to_dict()
            
            elif node_type == "z3_theorem_prover":
                state = await self.create_theorem_prover_node(
                    theorem_statement=inputs.get("theorem_statement", ""),
                    smtlib_format=inputs.get("smtlib_format", False),
                    node_id=node_id
                )
                return state.to_dict()
            
            elif node_type == "z3_leanaide_cross_verify":
                state = await self.create_cross_verification_node(
                    problem_text=inputs.get("problem_text", ""),
                    strategy=inputs.get("strategy", "adaptive"),
                    node_id=node_id
                )
                return state.to_dict()
            
            elif node_type == "z3_smt_solver":
                # Use constraint solver with SMT-LIB content
                state = await self.create_solver_node(
                    problem_text=inputs.get("smtlib_content", ""),
                    node_id=node_id
                )
                return state.to_dict()

            elif node_type == "z3_web3_invariant_translate":
                if translate_solidity_assignment_to_z3 is None:
                    return {"status": "error", "error": "Web3 invariant translator unavailable"}
                translation = translate_solidity_assignment_to_z3(
                    statement=inputs.get("statement", ""),
                    non_negative_target=bool(inputs.get("non_negative_target", True)),
                    max_withdraw_expr=inputs.get("max_withdraw_expr"),
                )
                result: Dict[str, Any] = {
                    "status": "success",
                    "node_id": node_id,
                    "translation": translation,
                }
                if bool(inputs.get("verify_translation", True)) and verify_solidity_invariant_translation is not None:
                    result["verification"] = verify_solidity_invariant_translation(
                        translation=translation,
                        assume_non_negative_amount=bool(
                            inputs.get("assume_non_negative_amount", True)
                        ),
                    )
                return result

            elif node_type == "z3_web3_exploit_witness":
                if solve_smart_contract_exploit_witness is None:
                    return {"status": "error", "error": "Web3 exploit witness solver unavailable"}
                witness = solve_smart_contract_exploit_witness(
                    additional_constraints=inputs.get("additional_constraints"),
                    timeout=float(inputs.get("timeout_seconds", 10.0)),
                )
                return {
                    "status": "success",
                    "node_id": node_id,
                    "result": witness,
                }

            elif node_type == "z3_web3_audit_exploit_verification":
                if translate_solidity_assignment_to_z3 is None:
                    return {"status": "error", "error": "Web3 invariant translator unavailable"}
                if solve_smart_contract_exploit_witness is None:
                    return {"status": "error", "error": "Web3 exploit witness solver unavailable"}

                translation = translate_solidity_assignment_to_z3(
                    statement=inputs.get("statement", ""),
                    non_negative_target=bool(inputs.get("non_negative_target", True)),
                    max_withdraw_expr=inputs.get("max_withdraw_expr"),
                )
                verification = None
                if bool(inputs.get("verify_translation", True)) and verify_solidity_invariant_translation is not None:
                    verification = verify_solidity_invariant_translation(
                        translation=translation,
                        assume_non_negative_amount=bool(inputs.get("assume_non_negative_amount", True)),
                    )

                witness = solve_smart_contract_exploit_witness(
                    additional_constraints=inputs.get("additional_constraints"),
                    timeout=float(inputs.get("timeout_seconds", 10.0)),
                )

                verified_exploit = bool(witness.get("satisfiable", False))
                if bool(inputs.get("verify_translation", True)) and isinstance(verification, dict):
                    verified_exploit = verified_exploit and bool(verification.get("proven", False))

                return {
                    "status": "success",
                    "node_id": node_id,
                    "translation": translation,
                    "verification": verification,
                    "result": witness,
                    "verified_exploit": verified_exploit,
                }
            
            else:
                return {
                    "error": f"Unknown node type: {node_type}",
                    "status": "error"
                }
                
        except Exception as e:
            logger.error(f"Node execution failed: {e}")
            return {
                "error": str(e),
                "status": "error",
                "node_id": node_id,
                "node_type": node_type
            }
    
    def get_all_node_states(self) -> Dict[str, Dict[str, Any]]:
        """Get all node states for UI display."""
        return {
            "classification_nodes": {
                node_id: state.to_dict()
                for node_id, state in self._classification_states.items()
            },
            "solver_nodes": {
                node_id: state.to_dict()
                for node_id, state in self._solver_states.items()
            },
            "prover_nodes": {
                node_id: state.to_dict()
                for node_id, state in self._prover_states.items()
            },
            "cross_verify_nodes": {
                node_id: state.to_dict()
                for node_id, state in self._cross_verify_states.items()
            }
        }
    
    def create_cav_nlp_ui(self) -> Optional[Dict[str, Any]]:
        """Create CAV-NLP enhanced UI components."""
        if self.use_cav_nlp:
            return {
                "formalize_panel": self._create_formalize_panel(),
                "verify_panel": self._create_verify_panel(),
                "export_panel": self._create_export_panel()
            }
        return None
    
    def _create_formalize_panel(self) -> Dict[str, Any]:
        """Create CAV-NLP formalization panel."""
        return {
            "type": "formalize_panel",
            "title": "CAV-NLP Formalization",
            "description": "Formalize natural language to Lean 4 code",
            "inputs": ["natural_language_text"],
            "outputs": ["formalized_code", "confidence_score"],
            "icon": "📝",
            "color": "#4f46e5"
        }
    
    def _create_verify_panel(self) -> Dict[str, Any]:
        """Create CAV-NLP verification panel."""
        return {
            "type": "verify_panel",
            "title": "CAV-NLP Verification",
            "description": "Verify constraints using enhanced Z3 solver",
            "inputs": ["constraints", "expected_result"],
            "outputs": ["verification_result", "proof", "counterexample"],
            "icon": "✓",
            "color": "#059669"
        }
    
    def _create_export_panel(self) -> Dict[str, Any]:
        """Create CAV-NLP export panel."""
        return {
            "type": "export_panel",
            "title": "CAV-NLP Export",
            "description": "Export formalized code to various formats",
            "inputs": ["code", "target_format"],
            "outputs": ["exported_code", "download_link"],
            "icon": "📤",
            "color": "#7c3aed"
        }


# =============================================================================
# Global Instance
# =============================================================================

_z3_bubblelabs_ui: Optional[Z3BubbleLabsUIManager] = None


def get_z3_bubblelabs_ui() -> Z3BubbleLabsUIManager:
    """Get global Z3 BubbleLabs UI manager."""
    global _z3_bubblelabs_ui
    if _z3_bubblelabs_ui is None:
        _z3_bubblelabs_ui = Z3BubbleLabsUIManager()
    return _z3_bubblelabs_ui


# =============================================================================
# BubbleLabs Tool Registration
# =============================================================================

def register_z3_leanaide_bubblelabs_tools():
    """
    Register Z3-LeanAIDE tools with BubbleLabs.
    
    This should be called during BubbleLabs initialization.
    """
    try:
        ui_manager = get_z3_bubblelabs_ui()
        
        # Get node definitions
        node_definitions = ui_manager.get_node_definitions()
        
        logger.info(f"Registering {len(node_definitions)} Z3-LeanAIDE node types with BubbleLabs")
        
        for node_def in node_definitions:
            logger.info(f"  Registered: {node_def['name']} ({node_def['type']})")
        
        return {
            "success": True,
            "nodes_registered": len(node_definitions),
            "node_types": [n['type'] for n in node_definitions]
        }
        
    except Exception as e:
        logger.error(f"Failed to register Z3-LeanAIDE tools: {e}")
        return {
            "success": False,
            "error": str(e)
        }


# =============================================================================
# Example Usage
# =============================================================================

async def example_classification_node():
    """Example: Create a classification node."""
    ui = get_z3_bubblelabs_ui()
    
    problem = "Prove that for all positive integers x, x + 1 > x"
    
    state = await ui.create_classification_node(problem)
    
    print("=" * 60)
    print("Problem Classification Example")
    print("=" * 60)
    print(f"Problem: {problem}")
    print(f"Classification: {state.classification}")
    print(f"Confidence: {state.confidence:.2f}")
    print(f"Recommended Solver: {state.recommended_solver}")
    print(f"Reasoning: {state.reasoning}")
    
    return state


async def example_solver_node():
    """Example: Create a constraint solver node."""
    ui = get_z3_bubblelabs_ui()
    
    problem = "Find x and y where x > 0, y = x + 5, and x < 10"
    
    variables = [
        {"name": "x", "type": "INTEGER"},
        {"name": "y", "type": "INTEGER"}
    ]
    
    constraints = [
        "(> x 0)",
        "(< x 10)",
        "(= y (+ x 5))"
    ]
    
    state = await ui.create_solver_node(problem, variables, constraints)
    
    print("\n" + "=" * 60)
    print("Constraint Solver Example")
    print("=" * 60)
    print(f"Problem: {problem}")
    print(f"Status: {state.status.value}")
    print(f"Result: {state.result_status}")
    print(f"Solution Found: {state.solution_found}")
    if state.variables:
        print("Variable Assignments:")
        for var in state.variables:
            print(f"  {var.variable_name} = {var.value}")
    
    return state


async def main():
    """Run all examples."""
    print("Z3-LeanAIDE BubbleLabs UI Integration")
    print("=" * 60)
    
    # Show status
    ui = get_z3_bubblelabs_ui()
    status = ui.get_status()
    
    print(f"\nUI Manager Status:")
    print(f"  Z3: {status['z3_available']}")
    print(f"  Z3-LeanAIDE Bridge: {status['z3_leanaide_available']}")
    print(f"  Full Integration: {status['full_integration_available']}")
    
    # Show node definitions
    print(f"\nAvailable Node Types:")
    for node_def in ui.get_node_definitions():
        print(f"  {node_def['icon']} {node_def['name']} ({node_def['type']})")
    
    # Run examples
    if status['full_integration_available']:
        await example_classification_node()
    
    if status['z3_available']:
        await example_solver_node()


if __name__ == "__main__":
    asyncio.run(main())
