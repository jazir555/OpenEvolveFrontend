"""
Matryoshka Execution Engine

Positions Matryoshka as a generalized execution engine within:
- ROMA's recursive decomposition framework
- Decomposition workflow's Blue/Red/Gold teams
- MDAP/MAKER's voting and error correction

Matryoshka becomes the "Executor" that:
- Explores problem spaces iteratively
- Executes symbolic commands via Nucleus/Lattice
- Maintains state across iterations
- Reports findings for aggregation

Key Insight: Matryoshka is a Recursive Language Model (RLM) that:
1. Uses LLM reasoning to output symbolic Nucleus commands
2. Executes via Lattice engine
3. Iterates based on observations
4. Handles arbitrary problem spaces through symbolic manipulation
"""

from __future__ import annotations

import json
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union, Set

# Configure logging
logger = logging.getLogger(__name__)

# ============================================================================
# ENUMERATIONS AND TYPES
# ============================================================================

class ExplorationMode(Enum):
    """Exploration strategies for problem spaces."""
    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    ADAPTIVE = "adaptive"
    GUIDED = "guided"


class FindingCategory(Enum):
    """Categories of findings during exploration."""
    FACT = "fact"
    RELATIONSHIP = "relationship"
    PATTERN = "pattern"
    CONSTRAINT = "constraint"
    VULNERABILITY = "vulnerability"
    OPTIMIZATION = "optimization"
    INSIGHT = "insight"


class ErrorType(Enum):
    """Types of failures during exploration."""
    TIMEOUT = "timeout"
    NO_RESULTS = "no_results"
    WRONG_RESULTS = "wrong_results"
    EXCEPTION = "exception"
    INVALID_COMMAND = "invalid_command"
    SAFETY_VIOLATION = "safety_violation"


class TeamRole(Enum):
    """Team roles in decomposition workflow."""
    BLUE = "blue"
    RED = "red"
    GOLD = "gold"


# ============================================================================
# MATRYOSHKA AS GENERALIZED EXECUTION ENGINE - CONFIGURATION
# ============================================================================

@dataclass
class MatryoshkaExecutionConfig:
    """
    Configuration for Matryoshka as execution engine.
    
    Not just for documents - for ANY problem space that can be:
    - Represented symbolically
    - Explored iteratively
    - Reasoned about via LLM
    """
    # Nucleus/Lattice execution
    max_iterations: int = 20
    nucleus_timeout_ms: int = 30000
    allow_code_execution: bool = False  # Safety: only symbolic Nucleus
    
    # State management (from unified memory)
    enable_state_tracking: bool = True
    state_storage_path: Optional[str] = None
    
    # ROMA integration
    report_intermediate_findings: bool = True  # For ROMA aggregation
    finding_batch_size: int = 5  # How often to report
    
    # Exploration strategy
    exploration_mode: ExplorationMode = ExplorationMode.ADAPTIVE
    backtrack_on_failure: bool = True
    
    # LLM configuration
    llm_model: str = "gpt-4"
    temperature: float = 0.7
    max_tokens: int = 2000
    
    # Safety and limits
    max_command_length: int = 1000
    max_findings_per_iteration: int = 10
    enable_safety_checks: bool = True
    
    # Integration settings
    roma_integration_enabled: bool = True
    maker_voting_enabled: bool = False
    decomposition_integration_enabled: bool = True


# ============================================================================
# EXECUTION STATE MANAGEMENT
# ============================================================================

@dataclass
class Finding:
    """A discovery during execution."""
    finding_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    iteration: int = 0
    category: FindingCategory = FindingCategory.FACT
    content: str = ""
    confidence: float = 0.0
    nucleus_command: Optional[str] = None
    handle_reference: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "iteration": self.iteration,
            "category": self.category.value,
            "content": self.content,
            "confidence": self.confidence,
            "nucleus_command": self.nucleus_command,
            "handle_reference": self.handle_reference,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }


@dataclass
class FailedAttempt:
    """Record of failed exploration for learning."""
    iteration: int = 0
    hypothesis: str = ""
    nucleus_command: str = ""
    failure_reason: str = ""
    error_type: ErrorType = ErrorType.EXCEPTION
    timestamp: datetime = field(default_factory=datetime.now)
    recovery_action: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "iteration": self.iteration,
            "hypothesis": self.hypothesis,
            "nucleus_command": self.nucleus_command,
            "failure_reason": self.failure_reason,
            "error_type": self.error_type.value,
            "timestamp": self.timestamp.isoformat(),
            "recovery_action": self.recovery_action
        }


@dataclass
class ExecutionState:
    """
    Current execution state that Matryoshka maintains.
    
    Similar to document state, but generalized for any problem:
    - Symbolic representations
    - Intermediate results
    - Exploration history
    - Current hypothesis
    """
    iteration: int = 0
    symbolic_state: Dict[str, Any] = field(default_factory=dict)
    exploration_path: List[str] = field(default_factory=list)
    current_hypothesis: Optional[str] = None
    accumulated_findings: List[Finding] = field(default_factory=list)
    failed_attempts: List[FailedAttempt] = field(default_factory=list)
    context_window: List[Dict[str, Any]] = field(default_factory=list)
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_finding(self, finding: Finding) -> None:
        """Add a finding to the accumulated findings."""
        self.accumulated_findings.append(finding)
        self.updated_at = datetime.now()
    
    def add_failure(self, failure: FailedAttempt) -> None:
        """Add a failed attempt for learning."""
        self.failed_attempts.append(failure)
        self.updated_at = datetime.now()
    
    def update_symbolic_state(self, key: str, value: Any) -> None:
        """Update the symbolic state."""
        self.symbolic_state[key] = value
        self.updated_at = datetime.now()
    
    def add_to_path(self, step: str) -> None:
        """Add a step to the exploration path."""
        self.exploration_path.append(step)
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "state_id": self.state_id,
            "iteration": self.iteration,
            "symbolic_state": self.symbolic_state,
            "exploration_path": self.exploration_path,
            "current_hypothesis": self.current_hypothesis,
            "accumulated_findings": [f.to_dict() for f in self.accumulated_findings],
            "failed_attempts": [f.to_dict() for f in self.failed_attempts],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ExecutionState:
        """Restore state from dictionary."""
        findings = [
            Finding(
                finding_id=f["finding_id"],
                iteration=f["iteration"],
                category=FindingCategory(f["category"]),
                content=f["content"],
                confidence=f["confidence"],
                nucleus_command=f.get("nucleus_command"),
                handle_reference=f.get("handle_reference"),
                timestamp=datetime.fromisoformat(f["timestamp"]),
                metadata=f.get("metadata", {})
            )
            for f in data.get("accumulated_findings", [])
        ]
        
        failures = [
            FailedAttempt(
                iteration=f["iteration"],
                hypothesis=f["hypothesis"],
                nucleus_command=f["nucleus_command"],
                failure_reason=f["failure_reason"],
                error_type=ErrorType(f["error_type"]),
                timestamp=datetime.fromisoformat(f["timestamp"]),
                recovery_action=f.get("recovery_action")
            )
            for f in data.get("failed_attempts", [])
        ]
        
        return cls(
            state_id=data.get("state_id", str(uuid.uuid4())),
            iteration=data["iteration"],
            symbolic_state=data.get("symbolic_state", {}),
            exploration_path=data.get("exploration_path", []),
            current_hypothesis=data.get("current_hypothesis"),
            accumulated_findings=findings,
            failed_attempts=failures,
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


# ============================================================================
# EXECUTION RESULTS
# ============================================================================

@dataclass
class ExecutionResult:
    """Result of a Matryoshka execution."""
    success: bool
    final_state: ExecutionState
    primary_finding: Optional[Finding] = None
    summary: str = ""
    execution_time_ms: float = 0.0
    iterations_completed: int = 0
    confidence_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_findings_by_category(self, category: FindingCategory) -> List[Finding]:
        """Get all findings of a specific category."""
        return [f for f in self.final_state.accumulated_findings if f.category == category]
    
    def get_high_confidence_findings(self, threshold: float = 0.8) -> List[Finding]:
        """Get findings with confidence above threshold."""
        return [f for f in self.final_state.accumulated_findings if f.confidence >= threshold]


@dataclass
class BranchResult:
    """Result of exploring a specific branch."""
    branch_id: str
    hypothesis: str
    result: ExecutionResult
    sub_branches: List[BranchResult] = field(default_factory=list)
    depth: int = 0
    parent_branch_id: Optional[str] = None
    
    def get_all_findings(self) -> List[Finding]:
        """Recursively get all findings from this branch and sub-branches."""
        findings = list(self.result.final_state.accumulated_findings)
        for sub in self.sub_branches:
            findings.extend(sub.get_all_findings())
        return findings


# ============================================================================
# PROBLEM SPACE DEFINITIONS
# ============================================================================

@dataclass
class ProblemSpace:
    """Abstract problem space that Matryoshka can explore."""
    space_type: str  # "document", "codebase", "database", "config", "abstract"
    representation: Any  # How the space is represented
    operations: List[str]  # Available Nucleus operations
    constraints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    space_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    def get_available_operations(self) -> List[str]:
        """Get list of available operations for this space."""
        return self.operations
    
    def validate_operation(self, operation: str) -> bool:
        """Check if an operation is valid for this space."""
        return operation in self.operations


@dataclass
class DocumentSpace(ProblemSpace):
    """Traditional Matryoshka document space."""
    document_path: Optional[str] = None
    document_content: Optional[str] = None
    document_type: str = "text"  # text, pdf, markdown, etc.
    
    def __post_init__(self):
        if not self.space_type:
            self.space_type = "document"
        if not self.operations:
            self.operations = [
                "read_section",
                "search_text",
                "extract_entities",
                "summarize",
                "find_relationships",
                "compare_sections"
            ]


@dataclass
class CodebaseSpace(ProblemSpace):
    """Code exploration space with symbol extraction."""
    repository_path: Optional[str] = None
    language: str = "python"
    entry_points: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.space_type:
            self.space_type = "codebase"
        if not self.operations:
            self.operations = [
                "find_symbol",
                "get_definition",
                "find_references",
                "analyze_dependencies",
                "extract_api",
                "find_patterns",
                "detect_vulnerabilities",
                "measure_complexity"
            ]


@dataclass
class DatabaseSpace(ProblemSpace):
    """Data exploration space with query capabilities."""
    connection_string: Optional[str] = None
    schema: Optional[Dict[str, Any]] = None
    dialect: str = "sql"
    
    def __post_init__(self):
        if not self.space_type:
            self.space_type = "database"
        if not self.operations:
            self.operations = [
                "execute_query",
                "describe_table",
                "find_relationships",
                "aggregate_data",
                "detect_anomalies",
                "generate_report"
            ]


@dataclass
class ConfigSpace(ProblemSpace):
    """Configuration space with constraint solving."""
    config_format: str = "yaml"  # yaml, json, toml, etc.
    schema_definition: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if not self.space_type:
            self.space_type = "config"
        if not self.operations:
            self.operations = [
                "validate_config",
                "check_constraints",
                "suggest_optimizations",
                "detect_conflicts",
                "generate_template"
            ]


@dataclass
class AbstractSpace(ProblemSpace):
    """Abstract problem space for arbitrary symbolic manipulation."""
    domain: str = "general"
    symbolic_representation: Optional[Any] = None
    
    def __post_init__(self):
        if not self.space_type:
            self.space_type = "abstract"
        if not self.operations:
            self.operations = [
                "apply_transformation",
                "check_property",
                "search_pattern",
                "infer_relationship",
                "verify_constraint"
            ]


# ============================================================================
# NUCLEUS/LATTICE COMMAND SYSTEM
# ============================================================================

@dataclass
class NucleusCommand:
    """A symbolic command for the Nucleus/Lattice engine."""
    command_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    command_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    safety_level: str = "restricted"  # restricted, standard, elevated
    timeout_ms: int = 30000
    
    def to_json(self) -> str:
        """Convert to JSON for execution."""
        return json.dumps({
            "command_id": self.command_id,
            "command_type": self.command_type,
            "parameters": self.parameters,
            "safety_level": self.safety_level,
            "timeout_ms": self.timeout_ms
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> NucleusCommand:
        """Parse from JSON."""
        data = json.loads(json_str)
        return cls(
            command_type=data["command_type"],
            parameters=data.get("parameters", {}),
            command_id=data.get("command_id", str(uuid.uuid4())[:8]),
            safety_level=data.get("safety_level", "restricted"),
            timeout_ms=data.get("timeout_ms", 30000)
        )


@dataclass
class NucleusResult:
    """Result from executing a Nucleus command."""
    command_id: str
    success: bool
    output: Any
    execution_time_ms: float
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    handle: Optional[str] = None  # Matryoshka handle reference


# ============================================================================
# LLM REASONING INTERFACE
# ============================================================================

class LLMReasoningEngine:
    """Interface for LLM reasoning in Matryoshka execution."""
    
    def __init__(self, model: str = "gpt-4", temperature: float = 0.7):
        self.model = model
        self.temperature = temperature
        self.reasoning_history: List[Dict[str, Any]] = []
    
    def reason_about_state(
        self,
        state: ExecutionState,
        task: str,
        problem_space: ProblemSpace,
        available_operations: List[str]
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about current state and determine next action.
        
        Returns:
            Dict with keys:
            - hypothesis: Current hypothesis about the problem
            - command: Nucleus command to execute
            - reasoning: Explanation of the reasoning
            - expected_outcome: What we expect to learn
        """
        # Build context from state
        context = self._build_context(state, task, problem_space)
        
        # This would integrate with actual LLM
        # For now, return structured reasoning placeholder
        reasoning = {
            "hypothesis": f"Exploring {problem_space.space_type} for: {task}",
            "command": self._generate_command(state, available_operations),
            "reasoning": f"Based on iteration {state.iteration}, exploring next aspect",
            "expected_outcome": "New findings or confirmation of hypothesis",
            "confidence": 0.75
        }
        
        self.reasoning_history.append({
            "iteration": state.iteration,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        })
        
        return reasoning
    
    def _build_context(
        self,
        state: ExecutionState,
        task: str,
        problem_space: ProblemSpace
    ) -> str:
        """Build context string for LLM prompting."""
        context_parts = [
            f"Task: {task}",
            f"Problem Space: {problem_space.space_type}",
            f"Iteration: {state.iteration}",
            f"Current Hypothesis: {state.current_hypothesis or 'None'}",
            f"Exploration Path: {' -> '.join(state.exploration_path[-5:])}",
            f"Findings so far: {len(state.accumulated_findings)}",
            f"Failed attempts: {len(state.failed_attempts)}"
        ]
        return "\n".join(context_parts)
    
    def _generate_command(
        self,
        state: ExecutionState,
        available_operations: List[str]
    ) -> NucleusCommand:
        """Generate next Nucleus command based on reasoning."""
        # Simplified: select operation based on state
        if not available_operations:
            available_operations = ["explore"]
        
        operation = available_operations[state.iteration % len(available_operations)]
        
        return NucleusCommand(
            command_type=operation,
            parameters={"iteration": state.iteration},
            safety_level="restricted"
        )


# ============================================================================
# CORE EXECUTION ENGINE
# ============================================================================

class MatryoshkaExecutionEngine:
    """
    Generalized execution engine using Matryoshka's RLM approach.
    
    Integrates with:
    - ROMA: As the recursive solver for sub-problems
    - Decomposition: As the Blue Team executor
    - MDAP/MAKER: For voting on execution paths
    
    Not limited to documents - can explore:
    - Codebases (via symbol extraction)
    - Data spaces (via symbolic manipulation)
    - Configuration spaces (via constraint solving)
    - Abstract problem spaces
    """
    
    def __init__(self, config: MatryoshkaExecutionConfig):
        self.config = config
        self.reasoning_engine = LLMReasoningEngine(
            model=config.llm_model,
            temperature=config.temperature
        )
        self.execution_history: List[Dict[str, Any]] = []
        self._initialize_storage()
        logger.info(f"Initialized MatryoshkaExecutionEngine with {config.exploration_mode.value} mode")
    
    def _initialize_storage(self) -> None:
        """Initialize state storage if enabled."""
        if self.config.enable_state_tracking and self.config.state_storage_path:
            path = Path(self.config.state_storage_path)
            path.parent.mkdir(parents=True, exist_ok=True)
    
    def _init_matryoshka(self) -> Any:
        """Initialize Matryoshka client (placeholder for actual integration)."""
        # This would initialize the actual Matryoshka RLM client
        return None
    
    def _init_memory(self) -> Any:
        """Initialize unified memory system (placeholder)."""
        # This would integrate with matryoshka_unified_memory_integration.py
        return None
    
    def execute(
        self,
        task: str,
        problem_space: ProblemSpace,
        initial_state: Optional[ExecutionState] = None
    ) -> ExecutionResult:
        """
        Execute task within problem space using iterative RLM approach.
        
        Args:
            task: What to accomplish (e.g., "Find security vulnerabilities")
            problem_space: The space to explore (code, data, config, etc.)
            initial_state: Optional state from previous execution
            
        Returns:
            ExecutionResult with findings and final state
        """
        import time
        start_time = time.time()
        
        # Initialize state
        state = initial_state or ExecutionState(iteration=0)
        state.current_hypothesis = task
        
        logger.info(f"Starting execution for task: {task}")
        logger.info(f"Problem space: {problem_space.space_type}")
        
        should_continue = True
        
        while should_continue and state.iteration < self.config.max_iterations:
            try:
                state, should_continue = self._iterate(state, task, problem_space)
                
                # Report intermediate findings if enabled
                if (self.config.report_intermediate_findings and 
                    state.iteration % self.config.finding_batch_size == 0):
                    self._report_intermediate_findings(state)
                    
            except Exception as e:
                logger.error(f"Error in iteration {state.iteration}: {e}")
                state.add_failure(FailedAttempt(
                    iteration=state.iteration,
                    hypothesis=state.current_hypothesis or "",
                    nucleus_command="",
                    failure_reason=str(e),
                    error_type=ErrorType.EXCEPTION
                ))
                
                if not self.config.backtrack_on_failure:
                    break
        
        execution_time = (time.time() - start_time) * 1000
        
        # Determine success based on findings
        success = len(state.accumulated_findings) > 0
        
        # Get primary finding (highest confidence)
        primary_finding = None
        if state.accumulated_findings:
            primary_finding = max(
                state.accumulated_findings,
                key=lambda f: f.confidence
            )
        
        result = ExecutionResult(
            success=success,
            final_state=state,
            primary_finding=primary_finding,
            summary=self._generate_summary(state, task),
            execution_time_ms=execution_time,
            iterations_completed=state.iteration,
            confidence_score=self._calculate_confidence(state)
        )
        
        # Save state if tracking enabled
        if self.config.enable_state_tracking:
            self._save_state(state)
        
        logger.info(f"Execution completed: {result.iterations_completed} iterations, "
                   f"{len(state.accumulated_findings)} findings")
        
        return result
    
    def _iterate(
        self,
        state: ExecutionState,
        task: str,
        problem_space: ProblemSpace
    ) -> Tuple[ExecutionState, bool]:
        """
        Single iteration:
        1. LLM reasons about current state
        2. Generates Nucleus command
        3. Executes via Lattice
        4. Observes results
        5. Updates state
        
        Returns:
            (new_state, should_continue)
        """
        state.iteration += 1
        logger.debug(f"Iteration {state.iteration}")
        
        # Step 1: LLM reasoning
        reasoning = self.reasoning_engine.reason_about_state(
            state, task, problem_space, problem_space.operations
        )
        
        state.current_hypothesis = reasoning["hypothesis"]
        
        # Step 2: Generate Nucleus command
        command = reasoning["command"]
        if isinstance(command, dict):
            command = NucleusCommand(
                command_type=command.get("command_type", "explore"),
                parameters=command.get("parameters", {})
            )
        
        # Step 3: Execute via Lattice (placeholder)
        nucleus_result = self._execute_nucleus_command(command, problem_space)
        
        # Step 4: Observe and update state
        if nucleus_result.success:
            # Create finding from result
            finding = Finding(
                iteration=state.iteration,
                category=FindingCategory.INSIGHT,
                content=str(nucleus_result.output),
                confidence=reasoning.get("confidence", 0.5),
                nucleus_command=command.command_type,
                handle_reference=nucleus_result.handle
            )
            state.add_finding(finding)
            state.update_symbolic_state(
                f"iteration_{state.iteration}",
                nucleus_result.output
            )
        else:
            # Record failure
            failure = FailedAttempt(
                iteration=state.iteration,
                hypothesis=state.current_hypothesis,
                nucleus_command=command.command_type,
                failure_reason=nucleus_result.error_message or "Unknown error",
                error_type=ErrorType.EXCEPTION
            )
            state.add_failure(failure)
        
        state.add_to_path(f"iter_{state.iteration}:{command.command_type}")
        
        # Step 5: Determine if should continue
        should_continue = self._should_continue(state, task)
        
        return state, should_continue
    
    def _execute_nucleus_command(
        self,
        command: NucleusCommand,
        problem_space: ProblemSpace
    ) -> NucleusResult:
        """
        Execute a Nucleus command via the Lattice engine.
        
        This is a placeholder for the actual Lattice execution.
        """
        # Validate command against problem space
        if not problem_space.validate_operation(command.command_type):
            return NucleusResult(
                command_id=command.command_id,
                success=False,
                output=None,
                execution_time_ms=0,
                error_message=f"Invalid operation '{command.command_type}' for {problem_space.space_type}"
            )
        
        # Simulate execution (replace with actual Lattice integration)
        import time
        start = time.time()
        
        # Placeholder: simulate successful execution
        output = {
            "command": command.command_type,
            "space": problem_space.space_type,
            "parameters": command.parameters,
            "result": f"Simulated {command.command_type} execution"
        }
        
        execution_time = (time.time() - start) * 1000
        
        return NucleusResult(
            command_id=command.command_id,
            success=True,
            output=output,
            execution_time_ms=execution_time,
            handle=f"handle_{command.command_id}"
        )
    
    def _should_continue(self, state: ExecutionState, task: str) -> bool:
        """Determine if exploration should continue."""
        # Stop if max findings reached
        if len(state.accumulated_findings) >= self.config.max_findings_per_iteration * self.config.max_iterations:
            return False
        
        # Stop if too many consecutive failures
        recent_failures = [
            f for f in state.failed_attempts 
            if f.iteration > state.iteration - 3
        ]
        if len(recent_failures) >= 3:
            logger.warning("Too many consecutive failures, stopping")
            return False
        
        # Continue based on exploration mode
        if self.config.exploration_mode == ExplorationMode.DEPTH_FIRST:
            # Continue until depth limit or solution found
            return state.iteration < self.config.max_iterations
        
        # Adaptive: continue if making progress
        recent_findings = [
            f for f in state.accumulated_findings
            if f.iteration > state.iteration - 3
        ]
        return len(recent_findings) > 0 or state.iteration < 5
    
    def explore_branch(
        self,
        branch_hypothesis: str,
        parent_state: ExecutionState,
        problem_space: ProblemSpace,
        depth: int = 0,
        max_depth: int = 5
    ) -> BranchResult:
        """
        Explore a specific branch/hypothesis.
        Used by ROMA for recursive solving.
        """
        branch_id = str(uuid.uuid4())[:8]
        logger.info(f"Exploring branch {branch_id} at depth {depth}: {branch_hypothesis}")
        
        # Create branch-specific state
        branch_state = ExecutionState(
            iteration=0,
            current_hypothesis=branch_hypothesis,
            symbolic_state=dict(parent_state.symbolic_state),
            exploration_path=list(parent_state.exploration_path)
        )
        
        # Execute exploration
        result = self.execute(branch_hypothesis, problem_space, branch_state)
        
        branch_result = BranchResult(
            branch_id=branch_id,
            hypothesis=branch_hypothesis,
            result=result,
            depth=depth
        )
        
        # Recursive exploration if needed and depth allows
        if depth < max_depth and result.success:
            sub_hypotheses = self._generate_sub_hypotheses(result)
            for sub_hypothesis in sub_hypotheses[:3]:  # Limit branching factor
                sub_result = self.explore_branch(
                    sub_hypothesis,
                    result.final_state,
                    problem_space,
                    depth + 1,
                    max_depth
                )
                sub_result.parent_branch_id = branch_id
                branch_result.sub_branches.append(sub_result)
        
        return branch_result
    
    def _generate_sub_hypotheses(self, result: ExecutionResult) -> List[str]:
        """Generate sub-hypotheses based on findings."""
        hypotheses = []
        for finding in result.final_state.accumulated_findings[:3]:
            hypotheses.append(f"Investigate: {finding.content[:50]}...")
        return hypotheses
    
    def _report_intermediate_findings(self, state: ExecutionState) -> None:
        """Report intermediate findings for ROMA aggregation."""
        recent_findings = [
            f for f in state.accumulated_findings
            if f.iteration > state.iteration - self.config.finding_batch_size
        ]
        logger.info(f"Intermediate report: {len(recent_findings)} new findings at iteration {state.iteration}")
    
    def _generate_summary(self, state: ExecutionState, task: str) -> str:
        """Generate execution summary."""
        summary_parts = [
            f"Task: {task}",
            f"Iterations: {state.iteration}",
            f"Total findings: {len(state.accumulated_findings)}",
            f"Failed attempts: {len(state.failed_attempts)}",
            f"Exploration path: {' -> '.join(state.exploration_path[-5:])}"
        ]
        return "\n".join(summary_parts)
    
    def _calculate_confidence(self, state: ExecutionState) -> float:
        """Calculate overall confidence score."""
        if not state.accumulated_findings:
            return 0.0
        
        total_confidence = sum(f.confidence for f in state.accumulated_findings)
        avg_confidence = total_confidence / len(state.accumulated_findings)
        
        # Adjust for failures
        failure_penalty = len(state.failed_attempts) * 0.05
        
        return max(0.0, min(1.0, avg_confidence - failure_penalty))
    
    def _save_state(self, state: ExecutionState) -> None:
        """Save state to storage."""
        if self.config.state_storage_path:
            try:
                state_path = Path(self.config.state_storage_path) / f"state_{state.state_id}.json"
                with open(state_path, 'w') as f:
                    json.dump(state.to_dict(), f, indent=2)
            except Exception as e:
                logger.error(f"Failed to save state: {e}")


# ============================================================================
# ROMA INTEGRATION
# ============================================================================

@dataclass
class SubProblem:
    """A sub-problem for ROMA solving."""
    problem_id: str
    description: str
    problem_space_type: str
    constraints: List[str] = field(default_factory=list)
    parent_problem_id: Optional[str] = None
    priority: int = 1


@dataclass
class SubProblemSolution:
    """Solution to a sub-problem."""
    problem_id: str
    solution_content: str
    findings: List[Finding] = field(default_factory=list)
    confidence: float = 0.0
    verification_status: str = "unverified"


@dataclass
class ROMAContext:
    """Context for ROMA operations."""
    context_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_solution_id: Optional[str] = None
    aggregation_strategy: str = "consensus"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CritiqueResult:
    """Result of critiquing a solution."""
    solution_id: str
    issues_found: List[str] = field(default_factory=list)
    severity_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    critique_confidence: float = 0.0


@dataclass
class VerificationResult:
    """Result of verifying a solution."""
    solution_id: str
    verified: bool
    evidence: List[str] = field(default_factory=list)
    failed_requirements: List[str] = field(default_factory=list)
    verification_confidence: float = 0.0


class ROMAMatryoshkaSolver:
    """
    Matryoshka as ROMA's recursive solver.
    
    Replaces or enhances ROMA's default solver with Matryoshka's
    iterative symbolic execution.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig()
        self.execution_engine = MatryoshkaExecutionEngine(self.config)
        self.solution_history: List[SubProblemSolution] = []
    
    def solve_subproblem(
        self,
        subproblem: SubProblem,
        context: ROMAContext
    ) -> SubProblemSolution:
        """
        Solve a sub-problem using Matryoshka execution.
        
        - Creates appropriate ProblemSpace
        - Executes exploration
        - Returns solution for ROMA aggregation
        """
        logger.info(f"Solving sub-problem {subproblem.problem_id}: {subproblem.description[:50]}...")
        
        # Create appropriate problem space
        problem_space = self._create_problem_space(subproblem)
        
        # Execute exploration
        result = self.execution_engine.execute(
            task=subproblem.description,
            problem_space=problem_space
        )
        
        # Build solution
        solution = SubProblemSolution(
            problem_id=subproblem.problem_id,
            solution_content=result.summary,
            findings=result.final_state.accumulated_findings,
            confidence=result.confidence_score,
            verification_status="pending"
        )
        
        self.solution_history.append(solution)
        return solution
    
    def _create_problem_space(self, subproblem: SubProblem) -> ProblemSpace:
        """Create appropriate problem space for sub-problem."""
        space_type = subproblem.problem_space_type.lower()
        
        if space_type == "document":
            return DocumentSpace(
                representation=subproblem.description,
                operations=["read", "search", "extract"],
                constraints=subproblem.constraints
            )
        elif space_type == "codebase":
            return CodebaseSpace(
                representation=subproblem.description,
                operations=["find_symbol", "analyze", "extract_api"],
                constraints=subproblem.constraints
            )
        elif space_type == "database":
            return DatabaseSpace(
                representation=subproblem.description,
                operations=["query", "aggregate", "analyze"],
                constraints=subproblem.constraints
            )
        else:
            return AbstractSpace(
                representation=subproblem.description,
                operations=["explore", "analyze", "synthesize"],
                constraints=subproblem.constraints,
                domain=space_type
            )
    
    def critique_solution(
        self,
        solution: SubProblemSolution,
        critique_criteria: List[str]
    ) -> CritiqueResult:
        """
        Critique solution using Matryoshka (Red Team role).
        
        - Explores solution for vulnerabilities
        - Checks against criteria
        - Returns critique for improvement
        """
        logger.info(f"Critiquing solution {solution.problem_id}")
        
        # Create critique task
        critique_task = f"Critique solution against criteria: {', '.join(critique_criteria)}"
        
        # Use execution engine to explore for issues
        problem_space = AbstractSpace(
            representation=solution.solution_content,
            operations=["analyze", "verify", "find_vulnerabilities"],
            domain="critique"
        )
        
        result = self.execution_engine.execute(critique_task, problem_space)
        
        # Extract issues from findings
        issues = [
            f.content for f in result.final_state.accumulated_findings
            if f.category == FindingCategory.VULNERABILITY
        ]
        
        recommendations = [
            f.content for f in result.final_state.accumulated_findings
            if f.category == FindingCategory.INSIGHT
        ]
        
        return CritiqueResult(
            solution_id=solution.problem_id,
            issues_found=issues,
            severity_score=len(issues) / max(len(critique_criteria), 1),
            recommendations=recommendations,
            critique_confidence=result.confidence_score
        )
    
    def verify_solution(
        self,
        solution: SubProblemSolution,
        requirements: List[str]
    ) -> VerificationResult:
        """
        Verify solution meets requirements (Gold Team role).
        
        - Systematically checks each requirement
        - Provides verification evidence
        """
        logger.info(f"Verifying solution {solution.problem_id}")
        
        # Create verification task
        verify_task = f"Verify solution meets requirements: {', '.join(requirements)}"
        
        problem_space = AbstractSpace(
            representation=solution.solution_content,
            operations=["verify", "check", "validate"],
            domain="verification"
        )
        
        result = self.execution_engine.execute(verify_task, problem_space)
        
        # Determine which requirements passed/failed
        evidence = [f.content for f in result.final_state.accumulated_findings]
        
        # Simple heuristic: if confidence > 0.7, consider verified
        verified = result.confidence_score > 0.7
        
        failed_reqs = []
        if not verified:
            failed_reqs = requirements  # Simplified
        
        return VerificationResult(
            solution_id=solution.problem_id,
            verified=verified,
            evidence=evidence,
            failed_requirements=failed_reqs,
            verification_confidence=result.confidence_score
        )


# ============================================================================
# DECOMPOSITION WORKFLOW INTEGRATION
# ============================================================================

@dataclass
class TeamContext:
    """Context for team execution."""
    team_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    role: TeamRole = TeamRole.BLUE
    parent_execution_id: Optional[str] = None
    collaboration_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    """Generic solution structure."""
    solution_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VulnerabilityReport:
    """Report of vulnerabilities found."""
    target_id: str
    vulnerabilities: List[Dict[str, Any]] = field(default_factory=list)
    risk_score: float = 0.0
    attack_vectors_tested: List[str] = field(default_factory=list)


@dataclass
class SuccessCriterion:
    """Criterion for success verification."""
    criterion_id: str
    description: str
    required: bool = True
    verification_method: str = "automated"


@dataclass
class VerificationReport:
    """Report of verification results."""
    target_id: str
    criteria_met: List[str] = field(default_factory=list)
    criteria_failed: List[str] = field(default_factory=list)
    overall_pass: bool = False
    verification_details: Dict[str, Any] = field(default_factory=dict)


class MatryoshkaBlueTeamExecutor:
    """
    Matryoshka as Blue Team executor in decomposition workflow.
    
    Executes sub-problems with iterative exploration and learning.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig()
        self.execution_engine = MatryoshkaExecutionEngine(self.config)
    
    def execute_subproblem(
        self,
        subproblem: SubProblem,
        team_context: TeamContext
    ) -> ExecutionResult:
        """Execute as Blue Team member."""
        logger.info(f"Blue Team executing sub-problem {subproblem.problem_id}")
        
        # Create problem space
        problem_space = self._create_blue_team_space(subproblem)
        
        # Execute with Blue Team specific configuration
        result = self.execution_engine.execute(
            task=subproblem.description,
            problem_space=problem_space
        )
        
        return result
    
    def _create_blue_team_space(self, subproblem: SubProblem) -> ProblemSpace:
        """Create problem space optimized for Blue Team execution."""
        return AbstractSpace(
            representation=subproblem.description,
            operations=["solve", "implement", "optimize"],
            constraints=subproblem.constraints,
            domain="blue_team_execution"
        )


class MatryoshkaRedTeamAnalyzer:
    """
    Matryoshka as Red Team analyzer.
    
    Adversarially explores solutions for weaknesses.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig() if config else MatryoshkaExecutionConfig()
        self.config.exploration_mode = ExplorationMode.DEPTH_FIRST  # Thorough analysis
        self.execution_engine = MatryoshkaExecutionEngine(self.config)
    
    def analyze_for_vulnerabilities(
        self,
        solution: Solution,
        attack_vectors: List[str]
    ) -> VulnerabilityReport:
        """Adversarial analysis."""
        logger.info(f"Red Team analyzing solution {solution.solution_id}")
        
        vulnerabilities = []
        
        for vector in attack_vectors:
            task = f"Analyze for vulnerability using attack vector: {vector}"
            
            problem_space = AbstractSpace(
                representation=solution.content,
                operations=["attack", "probe", "exploit"],
                domain="adversarial"
            )
            
            result = self.execution_engine.execute(task, problem_space)
            
            for finding in result.final_state.accumulated_findings:
                if finding.category == FindingCategory.VULNERABILITY:
                    vulnerabilities.append({
                        "vector": vector,
                        "description": finding.content,
                        "confidence": finding.confidence
                    })
        
        risk_score = sum(v["confidence"] for v in vulnerabilities) / max(len(attack_vectors), 1)
        
        return VulnerabilityReport(
            target_id=solution.solution_id,
            vulnerabilities=vulnerabilities,
            risk_score=risk_score,
            attack_vectors_tested=attack_vectors
        )


class MatryoshkaGoldTeamVerifier:
    """
    Matryoshka as Gold Team verifier.
    
    Systematic verification against success criteria.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig() if config else MatryoshkaExecutionConfig()
        self.config.exploration_mode = ExplorationMode.BREADTH_FIRST  # Comprehensive coverage
        self.execution_engine = MatryoshkaExecutionEngine(self.config)
    
    def verify_against_criteria(
        self,
        solution: Solution,
        criteria: List[SuccessCriterion]
    ) -> VerificationReport:
        """Systematic verification."""
        logger.info(f"Gold Team verifying solution {solution.solution_id}")
        
        criteria_met = []
        criteria_failed = []
        verification_details = {}
        
        for criterion in criteria:
            task = f"Verify criterion: {criterion.description}"
            
            problem_space = AbstractSpace(
                representation=solution.content,
                operations=["verify", "validate", "check"],
                domain="verification"
            )
            
            result = self.execution_engine.execute(task, problem_space)
            
            # Determine if criterion is met based on confidence
            met = result.confidence_score > 0.7
            
            if met:
                criteria_met.append(criterion.criterion_id)
            else:
                criteria_failed.append(criterion.criterion_id)
            
            verification_details[criterion.criterion_id] = {
                "met": met,
                "confidence": result.confidence_score,
                "evidence": [f.content for f in result.final_state.accumulated_findings]
            }
        
        overall_pass = len(criteria_failed) == 0 or all(
            not c.required for c in criteria if c.criterion_id in criteria_failed
        )
        
        return VerificationReport(
            target_id=solution.solution_id,
            criteria_met=criteria_met,
            criteria_failed=criteria_failed,
            overall_pass=overall_pass,
            verification_details=verification_details
        )


# ============================================================================
# MDAP/MAKER INTEGRATION
# ============================================================================

@dataclass
class ExplorationStrategy:
    """A strategy for exploration."""
    strategy_id: str
    name: str
    description: str
    exploration_mode: ExplorationMode
    priority: int = 1
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VotedExplorationResult:
    """Result of voting-based exploration."""
    winning_strategy: ExplorationStrategy
    result: ExecutionResult
    vote_tally: Dict[str, int] = field(default_factory=dict)
    rounds: int = 0
    all_results: Dict[str, ExecutionResult] = field(default_factory=dict)


class MatryoshkaVotingExplorer:
    """
    Matryoshka exploration with MAKER voting consensus.
    
    Multiple exploration strategies voted on,
    first-to-ahead-by-k determines path.
    """
    
    def __init__(self, config: Optional[MatryoshkaExecutionConfig] = None):
        self.config = config or MatryoshkaExecutionConfig()
        self.execution_engine = MatryoshkaExecutionEngine(self.config)
    
    def explore_with_voting(
        self,
        task: str,
        problem_space: ProblemSpace,
        exploration_strategies: List[ExplorationStrategy],
        k_ahead: int = 3
    ) -> VotedExplorationResult:
        """
        Explore using MAKER voting to select best path.
        
        Args:
            task: The task to accomplish
            problem_space: Space to explore
            exploration_strategies: Different strategies to try
            k_ahead: First-to-ahead-by-k threshold
            
        Returns:
            VotedExplorationResult with winning strategy
        """
        logger.info(f"Starting voting exploration with {len(exploration_strategies)} strategies")
        
        if not exploration_strategies:
            raise ValueError("At least one exploration strategy required")
        
        # Initialize voting
        votes: Dict[str, int] = {s.strategy_id: 0 for s in exploration_strategies}
        results: Dict[str, ExecutionResult] = {}
        
        round_num = 0
        max_rounds = 10
        
        while round_num < max_rounds:
            round_num += 1
            logger.debug(f"Voting round {round_num}")
            
            # Execute each strategy
            for strategy in exploration_strategies:
                if strategy.strategy_id in results:
                    continue  # Already executed
                
                # Configure engine for this strategy
                self.config.exploration_mode = strategy.exploration_mode
                
                # Execute
                result = self.execution_engine.execute(task, problem_space)
                results[strategy.strategy_id] = result
                
                # Award votes based on confidence
                votes[strategy.strategy_id] = int(result.confidence_score * 10)
            
            # Check for winner (first to ahead by k)
            sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_votes) >= 2:
                leader, leader_votes = sorted_votes[0]
                second, second_votes = sorted_votes[1]
                
                if leader_votes - second_votes >= k_ahead:
                    logger.info(f"Strategy {leader} wins with {leader_votes} votes")
                    winning_strategy = next(
                        s for s in exploration_strategies if s.strategy_id == leader
                    )
                    return VotedExplorationResult(
                        winning_strategy=winning_strategy,
                        result=results[leader],
                        vote_tally=votes,
                        rounds=round_num,
                        all_results=results
                    )
        
        # No clear winner, pick highest votes
        winner_id = max(votes, key=votes.get)
        winning_strategy = next(s for s in exploration_strategies if s.strategy_id == winner_id)
        
        return VotedExplorationResult(
            winning_strategy=winning_strategy,
            result=results[winner_id],
            vote_tally=votes,
            rounds=round_num,
            all_results=results
        )


# ============================================================================
# FACTORY FUNCTIONS
# ============================================================================

def create_matryoshka_solver_for_roma(
    config: Optional[MatryoshkaExecutionConfig] = None
) -> ROMAMatryoshkaSolver:
    """Create Matryoshka solver for ROMA integration."""
    return ROMAMatryoshkaSolver(config)


def create_matryoshka_executor_for_decomposition(
    team_role: str = "blue",
    config: Optional[MatryoshkaExecutionConfig] = None
) -> Union[MatryoshkaBlueTeamExecutor, MatryoshkaRedTeamAnalyzer, MatryoshkaGoldTeamVerifier]:
    """Create Matryoshka executor for decomposition workflow."""
    role = TeamRole(team_role.lower())
    
    if role == TeamRole.BLUE:
        return MatryoshkaBlueTeamExecutor(config)
    elif role == TeamRole.RED:
        return MatryoshkaRedTeamAnalyzer(config)
    elif role == TeamRole.GOLD:
        return MatryoshkaGoldTeamVerifier(config)
    else:
        raise ValueError(f"Unknown team role: {team_role}")


def create_matryoshka_voting_explorer(
    config: Optional[MatryoshkaExecutionConfig] = None
) -> MatryoshkaVotingExplorer:
    """Create Matryoshka explorer with MAKER voting."""
    return MatryoshkaVotingExplorer(config)


def create_problem_space(
    space_type: str,
    representation: Any,
    **kwargs
) -> ProblemSpace:
    """
    Factory function to create appropriate problem space.
    
    Args:
        space_type: Type of space (document, codebase, database, config, abstract)
        representation: How the space is represented
        **kwargs: Additional arguments for specific space types
        
    Returns:
        Appropriate ProblemSpace subclass
    """
    space_type = space_type.lower()
    
    if space_type == "document":
        return DocumentSpace(
            representation=representation,
            document_path=kwargs.get("document_path"),
            document_content=kwargs.get("document_content"),
            document_type=kwargs.get("document_type", "text"),
            operations=kwargs.get("operations", ["read", "search", "extract"]),
            constraints=kwargs.get("constraints", [])
        )
    elif space_type == "codebase":
        return CodebaseSpace(
            representation=representation,
            repository_path=kwargs.get("repository_path"),
            language=kwargs.get("language", "python"),
            entry_points=kwargs.get("entry_points", []),
            operations=kwargs.get("operations", ["find_symbol", "analyze"]),
            constraints=kwargs.get("constraints", [])
        )
    elif space_type == "database":
        return DatabaseSpace(
            representation=representation,
            connection_string=kwargs.get("connection_string"),
            schema=kwargs.get("schema"),
            dialect=kwargs.get("dialect", "sql"),
            operations=kwargs.get("operations", ["query", "aggregate"]),
            constraints=kwargs.get("constraints", [])
        )
    elif space_type == "config":
        return ConfigSpace(
            representation=representation,
            config_format=kwargs.get("config_format", "yaml"),
            schema_definition=kwargs.get("schema_definition"),
            operations=kwargs.get("operations", ["validate", "check"]),
            constraints=kwargs.get("constraints", [])
        )
    else:
        return AbstractSpace(
            representation=representation,
            domain=space_type,
            operations=kwargs.get("operations", ["explore", "analyze"]),
            constraints=kwargs.get("constraints", []),
            symbolic_representation=kwargs.get("symbolic_representation")
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def merge_execution_results(results: List[ExecutionResult]) -> ExecutionResult:
    """Merge multiple execution results into one."""
    if not results:
        return ExecutionResult(success=False, final_state=ExecutionState(iteration=0))
    
    # Use the first result as base
    base = results[0]
    
    # Merge findings
    all_findings = []
    for r in results:
        all_findings.extend(r.final_state.accumulated_findings)
    
    # Create merged state
    merged_state = ExecutionState(
        iteration=sum(r.final_state.iteration for r in results),
        accumulated_findings=all_findings,
        failed_attempts=sum((r.final_state.failed_attempts for r in results), [])
    )
    
    # Calculate aggregate confidence
    avg_confidence = sum(r.confidence_score for r in results) / len(results)
    
    return ExecutionResult(
        success=any(r.success for r in results),
        final_state=merged_state,
        summary=f"Merged {len(results)} execution results",
        execution_time_ms=sum(r.execution_time_ms for r in results),
        iterations_completed=sum(r.iterations_completed for r in results),
        confidence_score=avg_confidence
    )


def create_execution_pipeline(
    stages: List[Tuple[str, ProblemSpace]],
    config: Optional[MatryoshkaExecutionConfig] = None
) -> Callable[[], List[ExecutionResult]]:
    """
    Create a multi-stage execution pipeline.
    
    Args:
        stages: List of (task, problem_space) tuples
        config: Optional execution configuration
        
    Returns:
        Function that executes the pipeline
    """
    engine = MatryoshkaExecutionEngine(config or MatryoshkaExecutionConfig())
    
    def execute_pipeline() -> List[ExecutionResult]:
        results = []
        current_state = None
        
        for task, space in stages:
            result = engine.execute(task, space, current_state)
            results.append(result)
            current_state = result.final_state
        
        return results
    
    return execute_pipeline


# ============================================================================
# MAIN ENTRY POINT (for testing)
# ============================================================================

if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create configuration
    config = MatryoshkaExecutionConfig(
        max_iterations=5,
        exploration_mode=ExplorationMode.ADAPTIVE,
        report_intermediate_findings=True
    )
    
    # Create execution engine
    engine = MatryoshkaExecutionEngine(config)
    
    # Create a problem space
    space = AbstractSpace(
        representation="Example problem space for demonstration",
        operations=["explore", "analyze", "synthesize"],
        domain="example"
    )
    
    # Execute
    result = engine.execute(
        task="Explore and analyze the problem space",
        problem_space=space
    )
    
    print(f"Execution successful: {result.success}")
    print(f"Iterations: {result.iterations_completed}")
    print(f"Findings: {len(result.final_state.accumulated_findings)}")
    print(f"Confidence: {result.confidence_score:.2f}")
    print(f"\nSummary:\n{result.summary}")
