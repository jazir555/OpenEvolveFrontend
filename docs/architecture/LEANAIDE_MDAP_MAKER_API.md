# LeanAide MDAP/MAKER API Reference

> **STATUS: implemented** (`MDAPOrchestrator`, `MDAPTask`, `MDAPStep`, `MDAPRunResult`, `MDAPStepResult`, `RedFlagger`, `RedFlagRules` in `engines/other/mdap_engine.py`; `ROMAMDAPMakerEngine`/`ROMAMDAPMakerConfig`/`ROMARedFlagger` in `engines/other/roma_mdap_maker_engine.py`; `MAKERWorkflowConfig` in `engines/other/openevolve_maker_integration.py` and `integrations/leanaide/leanaide_mdap_workflow.py`; MAKER voting in `integrations/leanaide/leanaide_maker.py` — `LeanTacticVoter`, `RandomVoter`, `HeuristicVoter`, `EvolutionaryVoter`, `LeanRedFlagRules`).
>
> **Integration backend:** the HTTP surface lives in `services/openevolve-api` (FastAPI, port 8000): `api/settings.py` serves `/api/settings/mdap-maker` and `/api/settings/roma-mdap-maker`; `api/mdap_maker.py` defines `/mdap-maker/status`, `/mdap-maker/solve`, `/roma-mdap-maker/status`, `/roma-mdap-maker/solve` (router present, not yet included in `main.py`). The BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts` forwards `/api/*` verbatim to this service (default `http://localhost:8000`).
>
> **Last reconciled: 2026-08-20**

**Document Version:** 1.0
**Date:** 2025-12-30
**Project:** OpenEvolve Frontend - LeanAide MDAP/MAKER Integration

---

## Table of Contents

1. [Core MDAP API](#1-core-mdap-api)
2. [MAKER API](#2-maker-api)
3. [ROMA-MDAP-MAKER API](#3-roma-mdap-maker-api)
4. [Workflow Integration API](#4-workflow-integration-api)
5. [Red-Flagging API](#5-red-flagging-api)
6. [Voting Strategies API](#6-voting-strategies-api)
7. [Configuration API](#7-configuration-api)
8. [Lean 4 Integration API](#8-lean-4-integration-api)

---

## 1. Core MDAP API

### 1.1 MDAPOrchestrator

Main orchestrator for multi-agent proof generation.

```python
class MDAPOrchestrator:
    """
    Orchestrates multi-agent proof generation using MDAP framework.

    Attributes:
        config (MDAPConfig): MDAP configuration
        model_config (ModelConfig): LLM configuration
        cache (MDAPCache): Response cache
    """

    def __init__(
        self,
        config: MDAPConfig,
        model_config: ModelConfig,
        cache_enabled: bool = True
    ):
        """
        Initialize MDAP orchestrator.

        Args:
            config: MDAP configuration parameters
            model_config: LLM model configuration
            cache_enabled: Enable response caching
        """
        pass

    async def run_task_async(
        self,
        task: MDAPTask
    ) -> MDAPRunResult:
        """
        Execute MDAP task asynchronously.

        Args:
            task: MDAP task to execute

        Returns:
            MDAPRunResult containing step results and metrics
        """
        pass

    def run_task(
        self,
        task: MDAPTask
    ) -> MDAPRunResult:
        """
        Execute MDAP task synchronously.

        Args:
            task: MDAP task to execute

        Returns:
            MDAPRunResult containing step results and metrics
        """
        pass
```

**Usage Example:**
```python
from mdap_engine import MDAPOrchestrator, MDAPConfig
from workflow_structures import ModelConfig

config = MDAPConfig(k_min=3, k_max=5)
model_config = ModelConfig(
    provider="openai",
    model="gpt-4o",
    api_key="your-key"
)

orchestrator = MDAPOrchestrator(config, model_config)
result = await orchestrator.run_task_async(task)
```

### 1.2 MDAPStep

Represents a single step in an MDAP task.

```python
@dataclass
class MDAPStep:
    """
    A single step in MDAP execution.

    Attributes:
        step_id (str): Unique step identifier
        prompt (str): Prompt for this step
        expected_schema (dict): JSON schema for output validation
        task_type (str): Type of task ("theorem_proving", "tactic_generation", etc.)
        priority (int): Step priority (higher = earlier execution)
        system_prompt (str): Override system prompt
        temperature_override (float): Override temperature
        max_tokens_override (int): Override max tokens
        stop_sequences (list): Override stop sequences
        metadata (dict): Additional metadata
    """
    step_id: str
    prompt: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Usage Example:**
```python
step = MDAPStep(
    step_id="proof_generation",
    prompt="Prove: ∀ n : Nat, n + 0 = n",
    task_type="theorem_proving",
    temperature_override=0.1,
    max_tokens_override=500
)
```

### 1.3 MDAPTask

Represents a complete MDAP task with multiple steps.

```python
@dataclass
class MDAPTask:
    """
    A multi-step MDAP task.

    Attributes:
        task_id (str): Unique task identifier
        description (str): Task description
        steps (list): List of MDAPStep objects
        max_retries (int): Maximum retries per step
        target_success_rate (float): Target success rate (0-1)
        metadata (dict): Additional metadata
    """
    task_id: str
    description: str
    steps: List[MDAPStep]
    max_retries: int = 2
    target_success_rate: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)
```

**Usage Example:**
```python
task = MDAPTask(
    task_id="add_zero_proof",
    description="Prove addition with zero",
    steps=[
        MDAPStep(step_id="analyze", prompt="Analyze the theorem structure"),
        MDAPStep(step_id="prove", prompt="Generate the proof"),
        MDAPStep(step_id="verify", prompt="Verify the proof")
    ],
    max_retries=3
)
```

### 1.4 MDAPRunResult

Result of MDAP task execution.

```python
@dataclass
class MDAPRunResult:
    """
    Result of MDAP task execution.

    Attributes:
        task_id (str): Task identifier
        step_results (dict): Step results by step_id
        metrics (dict): Execution metrics
        success (bool): Overall success status
        proof (str): Generated proof (if successful)
        errors (list): List of errors
    """
    task_id: str
    step_results: Dict[str, MDAPStepResult]
    metrics: Dict[str, Any]
    success: bool = False
    proof: Optional[str] = None
    errors: List[str] = field(default_factory=list)
```

**Usage Example:**
```python
result = await orchestrator.run_task_async(task)

if result.success:
    print(f"Proof: {result.proof}")
    print(f"Metrics: {result.metrics}")
else:
    for error in result.errors:
        print(f"Error: {error}")
```

### 1.5 MDAPStepResult

Result of individual MDAP step execution.

```python
@dataclass
class MDAPStepResult:
    """
    Result of MDAP step execution.

    Attributes:
        step_id (str): Step identifier
        vote_result (MDAPVoteResult): Voting result
        status (str): Step status ("success", "failed", "retrying")
        retries (int): Number of retries attempted
    """
    step_id: str
    vote_result: MDAPVoteResult
    status: str
    retries: int
```

### 1.6 MDAPVoteResult

Result of voting aggregation.

```python
@dataclass
class MDAPVoteResult:
    """
    Result of voting aggregation.

    Attributes:
        winner (any): Winning candidate
        votes (dict): Vote counts by candidate
        red_flags (int): Number of red-flagged candidates
        confidence (float): Confidence in winner
        attempts (int): Number of voting attempts
        duration_seconds (float): Time taken for voting
        flagged_reasons (list): Reasons for red flags
        errors (list): List of errors
    """
    winner: Optional[Any]
    votes: Dict[str, int]
    red_flags: int
    confidence: float
    attempts: int
    duration_seconds: float
    flagged_reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
```

---

## 2. MAKER API

### 2.1 MAKER Workflow Integration

```python
def generate_solution_with_maker_v2(
    sub_problem: SubProblem,
    team: Team,
    formatted_user_prompt: str,
    system_message: str,
    workflow_state: WorkflowState,
    emit_info: Optional[callable] = None,
    emit_success: Optional[callable] = None,
    emit_warning: Optional[callable] = None
) -> Optional[str]:
    """
    Generate solution for sub-problem using MAKER framework.

    Args:
        sub_problem: The sub-problem to solve
        team: The team assigned to solve
        formatted_user_prompt: Formatted prompt template
        system_message: System prompt for LLM
        workflow_state: Current workflow state
        emit_info: Optional info logging function
        emit_success: Optional success logging function
        emit_warning: Optional warning logging function

    Returns:
        Solution string if successful, None otherwise
    """
    pass
```

**Usage Example:**
```python
from maker_workflow_integration import generate_solution_with_maker_v2

solution = await generate_solution_with_maker_v2(
    sub_problem=sub_problem,
    team=team,
    formatted_user_prompt=prompt,
    system_message=system_msg,
    workflow_state=state
)
```

### 2.2 MAKER Workflow Configuration

```python
@dataclass
class MAKERWorkflowConfig:
    """
    Configuration for MAKER workflow integration.

    Attributes:
        mode (MAKERMode): MAKER execution mode
        k_ahead (int): First-K-ahead threshold
        max_depth (int): Maximum decomposition depth
        enable_red_flagging (bool): Enable red-flagging
        max_token_length (int): Maximum token length
    """
    mode: MAKERMode = MAKERMode.SEQUENTIAL
    k_ahead: int = 3
    max_depth: int = 5
    enable_red_flagging: bool = True
    max_token_length: int = 750
```

### 2.3 MAKER Mode

```python
class MAKERMode(Enum):
    """MAKER execution modes"""
    SEQUENTIAL = "sequential"  # Step-by-step execution
    PARALLEL = "parallel"      # Parallel execution
    RECURSIVE = "recursive"    # Recursive decomposition
    HYBRID = "hybrid"          # Hybrid approach
```

**Usage Example:**
```python
from maker_workflow_integration import MAKERMode

config = MAKERWorkflowConfig(
    mode=MAKERMode.RECURSIVE,
    k_ahead=3,
    max_depth=5
)
```

---

## 3. ROMA-MDAP-MAKER API

### 3.1 ROMAMDAPMakerEngine

Main engine for ROMA-MDAP-MAKER integration.

```python
class ROMAMDAPMakerEngine:
    """
    ROMA-MDAP-MAKER integration engine.

    Combines ROMA decomposition with MAKER error correction.
    """

    def __init__(
        self,
        config: ROMAMDAPMakerConfig,
        api_key: Optional[str] = None
    ):
        """
        Initialize ROMA-MDAP-MAKER engine.

        Args:
            config: ROMA-MDAP-MAKER configuration
            api_key: Optional API key for LLM provider
        """
        pass

    async def solve_with_romamdap(
        self,
        theorem: str,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Solve theorem using ROMA-MDAP-MAKER.

        Args:
            theorem: Theorem statement to prove
            context: Optional context (imports, lemmas)

        Returns:
            Dictionary with proof results and metrics
        """
        pass

    def decompose_theorem(
        self,
        theorem: str
    ) -> Dict[str, Any]:
        """
        Decompose theorem using ROMA.

        Args:
            theorem: Theorem to decompose

        Returns:
            ROMA DAG structure
        """
        pass
```

**Usage Example:**
```python
from roma_mdap_maker_engine import ROMAMDAPMakerEngine, ROMAMDAPMakerConfig

config = ROMAMDAPMakerConfig(
    mdap_k_ahead=3,
    roma_max_depth_solving=2
)

engine = ROMAMDAPMakerEngine(config, api_key="your-key")
result = await engine.solve_with_romamdap(
    theorem="theorem mul_comm (a b : Nat) : a * b = b * a"
)
```

### 3.2 ROMAMDAPMakerConfig

Configuration for ROMA-MDAP-MAKER.

```python
@dataclass
class ROMAMDAPMakerConfig:
    """
    Configuration for ROMA-MDAP-MAKER integration.

    ROMA Settings:
        roma_max_depth_analysis (int): Max decomposition depth for analysis
        roma_max_depth_solving (int): Max decomposition depth for solving
        roma_execution_mode (str): "recursive" or "event_driven"
        roma_enable_checkpoints (bool): Enable checkpointing
        roma_enable_logging (bool): Enable detailed logging

    MDAP/MAKER Settings:
        mdap_enabled (bool): Enable MDAP voting
        mdap_k_ahead (int): First-K-ahead threshold
        mdap_max_samples (int): Max samples per voting round
        mdap_enable_red_flagging (bool): Enable red-flagging
        mdap_max_token_length (int): Max response tokens
        mdap_min_confidence (float): Min agent confidence

    Integration Settings:
        apply_maker_to_roma_atomic (bool): Apply MAKER to atomic tasks
        apply_maker_to_roma_planning (bool): Apply to planning
        aggregate_maker_results (bool): Aggregate voted results
        enable_hierarchical_voting (bool): Enable hierarchical voting
        enable_adaptive_k (bool): Enable adaptive k-selection

    Caching:
        enable_caching (bool): Enable caching
        cache_ttl_seconds (int): Cache TTL
        cache_max_size (int): Max cache size

    Fault Tolerance:
        max_retries (int): Max retries
        timeout_seconds (int): Timeout
        fallback_policy (str): Fallback policy

    Provider Settings:
        provider (str): LLM provider
        api_key (str): API key
        model (str): Model name
        temperature (float): Temperature

    Metadata:
        metadata (dict): Additional metadata
    """
    # ROMA settings
    roma_max_depth_analysis: int = 3
    roma_max_depth_solving: int = 2
    roma_execution_mode: str = "recursive"
    roma_enable_checkpoints: bool = False
    roma_enable_logging: bool = False

    # MDAP/MAKER settings
    mdap_enabled: bool = True
    mdap_k_ahead: int = 3
    mdap_max_samples: int = 100
    mdap_enable_red_flagging: bool = True
    mdap_max_token_length: int = 750
    mdap_min_confidence: float = 0.2

    # Integration settings
    apply_maker_to_roma_atomic: bool = True
    apply_maker_to_roma_planning: bool = False
    aggregate_maker_results: bool = True
    enable_hierarchical_voting: bool = True
    enable_adaptive_k: bool = True

    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size: int = 10000

    # Fault tolerance
    max_retries: int = 3
    timeout_seconds: int = 300
    fallback_policy: str = "escalate_then_best_effort"

    # Provider settings
    provider: str = "openai"
    api_key: Optional[str] = None
    model: str = "gpt-4o-mini"
    temperature: float = 0.1

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 4. Workflow Integration API

### 4.1 Workflow Configuration Builders

```python
def build_maker_config_from_workflow(
    workflow_state: WorkflowState,
    sub_problem: SubProblem
) -> MAKERWorkflowConfig:
    """
    Build MAKER configuration from workflow state and sub-problem.

    Args:
        workflow_state: Current workflow state
        sub_problem: Sub-problem to solve

    Returns:
        MAKERWorkflowConfig object
    """
    pass

def resolve_maker_enabled(
    workflow_state: WorkflowState,
    sub_problem: SubProblem
) -> bool:
    """
    Determine if MAKER should be enabled for this sub-problem.

    Args:
        workflow_state: Current workflow state
        sub_problem: Sub-problem to check

    Returns:
        True if MAKER should be used, False otherwise
    """
    pass
```

**Usage Example:**
```python
from maker_workflow_integration import (
    build_maker_config_from_workflow,
    resolve_maker_enabled
)

enabled = resolve_maker_enabled(state, sub_problem)
if enabled:
    config = build_maker_config_from_workflow(state, sub_problem)
```

### 4.2 SubProblem Integration

```python
@dataclass
class SubProblem:
    """
    Sub-problem for workflow integration.

    Attributes:
        id (str): Unique identifier
        title (str): Problem title
        description (str): Problem description
        dependencies (list): List of dependency IDs
        estimated_effort (int): Estimated effort (1-100)
        metadata (dict): Additional metadata
    """
    id: str
    title: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    estimated_effort: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 5. Red-Flagging API

### 5.1 RedFlagger

Main red-flagging class.

```python
class RedFlagger:
    """
    Red-flags low-quality or invalid responses.

    Attributes:
        rules (RedFlagRules): Red-flagging rules
    """

    def __init__(self, rules: RedFlagRules):
        """
        Initialize red-flagger.

        Args:
            rules: Red-flagging rules
        """
        pass

    def is_flagged(
        self,
        raw_text: str,
        candidate: Any,
        schema: Optional[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Check if response should be red-flagged.

        Args:
            raw_text: Raw response text
            candidate: Parsed candidate
            schema: Optional JSON schema

        Returns:
            Tuple of (is_flagged, list_of_reasons)
        """
        pass
```

**Usage Example:**
```python
from mdap_engine import RedFlagger, RedFlagRules

rules = RedFlagRules(
    max_tokens=750,
    min_confidence=0.3,
    blocked_patterns=["ERROR", "sorry"]
)

flagger = RedFlagger(rules)
is_flagged, reasons = flagger.is_flagged(response_text, candidate, schema)

if is_flagged:
    print(f"Flagged: {reasons}")
```

### 5.2 RedFlagRules

Red-flagging configuration.

```python
@dataclass
class RedFlagRules:
    """
    Red-flagging rules.

    Attributes:
        max_tokens (int): Maximum tokens allowed
        max_characters (int): Maximum characters allowed
        blocked_patterns (list): Blocked regex patterns
        min_confidence (float): Minimum confidence
        require_schema_match (bool): Require schema validation
    """
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    blocked_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.2
    require_schema_match: bool = True
```

### 5.3 ROMARedFlagger

Extended red-flagger for ROMA-MDAP-MAKER.

```python
class ROMARedFlagger(RedFlagger):
    """
    Enhanced red-flagging for ROMA-MDAP-MAKER.
    """

    def check_roma_decomposition_red_flags(
        self,
        romadag: Dict[str, Any]
    ) -> List[str]:
        """
        Check ROMA decomposition for structural issues.

        Args:
            romadag: ROMA DAG structure

        Returns:
            List of red flag reasons (empty if no flags)
        """
        pass

    def check_roma_planning_red_flags(
        self,
        subtask: Dict[str, Any]
    ) -> List[str]:
        """
        Check ROMA planned subtask for quality issues.

        Args:
            subtask: ROMA subtask structure

        Returns:
            List of red flag reasons
        """
        pass
```

### 5.4 ROMARedFlagRules

Extended rules for ROMA.

```python
@dataclass
class ROMARedFlagRules(RedFlagRules):
    """
    Enhanced red-flag rules for ROMA-MDAP-MAKER.

    Additional Attributes:
        max_roma_depth (int): Maximum ROMA decomposition depth
        max_dag_nodes (int): Maximum DAG nodes
        allow_cyclic_dependencies (bool): Allow cycles
        min_subtask_description_length (int): Min description length
        max_balance_ratio (float): Max balance ratio
    """
    max_roma_depth: int = 5
    max_dag_nodes: int = 1000
    allow_cyclic_dependencies: bool = False
    min_subtask_description_length: int = 20
    max_balance_ratio: float = 10.0
```

---

## 6. Voting Strategies API

### 6.1 HierarchicalVotingStrategy

Hierarchical voting for ROMA-MDAP-MAKER.

```python
class HierarchicalVotingStrategy:
    """
    Hierarchical voting across ROMA decomposition levels.

    Aggregates votes from atomic tasks up through decomposition levels.
    """

    def __init__(
        self,
        base_strategy: str = "first_k_ahead",
        k_ahead: int = 3
    ):
        """
        Initialize hierarchical voting.

        Args:
            base_strategy: Base voting strategy
            k_ahead: First-K-ahead threshold
        """
        pass

    def aggregate_hierarchical(
        self,
        dag: Dict[str, Any],
        results: Dict[str, MDAPVoteResult]
    ) -> Dict[str, Any]:
        """
        Aggregate votes hierarchically.

        Args:
            dag: ROMA DAG structure
            results: Results by node ID

        Returns:
            Aggregated results with hierarchy
        """
        pass
```

### 6.2 AdaptiveKSelector

Adaptive k-selection for MAKER.

```python
class AdaptiveKSelector:
    """
    Adaptive k-selection for MAKER voting.

    Adjusts k based on confidence, difficulty, and performance.
    """

    def __init__(
        self,
        k_min: int = 2,
        k_max: int = 8,
        confidence_threshold: float = 0.8
    ):
        """
        Initialize adaptive k-selector.

        Args:
            k_min: Minimum k value
            k_max: Maximum k value
            confidence_threshold: Confidence threshold for early stopping
        """
        pass

    def select_k(
        self,
        step: MDAPStep,
        history: List[MDAPStepResult]
    ) -> int:
        """
        Select k for current step.

        Args:
            step: Current step
            history: Previous step results

        Returns:
            Selected k value
        """
        pass

    def should_stop_early(
        self,
        votes: Dict[str, int],
        k_ahead: int
    ) -> bool:
        """
        Check if voting should stop early.

        Args:
            votes: Current vote counts
            k_ahead: K-ahead threshold

        Returns:
            True if should stop, False otherwise
        """
        pass
```

---

## 7. Configuration API

### 7.1 MDAPConfig

```python
@dataclass
class MDAPConfig:
    """
    MDAP configuration.

    Voting Parameters:
        k_min (int): Minimum agents for consensus
        k_max (int): Maximum agents to run
        max_votes_per_step (int): Maximum voting rounds

    Execution Parameters:
        timeout_seconds (int): Timeout per step

    Red-Flagging:
        red_flag_rules (RedFlagRules): Red-flagging rules

    Fallback:
        fallback_policy (str): Fallback policy

    Caching:
        cache_ttl_seconds (int): Cache TTL
        cache_max_size (int): Max cache size
    """
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 50
    timeout_seconds: int = 60
    red_flag_rules: RedFlagRules = field(default_factory=RedFlagRules)
    fallback_policy: str = "escalate_then_best_effort"
    cache_ttl_seconds: Optional[int] = None
    cache_max_size: int = 5000
```

### 7.2 MDAPCache

Caching for MDAP responses.

```python
class MDAPCache:
    """
    Cache for MDAP responses.

    Attributes:
        max_size (int): Maximum cache size
        ttl_seconds (int): Time-to-live in seconds
    """

    def __init__(self, max_size: int, ttl_seconds: int):
        """
        Initialize cache.

        Args:
            max_size: Maximum cache size
            ttl_seconds: Time-to-live in seconds
        """
        pass

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    def set(self, key: str, value: Any) -> None:
        """Set value in cache."""
        pass

    def clear(self) -> None:
        """Clear cache."""
        pass
```

**Usage Example:**
```python
from mdap_engine import MDAPCache

cache = MDAPCache(max_size=1000, ttl_seconds=3600)
cache.set("key", {"value": "data"})
result = cache.get("key")
```

---

## 8. Lean 4 Integration API

### 8.1 Lean4Verifier

Lean 4 proof verification.

```python
class Lean4Verifier:
    """
    Lean 4 proof verifier.

    Verifies Lean 4 proofs against theorems.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:7654",
        timeout: int = 300
    ):
        """
        Initialize verifier.

        Args:
            server_url: Lean 4 server URL
            timeout: Verification timeout
        """
        pass

    async def verify(
        self,
        proof: str,
        theorem: Optional[str] = None
    ) -> Lean4VerificationResult:
        """
        Verify proof against theorem.

        Args:
            proof: Lean 4 proof code
            theorem: Optional theorem statement

        Returns:
            Lean4VerificationResult
        """
        pass
```

### 8.2 Lean4VerificationResult

```python
@dataclass
class Lean4VerificationResult:
    """
    Result of Lean 4 verification.

    Attributes:
        is_valid (bool): Whether proof is valid
        errors (list): List of errors
        warnings (list): List of warnings
        duration_seconds (float): Verification time
        lean_output (str): Lean 4 output
    """
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    lean_output: str = ""
```

### 8.3 Lean4ServerConfig

```python
@dataclass
class Lean4ServerConfig:
    """
    Lean 4 server configuration.

    Attributes:
        host (str): Server host
        port (int): Server port
        timeout (int): Timeout
        persistent (bool): Keep server running
        enable_simulation_fallback (bool): Use simulation fallback
        worker_processes (int): Number of worker processes
    """
    host: str = "localhost"
    port: int = 7654
    timeout: int = 300
    persistent: bool = True
    enable_simulation_fallback: bool = True
    worker_processes: int = 4
```

---

## Appendix A: Type Definitions

### A.1 ModelConfig

```python
@dataclass
class ModelConfig:
    """
    LLM model configuration.

    Attributes:
        provider (str): LLM provider ("openai", "anthropic", etc.)
        model (str): Model name
        api_key (str): API key
        temperature (float): Temperature
        max_tokens (int): Max tokens
        top_p (float): Top-p sampling
        frequency_penalty (float): Frequency penalty
        presence_penalty (float): Presence penalty
    """
    provider: str
    model: str
    api_key: str
    temperature: float = 0.1
    max_tokens: int = 1000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
```

### A.2 Team

```python
@dataclass
class Team:
    """
    Team configuration.

    Attributes:
        team_id (str): Team identifier
        name (str): Team name
        model_config (ModelConfig): Model configuration
        agents (list): List of agent configurations
        metadata (dict): Additional metadata
    """
    team_id: str
    name: str
    model_config: ModelConfig
    agents: List[AgentConfig] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### A.3 WorkflowState

```python
@dataclass
class WorkflowState:
    """
    Workflow state.

    Attributes:
        maker_enabled (bool): MAKER enabled flag
        maker_config (dict): MAKER configuration
        mdap_config (dict): MDAP configuration
        metadata (dict): Additional metadata
    """
    maker_enabled: Optional[bool] = None
    maker_config: Dict[str, Any] = field(default_factory=dict)
    mdap_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

**Document End**

For more information, see:
- `LEANAIDE_MDAP_MAKER_GUIDE.md` - Complete usage guide
- `LEANAIDE_MDAP_MAKER_EXAMPLES.md` - Real-world examples
- `LEANAIDE_MDAP_ARCHITECTURE.md` - Architecture diagrams
