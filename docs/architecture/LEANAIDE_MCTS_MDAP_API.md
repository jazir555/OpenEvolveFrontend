# LeanAide MCTS-MDAP API Reference

> **STATUS: implemented** (see `integrations/leanaide/leanaide_mcts_mdap.py` — `MDAPMCTS`, `MDAPMCTSConfig`, `MDAPMCTSResult`, `MCTSSelectionPolicy`, `LeanAIDEMCTSMdap`; plus `integrations/leanaide/leanaide_mcts_mdap_complete.py` and `engines/mcts_mdap/mdap_maker_mcts_unified.py`).
>
> **Integration backend:** these are library modules; they are not exposed as HTTP routes. The distribution's real backend is `services/openevolve-api` (FastAPI, port 8000) which mounts all `/api/*` route groups, fronted by the BubbleLab Hono proxy at `apps/bubblelab-api/src/routes/openevolve.ts`.
>
> **Last reconciled: 2026-08-20**

## Table of Contents

1. [Core Classes](#core-classes)
2. [Data Structures](#data-structures)
3. [Configuration](#configuration)
4. [Search Functions](#search-functions)
5. [Utility Functions](#utility-functions)
6. [Type Definitions](#type-definitions)
7. [Error Handling](#error-handling)
8. [Integration API](#integration-api)

---

## Core Classes

### MDAPMCTS

The main orchestrator combining MCTS with MDAP voting.

```python
class MDAPMCTS:
    """
    MCTS enhanced with MDAP voting for Lean 4 theorem proving.

    This class integrates Monte Carlo Tree Search with Multi-Agent
    Decomposition (MDAP) to provide robust tactic selection through
    agent voting and red-flagging.

    Attributes:
        config: MCTS configuration
        mdap_config: MDAP configuration
        team: Agent team for voting
        selection: MCTS selection phase
        expansion: MCTS expansion phase (MDAP-enhanced)
        simulation: MCTS simulation phase (MAKER-enhanced)
        backpropagation: MCTS backpropagation phase
        metrics: Search metrics
    """

    def __init__(
        self,
        config: MCTSConfig,
        mdap_config: Optional[MDAPConfig] = None,
        team: Optional[Team] = None
    ):
        """
        Initialize MDAP-MCTS searcher.

        Args:
            config: MCTS configuration
            mdap_config: Optional MDAP configuration for voting
            team: Optional agent team for voting

        Raises:
            ValueError: If config is None
        """
```

#### Methods

##### search()

```python
def search(
    self,
    initial_state: ProofState,
    time_limit: Optional[float] = None,
    iteration_limit: Optional[int] = None,
    progress_callback: Optional[Callable[[int, MCTSNode], None]] = None
) -> MCTSResult:
    """
    Perform MDAP-MCTS search from initial state.

    Args:
        initial_state: Starting proof state
        time_limit: Optional time limit in seconds
        iteration_limit: Optional iteration limit
        progress_callback: Optional callback for progress updates

    Returns:
        MCTSResult: Search result including best proof found

    Example:
        ```python
        mcts_mdap = MDAPMCTS(config, mdap_config, team)
        state = ProofState(goals=["forall (a b : Nat), a + b = b + a"])

        result = mcts_mdap.search(
            state,
            time_limit=60.0,
            progress_callback=lambda i, n: print(f"Iteration {i}")
        )

        if result.success:
            print(f"Found proof in {result.search_iterations} iterations")
        ```
    """
```

##### search_with_maker()

```python
def search_with_maker(
    self,
    initial_state: ProofState,
    maker_engine: MAKEREngine,
    time_limit: Optional[float] = None
) -> MCTSResult:
    """
    Perform MCTS search with MAKER-enhanced simulation.

    Args:
        initial_state: Starting proof state
        maker_engine: MAKER engine for simulation voting
        time_limit: Optional time limit in seconds

    Returns:
        MCTSResult: Search result

    Example:
        ```python
        maker_engine = MAKEREngine(team, k_ahead=3)
        result = mcts_mdap.search_with_maker(state, maker_engine, time_limit=120.0)
        ```
    """
```

### MDAPMCTSNode

Enhanced MCTS node with voting metadata.

```python
class MDAPMCTSNode(MCTSNode):
    """
    MCTS node enhanced with MDAP voting information.

    Extends MCTSNode with:
    - Agent votes for action selection
    - Red flags for unreliable actions
    - Vote confidence tracking
    - MAKER scores for quality assessment

    Attributes:
        state: Proof state at this node
        parent: Parent node
        action: Action that led to this node
        children: Child nodes
        N: Visit count
        W: Total reward
        Q: Average reward (W / N)
        agent_votes: Votes received from agents {action: count}
        red_flags: Red flags for actions {action: [reasons]}
        vote_confidence: Confidence in selected action (0-1)
        maker_score: Quality score from MAKER (0-1)
    """

    def __init__(self, state: ProofState, parent: Optional['MDAPMCTSNode'] = None):
        """
        Initialize MDAP-MCTS node.

        Args:
            state: Proof state
            parent: Optional parent node
        """
```

#### Methods

##### add_agent_vote()

```python
def add_agent_vote(self, action: str, count: int = 1):
    """
    Add agent votes for an action.

    Args:
        action: Tactic action
        count: Number of votes to add (default: 1)

    Example:
        ```python
        node.add_agent_vote("intros", 3)
        node.add_agent_vote("apply", 2)
        ```
    """
```

##### add_red_flag()

```python
def add_red_flag(self, action: str, reason: str):
    """
    Add red flag for an action.

    Args:
        action: Tactic action
        reason: Reason for red flag

    Example:
        ```python
        node.add_red_flag("apply invalid_lemma", "lemma_not_found")
        ```
    """
```

##### get_vote_summary()

```python
def get_vote_summary(self) -> Dict[str, Any]:
    """
    Get summary of voting information.

    Returns:
        Dict with keys:
        - 'total_votes': Total votes cast
        - 'action_votes': Votes per action
        - 'winner': Winning action
        - 'confidence': Confidence in winner (0-1)
        - 'red_flags': Red flags per action

    Example:
        ```python
        summary = node.get_vote_summary()
        print(f"Winner: {summary['winner']} with confidence {summary['confidence']:.2%}")
        ```
    """
```

### MDAPMCTSExpansion

Enhanced expansion phase with agent voting.

```python
class MDAPMCTSExpansion(MCTSExpansion):
    """
    MCTS expansion phase enhanced with MDAP voting.

    Instead of expanding with a single action, uses multi-agent
    voting to select the best action with consensus.

    Attributes:
        config: MCTS configuration
        mdap_orchestrator: Optional MDAP orchestrator for voting
    """

    def __init__(
        self,
        config: MCTSConfig,
        mdap_orchestrator: Optional[MDAPOrchestrator] = None
    ):
        """
        Initialize MDAP-enhanced expansion.

        Args:
            config: MCTS configuration
            mdap_orchestrator: Optional MDAP orchestrator
        """
```

#### Methods

##### expand()

```python
def expand(
    self,
    node: MDAPMCTSNode,
    available_actions: List[Tactic],
    policy_probs: Optional[Dict[str, float]] = None,
    mdap_orchestrator: Optional[MDAPOrchestrator] = None,
    k_ahead: int = 3
) -> Optional[MDAPMCTSNode]:
    """
    Expand node with voting-enhanced action selection.

    Args:
        node: Node to expand
        available_actions: Available tactic actions
        policy_probs: Optional policy probabilities
        mdap_orchestrator: MDAP orchestrator for voting
        k_ahead: First-to-ahead-by-k threshold

    Returns:
        Expanded child node, or None if expansion fails

    Example:
        ```python
        available = [Tactic("intros"), Tactic("apply Nat.add_comm")]
        child = expansion.expand(node, available, mdap_orchestrator=mdap, k_ahead=3)
        ```
    """
```

##### vote_on_actions()

```python
def vote_on_actions(
    self,
    state: ProofState,
    actions: List[Tactic],
    orchestrator: MDAPOrchestrator,
    k_ahead: int
) -> Tuple[Optional[Tactic], Dict[str, int], Dict[str, List[str]]]:
    """
    Have agents vote on best action.

    Args:
        state: Current proof state
        actions: Candidate actions
        orchestrator: MDAP orchestrator
        k_ahead: Voting threshold

    Returns:
        Tuple of (winner_action, vote_counts, red_flags)

    Example:
        ```python
        winner, votes, flags = expansion.vote_on_actions(
            state, actions, mdap, k_ahead=3
        )
        print(f"Winner: {winner} with {votes[winner.name]} votes")
        ```
    """
```

### MDAPMCTSSimulation

Enhanced simulation phase with MAKER voting.

```python
class MDAPMCTSSimulation(MCTSSimulation):
    """
    MCTS simulation phase enhanced with MAKER voting.

    Instead of random or heuristic rollouts, uses MAKER
    voting to select high-quality tactics during simulation.

    Attributes:
        config: MCTS configuration
        maker_engine: Optional MAKER engine for voting
    """

    def __init__(
        self,
        config: MCTSConfig,
        maker_engine: Optional[MAKEREngine] = None
    ):
        """
        Initialize MAKER-enhanced simulation.

        Args:
            config: MCTS configuration
            maker_engine: Optional MAKER engine
        """
```

#### Methods

##### simulate()

```python
def simulate(
    self,
    node: MDAPMCTSNode,
    maker_engine: Optional[MAKEREngine] = None,
    max_depth: Optional[int] = None
) -> float:
    """
    Simulate from node with voting-enhanced tactic selection.

    Args:
        node: Node to simulate from
        maker_engine: MAKER engine for voting
        max_depth: Optional depth limit

    Returns:
        Simulated reward (0-1)

    Example:
        ```python
        reward = simulation.simulate(node, maker_engine=maker, max_depth=20)
        print(f"Simulation reward: {reward:.2f}")
        ```
    """
```

##### simulate_with_voting()

```python
def simulate_with_voting(
    self,
    state: ProofState,
    maker_engine: MAKEREngine,
    max_depth: int,
    progress_callback: Optional[Callable[[int, ProofState], None]] = None
) -> float:
    """
    Perform simulation with MAKER voting at each step.

    Args:
        state: Starting state
        maker_engine: MAKER engine
        max_depth: Maximum depth
        progress_callback: Optional progress callback

    Returns:
        Final reward (0-1)

    Example:
        ```python
        reward = simulation.simulate_with_voting(
            state, maker_engine, max_depth=50,
            progress_callback=lambda d, s: print(f"Depth {d}")
        )
        ```
    """
```

---

## Data Structures

### MCTSConfig

Configuration for MCTS search.

```python
@dataclass
class MCTSConfig:
    """
    MCTS configuration parameters.

    Attributes:
        max_iterations: Maximum MCTS iterations
        time_budget: Maximum search time in seconds
        c_param: UCT exploration constant
        rollout_depth: Maximum rollout depth
        rollout_policy: Rollout policy type
        parallel_simulations: Number of parallel simulations
        enable_transposition_table: Enable state reuse
        enable_amaf: Enable All-Moves-As-First
        progressive_widening: Enable progressive widening
        early_termination: Stop early if proof found
        temperature: Temperature for final selection
        max_tree_depth: Maximum tree depth
        cache_size_mb: Transposition table cache size
    """
    max_iterations: int = 1000
    time_budget: float = 60.0
    c_param: float = 1.414
    rollout_depth: int = 100
    rollout_policy: str = "heuristic"
    parallel_simulations: int = 4
    enable_transposition_table: bool = True
    enable_amaf: bool = True
    amaf_alpha: float = 0.5
    progressive_widening: bool = True
    widening_factor: float = 0.5
    early_termination: bool = True
    min_visits_for_confidence: int = 10
    temperature: float = 0.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    max_tree_depth: int = 50
    pruning_threshold: float = 0.1
    cache_size_mb: int = 500

    # LeanAide-specific
    server_url: str = "http://localhost:7654"
    verification_timeout: float = 30.0
    enable_caching: bool = True
    max_proof_states: int = 10000
```

### MDAPConfig

Configuration for MDAP voting.

```python
@dataclass
class MDAPConfig:
    """
    MDAP configuration parameters.

    Attributes:
        k_min: Minimum k for first-to-ahead-by-k voting
        k_max: Maximum k for voting
        max_votes_per_step: Maximum voting rounds per step
        timeout_seconds: Timeout per voting step
        red_flag_rules: Red-flagging rules
        fallback_policy: Policy for handling failures
        cache_ttl_seconds: Cache time-to-live
        cache_max_size: Maximum cache size
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

### RedFlagRules

Rules for red-flagging unreliable responses.

```python
@dataclass
class RedFlagRules:
    """
    Red-flagging rules for MDAP.

    Attributes:
        max_tokens: Maximum response tokens
        max_characters: Maximum response characters
        blocked_patterns: Regex patterns to block
        min_confidence: Minimum confidence score
        require_schema_match: Require schema validation
    """
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    blocked_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.2
    require_schema_match: bool = True
```

### MAKERConfig

Configuration for MAKER engine.

```python
@dataclass
class MAKERConfig:
    """
    MAKER configuration parameters.

    Attributes:
        k_ahead: First-to-ahead-by-k threshold
        max_token_length: Maximum response length
        max_steps: Maximum solution steps
        enable_first_to_ahead: Use first-to-ahead-by-k
        enable_red_flagging: Enable red-flagging
        temperature_first: Temperature for first vote
        temperature_subsequent: Temperature for subsequent votes
    """
    k_ahead: int = 3
    max_token_length: int = 750
    max_steps: int = 1000
    enable_first_to_ahead: bool = True
    enable_red_flagging: bool = True
    temperature_first: float = 0.0
    temperature_subsequent: float = 0.1
```

### MCTSResult

Result of MCTS search.

```python
@dataclass
class MCTSResult:
    """
    Result of MCTS proof search.

    Attributes:
        best_proof: Best proof found
        success: Whether complete proof was found
        search_iterations: Number of iterations performed
        time_elapsed: Total time elapsed
        nodes_visited: Total nodes visited
        tree_depth: Maximum tree depth
        win_rate: Estimated win rate
        confidence: Confidence in result
        proof_path: Path from root to best proof
        search_statistics: Detailed search stats
        tree_statistics: Tree statistics
    """
    best_proof: Optional[LeanProof] = None
    success: bool = False
    search_iterations: int = 0
    time_elapsed: float = 0.0
    nodes_visited: int = 0
    tree_depth: int = 0
    win_rate: float = 0.0
    confidence: float = 0.0
    proof_path: List[MCTSNode] = field(default_factory=list)
    search_statistics: Dict[str, Any] = field(default_factory=dict)
    tree_statistics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
```

### ProofState

Represents a Lean 4 proof state.

```python
@dataclass
class ProofState:
    """
    Lean 4 proof state.

    Attributes:
        goals: Current unsolved goals
        context: Current proof context (hypotheses)
        tactics_sequence: Tactics applied so far
        depth: Depth in proof tree
        is_complete: Whether all goals are solved
        hash: Unique hash of state
    """
    goals: List[str] = field(default_factory=list)
    context: List[str] = field(default_factory=list)
    tactics_sequence: List[Tactic] = field(default_factory=list)
    depth: int = 0
    is_complete: bool = False
    hash: str = field(default="")
```

### Tactic

Represents a Lean 4 tactic.

```python
@dataclass
class Tactic:
    """
    Lean 4 tactic.

    Attributes:
        name: Tactic name (e.g., "intros", "apply")
        params: Tactic parameters
        metadata: Additional metadata
    """
    name: str
    params: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## Configuration

### Creating MCTS Configuration

```python
# Basic configuration
config = MCTSConfig(
    max_iterations=1000,
    time_budget=60.0,
    c_param=1.414
)

# Advanced configuration
config = MCTSConfig(
    max_iterations=2000,
    time_budget=120.0,
    c_param=1.8,
    rollout_depth=100,
    rollout_policy="heuristic",
    parallel_simulations=8,
    enable_transposition_table=True,
    enable_amaf=True,
    progressive_widening=True,
    early_termination=True,
    max_tree_depth=50,
    cache_size_mb=500
)

# Quality-focused configuration
config = MCTSConfig(
    max_iterations=5000,
    time_budget=600.0,
    c_param=2.0,  # More exploration
    rollout_depth=200,
    parallel_simulations=2,  # Fewer, higher-quality simulations
    enable_amaf=True,
    enable_transposition_table=True
)

# Speed-focused configuration
config = MCTSConfig(
    max_iterations=500,
    time_budget=10.0,
    c_param=1.2,  # Less exploration
    rollout_depth=20,
    parallel_simulations=8,  # More parallelism
    enable_amaf=False,
    enable_transposition_table=False
)
```

### Creating MDAP Configuration

```python
# Basic configuration
mdap_config = MDAPConfig(
    k_min=2,
    k_max=5,
    max_votes_per_step=20
)

# Conservative configuration (high quality)
mdap_config = MDAPConfig(
    k_min=3,
    k_max=8,
    max_votes_per_step=50,
    timeout_seconds=60,
    red_flag_rules=RedFlagRules(
        max_tokens=500,
        min_confidence=0.5,
        require_schema_match=True
    )
)

# Aggressive configuration (fast)
mdap_config = MDAPConfig(
    k_min=1,
    k_max=3,
    max_votes_per_step=10,
    timeout_seconds=15,
    red_flag_rules=RedFlagRules(
        max_tokens=1000,
        min_confidence=0.1,
        require_schema_match=False
    )
)

# With caching
mdap_config = MDAPConfig(
    k_min=2,
    k_max=5,
    cache_ttl_seconds=3600,  # 1 hour
    cache_max_size=10000
)
```

### Creating MAKER Configuration

```python
# Basic configuration
maker_config = {
    "k_ahead": 3,
    "max_token_length": 750,
    "max_steps": 1000,
    "enable_first_to_ahead": True,
    "enable_red_flagging": True
}

# Recursive solver configuration
recursive_config = {
    "max_depth": 5,
    "k_ahead": 3,
    "num_candidates": 5,
    "max_token_length": 750
}
```

### Creating Agent Team

```python
team = Team(
    team_id="theorem_proving_team",
    name="Theorem Proving Team",
    members=[
        ModelConfig(
            model_id="gpt-4",
            api_key=os.getenv("OPENAI_API_KEY"),
            api_base="https://api.openai.com/v1",
            temperature=0.0,
            max_tokens=750,
            problem_type_specialization=["theorem_proving", "induction"]
        ),
        ModelConfig(
            model_id="claude-3-opus",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            api_base="https://api.anthropic.com/v1",
            temperature=0.0,
            max_tokens=750,
            problem_type_specialization=["tactic_selection", "proof_refinement"]
        ),
        ModelConfig(
            model_id="gemini-pro",
            api_key=os.getenv("GOOGLE_API_KEY"),
            api_base="https://generativelanguage.googleapis.com/v1",
            temperature=0.1,
            max_tokens=750,
            problem_type_specialization=["lemma_selection", "rewriting"]
        )
    ]
)
```

---

## Search Functions

### search_proof_with_mcts()

Basic MCTS search without voting.

```python
def search_proof_with_mcts(
    initial_state: ProofState,
    config: Optional[MCTSConfig] = None,
    progress_callback: Optional[Callable[[int], None]] = None
) -> MCTSResult:
    """
    Search for proof using pure MCTS.

    Args:
        initial_state: Initial proof state
        config: Optional MCTS configuration
        progress_callback: Optional progress callback

    Returns:
        MCTSResult: Search result

    Example:
        ```python
        state = ProofState(goals=["forall (a b : Nat), a + b = b + a"])
        result = search_proof_with_mcts(state, config=MCTSConfig())

        if result.success:
            print(f"Found proof in {result.search_iterations} iterations")
        else:
            print("Proof not found")
        ```
    """
```

### search_with_mdap_mcts()

MCTS search with MDAP voting.

```python
def search_with_mdap_mcts(
    initial_state: ProofState,
    mcts_config: MCTSConfig,
    mdap_config: MDAPConfig,
    team: Team,
    progress_callback: Optional[Callable[[int], None]] = None
) -> MCTSResult:
    """
    Search for proof using MCTS with MDAP voting.

    Args:
        initial_state: Initial proof state
        mcts_config: MCTS configuration
        mdap_config: MDAP configuration
        team: Agent team
        progress_callback: Optional progress callback

    Returns:
        MCTSResult: Search result

    Example:
        ```python
        state = ProofState(goals=["forall (a b : Nat), a + b = b + a"])
        mcts_config = MCTSConfig(max_iterations=1000)
        mdap_config = MDAPConfig(k_min=2, k_max=5)

        result = search_with_mdap_mcts(state, mcts_config, mdap_config, team)

        print(f"Success: {result.success}")
        print(f"Iterations: {result.search_iterations}")
        print(f"Win rate: {result.win_rate:.2%}")
        ```
    """
```

### search_with_maker_mcts()

MCTS search with MAKER simulation.

```python
def search_with_maker_mcts(
    initial_state: ProofState,
    mcts_config: MCTSConfig,
    maker_engine: MAKEREngine,
    progress_callback: Optional[Callable[[int], None]] = None
) -> MCTSResult:
    """
    Search for proof using MCTS with MAKER simulation.

    Args:
        initial_state: Initial proof state
        mcts_config: MCTS configuration
        maker_engine: MAKER engine for simulation
        progress_callback: Optional progress callback

    Returns:
        MCTSResult: Search result

    Example:
        ```python
        maker_engine = MAKEREngine(team, k_ahead=3, max_steps=100)
        result = search_with_maker_mcts(state, mcts_config, maker_engine)

        print(f"Proof found: {result.success}")
        ```
    """
```

### decompose_and_search()

Recursive decomposition with MCTS.

```python
def decompose_and_search(
    task: str,
    context: Dict[str, Any],
    maker_engine: MAKEREngine,
    mcts_config: MCTSConfig,
    max_depth: int = 5
) -> Tuple[Any, MAKERRunMetrics]:
    """
    Decompose task and solve subproblems with MCTS.

    Args:
        task: Task description
        context: Task context
        maker_engine: MAKER engine for decomposition
        mcts_config: MCTS configuration
        max_depth: Maximum decomposition depth

    Returns:
        Tuple of (solution, metrics)

    Example:
        ```python
        task = "Prove that addition is commutative"
        context = {"type": "theorem", "domain": "natural_numbers"}

        solution, metrics = decompose_and_search(
            task, context, maker_engine, mcts_config, max_depth=3
        )

        print(f"Solution: {solution}")
        print(f"Decompositions: {metrics.decompositions}")
        ```
    """
```

---

## Utility Functions

### create_tactic_prompt()

Create prompt for tactic selection.

```python
def create_tactic_prompt(state: ProofState) -> str:
    """
    Create prompt for tactic selection from proof state.

    Args:
        state: Current proof state

    Returns:
        Prompt string for LLM

    Example:
        ```python
        state = ProofState(goals=["forall (a b : Nat), a + b = b + a"])
        prompt = create_tactic_prompt(state)
        # Returns: "Current goals: [forall (a b : Nat), a + b = b + a]\n..."
        ```
    """
```

### parse_tactic_response()

Parse LLM response to extract tactic.

```python
def parse_tactic_response(response: str) -> Tuple[Optional[Tactic], Optional[str]]:
    """
    Parse LLM response to extract tactic action.

    Args:
        response: LLM response text

    Returns:
        Tuple of (tactic, next_state)

    Example:
        ```python
        response = "action = intros\nnext_state = ..."
        tactic, next_state = parse_tactic_response(response)

        if tactic:
            print(f"Tactic: {tactic.name}")
        ```
    """
```

### estimate_proof_progress()

Estimate proof progress from state.

```python
def estimate_proof_progress(state: ProofState) -> float:
    """
    Estimate proof progress (0-1) from state.

    Args:
        state: Current proof state

    Returns:
        Progress estimate (0=just started, 1=complete)

    Example:
        ```python
        progress = estimate_proof_progress(state)
        print(f"Proof progress: {progress:.1%}")
        ```
    """
```

### get_applicable_tactics()

Get applicable tactics for state.

```python
def get_applicable_tactics(
    state: ProofState,
    lean_server_url: str = "http://localhost:7654"
) -> List[Tactic]:
    """
    Get applicable tactics from Lean server.

    Args:
        state: Current proof state
        lean_server_url: Lean server URL

    Returns:
        List of applicable tactics

    Example:
        ```python
        tactics = get_applicable_tactics(state)
        print(f"Applicable tactics: {[t.name for t in tactics]}")
        ```
    """
```

### compute_uct_value()

Compute UCT value for node.

```python
def compute_uct_value(
    node: MCTSNode,
    c_param: float,
    value_normalization: Optional[float] = None
) -> float:
    """
    Compute UCT (Upper Confidence Bound) value for node.

    Args:
        node: MCTS node
        c_param: Exploration constant
        value_normalization: Optional value normalization

    Returns:
        UCT value

    Example:
        ```python
        uct = compute_uct_value(node, c_param=1.414)
        print(f"UCT value: {uct:.4f}")
        ```
    """
```

---

## Type Definitions

### VoteDict

```python
VoteDict = Dict[str, int]  # Maps action to vote count
```

### RedFlagDict

```python
RedFlagDict = Dict[str, List[str]]  # Maps action to red flag reasons
```

### ProgressCallback

```python
ProgressCallback = Callable[[int, MCTSNode], None]
```

### TacticParser

```python
TacticParser = Callable[[str], Tuple[Optional[Tactic], Optional[str]]]
```

---

## Error Handling

### MDAPMCTSError

Base exception for MDAP-MCTS errors.

```python
class MDAPMCTSError(Exception):
    """Base exception for MDAP-MCTS errors."""
    pass
```

### VotingError

Raised when voting fails.

```python
class VotingError(MDAPMCTSError):
    """Raised when voting fails to converge."""

    def __init__(self, message: str, votes: Dict[str, int]):
        super().__init__(message)
        self.votes = votes
```

### RedFlagError

Raised when all actions are red-flagged.

```python
class RedFlagError(MDAPMCTSError):
    """Raised when all actions are red-flagged."""

    def __init__(self, message: str, red_flags: Dict[str, List[str]]):
        super().__init__(message)
        self.red_flags = red_flags
```

### TimeoutError

Raised when search times out.

```python
class TimeoutError(MDAPMCTSError):
    """Raised when search exceeds time limit."""

    def __init__(self, message: str, elapsed_time: float):
        super().__init__(message)
        self.elapsed_time = elapsed_time
```

### Error Handling Example

```python
try:
    result = search_with_mdap_mcts(state, mcts_config, mdap_config, team)
except VotingError as e:
    logger.error(f"Voting failed: {e}")
    logger.error(f"Final votes: {e.votes}")
    # Fall back to pure MCTS
    result = search_proof_with_mcts(state, mcts_config)
except RedFlagError as e:
    logger.warning(f"All actions red-flagged: {e.red_flags}")
    # Relax red-flagging and retry
    mdap_config.red_flag_rules.min_confidence = 0.1
    result = search_with_mdap_mcts(state, mcts_config, mdap_config, team)
except TimeoutError as e:
    logger.warning(f"Search timed out after {e.elapsed_time:.1f}s")
    # Return best partial result
    result = e.partial_result
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    raise
```

---

## Integration API

### LeanAide Integration

```python
class LeanAideMCTSIntegration:
    """
    Integration layer for LeanAide + MDAP-MCTS.

    Provides high-level API for combining Lean 4 verification
    with MDAP-MCTS search.
    """

    def __init__(
        self,
        lean_server_url: str = "http://localhost:7654",
        mcts_config: Optional[MCTSConfig] = None,
        mdap_config: Optional[MDAPConfig] = None
    ):
        """
        Initialize integration.

        Args:
            lean_server_url: Lean 4 server URL
            mcts_config: Optional MCTS config
            mdap_config: Optional MDAP config
        """
```

#### Methods

##### prove_theorem()

```python
def prove_theorem(
    self,
    theorem_name: str,
    theorem_statement: str,
    team: Team,
    time_limit: float = 60.0
) -> MCTSResult:
    """
    Prove theorem using MDAP-MCTS.

    Args:
        theorem_name: Name of theorem
        theorem_statement: Theorem statement
        team: Agent team
        time_limit: Time limit in seconds

    Returns:
        MCTSResult: Proof result

    Example:
        ```python
        integration = LeanAideMCTSIntegration()
        result = integration.prove_theorem(
            "add_comm",
            "forall (a b : Nat), a + b = b + a",
            team,
            time_limit=120.0
        )

        if result.success:
            print(f"Proof found!")
            print(result.best_proof)
        ```
    """
```

##### verify_proof()

```python
def verify_proof(
    self,
    theorem_statement: str,
    proof_tactics: List[Tactic]
) -> VerificationResult:
    """
    Verify proof using Lean 4 server.

    Args:
        theorem_statement: Theorem statement
        proof_tactics: Proof tactics

    Returns:
        VerificationResult: Verification result

    Example:
        ```python
        tactics = [Tactic("intros"), Tactic("apply", ["Nat.add_comm"])]
        result = integration.verify_proof(
            "forall (a b : Nat), a + b = b + a",
            tactics
        )

        if result.is_valid:
            print("Proof is valid!")
        ```
    """
```

---

## Quick Reference

### Common Usage Patterns

**Pattern 1: Basic MCTS Search**
```python
config = MCTSConfig(max_iterations=1000)
result = search_proof_with_mcts(state, config)
```

**Pattern 2: MCTS with MDAP Voting**
```python
mcts_config = MCTSConfig(max_iterations=1000)
mdap_config = MDAPConfig(k_min=2, k_max=5)
result = search_with_mdap_mcts(state, mcts_config, mdap_config, team)
```

**Pattern 3: MCTS with MAKER Simulation**
```python
maker_engine = MAKEREngine(team, k_ahead=3)
result = search_with_maker_mcts(state, mcts_config, maker_engine)
```

**Pattern 4: Full MDAP-MCTS-MAKER**
```python
mcts_mdap = MDAPMCTS(mcts_config, mdap_config, team)
maker_engine = MAKEREngine(team, k_ahead=3)
result = mcts_mdap.search_with_maker(state, maker_engine)
```

**Pattern 5: Decomposition + MCTS**
```python
solution, metrics = decompose_and_search(
    task, context, maker_engine, mcts_config, max_depth=3
)
```

### Configuration Templates

| Use Case | MCTS Config | MDAP Config | MAKER Config |
|----------|-------------|-------------|--------------|
| Simple Proof | `max_iterations=500` | `k_max=3` | Not needed |
| Medium Proof | `max_iterations=1000` | `k_max=5` | `k_ahead=3` |
| Complex Proof | `max_iterations=2000` | `k_max=8` | `k_ahead=5` |
| Time-Critical | `time_budget=10.0` | `k_max=2` | Not needed |
| Quality-Critical | `max_iterations=5000` | `k_max=8` | `k_ahead=5` |

For more examples, see `LEANAIDE_MCTS_MDAP_EXAMPLES.md`.
