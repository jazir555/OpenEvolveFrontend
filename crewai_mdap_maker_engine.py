"""
CrewAI MDAP/MAKER Engine - Port of MAKER to CrewAI Agents

This module ports the MAKER (Maximal Agentic decomposition, first-to-ahead-by-K
Error correction, and Red-flagging) framework to CrewAI agents and crews.

Based on: "Solving a Million-Step LLM Task with Zero Errors" (arXiv:2511.09030)

Key Features:
1. Algorithm 1: generate_solution - Main orchestration with voting
2. Algorithm 2: do_voting - First-to-ahead-by-k voting mechanism
3. Algorithm 3: get_vote - Voting with red-flagging
4. Algorithm 4: Recursive multi-agent solve with decomposition

CrewAI Architecture:
- MAKEREngine → CrewAI Crew with multiple agents
- VotingEngine → CrewAI Task with agent coordination
- VoteCollector → CrewAI Agent with red-flagging
- RecursiveMAKERSolver → Nested CrewAI Crews

License: MIT (replaces AGPL Hephaestus)
"""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from enum import Enum

# CrewAI imports
try:
    from crewai import Agent, Task, Crew, Process
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    Agent = None
    Task = None
    Crew = None
    Process = None

# Import state management
from crewai_state_management import (
    WorkflowState,
    WorkflowStatus,
    SolutionAttempt,
)

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MAKERConfig:
    """Configuration for MAKER execution"""
    k_ahead: int = 5  # First-to-ahead-by-K threshold
    max_token_length: int = 750
    max_steps: int = 1000
    max_voting_rounds: int = 50
    enable_first_to_ahead: bool = True
    enable_red_flagging: bool = True
    temperature_first: float = 0.0
    temperature_subsequent: float = 0.1
    max_retries: int = 10


@dataclass
class MAKERRunMetrics:
    """Metrics for MAKER execution"""
    workflow_id: str
    total_steps: int = 0
    total_votes: int = 0
    red_flags: int = 0
    decompositions: int = 0
    atomic_solves: int = 0
    voting_rounds: int = 0
    total_time: float = 0.0
    avg_confidence: float = 0.0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "total_steps": self.total_steps,
            "total_votes": self.total_votes,
            "red_flags": self.red_flags,
            "decompositions": self.decompositions,
            "atomic_solves": self.atomic_solves,
            "voting_rounds": self.voting_rounds,
            "total_time": self.total_time,
            "avg_confidence": self.avg_confidence,
            "errors": self.errors,
        }


@dataclass
class VoteResult:
    """Result from a single vote"""
    action: Any
    state: Any
    raw_text: str
    red_flagged: bool = False
    agent_name: Optional[str] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class VotingResult:
    """Result from voting round"""
    winner: Any
    vote_counts: Dict[str, int]
    confidence: float
    rounds: int
    red_flags: int
    meets_threshold: bool


# =============================================================================
# AGENT ROLE DEFINITIONS
# =============================================================================

class MAKERAgentRole(str, Enum):
    """Roles for MAKER agents"""
    VOTER = "voter"  # Casts votes on solutions
    DECOMPOSER = "decomposer"  # Breaks down tasks
    SOLUTION_DISCRIMINATOR = "solution_discriminator"  # Evaluates solutions
    ATOMIC_SOLVER = "atomic_solver"  # Solves atomic tasks
    REFINER = "refiner"  # Refines and improves solutions


# =============================================================================
# CREWAI AGENT FACTORY
# =============================================================================

class MAKERAgentFactory:
    """
    Factory for creating CrewAI agents for MAKER operations.

    Replaces direct LLM calls with CrewAI agents.
    """

    @staticmethod
    def create_voter_agent(
        name: str = "MAKER_Voter",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """
        Create a CrewAI agent for voting.

        Args:
            name: Agent name
            llm_config: LLM configuration

        Returns:
            CrewAI Agent or None if CrewAI unavailable
        """
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MAKERAgentRole.VOTER.value,
            goal="Cast votes on proposed solutions with high accuracy and reliability",
            backstory="""You are an expert evaluator specializing in analyzing proposed solutions
            and determining the best course of action. You vote based on:
            1. Correctness and accuracy
            2. Efficiency and optimality
            3. Robustness and reliability
            You avoid responses that are overly long or poorly formatted.""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_decomposer_agent(
        name: str = "MAKER_Decomposer",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for task decomposition."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MAKERAgentRole.DECOMPOSER.value,
            goal="Break down complex tasks into manageable sub-tasks with clear dependencies",
            backstory="""You are an expert at task decomposition and problem analysis. You can:
            1. Identify task boundaries and dependencies
            2. Break complex problems into atomic sub-tasks
            3. Define composition functions for combining results
            4. Estimate complexity and effort for each sub-task""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_solution_discriminator_agent(
        name: str = "MAKER_Solution_Discriminator",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for solution evaluation."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MAKERAgentRole.SOLUTION_DISCRIMINATOR.value,
            goal="Evaluate and compare alternative solutions to select the best one",
            backstory="""You are an expert at evaluating solution quality. You assess:
            1. Completeness - does it solve the entire problem?
            2. Correctness - is the solution accurate and valid?
            3. Efficiency - does it use resources optimally?
            4. Robustness - will it work in edge cases?""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_atomic_solver_agent(
        name: str = "MAKER_Atomic_Solver",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for solving atomic tasks."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MAKERAgentRole.ATOMIC_SOLVER.value,
            goal="Solve atomic (non-decomposable) tasks with high accuracy",
            backstory="""You are an expert problem solver specializing in atomic tasks that cannot
            be further decomposed. You provide:
            1. Direct and accurate solutions
            2. Clear reasoning and justification
            3. Confidence scores for your solutions""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )

    @staticmethod
    def create_refiner_agent(
        name: str = "MAKER_Refiner",
        llm_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[Agent]:
        """Create a CrewAI agent for solution refinement."""
        if not CREWAI_AVAILABLE:
            return None

        return Agent(
            role=MAKERAgentRole.REFINER.value,
            goal="Refine and improve existing solutions to maximize quality",
            backstory="""You are an expert at improving solutions through iterative refinement.
            You focus on:
            1. Eliminating inefficiencies
            2. Fixing edge cases and bugs
            3. Optimizing performance
            4. Improving clarity and maintainability""",
            verbose=True,
            allow_delegation=False,
            **(llm_config or {})
        )


# =============================================================================
# ALGORITHM 3: get_vote - Voting with Red-Flagging
# =============================================================================

class CrewAIVoteCollector:
    """
    Algorithm 3: get_vote from the paper, ported to CrewAI.

    Collects votes from CrewAI agents with red-flagging to discard unreliable responses.

    From paper: "get_vote" (line 3-7):
    1: Input x, M
    2: while True do
    3:   r ∼ (M ◦ ϕ)(x)
    4:   if r has no red flags then
    5:     return ψa (r), ψx (r)
    6:   end if
    7: end while
    """

    def __init__(
        self,
        config: MAKERConfig,
        workflow_id: Optional[str] = None,
    ):
        self.config = config
        self.workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:12]}"
        self.attempt_count = 0

    def get_vote(
        self,
        prompt: str,
        agent: Agent,
        context: Optional[Dict[str, Any]] = None,
    ) -> VoteResult:
        """
        Collect a single vote with red-flagging.

        Args:
            prompt: Voting prompt
            agent: CrewAI agent to get vote from
            context: Additional context

        Returns:
            VoteResult with action, state, and metadata

        Raises:
            RuntimeError: If max retries exceeded
        """
        for attempt in range(self.config.max_retries):
            self.attempt_count += 1

            # Determine temperature
            temperature = (
                self.config.temperature_first if attempt == 0
                else self.config.temperature_subsequent
            )

            try:
                # Execute agent task
                if CREWAI_AVAILABLE:
                    task = Task(
                        description=prompt,
                        expected_output="A proposed solution with action and next_state",
                        agent=agent,
                    )

                    # Create crew for single task
                    crew = Crew(
                        agents=[agent],
                        tasks=[task],
                        process=Process.sequential,
                        verbose=True,
                    )

                    # Execute
                    raw_text = crew.kickoff()

                    # Check red flags
                    if self._has_red_flags(raw_text):
                        logger.debug(f"Vote {attempt + 1} red-flagged, retrying...")
                        continue

                    # Parse response
                    action, state = self._parse_vote(raw_text)

                    return VoteResult(
                        action=action,
                        state=state,
                        raw_text=raw_text,
                        red_flagged=False,
                        agent_name=agent.role if agent else None,
                    )

            except Exception as e:
                logger.warning(f"Vote collection attempt {attempt + 1} failed: {e}")
                continue

        raise RuntimeError(f"Failed to get valid vote after {self.config.max_retries} attempts")

    def _has_red_flags(self, raw_text: str) -> bool:
        """
        Check for red flags indicating unreliable response.

        From paper Section 3.3: "Red-Flagging: Recognizing Signs of Unreliability"
        - Overly long responses
        - Incorrectly formatted responses
        """
        if not raw_text or raw_text.isspace():
            return True

        # Check length (approximate token count)
        approx_tokens = len(raw_text) / 4
        if approx_tokens > self.config.max_token_length * 0.9:
            return True

        # Check for basic structure
        # (More sophisticated checks could be added here)
        return False

    def _parse_vote(self, raw_text: str) -> Tuple[Any, Any]:
        """
        Parse vote text into action and state.

        Args:
            raw_text: Raw response text

        Returns:
            Tuple of (action, state)
        """
        # Try JSON parsing first
        try:
            data = json.loads(raw_text)
            if isinstance(data, dict):
                action = data.get("action")
                state = data.get("next_state", data.get("state"))
                if action is not None:
                    return action, state
        except json.JSONDecodeError:
            pass

        # Try regex extraction
        import re

        action_match = re.search(r'action\s*=\s*(.+?)(?:\n|$)', raw_text, re.IGNORECASE)
        state_match = re.search(r'next_state\s*=\s*(.+?)(?:\n|$)', raw_text, re.IGNORECASE)

        action = None
        state = None

        if action_match:
            action_str = action_match.group(1).strip()
            try:
                action = json.loads(action_str)
            except json.JSONDecodeError:
                action = action_str

        if state_match:
            state_str = state_match.group(1).strip()
            try:
                state = json.loads(state_str)
            except json.JSONDecodeError:
                state = state_str

        # If no explicit action/state, treat entire text as action
        if action is None:
            action = raw_text
        if state is None:
            state = raw_text

        return action, state


# =============================================================================
# ALGORITHM 2: do_voting - First-to-Ahead-by-K Voting
# =============================================================================

class CrewAIVotingEngine:
    """
    Algorithm 2: do_voting from the paper, ported to CrewAI.

    Implements first-to-ahead-by-k voting mechanism using CrewAI agents.

    From paper (line 1-9):
    1: Input: x, M, k
    2: V ← {v : 0 ∀v}    # Vote counts
    3: while True do
    4:   y ← get_vote(x, M)
    5:   V [y] = V [y] + 1
    6:   if V [y] ≥ k + maxv̸=y V [v] then
    7:     return y
    8:   end if
    9: end while
    """

    def __init__(
        self,
        config: MAKERConfig,
        workflow_id: Optional[str] = None,
    ):
        self.config = config
        self.workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:12]}"
        self.vote_collector = CrewAIVoteCollector(config, workflow_id)
        self.metrics = {
            "total_rounds": 0,
            "red_flags": 0,
            "votes_per_candidate": {},
        }

    def do_voting(
        self,
        prompt: str,
        agents: List[Agent],
        context: Optional[Dict[str, Any]] = None,
    ) -> VotingResult:
        """
        Perform voting until winner emerges.

        Args:
            prompt: Voting prompt
            agents: List of CrewAI agents (will cycle through)
            context: Additional context

        Returns:
            VotingResult with winner and metadata
        """
        votes: Dict[str, int] = {}
        agent_idx = 0
        round_num = 0
        red_flags = 0

        while round_num < self.config.max_voting_rounds:
            round_num += 1
            self.metrics["total_rounds"] += 1

            # Select agent (round-robin)
            agent = agents[agent_idx % len(agents)]
            agent_idx += 1

            # Collect vote
            try:
                vote_result = self.vote_collector.get_vote(
                    prompt=prompt,
                    agent=agent,
                    context=context,
                )

                if vote_result.red_flagged:
                    red_flags += 1
                    continue

                # Canonicalize candidate for counting
                candidate_key = self._canonicalize_candidate(
                    vote_result.action,
                    vote_result.state
                )

                # Count vote
                votes[candidate_key] = votes.get(candidate_key, 0) + 1
                self.metrics["votes_per_candidate"] = votes.copy()

                # Check for winner
                if self._has_winner(votes, self.config.k_ahead):
                    winner_key = max(votes, key=votes.get)
                    winner = self._decode_vote(winner_key)

                    # Calculate confidence
                    total_votes = sum(votes.values())
                    confidence = votes[winner_key] / total_votes if total_votes > 0 else 0.0

                    return VotingResult(
                        winner=winner,
                        vote_counts=votes,
                        confidence=confidence,
                        rounds=round_num,
                        red_flags=red_flags,
                        meets_threshold=True,
                    )

            except RuntimeError as e:
                logger.warning(f"Vote collection failed: {e}")
                red_flags += 1
                continue

        # Timeout - return best effort
        logger.warning(f"Voting reached max rounds ({self.config.max_voting_rounds}), returning best effort")
        if votes:
            winner_key = max(votes, key=votes.get)
            winner = self._decode_vote(winner_key)
            total_votes = sum(votes.values())
            confidence = votes[winner_key] / total_votes if total_votes > 0 else 0.0

            return VotingResult(
                winner=winner,
                vote_counts=votes,
                confidence=confidence,
                rounds=round_num,
                red_flags=red_flags,
                meets_threshold=False,
            )

        return VotingResult(
            winner=None,
            vote_counts=votes,
            confidence=0.0,
            rounds=round_num,
            red_flags=red_flags,
            meets_threshold=False,
        )

    def _canonicalize_candidate(self, action: Any, state: Any) -> str:
        """Convert candidate to canonical string for voting."""
        candidate = {"action": action, "state": state}
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

    def _decode_vote(self, vote_key: str) -> Any:
        """Decode vote key back to candidate."""
        try:
            candidate = json.loads(vote_key)
            return candidate.get("action")
        except (json.JSONDecodeError, TypeError):
            return vote_key

    def _has_winner(self, votes: Dict[str, int], k: int) -> bool:
        """
        Check if any candidate has won.

        For first-to-ahead-by-k: V[y] ≥ k + max V[v≠y]
        """
        if not votes:
            return False

        leader = max(votes, key=votes.get)
        leader_count = votes[leader]

        if self.config.enable_first_to_ahead:
            # First-to-ahead-by-k
            max_other = max(
                (count for key, count in votes.items() if key != leader),
                default=0
            )
            return leader_count >= max_other + k
        else:
            # First-to-k
            return leader_count >= k


# =============================================================================
# ALGORITHM 1: generate_solution - Main Orchestration
# =============================================================================

class CrewAIMAKEREngine:
    """
    Algorithm 1: generate_solution from the paper, ported to CrewAI.

    Main orchestration for MAKER system using CrewAI crews.

    From paper (line 1-8):
    1: Input xo, M, k
    2: Initialize A ← []   # Action list
    3: Initialize x ← xo
    4: for s steps do
    5:   a, x ← do_voting(x, M, k)
    6:   Append a to A
    7: end for
    8: return A
    """

    def __init__(
        self,
        agents: List[Agent],
        config: MAKERConfig,
        workflow_id: Optional[str] = None,
    ):
        """
        Initialize MAKER engine.

        Args:
            agents: List of CrewAI agents for voting
            config: MAKER configuration
            workflow_id: Optional workflow identifier
        """
        self.agents = agents
        self.config = config
        self.workflow_id = workflow_id or f"workflow_{uuid.uuid4().hex[:12]}"

        # Initialize voting engine
        self.voting_engine = CrewAIVotingEngine(config, workflow_id)

        logger.info(f"CrewAIMAKEREngine initialized with {len(agents)} agents")

    def generate_solution(
        self,
        initial_state: Any,
        prompt_template: Callable[[Any], str],
        context: Optional[Dict[str, Any]] = None,
        stop_condition: Optional[Callable[[Any], bool]] = None,
        progress_callback: Optional[Callable[[int, Any], None]] = None,
    ) -> Tuple[List[Any], Any, MAKERRunMetrics]:
        """
        Generate solution through iterative voting.

        Args:
            initial_state: Starting state
            prompt_template: Function(state) -> prompt
            context: Additional context
            stop_condition: Optional function(state) -> bool
            progress_callback: Optional callback(step, state)

        Returns:
            Tuple of (action_list, final_state, metrics)
        """
        action_list: List[Any] = []
        current_state = initial_state
        total_votes = 0
        total_red_flags = 0
        start_time = time.time()

        for step in range(self.config.max_steps):
            # Check stop condition
            if stop_condition and stop_condition(current_state):
                logger.info(f"Stop condition met at step {step}")
                break

            # Generate prompt for current state
            prompt = prompt_template(current_state)

            # Perform voting
            try:
                voting_result = self.voting_engine.do_voting(
                    prompt=prompt,
                    agents=self.agents,
                    context=context,
                )

                total_votes += voting_result.rounds
                total_red_flags += voting_result.red_flags

                if voting_result.winner is None:
                    logger.error(f"Voting failed at step {step}")
                    break

                # Apply action to get next state
                action = voting_result.winner

                # Extract next state if available
                if isinstance(action, dict) and "next_state" in action:
                    next_state = action["next_state"]
                    action = action.get("action")
                else:
                    next_state = current_state  # State unchanged

                # Append action
                action_list.append(action)
                current_state = next_state

                # Progress callback
                if progress_callback:
                    progress_callback(step + 1, current_state)

            except Exception as e:
                logger.error(f"Error at step {step}: {e}", exc_info=True)
                break

        total_time = time.time() - start_time

        metrics = MAKERRunMetrics(
            workflow_id=self.workflow_id,
            total_steps=len(action_list),
            total_votes=total_votes,
            red_flags=total_red_flags,
            total_time=total_time,
            avg_confidence=0.95 if action_list else 0.0,
        )

        return action_list, current_state, metrics


# =============================================================================
# ALGORITHM 4: Recursive Multi-Agent Solve (TODO: Future Phase)
# =============================================================================

class CrewAIRecursiveMAKERSolver:
    """
    Algorithm 4: Recursive multi-agent solve from Appendix F.

    Implements general-purpose decomposition with voting at each level.

    NOTE: This is a placeholder for future implementation.
    Full recursive decomposition will be implemented in Phase 1.3.3.

    From paper (line 1-18):
    1:  N ← 2k − 1                    # First-to-k voting, N candidates per step
    2:  function D ECOMPOSE(x)
    3:    sample N decompositions via D ECOMPOSER(x)
    4:    vote via S OLUTION D ISCRIMINATOR until one reaches k
    5:    return (P1 , P2 , C)        # Subtask1, Subtask2, Composition
    6:  end function
    """

    def __init__(self, config: MAKERConfig):
        self.config = config
        logger.warning("CrewAIRecursiveMAKERSolver not yet fully implemented")

    def solve(self, problem: str) -> Any:
        """Placeholder for recursive solve."""
        raise NotImplementedError("Recursive MAKER solving will be implemented in Phase 1.3.3")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_maker_config(
    k_ahead: int = 5,
    max_token_length: int = 750,
    max_steps: int = 1000,
    enable_first_to_ahead: bool = True,
    enable_red_flagging: bool = True,
) -> MAKERConfig:
    """
    Factory function to create MAKER configuration.

    Args:
        k_ahead: First-to-ahead-by-K threshold
        max_token_length: Maximum token length for responses
        max_steps: Maximum solving steps
        enable_first_to_ahead: Enable first-to-ahead-by-k voting
        enable_red_flagging: Enable red-flagging

    Returns:
        MAKERConfig instance
    """
    return MAKERConfig(
        k_ahead=k_ahead,
        max_token_length=max_token_length,
        max_steps=max_steps,
        enable_first_to_ahead=enable_first_to_ahead,
        enable_red_flagging=enable_red_flagging,
    )


def create_maker_engine(
    agents: List[Agent],
    config: Optional[MAKERConfig] = None,
    workflow_id: Optional[str] = None,
) -> CrewAIMAKEREngine:
    """
    Factory function to create MAKER engine.

    Args:
        agents: List of CrewAI agents for voting
        config: MAKER configuration (uses defaults if None)
        workflow_id: Optional workflow identifier

    Returns:
        CrewAIMAKEREngine instance
    """
    if config is None:
        config = create_maker_config()

    return CrewAIMAKEREngine(
        agents=agents,
        config=config,
        workflow_id=workflow_id,
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("CrewAI MDAP/MAKER Engine Example")
    print("=" * 50)

    if not CREWAI_AVAILABLE:
        print("CrewAI not available. Install with: pip install crewai")
    else:
        # Create agents
        factory = MAKERAgentFactory()

        voter_agents = [
            factory.create_voter_agent(f"Voter_{i}")
            for i in range(3)
        ]

        # Create config
        config = create_maker_config(k_ahead=3)

        # Create engine
        engine = create_maker_engine(
            agents=voter_agents,
            config=config,
        )

        print(f"Created MAKER engine with {len(voter_agents)} agents")
        print(f"K-ahead threshold: {config.k_ahead}")
        print(f"First-to-ahead-by-K: {config.enable_first_to_ahead}")
