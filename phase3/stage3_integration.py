"""
Stage 3 Integration: Monte Carlo Nest

Integrates Γ₁ (ACI Analyzer), Γ₂ (MCTS Search), and Γ₃ (Statistical Validator)
for the Monte Carlo Refinement phase of RESE.

Architecture:
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│    Γ₁ ACI    │───▶│  Γ₂ MCTS     │───▶│   Γ₃ Stats   │
│   Analyzer   │    │    Search    │    │  Validator   │
└──────────────┘    └──────────────┘    └──────────────┘

Author: Agent D2 (Γ₂/Γ₃ Specialist)
Created: 2025-12-31
Status: 🟢 Active Implementation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable, Any
from enum import Enum
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Try to import RESE modules
try:
    from phase3.mcts_search import (
        MCTSState, MCTSConfig, MCTSSearch, ParallelMCTS,
        PlayoutStrategy
    )
except ImportError:
    MCTSState = None
    MCTSConfig = None
    MCTSSearch = None
    ParallelMCTS = None

try:
    from phase3.statistical_validator import (
        StatisticalValidator, ValidationResult, ValidationConfig
    )
except ImportError:
    StatisticalValidator = None
    ValidationResult = None
    ValidationConfig = None

try:
    from phase3.aci_analyzer import ACIAnalyzer
except ImportError:
    ACIAnalyzer = None


class AgentStrategy(Enum):
    """Strategies for different MCTS agents"""
    EXPLOIT = "exploit"  # Heavy exploitation (low C, high ACI trust)
    EXPLORE = "explore"  # Heavy exploration (high C, low ACI trust)
    BALANCED = "balanced"  # Balanced exploration/exploitation
    ADAPTIVE = "adaptive"  # ACI-adaptive


@dataclass
class NestConfig:
    """Configuration for Monte Carlo Nest"""
    # Agent configuration
    num_agents: int = 4
    agent_strategies: List[AgentStrategy] = field(default_factory=lambda: [
        AgentStrategy.EXPLOIT,
        AgentStrategy.EXPLORE,
        AgentStrategy.BALANCED,
        AgentStrategy.ADAPTIVE
    ])

    # MCTS configuration (base)
    mcts_iterations: int = 500
    mcts_playout_depth: int = 50

    # ACI guidance
    aci_guided: bool = True
    early_stopping: bool = True

    # Validation
    validate_results: bool = True
    confidence_level: float = 0.95

    # Parallelization
    parallel_agents: bool = True
    max_workers: int = 4

    # Stopping criteria
    max_time_seconds: float = 300.0  # 5 minutes
    convergence_required: bool = True

    # Debugging
    verbose: bool = True


@dataclass
class AgentResult:
    """Result from a single MCTS agent"""
    agent_id: int
    strategy: AgentStrategy
    best_value: float
    best_node: Any  # MCTSNode
    search_info: Dict
    validation: Optional[ValidationResult] = None

    @property
    def is_confident(self) -> bool:
        """Check if this result is confident (validated)"""
        if self.validation is None:
            return False
        return self.validation.is_confident()


@dataclass
class NestResult:
    """Result from Monte Carlo Nest"""
    best_agent_result: AgentResult
    all_agent_results: List[AgentResult]
    aggregated_value: float
    confidence: float
    elapsed_time: float
    converged: bool
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> str:
        """Generate summary string"""
        lines = [
            "=== Monte Carlo Nest Result ===",
            f"Best Agent: {self.best_agent_result.strategy.value} (Agent {self.best_agent_result.agent_id})",
            f"Best Value: {self.best_agent_result.best_value:.4f}",
            f"Aggregated Value: {self.aggregated_value:.4f}",
            f"Confidence: {self.confidence:.2f}",
            f"Converged: {self.converged}",
            f"Elapsed Time: {self.elapsed_time:.2f}s",
            f"Total Agents: {len(self.all_agent_results)}",
        ]

        if self.best_agent_result.validation:
            lines.append(f"\nBest Agent Validation:")
            lines.append(f"  CI: {self.best_agent_result.validation.confidence_interval}")
            lines.append(f"  Convergence: {self.best_agent_result.validation.convergence}")

        return "\n".join(lines)


class MonteCarloNest:
    """
    Monte Carlo Nest: Integrates Γ₁, Γ₂, Γ₃ for adaptive search.

    Workflow:
    1. Calculate ACI for initial problem (Γ₁)
    2. Launch multiple MCTS agents with diverse strategies (Γ₂)
    3. Validate results with statistical tests (Γ₃)
    4. Aggregate and return best validated solution
    """

    def __init__(self, config: NestConfig = None, aci_analyzer: ACIAnalyzer = None):
        """
        Initialize Monte Carlo Nest.

        Args:
            config: Nest configuration
            aci_analyzer: ACI analyzer (Γ₁)
        """
        self.config = config or NestConfig()
        self.aci_analyzer = aci_analyzer
        self.validator = StatisticalValidator() if self.config.validate_results else None

    def search(self,
               initial_state: MCTSState,
               action_generator: Callable[[MCTSState], List[Any]],
               state_transition: Callable[[MCTSState, Any], MCTSState],
               value_function: Callable[[MCTSState], float]) -> NestResult:
        """
        Run Monte Carlo Nest search.

        Args:
            initial_state: Starting state
            action_generator: Generate available actions
            state_transition: Apply action to get new state
            value_function: Evaluate state quality

        Returns:
            NestResult with best solution and metadata
        """
        start_time = time.time()

        # Step 1: Calculate ACI (Γ₁)
        initial_aci = self._calculate_aci(initial_state)

        if self.config.verbose:
            print(f"[Nest] Initial ACI: {initial_aci.get('ACI', 'N/A'):.3f}")

        # Step 2: Configure agents based on ACI
        agent_configs = self._create_agent_configs(initial_aci)

        # Step 3: Run agents in parallel
        agent_results = self._run_agents(
            initial_state,
            action_generator,
            state_transition,
            value_function,
            agent_configs
        )

        # Step 4: Validate results (Γ₃)
        if self.validator:
            agent_results = self._validate_agents(agent_results)

        # Step 5: Aggregate results
        best_result, aggregated = self._aggregate_results(agent_results)

        elapsed_time = time.time() - start_time

        # Compile final result
        nest_result = NestResult(
            best_agent_result=best_result,
            all_agent_results=agent_results,
            aggregated_value=aggregated['value'],
            confidence=aggregated['confidence'],
            elapsed_time=elapsed_time,
            converged=aggregated['converged'],
            metadata={
                'initial_aci': initial_aci,
                'num_agents': len(agent_results),
                'agent_strategies': [r.strategy.value for r in agent_results],
                'validation_summary': aggregated.get('validation_summary')
            }
        )

        if self.config.verbose:
            print(nest_result.summary())

        return nest_result

    def _calculate_aci(self, state: MCTSState) -> Dict:
        """Calculate ACI for state (Γ₁)"""
        if self.aci_analyzer is None:
            # Return default if no analyzer
            return {'ACI': 0.5, 'confidence': 0.0}

        try:
            aci_result = self.aci_analyzer.calculate(state)
            return aci_result
        except Exception as e:
            if self.config.verbose:
                print(f"[Nest] ACI calculation failed: {e}")
            return {'ACI': 0.5, 'confidence': 0.0}

    def _create_agent_configs(self, aci_result: Dict) -> List[MCTSConfig]:
        """Create MCTS configurations for each agent strategy"""
        aci_score = aci_result.get('ACI', 0.5)
        configs = []

        for strategy in self.config.agent_strategies:
            config = self._create_config_for_strategy(strategy, aci_score)
            configs.append(config)

        return configs

    def _create_config_for_strategy(self, strategy: AgentStrategy, aci_score: float) -> MCTSConfig:
        """Create MCTS config for specific strategy"""
        base_config = MCTSConfig(
            max_iterations=self.config.mcts_iterations,
            max_playout_depth=self.config.mcts_playout_depth,
            verbose=self.config.verbose,
            aci_guided=self.config.aci_guided
        )

        if strategy == AgentStrategy.EXPLOIT:
            # Heavy exploitation: low C, trust ACI
            base_config.exploration_constant = 0.7
            base_config.adaptive_c = False  # Fixed low C
            base_config.playout_strategy = PlayoutStrategy.CAUSALLY_GUIDED

        elif strategy == AgentStrategy.EXPLORE:
            # Heavy exploration: high C, don't trust ACI
            base_config.exploration_constant = 2.0
            base_config.adaptive_c = False  # Fixed high C
            base_config.playout_strategy = PlayoutStrategy.RANDOM

        elif strategy == AgentStrategy.BALANCED:
            # Balanced: standard UCB
            base_config.exploration_constant = 1.41
            base_config.adaptive_c = False
            base_config.playout_strategy = PlayoutStrategy.HEURISTIC_GUIDED

        elif strategy == AgentStrategy.ADAPTIVE:
            # ACI-adaptive: adjust parameters based on ACI
            base_config.adaptive_c = True
            base_config.playout_strategy = PlayoutStrategy.ADAPTIVE

        return base_config

    def _run_agents(self,
                   initial_state: MCTSState,
                   action_generator: Callable,
                   state_transition: Callable,
                   value_function: Callable,
                   configs: List[MCTSConfig]) -> List[AgentResult]:
        """Run all MCTS agents (in parallel if configured)"""

        if self.config.parallel_agents and len(configs) > 1:
            return self._run_agents_parallel(
                initial_state, action_generator, state_transition,
                value_function, configs
            )
        else:
            return self._run_agents_sequential(
                initial_state, action_generator, state_transition,
                value_function, configs
            )

    def _run_agents_sequential(self,
                              initial_state: MCTSState,
                              action_generator: Callable,
                              state_transition: Callable,
                              value_function: Callable,
                              configs: List[MCTSConfig]) -> List[AgentResult]:
        """Run agents sequentially"""
        results = []

        for agent_id, config in enumerate(configs):
            strategy = self.config.agent_strategies[agent_id]

            if self.config.verbose:
                print(f"[Nest] Agent {agent_id} ({strategy.value}) starting...")

            mcts = MCTSSearch(config=config, aci_analyzer=self.aci_analyzer)

            best_node, search_info = mcts.search(
                initial_state,
                action_generator,
                state_transition,
                value_function
            )

            result = AgentResult(
                agent_id=agent_id,
                strategy=strategy,
                best_value=search_info['best_value'],
                best_node=best_node,
                search_info=search_info
            )

            results.append(result)

            if self.config.verbose:
                print(f"[Nest] Agent {agent_id} complete: value={result.best_value:.4f}")

        return results

    def _run_agents_parallel(self,
                            initial_state: MCTSState,
                            action_generator: Callable,
                            state_transition: Callable,
                            value_function: Callable,
                            configs: List[MCTSConfig]) -> List[AgentResult]:
        """Run agents in parallel"""
        results = []
        max_workers = min(len(configs), self.config.max_workers)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all agents
            futures = {}
            for agent_id, config in enumerate(configs):
                strategy = self.config.agent_strategies[agent_id]

                future = executor.submit(
                    self._run_single_agent,
                    agent_id,
                    strategy,
                    config,
                    initial_state,
                    action_generator,
                    state_transition,
                    value_function
                )
                futures[future] = (agent_id, strategy)

            # Collect results
            for future in as_completed(futures):
                agent_id, strategy = futures[future]
                result = future.result()
                results.append(result)

                if self.config.verbose:
                    print(f"[Nest] Agent {agent_id} ({strategy.value}) complete: "
                          f"value={result.best_value:.4f}")

        # Sort by agent_id
        results.sort(key=lambda r: r.agent_id)

        return results

    def _run_single_agent(self,
                         agent_id: int,
                         strategy: AgentStrategy,
                         config: MCTSConfig,
                         initial_state: MCTSState,
                         action_generator: Callable,
                         state_transition: Callable,
                         value_function: Callable) -> AgentResult:
        """Run a single agent (for parallel execution)"""
        mcts = MCTSSearch(config=config, aci_analyzer=self.aci_analyzer)

        best_node, search_info = mcts.search(
            initial_state,
            action_generator,
            state_transition,
            value_function
        )

        return AgentResult(
            agent_id=agent_id,
            strategy=strategy,
            best_value=search_info['best_value'],
            best_node=best_node,
            search_info=search_info
        )

    def _validate_agents(self, agent_results: List[AgentResult]) -> List[AgentResult]:
        """Validate agent results (Γ₃)"""
        for result in agent_results:
            # Extract value history from search info
            value_history = result.search_info.get('value_history', [])

            # Create validation result
            validation = self.validator.validate_mcts_results(
                results=[result.best_value],  # Single result
                value_history=value_history
            )

            result.validation = validation

        return agent_results

    def _aggregate_results(self, agent_results: List[AgentResult]) -> Tuple[AgentResult, Dict]:
        """Aggregate results from all agents"""

        # Filter confident results (if validated)
        confident_results = [r for r in agent_results if r.is_confident]
        if confident_results:
            candidates = confident_results
        else:
            candidates = agent_results

        # Best result
        best = max(candidates, key=lambda r: r.best_value)

        # Aggregated value (weighted by confidence if available)
        if self.validator and all(r.validation for r in candidates):
            # Weight by inverse CI width (narrower CI = higher weight)
            weights = []
            for r in candidates:
                ci_width = r.validation.confidence_interval.width
                weight = 1.0 / (ci_width + 1e-10)
                weights.append(weight)

            weights = np.array(weights)
            weights /= weights.sum()  # Normalize

            values = np.array([r.best_value for r in candidates])
            aggregated_value = float(np.dot(values, weights))

            # Overall confidence
            avg_ci_width = np.mean([r.validation.confidence_interval.width
                                   for r in candidates])
            confidence = 1.0 / (1.0 + avg_ci_width)

        else:
            # Simple average
            aggregated_value = np.mean([r.best_value for r in candidates])
            confidence = 0.5

        # Check convergence (all agents converged?)
        converged = all(r.search_info.get('converged', False)
                       for r in candidates)

        # Validation summary
        validation_summary = {
            'num_confident': len(confident_results),
            'num_total': len(agent_results),
            'best_strategy': best.strategy.value,
            'value_range': (min(r.best_value for r in agent_results),
                           max(r.best_value for r in agent_results))
        }

        aggregated = {
            'value': aggregated_value,
            'confidence': confidence,
            'converged': converged,
            'validation_summary': validation_summary
        }

        return best, aggregated


# Convenience function
def quick_nest_search(initial_state: MCTSState,
                     action_generator: Callable,
                     state_transition: Callable,
                     value_function: Callable,
                     num_agents: int = 4,
                     iterations_per_agent: int = 500) -> NestResult:
    """
    Convenience function for quick Monte Carlo Nest search.

    Args:
        initial_state: Starting state
        action_generator: Generate actions
        state_transition: Apply actions
        value_function: Evaluate states
        num_agents: Number of parallel agents
        iterations_per_agent: MCTS iterations per agent

    Returns:
        NestResult
    """
    config = NestConfig(
        num_agents=num_agents,
        mcts_iterations=iterations_per_agent,
        verbose=False
    )

    nest = MonteCarloNest(config)

    return nest.search(
        initial_state,
        action_generator,
        state_transition,
        value_function
    )


# Example usage
if __name__ == "__main__":
    print("Monte Carlo Nest (Stage 3 Integration) - Ready")
    print("=" * 60)

    # Simple example
    print("\nExample: Multi-agent optimization")
    print("-" * 60)

    # Define simple state space
    class SimpleState:
        def __init__(self, value=0, depth=0):
            self.value_val = value
            self.depth_val = depth

        @property
        def value(self):
            return self.value_val

        def is_terminal(self):
            return self.depth_val >= 10

    initial = SimpleState(value=0, depth=0)

    def actions(state):
        if state.depth_val >= 10:
            return []
        return ['+1', '-1']

    def transition(state, action):
        new_value = state.value_val + (1 if action == '+1' else -1)
        new_depth = state.depth_val + 1
        return SimpleState(new_value, new_depth)

    def value_fn(state):
        return state.value_val

    # Run nest
    config = NestConfig(
        num_agents=4,
        mcts_iterations=200,
        verbose=True
    )

    nest = MonteCarloNest(config)

    result = nest.search(initial, actions, transition, value_fn)

    print("\n" + result.summary())

    print("\n" + "=" * 60)
    print("Monte Carlo Nest - Test Complete")
