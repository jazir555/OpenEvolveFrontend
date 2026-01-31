"""
Three-Round Gauntlet Integration Example
========================================

Complete example showing how to integrate the 3-round gauntlet orchestrator
into an evolutionary optimization workflow.

Author: OpenEvolve Integration Examples
Date: 2026-01-30
"""

import asyncio
import logging
from typing import List, Dict, Any
from openevolve.gauntlets.three_round_orchestrator import (
    ThreeRoundGauntletOrchestrator,
    ThreeRoundConfig,
    FullGauntletResult,
    create_domain_config
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvolutionaryWorkflowWithGauntlet:
    """
    Example evolutionary workflow that uses 3-round gauntlet for solution filtering.

    This demonstrates how to integrate the gauntlet system into an optimization loop.
    """

    def __init__(self, domain: str = 'general'):
        """
        Initialize workflow with domain-specific configuration.

        Args:
            domain: Application domain (finance, science, web, etc.)
        """
        self.domain = domain

        # Get domain-tuned gauntlet configuration
        self.gauntlet_config = create_domain_config(domain)

        # Initialize orchestrator
        self.orchestrator = ThreeRoundGauntletOrchestrator(
            config=self.gauntlet_config
        )

        logger.info(f"Initialized workflow for domain: {domain}")
        logger.info(f"Round 1 threshold: {self.gauntlet_config.round1_threshold}")
        logger.info(f"Round 2 threshold: {self.gauntlet_config.round2_threshold}")
        logger.info(f"Round 3 threshold: {self.gauntlet_config.round3_threshold}")

    async def evaluate_solutions(
        self,
        solutions: List[str],
        problem: str
    ) -> List[FullGauntletResult]:
        """
        Evaluate multiple solutions through the gauntlet.

        Args:
            solutions: List of candidate solutions
            problem: Problem statement

        Returns:
            List of FullGauntletResult, sorted by score (descending)
        """
        logger.info(f"Evaluating {len(solutions)} solutions...")

        results = []

        for i, solution in enumerate(solutions, 1):
            logger.info(f"Evaluating solution {i}/{len(solutions)}")

            try:
                result = await self.orchestrator.run_full_gauntlet(
                    solution=solution,
                    problem=problem,
                    domain=self.domain
                )

                results.append(result)

                logger.info(
                    f"  Solution {i}: {'PASS' if result.passed else 'FAIL'}, "
                    f"score={result.final_score:.3f}, "
                    f"rounds={result.rounds_completed}"
                )

            except Exception as e:
                logger.error(f"  Solution {i} evaluation failed: {e}")

        # Sort by score (descending)
        results.sort(key=lambda r: r.final_score, reverse=True)

        return results

    async def evolve_with_gauntlet_filtering(
        self,
        problem: str,
        initial_population: List[str],
        generations: int = 5
    ) -> Dict[str, Any]:
        """
        Run evolutionary optimization with gauntlet-based filtering.

        The gauntlet acts as a quality filter, allowing only high-quality
        solutions to reproduce in subsequent generations.

        Args:
            problem: Problem to solve
            initial_population: Initial candidate solutions
            generations: Number of evolutionary generations

        Returns:
            Dictionary with best solution and evolution statistics
        """
        logger.info(f"Starting evolutionary optimization ({generations} generations)")

        population = initial_population
        generation_stats = []
        best_overall = None
        best_score = 0.0

        for gen in range(generations):
            logger.info(f"\n=== Generation {gen + 1}/{generations} ===")

            # Evaluate current population
            results = await self.evaluate_solutions(population, problem)

            # Track statistics
            passed_count = sum(1 for r in results if r.passed)
            avg_score = sum(r.final_score for r in results) / len(results)
            max_score = results[0].final_score if results else 0.0

            stats = {
                'generation': gen + 1,
                'population_size': len(population),
                'passed': passed_count,
                'pass_rate': passed_count / len(population) if population else 0.0,
                'avg_score': avg_score,
                'max_score': max_score
            }
            generation_stats.append(stats)

            logger.info(f"  Passed: {passed_count}/{len(population)} ({stats['pass_rate']*100:.1f}%)")
            logger.info(f"  Avg score: {avg_score:.3f}")
            logger.info(f"  Max score: {max_score:.3f}")

            # Update best overall
            if results and results[0].final_score > best_score:
                best_overall = results[0]
                best_score = results[0].final_score

            # Select survivors (only passed solutions)
            survivors = [r for r in results if r.passed]

            if not survivors:
                logger.warning("  No solutions passed! Using top 50% by score.")
                survivors = results[:len(results)//2]

            if gen < generations - 1:
                # Create next generation (mutation/crossover would go here)
                # For this example, just use survivors as next population
                population = [r.solution for r in survivors]

                # Ensure we don't lose diversity
                if len(population) < len(initial_population) // 2:
                    # Add some diversity by including some lower-scoring solutions
                    additional = [r.solution for r in results[len(survivors):len(survivors)+len(population)]]
                    population.extend(additional)

        logger.info("\n=== Evolution Complete ===")

        return {
            'best_solution': best_overall.solution if best_overall else None,
            'best_score': best_score,
            'generation_stats': generation_stats,
            'final_result': best_overall
        }

    def generate_evolution_report(self, evolution_result: Dict[str, Any]) -> str:
        """Generate comprehensive evolution report"""
        lines = [
            "=" * 80,
            "EVOLUTIONARY OPTIMIZATION WITH GAUNTLET FILTERING",
            "=" * 80,
            f"Domain: {self.domain}",
            "",
            "FINAL OUTCOME",
            "-" * 80,
            f"Best Score: {evolution_result['best_score']:.3f}",
            "",
        ]

        if evolution_result['final_result']:
            result = evolution_result['final_result']
            lines.extend([
                f"Rounds Completed: {result.rounds_completed}",
                f"Passed: {result.passed}",
                "",
                "PER-GENERATION STATISTICS",
                "-" * 80,
                "Gen | Population | Passed | Pass Rate | Avg Score | Max Score",
                "-" * 80
            ])

            for stats in evolution_result['generation_stats']:
                lines.append(
                    f"{stats['generation']:3d} | "
                    f"{stats['population_size']:9d} | "
                    f"{stats['passed']:6d} | "
                    f"{stats['pass_rate']*100:8.1f}% | "
                    f"{stats['avg_score']:8.3f} | "
                    f"{stats['max_score']:9.3f}"
                )

        lines.extend([
            "",
            "=" * 80,
            "END OF REPORT",
            "=" * 80
        ])

        return "\n".join(lines)


async def finance_trading_example():
    """Example: Trading strategy optimization"""
    print("\n" + "="*80)
    print("FINANCE TRADING STRATEGY OPTIMIZATION")
    print("="*80 + "\n")

    workflow = EvolutionaryWorkflowWithGauntlet(domain='finance')

    problem = """
    Develop a momentum-based trading strategy that:
    1. Identifies trending assets
    2. Generates buy/sell signals
    3. Manages risk through position sizing
    4. Handles market volatility
    """

    # Initial population of trading strategies
    population = [
        """
def momentum_strategy_v1(returns, lookback=20):
    import numpy as np
    momentum = returns.rolling(lookback).mean()
    signals = np.where(momentum > 0, 1, -1)
    return signals
""",
        """
def momentum_strategy_v2(returns, lookback=20, threshold=0.02):
    import numpy as np
    momentum = returns.rolling(lookback).mean()
    signals = np.where(abs(momentum) > threshold, np.sign(momentum), 0)
    return signals
""",
        """
def momentum_strategy_v3(returns, short_lookback=10, long_lookback=30):
    import numpy as np
    short_momentum = returns.rolling(short_lookback).mean()
    long_momentum = returns.rolling(long_lookback).mean()
    signals = np.where(short_momentum > long_momentum, 1, -1)
    return signals
"""
    ]

    # Run evolution
    result = await workflow.evolve_with_gauntlet_filtering(
        problem=problem,
        initial_population=population,
        generations=3
    )

    # Print report
    print(workflow.generate_evolution_report(result))


async def science_experimental_design_example():
    """Example: Scientific experimental design optimization"""
    print("\n" + "="*80)
    print("SCIENTIFIC EXPERIMENTAL DESIGN OPTIMIZATION")
    print("="*80 + "\n")

    workflow = EvolutionaryWorkflowWithGauntlet(domain='science')

    problem = """
    Design a randomized controlled trial (RCT) to test the efficacy of a new drug:
    1. Determine appropriate sample size
    2. Specify randomization method
    3. Define control measures
    4. Outline statistical analysis plan
    """

    population = [
        """
# RCT Design Version 1

Sample Size: 100 participants per group
Randomization: Simple randomization
Controls: Placebo only
Analysis: t-test comparing group means

Strengths: Simple design
Weaknesses: May be underpowered, no stratification
""",
        """
# RCT Design Version 2

Sample Size: 200 per group (power analysis: 80% power, α=0.05, d=0.5)
Randomization: Block randomization (block size=8)
Controls: Placebo + active comparator
Analysis: ANCOVA with baseline adjustment

Strengths: Adequate power, stratification
Weaknesses: More complex
""",
        """
# RCT Design Version 3

Sample Size: 150 per group (adaptive design)
Randomization: Stratified by age and severity
Controls: Placebo, dose-response arms
Analysis: Mixed-effects model with repeated measures

Strengths: Adaptive, comprehensive controls
Weaknesses: Complex statistical analysis
"""
    ]

    result = await workflow.evolve_with_gauntlet_filtering(
        problem=problem,
        initial_population=population,
        generations=3
    )

    print(workflow.generate_evolution_report(result))


async def web_component_example():
    """Example: Web component optimization"""
    print("\n" + "="*80)
    print("WEB COMPONENT OPTIMIZATION")
    print("="*80 + "\n")

    workflow = EvolutionaryWorkflowWithGauntlet(domain='web')

    problem = """
    Create a React user profile component with:
    1. Async data fetching
    2. Error handling
    3. Loading states
    4. Responsive design
    5. Accessibility features
    """

    population = [
        """
// UserProfile v1 - Basic
function UserProfile({userId}) {
  const [user, setUser] = useState(null);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(r => r.json())
      .then(setUser);
  }, [userId]);

  return user ? <div>{user.name}</div> : <div>Loading...</div>;
}
""",
        """
// UserProfile v2 - With error handling
function UserProfile({userId}) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/users/${userId}`)
      .then(r => r.json())
      .then(data => { setUser(data); setLoading(false); })
      .catch(err => { setError(err); setLoading(false); });
  }, [userId]);

  if (loading) return <Spinner />;
  if (error) return <Error message={error.message} />;
  return <div className="profile">{user.name}</div>;
}
""",
        """
// UserProfile v3 - Complete with accessibility
function UserProfile({userId}) {
  const [user, setUser] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchUser(userId).then(setUser).catch(setError).finally(() => setLoading(false));
  }, [userId]);

  return (
    <article className="user-profile" aria-busy={loading}>
      {loading && <Spinner aria-label="Loading user profile"/>}
      {error && <ErrorMessage error={error} />}
      {user && (
        <Fragment>
          <Avatar src={user.avatar} alt={`Photo of ${user.name}`}/>
          <h1>{user.name}</h1>
          <p>{user.bio}</p>
        </Fragment>
      )}
    </article>
  );
}
"""
    ]

    result = await workflow.evolve_with_gauntlet_filtering(
        problem=problem,
        initial_population=population,
        generations=3
    )

    print(workflow.generate_evolution_report(result))


async def main():
    """Run all examples"""
    print("\n" + "="*80)
    print("THREE-ROUND GAUNTLET INTEGRATION EXAMPLES")
    print("="*80)

    # Finance example
    await finance_trading_example()

    # Science example
    await science_experimental_design_example()

    # Web example
    await web_component_example()

    print("\n" + "="*80)
    print("ALL EXAMPLES COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
