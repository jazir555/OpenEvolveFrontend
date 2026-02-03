"""
Insurance Reserve Evolver - Evolve bond portfolios that survive stress tests

LoongFlow Role: Plans regulatory stress test scenarios
OpenEvolve Role: Evolves bond portfolio that survives crises without breaching RBC

This module implements the core evolution engine for insurance reserve portfolios,
using a combination of LoongFlow's planning capabilities (for stress scenario
generation) and OpenEvolve's evolutionary algorithms (for portfolio optimization).

Author: AI Architecture Team
Date: 2026-01-30
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from .models import (
    Portfolio,
    Bond,
    PortfolioConstraints,
    StressScenario,
    StressTestResult,
    InsuranceEvolutionResult,
    ScenarioEvolutionResult,
    CreditRating
)
from .rbc_calculator import RBCCalculator
from .stress_generator import StressScenarioGenerator


logger = logging.getLogger(__name__)


class InsuranceReserveEvolver:
    """
    Evolve insurance reserve portfolios that survive regulatory stress tests.

    This class orchestrates the evolution of bond portfolios designed to
    withstand extreme market scenarios while maintaining regulatory compliance
    with Risk-Based Capital (RBC) requirements.

    Stress Scenarios (LoongFlow-planned):
    - 2008 GFC + COVID simultaneous
    - Interest rate spike (300bps in 3 months)
    - Corporate bond downgrade cascade
    - Mortality surge (+20% deaths)

    Constraints:
    - Maintain 350% RBC ratio through all scenarios
    - Duration < 7 years
    - Investment grade minimum (BBB-)

    Example:
        >>> evolver = InsuranceReserveEvolver(config={
        ...     "data_source": "CRSP_BOND_API",
        ...     "max_iterations": 100
        ... })
        >>> result = await evolver.evolve_reserve_portfolio(
        ...     reserve_requirements={
        ...         "policy_liabilities": 1_000_000_000,
        ...         "minimum_rbc": 350
        ...     },
        ...     constraints=PortfolioConstraints(
        ...         max_duration=7.0,
        ...         min_credit_quality="BBB-"
        ...     )
        ... )
        >>> print(f"Min RBC: {result.min_rbc_ratio}%")
        >>> print(f"Compliant: {result.regulatory_compliant}")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the insurance reserve evolver.

        Args:
            config: Configuration dictionary with keys:
                - data_source: Market data source (default: "mock")
                - max_iterations: Maximum evolution iterations (default: 100)
                - population_size: Population size for evolution (default: 50)
                - mutation_rate: Mutation rate (default: 0.1)
        """
        self.config = config or {}
        self.max_iterations = self.config.get("max_iterations", 100)
        self.population_size = self.config.get("population_size", 50)
        self.mutation_rate = self.config.get("mutation_rate", 0.1)

        # Initialize components
        self.rbc_calculator = RBCCalculator()
        self.stress_generator = StressScenarioGenerator()

        # Track policy liabilities
        self.policy_liabilities: Optional[float] = None

        logger.info("InsuranceReserveEvolver initialized with config: %s", self.config)

    async def evolve_reserve_portfolio(
        self,
        reserve_requirements: Dict[str, float],
        constraints: PortfolioConstraints
    ) -> InsuranceEvolutionResult:
        """
        Evolve portfolio that survives stress tests.

        This is the main entry point for the insurance reserve evolution process.
        It follows a three-phase approach:

        1. PLAN PHASE: LoongFlow generates comprehensive stress scenarios
        2. EXECUTE PHASE: OpenEvolve evolves portfolios for each scenario
        3. SUMMARIZE PHASE: Find the most robust portfolio across all scenarios

        Args:
            reserve_requirements:
                - policy_liabilities: Present value of future claims
                - minimum_rbc: Required RBC ratio (typically 350%)
            constraints:
                - max_duration: Maximum portfolio duration
                - min_credit_quality: Minimum rating (BBB-, Baa3)
                - max_concentration: Max exposure to any sector

        Returns:
            InsuranceEvolutionResult containing:
                - portfolio: The evolved portfolio
                - stress_test_results: Results for each scenario
                - min_rbc_ratio: Minimum RBC ratio across all scenarios
                - regulatory_compliant: Whether portfolio meets 350% threshold

        Example:
            >>> result = await evolver.evolve_reserve_portfolio(
            ...     reserve_requirements={
            ...         "policy_liabilities": 1_000_000_000,
            ...         "minimum_rbc": 350
            ...     },
            ...     constraints=PortfolioConstraints(max_duration=7.0)
            ... )
            >>> assert result.min_rbc_ratio >= 350
            >>> assert result.regulatory_compliant
        """
        logger.info("Starting insurance reserve portfolio evolution")
        logger.info("Policy liabilities: $%s", f"{reserve_requirements['policy_liabilities']:,.0f}")
        logger.info("Constraints: %s", constraints)

        # Store policy liabilities for RBC calculations
        self.policy_liabilities = reserve_requirements["policy_liabilities"]
        minimum_rbc = reserve_requirements.get("minimum_rbc", 350.0)

        # === PLAN PHASE: Generate stress scenarios ===
        logger.info("PLAN PHASE: Generating stress scenarios")
        stress_scenarios = await self._plan_stress_scenarios(
            reserve_requirements,
            constraints
        )
        logger.info("Generated %d stress scenarios", len(stress_scenarios))

        # === EXECUTE PHASE: Evolve portfolios for each scenario ===
        logger.info("EXECUTE PHASE: Evolving portfolios for scenarios")
        best_portfolios = []
        scenario_results = {}

        for i, scenario in enumerate(stress_scenarios):
            logger.info("Evolving for scenario %d/%d: %s",
                       i+1, len(stress_scenarios), scenario.name)

            result = await self._evolve_for_scenario(
                scenario=scenario,
                constraints=constraints
            )
            best_portfolios.append(result.best_portfolio)
            scenario_results[scenario.name] = result

        # === SUMMARIZE PHASE: Find robust portfolio ===
        logger.info("SUMMARIZE PHASE: Finding robust portfolio")
        robust_portfolio = await self._find_robust_portfolio(
            portfolios=best_portfolios,
            scenarios=stress_scenarios
        )

        # === VALIDATION: Run full stress test suite ===
        logger.info("VALIDATION: Running full stress test suite")
        validation = await self._validate_regulatory(
            portfolio=robust_portfolio,
            scenarios=stress_scenarios,
            minimum_rbc=minimum_rbc
        )

        # Compile results
        result = InsuranceEvolutionResult(
            portfolio=robust_portfolio,
            stress_test_results=validation.stress_results,
            min_rbc_ratio=validation.min_rbc,
            regulatory_compliant=validation.compliant,
            evolution_iterations=self.max_iterations,
            scenarios_tested=[s.name for s in stress_scenarios],
            metadata={
                "policy_liabilities": self.policy_liabilities,
                "minimum_rbc_required": minimum_rbc,
                "num_scenarios": len(stress_scenarios),
                "evolution_timestamp": datetime.now().isoformat()
            }
        )

        logger.info("Evolution complete. Min RBC: %.2f%%, Compliant: %s",
                   result.min_rbc_ratio, result.regulatory_compliant)

        return result

    async def _plan_stress_scenarios(
        self,
        requirements: Dict[str, float],
        constraints: PortfolioConstraints
    ) -> List[StressScenario]:
        """
        LoongFlow plans comprehensive stress scenarios.

        In a full implementation, this would use LoongFlow's planning capabilities
        to generate diverse, realistic stress scenarios. For now, we use the
        StressScenarioGenerator to create predefined scenarios.

        Args:
            requirements: Reserve requirements
            constraints: Portfolio constraints

        Returns:
            List of stress scenarios
        """
        # In production, would call LoongFlow:
        # loongflow_result = await self.loongflow.plan(
        #     task="plan_insurance_stress_scenarios",
        #     prompt=prompt
        # )
        # return loongflow_result.scenarios

        # For now, use predefined scenarios
        scenarios = []

        # Historical crises
        scenarios.append(self.stress_generator.gfc_plus_covid())
        scenarios.append(self.stress_generator.rate_shock_up())
        scenarios.append(self.stress_generator.rate_shock_down())
        scenarios.append(self.stress_generator.credit_downgrade_cascade())

        # Insurance-specific
        scenarios.append(self.stress_generator.mortality_surge())
        scenarios.append(self.stress_generator.natural_catastrophe())

        logger.info("Generated %d stress scenarios", len(scenarios))
        return scenarios

    async def _evolve_for_scenario(
        self,
        scenario: StressScenario,
        constraints: PortfolioConstraints
    ) -> ScenarioEvolutionResult:
        """
        OpenEvolve evolves portfolio for specific scenario.

        This uses evolutionary algorithms to find the optimal portfolio
        that maximizes RBC ratio under the given stress scenario.

        Args:
            scenario: Stress scenario to optimize for
            constraints: Portfolio constraints

        Returns:
            ScenarioEvolutionResult with best portfolio for this scenario
        """
        # Generate initial population
        population = self._generate_portfolio_variants(
            constraints=constraints,
            n_variants=self.population_size
        )

        # Evolution loop
        best_result = None
        best_score = float('-inf')

        for generation in range(self.max_iterations):
            # Evaluate each portfolio
            scored_population = []

            for portfolio in population:
                # Backtest through scenario
                stress_result = await self._simulate_stress_scenario(
                    portfolio=portfolio,
                    scenario=scenario
                )

                # Calculate fitness
                score = self._calculate_fitness(
                    stress_result=stress_result,
                    constraints=constraints
                )

                scored_population.append((portfolio, stress_result, score))

                # Track best
                if score > best_score:
                    best_score = score
                    best_result = scored_population[-1]

            # Selection
            scored_population.sort(key=lambda x: x[2], reverse=True)
            top_survivors = scored_population[:self.population_size // 2]

            # Crossover and mutation
            new_population = [p[0] for p in top_survivors]

            while len(new_population) < self.population_size:
                # Select parents
                parent1 = top_survivors[np.random.randint(len(top_survivors))][0]
                parent2 = top_survivors[np.random.randint(len(top_survivors))][0]

                # Crossover
                child = self._crossover_portfolios(parent1, parent2)

                # Mutate
                if np.random.random() < self.mutation_rate:
                    child = self._mutate_portfolio(child, constraints)

                # Validate
                if self._validate_constraints(child, constraints):
                    new_population.append(child)

            population = new_population

            if generation % 10 == 0:
                logger.info("Generation %d: Best score = %.2f, RBC = %.2f%%",
                           generation, best_score, best_result[1].rbc_ratio_final)

        # Compile results
        best_portfolio = best_result[0]
        best_stress_result = best_result[1]

        return ScenarioEvolutionResult(
            best_portfolio=best_portfolio,
            best_rbc=best_stress_result.rbc_ratio_final,
            all_results=[]  # Could store all results if needed
        )

    async def _find_robust_portfolio(
        self,
        portfolios: List[Portfolio],
        scenarios: List[StressScenario]
    ) -> Portfolio:
        """
        Find portfolio that performs best across all scenarios.

        Args:
            portfolios: List of candidate portfolios
            scenarios: All stress scenarios

        Returns:
            Most robust portfolio
        """
        best_portfolio = None
        best_min_rbc = float('inf')
        best_avg_rbc = 0.0

        for portfolio in portfolios:
            rbc_ratios = []

            # Test against all scenarios
            for scenario in scenarios:
                stress_result = await self._simulate_stress_scenario(
                    portfolio=portfolio,
                    scenario=scenario
                )
                rbc_ratios.append(stress_result.rbc_ratio_final)

            min_rbc = min(rbc_ratios)
            avg_rbc = np.mean(rbc_ratios)

            # Prefer portfolio with highest minimum RBC
            if min_rbc > best_min_rbc or \
               (min_rbc == best_min_rbc and avg_rbc > best_avg_rbc):
                best_min_rbc = min_rbc
                best_avg_rbc = avg_rbc
                best_portfolio = portfolio

        logger.info("Robust portfolio: Min RBC = %.2f%%, Avg RBC = %.2f%%",
                   best_min_rbc, best_avg_rbc)

        return best_portfolio

    async def _validate_regulatory(
        self,
        portfolio: Portfolio,
        scenarios: List[StressScenario],
        minimum_rbc: float
    ) -> Any:
        """
        Validate portfolio meets regulatory requirements.

        Args:
            portfolio: Portfolio to validate
            scenarios: All stress scenarios
            minimum_rbc: Required minimum RBC ratio

        Returns:
            Validation result
        """
        stress_results = {}
        min_rbc = float('inf')

        for scenario in scenarios:
            stress_result = await self._simulate_stress_scenario(
                portfolio=portfolio,
                scenario=scenario
            )
            stress_results[scenario.name] = stress_result
            min_rbc = min(min_rbc, stress_result.rbc_ratio_final)

        compliant = min_rbc >= minimum_rbc

        # Create validation result object
        class ValidationResult:
            def __init__(self, stress_results, min_rbc, compliant):
                self.stress_results = stress_results
                self.min_rbc = min_rbc
                self.compliant = compliant

        return ValidationResult(stress_results, min_rbc, compliant)

    async def _simulate_stress_scenario(
        self,
        portfolio: Portfolio,
        scenario: StressScenario
    ) -> StressTestResult:
        """
        Simulate stress scenario on portfolio.

        Args:
            portfolio: Portfolio to test
            scenario: Stress scenario

        Returns:
            StressTestResult
        """
        initial_value = portfolio.total_value

        # Apply scenario shocks
        losses = 0.0

        for bond in portfolio.bonds:
            # Apply credit shock (downgrades and defaults)
            if "defaults" in scenario.shocks:
                default_rate = self._get_default_rate(bond.rating, scenario.shocks["defaults"])
                losses += bond.market_value * default_rate

            # Apply spread shock
            if "corporate_bonds_oas" in scenario.shocks:
                spread_increase = scenario.shocks["corporate_bonds_oas"] / 10000  # bps to decimal
                duration_impact = bond.duration * spread_increase
                losses += bond.market_value * duration_impact

            # Apply equity shock (if applicable)
            if "equities" in scenario.shocks:
                # Assume some allocation to equities
                losses += portfolio.total_value * 0.1 * abs(scenario.shocks["equities"])

        # Apply interest rate shock
        if "treasury_yields" in scenario.shocks:
            rate_change = scenario.shocks["treasury_yields"] / 10000  # bps to decimal
            duration_impact = portfolio.duration * rate_change
            losses += portfolio.total_value * duration_impact

        final_value = initial_value - losses
        loss_percentage = (losses / initial_value) * 100 if initial_value > 0 else 0

        # Calculate RBC
        initial_rbc = self.rbc_calculator.calculate(
            portfolio_value=initial_value,
            liabilities=self.policy_liabilities,
            portfolio=portfolio
        )

        # Create stressed portfolio
        stressed_portfolio = Portfolio(
            bonds=[b for b in portfolio.bonds if b.market_value > 0],
            cash=portfolio.cash,
            total_value=final_value
        )

        final_rbc = self.rbc_calculator.calculate(
            portfolio_value=final_value,
            liabilities=self.policy_liabilities,
            portfolio=stressed_portfolio
        )

        return StressTestResult(
            scenario_name=scenario.name,
            initial_value=initial_value,
            final_value=final_value,
            loss_amount=losses,
            loss_percentage=loss_percentage,
            rbc_ratio_initial=initial_rbc,
            rbc_ratio_final=final_rbc,
            breaches_rbc=final_rbc < 350,
            details={
                "scenario_duration": scenario.duration_months,
                "shocks_applied": scenario.shocks
            }
        )

    def _get_default_rate(self, rating: CreditRating, defaults: Dict[str, float]) -> float:
        """Get default rate for rating"""
        rating_map = {
            CreditRating.AAA: "aaa",
            CreditRating.AA: "aa",
            CreditRating.A: "a",
            CreditRating.BBB: "bbb",
            CreditRating.BB: "bb",
            CreditRating.B: "b"
        }
        key = rating_map.get(rating, "bbb")
        return defaults.get(key, 0.0)

    def _calculate_fitness(
        self,
        stress_result: StressTestResult,
        constraints: PortfolioConstraints
    ) -> float:
        """
        Calculate fitness score for portfolio.

        Args:
            stress_result: Stress test result
            constraints: Portfolio constraints

        Returns:
            Fitness score (higher is better)
        """
        # Base score: RBC ratio
        score = stress_result.rbc_ratio_final

        # Heavy penalty for RBC breach
        if stress_result.breaches_rbc:
            score -= 1000

        # Penalty for large losses
        score -= stress_result.loss_percentage * 2

        # Bonus for low duration
        if stress_result.details.get("duration", 7.0) < constraints.max_duration:
            score += 5

        return score

    def _generate_portfolio_variants(
        self,
        constraints: PortfolioConstraints,
        n_variants: int
    ) -> List[Portfolio]:
        """Generate initial population of portfolios"""
        portfolios = []

        for _ in range(n_variants):
            # Generate random portfolio
            n_bonds = np.random.randint(
                constraints.min_diversification,
                constraints.min_diversification * 2
            )

            bonds = []
            total_value = 0.0

            for i in range(n_bonds):
                # Generate random bond
                rating = np.random.choice([
                    CreditRating.AAA, CreditRating.AA, CreditRating.A,
                    CreditRating.BBB
                ])

                par_value = np.random.uniform(10_000_000, 50_000_000)
                market_value = par_value * np.random.uniform(0.95, 1.05)
                book_value = par_value

                bond = Bond(
                    ticker=f"BOND{i}",
                    rating=rating,
                    par_value=par_value,
                    market_value=market_value,
                    book_value=book_value,
                    duration=np.random.uniform(2.0, constraints.max_duration),
                    convexity=np.random.uniform(50, 150),
                    yield_to_maturity=np.random.uniform(0.02, 0.06),
                    sector=np.random.choice(["Government", "Corporate", "Municipal", "MBS"]),
                    coupon_rate=np.random.uniform(0.02, 0.05),
                    maturity_date=datetime(2035, 1, 1)
                )

                bonds.append(bond)
                total_value += market_value

            cash = total_value * constraints.liquidity_requirement

            portfolio = Portfolio(
                bonds=bonds,
                cash=cash,
                total_value=total_value + cash
            )

            if self._validate_constraints(portfolio, constraints):
                portfolios.append(portfolio)

        return portfolios

    def _crossover_portfolios(self, parent1: Portfolio, parent2: Portfolio) -> Portfolio:
        """Combine two portfolios"""
        # Take bonds from both parents
        all_bonds = parent1.bonds + parent2.bonds

        # Randomly select subset
        n_bonds = min(len(all_bonds), np.random.randint(20, 40))
        selected_bonds = np.random.choice(all_bonds, n_bonds, replace=False).tolist()

        total_value = sum(b.market_value for b in selected_bonds)
        cash = total_value * 0.1

        return Portfolio(
            bonds=selected_bonds,
            cash=cash,
            total_value=total_value + cash
        )

    def _mutate_portfolio(self, portfolio: Portfolio, constraints: PortfolioConstraints) -> Portfolio:
        """Mutate portfolio"""
        # Randomly adjust one bond
        if not portfolio.bonds:
            return portfolio

        mutated_bonds = portfolio.bonds.copy()
        idx = np.random.randint(len(mutated_bonds))
        bond = mutated_bonds[idx]

        # Adjust duration
        bond.duration = np.clip(
            bond.duration + np.random.uniform(-0.5, 0.5),
            1.0,
            constraints.max_duration
        )

        # Adjust market value slightly
        bond.market_value *= np.random.uniform(0.95, 1.05)

        return Portfolio(
            bonds=mutated_bonds,
            cash=portfolio.cash,
            total_value=sum(b.market_value for b in mutated_bonds) + portfolio.cash
        )

    def _validate_constraints(self, portfolio: Portfolio, constraints: PortfolioConstraints) -> bool:
        """Validate portfolio meets constraints"""
        # Check duration
        if portfolio.duration > constraints.max_duration:
            return False

        # Check credit quality
        min_rating = CreditRating.from_string(constraints.min_credit_quality)
        if portfolio.credit_quality < min_rating:
            return False

        # Check diversification
        if len(portfolio.bonds) < constraints.min_diversification:
            return False

        # Check concentration
        sector_values = {}
        for bond in portfolio.bonds:
            sector_values[bond.sector] = sector_values.get(bond.sector, 0) + bond.market_value

        for sector_value in sector_values.values():
            if sector_value / portfolio.total_value > constraints.max_concentration:
                return False

        return True
