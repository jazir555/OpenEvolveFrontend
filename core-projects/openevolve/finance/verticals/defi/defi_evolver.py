"""
DeFiProtocolEvolver - Evolve lending protocol parameters

LoongFlow Role: Plans black swan attacks (oracle manipulation, flash loan cascading)
OpenEvolve Role: Evolves parameter settings that survive historical exploits
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import asyncio

from openevolve.finance.verticals.defi.base_evolution_agent import FinancialEvolutionAgent


@dataclass
class ProtocolConstraints:
    """Constraints for protocol parameters"""
    max_collateral_factor: float  # Maximum CF (e.g., 0.80 for 80%)
    min_liquidation_bonus: float  # Minimum bonus (e.g., 0.05 for 5%)
    target_utilization: float  # Target capital efficiency (e.g., 0.80)
    max_bad_debt_threshold: float = 0.01  # 1% of TVL
    min_liquidity_threshold: float = 1_000_000  # $1M minimum


@dataclass
class ProtocolParameters:
    """Evolved protocol parameters"""
    protocol: str
    collateral_factors: Dict[str, float]  # asset -> CF
    liquidation_thresholds: Dict[str, float]  # asset -> threshold
    liquidation_bonuses: Dict[str, float]  # asset -> bonus
    price_oracle_type: str  # "spot", "twap", "median", "chainlink"
    circuit_breaker_threshold: float  # Price change threshold
    min_liquidity_required: float
    max_price_impact: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "protocol": self.protocol,
            "collateral_factors": self.collateral_factors,
            "liquidation_thresholds": self.liquidation_thresholds,
            "liquidation_bonuses": self.liquidation_bonuses,
            "price_oracle_type": self.price_oracle_type,
            "circuit_breaker_threshold": self.circuit_breaker_threshold,
            "min_liquidity_required": self.min_liquidity_required,
            "max_price_impact": self.max_price_impact,
        }


@dataclass
class DeFiAttackScenario:
    """Attack scenario for testing"""
    name: str
    description: str
    attack_type: str
    attack_steps: List[Dict[str, Any]]
    expected_profit: float
    attack_vectors: List[str]
    difficulty: str = "medium"  # easy, medium, hard, extreme


@dataclass
class DeFiAttackResult:
    """Result of attack simulation"""
    survived: bool
    attacker_profit: float
    protocol_loss: float
    bad_debt: float
    failure_point: Optional[Dict[str, Any]]
    capital_efficiency: float = 0.0
    utilization: float = 0.0


@dataclass
class DeFiScenarioResult:
    """Result of evolution for one scenario"""
    scenario: DeFiAttackScenario
    best_parameters: ProtocolParameters
    best_result: DeFiAttackResult
    all_results: List[Tuple[ProtocolParameters, DeFiAttackResult, float]]


@dataclass
class ParameterValidation:
    """Validation results for parameters"""
    meets_constraints: bool
    constraint_violations: List[str]
    scenario_results: Dict[str, bool]
    risk_score: float  # 0-100, lower is better
    capital_efficiency_score: float  # 0-100, higher is better


@dataclass
class HistoricalSimulation:
    """Historical event simulation results"""
    event_results: List[Dict[str, Any]]
    avg_utilization: float
    max_bad_debt: float
    survived_all_events: bool
    total_events: int = 0


@dataclass
class DeFiEvolutionResult:
    """Complete evolution result"""
    parameters: ProtocolParameters
    validation: ParameterValidation
    attack_survival: Dict[str, bool]
    historical_performance: HistoricalSimulation
    capital_efficiency: float
    evolution_time: float = 0.0
    generations: int = 0
    timestamp: datetime = field(default_factory=datetime.utcnow)


class DeFiProtocolEvolver(FinancialEvolutionAgent):
    """
    Evolve DeFi lending protocol parameters that survive exploits.

    Protocols:
    - Compound, Aave, Venus (lending)
    - Uniswap, Curve (DEX)

    Parameters to evolve:
    - Collateral factors (how much can you borrow?)
    - Liquidation thresholds (when to liquidate?)
    - Liquidation bonuses (incentive for liquidators)
    - Price oracle choices (TWAP, median, specialized)
    - Circuit breaker thresholds

    Attack scenarios:
    - Oracle manipulation (spot price spikes)
    - Flash loan attacks (collateral -> borrow -> dump -> repay)
    - Cascading liquidations (systemic risk)
    - Token peg failures (stablecoin de-peg)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        from openevolve.finance.verticals.defi.defi_simulator import DeFiProtocolSimulator
        from openevolve.finance.verticals.defi.attack_generator import DeFiAttackGenerator
        from openevolve.finance.verticals.defi.historical_exploits import HISTORICAL_EXPLOITS

        self.defi_simulator = DeFiProtocolSimulator()
        self.attack_generator = DeFiAttackGenerator()
        self.historical_exploits = HISTORICAL_EXPLOITS

        # Evolution parameters
        self.population_size = config.get("population_size", 100)
        self.generations = config.get("generations", 50)
        self.mutation_rate = config.get("mutation_rate", 0.2)
        self.elitism_rate = config.get("elitism_rate", 0.1)

    async def evolve_protocol_parameters(
        self,
        protocol: str,
        assets: List[str],
        constraints: ProtocolConstraints
    ) -> DeFiEvolutionResult:
        """
        Evolve protocol parameters that survive attacks.

        Args:
            protocol: "compound", "aave", "venus", etc.
            assets: List of supported assets (e.g., ["ETH", "USDC", "WBTC"])
            constraints:
                - max_collateral_factor: Maximum CF (e.g., 80%)
                - min_liquidation_bonus: Minimum bonus (e.g., 5%)
                - target_utilization: Target capital efficiency (e.g., 80%)

        Returns:
            Parameter settings that survived all attack scenarios
        """
        start_time = datetime.utcnow()

        # === PLAN PHASE: LoongFlow generates attack scenarios ===
        self.logger.info(f"Planning attack scenarios for {protocol}...")
        attack_scenarios = await self._plan_attack_scenarios(
            protocol=protocol,
            assets=assets
        )

        self.logger.info(f"Generated {len(attack_scenarios)} attack scenarios")

        # === EXECUTE PHASE: OpenEvolve evolves parameters ===
        best_parameters = []

        for generation in range(self.generations):
            self.logger.info(f"Generation {generation + 1}/{self.generations}")

            # Evolve for each scenario
            for scenario in attack_scenarios:
                result = await self._evolve_for_scenario(
                    scenario=scenario,
                    protocol=protocol,
                    assets=assets,
                    constraints=constraints
                )
                best_parameters.append(result.best_parameters)

        # === SUMMARIZE PHASE: Find robust parameters ===
        self.logger.info("Finding robust parameters across all scenarios...")
        robust_parameters = await self._find_robust_parameters(
            parameter_sets=best_parameters,
            scenarios=attack_scenarios
        )

        # Validate meets constraints
        validation = await self._validate_parameters(
            robust_parameters,
            constraints
        )

        # Simulate historical performance
        self.logger.info("Simulating historical events...")
        historical_sim = await self.defi_simulator.simulate_history(
            parameters=robust_parameters,
            protocol=protocol,
            assets=assets
        )

        evolution_time = (datetime.utcnow() - start_time).total_seconds()

        return DeFiEvolutionResult(
            parameters=robust_parameters,
            validation=validation,
            attack_survival=validation.scenario_results,
            historical_performance=historical_sim,
            capital_efficiency=historical_sim.avg_utilization,
            evolution_time=evolution_time,
            generations=self.generations,
            timestamp=datetime.utcnow()
        )

    async def _plan_attack_scenarios(
        self,
        protocol: str,
        assets: List[str]
    ) -> List[DeFiAttackScenario]:
        """LoongFlow plans comprehensive attack scenarios"""

        # Get historical exploits from memory
        historical_exploits = self.memory.get_defi_exploits() if hasattr(self.memory, 'get_defi_exploits') else self.historical_exploits

        # Generate scenarios
        scenarios = []

        # 1. Flash loan attacks
        scenarios.append(self.attack_generator.generate_flash_loan_attack(assets))

        # 2. Oracle manipulation
        scenarios.append(self.attack_generator.generate_oracle_manipulation(assets))

        # 3. Cascading liquidations
        scenarios.append(self.attack_generator.generate_cascading_liquidation(assets))

        # 4. Stablecoin de-peg
        scenarios.append(self.attack_generator.generate_stablecoin_depeg(assets))

        # 5. Smart contract bugs
        scenarios.append(self.attack_generator.generate_reentrancy_attack(assets))

        # 6. Learn from historical exploits
        for exploit_name, exploit_data in historical_exploits.items():
            scenario = self.attack_generator.generate_historical_exploit_scenario(
                exploit_name,
                exploit_data,
                assets
            )
            if scenario:
                scenarios.append(scenario)

        # Use LoongFlow to enhance scenarios
        if self.loongflow:
            prompt = f"""
            You are planning attack scenarios for {protocol} lending protocol.

            Supported assets: {assets}

            Historical exploits to learn from:
            {self._format_exploits(historical_exploits)}

            We have generated {len(scenarios)} initial scenarios.

            Review and enhance these scenarios by:
            1. Identifying missing attack vectors
            2. Suggesting realistic parameter variations
            3. Considering multi-step attack combinations
            4. Assessing difficulty and likelihood

            Return enhanced scenarios as JSON.
            """

            try:
                loongflow_result = await self.loongflow.plan(
                    task="plan_defi_attack_scenarios",
                    prompt=prompt
                )

                if hasattr(loongflow_result, 'scenarios'):
                    scenarios = loongflow_result.scenarios
            except Exception as e:
                self.logger.warning(f"LoongFlow planning failed: {e}, using generated scenarios")

        return scenarios

    async def _evolve_for_scenario(
        self,
        scenario: DeFiAttackScenario,
        protocol: str,
        assets: List[str],
        constraints: ProtocolConstraints
    ) -> DeFiScenarioResult:
        """OpenEvolve evolves parameters for specific attack"""

        # Generate parameter variants
        parameter_sets = self._generate_parameter_variants(
            protocol=protocol,
            assets=assets,
            constraints=constraints,
            n_variants=self.population_size
        )

        # Simulate attack on each parameter set
        results = []
        for params in parameter_sets:
            result = await self.defi_simulator.simulate_attack(
                parameters=params,
                protocol=protocol,
                assets=assets,
                attack=scenario
            )
            results.append((params, result))

        # Score by attack resilience + capital efficiency
        scored_results = []
        for params, result in results:
            # Survival score (did protocol survive?)
            survival_score = 1000 if result.survived else -1000

            # Capital efficiency score (higher utilization = better)
            efficiency_score = result.capital_efficiency * 100

            # Risk score (lower bad debt = better)
            risk_score = -result.bad_debt * 1000

            # Combined score
            score = survival_score + efficiency_score + risk_score

            scored_results.append((params, result, score))

        # Rank by score
        scored_results.sort(key=lambda x: x[2], reverse=True)

        return DeFiScenarioResult(
            scenario=scenario,
            best_parameters=scored_results[0][0],
            best_result=scored_results[0][1],
            all_results=scored_results
        )

    def _generate_parameter_variants(
        self,
        protocol: str,
        assets: List[str],
        constraints: ProtocolConstraints,
        n_variants: int = 100
    ) -> List[ProtocolParameters]:
        """Generate random parameter variants within constraints"""

        variants = []

        oracle_types = ["spot", "twap", "median", "chainlink"]

        for i in range(n_variants):
            # Random collateral factors (0.5 to max)
            collateral_factors = {
                asset: np.random.uniform(0.5, constraints.max_collateral_factor)
                for asset in assets
            }

            # Liquidation thresholds (CF + 5% to CF + 15%)
            liquidation_thresholds = {
                asset: cf + np.random.uniform(0.05, 0.15)
                for asset, cf in collateral_factors.items()
            }

            # Liquidation bonuses (min to 15%)
            liquidation_bonuses = {
                asset: np.random.uniform(constraints.min_liquidation_bonus, 0.15)
                for asset in assets
            }

            # Price oracle type (prefer safer options)
            oracle_type = np.random.choice(
                oracle_types,
                p=[0.1, 0.3, 0.4, 0.2]  # Weight towards safer options
            )

            # Circuit breaker threshold (5% to 20%)
            circuit_breaker = np.random.uniform(0.05, 0.20)

            # Min liquidity (1M to 10M)
            min_liquidity = np.random.uniform(
                constraints.min_liquidity_threshold,
                10_000_000
            )

            # Max price impact (1% to 10%)
            max_price_impact = np.random.uniform(0.01, 0.10)

            variants.append(
                ProtocolParameters(
                    protocol=protocol,
                    collateral_factors=collateral_factors,
                    liquidation_thresholds=liquidation_thresholds,
                    liquidation_bonuses=liquidation_bonuses,
                    price_oracle_type=oracle_type,
                    circuit_breaker_threshold=circuit_breaker,
                    min_liquidity_required=min_liquidity,
                    max_price_impact=max_price_impact
                )
            )

        return variants

    async def _find_robust_parameters(
        self,
        parameter_sets: List[ProtocolParameters],
        scenarios: List[DeFiAttackScenario]
    ) -> ProtocolParameters:
        """Find parameters that perform well across all scenarios"""

        if not parameter_sets:
            raise ValueError("No parameter sets to evaluate")

        # Score each parameter set across all scenarios
        scored_params = []

        for params in parameter_sets:
            total_score = 0
            survival_count = 0

            for scenario in scenarios:
                result = await self.defi_simulator.simulate_attack(
                    parameters=params,
                    protocol=params.protocol,
                    assets=list(params.collateral_factors.keys()),
                    attack=scenario
                )

                if result.survived:
                    survival_count += 1
                    total_score += 1000  # Survival bonus

                total_score += result.capital_efficiency * 100
                total_score -= result.bad_debt * 1000

            # Prefer parameters that survive more scenarios
            total_score += survival_count * 500

            scored_params.append((params, total_score, survival_count))

        # Sort by score
        scored_params.sort(key=lambda x: x[1], reverse=True)

        # Return best
        return scored_params[0][0]

    async def _validate_parameters(
        self,
        parameters: ProtocolParameters,
        constraints: ProtocolConstraints
    ) -> ParameterValidation:
        """Validate parameters meet constraints"""

        violations = []

        # Check collateral factors
        for asset, cf in parameters.collateral_factors.items():
            if cf > constraints.max_collateral_factor:
                violations.append(
                    f"{asset} collateral factor {cf:.2f} exceeds max {constraints.max_collateral_factor:.2f}"
                )

        # Check liquidation bonuses
        for asset, bonus in parameters.liquidation_bonuses.items():
            if bonus < constraints.min_liquidation_bonus:
                violations.append(
                    f"{asset} liquidation bonus {bonus:.2f} below min {constraints.min_liquidation_bonus:.2f}"
                )

        # Check liquidation thresholds > collateral factors
        for asset in parameters.collateral_factors:
            cf = parameters.collateral_factors[asset]
            threshold = parameters.liquidation_thresholds.get(asset, 0)
            if threshold <= cf:
                violations.append(
                    f"{asset} liquidation threshold {threshold:.2f} must be > collateral factor {cf:.2f}"
                )

        meets_constraints = len(violations) == 0

        # Score risk (0-100, lower is better)
        risk_factors = []

        # High collateral factors increase risk
        avg_cf = np.mean(list(parameters.collateral_factors.values()))
        risk_factors.append(avg_cf * 50)  # 0-50 points

        # Weak oracle increases risk
        oracle_risk = {
            "chainlink": 0,
            "median": 10,
            "twap": 20,
            "spot": 50
        }
        risk_factors.append(oracle_risk.get(parameters.price_oracle_type, 30))

        # Loose circuit breaker increases risk
        risk_factors.append(parameters.circuit_breaker_threshold * 100)

        risk_score = min(100, sum(risk_factors))

        # Capital efficiency score (0-100, higher is better)
        efficiency_score = avg_cf * 100

        return ParameterValidation(
            meets_constraints=meets_constraints,
            constraint_violations=violations,
            scenario_results={},  # Will be populated after simulations
            risk_score=risk_score,
            capital_efficiency_score=efficiency_score
        )

    def _format_exploits(self, exploits: Dict[str, Dict[str, Any]]) -> str:
        """Format exploits for prompt"""
        formatted = []
        for name, data in exploits.items():
            formatted.append(
                f"- {name} ({data.get('date', 'unknown')}): "
                f"{data.get('attack_type', 'unknown')} - "
                f"${data.get('loss_usd', 0):,} loss"
            )
        return "\n".join(formatted)
