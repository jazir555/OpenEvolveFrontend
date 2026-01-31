"""
Test DeFi Protocol Evolver
"""

import pytest
from datetime import datetime
from openevolve.finance.verticals.defi import (
    DeFiProtocolEvolver,
    ProtocolConstraints,
    ProtocolParameters,
    DeFiAttackScenario,
)
from openevolve.finance.verticals.defi.historical_exploits import (
    HISTORICAL_EXPLOITS,
    get_exploit_lessons,
    get_exploits_by_type,
    get_total_loss_by_type,
    get_comprehensive_summary,
)


@pytest.fixture
def basic_config():
    """Basic configuration for testing"""
    return {
        "population_size": 10,
        "generations": 2,
        "mutation_rate": 0.2,
        "elitism_rate": 0.1,
    }


@pytest.fixture
def basic_constraints():
    """Basic protocol constraints for testing"""
    return ProtocolConstraints(
        max_collateral_factor=0.80,
        min_liquidation_bonus=0.05,
        target_utilization=0.80,
        max_bad_debt_threshold=0.01,
        min_liquidity_threshold=1_000_000,
    )


@pytest.mark.asyncio
async def test_defi_evolver_initialization(basic_config):
    """Test DeFi evolver can be initialized"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    assert evolver is not None
    assert evolver.population_size == 10
    assert evolver.generations == 2
    assert evolver.defi_simulator is not None
    assert evolver.attack_generator is not None


@pytest.mark.asyncio
async def test_generate_parameter_variants(basic_config, basic_constraints):
    """Test parameter variant generation"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    variants = evolver._generate_parameter_variants(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"],
        constraints=basic_constraints,
        n_variants=10
    )

    assert len(variants) == 10

    # Check all variants meet constraints
    for params in variants:
        assert params.protocol == "compound"
        assert "ETH" in params.collateral_factors
        assert "USDC" in params.collateral_factors
        assert "WBTC" in params.collateral_factors

        # Check collateral factors within bounds
        for asset, cf in params.collateral_factors.items():
            assert 0.5 <= cf <= basic_constraints.max_collateral_factor

        # Check liquidation thresholds > collateral factors
        for asset in params.collateral_factors:
            cf = params.collateral_factors[asset]
            threshold = params.liquidation_thresholds.get(asset, 0)
            assert threshold > cf

        # Check liquidation bonuses
        for asset, bonus in params.liquidation_bonuses.items():
            assert bonus >= basic_constraints.min_liquidation_bonus


@pytest.mark.asyncio
async def test_validate_parameters(basic_config, basic_constraints):
    """Test parameter validation"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    # Valid parameters
    valid_params = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.75, "USDC": 0.80},
        liquidation_thresholds={"ETH": 0.85, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    validation = await evolver._validate_parameters(valid_params, basic_constraints)

    assert validation.meets_constraints is True
    assert len(validation.constraint_violations) == 0
    assert 0 <= validation.risk_score <= 100
    assert 0 <= validation.capital_efficiency_score <= 100

    # Invalid parameters (collateral factor too high)
    invalid_params = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.90, "USDC": 0.80},  # ETH CF too high
        liquidation_thresholds={"ETH": 0.95, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    validation = await evolver._validate_parameters(invalid_params, basic_constraints)

    assert validation.meets_constraints is False
    assert len(validation.constraint_violations) > 0
    assert any("collateral factor" in v.lower() for v in validation.constraint_violations)


@pytest.mark.asyncio
async def test_plan_attack_scenarios(basic_config):
    """Test attack scenario planning"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    scenarios = await evolver._plan_attack_scenarios(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"]
    )

    assert len(scenarios) > 0

    # Check scenarios have required fields
    for scenario in scenarios:
        assert isinstance(scenario, DeFiAttackScenario)
        assert scenario.name is not None
        assert scenario.description is not None
        assert scenario.attack_type is not None
        assert len(scenario.attack_steps) > 0
        assert scenario.expected_profit >= 0


@pytest.mark.asyncio
async def test_evolve_for_scenario(basic_config, basic_constraints):
    """Test evolution for single scenario"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    # Create simple attack scenario
    scenario = DeFiAttackScenario(
        name="test_flash_loan",
        description="Test flash loan attack",
        attack_type="flash_loan",
        attack_steps=[
            {"step": 1, "action": "flash_loan_borrow", "asset": "USDC", "amount": 1000000},
        ],
        expected_profit=100000,
        attack_vectors=["flash_loan"],
        difficulty="easy"
    )

    result = await evolver._evolve_for_scenario(
        scenario=scenario,
        protocol="compound",
        assets=["ETH", "USDC"],
        constraints=basic_constraints
    )

    assert result.scenario == scenario
    assert result.best_parameters is not None
    assert result.best_result is not None
    assert len(result.all_results) > 0


@pytest.mark.asyncio
async def test_find_robust_parameters(basic_config):
    """Test finding robust parameters across scenarios"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    # Create parameter sets
    params1 = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.75, "USDC": 0.80},
        liquidation_thresholds={"ETH": 0.85, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    params2 = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.70, "USDC": 0.75},
        liquidation_thresholds={"ETH": 0.80, "USDC": 0.85},
        liquidation_bonuses={"ETH": 0.10, "USDC": 0.08},
        price_oracle_type="twap",
        circuit_breaker_threshold=0.08,
        min_liquidity_required=2_000_000,
        max_price_impact=0.03
    )

    scenarios = [
        DeFiAttackScenario(
            name="test1",
            description="Test 1",
            attack_type="oracle_manipulation",
            attack_steps=[],
            expected_profit=0,
            attack_vectors=[],
        )
    ]

    robust_params = await evolver._find_robust_parameters(
        parameter_sets=[params1, params2],
        scenarios=scenarios
    )

    assert robust_params is not None
    assert robust_params.protocol == "compound"


@pytest.mark.asyncio
async def test_end_to_end_evolution(basic_config, basic_constraints):
    """Test end-to-end parameter evolution"""
    evolver = DeFiProtocolEvolver(config=basic_config)

    result = await evolver.evolve_protocol_parameters(
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"],
        constraints=basic_constraints
    )

    # Check result structure
    assert result is not None
    assert isinstance(result.parameters, ProtocolParameters)
    assert result.validation is not None
    assert isinstance(result.attack_survival, dict)
    assert result.historical_performance is not None
    assert 0 <= result.capital_efficiency <= 1

    # Check timestamp
    assert isinstance(result.timestamp, datetime)
    assert result.evolution_time >= 0

    # Check parameters
    assert result.parameters.protocol == "compound"
    assert "ETH" in result.parameters.collateral_factors
    assert "USDC" in result.parameters.collateral_factors
    assert "WBTC" in result.parameters.collateral_factors


def test_protocol_parameters_to_dict():
    """Test ProtocolParameters serialization"""
    params = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.75, "USDC": 0.80},
        liquidation_thresholds={"ETH": 0.85, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    params_dict = params.to_dict()

    assert params_dict["protocol"] == "compound"
    assert params_dict["collateral_factors"]["ETH"] == 0.75
    assert params_dict["price_oracle_type"] == "chainlink"
    assert params_dict["circuit_breaker_threshold"] == 0.10


def test_historical_exploits_database():
    """Test historical exploits database"""
    assert len(HISTORICAL_EXPLOITS) > 0

    # Check exploit structure
    for name, data in HISTORICAL_EXPLOITS.items():
        assert "date" in data
        assert "protocol" in data
        assert "attack_type" in data
        assert "loss_usd" in data
        assert "description" in data
        assert "lessons" in data


def test_get_exploit_lessons():
    """Test getting lessons from specific exploit"""
    lessons = get_exploit_lessons("bzx_2020")

    assert isinstance(lessons, list)
    assert len(lessons) > 0
    assert any("oracle" in lesson.lower() for lesson in lessons)


def test_get_exploits_by_type():
    """Test filtering exploits by type"""
    oracle_exploits = get_exploits_by_type("oracle_manipulation")

    assert len(oracle_exploits) > 0
    for name, data in oracle_exploits.items():
        assert data["attack_type"] == "oracle_manipulation"


def test_get_total_loss_by_type():
    """Test calculating total losses by type"""
    losses = get_total_loss_by_type()

    assert len(losses) > 0
    assert all(loss > 0 for loss in losses.values())

    # Oracle manipulation should have significant losses
    if "oracle_manipulation" in losses:
        assert losses["oracle_manipulation"] > 1_000_000


def test_get_comprehensive_summary():
    """Test comprehensive exploit summary"""
    summary = get_comprehensive_summary()

    assert "total_exploits" in summary
    assert "total_loss_usd" in summary
    assert "losses_by_attack_type" in summary
    assert "top_5_destructive" in summary
    assert "most_common_lessons" in summary

    assert summary["total_exploits"] > 0
    assert summary["total_loss_usd"] > 0
    assert len(summary["top_5_destructive"]) <= 5


@pytest.mark.asyncio
async def test_attack_scenario_generation():
    """Test attack scenario generation"""
    from openevolve.finance.verticals.defi.attack_generator import DeFiAttackGenerator

    generator = DeFiAttackGenerator()

    # Test flash loan attack
    flash_loan_scenario = generator.generate_flash_loan_attack(["ETH", "USDC", "WBTC"])
    assert flash_loan_scenario.attack_type == "flash_loan"
    assert len(flash_loan_scenario.attack_steps) > 0

    # Test oracle manipulation
    oracle_scenario = generator.generate_oracle_manipulation(["ETH", "USDC"])
    assert oracle_scenario.attack_type == "oracle_manipulation"
    assert len(oracle_scenario.attack_steps) > 0

    # Test cascading liquidation
    cascade_scenario = generator.generate_cascading_liquidation(["ETH", "USDC"])
    assert cascade_scenario.attack_type == "cascading_liquidation"

    # Test stablecoin de-peg
    depeg_scenario = generator.generate_stablecoin_depeg(["USDC", "ETH"])
    assert depeg_scenario.attack_type == "stablecoin_depeg"

    # Test reentrancy
    reentrancy_scenario = generator.generate_reentrancy_attack(["ETH", "USDC"])
    assert reentrancy_scenario.attack_type == "smart_contract_bug"


@pytest.mark.asyncio
async def test_comprehensive_attack_suite():
    """Test comprehensive attack suite generation"""
    from openevolve.finance.verticals.defi.attack_generator import DeFiAttackGenerator

    generator = DeFiAttackGenerator()
    scenarios = generator.generate_comprehensive_attack_suite(["ETH", "USDC", "WBTC"])

    assert len(scenarios) >= 5  # At least the basic scenarios

    # Check variety
    attack_types = set(s.attack_type for s in scenarios)
    assert "flash_loan" in attack_types
    assert "oracle_manipulation" in attack_types


@pytest.mark.asyncio
async def test_defi_simulator_attack():
    """Test DeFi simulator attack simulation"""
    from openevolve.finance.verticals.defi.defi_simulator import DeFiProtocolSimulator

    simulator = DeFiProtocolSimulator()

    params = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.75, "USDC": 0.80},
        liquidation_thresholds={"ETH": 0.85, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    attack = DeFiAttackScenario(
        name="test_attack",
        description="Test attack",
        attack_type="flash_loan",
        attack_steps=[
            {"step": 1, "action": "flash_loan_borrow", "asset": "USDC", "amount": 1000000},
        ],
        expected_profit=100000,
        attack_vectors=["flash_loan"],
    )

    result = await simulator.simulate_attack(
        parameters=params,
        protocol="compound",
        assets=["ETH", "USDC"],
        attack=attack
    )

    assert result is not None
    assert isinstance(result.survived, bool)
    assert isinstance(result.bad_debt, float)
    assert result.bad_debt >= 0


@pytest.mark.asyncio
async def test_defi_simulator_history():
    """Test DeFi simulator historical event simulation"""
    from openevolve.finance.verticals.defi.defi_simulator import DeFiProtocolSimulator

    simulator = DeFiProtocolSimulator()

    params = ProtocolParameters(
        protocol="compound",
        collateral_factors={"ETH": 0.75, "USDC": 0.80},
        liquidation_thresholds={"ETH": 0.85, "USDC": 0.90},
        liquidation_bonuses={"ETH": 0.08, "USDC": 0.05},
        price_oracle_type="chainlink",
        circuit_breaker_threshold=0.10,
        min_liquidity_required=1_000_000,
        max_price_impact=0.05
    )

    result = await simulator.simulate_history(
        parameters=params,
        protocol="compound",
        assets=["ETH", "USDC", "WBTC"]
    )

    assert result is not None
    assert len(result.event_results) > 0
    assert isinstance(result.avg_utilization, float)
    assert isinstance(result.max_bad_debt, float)
    assert isinstance(result.survived_all_events, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
