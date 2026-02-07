"""
Domain Optimizer Tests
Comprehensive tests for all 6 domain optimizers

Tests:
- Configuration validation
- Sub-domain configuration
- Domain-specific evaluation
- Constraint validation
- Utility methods

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
from openevolve.domain import (
    FinanceOptimizer,
    TradingOptimizer,
    ScienceOptimizer,
    EngineeringOptimizer,
    PharmaOptimizer,
    WebDesignOptimizer,
    detect_domain,
    get_optimizer
)
from openevolve.unified.config import EvolutionMode, DomainType


# ============================================================================
# TEST UTILITIES
# ============================================================================

class TestDomainDetection:
    """Test domain auto-detection"""

    def test_detect_finance_domain(self):
        """Test finance domain detection"""
        problem = "Optimize portfolio allocation for maximum return and minimum risk"
        domain = detect_domain(problem)
        assert domain == "finance"

    def test_detect_trading_domain(self):
        """Test trading domain detection"""
        problem = "Develop trading strategy with entry and exit rules"
        domain = detect_domain(problem)
        assert domain == "trading"

    def test_detect_science_domain(self):
        """Test science domain detection"""
        problem = "Design experiments to test scientific hypothesis"
        domain = detect_domain(problem)
        assert domain == "science"

    def test_detect_engineering_domain(self):
        """Test engineering domain detection"""
        problem = "Optimize structural design for minimum weight with FEA simulation"
        domain = detect_domain(problem)
        assert domain == "engineering"

    def test_detect_pharma_domain(self):
        """Test pharma domain detection"""
        problem = "Optimize molecule for high binding affinity and low toxicity"
        domain = detect_domain(problem)
        assert domain == "pharma"

    def test_detect_web_design_domain(self):
        """Test web design domain detection"""
        problem = "Optimize landing page for maximum conversion rate"
        domain = detect_domain(problem)
        assert domain == "web_design"

    def test_detect_general_domain(self):
        """Test general domain when no keywords match"""
        problem = "Solve optimization problem"
        domain = detect_domain(problem)
        assert domain == "general"


# ============================================================================
# FINANCE OPTIMIZER TESTS
# ============================================================================

class TestFinanceOptimizer:
    """Test Finance domain optimizer"""

    def test_finance_optimizer_init(self):
        """Test finance optimizer initialization"""
        optimizer = FinanceOptimizer()
        assert optimizer.domain_name == "finance"
        assert optimizer.sub_domain == "general"

    def test_finance_recommended_system(self):
        """Test finance system recommendation"""
        optimizer = FinanceOptimizer()
        assert optimizer.get_recommended_system() == "loongflow"

    def test_finance_recommended_mode(self):
        """Test finance mode recommendation"""
        optimizer = FinanceOptimizer()
        assert optimizer.get_recommended_mode() == "pes"

    def test_finance_domain_metrics(self):
        """Test finance domain metrics"""
        optimizer = FinanceOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "volatility" in metrics

    def test_finance_general_config(self):
        """Test general finance configuration"""
        optimizer = FinanceOptimizer()
        config = optimizer.get_default_config()

        # Compare by value since enums may be from different modules
        assert config.evolution_mode.value == EvolutionMode.PES.value
        assert config.pes.enabled is True
        assert config.max_iterations == 50

    def test_finance_portfolio_config(self):
        """Test portfolio sub-domain configuration"""
        optimizer = FinanceOptimizer(sub_domain="portfolio")
        config = optimizer.config

        assert config.mo.enabled is True
        assert "return" in config.mo.objectives
        assert "risk" in config.mo.objectives

    def test_finance_risk_config(self):
        """Test risk sub-domain configuration"""
        optimizer = FinanceOptimizer(sub_domain="risk")
        config = optimizer.config

        assert config.max_iterations == 40
        assert config.llm.plan_temperature == 0.5

    def test_finance_asset_allocation_config(self):
        """Test asset allocation sub-domain configuration"""
        optimizer = FinanceOptimizer(sub_domain="asset_allocation")
        config = optimizer.config

        assert config.max_iterations == 60
        assert config.pes.plan_iterations == 2

    def test_finance_evaluation(self):
        """Test finance-specific evaluation"""
        optimizer = FinanceOptimizer()
        solution = "AAPL = 0.3, GOOGL = 0.4, MSFT = 0.3"
        problem = "Optimize portfolio"
        metrics = optimizer.evaluate_solution(solution, problem)

        assert "sharpe_ratio" in metrics
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_finance_portfolio_parsing(self):
        """Test portfolio parsing"""
        optimizer = FinanceOptimizer()
        solution = "AAPL = 0.3, GOOGL = 0.4, MSFT = 0.3"
        portfolio = optimizer._parse_portfolio(solution)

        assert isinstance(portfolio, dict)
        # Would check specific parsing in production

    def test_finance_constraints(self):
        """Test finance constraints"""
        optimizer = FinanceOptimizer()
        constraints = optimizer.get_portfolio_constraints(
            max_assets=30,
            min_weight=0.01
        )

        assert constraints["max_assets"] == 30
        assert constraints["min_weight"] == 0.01

    def test_finance_validation(self):
        """Test portfolio validation"""
        optimizer = FinanceOptimizer()
        portfolio = {"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3}
        constraints = {"max_assets": 5}

        is_valid, violations = optimizer.validate_portfolio(portfolio, constraints)
        assert is_valid is True  # 3 assets <= 5
        assert len(violations) == 0

    def test_finance_sub_domain_list(self):
        """Test sub-domain listing"""
        optimizer = FinanceOptimizer()
        sub_domains = optimizer.list_sub_domains()

        assert "general" in sub_domains
        assert "portfolio" in sub_domains
        assert "risk" in sub_domains
        assert "asset_allocation" in sub_domains


# ============================================================================
# TRADING OPTIMIZER TESTS
# ============================================================================

class TestTradingOptimizer:
    """Test Trading domain optimizer"""

    def test_trading_optimizer_init(self):
        """Test trading optimizer initialization"""
        optimizer = TradingOptimizer()
        assert optimizer.domain_name == "trading"
        assert optimizer.sub_domain == "general"

    def test_trading_recommended_system(self):
        """Test trading system recommendation"""
        optimizer = TradingOptimizer()
        assert optimizer.get_recommended_system() == "openevolve"

    def test_trading_recommended_mode(self):
        """Test trading mode recommendation"""
        optimizer = TradingOptimizer()
        assert optimizer.get_recommended_mode() == "adversarial"

    def test_trading_domain_metrics(self):
        """Test trading domain metrics"""
        optimizer = TradingOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "win_rate" in metrics
        assert "profit_factor" in metrics

    def test_trading_general_config(self):
        """Test general trading configuration"""
        optimizer = TradingOptimizer()
        config = optimizer.get_default_config()

        assert config.evolution_mode == EvolutionMode.ADVERSARIAL
        assert config.adversarial.enabled is True
        assert config.adversarial.adversarial_rounds == 20

    def test_trading_strategy_config(self):
        """Test strategy sub-domain configuration"""
        optimizer = TradingOptimizer(sub_domain="strategy")
        config = optimizer.config

        assert config.adversarial.adversarial_rounds == 30
        assert config.llm.temperature == 0.9

    def test_trading_signal_config(self):
        """Test signal sub-domain configuration"""
        optimizer = TradingOptimizer(sub_domain="signal")
        config = optimizer.config

        assert config.adversarial.adversarial_rounds == 15
        assert config.llm.temperature == 0.6

    def test_trading_parameter_config(self):
        """Test parameter sub-domain configuration"""
        optimizer = TradingOptimizer(sub_domain="parameter")
        config = optimizer.config

        assert config.max_iterations == 100
        assert config.llm.temperature == 0.4

    def test_trading_evaluation(self):
        """Test trading-specific evaluation"""
        optimizer = TradingOptimizer()
        solution = "def strategy(): pass"
        problem = "Develop momentum strategy"
        metrics = optimizer.evaluate_solution(solution, problem)

        assert "sharpe_ratio" in metrics
        assert isinstance(metrics["sharpe_ratio"], float)

    def test_trading_scenario_generation(self):
        """Test adversarial scenario generation"""
        optimizer = TradingOptimizer()
        scenarios = optimizer.generate_adversarial_scenarios(None)

        assert len(scenarios) == 3
        assert any(s["type"] == "regime_change" for s in scenarios)
        assert any(s["type"] == "volatility_spike" for s in scenarios)

    def test_trading_validation(self):
        """Test strategy validation"""
        optimizer = TradingOptimizer()
        metrics = {"max_drawdown": 0.15, "win_rate": 0.55, "profit_factor": 2.2}
        constraints = {"max_drawdown": 0.2, "min_win_rate": 0.5}

        is_valid, violations = optimizer.validate_strategy(metrics, constraints)
        assert is_valid is True
        assert len(violations) == 0


# ============================================================================
# SCIENCE OPTIMIZER TESTS
# ============================================================================

class TestScienceOptimizer:
    """Test Science domain optimizer"""

    def test_science_optimizer_init(self):
        """Test science optimizer initialization"""
        optimizer = ScienceOptimizer()
        assert optimizer.domain_name == "science"

    def test_science_recommended_system(self):
        """Test science system recommendation"""
        optimizer = ScienceOptimizer()
        assert optimizer.get_recommended_system() == "hybrid"

    def test_science_domain_metrics(self):
        """Test science domain metrics"""
        optimizer = ScienceOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "statistical_power" in metrics
        assert "cost_efficiency" in metrics
        assert "discovery_rate" in metrics

    def test_science_general_config(self):
        """Test general science configuration"""
        optimizer = ScienceOptimizer()
        config = optimizer.get_default_config()

        assert config.evolution_mode == EvolutionMode.QD
        assert config.qd.enabled is True
        assert config.max_iterations == 20

    def test_science_experimental_design_config(self):
        """Test experimental design configuration"""
        optimizer = ScienceOptimizer(sub_domain="experimental_design")
        config = optimizer.config

        assert config.qd.grid_resolution == 20
        assert config.mo.enabled is True

    def test_science_data_analysis_config(self):
        """Test data analysis configuration"""
        optimizer = ScienceOptimizer(sub_domain="data_analysis")
        config = optimizer.config

        assert config.evolution_mode == EvolutionMode.STANDARD
        assert config.max_iterations == 50

    def test_science_doe_suggestions(self):
        """Test DOE parameter suggestions"""
        optimizer = ScienceOptimizer()
        doe_params = optimizer.suggest_doe_parameters(
            "Optimize reaction",
            num_factors=5,
            resolution="IV"
        )

        assert doe_params["design_type"] == "fractional_factorial"
        assert doe_params["num_factors"] == 5

    def test_science_cost_estimation(self):
        """Test experiment cost estimation"""
        optimizer = ScienceOptimizer()
        cost = optimizer.estimate_experiment_cost(
            {},
            {"max_experiments": 20, "cost_per_experiment": 2500}
        )

        assert cost == 50000


# ============================================================================
# ENGINEERING OPTIMIZER TESTS
# ============================================================================

class TestEngineeringOptimizer:
    """Test Engineering domain optimizer"""

    def test_engineering_optimizer_init(self):
        """Test engineering optimizer initialization"""
        optimizer = EngineeringOptimizer()
        assert optimizer.domain_name == "engineering"

    def test_engineering_recommended_system(self):
        """Test engineering system recommendation"""
        optimizer = EngineeringOptimizer()
        assert optimizer.get_recommended_system() == "hybrid"

    def test_engineering_domain_metrics(self):
        """Test engineering domain metrics"""
        optimizer = EngineeringOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "performance" in metrics
        assert "safety_margin" in metrics
        assert "reliability" in metrics

    def test_engineering_general_config(self):
        """Test general engineering configuration"""
        optimizer = EngineeringOptimizer()
        config = optimizer.get_default_config()

        assert config.evolution_mode == EvolutionMode.PES
        assert config.pes.enabled is True
        assert config.adversarial.enabled is True

    def test_engineering_structural_config(self):
        """Test structural configuration"""
        optimizer = EngineeringOptimizer(sub_domain="structural")
        config = optimizer.config

        assert config.mo.enabled is True
        assert "weight" in config.mo.objectives

    def test_engineering_safety_scenarios(self):
        """Test safety scenario generation"""
        optimizer = EngineeringOptimizer()
        scenarios = optimizer.generate_safety_scenarios("structural")

        assert len(scenarios) > 0
        assert any(s["type"] == "load_exceedance" for s in scenarios)

    def test_engineering_validation(self):
        """Test design validation"""
        optimizer = EngineeringOptimizer()
        metrics = {"safety_margin": 2.5, "weight": 0.65, "cost": 0.70}
        constraints = {"min_safety_factor": 2.0}

        is_valid, violations = optimizer.validate_design(metrics, constraints)
        assert is_valid is True


# ============================================================================
# PHARMA OPTIMIZER TESTS
# ============================================================================

class TestPharmaOptimizer:
    """Test Pharma domain optimizer"""

    def test_pharma_optimizer_init(self):
        """Test pharma optimizer initialization"""
        optimizer = PharmaOptimizer()
        assert optimizer.domain_name == "pharma"

    def test_pharma_recommended_system(self):
        """Test pharma system recommendation"""
        optimizer = PharmaOptimizer()
        assert optimizer.get_recommended_system() == "openevolve"

    def test_pharma_recommended_mode(self):
        """Test pharma mode recommendation"""
        optimizer = PharmaOptimizer()
        assert optimizer.get_recommended_mode() == "qd"

    def test_pharma_domain_metrics(self):
        """Test pharma domain metrics"""
        optimizer = PharmaOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "binding_affinity" in metrics
        assert "solubility" in metrics
        assert "toxicity" in metrics

    def test_pharma_general_config(self):
        """Test general pharma configuration"""
        optimizer = PharmaOptimizer()
        config = optimizer.get_default_config()

        assert config.evolution_mode == EvolutionMode.QD
        assert config.qd.enabled is True
        assert config.qd.archive_size == 10000

    def test_pharma_molecular_config(self):
        """Test molecular configuration"""
        optimizer = PharmaOptimizer(sub_domain="molecular")
        config = optimizer.config

        assert config.mo.enabled is True
        assert config.qd.grid_resolution == 25

    def test_pharma_validation(self):
        """Test molecule validation"""
        optimizer = PharmaOptimizer()
        metrics = {"toxicity": 0.25, "solubility": 0.75, "binding_affinity": 0.85}
        constraints = {"max_toxicity": 0.3, "min_solubility": 0.5}

        is_valid, violations = optimizer.validate_molecule(metrics, constraints)
        assert is_valid is True

    def test_pharma_drug_likeness(self):
        """Test drug-likeness calculation"""
        optimizer = PharmaOptimizer()
        molecule = {"molecular_weight": 400, "logp": 3, "hbd": 3, "hba": 6}
        score = optimizer.calculate_drug_likeness(molecule)

        assert 0.0 <= score <= 1.0


# ============================================================================
# WEB DESIGN OPTIMIZER TESTS
# ============================================================================

class TestWebDesignOptimizer:
    """Test Web Design domain optimizer"""

    def test_web_design_optimizer_init(self):
        """Test web design optimizer initialization"""
        optimizer = WebDesignOptimizer()
        assert optimizer.domain_name == "web_design"

    def test_web_design_recommended_system(self):
        """Test web design system recommendation"""
        optimizer = WebDesignOptimizer()
        assert optimizer.get_recommended_system() == "openevolve"

    def test_web_design_recommended_mode(self):
        """Test web design mode recommendation"""
        optimizer = WebDesignOptimizer()
        assert optimizer.get_recommended_mode() == "standard"

    def test_web_design_domain_metrics(self):
        """Test web design domain metrics"""
        optimizer = WebDesignOptimizer()
        metrics = optimizer.get_domain_metrics()
        assert "conversion_rate" in metrics
        assert "bounce_rate" in metrics
        assert "time_on_page" in metrics

    def test_web_design_general_config(self):
        """Test general web design configuration"""
        optimizer = WebDesignOptimizer()
        config = optimizer.get_default_config()

        # Compare by value since enums may be from different modules
        assert config.evolution_mode.value == EvolutionMode.STANDARD.value
        assert config.max_iterations == 100

    def test_web_design_landing_page_config(self):
        """Test landing page configuration"""
        optimizer = WebDesignOptimizer(sub_domain="landing_page")
        config = optimizer.config

        assert config.max_iterations == 150
        assert config.llm.temperature == 0.8

    def test_web_design_validation(self):
        """Test design validation"""
        optimizer = WebDesignOptimizer()
        metrics = {
            "load_time": 0.90,
            "accessibility_score": 0.85,
            "seo_score": 0.80
        }
        constraints = {"max_load_time": 3.0, "min_accessibility": 0.8}

        is_valid, violations = optimizer.validate_design(metrics, constraints)
        assert is_valid is True

    def test_web_design_suggestions(self):
        """Test improvement suggestions"""
        optimizer = WebDesignOptimizer()
        metrics = {"conversion_rate": 0.03}
        targets = {"conversion_rate": 0.08}

        suggestions = optimizer.suggest_improvements(metrics, targets)
        assert len(suggestions) > 0


# ============================================================================
# FACTORY TESTS
# ============================================================================

class TestOptimizerFactory:
    """Test optimizer factory functions"""

    def test_get_finance_optimizer(self):
        """Test getting finance optimizer"""
        optimizer = get_optimizer("finance")
        assert isinstance(optimizer, FinanceOptimizer)

    def test_get_trading_optimizer(self):
        """Test getting trading optimizer"""
        optimizer = get_optimizer("trading")
        assert isinstance(optimizer, TradingOptimizer)

    def test_get_science_optimizer(self):
        """Test getting science optimizer"""
        optimizer = get_optimizer("science")
        assert isinstance(optimizer, ScienceOptimizer)

    def test_get_engineering_optimizer(self):
        """Test getting engineering optimizer"""
        optimizer = get_optimizer("engineering")
        assert isinstance(optimizer, EngineeringOptimizer)

    def test_get_pharma_optimizer(self):
        """Test getting pharma optimizer"""
        optimizer = get_optimizer("pharma")
        assert isinstance(optimizer, PharmaOptimizer)

    def test_get_web_design_optimizer(self):
        """Test getting web design optimizer"""
        optimizer = get_optimizer("web_design")
        assert isinstance(optimizer, WebDesignOptimizer)

    def test_get_web_alias(self):
        """Test web alias for web_design"""
        optimizer = get_optimizer("web")
        assert isinstance(optimizer, WebDesignOptimizer)

    def test_get_default_optimizer(self):
        """Test default optimizer (finance)"""
        optimizer = get_optimizer("unknown")
        assert isinstance(optimizer, FinanceOptimizer)

    def test_get_optimizer_with_sub_domain(self):
        """Test getting optimizer with sub-domain"""
        optimizer = get_optimizer("finance", "portfolio")
        assert isinstance(optimizer, FinanceOptimizer)
        assert optimizer.sub_domain == "portfolio"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestDomainIntegration:
    """Integration tests for domain optimizers"""

    def test_all_optimizers_have_configs(self):
        """Test all optimizers have valid configurations"""
        optimizers = [
            FinanceOptimizer(),
            TradingOptimizer(),
            ScienceOptimizer(),
            EngineeringOptimizer(),
            PharmaOptimizer(),
            WebDesignOptimizer()
        ]

        for optimizer in optimizers:
            config = optimizer.get_default_config()
            assert config is not None
            assert config.domain.value == optimizer.domain_name

    def test_all_optimizers_have_metrics(self):
        """Test all optimizers have domain metrics"""
        optimizers = [
            FinanceOptimizer(),
            TradingOptimizer(),
            ScienceOptimizer(),
            EngineeringOptimizer(),
            PharmaOptimizer(),
            WebDesignOptimizer()
        ]

        for optimizer in optimizers:
            metrics = optimizer.get_domain_metrics()
            assert len(metrics) > 0
            assert isinstance(metrics, list)

    def test_all_optimizers_have_sub_domains(self):
        """Test all optimizers have sub-domain configurations"""
        optimizers = [
            ("finance", FinanceOptimizer()),
            ("trading", TradingOptimizer()),
            ("science", ScienceOptimizer()),
            ("engineering", EngineeringOptimizer()),
            ("pharma", PharmaOptimizer()),
            ("web_design", WebDesignOptimizer())
        ]

        for domain, optimizer in optimizers:
            sub_domains = optimizer.list_sub_domains()
            assert len(sub_domains) >= 3  # At least general + 2 sub-domains

            # Test each sub-domain has valid config
            for sub_domain in sub_domains:
                config = optimizer.get_sub_domain_config(sub_domain)
                assert config is not None

    def test_all_optimizers_can_evaluate(self):
        """Test all optimizers can evaluate solutions"""
        optimizers = [
            (FinanceOptimizer(), "AAPL = 0.5, GOOGL = 0.5"),
            (TradingOptimizer(), "def strategy(): pass"),
            (ScienceOptimizer(), "Run experiment with X=5"),
            (EngineeringOptimizer(), "Design: weight=100, strength=500"),
            (PharmaOptimizer(), "SMILES: CCO"),
            (WebDesignOptimizer(), "<html><body>Test</body></html>")
        ]

        for optimizer, solution in optimizers:
            metrics = optimizer.evaluate_solution(solution, "test problem")
            assert isinstance(metrics, dict)
            assert len(metrics) > 0
