"""
Test RBC Calculator

Unit tests for Risk-Based Capital calculation.

Author: AI Architecture Team
Date: 2026-01-30
"""

import pytest
from datetime import datetime

from openevolve.finance.verticals.insurance import (
    RBCCalculator,
    Portfolio,
    Bond,
    CreditRating
)


@pytest.fixture
def rbc_calculator():
    """Create RBC calculator instance"""
    return RBCCalculator()


@pytest.fixture
def simple_portfolio():
    """Create simple test portfolio"""
    bonds = [
        Bond(
            ticker="BOND1",
            rating=CreditRating.AAA,
            par_value=100_000_000,
            market_value=105_000_000,
            book_value=100_000_000,
            duration=5.0,
            convexity=50.0,
            yield_to_maturity=0.04,
            sector="Government",
            coupon_rate=0.04,
            maturity_date=datetime(2030, 1, 1)
        ),
        Bond(
            ticker="BOND2",
            rating=CreditRating.BBB,
            par_value=50_000_000,
            market_value=48_000_000,
            book_value=50_000_000,
            duration=4.0,
            convexity=40.0,
            yield_to_maturity=0.05,
            sector="Corporate",
            coupon_rate=0.05,
            maturity_date=datetime(2028, 1, 1)
        )
    ]

    return Portfolio(
        bonds=bonds,
        cash=10_000_000,
        total_value=163_000_000
    )


class TestRBCCalculatorBasics:
    """Test basic RBC calculator functionality"""

    def test_initialization(self, rbc_calculator):
        """Test calculator initialization"""
        assert rbc_calculator.minimum_rbc_ratio == 350.0
        assert rbc_calculator.RISK_FACTORS is not None
        assert len(rbc_calculator.RISK_FACTORS) > 0

    def test_calculate_simple(self, rbc_calculator, simple_portfolio):
        """Test simple RBC calculation"""
        rbc_ratio = rbc_calculator.calculate(
            portfolio_value=simple_portfolio.total_value,
            liabilities=100_000_000,
            portfolio=simple_portfolio
        )

        assert rbc_ratio > 0
        assert isinstance(rbc_ratio, float)

    def test_calculate_detailed(self, rbc_calculator, simple_portfolio):
        """Test detailed RBC calculation"""
        result = rbc_calculator.calculate_detailed(
            portfolio_value=simple_portfolio.total_value,
            liabilities=100_000_000,
            portfolio=simple_portfolio
        )

        # Check all fields are populated
        assert result.tac > 0
        assert result.rbc_required > 0
        assert result.rbc_ratio > 0
        assert result.c1_risk > 0  # Should have bond risk
        assert isinstance(result.compliant, bool)
        assert "action_level" in result.details


class TestTACCalculation:
    """Test Total Adjusted Capital calculation"""

    def test_tac_without_portfolio(self, rbc_calculator):
        """Test TAC calculation without portfolio (simple case)"""
        tac = rbc_calculator._calculate_tac(
            portfolio_value=1_000_000_000,
            portfolio=None
        )

        assert tac == 1_000_000_000

    def test_tac_with_portfolio_at_par(self, rbc_calculator, simple_portfolio):
        """Test TAC when all bonds at or above book value"""
        tac = rbc_calculator._calculate_tac(
            portfolio_value=simple_portfolio.total_value,
            portfolio=simple_portfolio
        )

        # Should be close to portfolio value (small AVR)
        assert tac > 0
        assert tac <= simple_portfolio.total_value


class TestRiskComponentCalculations:
    """Test individual risk component calculations"""

    def test_c0_risk(self, rbc_calculator, simple_portfolio):
        """Test C0 (affiliate) risk calculation"""
        c0 = rbc_calculator._calculate_c0_risk(simple_portfolio)
        assert c0 == 0  # No affiliate risk for bond portfolio

    def test_c1_risk_by_rating(self, rbc_calculator):
        """Test C1 (fixed income) risk varies by rating"""
        # AAA bond should have lower risk than BBB
        aaa_portfolio = Portfolio(
            bonds=[Bond(
                ticker="AAA_BOND",
                rating=CreditRating.AAA,
                par_value=100_000_000,
                market_value=100_000_000,
                book_value=100_000_000,
                duration=5.0,
                convexity=50.0,
                yield_to_maturity=0.04,
                sector="Government",
                coupon_rate=0.04,
                maturity_date=datetime(2030, 1, 1)
            )],
            cash=0,
            total_value=100_000_000
        )

        bbb_portfolio = Portfolio(
            bonds=[Bond(
                ticker="BBB_BOND",
                rating=CreditRating.BBB,
                par_value=100_000_000,
                market_value=100_000_000,
                book_value=100_000_000,
                duration=5.0,
                convexity=50.0,
                yield_to_maturity=0.05,
                sector="Corporate",
                coupon_rate=0.05,
                maturity_date=datetime(2030, 1, 1)
            )],
            cash=0,
            total_value=100_000_000
        )

        aaa_risk = rbc_calculator._calculate_c1_risk(aaa_portfolio)
        bbb_risk = rbc_calculator._calculate_c1_risk(bbb_portfolio)

        assert aaa_risk < bbb_risk  # AAA should have lower risk charge

    def test_c2_risk(self, rbc_calculator, simple_portfolio):
        """Test C2 (equity) risk calculation"""
        c2 = rbc_calculator._calculate_c2_risk(simple_portfolio)

        # Should be ~10% of portfolio value * 20% charge = 2% of portfolio
        expected_c2 = simple_portfolio.total_value * 0.10 * 0.20  # 2% of portfolio
        # Allow for reasonable variance (test expects actual calculation to match)
        assert abs(c2 - expected_c2) < (expected_c2 * 0.01)  # Within 1%

    def test_rbc_with_covariance(self, rbc_calculator):
        """Test RBC calculation with covariance adjustment"""
        # Simple case: only C1 risk
        rbc = rbc_calculator._calculate_rbc_with_covariance(
            c0=0,
            c1=10_000_000,
            c2=0,
            c3=0,
            c4=0
        )

        assert rbc > 0
        # Should be close to C1 when it's the only component
        assert 9_000_000 < rbc < 11_000_000


class TestActionLevels:
    """Test regulatory action level determination"""

    def test_compliant_level(self, rbc_calculator):
        """Test compliant action level (350%+)"""
        action = rbc_calculator._get_action_level(400)
        assert "Compliant" in action

    def test_monitoring_zone(self, rbc_calculator):
        """Test monitoring zone (250-350%)"""
        action = rbc_calculator._get_action_level(300)
        assert "Monitoring" in action

    def test_company_action_level(self, rbc_calculator):
        """Test company action level (200-250%)"""
        action = rbc_calculator._get_action_level(225)
        assert "Company Action" in action

    def test_regulatory_action_level(self, rbc_calculator):
        """Test regulatory action level (150-200%)"""
        action = rbc_calculator._get_action_level(175)
        assert "Regulatory Action" in action

    def test_authorized_control_level(self, rbc_calculator):
        """Test authorized control level (100-150%)"""
        action = rbc_calculator._get_action_level(125)
        assert "Authorized Control" in action

    def test_mandatory_control_level(self, rbc_calculator):
        """Test mandatory control level (<100%)"""
        action = rbc_calculator._get_action_level(75)
        assert "Mandatory Control" in action


class TestCapitalRequirements:
    """Test capital requirement calculations"""

    def test_capital_required_basic(self, rbc_calculator):
        """Test basic capital required calculation"""
        capital = rbc_calculator.calculate_capital_required(
            liabilities=1_000_000_000,
            target_rbc_ratio=350.0
        )

        assert capital > 0
        # Should be around $350M (10% RBC * 350%)
        assert 300_000_000 < capital < 400_000_000

    def test_capital_required_higher_ratio(self, rbc_calculator):
        """Test capital required with higher target ratio"""
        capital_350 = rbc_calculator.calculate_capital_required(
            liabilities=1_000_000_000,
            target_rbc_ratio=350.0
        )

        capital_500 = rbc_calculator.calculate_capital_required(
            liabilities=1_000_000_000,
            target_rbc_ratio=500.0
        )

        assert capital_500 > capital_350


class TestStressTesting:
    """Test RBC stress testing functionality"""

    def test_stress_test_basic(self, rbc_calculator, simple_portfolio):
        """Test basic stress test"""
        result = rbc_calculator.stress_test_rbc(
            portfolio=simple_portfolio,
            scenario_shocks={"corporate_spread": 300},
            liabilities=100_000_000
        )

        assert "rbc_ratio" in result
        assert "loss" in result
        assert "loss_percentage" in result
        assert "compliant" in result
        assert result["loss"] > 0

    def test_stress_test_spread_shock(self, rbc_calculator, simple_portfolio):
        """Test stress test with spread shock"""
        result = rbc_calculator.stress_test_rbc(
            portfolio=simple_portfolio,
            scenario_shocks={"corporate_spread": 500},
            liabilities=100_000_000
        )

        assert result["loss_percentage"] > 0
        # Larger spread should cause larger loss
        assert result["loss"] > 0

    def test_stress_test_no_shocks(self, rbc_calculator, simple_portfolio):
        """Test stress test with no shocks (baseline)"""
        result = rbc_calculator.stress_test_rbc(
            portfolio=simple_portfolio,
            scenario_shocks={},
            liabilities=100_000_000
        )

        # Should have minimal loss without shocks
        assert result["loss"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
