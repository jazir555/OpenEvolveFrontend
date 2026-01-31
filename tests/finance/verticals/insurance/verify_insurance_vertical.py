"""
Insurance Vertical Verification Script

Quick verification that all components are working correctly.

Author: AI Architecture Team
Date: 2026-01-30
"""

import sys
from datetime import datetime


def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")

    try:
        from openevolve.finance.verticals.insurance import (
            InsuranceReserveEvolver,
            RBCCalculator,
            StressScenarioGenerator,
            PortfolioConstraints,
            Portfolio,
            Bond,
            CreditRating
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False


def test_rbc_calculator():
    """Test RBC calculator"""
    print("\nTesting RBC Calculator...")

    try:
        from openevolve.finance.verticals.insurance import (
            RBCCalculator,
            Portfolio,
            Bond,
            CreditRating
        )

        # Create test portfolio
        portfolio = Portfolio(
            bonds=[
                Bond(
                    ticker="TEST1",
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
                )
            ],
            cash=10_000_000,
            total_value=115_000_000
        )

        # Calculate RBC
        calculator = RBCCalculator()
        rbc_ratio = calculator.calculate(
            portfolio_value=portfolio.total_value,
            liabilities=100_000_000,
            portfolio=portfolio
        )

        assert rbc_ratio > 0, "RBC ratio should be positive"
        print(f"✓ RBC Calculator working (RBC: {rbc_ratio:.2f}%)")
        return True

    except Exception as e:
        print(f"✗ RBC Calculator test failed: {e}")
        return False


def test_stress_generator():
    """Test stress scenario generator"""
    print("\nTesting Stress Scenario Generator...")

    try:
        from openevolve.finance.verticals.insurance import StressScenarioGenerator

        generator = StressScenarioGenerator()

        # Generate scenarios
        scenarios = generator.generate_all_scenarios()

        assert len(scenarios) > 0, "Should generate scenarios"

        for scenario in scenarios:
            assert scenario.name, "Scenario should have name"
            assert scenario.shocks, "Scenario should have shocks"
            assert scenario.duration_months > 0, "Duration should be positive"

        print(f"✓ Stress Scenario Generator working ({len(scenarios)} scenarios)")
        return True

    except Exception as e:
        print(f"✗ Stress Scenario Generator test failed: {e}")
        return False


def test_portfolio_constraints():
    """Test portfolio constraints"""
    print("\nTesting Portfolio Constraints...")

    try:
        from openevolve.finance.verticals.insurance import (
            PortfolioConstraints,
            Portfolio,
            Bond,
            CreditRating
        )

        # Create constraints
        constraints = PortfolioConstraints(
            max_duration=7.0,
            min_credit_quality="BBB-",
            max_concentration=0.30,
            min_diversification=20
        )

        # Create compliant portfolio
        portfolio = Portfolio(
            bonds=[
                Bond(
                    ticker=f"BOND{i}",
                    rating=CreditRating.AAA,
                    par_value=10_000_000,
                    market_value=10_500_000,
                    book_value=10_000_000,
                    duration=5.0,
                    convexity=50.0,
                    yield_to_maturity=0.04,
                    sector="Government",
                    coupon_rate=0.04,
                    maturity_date=datetime(2030, 1, 1)
                )
                for i in range(25)  # 25 bonds
            ],
            cash=20_000_000,
            total_value=282_500_000
        )

        # Check properties
        assert portfolio.duration <= constraints.max_duration, "Duration constraint"
        assert len(portfolio.bonds) >= constraints.min_diversification, "Diversification constraint"

        print("✓ Portfolio Constraints working")
        return True

    except Exception as e:
        print(f"✗ Portfolio Constraints test failed: {e}")
        return False


def test_evolver_initialization():
    """Test evolver initialization"""
    print("\nTesting Insurance Reserve Evolver...")

    try:
        from openevolve.finance.verticals.insurance import InsuranceReserveEvolver

        # Initialize evolver
        evolver = InsuranceReserveEvolver(config={
            "max_iterations": 10,
            "population_size": 10
        })

        assert evolver.max_iterations == 10
        assert evolver.population_size == 10
        assert evolver.rbc_calculator is not None
        assert evolver.stress_generator is not None

        print("✓ Insurance Reserve Evolver initialization successful")
        return True

    except Exception as e:
        print(f"✗ Evolver initialization test failed: {e}")
        return False


def main():
    """Run all verification tests"""
    print("=" * 80)
    print("INSURANCE VERTICAL VERIFICATION")
    print("=" * 80)

    tests = [
        ("Imports", test_imports),
        ("RBC Calculator", test_rbc_calculator),
        ("Stress Generator", test_stress_generator),
        ("Portfolio Constraints", test_portfolio_constraints),
        ("Evolver Initialization", test_evolver_initialization)
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} test crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total} tests")

    if passed == total:
        print("\n🎉 All tests passed! Insurance vertical is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
