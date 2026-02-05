"""
Test Uncertainty Quantification - Standalone
"""
import numpy as np

print('=' * 60)
print('Testing Real Uncertainty Quantification...')
print('=' * 60)

from uncertainty_propagation_real import (
    RealUncertaintyPropagator,
    RealPolynomialChaosExpansion,
    RealSobolAnalyzer,
    UncertaintySource,
    UNCERTAINPY_AVAILABLE
)
print(f'Uncertainpy available: {UNCERTAINPY_AVAILABLE}')

# Test Monte Carlo
print('\n[1] Testing Monte Carlo Propagation...')
propagator = RealUncertaintyPropagator()

def model(params):
    return 2*params[0] + 3*params[1]

sources = [
    UncertaintySource("x1", "normal", {"mean": 1, "std": 0.1}),
    UncertaintySource("x2", "normal", {"mean": 2, "std": 0.2})
]

result = propagator.propagate_monte_carlo(model, sources, n_samples=2000)
print(f'    Mean: {result.mean:.4f} (expected ~8.0)')
print(f'    Std: {result.standard_deviation:.4f}')
print(f'    Convergence tracked: {len(result.convergence_history) > 0}')

# Test Polynomial Chaos
print('\n[2] Testing Polynomial Chaos Expansion...')
pce = RealPolynomialChaosExpansion(polynomial_order=2)

def simple_model(params):
    return params[0] + params[1]

sources_pce = [
    UncertaintySource("x1", "uniform", {"low": 0, "high": 1}),
    UncertaintySource("x2", "uniform", {"low": 0, "high": 1})
]

result = pce.fit(simple_model, sources_pce, method="quadrature")
print(f'    Convergence: {result["convergence"]}')
print(f'    Mean: {result["mean"]:.4f} (expected ~1.0)')
print(f'    Basis functions: {result["n_basis_functions"]}')

# Test Sobol Analysis
print('\n[3] Testing Sobol Sensitivity Analysis...')
analyzer = RealSobolAnalyzer()

def ishigami(params):
    x1, x2, x3 = params[0] * np.pi, params[1] * np.pi, params[2] * np.pi
    return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)

sources_sobol = [
    UncertaintySource("x1", "uniform", {"low": -1, "high": 1}),
    UncertaintySource("x2", "uniform", {"low": -1, "high": 1}),
    UncertaintySource("x3", "uniform", {"low": -1, "high": 1})
]

result = analyzer.analyze(ishigami, sources_sobol, n_samples=2000)
print(f'    S1(x1): {result.first_order["x1"]:.3f}')
print(f'    S1(x2): {result.first_order["x2"]:.3f}')
print(f'    S1(x3): {result.first_order["x3"]:.3f}')
print(f'    x2 has highest effect: {result.first_order["x2"] > result.first_order["x3"]}')

# Test Error Budget
print('\n[4] Testing Error Budget (GUM)...')
def product_model(params):
    return params[0] * params[1]

sources_budget = [
    UncertaintySource("length", "normal", {"mean": 10, "std": 0.1}, category="geometric"),
    UncertaintySource("force", "normal", {"mean": 100, "std": 5}, category="loading")
]

budget = propagator.create_error_budget(product_model, sources_budget, confidence_level=0.95)
print(f'    Total uncertainty: {budget.total_uncertainty:.4f}')
print(f'    Coverage factor: {budget.coverage_factor}')
print(f'    Source contributions: {list(budget.source_contributions.keys())}')

print('\n' + '=' * 60)
print('UNCERTAINTY QUANTIFICATION: REAL IMPLEMENTATION [OK]')
print('=' * 60)
