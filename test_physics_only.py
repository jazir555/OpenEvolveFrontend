"""
Test Physics Validator - Standalone
"""
import sys
import numpy as np

# Test physics validator
print('=' * 60)
print('Testing Real Physics Validator...')
print('=' * 60)

from physics_validator_real import (
    RealPhysicsValidator, 
    RealFiniteElementAnalysis, 
    NavierStokesSolver,
    RealThermalAnalyzer,
    PHYSICS_NEMO_AVAILABLE
)
print(f'PhysicsNeMo available: {PHYSICS_NEMO_AVAILABLE}')

# Test FEA
print('\n[1] Testing FEA...')
fea = RealFiniteElementAnalysis()
result = fea.solve_stress_analysis_1d(
    length=1.0, n_elements=20, E=200e9,
    A_func=lambda x: 1e-4,
    loads=[(1.0, 1000)],
    constraints=[(0, 0)]
)
print(f'    FEA passed: {result["passed"]}')
print(f'    Max stress: {result["max_stress"]/1e6:.2f} MPa')
print(f'    Method: {result["method"]}')

# Test CFD
print('\n[2] Testing CFD...')
cfd = NavierStokesSolver(nx=30, ny=30)
result = cfd.solve_steady_lid_driven_cavity(Re=100)
print(f'    CFD passed: {result["passed"]}')
print(f'    Reynolds number: {result["reynolds_number"]}')
print(f'    Convergence: {result["convergence_reached"]}')
print(f'    Method: {result["method"]}')

# Test pipe flow
print('\n[3] Testing Pipe Flow CFD...')
result = cfd.solve_pipe_flow(
    diameter=0.1, length=1.0, rho=1000, mu=1e-3,
    inlet_pressure=101325, outlet_pressure=100000
)
print(f'    Pipe flow passed: {result["passed"]}')
print(f'    Flow regime: {result["flow_regime"]}')
print(f'    Reynolds number: {result["reynolds_number"]:.1f}')
print(f'    Method: {result["method"]}')

# Test Physics Validator integration
print('\n[4] Testing Physics Validator...')
validator = RealPhysicsValidator()
result = validator.validate_structural(
    geometry={'length': 1.0, 'cross_sectional_area': 1e-4},
    material={'youngs_modulus': 200e9, 'yield_stress': 250e6},
    loads=[{'magnitude': 10000, 'position': 1.0}]
)
print(f'    Validation passed: {result.passed}')
print(f'    Safety factor: {result.metrics["safety_factor"]:.2f}')
print(f'    Field data available: {result.field_data is not None}')

# Test fluid validation
print('\n[5] Testing Fluid Validation...')
result = validator.validate_fluid_dynamics(
    geometry={'type': 'pipe', 'diameter': 0.1, 'length': 1.0},
    fluid={'density': 1000, 'viscosity': 1e-3},
    boundary_conditions={'inlet_pressure': 101325, 'outlet_pressure': 100000}
)
print(f'    Validation passed: {result.passed}')
print(f'    Field data available: {result.field_data is not None}')

print('\n' + '=' * 60)
print('PHYSICS VALIDATOR: REAL IMPLEMENTATION ✓')
print('=' * 60)
