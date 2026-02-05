"""
Standalone Tests for Enhanced E2E Invention Planner Components

Tests only the enhanced components without requiring the full base planner.

Author: OpenEvolve
Version: 2.0.0
"""

import pytest
import numpy as np
import sys
import os

# Test physics validation
def test_physics_import():
    """Test that enhanced physics validator can be imported"""
    from physics_validator_enhanced import (
        EnhancedPhysicsValidator,
        PhysicsDomain,
        FEASimulator,
        CFDSimulator,
        ThermalAnalyzer
    )
    assert True

def test_fea_stress_analysis():
    """Test FEA stress analysis"""
    from physics_validator_enhanced import FEASimulator
    
    fea = FEASimulator()
    
    geometry = {
        'length': 2.0,
        'cross_sectional_area': 0.01,
        'surface_area': 0.5
    }
    material = {
        'youngs_modulus': 200e9,
        'yield_stress': 250e6,
        'poisson_ratio': 0.3
    }
    loads = [{'magnitude': 50000, 'direction': 'axial'}]
    
    result = fea.analyze_stress(geometry, material, loads, [])
    
    assert 'max_stress' in result
    assert 'safety_factor' in result
    assert result['safety_factor'] > 0
    print(f"FEA Result: max_stress={result['max_stress']:.2e} Pa, safety_factor={result['safety_factor']:.2f}")

def test_fea_modal_analysis():
    """Test FEA modal analysis"""
    from physics_validator_enhanced import FEASimulator
    
    fea = FEASimulator()
    
    geometry = {'mass': 10.0, 'length': 1.0, 'cross_sectional_area': 0.01}
    material = {'youngs_modulus': 200e9}
    
    result = fea.modal_analysis(geometry, material, num_modes=5)
    
    assert 'natural_frequencies' in result
    assert len(result['natural_frequencies']) == 5
    assert all(f > 0 for f in result['natural_frequencies'])
    print(f"Modal Analysis: frequencies={result['natural_frequencies']}")

def test_cfd_flow_simulation():
    """Test CFD flow simulation"""
    from physics_validator_enhanced import CFDSimulator
    
    cfd = CFDSimulator()
    
    geometry = {
        'length': 10.0,
        'diameter': 0.5,
        'characteristic_length': 0.5
    }
    fluid = {
        'density': 1000,
        'viscosity': 1e-3
    }
    bc = {'inlet_velocity': 2.0}
    
    result = cfd.simulate_flow(geometry, fluid, bc)
    
    assert 'reynolds_number' in result
    assert 'pressure_drop' in result
    assert 'flow_regime' in result
    assert result['flow_regime'] in ['laminar', 'turbulent']
    print(f"CFD Result: Re={result['reynolds_number']:.1f}, regime={result['flow_regime']}")

def test_thermal_steady_state():
    """Test steady-state thermal analysis"""
    from physics_validator_enhanced import ThermalAnalyzer
    
    thermal = ThermalAnalyzer()
    
    geometry = {'surface_area': 1.0, 'volume': 0.1}
    material = {
        'thermal_conductivity': 50,
        'density': 7850,
        'specific_heat': 420
    }
    heat_sources = [{'power': 1000, 'volume': 0.01}]
    boundary_temps = {'ambient': 300}
    
    result = thermal.steady_state_temperature(
        geometry, material, heat_sources, boundary_temps
    )
    
    assert 'max_temperature' in result
    assert result['max_temperature'] > boundary_temps['ambient']
    print(f"Thermal Result: max_temp={result['max_temperature']:.1f}K")

def test_enhanced_physics_validator():
    """Test enhanced physics validator"""
    from physics_validator_enhanced import EnhancedPhysicsValidator, PhysicsDomain
    
    validator = EnhancedPhysicsValidator()
    
    invention_spec = {
        'geometry': {
            'length': 2.0,
            'cross_sectional_area': 0.01,
            'surface_area': 0.5,
            'mass': 10.0
        },
        'material_properties': {
            'youngs_modulus': 200e9,
            'yield_stress': 250e6
        },
        'loads': [{'magnitude': 50000, 'direction': 'axial'}]
    }
    
    results = validator.validate_physics_comprehensive(
        invention_spec,
        [PhysicsDomain.STRUCTURAL, PhysicsDomain.MECHANICS]
    )
    
    assert 'structural' in results
    assert 'mechanics' in results
    print(f"Physics Validation: {len(results)} domains validated")


# Test uncertainty propagation
def test_uncertainty_import():
    """Test that enhanced uncertainty propagator can be imported"""
    from uncertainty_propagation_enhanced import (
        EnhancedUncertaintyPropagator,
        UncertaintySource,
        comprehensive_error_analysis
    )
    assert True

def test_uncertainty_source_sampling():
    """Test uncertainty source sampling"""
    from uncertainty_propagation_enhanced import UncertaintySource
    
    source = UncertaintySource(
        name="test_parameter",
        distribution="normal",
        parameters={'mean': 10.0, 'std': 1.0}
    )
    
    samples = source.sample(1000)
    
    assert len(samples) == 1000
    assert abs(np.mean(samples) - 10.0) < 0.5
    assert abs(np.std(samples) - 1.0) < 0.2
    print(f"Uncertainty Sampling: mean={np.mean(samples):.2f}, std={np.std(samples):.2f}")

def test_monte_carlo_propagation():
    """Test Monte Carlo uncertainty propagation"""
    from uncertainty_propagation_enhanced import (
        EnhancedUncertaintyPropagator,
        UncertaintySource
    )
    
    propagator = EnhancedUncertaintyPropagator(random_seed=42)
    
    def model(params):
        return params[0] + params[1]
    
    uncertainty_sources = [
        UncertaintySource("x1", "normal", {'mean': 10.0, 'std': 1.0}),
        UncertaintySource("x2", "normal", {'mean': 5.0, 'std': 0.5})
    ]
    
    result = propagator.propagate_monte_carlo(
        model, uncertainty_sources, n_samples=5000
    )
    
    assert abs(result.mean - 15.0) < 0.5
    assert result.standard_deviation > 0
    print(f"Monte Carlo: mean={result.mean:.2f}, std={result.standard_deviation:.2f}")

def test_sobol_sensitivity():
    """Test Sobol sensitivity analysis"""
    from uncertainty_propagation_enhanced import (
        EnhancedUncertaintyPropagator,
        UncertaintySource
    )
    
    propagator = EnhancedUncertaintyPropagator(random_seed=42)
    
    def ishigami(params):
        x1, x2, x3 = params[0], params[1], params[2]
        return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)
    
    uncertainty_sources = [
        UncertaintySource("x1", "uniform", {'low': -np.pi, 'high': np.pi}),
        UncertaintySource("x2", "uniform", {'low': -np.pi, 'high': np.pi}),
        UncertaintySource("x3", "uniform", {'low': -np.pi, 'high': np.pi})
    ]
    
    sobol = propagator.compute_sobol_indices(
        ishigami, uncertainty_sources, n_samples=3000
    )
    
    assert 'x1' in sobol.first_order
    assert 'x2' in sobol.first_order
    assert 'x3' in sobol.first_order
    print(f"Sobol Indices: x1={sobol.first_order['x1']:.3f}, x2={sobol.first_order['x2']:.3f}")


def test_error_budget():
    """Test error budget creation"""
    from uncertainty_propagation_enhanced import (
        EnhancedUncertaintyPropagator,
        UncertaintySource
    )
    
    propagator = EnhancedUncertaintyPropagator()
    
    def model(params):
        return params[0] * params[1]
    
    uncertainty_sources = [
        UncertaintySource("gain", "normal", {'mean': 2.0, 'std': 0.1}),
        UncertaintySource("offset", "normal", {'mean': 1.0, 'std': 0.05})
    ]
    
    budget = propagator.create_error_budget(
        model, uncertainty_sources, confidence_level=0.95
    )
    
    assert budget.total_uncertainty > 0
    assert budget.coverage_factor == 2.0
    print(f"Error Budget: total_unc={budget.total_uncertainty:.3f}, k={budget.coverage_factor}")


# Test SOP generation
import asyncio

@pytest.mark.asyncio
async def test_sop_import():
    """Test that enhanced SOP generator can be imported"""
    from sop_generator_enhanced import (
        EnhancedSOPGenerator,
        LLM4IASIntegration,
        IndustryStandard
    )
    assert True

@pytest.mark.asyncio
async def test_manufacturing_sop():
    """Test manufacturing SOP generation"""
    from sop_generator_enhanced import EnhancedSOPGenerator, IndustryStandard
    
    generator = EnhancedSOPGenerator()
    
    product_spec = {
        'name': 'Test Product',
        'critical_characteristics': ['dimension', 'weight'],
        'hazards': [
            {'type': 'mechanical', 'description': 'Moving parts', 'risk': 'Medium'}
        ]
    }
    
    result = await generator.generate_manufacturing_sop(
        product_name="Test Product",
        product_spec=product_spec,
        equipment_list=['Machine A', 'Machine B', 'Tool C'],
        industry_standard=IndustryStandard.ISO_9001,
        include_qc=True,
        include_safety=True
    )
    
    assert 'sop_type' in result
    assert 'manufacturing_process' in result
    print(f"Manufacturing SOP: {result['sop_type']}")

@pytest.mark.asyncio
async def test_assembly_sop():
    """Test assembly SOP generation"""
    from sop_generator_enhanced import EnhancedSOPGenerator
    
    generator = EnhancedSOPGenerator()
    
    bom = [
        {'part_number': '001', 'description': 'Base plate'},
        {'part_number': '002', 'description': 'Mounting bracket'}
    ]
    
    sequence = [
        {
            'description': 'Attach mounting bracket to base plate',
            'components': ['001', '002'],
            'tools': ['wrench'],
            'torque': {'bolt_1': 25.0}
        }
    ]
    
    result = await generator.generate_assembly_sop(
        assembly_name="Test Assembly",
        bill_of_materials=bom,
        assembly_sequence=sequence,
        tools_required=['wrench', 'screwdriver']
    )
    
    assert 'sop_type' in result
    assert 'assembly_instructions' in result
    print(f"Assembly SOP: {len(result['assembly_instructions'])} steps")

@pytest.mark.asyncio
async def test_complete_sop_package():
    """Test complete invention SOP generation"""
    from sop_generator_enhanced import EnhancedSOPGenerator
    
    generator = EnhancedSOPGenerator()
    
    invention_spec = {
        'name': 'Test Invention',
        'manufacturing': {'process_type': 'assembly', 'cycle_time': 30},
        'assembly': {
            'bom': [{'part': 'A'}, {'part': 'B'}],
            'sequence': [{'step': 1, 'description': 'Assemble A and B'}],
            'tools': ['wrench']
        },
        'testing': {
            'type': 'Functional',
            'parameters': {'param1': {'value': 10}},
            'acceptance': 'Pass',
            'equipment': ['tester']
        },
        'equipment': [
            {'id': 'EQ001', 'name': 'Machine 1', 'maintenance_type': 'Preventive'}
        ],
        'hazards': [
            {'type': 'mechanical', 'description': 'Pinch point', 'risk': 'Medium'}
        ]
    }
    
    result = await generator.generate_complete_invention_sop(invention_spec)
    
    assert 'document_title' in result
    assert 'sections' in result
    print(f"Complete SOP: {result['document_title']}")
    print(f"Sections: {list(result['sections'].keys())}")


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING ENHANCED E2E INVENTION PLANNER TESTS")
    print("=" * 80)
    
    # Run physics tests
    print("\n--- Physics Validation Tests ---")
    test_physics_import()
    print("[PASS] Import successful")
    
    test_fea_stress_analysis()
    print("[PASS] FEA stress analysis passed")
    
    test_fea_modal_analysis()
    print("[PASS] FEA modal analysis passed")
    
    test_cfd_flow_simulation()
    print("[PASS] CFD flow simulation passed")
    
    test_thermal_steady_state()
    print("[PASS] Thermal analysis passed")
    
    test_enhanced_physics_validator()
    print("[PASS] Enhanced physics validator passed")
    
    # Run uncertainty tests
    print("\n--- Uncertainty Propagation Tests ---")
    test_uncertainty_import()
    print("[PASS] Import successful")
    
    test_uncertainty_source_sampling()
    print("[PASS] Uncertainty sampling passed")
    
    test_monte_carlo_propagation()
    print("[PASS] Monte Carlo propagation passed")
    
    test_sobol_sensitivity()
    print("[PASS] Sobol sensitivity passed")
    
    test_error_budget()
    print("[PASS] Error budget passed")
    
    # Run SOP tests
    print("\n--- SOP Generation Tests ---")
    asyncio.run(test_sop_import())
    print("[PASS] Import successful")
    
    asyncio.run(test_manufacturing_sop())
    print("[PASS] Manufacturing SOP passed")
    
    asyncio.run(test_assembly_sop())
    print("[PASS] Assembly SOP passed")
    
    asyncio.run(test_complete_sop_package())
    print("[PASS] Complete SOP package passed")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
