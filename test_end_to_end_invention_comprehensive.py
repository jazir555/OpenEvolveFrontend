"""
Comprehensive Tests for Enhanced End-to-End Invention Planner

Tests all enhanced components:
1. Physics validation with FEA, CFD, thermal analysis
2. Error analysis with Monte Carlo, Sobol, PCE
3. SOP generation with industrial automation
4. Complete pipeline integration

Author: OpenEvolve
Version: 2.0.0
"""

import pytest
import asyncio
import numpy as np
from typing import Dict, List, Any
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import enhanced components
try:
    from physics_validator_enhanced import (
        EnhancedPhysicsValidator,
        PhysicsDomain,
        validate_physics_with_simulation,
        FEASimulator,
        CFDSimulator,
        ThermalAnalyzer,
        PDESolver
    )
    PHYSICS_AVAILABLE = True
except ImportError:
    PHYSICS_AVAILABLE = False
    print("Warning: Enhanced physics validator not available")

try:
    from uncertainty_propagation_enhanced import (
        EnhancedUncertaintyPropagator,
        UncertaintySource,
        PolynomialChaosExpansion,
        SobolSensitivityAnalyzer,
        comprehensive_error_analysis
    )
    UNCERTAINTY_AVAILABLE = True
except ImportError:
    UNCERTAINTY_AVAILABLE = False
    print("Warning: Enhanced uncertainty propagation not available")

try:
    from sop_generator_enhanced import (
        EnhancedSOPGenerator,
        LLM4IASIntegration,
        generate_industrial_sop,
        SOPType,
        IndustryStandard
    )
    SOP_AVAILABLE = True
except ImportError:
    SOP_AVAILABLE = False
    print("Warning: Enhanced SOP generator not available")

try:
    from e2e_invention_planner_enhanced import (
        EnhancedEndToEndPlanner,
        run_enhanced_invention_planning,
        get_enhanced_planner_status
    )
    E2E_AVAILABLE = True
except ImportError:
    E2E_AVAILABLE = False
    print("Warning: Enhanced E2E planner not available")


# ============================================================================
# Physics Validation Tests
# ============================================================================

@pytest.mark.skipif(not PHYSICS_AVAILABLE, reason="Enhanced physics not available")
class TestPhysicsValidation:
    """Test enhanced physics validation"""
    
    def test_fea_simulator_initialization(self):
        """Test FEA simulator initialization"""
        fea = FEASimulator()
        assert fea is not None
    
    def test_fea_stress_analysis(self):
        """Test FEA stress analysis"""
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
    
    def test_fea_modal_analysis(self):
        """Test FEA modal analysis"""
        fea = FEASimulator()
        
        geometry = {'mass': 10.0, 'length': 1.0, 'cross_sectional_area': 0.01}
        material = {'youngs_modulus': 200e9}
        
        result = fea.modal_analysis(geometry, material, num_modes=5)
        
        assert 'natural_frequencies' in result
        assert len(result['natural_frequencies']) == 5
        assert all(f > 0 for f in result['natural_frequencies'])
    
    def test_cfd_simulator_initialization(self):
        """Test CFD simulator initialization"""
        cfd = CFDSimulator()
        assert cfd is not None
    
    def test_cfd_flow_simulation(self):
        """Test CFD flow simulation"""
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
    
    def test_thermal_analyzer_initialization(self):
        """Test thermal analyzer initialization"""
        thermal = ThermalAnalyzer()
        assert thermal is not None
    
    def test_thermal_steady_state(self):
        """Test steady-state thermal analysis"""
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
    
    def test_pde_solver_initialization(self):
        """Test PDE solver initialization"""
        solver = PDESolver()
        assert solver is not None
    
    def test_enhanced_physics_validator_initialization(self):
        """Test enhanced physics validator initialization"""
        validator = EnhancedPhysicsValidator()
        assert validator is not None
        assert validator.physicsnemo is not None
        assert validator.pde_solver is not None
        assert validator.fea is not None
        assert validator.cfd is not None
        assert validator.thermal is not None
    
    def test_physics_validation_structural(self):
        """Test physics validation for structural domain"""
        validator = EnhancedPhysicsValidator()
        
        invention_spec = {
            'geometry': {
                'length': 2.0,
                'cross_sectional_area': 0.01,
                'surface_area': 0.5
            },
            'material_properties': {
                'youngs_modulus': 200e9,
                'yield_stress': 250e6,
                'poisson_ratio': 0.3
            },
            'loads': [{'magnitude': 50000, 'direction': 'axial'}]
        }
        
        result = validator._validate_structural(invention_spec)
        
        assert result.domain == PhysicsDomain.STRUCTURAL
        assert result.simulation_type == "fea_stress_analysis"
        assert 'max_stress' in result.metrics
    
    def test_physics_validation_thermal(self):
        """Test physics validation for thermal domain"""
        validator = EnhancedPhysicsValidator()
        
        invention_spec = {
            'geometry': {'surface_area': 1.0, 'volume': 0.1},
            'thermal_properties': {
                'thermal_conductivity': 50,
                'density': 7850,
                'specific_heat': 420
            },
            'heat_sources': [{'power': 1000, 'volume': 0.01}],
            'boundary_temperatures': {'ambient': 300},
            'max_operating_temperature': 500
        }
        
        result = validator._validate_thermal(invention_spec)
        
        assert result.domain == PhysicsDomain.THERMAL
        assert 'max_temperature' in result.metrics
    
    def test_physics_validation_comprehensive(self):
        """Test comprehensive physics validation"""
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


# ============================================================================
# Uncertainty Propagation Tests
# ============================================================================

@pytest.mark.skipif(not UNCERTAINTY_AVAILABLE, reason="Enhanced uncertainty not available")
class TestUncertaintyPropagation:
    """Test enhanced uncertainty propagation"""
    
    def test_uncertainty_source_sampling(self):
        """Test uncertainty source sampling"""
        source = UncertaintySource(
            name="test_parameter",
            distribution="normal",
            parameters={'mean': 10.0, 'std': 1.0}
        )
        
        samples = source.sample(1000)
        
        assert len(samples) == 1000
        assert abs(np.mean(samples) - 10.0) < 0.1
        assert abs(np.std(samples) - 1.0) < 0.1
    
    def test_monte_carlo_propagation(self):
        """Test Monte Carlo uncertainty propagation"""
        propagator = EnhancedUncertaintyPropagator(random_seed=42)
        
        # Simple model: y = x1 + x2
        def model(params):
            return params[0] + params[1]
        
        uncertainty_sources = [
            UncertaintySource(
                name="x1",
                distribution="normal",
                parameters={'mean': 10.0, 'std': 1.0}
            ),
            UncertaintySource(
                name="x2",
                distribution="normal",
                parameters={'mean': 5.0, 'std': 0.5}
            )
        ]
        
        result = propagator.propagate_monte_carlo(
            model, uncertainty_sources, n_samples=10000
        )
        
        assert abs(result.mean - 15.0) < 0.1
        assert result.standard_deviation > 0
        assert result.confidence_interval_95[0] < result.mean
        assert result.confidence_interval_95[1] > result.mean
    
    def test_sobol_sensitivity_analysis(self):
        """Test Sobol sensitivity analysis"""
        propagator = EnhancedUncertaintyPropagator(random_seed=42)
        
        # Ishigami function (common test function for sensitivity analysis)
        def ishigami(params):
            x1, x2, x3 = params[0], params[1], params[2]
            return np.sin(x1) + 7 * np.sin(x2)**2 + 0.1 * x3**4 * np.sin(x1)
        
        uncertainty_sources = [
            UncertaintySource("x1", "uniform", {'low': -np.pi, 'high': np.pi}),
            UncertaintySource("x2", "uniform", {'low': -np.pi, 'high': np.pi}),
            UncertaintySource("x3", "uniform", {'low': -np.pi, 'high': np.pi})
        ]
        
        sobol = propagator.compute_sobol_indices(
            ishigami, uncertainty_sources, n_samples=5000
        )
        
        assert 'x1' in sobol.first_order
        assert 'x2' in sobol.first_order
        assert 'x3' in sobol.first_order
        
        # x2 should have highest total effect
        most_important = sobol.get_most_important(1)[0][0]
        assert most_important in ['x1', 'x2', 'x3']
    
    def test_error_budget_creation(self):
        """Test error budget creation"""
        propagator = EnhancedUncertaintyPropagator()
        
        def model(params):
            return params[0] * params[1]
        
        uncertainty_sources = [
            UncertaintySource(
                name="gain",
                distribution="normal",
                parameters={'mean': 2.0, 'std': 0.1}
            ),
            UncertaintySource(
                name="offset",
                distribution="normal",
                parameters={'mean': 1.0, 'std': 0.05}
            )
        ]
        
        budget = propagator.create_error_budget(
            model, uncertainty_sources, confidence_level=0.95
        )
        
        assert budget.total_uncertainty > 0
        assert budget.coverage_factor == 2.0
        assert len(budget.source_contributions) > 0
    
    def test_comprehensive_error_analysis(self):
        """Test comprehensive error analysis function"""
        
        invention_spec = {
            'uncertainty_sources': [
                {
                    'name': 'param1',
                    'distribution': 'normal',
                    'parameters': {'mean': 10, 'std': 1},
                    'category': 'equipment'
                },
                {
                    'name': 'param2',
                    'distribution': 'normal',
                    'parameters': {'mean': 5, 'std': 0.5},
                    'category': 'material'
                }
            ]
        }
        
        def model(params):
            return params[0] + params[1]
        
        result = comprehensive_error_analysis(
            invention_spec, model, n_samples=5000,
            include_sensitivity=True, include_error_budget=True
        )
        
        assert 'propagation' in result
        assert 'sensitivity_analysis' in result
        assert 'error_budget' in result


# ============================================================================
# SOP Generation Tests
# ============================================================================

@pytest.mark.skipif(not SOP_AVAILABLE, reason="Enhanced SOP not available")
class TestSOPGeneration:
    """Test enhanced SOP generation"""
    
    @pytest.mark.asyncio
    async def test_llm4ias_integration(self):
        """Test LLM4IAS integration"""
        llm4ias = LLM4IASIntegration()
        assert llm4ias is not None
    
    @pytest.mark.asyncio
    async def test_manufacturing_sop_generation(self):
        """Test manufacturing SOP generation"""
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
        
        assert result['sop_type'] == SOPType.MANUFACTURING.value
        assert 'manufacturing_process' in result
    
    @pytest.mark.asyncio
    async def test_assembly_sop_generation(self):
        """Test assembly SOP generation"""
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
        
        assert result['sop_type'] == SOPType.ASSEMBLY.value
        assert 'assembly_instructions' in result
    
    @pytest.mark.asyncio
    async def test_testing_sop_generation(self):
        """Test testing SOP generation"""
        generator = EnhancedSOPGenerator()
        
        test_params = {
            'voltage': {'value': 12.0, 'unit': 'V', 'tolerance': 0.5},
            'current': {'value': 2.0, 'unit': 'A', 'tolerance': 0.1}
        }
        
        result = await generator.generate_testing_sop(
            test_name="Electrical Test",
            test_type="Functional",
            test_parameters=test_params,
            acceptance_criteria="Voltage within ±0.5V, Current within ±0.1A",
            equipment_required=['Multimeter', 'Power supply']
        )
        
        assert result['sop_type'] == SOPType.TESTING.value
        assert 'procedure' in result
    
    @pytest.mark.asyncio
    async def test_complete_invention_sop(self):
        """Test complete invention SOP generation"""
        generator = EnhancedSOPGenerator()
        
        invention_spec = {
            'name': 'Test Invention',
            'manufacturing': {
                'process_type': 'assembly',
                'cycle_time': 30
            },
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
        assert 'manufacturing' in result['sections']
        assert 'assembly' in result['sections']
        assert 'testing' in result['sections']


# ============================================================================
# E2E Integration Tests
# ============================================================================

@pytest.mark.skipif(not E2E_AVAILABLE, reason="Enhanced E2E not available")
class TestE2EIntegration:
    """Test end-to-end integration"""
    
    def test_planner_initialization(self):
        """Test planner initialization"""
        planner = EnhancedEndToEndPlanner(use_enhanced=True)
        assert planner is not None
        assert planner.use_enhanced == True
    
    @pytest.mark.asyncio
    async def test_complete_planning_run(self):
        """Test complete planning run with all enhancements"""
        planner = EnhancedEndToEndPlanner(use_enhanced=True)
        
        invention_spec = {
            'name': 'Test Device',
            'geometry': {
                'length': 1.0,
                'cross_sectional_area': 0.01,
                'surface_area': 0.5
            },
            'material_properties': {
                'youngs_modulus': 200e9,
                'yield_stress': 250e6
            },
            'loads': [{'magnitude': 10000, 'direction': 'axial'}],
            'thermal_properties': {
                'thermal_conductivity': 50
            },
            'uncertainty_sources': [
                {
                    'name': 'load',
                    'distribution': 'normal',
                    'parameters': {'mean': 10000, 'std': 500},
                    'category': 'equipment'
                }
            ]
        }
        
        result = await planner.plan_invention_complete(
            prompt="Create a test device with structural and thermal requirements",
            domain="engineering",
            invention_spec=invention_spec,
            enable_physics_simulation=True,
            enable_uncertainty_analysis=True,
            enable_enhanced_sop=True
        )
        
        assert result['planning_complete'] == True
        assert 'enhanced_validations' in result
        assert result['enhanced_validations']['physics_validation']['enabled'] == True
        assert result['enhanced_validations']['error_analysis']['enabled'] == True
    
    @pytest.mark.asyncio
    async def test_planning_without_spec(self):
        """Test planning without detailed spec (base only)"""
        planner = EnhancedEndToEndPlanner(use_enhanced=True)
        
        result = await planner.plan_invention_complete(
            prompt="Create a room-temperature superconductor",
            domain="physics",
            enable_physics_simulation=True,
            enable_uncertainty_analysis=True,
            enable_enhanced_sop=True
        )
        
        assert result['planning_complete'] == True
        # Without spec, enhanced features are skipped
        assert result['enhanced_validations']['physics_validation']['completed'] == False
    
    def test_planner_status(self):
        """Test planner status function"""
        status = get_enhanced_planner_status()
        
        assert 'version' in status
        assert 'components' in status
        assert status['version'] == '2.0.0'
    
    @pytest.mark.asyncio
    async def test_convenience_function(self):
        """Test convenience function"""
        invention_spec = {
            'name': 'Quick Test',
            'uncertainty_sources': [
                {
                    'name': 'param',
                    'distribution': 'normal',
                    'parameters': {'mean': 10, 'std': 1}
                }
            ]
        }
        
        result = await run_enhanced_invention_planning(
            prompt="Quick test invention",
            invention_spec=invention_spec,
            domain="test",
            enable_all_enhancements=True
        )
        
        assert result['planning_complete'] == True


# ============================================================================
# Performance and Edge Case Tests
# ============================================================================

@pytest.mark.skipif(not PHYSICS_AVAILABLE, reason="Physics not available")
class TestPhysicsEdgeCases:
    """Test physics validation edge cases"""
    
    def test_high_load_stress(self):
        """Test stress analysis with very high load"""
        fea = FEASimulator()
        
        geometry = {'length': 1.0, 'cross_sectional_area': 0.001, 'surface_area': 0.1}
        material = {'youngs_modulus': 200e9, 'yield_stress': 250e6}
        loads = [{'magnitude': 500000, 'direction': 'axial'}]  # Very high load
        
        result = fea.analyze_stress(geometry, material, loads, [])
        
        # Should fail (safety factor < 1.5)
        assert result['passed'] == False or result['safety_factor'] < 1.5
    
    def test_thermal_runaway(self):
        """Test thermal analysis with excessive heat"""
        thermal = ThermalAnalyzer()
        
        geometry = {'surface_area': 0.1, 'volume': 0.01}
        material = {'thermal_conductivity': 1, 'density': 1000, 'specific_heat': 1000}
        heat_sources = [{'power': 10000, 'volume': 0.001}]  # Very high heat
        boundary_temps = {'ambient': 300}
        
        result = thermal.steady_state_temperature(
            geometry, material, heat_sources, boundary_temps
        )
        
        # Temperature should be very high
        assert result['max_temperature'] > 1000


@pytest.mark.skipif(not UNCERTAINTY_AVAILABLE, reason="Uncertainty not available")
class TestUncertaintyEdgeCases:
    """Test uncertainty propagation edge cases"""
    
    def test_zero_variance(self):
        """Test with zero variance (deterministic)"""
        propagator = EnhancedUncertaintyPropagator()
        
        def model(params):
            return params[0] + params[1]
        
        uncertainty_sources = [
            UncertaintySource("x1", "normal", {'mean': 10, 'std': 0}),
            UncertaintySource("x2", "normal", {'mean': 5, 'std': 0})
        ]
        
        result = propagator.propagate_monte_carlo(
            model, uncertainty_sources, n_samples=100
        )
        
        assert abs(result.mean - 15) < 0.001
        assert result.standard_deviation < 0.001
    
    def test_large_variance(self):
        """Test with very large variance"""
        propagator = EnhancedUncertaintyPropagator()
        
        def model(params):
            return params[0]
        
        uncertainty_sources = [
            UncertaintySource("x", "normal", {'mean': 100, 'std': 1000})
        ]
        
        result = propagator.propagate_monte_carlo(
            model, uncertainty_sources, n_samples=10000
        )
        
        assert result.standard_deviation > 500


# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
