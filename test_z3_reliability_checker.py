"""
Unit tests for Z3 Reliability Checker.
"""

import unittest
from unittest.mock import Mock, patch

try:
    from z3_reliability_checker import (
        Z3ReliabilityChecker,
        ComponentReliabilityModel,
        ReliabilityConstraint,
        ReliabilityProperty,
        ContractSpecification,
        get_z3_reliability_checker,
        get_roma_z3_integration
    )
    Z3_MODULE_AVAILABLE = True
except ImportError:
    Z3_MODULE_AVAILABLE = False


@unittest.skipUnless(Z3_MODULE_AVAILABLE, "Z3 reliability checker not available")
class TestZ3ReliabilityChecker(unittest.TestCase):
    """Test cases for Z3ReliabilityChecker."""
    
    def setUp(self):
        self.checker = Z3ReliabilityChecker()
    
    def test_component_model_creation(self):
        """Test ComponentReliabilityModel creation."""
        model = ComponentReliabilityModel(
            component_id="test_component",
            availability=0.99,
            mtbf_hours=8760.0,
            mttr_hours=1.0
        )
        
        self.assertEqual(model.component_id, "test_component")
        self.assertEqual(model.availability, 0.99)
        self.assertEqual(model.calculate_availability(), 8760.0 / 8761.0)
    
    def test_component_to_z3_variables(self):
        """Test conversion to Z3 variables."""
        model = ComponentReliabilityModel(component_id="test")
        variables = model.to_z3_variables()
        
        self.assertEqual(len(variables), 3)
        self.assertTrue(any("availability" in v.name for v in variables))
        self.assertTrue(any("mtbf" in v.name for v in variables))
    
    def test_reliability_constraint_to_smtlib(self):
        """Test ReliabilityConstraint to SMT-LIB conversion."""
        constraint = ReliabilityConstraint(
            property_type=ReliabilityProperty.AVAILABILITY,
            threshold=0.99,
            target_component="test"
        )
        
        smtlib = constraint.to_smtlib()
        self.assertIn("availability_availability_test", smtlib)
        self.assertIn("0.99", smtlib)
    
    def test_contract_specification(self):
        """Test ContractSpecification creation."""
        contract = ContractSpecification(
            contract_id="test_contract",
            provider="service_a",
            consumer="service_b",
            preconditions=["auth_valid"],
            postconditions=["data_returned"],
            reliability_slo=0.995
        )
        
        self.assertEqual(contract.contract_id, "test_contract")
        self.assertEqual(contract.provider, "service_a")
        self.assertEqual(contract.reliability_slo, 0.995)
        
        # Test constraint generation
        constraints = contract.to_z3_constraints()
        self.assertEqual(len(constraints), 3)  # 1 precondition + 1 postcondition + 1 SLO
    
    def test_get_status(self):
        """Test get_status method."""
        status = self.checker.get_status()
        
        self.assertIn("z3_available", status)
        self.assertIn("statistics", status)
        self.assertIn("cache_size", status)
    
    @patch('z3_reliability_checker.Z3_AVAILABLE', False)
    def test_verify_component_reliability_no_z3(self):
        """Test component verification when Z3 not available."""
        model = ComponentReliabilityModel(component_id="test")
        requirements = [ReliabilityConstraint(ReliabilityProperty.AVAILABILITY, 0.99)]
        
        result = self.checker.verify_component_reliability(model, requirements)
        
        self.assertFalse(result.success)
        self.assertFalse(result.verified)
        self.assertTrue(len(result.violations) > 0)


class TestROMAZ3Integration(unittest.TestCase):
    """Test cases for ROMA-Z3 integration."""
    
    @unittest.skipUnless(Z3_MODULE_AVAILABLE, "Z3 reliability checker not available")
    def test_get_roma_z3_integration(self):
        """Test getting ROMA-Z3 integration."""
        integration = get_roma_z3_integration()
        self.assertIsNotNone(integration)
        self.assertIsNotNone(integration.checker)


if __name__ == "__main__":
    unittest.main()
