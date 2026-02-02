"""
Unit tests for Z3 Decomposition Validator.
"""

import unittest
from unittest.mock import Mock, patch

try:
    from decomposition_z3_validator import (
        Z3DecompositionValidator,
        DecompositionProperty,
        SubProblemModel,
        EntanglementSpecification,
        get_z3_decomposition_validator,
        get_decomposition_engine_z3_integration
    )
    Z3_MODULE_AVAILABLE = True
except ImportError:
    Z3_MODULE_AVAILABLE = False


@unittest.skipUnless(Z3_MODULE_AVAILABLE, "Z3 decomposition validator not available")
class TestZ3DecompositionValidator(unittest.TestCase):
    """Test cases for Z3DecompositionValidator."""
    
    def setUp(self):
        self.validator = Z3DecompositionValidator()
    
    def test_decomposition_property_enum(self):
        """Test DecompositionProperty enum values."""
        self.assertEqual(DecompositionProperty.COMPLETENESS.value, "completeness")
        self.assertEqual(DecompositionProperty.SOUNDNESS.value, "soundness")
        self.assertEqual(DecompositionProperty.INDEPENDENCE.value, "independence")
    
    def test_subproblem_model_creation(self):
        """Test SubProblemModel creation."""
        model = SubProblemModel(
            subproblem_id="sp1",
            complexity_score=5.0
        )
        
        self.assertEqual(model.subproblem_id, "sp1")
        self.assertEqual(model.complexity_score, 5.0)
        self.assertEqual(len(model.variables), 0)
        self.assertEqual(len(model.constraints), 0)
    
    def test_entanglement_specification(self):
        """Test EntanglementSpecification creation."""
        ent = EntanglementSpecification(
            entanglement_id="ent1",
            source_subproblem="sp1",
            target_subproblem="sp2",
            shared_variables=["x", "y"],
            strength="strong"
        )
        
        self.assertEqual(ent.entanglement_id, "ent1")
        self.assertEqual(ent.source_subproblem, "sp1")
        self.assertEqual(ent.target_subproblem, "sp2")
        self.assertEqual(len(ent.shared_variables), 2)
    
    def test_get_status(self):
        """Test get_status method."""
        status = self.validator.get_status()
        
        self.assertIn("z3_available", status)
        self.assertIn("decomposition_available", status)
        self.assertIn("statistics", status)
    
    @patch('decomposition_z3_validator.Z3_AVAILABLE', False)
    def test_validate_decomposition_no_z3(self):
        """Test validation when Z3 not available."""
        result = self.validator.validate_decomposition(
            original_problem="test",
            subproblems=[],
            entanglements=[]
        )
        
        self.assertFalse(result.success)
        self.assertFalse(result.valid)
    
    def test_analyze_decomposition_quality(self):
        """Test decomposition quality analysis."""
        subproblems = [
            SubProblemModel(subproblem_id="sp1", complexity_score=3.0),
            SubProblemModel(subproblem_id="sp2", complexity_score=4.0)
        ]
        entanglements = [
            EntanglementSpecification(
                entanglement_id="ent1",
                source_subproblem="sp1",
                target_subproblem="sp2"
            )
        ]
        
        quality = self.validator.analyze_decomposition_quality(
            original_problem="test",
            subproblems=subproblems,
            entanglements=entanglements
        )
        
        self.assertIn("subproblem_count", quality)
        self.assertIn("entanglement_count", quality)
        self.assertIn("parallelization_score", quality)
        self.assertEqual(quality["subproblem_count"], 2)


class TestDecompositionEngineIntegration(unittest.TestCase):
    """Test cases for Decomposition Engine integration."""
    
    @unittest.skipUnless(Z3_MODULE_AVAILABLE, "Z3 decomposition validator not available")
    def test_get_decomposition_engine_z3_integration(self):
        """Test getting decomposition engine integration."""
        integration = get_decomposition_engine_z3_integration()
        self.assertIsNotNone(integration)
        self.assertIsNotNone(integration.validator)


if __name__ == "__main__":
    unittest.main()
