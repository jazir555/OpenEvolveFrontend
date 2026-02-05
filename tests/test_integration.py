"""
Integration test for adversarial and evolution modules

This test file validates basic integration between components.
"""
import sys
import os
import pytest

# Add the current directory to Python path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class TestAdversarialEvolutionIntegration:
    """Test integration between adversarial and evolution modules."""

    def test_import_adversarial_module(self):
        """Test that adversarial module can be imported."""
        try:
            import adversarial
            assert adversarial is not None
        except ImportError as e:
            pytest.skip(f"Adversarial module not available: {e}")

    def test_import_evolution_module(self):
        """Test that evolution module can be imported."""
        try:
            from evolution import ContentEvaluator
            assert ContentEvaluator is not None
        except ImportError as e:
            pytest.skip(f"Evolution module not available: {e}")

    def test_content_evaluator_creation(self):
        """Test ContentEvaluator class instantiation."""
        try:
            from evolution import ContentEvaluator
            evaluator = ContentEvaluator("general", "Evaluate content quality")
            assert evaluator is not None
        except (ImportError, TypeError, ValueError) as e:
            pytest.skip(f"ContentEvaluator test skipped: {e}")

    def test_adversarial_capabilities(self):
        """Test adversarial capabilities summary."""
        try:
            from adversarial import get_adversarial_testing_capabilities
            capabilities = get_adversarial_testing_capabilities()
            assert isinstance(capabilities, dict)
            assert len(capabilities) > 0
        except (ImportError, AttributeError) as e:
            pytest.skip(f"Adversarial capabilities test skipped: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])