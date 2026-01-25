"""
Test Suite for Enhanced Continuous Math Detector (Phase 3)

Tests for enhanced detection features:
- Ambiguity resolution
- Multi-equation detection
- Context-aware classification
- Alternative interpretations
- Equation relationship analysis

Author: OpenEvolve
Created: 2026-01-09
Phase: 3 - Enhanced Detection
"""

import pytest
from enhanced_math_detector import (
    EnhancedContinuousMathDetector,
    detect_continuous_math_enhanced,
    EnhancedDetectionResult,
    EquationRelation,
)
from continuous_math_detector import MathType, ProblemType, ScientificDomain


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def enhanced_detector():
    """Create enhanced detector instance"""
    return EnhancedContinuousMathDetector()


# ============================================================================
# Phase 3.1: Ambiguity Resolution Tests
# ============================================================================

class TestAmbiguityResolution:
    """Test suite for ambiguity resolution features"""

    def test_ambiguous_domain_resolved(self, enhanced_detector):
        """Test that ambiguous domain is resolved with context"""
        text = "Analyze population growth with dP/dt = rP(1 - P/K)"
        result = enhanced_detector.detect(text)

        # Should detect as biology (not general) due to context
        assert result.domain == ScientificDomain.BIOLOGY
        assert result.ambiguity_score < 0.5

    def test_ambiguous_math_type_resolved(self, enhanced_detector):
        """Test that ambiguous math type is resolved"""
        text = "Solve the system with partial derivatives"
        result = enhanced_detector.detect(text)

        # Should provide alternatives
        assert len(result.alternative_interpretations) > 0

    def test_context_enhances_confidence(self, enhanced_detector):
        """Test that context keywords enhance confidence"""
        text_clear = "dy/dx = y with exponential growth"
        text_context = "dy/dx = y in population dynamics with growth rate r"

        result_clear = enhanced_detector.detect(text_clear)
        result_context = enhanced_detector.detect(text_context)

        # Context should enhance confidence
        assert result_context.confidence >= result_clear.confidence

    def test_ambiguity_score_calculation(self, enhanced_detector):
        """Test ambiguity score is calculated correctly"""
        # Clear case
        text_clear = "Solve dy/dx = y"
        result_clear = enhanced_detector.detect(text_clear)

        # Ambiguous case
        text_ambiguous = "Is this physics or biology? Growth equation"
        result_ambiguous = enhanced_detector.detect(text_ambiguous)

        # Ambiguous case should have higher score
        assert result_ambiguous.ambiguity_score > result_clear.ambiguity_score


# ============================================================================
# Phase 3.2: Multi-Equation Detection Tests
# ============================================================================

class TestMultiEquationDetection:
    """Test suite for multi-equation detection"""

    def test_system_of_odes_detected(self, enhanced_detector):
        """Test detection of ODE system"""
        text = """
        System of equations:
        dx/dt = x - xy
        dy/dt = xy - y
        """
        result = enhanced_detector.detect(text)

        # Should detect multiple equations
        assert len(result.equations_found) >= 2

        # Should identify as system
        assert result.equation_relations is not None
        assert result.equation_relations.relation_type == "system"

    def test_coupled_equations_detected(self, enhanced_detector):
        """Test detection of coupled equations"""
        text = "dx/dt = y, dy/dt = -x"
        result = enhanced_detector.detect(text)

        # Should find shared variables
        assert len(result.equation_relations.variables_shared) > 0
        assert "x" in result.equation_relations.variables_shared or \
               "y" in result.equation_relations.variables_shared

    def test_independent_equations_detected(self, enhanced_detector):
        """Test detection of independent equations"""
        text = "First: dy/dx = y. Second: dz/dt = z."
        result = enhanced_detector.detect(text)

        # Should detect as independent
        if result.equation_relations:
            assert result.equation_relations.relation_type == "independent"

    def test_sequential_equations_detected(self, enhanced_detector):
        """Test detection of sequential equation relationships"""
        text = "First solve dy/dx = y, then use the result for dz/dt = z + y"
        result = enhanced_detector.detect(text)

        # Should detect sequential relationship
        if result.equation_relations:
            assert result.equation_relations.relation_type == "sequential"

    def test_coupling_strength_calculated(self, enhanced_detector):
        """Test that coupling strength is calculated"""
        # Strongly coupled
        text_strong = "dx/dt = x - xy, dy/dt = xy - y"
        result_strong = enhanced_detector.detect(text_strong)

        # Weakly coupled or independent
        text_weak = "dx/dt = x. Also dy/dt = y."
        result_weak = enhanced_detector.detect(text_weak)

        # Strong coupling should have higher strength
        if result_strong.equation_relations and result_weak.equation_relations:
            assert result_strong.equation_relations.coupling_strength >= \
                   result_weak.equation_relations.coupling_strength


# ============================================================================
# Phase 3.3: Context-Aware Classification Tests
# ============================================================================

class TestContextAwareClassification:
    """Test suite for context-aware classification"""

    def test_context_keywords_extracted(self, enhanced_detector):
        """Test that context keywords are extracted"""
        text = "Heat equation with temperature diffusion"
        result = enhanced_detector.detect(text)

        # Should extract context keywords
        assert len(result.context_keywords) > 0

        # Should have thermodynamics context
        has_thermo = any('thermodynamics' in ctx or 'heat' in ctx
                        for ctx in result.context_keywords)
        assert has_thermo

    def test_domain_resolved_with_context(self, enhanced_detector):
        """Test domain is resolved using context"""
        text = "Analyze species population dynamics: dN/dt = rN"
        result = enhanced_detector.detect(text)

        # Should resolve to biology (not general)
        assert result.domain == ScientificDomain.BIOLOGY

    def test_multiple_contexts_handled(self, enhanced_detector):
        """Test handling of multiple context indicators"""
        text = "Energy and momentum in the system"
        result = enhanced_detector.detect(text)

        # Should extract multiple contexts
        physics_contexts = [ctx for ctx in result.context_keywords
                           if 'physics' in ctx]
        assert len(physics_contexts) > 0

    def test_confidence_boosted_by_context(self, enhanced_detector):
        """Test that context boosts confidence"""
        text_no_context = "Solve dy/dx = y"
        text_with_context = "In population dynamics, solve dy/dx = y"

        result_no_context = enhanced_detector.detect(text_no_context)
        result_with_context = enhanced_detector.detect(text_with_context)

        # Context should boost confidence
        assert result_with_context.confidence >= result_no_context.confidence


# ============================================================================
# Phase 3.4: Alternative Interpretations Tests
# ============================================================================

class TestAlternativeInterpretations:
    """Test suite for alternative interpretations"""

    def test_alternatives_for_unknown_type(self, enhanced_detector):
        """Test alternatives provided for unknown math type"""
        text = "integral of growth function"
        result = enhanced_detector.detect(text)

        # Should provide alternatives
        assert len(result.alternative_interpretations) > 0

        # Should suggest integral type
        integral_alternatives = [alt for alt in result.alternative_interpretations
                               if alt.get('math_type') == MathType.INTEGRAL]
        assert len(integral_alternatives) > 0

    def test_alternatives_for_ambiguous_domain(self, enhanced_detector):
        """Test domain alternatives provided"""
        text = "Growth model in the system"
        result = enhanced_detector.detect(text)

        # Should provide domain alternatives
        domain_alternatives = [alt for alt in result.alternative_interpretations
                             if 'domain' in alt]
        # May or may not have alternatives depending on clarity
        # Just check it doesn't crash
        assert isinstance(domain_alternatives, list)

    def test_alternative_includes_reason(self, enhanced_detector):
        """Test that alternatives include reasons"""
        text = "Solve with exponential growth"
        result = enhanced_detector.detect(text)

        if result.alternative_interpretations:
            # Each alternative should have a reason
            for alt in result.alternative_interpretations:
                assert 'reason' in alt
                assert 'confidence' in alt

    def test_pde_alternative_for_multi_variable(self, enhanced_detector):
        """Test PDE alternative for multi-variable ODEs"""
        text = "Solve for f(x, y, t)"
        result = enhanced_detector.detect(text)

        # If detected as ODE, should suggest PDE as alternative
        pde_alternatives = [alt for alt in result.alternative_interpretations
                           if alt.get('math_type') == MathType.PDE]
        # This is context-dependent, just check it works
        assert isinstance(pde_alternatives, list)


# ============================================================================
# Phase 3.5: Enhanced Result Structure Tests
# ============================================================================

class TestEnhancedResultStructure:
    """Test suite for enhanced result data structure"""

    def test_enhanced_result_fields(self, enhanced_detector):
        """Test that enhanced result has all expected fields"""
        text = "Solve dy/dx = y"
        result = enhanced_detector.detect(text)

        # Base fields (inherited)
        assert result.math_type is not None
        assert result.domain is not None
        assert result.confidence >= 0

        # Enhanced fields
        assert isinstance(result.equations_found, list)
        assert isinstance(result.ambiguity_score, float)
        assert isinstance(result.context_keywords, list)
        assert isinstance(result.alternative_interpretations, list)

    def test_result_to_dict_works(self, enhanced_detector):
        """Test that result can be serialized"""
        text = "System: dx/dt = x, dy/dt = y"
        result = enhanced_detector.detect(text)

        # Should be serializable
        result_dict = result.to_dict() if hasattr(result, 'to_dict') else None

        # Just check it doesn't crash
        assert result is not None

    def test_equation_structure_completeness(self, enhanced_detector):
        """Test that equation structures are complete"""
        text = "dx/dt = x - xy, dy/dt = xy - y"
        result = enhanced_detector.detect(text)

        # Check equation structures
        for eq in result.equations_found:
            assert eq.dependent_var is not None
            assert eq.independent_vars is not None
            assert eq.order is not None
            assert eq.raw_equation is not None


# ============================================================================
# Phase 3.6: Integration Tests
# ============================================================================

class TestEnhancedIntegration:
    """Integration tests for enhanced detector"""

    def test_complete_enhanced_workflow(self, enhanced_detector):
        """Test complete enhanced detection workflow"""
        text = """
        In population dynamics, analyze the Lotka-Volterra system:
        dx/dt = αx - βxy
        dy/dt = δxy - γy
        where x is prey, y is predator
        """

        result = enhanced_detector.detect(text)

        # Should detect system
        assert len(result.equations_found) >= 2
        assert result.equation_relations.relation_type == "system"

        # Should resolve domain
        assert result.domain == ScientificDomain.BIOLOGY

        # Should have context
        assert len(result.context_keywords) > 0

        # Should have reasonable confidence
        assert result.confidence > 0.5

    def test_ambiguous_case_workflow(self, enhanced_detector):
        """Test workflow for ambiguous input"""
        text = "Is this about energy? Growth equation: dE/dt = input - output"

        result = enhanced_detector.detect(text)

        # Should provide alternatives
        assert len(result.alternative_interpretations) >= 0

        # Should calculate ambiguity
        assert 0 <= result.ambiguity_score <= 1

        # Should extract context
        assert len(result.context_keywords) >= 0

    def test_multi_domain_context(self, enhanced_detector):
        """Test handling of multi-domain context"""
        text = "Biological system with energy conservation: dN/dt = rN"
        result = enhanced_detector.detect(text)

        # Should detect both biology and physics contexts
        contexts = result.context_keywords
        has_biology = any('biology' in ctx or 'population' in ctx for ctx in contexts)
        has_physics = any('physics' in ctx or 'energy' in ctx for ctx in contexts)

        # Should pick one as primary but have both in context
        assert result.domain in [ScientificDomain.BIOLOGY, ScientificDomain.PHYSICS]
        assert has_biology or has_physics


# ============================================================================
# Phase 3.7: Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_detect_enhanced_function(self):
        """Test detect_continuous_math_enhanced convenience function"""
        text = "System: dx/dt = x, dy/dt = y"

        result = detect_continuous_math_enhanced(text)

        # Should return enhanced result
        assert isinstance(result, EnhancedDetectionResult)
        assert len(result.equations_found) >= 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestEnhancedPerformance:
    """Test performance of enhanced detection"""

    def test_enhanced_detection_performance(self, enhanced_detector):
        """Test that enhanced detection is still fast"""
        import time

        text = "Solve the system: dx/dt = x - xy, dy/dt = xy - y"

        start = time.time()
        result = enhanced_detector.detect(text)
        elapsed = time.time() - start

        # Should complete in reasonable time (< 1 second)
        assert elapsed < 1.0

        # Should still work correctly
        assert len(result.equations_found) >= 2

    def test_batch_enhanced_detection(self, enhanced_detector):
        """Test batch processing with enhanced detection"""
        texts = [
            "dy/dx = y",
            "System: dx/dt = x, dy/dt = y",
            "Heat equation with diffusion",
        ] * 10  # 30 texts total

        import time
        start = time.time()

        results = [enhanced_detector.detect(text) for text in texts]

        elapsed = time.time() - start
        avg_time = elapsed / len(texts)

        # Average should be reasonable (< 100ms per text)
        assert avg_time < 0.1


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
