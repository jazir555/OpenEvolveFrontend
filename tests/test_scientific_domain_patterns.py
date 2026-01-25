"""
Test Suite for Scientific Domain Patterns

Comprehensive tests for B.3 Scientific Domain Patterns functionality.
Tests all domain-specific capabilities: equation templates, parameter conventions,
solution methods, boundary conditions, and verification patterns.

Author: OpenEvolve
Created: 2026-01-09
Phase: 2 - LeanAide Enhancement (Task B.3)
"""

import pytest
from scientific_domain_patterns import (
    ScientificDomainPatterns,
    EquationTemplate,
    ParameterConvention,
    DomainKnowledge,
    get_domain_patterns,
    get_equation_template,
)
from continuous_math_detector import ScientificDomain, MathType, ProblemType


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def patterns():
    """Create a ScientificDomainPatterns instance for testing"""
    return ScientificDomainPatterns()


# ============================================================================
# B.3.1: Domain Knowledge Tests
# ============================================================================

class TestDomainKnowledge:
    """Test suite for domain knowledge base"""

    def test_physics_domain_knowledge_exists(self, patterns):
        """Test that physics domain knowledge is available"""
        knowledge = patterns.get_domain_knowledge(ScientificDomain.PHYSICS)

        assert knowledge is not None
        assert knowledge.domain == ScientificDomain.PHYSICS
        assert len(knowledge.equation_templates) > 0

    def test_chemistry_domain_knowledge_exists(self, patterns):
        """Test that chemistry domain knowledge is available"""
        knowledge = patterns.get_domain_knowledge(ScientificDomain.CHEMISTRY)

        assert knowledge is not None
        assert knowledge.domain == ScientificDomain.CHEMISTRY
        assert len(knowledge.equation_templates) > 0

    def test_biology_domain_knowledge_exists(self, patterns):
        """Test that biology domain knowledge is available"""
        knowledge = patterns.get_domain_knowledge(ScientificDomain.BIOLOGY)

        assert knowledge is not None
        assert knowledge.domain == ScientificDomain.BIOLOGY
        assert len(knowledge.equation_templates) > 0

    def test_engineering_domain_knowledge_exists(self, patterns):
        """Test that engineering domain knowledge is available"""
        knowledge = patterns.get_domain_knowledge(ScientificDomain.ENGINEERING)

        assert knowledge is not None
        assert knowledge.domain == ScientificDomain.ENGINEERING
        assert len(knowledge.equation_templates) > 0

    def test_economics_domain_knowledge_exists(self, patterns):
        """Test that economics domain knowledge is available"""
        knowledge = patterns.get_domain_knowledge(ScientificDomain.ECONOMICS)

        assert knowledge is not None
        assert knowledge.domain == ScientificDomain.ECONOMICS
        assert len(knowledge.equation_templates) > 0

    def test_all_domains_initialized(self, patterns):
        """Test that all 5 scientific domains are initialized"""
        domains = [
            ScientificDomain.PHYSICS,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.BIOLOGY,
            ScientificDomain.ENGINEERING,
            ScientificDomain.ECONOMICS
        ]

        for domain in domains:
            knowledge = patterns.get_domain_knowledge(domain)
            assert knowledge is not None, f"{domain.value} domain not initialized"


# ============================================================================
# B.3.2: Equation Templates Tests
# ============================================================================

class TestEquationTemplates:
    """Test suite for equation templates"""

    def test_physics_has_newtons_second_law(self, patterns):
        """Test physics domain has Newton's Second Law template"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        assert any(t.name == "Newton's Second Law" for t in templates)

    def test_physics_has_heat_equation(self, patterns):
        """Test physics domain has Heat Equation template"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        assert any(t.name == "Heat Equation" for t in templates)

    def test_physics_has_wave_equation(self, patterns):
        """Test physics domain has Wave Equation template"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        assert any(t.name == "Wave Equation" for t in templates)

    def test_physics_has_schrodinger_equation(self, patterns):
        """Test physics domain has Schrödinger Equation template"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        assert any("Schrödinger" in t.name for t in templates)

    def test_chemistry_has_rate_equation(self, patterns):
        """Test chemistry domain has Rate Equation template"""
        templates = patterns.get_equation_templates(ScientificDomain.CHEMISTRY)

        assert any("Rate" in t.name and "Equation" in t.name for t in templates)

    def test_chemistry_has_michaelis_menten(self, patterns):
        """Test chemistry domain has Michaelis-Menten template"""
        templates = patterns.get_equation_templates(ScientificDomain.CHEMISTRY)

        assert any("Michaelis-Menten" in t.name for t in templates)

    def test_biology_has_lotka_volterra(self, patterns):
        """Test biology domain has Lotka-Volterra template"""
        templates = patterns.get_equation_templates(ScientificDomain.BIOLOGY)

        assert any("Lotka-Volterra" in t.name for t in templates)

    def test_biology_has_sir_model(self, patterns):
        """Test biology domain has SIR Model template"""
        templates = patterns.get_equation_templates(ScientificDomain.BIOLOGY)

        assert any("SIR" in t.name for t in templates)

    def test_economics_has_black_scholes(self, patterns):
        """Test economics domain has Black-Scholes template"""
        templates = patterns.get_equation_templates(ScientificDomain.ECONOMICS)

        assert any("Black-Scholes" in t.name for t in templates)

    def test_equation_template_structure(self, patterns):
        """Test that equation templates have correct structure"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        assert len(templates) > 0

        for template in templates:
            assert template.name is not None
            assert template.domain == ScientificDomain.PHYSICS
            assert template.category is not None
            assert template.equation_pattern is not None
            assert template.description is not None
            assert isinstance(template.parameters, dict)
            assert isinstance(template.typical_conditions, list)
            assert template.solution_method is not None

    def test_template_has_lean4_code(self, patterns):
        """Test that templates include Lean 4 code"""
        templates = patterns.get_equation_templates(ScientificDomain.PHYSICS)

        # At least some templates should have Lean 4 templates
        has_lean4 = any(t.lean4_template is not None for t in templates)
        assert has_lean4

    def test_filter_by_category(self, patterns):
        """Test filtering equation templates by category"""
        # Get only thermodynamics templates
        templates = patterns.get_equation_templates(
            ScientificDomain.PHYSICS,
            category="thermodynamics"
        )

        assert all(t.category == "thermodynamics" for t in templates)
        assert len(templates) > 0


# ============================================================================
# B.3.3: Parameter Conventions Tests
# ============================================================================

class TestParameterConventions:
    """Test suite for parameter conventions"""

    def test_physics_has_planck_constant(self, patterns):
        """Test physics domain has Planck constant convention"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.PHYSICS)

        assert any("Planck" in c.parameter or "hbar" in c.symbol for c in conventions)

    def test_physics_has_speed_of_light(self, patterns):
        """Test physics domain has speed of light convention"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.PHYSICS)

        assert any("speed of light" in c.parameter.lower() or c.symbol == "c" for c in conventions)

    def test_chemistry_has_avogadros_number(self, patterns):
        """Test chemistry domain has Avogadro's number convention"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.CHEMISTRY)

        assert any("Avogadro" in c.parameter or c.symbol == "N_A" for c in conventions)

    def test_chemistry_has_gas_constant(self, patterns):
        """Test chemistry domain has gas constant convention"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.CHEMISTRY)

        assert any("Gas" in c.parameter or c.symbol == "R" for c in conventions)

    def test_parameter_convention_structure(self, patterns):
        """Test that parameter conventions have correct structure"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.PHYSICS)

        assert len(conventions) > 0

        for convention in conventions:
            assert convention.domain == ScientificDomain.PHYSICS
            assert convention.parameter is not None
            assert convention.symbol is not None
            assert convention.description is not None
            assert isinstance(convention.typical_values, list)

    def test_parameter_has_units(self, patterns):
        """Test that parameter conventions include units"""
        conventions = patterns.get_parameter_conventions(ScientificDomain.PHYSICS)

        # At least some should have units
        has_units = any(c.units is not None for c in conventions)
        assert has_units


# ============================================================================
# B.3.4: Solution Methods Tests
# ============================================================================

class TestSolutionMethods:
    """Test suite for solution methods"""

    def test_physics_has_separation_of_variables(self, patterns):
        """Test physics domain has separation of variables method"""
        methods = patterns.get_solution_methods(ScientificDomain.PHYSICS)

        assert any("separation" in m.lower() for m in methods)

    def test_physics_has_fourier_series(self, patterns):
        """Test physics domain has Fourier series method"""
        methods = patterns.get_solution_methods(ScientificDomain.PHYSICS)

        assert any("fourier" in m.lower() for m in methods)

    def test_chemistry_has_steady_state_approximation(self, patterns):
        """Test chemistry domain has steady-state approximation"""
        methods = patterns.get_solution_methods(ScientificDomain.CHEMISTRY)

        assert any("steady" in m.lower() or "equilibrium" in m.lower() for m in methods)

    def test_biology_has_phase_plane_analysis(self, patterns):
        """Test biology domain has phase plane analysis"""
        methods = patterns.get_solution_methods(ScientificDomain.BIOLOGY)

        assert any("phase" in m.lower() for m in methods)

    def test_engineering_has_laplace_transform(self, patterns):
        """Test engineering domain has Laplace transform"""
        methods = patterns.get_solution_methods(ScientificDomain.ENGINEERING)

        assert any("laplace" in m.lower() for m in methods)

    def test_economics_has_ito_calculus(self, patterns):
        """Test economics domain has Itô calculus"""
        methods = patterns.get_solution_methods(ScientificDomain.ECONOMICS)

        assert any("ito" in m.lower() for m in methods)

    def test_solution_methods_not_empty(self, patterns):
        """Test that all domains have solution methods"""
        domains = [
            ScientificDomain.PHYSICS,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.BIOLOGY,
            ScientificDomain.ENGINEERING,
            ScientificDomain.ECONOMICS
        ]

        for domain in domains:
            methods = patterns.get_solution_methods(domain)
            assert len(methods) > 0, f"{domain.value} has no solution methods"


# ============================================================================
# B.3.5: Boundary Conditions Tests
# ============================================================================

class TestBoundaryConditions:
    """Test suite for boundary conditions"""

    def test_physics_has_dirichlet_condition(self, patterns):
        """Test physics domain has Dirichlet boundary condition"""
        conditions = patterns.get_boundary_conditions(ScientificDomain.PHYSICS)

        assert any("Dirichlet" in c for c in conditions)

    def test_physics_has_neumann_condition(self, patterns):
        """Test physics domain has Neumann boundary condition"""
        conditions = patterns.get_boundary_conditions(ScientificDomain.PHYSICS)

        assert any("Neumann" in c for c in conditions)

    def test_chemistry_has_no_flux_condition(self, patterns):
        """Test chemistry domain has no-flux condition"""
        conditions = patterns.get_boundary_conditions(ScientificDomain.CHEMISTRY)

        assert any("flux" in c.lower() for c in conditions)

    def test_biology_has_nonnegativity_condition(self, patterns):
        """Test biology domain has non-negativity condition"""
        conditions = patterns.get_boundary_conditions(ScientificDomain.BIOLOGY)

        assert any("negative" in c.lower() for c in conditions)

    def test_engineering_has_stability_condition(self, patterns):
        """Test engineering domain has stability condition"""
        conditions = patterns.get_boundary_conditions(ScientificDomain.ENGINEERING)

        assert any("stability" in c.lower() or "stable" in c.lower() for c in conditions)


# ============================================================================
# B.3.6: Verification Patterns Tests
# ============================================================================

class TestVerificationPatterns:
    """Test suite for verification patterns"""

    def test_physics_has_energy_conservation(self, patterns):
        """Test physics domain has energy conservation pattern"""
        patterns_list = patterns.get_verification_patterns(ScientificDomain.PHYSICS)

        assert any("energy" in p.lower() and "conservation" in p.lower() for p in patterns_list)

    def test_chemistry_has_mass_conservation(self, patterns):
        """Test chemistry domain has mass conservation pattern"""
        patterns_list = patterns.get_verification_patterns(ScientificDomain.CHEMISTRY)

        assert any("mass" in p.lower() and "conservation" in p.lower() for p in patterns_list)

    def test_biology_has_stability_pattern(self, patterns):
        """Test biology domain has stability verification pattern"""
        patterns_list = patterns.get_verification_patterns(ScientificDomain.BIOLOGY)

        assert any("stability" in p.lower() for p in patterns_list)

    def test_engineering_has_bibostability_pattern(self, patterns):
        """Test engineering domain has BIBO stability pattern"""
        patterns_list = patterns.get_verification_patterns(ScientificDomain.ENGINEERING)

        assert any("bibo" in p.lower() or "stability" in p.lower() for p in patterns_list)

    def test_economics_has_no_arbitrage_pattern(self, patterns):
        """Test economics domain has no-arbitrage pattern"""
        patterns_list = patterns.get_verification_patterns(ScientificDomain.ECONOMICS)

        assert any("arbitrage" in p.lower() for p in patterns_list)


# ============================================================================
# B.3.7: Named Problems Tests
# ============================================================================

class TestNamedProblems:
    """Test suite for named problems"""

    def test_physics_has_harmonic_oscillator(self, patterns):
        """Test physics domain has harmonic oscillator problem"""
        description = patterns.find_named_problem(ScientificDomain.PHYSICS, "Harmonic Oscillator")

        assert description is not None
        assert len(description) > 0

    def test_physics_has_hydrogen_atom(self, patterns):
        """Test physics domain has hydrogen atom problem"""
        description = patterns.find_named_problem(ScientificDomain.PHYSICS, "Hydrogen Atom")

        assert description is not None

    def test_biology_has_predator_prey(self, patterns):
        """Test biology domain has predator-prey problem"""
        description = patterns.find_named_problem(ScientificDomain.BIOLOGY, "Competitive exclusion")

        assert description is not None

    def test_economics_has_option_pricing(self, patterns):
        """Test economics domain has option pricing problem"""
        description = patterns.find_named_problem(ScientificDomain.ECONOMICS, "Option pricing")

        assert description is not None

    def test_nonexistent_problem_returns_none(self, patterns):
        """Test that non-existent problem returns None"""
        description = patterns.find_named_problem(ScientificDomain.PHYSICS, "Nonexistent Problem")

        assert description is None


# ============================================================================
# B.3.8: Template Matching Tests
# ============================================================================

class TestTemplateMatching:
    """Test suite for equation template matching"""

    def test_match_heat_equation(self, patterns):
        """Test matching heat equation"""
        equation = "∂u/∂t = α ∂²u/∂x²"
        template = patterns.match_equation_to_template(equation, ScientificDomain.PHYSICS)

        assert template is not None
        assert "Heat" in template.name or "heat" in template.name

    def test_match_wave_equation(self, patterns):
        """Test matching wave equation"""
        equation = "∂²u/∂t² = c² ∂²u/∂x²"
        template = patterns.match_equation_to_template(equation, ScientificDomain.PHYSICS)

        assert template is not None
        assert "Wave" in template.name or "wave" in template.name

    def test_match_lotka_volterra(self, patterns):
        """Test matching Lotka-Volterra equations"""
        equation = "dx/dt = αx - βxy, dy/dt = δxy - γy"
        template = patterns.match_equation_to_template(equation, ScientificDomain.BIOLOGY)

        assert template is not None
        assert "Lotka-Volterra" in template.name

    def test_match_black_scholes(self, patterns):
        """Test matching Black-Scholes equation"""
        equation = "∂V/∂t + ½σ²S²∂²V/∂S² + rS∂V/∂S - rV = 0"
        template = patterns.match_equation_to_template(equation, ScientificDomain.ECONOMICS)

        assert template is not None
        assert "Black-Scholes" in template.name


# ============================================================================
# B.3.9: Solution Method Recommendation Tests
# ============================================================================

class TestSolutionMethodRecommendation:
    """Test suite for solution method recommendations"""

    def test_recommend_for_ivp(self, patterns):
        """Test solution method recommendation for IVP"""
        methods = patterns.recommend_solution_method(
            ScientificDomain.PHYSICS,
            MathType.ODE,
            ProblemType.INITIAL_VALUE
        )

        assert len(methods) > 0
        assert isinstance(methods, list)

    def test_recommend_for_bvp(self, patterns):
        """Test solution method recommendation for BVP"""
        methods = patterns.recommend_solution_method(
            ScientificDomain.PHYSICS,
            MathType.PDE,
            ProblemType.BOUNDARY_VALUE
        )

        assert len(methods) > 0

    def test_recommend_for_biology(self, patterns):
        """Test solution method recommendation for biology domain"""
        methods = patterns.recommend_solution_method(
            ScientificDomain.BIOLOGY,
            MathType.ODE,
            ProblemType.INITIAL_VALUE
        )

        assert len(methods) > 0

    def test_recommend_fallback_to_numerical(self, patterns):
        """Test fallback to numerical methods when no specific method"""
        # Use a combination that might not have specific methods
        methods = patterns.recommend_solution_method(
            ScientificDomain.ENGINEERING,
            MathType.SDE,
            ProblemType.INITIAL_VALUE
        )

        assert len(methods) > 0
        # Should fallback to numerical methods
        assert any("numerical" in m.lower() for m in methods)


# ============================================================================
# B.3.10: Domain Summary Tests
# ============================================================================

class TestDomainSummary:
    """Test suite for domain summaries"""

    def test_physics_domain_summary(self, patterns):
        """Test physics domain summary"""
        summary = patterns.get_domain_summary(ScientificDomain.PHYSICS)

        assert summary["domain"] == "physics"
        assert "num_equation_templates" in summary
        assert summary["num_equation_templates"] > 0
        assert "categories" in summary
        assert len(summary["categories"]) > 0

    def test_all_domains_have_summaries(self, patterns):
        """Test that all domains have summaries"""
        domains = [
            ScientificDomain.PHYSICS,
            ScientificDomain.CHEMISTRY,
            ScientificDomain.BIOLOGY,
            ScientificDomain.ENGINEERING,
            ScientificDomain.ECONOMICS
        ]

        for domain in domains:
            summary = patterns.get_domain_summary(domain)
            assert summary["domain"] == domain.value
            assert "status" not in summary or summary["status"] != "not_available"


# ============================================================================
# B.3.11: Convenience Functions Tests
# ============================================================================

class TestConvenienceFunctions:
    """Test suite for convenience functions"""

    def test_get_domain_patterns_singleton(self):
        """Test get_domain_patterns returns instance"""
        patterns1 = get_domain_patterns()
        patterns2 = get_domain_patterns()

        assert isinstance(patterns1, ScientificDomainPatterns)
        assert isinstance(patterns2, ScientificDomainPatterns)

    def test_get_equation_template_by_name(self):
        """Test getting equation template by name"""
        template = get_equation_template(ScientificDomain.PHYSICS, "Heat Equation")

        assert template is not None
        assert template.name == "Heat Equation"
        assert template.domain == ScientificDomain.PHYSICS

    def test_get_nonexistent_template_returns_none(self):
        """Test that non-existent template returns None"""
        template = get_equation_template(ScientificDomain.PHYSICS, "Nonexistent Equation")

        assert template is None


# ============================================================================
# B.3.12: Integration Tests
# ============================================================================

class TestDomainPatternsIntegration:
    """Integration tests for domain patterns"""

    def test_complete_workflow_physics(self, patterns):
        """Test complete workflow for physics problem"""
        # Step 1: Get template
        template = get_equation_template(ScientificDomain.PHYSICS, "Heat Equation")
        assert template is not None

        # Step 2: Get parameters
        conventions = patterns.get_parameter_conventions(ScientificDomain.PHYSICS)
        assert len(conventions) > 0

        # Step 3: Get solution methods
        methods = patterns.get_solution_methods(ScientificDomain.PHYSICS)
        assert len(methods) > 0

        # Step 4: Get verification patterns
        verifications = patterns.get_verification_patterns(ScientificDomain.PHYSICS)
        assert len(verifications) > 0

    def test_complete_workflow_biology(self, patterns):
        """Test complete workflow for biology problem"""
        # Match equation
        equation = "dx/dt = αx - βxy"
        template = patterns.match_equation_to_template(equation, ScientificDomain.BIOLOGY)
        assert template is not None

        # Get solution methods
        methods = patterns.recommend_solution_method(
            ScientificDomain.BIOLOGY,
            MathType.ODE,
            ProblemType.INITIAL_VALUE
        )
        assert len(methods) > 0

    def test_cross_domain_comparison(self, patterns):
        """Test comparing patterns across domains"""
        physics_summary = patterns.get_domain_summary(ScientificDomain.PHYSICS)
        chemistry_summary = patterns.get_domain_summary(ScientificDomain.CHEMISTRY)

        # Both should have templates
        assert physics_summary["num_equation_templates"] > 0
        assert chemistry_summary["num_equation_templates"] > 0

        # Categories should differ
        physics_categories = set(physics_summary["categories"])
        chemistry_categories = set(chemistry_summary["categories"])

        # At least some categories should be different
        assert len(physics_categories.symmetric_difference(chemistry_categories)) > 0


# ============================================================================
# Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
