"""
Test Suite for LeanAide-Enhanced Decomposition Engine

Comprehensive tests for mathematical problem detection, LeanAide integration,
and evolutionary proof generation configuration.
"""

import asyncio
import pytest
import logging
from typing import Dict, Any

# Import the enhanced decomposition engine
try:
    from decomposition_engine_lean_enhanced import (
        LeanMathematicalDetector,
        LeanEnhancedDecompositionEngine,
        LeanSubProblemDecomposer,
        EvolutionaryStrategySuggestor,
        MathematicalProblemMetadata,
        MathematicalProblemType,
        EvolutionaryStrategyType,
        detect_and_route_mathematical_problem,
        generate_evolutionary_config
    )
    from leanaide_decomposition_integration import (
        MathematicalDomain,
        ComponentType
    )
    from sovereign_data_models import (
        ProblemDefinition,
        DomainContext,
        ComplexityScore
    )
    LEAN_ENHANCED_ENGINE_AVAILABLE = True
except ImportError as e:
    LEAN_ENHANCED_ENGINE_AVAILABLE = False
    print(f"Warning: Lean-enhanced engine not available: {e}")

logger = logging.getLogger(__name__)


# =============================================================================
# TEST FIXTURES
# =============================================================================

@pytest.fixture
def mathematical_problems() -> Dict[str, Dict[str, str]]:
    """Sample mathematical problems for testing."""
    return {
        "infinite_primes": {
            "title": "Infinite Primes Theorem",
            "description": """
            Prove that there are infinitely many prime numbers.

            Hint: Assume there are finitely many primes p1, p2, ..., pn.
            Consider the number N = p1 * p2 * ... * pn + 1.
            Show that N must have a prime divisor not in the list.
            """,
            "expected_domain": MathematicalDomain.NUMBER_THEORY,
            "expected_difficulty": 6
        },
        "irrational_sqrt2": {
            "title": "Irrationality of sqrt(2)",
            "description": """
            Prove that sqrt(2) is irrational.

            Use proof by contradiction: assume sqrt(2) = a/b where a, b are
            integers with no common factors. Show that this leads to a
            contradiction.
            """,
            "expected_domain": MathematicalDomain.NUMBER_THEORY,
            "expected_difficulty": 5
        },
        "group_theory": {
            "title": "Lagrange's Theorem",
            "description": """
            Prove Lagrange's Theorem: For any finite group G and subgroup H,
            the order of H divides the order of G.

            This is a fundamental result in group theory with applications
            throughout algebra.
            """,
            "expected_domain": MathematicalDomain.ALGEBRA,
            "expected_difficulty": 7
        },
        "continuity": {
            "title": "Epsilon-Delta Continuity",
            "description": """
            Prove that the function f(x) = x^2 is continuous at every point
            using the epsilon-delta definition of continuity.

            For any c in R and epsilon > 0, find delta > 0 such that
            |x - c| < delta implies |f(x) - f(c)| < epsilon.
            """,
            "expected_domain": MathematicalDomain.ANALYSIS,
            "expected_difficulty": 6
        },
        "topology": {
            "title": "Compactness in R",
            "description": """
            Prove the Heine-Borel Theorem: A subset of R^n is compact
            if and only if it is closed and bounded.

            This requires understanding of open covers, sequential compactness,
            and the relationship between these concepts.
            """,
            "expected_domain": MathematicalDomain.TOPOLOGY,
            "expected_difficulty": 8
        },
        "general": {
            "title": "Build a web application",
            "description": """
            Create a web application with user authentication, database integration,
            and a responsive frontend using React and Node.js.
            """,
            "expected_domain": None,
            "expected_difficulty": 0
        }
    }


@pytest.fixture
def detector():
    """Create a mathematical detector instance."""
    return LeanMathematicalDetector(enable_llm=False)  # Use heuristic for tests


@pytest.fixture
def engine():
    """Create a Lean-enhanced decomposition engine."""
    if not LEAN_ENHANCED_ENGINE_AVAILABLE:
        pytest.skip("Lean-enhanced engine not available")
    return LeanEnhancedDecompositionEngine(
        enable_lean_detection=True,
        enable_evolution=True
    )


# =============================================================================
# MATHEMATICAL DETECTION TESTS
# =============================================================================

class TestMathematicalDetection:
    """Test mathematical problem detection."""

    @pytest.mark.asyncio
    async def test_detect_mathematical_problem(self, detector, mathematical_problems):
        """Test detection of mathematical problems."""
        # Test mathematical problems
        for problem_name, problem_data in mathematical_problems.items():
            if problem_name == "general":
                continue  # Skip non-mathematical problem

            metadata = detector.detect_mathematical_problem(
                problem_data["description"],
                problem_data["title"]
            )

            assert metadata.is_mathematical, f"Failed to detect {problem_name} as mathematical"
            assert metadata.domain == problem_data["expected_domain"], \
                f"Domain mismatch for {problem_name}: got {metadata.domain}, expected {problem_data['expected_domain']}"

            logger.info(f"[OK] {problem_name}: domain={metadata.domain.value}, difficulty={metadata.proof_difficulty}")

    @pytest.mark.asyncio
    async def test_detect_non_mathematical_problem(self, detector, mathematical_problems):
        """Test that non-mathematical problems are correctly identified."""
        problem = mathematical_problems["general"]
        metadata = detector.detect_mathematical_problem(
            problem["description"],
            problem["title"]
        )

        assert not metadata.is_mathematical
        assert metadata.domain is None

    def test_classify_problem_type(self, detector):
        """Test problem type classification."""
        test_cases = [
            ("Prove that all primes are odd", MathematicalProblemType.THEOREM_PROOF),
            ("Lemma: Every subgroup of a cyclic group is cyclic", MathematicalProblemType.LEMMA_PROOF),
            ("Define a group as a set with an operation...", MathematicalProblemType.DEFINITION_FORMALIZATION),
            ("Investigate the Goldbach conjecture", MathematicalProblemType.CONJECTURE_INVESTIGATION),
            ("Exercise 1.5: Find the derivative", MathematicalProblemType.EXERCISE_SOLUTION),
        ]

        for text, expected_type in test_cases:
            detected_type = detector._classify_problem_type(text.lower())
            assert detected_type == expected_type, f"Type mismatch for: {text}"

    def test_identify_domain(self, detector):
        """Test domain identification."""
        test_cases = [
            ("The group is abelian", MathematicalDomain.ALGEBRA),
            ("The limit converges to zero", MathematicalDomain.ANALYSIS),
            ("The space is compact", MathematicalDomain.TOPOLOGY),
            ("Every integer has a prime factor", MathematicalDomain.NUMBER_THEORY),
            ("The graph is connected", MathematicalDomain.COMBINATORICS),
            ("The triangle is isosceles", MathematicalDomain.GEOMETRY),
            ("The proposition implies the conclusion", MathematicalDomain.LOGIC),
            ("The set is uncountable", MathematicalDomain.SET_THEORY),
        ]

        for text, expected_domain in test_cases:
            detected_domain = detector._identify_domain(text.lower())
            assert detected_domain == expected_domain, f"Domain mismatch for: {text}"

    def test_estimate_proof_difficulty(self, detector):
        """Test proof difficulty estimation."""
        test_cases = [
            ("Prove 1+1=2", 3, MathematicalProblemType.COMPUTATION_PROBLEM),
            ("Prove there are infinite primes", 6, MathematicalProblemType.THEOREM_PROOF),
            ("Investigate the Riemann Hypothesis", 9, MathematicalProblemType.CONJECTURE_INVESTIGATION),
        ]

        for text, min_difficulty, problem_type in test_cases:
            difficulty = detector._estimate_proof_difficulty(text.lower(), problem_type)
            assert difficulty >= min_difficulty, f"Difficulty too low for: {text} (got {difficulty}, expected >= {min_difficulty})"
            assert 1 <= difficulty <= 10, f"Difficulty out of range: {difficulty}"


# =============================================================================
# EVOLUTIONARY STRATEGY TESTS
# =============================================================================

class TestEvolutionaryStrategy:
    """Test evolutionary strategy suggestion."""

    def test_suggest_evolutionary_strategy(self, detector):
        """Test evolutionary strategy suggestion."""
        test_cases = [
            # (difficulty, formalization_complexity, expected_strategy_range)
            (4, 4, None),  # Too simple for evolution
            (6, 6, EvolutionaryStrategyType.STANDARD_EVOLUTION),
            (7, 7, EvolutionaryStrategyType.HYBRID_EVOLUTIONARY),
            (9, 9, EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION),
        ]

        for difficulty, complexity, expected_strategy in test_cases:
            # Create mock metadata
            metadata = MathematicalProblemMetadata(
                is_mathematical=True,
                proof_difficulty=difficulty,
                formalization_complexity=complexity
            )

            # Detect to get strategy
            test_text = f"Prove something with difficulty {difficulty}"
            detected_metadata = detector.detect_mathematical_problem(test_text)
            detected_strategy = detected_metadata.recommended_evolutionary_strategy

            if expected_strategy is None:
                assert detected_strategy is None, f"Expected no strategy for difficulty {difficulty}"
            else:
                assert detected_strategy is not None, f"Expected strategy for difficulty {difficulty}"
                # Check if strategy is in appropriate range
                if difficulty >= 9:
                    assert detected_strategy in [
                        EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION,
                        EvolutionaryStrategyType.HYBRID_EVOLUTIONARY
                    ], f"Strategy {detected_strategy} not appropriate for difficulty {difficulty}"

    @pytest.mark.asyncio
    async def test_generate_evolutionary_config(self):
        """Test evolutionary configuration generation."""
        suggestor = EvolutionaryStrategySuggestor()

        # Test various metadata scenarios
        test_cases = [
            # Simple problem - no evolution
            MathematicalProblemMetadata(
                is_mathematical=True,
                proof_difficulty=4,
                formalization_complexity=4
            ),
            # Medium problem - standard evolution
            MathematicalProblemMetadata(
                is_mathematical=True,
                proof_difficulty=6,
                formalization_complexity=6,
                recommended_evolutionary_strategy=EvolutionaryStrategyType.STANDARD_EVOLUTION
            ),
            # Hard problem - adversarial evolution
            MathematicalProblemMetadata(
                is_mathematical=True,
                proof_difficulty=9,
                formalization_complexity=9,
                recommended_evolutionary_strategy=EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION
            ),
        ]

        for metadata in test_cases:
            config = suggestor.suggest_strategy(metadata)

            if metadata.proof_difficulty < 5:
                assert not config.get("enable_evolution"), "Evolution should not be enabled for simple problems"
            else:
                assert config.get("enable_evolution"), "Evolution should be enabled for complex problems"
                assert "population_size" in config
                assert "max_generations" in config
                assert config["population_size"] >= 10
                assert config["max_generations"] >= 20

                # Check strategy-specific configs
                if metadata.recommended_evolutionary_strategy == EvolutionaryStrategyType.ADVERSARIAL_EVOLUTION:
                    assert "adversarial_epochs" in config
                elif metadata.recommended_evolutionary_strategy == EvolutionaryStrategyType.SELF_PLAY:
                    assert "self_play_episodes" in config


# =============================================================================
# DECOMPOSITION ENGINE TESTS
# =============================================================================

class TestLeanEnhancedDecomposition:
    """Test Lean-enhanced decomposition engine."""

    @pytest.mark.asyncio
    async def test_detect_and_route(self, mathematical_problems):
        """Test detection and routing of mathematical problems."""
        if not LEAN_ENHANCED_ENGINE_AVAILABLE:
            pytest.skip("Lean-enhanced engine not available")

        # Test mathematical problem
        problem = mathematical_problems["infinite_primes"]
        problem_def = ProblemDefinition(
            id="test_001",
            title=problem["title"],
            description=problem["description"],
            problem_type="theorem_proof",
            domain_context=DomainContext(
                domain="number_theory",
                subdomain=None,
                related_domains=[],
                domain_knowledge={}
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=2.0,
                domain_complexity=5.0,
                integration_complexity=3.0,
                overall_complexity=4.0,
                explanation="Test problem"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )

        plan, metadata = await detect_and_route_mathematical_problem(
            problem_def,
            enable_lean=True,
            enable_evolution=True
        )

        assert metadata is not None
        assert metadata.is_mathematical
        assert plan is not None

        logger.info(f"[OK] Detection and routing: domain={metadata.domain.value if metadata.domain else 'N/A'}")

    @pytest.mark.asyncio
    async def test_non_mathematical_routing(self, mathematical_problems):
        """Test routing of non-mathematical problems."""
        if not LEAN_ENHANCED_ENGINE_AVAILABLE:
            pytest.skip("Lean-enhanced engine not available")

        problem = mathematical_problems["general"]
        problem_def = ProblemDefinition(
            id="test_002",
            title=problem["title"],
            description=problem["description"],
            problem_type="software_development",
            domain_context=DomainContext(
                domain="software_engineering",
                subdomain="web_development",
                related_domains=[],
                domain_knowledge={}
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=5.0,
                computational_complexity=5.0,
                domain_complexity=5.0,
                integration_complexity=5.0,
                overall_complexity=5.0,
                explanation="Test problem"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )

        plan, metadata = await detect_and_route_mathematical_problem(
            problem_def,
            enable_lean=True
        )

        assert metadata is not None
        assert not metadata.is_mathematical
        assert plan is None  # No Lean decomposition for non-mathematical

        logger.info("[OK] Non-mathematical problem correctly identified")


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, mathematical_problems):
        """Test complete workflow from detection to evolutionary config."""
        if not LEAN_ENHANCED_ENGINE_AVAILABLE:
            pytest.skip("Lean-enhanced engine not available")

        # Step 1: Detect mathematical problem
        problem = mathematical_problems["infinite_primes"]
        detector = LeanMathematicalDetector(enable_llm=False)
        metadata = detector.detect_mathematical_problem(
            problem["description"],
            problem["title"]
        )

        assert metadata.is_mathematical
        logger.info(f"[OK] Step 1: Detected as {metadata.domain.value}")

        # Step 2: Generate evolutionary config
        config = await generate_evolutionary_config(metadata)
        if metadata.requires_evolution:
            assert config.get("enable_evolution")
            logger.info(f"[OK] Step 2: Evolutionary config generated with strategy {config.get('strategy_type')}")
        else:
            logger.info("[OK] Step 2: Evolution not required for this problem")

        # Step 3: Create problem definition
        problem_def = ProblemDefinition(
            id="integration_test_001",
            title=problem["title"],
            description=problem["description"],
            problem_type="theorem_proof",
            domain_context=DomainContext(
                domain=metadata.domain.value if metadata.domain else "general",
                subdomain=None,
                related_domains=[],
                domain_knowledge={}
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=float(metadata.proof_difficulty),
                computational_complexity=float(metadata.formalization_complexity * 0.8),
                domain_complexity=float(metadata.formalization_complexity * 0.9),
                integration_complexity=float(metadata.formalization_complexity * 0.7),
                overall_complexity=float(metadata.formalization_complexity),
                explanation="Integration test problem"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )

        # Step 4: Decompose with Lean-enhanced engine
        engine = LeanEnhancedDecompositionEngine(
            enable_lean_detection=True,
            enable_evolution=True
        )
        plan = await engine.decompose_with_leanaide(problem_def)

        assert plan is not None
        assert len(plan.sub_problems) > 0
        logger.info(f"[OK] Step 3: Decomposition created {len(plan.sub_problems)} sub-problems")

        # Step 5: Verify Lean metadata in sub-problems
        lean_subproblems = [
            sp for sp in plan.sub_problems
            if sp.metadata.get("lean_formalization")
        ]

        logger.info(f"[OK] Step 4: Found {len(lean_subproblems)} Lean-formalizable sub-problems")

        # Step 6: Check evolutionary config in sub-problems
        for sp in lean_subproblems:
            evo_config = sp.metadata.get("evolutionary_config")
            if evo_config:
                logger.info(f"  - {sp.title}: evolution={evo_config.get('strategy_type')}")


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests for the Lean-enhanced engine."""

    @pytest.mark.asyncio
    async def test_detection_performance(self, mathematical_problems):
        """Test detection performance."""
        import time

        detector = LeanMathematicalDetector(enable_llm=False)

        # Test each problem
        for problem_name, problem_data in mathematical_problems.items():
            start_time = time.time()
            metadata = detector.detect_mathematical_problem(
                problem_data["description"],
                problem_data["title"]
            )
            elapsed = time.time() - start_time

            # Detection should be fast (< 1 second for heuristic)
            assert elapsed < 1.0, f"Detection too slow for {problem_name}: {elapsed}s"
            logger.info(f"[OK] {problem_name}: detected in {elapsed:.3f}s")

    @pytest.mark.asyncio
    async def test_decomposition_performance(self, mathematical_problems):
        """Test decomposition performance."""
        if not LEAN_ENHANCED_ENGINE_AVAILABLE:
            pytest.skip("Lean-enhanced engine not available")

        import time

        engine = LeanEnhancedDecompositionEngine(
            enable_lean_detection=True,
            enable_evolution=True
        )

        # Test a simple mathematical problem
        problem = mathematical_problems["infinite_primes"]
        problem_def = ProblemDefinition(
            id="perf_test_001",
            title=problem["title"],
            description=problem["description"],
            problem_type="theorem_proof",
            domain_context=DomainContext(
                domain="number_theory",
                subdomain=None,
                related_domains=[],
                domain_knowledge={}
            ),
            complexity_score=ComplexityScore(
                cognitive_complexity=6.0,
                computational_complexity=2.0,
                domain_complexity=5.0,
                integration_complexity=3.0,
                overall_complexity=4.0,
                explanation="Performance test"
            ),
            constraints=[],
            success_criteria=[],
            stakeholders=[],
            resources_available={}
        )

        start_time = time.time()
        plan = await engine.decompose_with_leanaide(problem_def)
        elapsed = time.time() - start_time

        # Decomposition should complete in reasonable time
        assert elapsed < 10.0, f"Decomposition too slow: {elapsed}s"
        logger.info(f"[OK] Decomposition completed in {elapsed:.3f}s")


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def run_tests():
    """Run all tests."""
    pytest.main([__file__, "-v", "-s"])


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run tests
    run_tests()
