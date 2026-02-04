"""Comprehensive tests for ICR (Iterative Contextual Refinements) integration.

Tests all components: Generator, Critic, Refiner, Judge, and ICREngine,
along with the Knowledge Graph integration layer.
"""

import pytest
import time
from typing import Dict, Any, List

from integrations.icr import (
    Generator,
    GenerationResult,
    GenerationStrategy,
    Critic,
    CritiqueResult,
    Issue,
    IssueType,
    Severity,
    Suggestion,
    CritiqueCriteria,
    Refiner,
    RefinementStrategy,
    RefinementTracker,
    Change,
    RefinedOutput,
    Judge,
    EvaluationResult,
    ComparisonResult,
    Criteria,
    EvaluationStatus,
    ICREngine,
    RefinementResult,
    IterationResult,
    refine_content,
)

from knowledge_engine.integrations.icr import (
    ICRKGIntegration,
    RefinedExtraction,
    ImprovedQuery,
    RefinedEntities,
    OptimizedKG,
    RefinedSchema,
)


# =============================================================================
# Generator Tests
# =============================================================================

class TestGenerator:
    """Tests for the Generator class."""
    
    def test_generator_initialization(self):
        """Test generator initialization with defaults."""
        gen = Generator()
        assert gen.default_strategy == GenerationStrategy.DIRECT
        assert gen.temperature == 0.7
        assert gen.max_tokens == 2048
        assert gen._backend is None
    
    def test_generator_custom_initialization(self):
        """Test generator with custom parameters."""
        gen = Generator(
            default_strategy=GenerationStrategy.CHAIN_OF_THOUGHT,
            temperature=0.5,
            max_tokens=1024,
        )
        assert gen.default_strategy == GenerationStrategy.CHAIN_OF_THOUGHT
        assert gen.temperature == 0.5
        assert gen.max_tokens == 1024
    
    def test_generator_basic(self):
        """Test basic generation."""
        gen = Generator()
        result = gen.generate("Test prompt")
        
        assert isinstance(result, GenerationResult)
        assert isinstance(result.content, str)
        assert len(result.content) > 0
        assert 0 <= result.confidence <= 1
        assert result.generation_time >= 0
        assert result.tokens_used >= 0
    
    def test_generator_with_context(self):
        """Test generation with context."""
        gen = Generator()
        context = {"language": "python", "style": "functional"}
        result = gen.generate("Write a function", context=context)
        
        assert isinstance(result, GenerationResult)
        assert result.metadata.get("language") == "python"
    
    def test_generator_variants(self):
        """Test variant generation."""
        gen = Generator()
        variants = gen.generate_variants("Test prompt", n=3)
        
        assert len(variants) == 3
        for variant in variants:
            assert isinstance(variant, GenerationResult)
            assert variant.metadata.get("variant_index") is not None
    
    def test_generator_variants_invalid_n(self):
        """Test variant generation with invalid n."""
        gen = Generator()
        with pytest.raises(ValueError):
            gen.generate_variants("Test", n=0)
    
    def test_generation_result_validation(self):
        """Test GenerationResult validation."""
        # Test confidence clamping
        result = GenerationResult(
            content="Test",
            confidence=1.5,  # Should be clamped to 1.0
        )
        assert result.confidence == 1.0
        
        result = GenerationResult(
            content="Test",
            confidence=-0.5,  # Should be clamped to 0.0
        )
        assert result.confidence == 0.0
    
    def test_generation_result_serialization(self):
        """Test GenerationResult serialization."""
        result = GenerationResult(
            content="Test content",
            confidence=0.85,
            generation_time=1.5,
            tokens_used=100,
        )
        
        data = result.to_dict()
        restored = GenerationResult.from_dict(data)
        
        assert restored.content == result.content
        assert restored.confidence == result.confidence
    
    def test_generator_stats(self):
        """Test generator statistics."""
        gen = Generator()
        
        # Generate some content
        gen.generate("Test 1")
        gen.generate("Test 2")
        
        stats = gen.get_stats()
        assert stats["total_generations"] == 2
        assert stats["has_backend"] is False
    
    def test_generator_set_backend(self):
        """Test setting generation backend."""
        gen = Generator()
        
        def mock_backend(prompt: str, params: Dict) -> str:
            return f"Generated: {prompt}"
        
        gen.set_backend(mock_backend)
        assert gen._backend is not None
        
        result = gen.generate("Test")
        assert "Generated: Test" in result.content


# =============================================================================
# Critic Tests
# =============================================================================

class TestCritic:
    """Tests for the Critic class."""
    
    def test_critic_initialization(self):
        """Test critic initialization."""
        critic = Critic()
        assert critic.auto_suggest is True
        assert critic._critique_count == 0
    
    def test_critic_basic_critique(self):
        """Test basic critique functionality."""
        critic = Critic()
        result = critic.critique("This is a test output.")
        
        assert isinstance(result, CritiqueResult)
        assert 0 <= result.score <= 1
        assert isinstance(result.issues, list)
        assert isinstance(result.suggestions, list)
        assert isinstance(result.strengths, list)
    
    def test_critique_result_properties(self):
        """Test CritiqueResult properties."""
        # Critique with critical issues
        issue = Issue(
            type=IssueType.ACCURACY,
            severity=Severity.CRITICAL,
            description="Critical issue",
        )
        result = CritiqueResult(
            score=0.5,
            issues=[issue],
            suggestions=[],
        )
        
        assert result.has_critical_issues is True
        assert result.issue_count["critical"] == 1
    
    def test_critique_without_critical_issues(self):
        """Test critique without critical issues."""
        issue = Issue(
            type=IssueType.CLARITY,
            severity=Severity.MINOR,
            description="Minor clarity issue",
        )
        result = CritiqueResult(
            score=0.8,
            issues=[issue],
            suggestions=[],
        )
        
        assert result.has_critical_issues is False
        assert result.issue_count["minor"] == 1
    
    def test_criteria_configuration(self):
        """Test critique criteria configuration."""
        criteria = CritiqueCriteria(
            check_accuracy=True,
            check_completeness=False,
            min_length=100,
            max_length=1000,
        )
        
        critic = Critic(default_criteria=criteria)
        
        # Short text should trigger completeness issue
        result = critic.critique("Short")
        completeness_issues = [i for i in result.issues if i.type == IssueType.COMPLETENESS]
        assert len(completeness_issues) > 0
    
    def test_criteria_required_elements(self):
        """Test criteria with required elements."""
        criteria = CritiqueCriteria(
            required_elements=["conclusion", "summary"],
        )
        
        critic = Critic(default_criteria=criteria)
        result = critic.critique("This text has no required elements.")
        
        completeness_issues = [i for i in result.issues if i.type == IssueType.COMPLETENESS]
        assert len(completeness_issues) >= 2
    
    def test_criteria_forbidden_patterns(self):
        """Test criteria with forbidden patterns."""
        criteria = CritiqueCriteria(
            forbidden_patterns=["badword"],
        )
        
        critic = Critic(default_criteria=criteria)
        result = critic.critique("This contains badword which is forbidden.")
        
        style_issues = [i for i in result.issues if i.type == IssueType.STYLE]
        assert len(style_issues) > 0
    
    def test_identify_issues_specific_types(self):
        """Test identifying specific issue types."""
        critic = Critic()
        
        issues = critic.identify_issues(
            "This is a test with vague words like maybe and perhaps.",
            issue_types={IssueType.CLARITY},
        )
        
        clarity_issues = [i for i in issues if i.type == IssueType.CLARITY]
        assert len(clarity_issues) > 0
    
    def test_suggestion_generation(self):
        """Test suggestion generation."""
        critic = Critic(auto_suggest=True)
        
        issue = Issue(
            type=IssueType.GRAMMAR,
            severity=Severity.MINOR,
            description="Grammar issue",
        )
        
        suggestions = critic.suggest_improvements("Test", [issue])
        assert len(suggestions) > 0
        assert suggestions[0].issue == issue
        assert suggestions[0].automated is True
    
    def test_issue_serialization(self):
        """Test Issue serialization."""
        issue = Issue(
            type=IssueType.ACCURACY,
            severity=Severity.MAJOR,
            description="Test issue",
            location="line 5",
        )
        
        data = issue.to_dict()
        restored = Issue.from_dict(data)
        
        assert restored.type == issue.type
        assert restored.severity == issue.severity
        assert restored.description == issue.description
        assert restored.location == issue.location
    
    def test_critique_serialization(self):
        """Test CritiqueResult serialization."""
        issue = Issue(
            type=IssueType.CLARITY,
            severity=Severity.MINOR,
            description="Test",
        )
        suggestion = Suggestion(
            issue=issue,
            fix="Fix it",
            priority=1,
        )
        
        result = CritiqueResult(
            score=0.8,
            issues=[issue],
            suggestions=[suggestion],
            strengths=["Good structure"],
        )
        
        data = result.to_dict()
        assert data["score"] == 0.8
        assert len(data["issues"]) == 1
        assert len(data["suggestions"]) == 1


# =============================================================================
# Refiner Tests
# =============================================================================

class TestRefiner:
    """Tests for the Refiner class."""
    
    def test_refiner_initialization(self):
        """Test refiner initialization."""
        refiner = Refiner()
        assert refiner.strategy == RefinementStrategy.HYBRID
        assert refiner.max_changes_per_iteration == 10
        assert refiner.preserve_structure is True
    
    def test_refiner_custom_initialization(self):
        """Test refiner with custom parameters."""
        refiner = Refiner(
            strategy=RefinementStrategy.INCREMENTAL,
            max_changes_per_iteration=5,
            preserve_structure=False,
        )
        assert refiner.strategy == RefinementStrategy.INCREMENTAL
        assert refiner.max_changes_per_iteration == 5
        assert refiner.preserve_structure is False
    
    def test_refinement_tracker(self):
        """Test RefinementTracker functionality."""
        tracker = RefinementTracker()
        
        tracker.start_tracking()
        assert tracker._start_time is not None
        
        # Record scores
        tracker.record_score(0.6)
        tracker.record_score(0.7)
        tracker.record_score(0.75)
        
        assert len(tracker.improvement_history) == 3
        assert tracker._iteration_count == 3
        
        # Check trend
        assert tracker.improvement_trend in ["improving", "stable"]
    
    def test_refinement_tracker_convergence(self):
        """Test convergence detection."""
        tracker = RefinementTracker()
        
        # Record stable scores (converged)
        for _ in range(5):
            tracker.record_score(0.85)
        
        assert tracker.has_converged is True
    
    def test_change_creation(self):
        """Test Change dataclass."""
        change = Change(
            description="Fixed grammar",
            issue_type=IssueType.GRAMMAR,
            before="teh",
            after="the",
        )
        
        assert change.description == "Fixed grammar"
        assert change.before == "teh"
        assert change.after == "the"
        assert change.timestamp is not None
    
    def test_refine_with_mock_critique(self):
        """Test refinement with mock critique."""
        refiner = Refiner(strategy=RefinementStrategy.INCREMENTAL)
        
        issue = Issue(
            type=IssueType.GRAMMAR,
            severity=Severity.MINOR,
            description="Double spaces",
        )
        suggestion = Suggestion(
            issue=issue,
            fix="Remove double spaces",
            automated=True,
        )
        
        class MockCritique:
            def __init__(self):
                self.issues = [issue]
                self.suggestions = [suggestion]
        
        critique = MockCritique()
        output = "This  has  double  spaces."
        
        result = refiner.refine(output, critique)
        
        assert isinstance(result, RefinedOutput)
        assert result.original_content == output
        assert result.change_count > 0
        assert result.improvement_score >= 0
    
    def test_apply_suggestion(self):
        """Test applying a single suggestion."""
        refiner = Refiner()
        
        issue = Issue(
            type=IssueType.CONCISENESS,
            severity=Severity.MINOR,
            description="Very redundant phrase",
        )
        suggestion = Suggestion(
            issue=issue,
            fix="Remove redundant words",
            automated=True,
        )
        
        output = "This is very good and very nice."
        result = refiner.apply_suggestion(output, suggestion)
        
        assert result.applied is True
        assert result.content != output
    
    def test_apply_suggestion_manual(self):
        """Test applying non-automated suggestion."""
        refiner = Refiner()
        
        issue = Issue(
            type=IssueType.ACCURACY,
            severity=Severity.MAJOR,
            description="Needs fact checking",
        )
        suggestion = Suggestion(
            issue=issue,
            fix="Verify facts",
            automated=False,  # Requires manual intervention
        )
        
        output = "Some statement."
        result = refiner.apply_suggestion(output, suggestion)
        
        assert result.applied is False
        assert result.content == output
    
    def test_merge_suggestions(self):
        """Test merging multiple suggestions."""
        refiner = Refiner()
        
        suggestions = [
            Suggestion(
                issue=Issue(IssueType.GRAMMAR, Severity.MINOR, "Issue 1"),
                fix="Fix 1",
                priority=1,
                automated=True,
            ),
            Suggestion(
                issue=Issue(IssueType.GRAMMAR, Severity.MINOR, "Issue 2"),
                fix="Fix 2",
                priority=2,
                automated=True,
            ),
        ]
        
        output = "This  has  issues."
        result = refiner.merge_suggestions(output, suggestions)
        
        assert len(result.applied_suggestions) > 0


# =============================================================================
# Judge Tests
# =============================================================================

class TestJudge:
    """Tests for the Judge class."""
    
    def test_judge_initialization(self):
        """Test judge initialization."""
        judge = Judge()
        assert judge.default_threshold == 0.85
        assert judge.use_critique_integration is True
    
    def test_judge_custom_initialization(self):
        """Test judge with custom parameters."""
        judge = Judge(
            default_threshold=0.9,
            use_critique_integration=False,
        )
        assert judge.default_threshold == 0.9
        assert judge.use_critique_integration is False
    
    def test_criteria_creation(self):
        """Test Criteria creation."""
        criteria = Criteria(
            accuracy=0.3,
            completeness=0.2,
            clarity=0.2,
            conciseness=0.1,
            correctness=0.15,
            consistency=0.05,
        )
        
        all_criteria = criteria.get_all_criteria()
        assert len(all_criteria) == 6
        assert abs(sum(all_criteria.values()) - 1.0) < 0.01
    
    def test_criteria_normalization(self):
        """Test criteria weight normalization."""
        # Weights that don't sum to 1
        criteria = Criteria(
            accuracy=1.0,
            completeness=1.0,
            clarity=1.0,
        )
        
        # Should be normalized
        assert abs(criteria.accuracy - 0.303) < 0.01  # 1/3.3 with all 6 default criteria
    
    def test_criteria_presets(self):
        """Test criteria presets."""
        strict = Criteria.strict()
        assert strict.correctness > strict.conciseness
        
        balanced = Criteria.balanced()
        assert abs(balanced.accuracy - balanced.completeness) < 0.01
        
        creative = Criteria.creative()
        assert creative.clarity > creative.correctness
    
    def test_evaluation_result(self):
        """Test EvaluationResult creation."""
        result = EvaluationResult(
            score=0.85,
            passed=True,
            criteria_scores={"accuracy": 0.9, "completeness": 0.8},
            feedback="Good work",
        )
        
        assert result.score == 0.85
        assert result.passed is True
        assert result.status == EvaluationStatus.PASSED
    
    def test_evaluation_result_score_clamping(self):
        """Test score clamping in EvaluationResult."""
        result = EvaluationResult(score=1.5, passed=False)
        assert result.score == 1.0
        
        result = EvaluationResult(score=-0.5, passed=False)
        assert result.score == 0.0
    
    def test_judge_evaluate(self):
        """Test basic evaluation."""
        judge = Judge()
        result = judge.evaluate("This is a well-structured output with good content.")
        
        assert isinstance(result, EvaluationResult)
        assert 0 <= result.score <= 1
        assert isinstance(result.criteria_scores, dict)
        assert len(result.feedback) > 0
    
    def test_judge_meets_threshold(self):
        """Test threshold checking."""
        judge = Judge(default_threshold=0.8)
        
        # High quality content should pass
        high_quality = "This is excellent content with perfect structure. It has clarity, completeness, and correctness."
        assert judge.meets_threshold(high_quality, threshold=0.8) is True
    
    def test_judge_compare(self):
        """Test comparing two outputs."""
        judge = Judge()
        
        original = "Basic output."
        refined = "Excellent output with great structure and clarity. Much better content here."
        
        result = judge.compare(original, refined)
        
        assert isinstance(result, ComparisonResult)
        assert result.winner in ["original", "refined", "tie"]
        assert isinstance(result.improvements, list)
        assert isinstance(result.regressions, list)
    
    def test_comparison_result_properties(self):
        """Test ComparisonResult properties."""
        # Improvement
        result = ComparisonResult(
            winner="refined",
            score_delta=0.15,
            improvements=["Better clarity"],
        )
        assert result.is_improvement is True
        assert result.is_regression is False
        
        # Regression
        result = ComparisonResult(
            winner="original",
            score_delta=-0.15,
            regressions=["Worse clarity"],
        )
        assert result.is_improvement is False
        assert result.is_regression is True
        
        # Tie
        result = ComparisonResult(
            winner="tie",
            score_delta=0.01,
        )
        assert result.is_improvement is False
        assert result.is_regression is False
    
    def test_evaluation_serialization(self):
        """Test EvaluationResult serialization."""
        result = EvaluationResult(
            score=0.85,
            passed=True,
            criteria_scores={"accuracy": 0.9},
            feedback="Good",
        )
        
        data = result.to_dict()
        assert data["score"] == 0.85
        assert data["passed"] is True
        assert data["status"] == "passed"


# =============================================================================
# ICREngine Tests
# =============================================================================

class TestICREngine:
    """Tests for the ICREngine class."""
    
    def test_engine_initialization(self):
        """Test engine initialization."""
        engine = ICREngine()
        
        assert isinstance(engine.generator, Generator)
        assert isinstance(engine.critic, Critic)
        assert isinstance(engine.refiner, Refiner)
        assert isinstance(engine.judge, Judge)
        assert engine.max_iterations == 5
        assert engine.quality_threshold == 0.9
    
    def test_engine_custom_initialization(self):
        """Test engine with custom parameters."""
        engine = ICREngine(
            max_iterations=10,
            quality_threshold=0.95,
            early_stopping=False,
            patience=3,
        )
        
        assert engine.max_iterations == 10
        assert engine.quality_threshold == 0.95
        assert engine.early_stopping is False
        assert engine.patience == 3
    
    def test_engine_refine_basic(self):
        """Test basic refinement."""
        engine = ICREngine(
            max_iterations=2,
            quality_threshold=0.95,  # High threshold to force multiple iterations
        )
        
        result = engine.refine("Write a Python function")
        
        assert isinstance(result, RefinementResult)
        assert isinstance(result.final_output, str)
        assert result.iterations >= 0
        assert isinstance(result.improvement_history, list)
        assert isinstance(result.critique_history, list)
    
    def test_engine_refine_with_initial_output(self):
        """Test refinement with initial output."""
        engine = ICREngine(max_iterations=2)
        
        initial = "def func():\n    pass  # TODO: implement"
        result = engine.refine(
            prompt="Improve this function",
            initial_output=initial,
        )
        
        assert isinstance(result, RefinementResult)
        assert result.metadata.get("generator_used") is False
    
    def test_engine_refine_already_high_quality(self):
        """Test refinement with already high-quality content."""
        engine = ICREngine(
            max_iterations=3,
            quality_threshold=0.6,  # Low threshold
        )
        
        # High quality initial content
        high_quality = """
This is excellent content with perfect structure.
It demonstrates clarity, completeness, and correctness.
The writing is clear and concise with proper grammar.
"""
        
        result = engine.refine(
            prompt="Improve this",
            initial_output=high_quality,
        )
        
        # Should stop early as content is already good
        assert result.stopped_reason == "threshold_met_initially" or result.iterations < 3
    
    def test_iterate_once(self):
        """Test single iteration."""
        engine = ICREngine()
        
        result = engine.iterate_once(
            current_output="Test content with issues.",
            iteration=1,
            threshold=0.9,
            context={},
        )
        
        assert isinstance(result, IterationResult)
        assert result.iteration == 1
        assert isinstance(result.output, str)
        assert isinstance(result.critique, CritiqueResult)
        assert isinstance(result.evaluation, EvaluationResult)
    
    def test_should_continue(self):
        """Test continuation logic."""
        engine = ICREngine(max_iterations=5, quality_threshold=0.9)
        
        # Should continue if below threshold and below max iterations
        assert engine.should_continue(0.7, 1) is True
        
        # Should not continue if above threshold
        assert engine.should_continue(0.95, 1) is False
        
        # Should not continue if at max iterations
        assert engine.should_continue(0.7, 5) is False
    
    def test_get_best_version(self):
        """Test getting best version from history."""
        engine = ICREngine()
        
        # Create mock iteration results
        class MockEval:
            def __init__(self, score):
                self.score = score
        
        class MockResult:
            def __init__(self, score):
                self.evaluation = MockEval(score)
        
        history = [
            MockResult(0.6),
            MockResult(0.8),
            MockResult(0.75),
        ]
        
        best = engine.get_best_version(history)
        assert best.evaluation.score == 0.8
    
    def test_quick_refine(self):
        """Test quick refine convenience method."""
        engine = ICREngine()
        
        result = engine.quick_refine("Test prompt", target_iterations=2)
        
        assert isinstance(result, str)
        assert len(result) > 0
    
    def test_batch_refine(self):
        """Test batch refinement."""
        engine = ICREngine(max_iterations=1)
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = engine.batch_refine(prompts)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, RefinementResult)
    
    def test_refinement_result_properties(self):
        """Test RefinementResult properties."""
        result = RefinementResult(
            final_output="Final",
            iterations=3,
            improvement_history=[0.6, 0.7, 0.8],
            final_score=0.8,
            critique_history=[],
        )
        
        assert result.total_improvement == pytest.approx(0.2, abs=0.001)
        assert result.average_improvement_per_iteration == pytest.approx(0.067, rel=0.01)
    
    def test_iteration_result_serialization(self):
        """Test IterationResult serialization."""
        critique = CritiqueResult(
            score=0.7,
            issues=[],
            suggestions=[],
        )
        evaluation = EvaluationResult(
            score=0.8,
            passed=True,
        )
        
        result = IterationResult(
            iteration=1,
            output="Test",
            critique=critique,
            evaluation=evaluation,
            improvement=0.1,
            converged=False,
            should_continue=True,
        )
        
        data = result.to_dict()
        assert data["iteration"] == 1
        assert data["improvement"] == 0.1
    
    def test_engine_stats(self):
        """Test engine statistics."""
        engine = ICREngine()
        
        # Run some operations
        engine.refine("Test", max_iterations=1)
        engine.refine("Test 2", max_iterations=1)
        
        stats = engine.get_stats()
        assert stats["total_runs"] == 2
        assert "generator" in stats
        assert "critic" in stats
        assert "refiner" in stats
        assert "judge" in stats
    
    def test_convenience_function(self):
        """Test refine_content convenience function."""
        result = refine_content(
            content="Test content to refine.",
            max_iterations=2,
            threshold=0.8,
        )
        
        assert isinstance(result, str)


# =============================================================================
# Knowledge Engine Integration Tests
# =============================================================================

class TestICRKGIntegration:
    """Tests for ICR KG Integration."""
    
    def test_integration_initialization(self):
        """Test KG integration initialization."""
        integration = ICRKGIntegration()
        
        assert isinstance(integration.engine, ICREngine)
        assert integration.max_iterations == 5
        assert integration.quality_threshold == 0.85
    
    def test_integration_custom_initialization(self):
        """Test KG integration with custom parameters."""
        integration = ICRKGIntegration(
            max_iterations=10,
            quality_threshold=0.9,
        )
        
        assert integration.max_iterations == 10
        assert integration.quality_threshold == 0.9
    
    def test_refine_kg_extraction(self):
        """Test KG extraction refinement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        text = "Apple Inc. was founded by Steve Jobs in Cupertino."
        
        result = integration.refine_kg_extraction(text)
        
        assert isinstance(result, RefinedExtraction)
        assert isinstance(result.entities, list)
        assert isinstance(result.relations, list)
        assert 0 <= result.confidence <= 1
        assert result.iterations >= 0
    
    def test_refine_entity_extraction(self):
        """Test entity extraction refinement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        text = "Microsoft and Google compete in cloud computing."
        initial_entities = [
            {"name": "Microsoft", "type": "Company"},
        ]
        
        result = integration.refine_entity_extraction(text, initial_entities)
        
        assert isinstance(result, RefinedExtraction)
        assert len(result.entities) >= 0
    
    def test_refine_relation_extraction(self):
        """Test relation extraction refinement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        text = "Apple makes iPhones and iPads."
        entities = [
            {"name": "Apple", "type": "Company"},
            {"name": "iPhone", "type": "Product"},
        ]
        initial_relations = []
        
        result = integration.refine_relation_extraction(text, entities, initial_relations)
        
        assert isinstance(result, RefinedExtraction)
        assert isinstance(result.relations, list)
    
    def test_improve_cypher_query(self):
        """Test Cypher query improvement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        query = "MATCH (n) RETURN n LIMIT 10"
        
        result = integration.improve_cypher_query(query)
        
        assert isinstance(result, ImprovedQuery)
        assert isinstance(result.query, str)
        assert len(result.query) > 0
        assert result.original_query == query
        assert isinstance(result.improvements, list)
        assert 0 <= result.performance_estimate <= 1
        assert 0 <= result.confidence <= 1
    
    def test_refine_entity_resolution(self):
        """Test entity resolution refinement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        entities = [
            {"name": "Apple Inc.", "type": "Company"},
            {"name": "Apple", "type": "Company"},
            {"name": "Microsoft", "type": "Company"},
        ]
        
        result = integration.refine_entity_resolution(entities)
        
        assert isinstance(result, RefinedEntities)
        assert isinstance(result.entities, list)
        assert result.duplicates_found >= 0
        assert result.merges_performed >= 0
        assert 0 <= result.confidence <= 1
    
    def test_optimize_kg_structure(self):
        """Test KG structure optimization."""
        integration = ICRKGIntegration(max_iterations=2)
        
        nodes = [
            {"id": 1, "type": "Person", "name": "Alice"},
            {"id": 2, "type": "Person", "name": "Bob"},
        ]
        edges = [
            {"source": 1, "target": 2, "type": "KNOWS"},
        ]
        
        result = integration.optimize_kg_structure(nodes, edges)
        
        assert isinstance(result, OptimizedKG)
        assert isinstance(result.optimizations, list)
        assert isinstance(result.metrics, dict)
    
    def test_iterative_schema_inference(self):
        """Test iterative schema inference."""
        integration = ICRKGIntegration(max_iterations=2)
        
        data = [
            {"type": "Person", "name": "Alice", "age": 30},
            {"type": "Person", "name": "Bob", "age": 25},
            {"type": "Company", "name": "Acme"},
        ]
        
        result = integration.iterative_schema_inference(data)
        
        assert isinstance(result, RefinedSchema)
        assert isinstance(result.schema, dict)
        assert isinstance(result.entity_types, list)
        assert isinstance(result.relation_types, list)
        assert 0 <= result.confidence <= 1
        assert 0 <= result.coverage <= 1
    
    def test_improve_kg_quality(self):
        """Test general KG quality improvement."""
        integration = ICRKGIntegration(max_iterations=2)
        
        kg = {
            "entities": [
                {"id": 1, "type": "Person", "name": "Alice"},
            ],
            "relations": [],
        }
        
        result = integration.improve_kg_quality(kg)
        
        assert isinstance(result, RefinedExtraction)
        assert result.iterations >= 0
    
    def test_converge_to_optimal(self):
        """Test generic convergence function."""
        integration = ICRKGIntegration(max_iterations=3)
        
        def judge_fn(state):
            # Simple judge: prefer longer strings
            return min(1.0, len(str(state)) / 100)
        
        result = integration.converge_to_optimal(
            initial="Start",
            judge_fn=judge_fn,
            max_iter=3,
        )
        
        assert "optimal" in result
        assert "score" in result
        assert "iterations" in result
        assert "history" in result
        assert isinstance(result["history"], list)
    
    def test_refined_extraction_dataclass(self):
        """Test RefinedExtraction dataclass."""
        result = RefinedExtraction(
            entities=[{"name": "Test"}],
            relations=[],
            confidence=0.85,
            iterations=3,
            improvement=0.15,
        )
        
        assert result.confidence == 0.85
        assert result.improvement == 0.15
    
    def test_improved_query_dataclass(self):
        """Test ImprovedQuery dataclass."""
        result = ImprovedQuery(
            query="MATCH (n) RETURN n",
            original_query="MATCH (n) RETURN n",
            improvements=["Added index"],
            performance_estimate=0.9,
            confidence=0.85,
        )
        
        assert result.performance_estimate == 0.9
        assert len(result.improvements) == 1
    
    def test_integration_stats(self):
        """Test integration statistics."""
        integration = ICRKGIntegration()
        
        # Perform operations
        integration.refine_kg_extraction("Test text")
        integration.improve_cypher_query("MATCH (n) RETURN n")
        
        stats = integration.get_stats()
        assert stats["operations_performed"] == 2
        assert "engine_stats" in stats


# =============================================================================
# Integration and E2E Tests
# =============================================================================

class TestIntegration:
    """Integration tests for the full ICR system."""
    
    def test_end_to_end_refinement(self):
        """Test complete end-to-end refinement flow."""
        # Create components
        generator = Generator()
        critic = Critic()
        refiner = Refiner()
        judge = Judge()
        
        engine = ICREngine(
            generator=generator,
            critic=critic,
            refiner=refiner,
            judge=judge,
            max_iterations=3,
            quality_threshold=0.8,
        )
        
        # Run refinement
        result = engine.refine("Write a Python docstring")
        
        # Verify results
        assert isinstance(result, RefinementResult)
        assert result.final_output is not None
        assert len(result.improvement_history) > 0
        assert result.final_score > 0
    
    def test_kg_integration_end_to_end(self):
        """Test complete KG integration flow."""
        integration = ICRKGIntegration(max_iterations=3)
        
        # Extract and refine
        text = """
        OpenAI is a company founded by Sam Altman, Greg Brockman, and others.
        It is based in San Francisco and develops AI systems like GPT-4.
        Microsoft has invested billions in OpenAI.
        """
        
        extraction = integration.refine_kg_extraction(text)
        
        # Improve any queries
        query = integration.improve_cypher_query(
            "MATCH (c:Company)-[:DEVELOPS]->(p:Product) RETURN c, p"
        )
        
        # Verify
        assert extraction.confidence > 0
        assert len(query.query) > 0
    
    def test_convergence_detection(self):
        """Test that convergence is properly detected."""
        engine = ICREngine(
            max_iterations=10,
            quality_threshold=0.99,  # High to force many iterations
            early_stopping=True,
            patience=2,
        )
        
        # Start with decent content
        initial = """
        This content is already reasonably good quality.
        It has proper structure and grammar.
        The sentences are clear and concise.
        """
        
        result = engine.refine(
            prompt="Improve this",
            initial_output=initial,
        )
        
        # Should have stopped due to no improvement (patience) or threshold
        assert result.iterations < 10 or result.stopped_reason in ["threshold_met", "no_improvement"]


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Tests for error handling."""
    
    def test_generation_error(self):
        """Test generation error handling."""
        gen = Generator()
        
        # Set a backend that raises an exception
        def failing_backend(prompt, params):
            raise RuntimeError("Generation failed")
        
        gen.set_backend(failing_backend)
        
        with pytest.raises(Exception):
            gen.generate("Test")


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """Performance-related tests."""
    
    def test_refinement_time_tracking(self):
        """Test that refinement time is tracked."""
        engine = ICREngine(max_iterations=2)
        
        start = time.time()
        result = engine.refine("Test prompt", max_iterations=2)
        elapsed = time.time() - start
        
        assert "total_time" in result.metadata  # Just verify the key exists
    
    def test_batch_processing(self):
        """Test batch processing performance."""
        engine = ICREngine(max_iterations=1)
        
        prompts = ["Prompt 1", "Prompt 2"]
        
        start = time.time()
        results = engine.batch_refine(prompts)
        elapsed = time.time() - start
        
        assert len(results) == 2
        # Should complete reasonably fast
        assert elapsed < 30  # Generous timeout


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
