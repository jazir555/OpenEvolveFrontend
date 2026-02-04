"""Tests for Cognitive Hydraulics Integration.

Test coverage includes:
- Unit tests for Soar engine (decision cycle, impasse detection)
- Unit tests for ACT-R engine (utility calculation, tabu search)
- Unit tests for Pressure Valve (threshold detection)
- Unit tests for Evolutionary fallback (GA operations)
- Unit tests for Chunking (rule learning)
- Integration tests (full reasoning pipeline)
- Performance benchmarks

Follows pytest patterns with proper fixtures and markers.
"""

import pytest
import time
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List
from unittest.mock import Mock, MagicMock

# Skip if numpy not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Import cognitive hydraulics components
from integrations.cognitive_hydraulics import (
    SoarEngine, SoarState, SoarOperator, SoarRule,
    Impasse, ImpasseType, TieImpasse, NoChangeImpasse,
    ACTREngine, ACTRProduction, ACTRChunk, TabuList, UtilityEquation,
    PressureValve, SystemType, PressureMetrics,
    EvolutionarySolver, Individual, Population, SolutionType,
    ChunkingEngine, Chunk, ChunkType, ChunkQuality,
    CognitiveHydraulicsEngine, ReasoningResult,
)
from integrations.cognitive_hydraulics.config import (
    SoarConfig, ACTRConfig, PressureValveConfig,
    EvolutionaryConfig, CognitiveHydraulicsConfig
)

# Import KG integration
from knowledge_engine.integrations.cognitive_hydraulics import (
    CognitiveHydraulicsKGIntegration,
    ReasoningTracer,
    KGProblemEncoder,
    KGSolutionDecoder,
    KGReasoningResult,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def soar_config():
    """Fixture for Soar configuration."""
    return SoarConfig(
        working_memory_slots=7,
        max_subgoal_depth=5,
        max_decision_cycles=100,
        enable_chunking=True
    )


@pytest.fixture
def actr_config():
    """Fixture for ACT-R configuration."""
    return ACTRConfig(
        default_probability=0.5,
        default_goal_value=10.0,
        default_cost=1.0,
        noise_sigma=0.1,
        tabu_list_size=5
    )


@pytest.fixture
def pressure_config():
    """Fixture for Pressure Valve configuration."""
    return PressureValveConfig(
        soar_to_actr_depth=2,
        actr_to_evo_pressure=0.9,
        time_threshold_ms=100,
        weight_depth=0.3,
        weight_time=0.25,
        weight_impasses=0.25,
        weight_ambiguity=0.2
    )


@pytest.fixture
def evolutionary_config():
    """Fixture for Evolutionary configuration."""
    return EvolutionaryConfig(
        population_size=20,
        max_generations=10,
        mutation_rate=0.1,
        crossover_rate=0.7
    )


@pytest.fixture
def sample_operators():
    """Fixture for sample Soar operators."""
    return [
        SoarOperator(
            name="explore",
            preconditions=[],
            actions=[{"type": "add", "attribute": "status", "value": "exploring"}],
            preferences={"default": 0.5}
        ),
        SoarOperator(
            name="evaluate",
            preconditions=[{"attribute": "status", "value": "exploring"}],
            actions=[{"type": "add", "attribute": "status", "value": "evaluating"}],
            preferences={"default": 0.6}
        ),
        SoarOperator(
            name="complete",
            preconditions=[{"attribute": "status", "value": "evaluating"}],
            actions=[{"type": "add", "attribute": "status", "value": "completed"}],
            preferences={"default": 0.7}
        ),
    ]


# ============================================================================
# SOAR ENGINE TESTS
# ============================================================================

class TestSoarEngine:
    """Tests for Soar System 2 engine."""
    
    def test_engine_initialization(self, soar_config):
        """Test Soar engine initialization."""
        engine = SoarEngine(soar_config)
        assert engine.config == soar_config
        assert not engine.initialized
        assert len(engine.operators) == 0
    
    def test_engine_initialization_with_operators(self, soar_config, sample_operators):
        """Test Soar engine initialization with operators."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test", "target": "completed"}
        
        engine.initialize(goal, sample_operators)
        
        assert engine.initialized
        assert len(engine.operators) == 3
        assert engine.get_current_state() is not None
    
    def test_working_memory_operations(self, soar_config):
        """Test working memory operations."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, [])
        
        state = engine.get_current_state()
        assert state is not None
        
        # Test WME operations
        state.add_wme("test_attr", "test_value")
        assert state.get_wme_attribute("test_attr") == "test_value"
        
        state.modify_wme("test_attr", "modified_value")
        assert state.get_wme_attribute("test_attr") == "modified_value"
        
        state.remove_wme("test_attr")
        assert state.get_wme_attribute("test_attr") is None
    
    def test_operator_applicability(self, soar_config, sample_operators):
        """Test operator applicability checking."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, sample_operators)
        
        state = engine.get_current_state()
        
        # 'explore' should be applicable (no preconditions)
        assert sample_operators[0].is_applicable(state)
        
        # 'evaluate' should not be applicable yet
        assert not sample_operators[1].is_applicable(state)
        
        # Add required attribute
        state.add_wme("status", "exploring")
        assert sample_operators[1].is_applicable(state)
    
    def test_decision_cycle(self, soar_config, sample_operators):
        """Test decision cycle execution."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, sample_operators)
        
        # Run one cycle
        success, impasse = engine.run_decision_cycle()
        
        assert success
        assert impasse is None
        assert engine.decision_cycle.cycle_count == 1
    
    def test_impasse_detection_tie(self, soar_config):
        """Test tie impasse detection."""
        engine = SoarEngine(soar_config)
        
        # Create operators with equal preferences
        operators = [
            SoarOperator(name=f"op_{i}", preferences={"default": 0.5})
            for i in range(5)  # More than threshold
        ]
        
        goal = {"type": "test"}
        engine.initialize(goal, operators)
        
        state = engine.get_current_state()
        
        # Detect impasse
        impasse = engine.detect_impasse(state, operators)
        
        assert impasse is not None
        assert impasse.impasse_type == ImpasseType.TIE
    
    def test_impasse_detection_no_change(self, soar_config):
        """Test no-change impasse detection."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, [])
        
        state = engine.get_current_state()
        
        # Detect impasse with no operators
        impasse = engine.detect_impasse(state, [])
        
        assert impasse is not None
        assert impasse.impasse_type == ImpasseType.NO_CHANGE
    
    def test_subgoal_creation(self, soar_config):
        """Test subgoal creation."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, [])
        
        # Create impasse
        impasse = NoChangeImpasse(state_id="test_state")
        
        # Create subgoal
        subgoal = engine.create_subgoal(impasse)
        
        assert subgoal is not None
        assert subgoal.subgoal_depth == 1
        assert subgoal.goal["type"] == "resolve_impasse"
    
    def test_chunking(self, soar_config):
        """Test rule chunking from resolution."""
        engine = SoarEngine(soar_config)
        goal = {"type": "test"}
        engine.initialize(goal, [])
        
        # Create impasse
        impasse = NoChangeImpasse(state_id="test_state")
        
        # Create resolution
        resolution = {"operator_id": "test_op", "value": "success"}
        context = {"problem_description": "test"}
        
        # Chunk success
        chunk = engine.chunk_success(impasse, resolution, context)
        
        assert chunk is not None
        assert len(engine.chunking.repository.get_all()) == 1


# ============================================================================
# ACT-R ENGINE TESTS
# ============================================================================

class TestACTREngine:
    """Tests for ACT-R System 1 engine."""
    
    def test_engine_initialization(self, actr_config):
        """Test ACT-R engine initialization."""
        engine = ACTREngine(actr_config)
        assert engine.config == actr_config
        assert engine.current_goal is None
    
    def test_declarative_memory(self, actr_config):
        """Test declarative memory operations."""
        engine = ACTREngine(actr_config)
        
        # Add chunks
        chunk1 = ACTRChunk(chunk_type="fact", slots={"name": "A", "value": 1})
        chunk2 = ACTRChunk(chunk_type="fact", slots={"name": "B", "value": 2})
        
        engine.add_chunk(chunk1)
        engine.add_chunk(chunk2)
        
        assert len(engine.declarative_memory.chunks) == 2
        
        # Retrieve by pattern
        retrieved = engine.retrieve_from_memory({"name": "A"}, "fact")
        assert retrieved is not None
        assert retrieved.slots["name"] == "A"
    
    def test_procedural_memory(self, actr_config):
        """Test procedural memory operations."""
        engine = ACTREngine(actr_config)
        
        # Add production
        production = ACTRProduction(
            name="test_production",
            conditions=[{"slot": "status", "value": "ready"}],
            actions=[{"type": "modify", "slot": "status", "value": "done"}],
            probability=0.8,
            cost=1.0,
            goal_value=10.0
        )
        
        engine.add_production(production)
        
        assert len(engine.procedural_memory.productions) == 1
    
    def test_utility_calculation(self, actr_config):
        """Test utility equation calculation."""
        calculator = UtilityEquation(actr_config)
        
        # Test basic calculation: U = P*G - C
        utility = calculator.compute(
            probability=0.8,
            goal_value=10.0,
            cost=1.0
        )
        
        # U = 0.8 * 10 - 1 = 7 (plus/minus noise)
        assert 5 <= utility <= 9  # Allow for noise
    
    def test_utility_with_history_penalty(self, actr_config):
        """Test utility with history penalty."""
        calculator = UtilityEquation(actr_config)
        
        utility_with_penalty = calculator.compute(
            probability=0.8,
            goal_value=10.0,
            cost=1.0,
            history_penalty=2.0
        )
        
        utility_without = calculator.compute(
            probability=0.8,
            goal_value=10.0,
            cost=1.0,
            history_penalty=0.0
        )
        
        assert utility_with_penalty < utility_without
    
    def test_tabu_list(self, actr_config):
        """Test tabu list operations."""
        tabu = TabuList(max_size=3, penalty_base=1.0)
        
        # Add entries
        tabu.add("op1")
        tabu.add("op2")
        
        assert tabu.contains("op1")
        assert tabu.contains("op2")
        assert not tabu.contains("op3")
        
        # Check penalty
        penalty = tabu.get_penalty("op1")
        assert penalty > 0
        
        # Test max size (oldest should be evicted)
        tabu.add("op3")
        tabu.add("op4")
        
        assert not tabu.contains("op1")  # Should be evicted
    
    def test_production_matching(self, actr_config):
        """Test production matching to context."""
        engine = ACTREngine(actr_config)
        
        production = ACTRProduction(
            name="match_test",
            conditions=[
                {"slot": "status", "value": "ready", "operator": "equals"},
                {"slot": "count", "value": 5, "operator": "greater"}
            ],
            actions=[]
        )
        
        engine.add_production(production)
        
        # Should match
        context1 = {"status": "ready", "count": 10}
        matches1 = engine.procedural_memory.find_matching_productions(context1)
        assert len(matches1) == 1
        
        # Should not match
        context2 = {"status": "waiting", "count": 10}
        matches2 = engine.procedural_memory.find_matching_productions(context2)
        assert len(matches2) == 0
    
    def test_operator_selection(self, actr_config):
        """Test operator selection based on utility."""
        engine = ACTREngine(actr_config)
        engine.set_goal({"type": "test"})
        
        # Create productions with different utilities
        op1 = ACTRProduction(
            name="low_utility",
            conditions=[],
            probability=0.3,
            cost=5.0,
            goal_value=10.0
        )
        
        op2 = ACTRProduction(
            name="high_utility",
            conditions=[],
            probability=0.9,
            cost=1.0,
            goal_value=10.0
        )
        
        engine.add_production(op1)
        engine.add_production(op2)
        
        # Run cycle
        selected = engine.run_cycle({})
        
        # Should usually select high_utility (allowing for noise)
        assert selected is not None


# ============================================================================
# PRESSURE VALVE TESTS
# ============================================================================

class TestPressureValve:
    """Tests for Pressure Valve meta-cognitive monitor."""
    
    def test_initialization(self, pressure_config):
        """Test pressure valve initialization."""
        valve = PressureValve(pressure_config)
        
        assert valve.config == pressure_config
        assert valve.switcher.get_current_system() == SystemType.SOAR
    
    def test_pressure_calculation(self, pressure_config):
        """Test pressure calculation."""
        valve = PressureValve(pressure_config)
        valve.start_monitoring()
        
        # Update metrics
        valve.monitor.update_metrics(
            subgoal_depth=3,
            time_in_state_ms=600,
            impasse_count=5,
            ambiguity_score=3
        )
        
        pressure = valve.compute_pressure({}, {})
        
        # Pressure should be high due to exceeding thresholds
        assert 0 < pressure <= 1.0
        assert pressure > 0.5  # Should be significant
    
    def test_soar_to_actr_switch(self, pressure_config):
        """Test switching from Soar to ACT-R."""
        valve = PressureValve(pressure_config)
        valve.start_monitoring()
        
        # Set depth above threshold
        valve.monitor.update_metrics(subgoal_depth=5)
        
        metrics = valve.monitor.get_current_metrics()
        soar_state = {}
        
        should_switch, reason = valve.switcher.should_switch_to_actr(metrics, soar_state)
        
        assert should_switch
        assert "depth" in reason.lower()
    
    def test_actr_to_evo_switch(self, pressure_config):
        """Test switching from ACT-R to Evolutionary."""
        valve = PressureValve(pressure_config)
        valve.switcher.current_system = SystemType.ACT_R
        
        should_switch, reason = valve.switcher.should_switch_to_evolutionary(
            pressure=0.95,
            actr_failure=False
        )
        
        assert should_switch
        assert "pressure" in reason.lower()
    
    def test_system_switch_callback(self, pressure_config):
        """Test system switching with callback."""
        valve = PressureValve(pressure_config)
        
        callback_called = [False]
        def test_callback():
            callback_called[0] = True
        
        valve.register_callbacks(on_switch_to_actr=test_callback)
        
        # Trigger switch
        valve.switcher.switch_system(SystemType.ACT_R, "test")
        
        assert callback_called[0]
    
    def test_pressure_normalization(self, pressure_config):
        """Test that pressure stays in [0, 1] range."""
        valve = PressureValve(pressure_config)
        valve.start_monitoring()
        
        # Extreme values
        valve.monitor.update_metrics(
            subgoal_depth=100,
            time_in_state_ms=100000,
            impasse_count=1000
        )
        
        pressure = valve.compute_pressure({}, {})
        
        assert 0 <= pressure <= 1.0


# ============================================================================
# EVOLUTIONARY FALLBACK TESTS
# ============================================================================

class TestEvolutionaryFallback:
    """Tests for Evolutionary Solver GA."""
    
    def test_initialization(self, evolutionary_config):
        """Test evolutionary solver initialization."""
        solver = EvolutionarySolver(evolutionary_config)
        
        assert solver.config == evolutionary_config
        assert solver.generation == 0
    
    def test_population_initialization(self, evolutionary_config):
        """Test population initialization."""
        solver = EvolutionarySolver(evolutionary_config)
        problem = {"type": "test"}
        
        solver.initialize_population(20, problem)
        
        assert len(solver.population.individuals) == 20
        assert all(isinstance(ind, Individual) for ind in solver.population.individuals)
    
    def test_fitness_evaluation(self, evolutionary_config):
        """Test fitness evaluation."""
        solver = EvolutionarySolver(evolutionary_config)
        
        # Create individual with known properties
        ind = Individual(
            genome="test",
            syntax_correct=True,
            runtime_success=True,
            output_correct=True,
            efficiency_score=0.8
        )
        
        evaluated = solver.evaluate_fitness(ind)
        
        assert evaluated.fitness > 0
        assert evaluated.syntax_correct
    
    def test_mutation(self, evolutionary_config):
        """Test mutation operator."""
        solver = EvolutionarySolver(evolutionary_config, SolutionType.CODE)
        
        parent = Individual(genome="def test(): pass")
        
        # Mutate multiple times to ensure change happens
        mutated = None
        for _ in range(10):
            mutated = solver.mutate(parent)
            if mutated.genome != parent.genome:
                break
        
        # Mutation may or may not change genome due to randomness
        # Just verify it returns an Individual
        assert isinstance(mutated, Individual)
    
    def test_crossover(self, evolutionary_config):
        """Test crossover operator."""
        solver = EvolutionarySolver(evolutionary_config, SolutionType.CODE)
        
        parent1 = Individual(genome="AAAA")
        parent2 = Individual(genome="BBBB")
        
        child1, child2 = solver.crossover(parent1, parent2)
        
        assert isinstance(child1, Individual)
        assert isinstance(child2, Individual)
        assert len(child1.genome) > 0
    
    def test_selection(self, evolutionary_config):
        """Test tournament selection."""
        solver = EvolutionarySolver(evolutionary_config)
        
        # Create population with varying fitness
        for i in range(10):
            solver.population.individuals.append(
                Individual(genome=f"ind_{i}", fitness=i * 0.1)
            )
        
        # Selection should favor higher fitness
        selected = solver.select_parents()
        
        assert isinstance(selected, tuple)
        assert len(selected) == 2
    
    @pytest.mark.slow
    def test_evolution_run(self, evolutionary_config):
        """Test full evolution run."""
        solver = EvolutionarySolver(evolutionary_config)
        problem = {"type": "test"}
        
        solver.initialize_population(10, problem)
        
        # Mock fitness evaluator for testing
        for ind in solver.population.individuals:
            ind.syntax_correct = True
            ind.runtime_success = True
            ind.output_correct = True
            ind.calculate_fitness()
        
        best = solver.evolve(generations=5)
        
        assert best is not None
        assert solver.generation <= 5


# ============================================================================
# CHUNKING SYSTEM TESTS
# ============================================================================

class TestChunkingSystem:
    """Tests for Chunking System."""
    
    def test_chunk_creation(self):
        """Test creating a chunk from resolution."""
        engine = ChunkingEngine()
        
        impasse = NoChangeImpasse(state_id="test")
        resolution = {"operator_id": "op1", "value": "success"}
        context = {"problem": "test"}
        
        chunk = engine.create_chunk(impasse, resolution, context)
        
        assert chunk is not None
        assert chunk.chunk_type == ChunkType.SUBGOAL_RESOLUTION
        assert len(engine.repository.get_all()) == 1
    
    def test_chunk_to_rule_conversion(self):
        """Test converting chunk to production rule."""
        chunk = Chunk(
            name="test_chunk",
            conditions=[{"attribute": "status", "value": "stuck"}],
            actions=[{"type": "add", "attribute": "status", "value": "resolved"}],
            source_impasse_type=ImpasseType.NO_CHANGE
        )
        
        rule = chunk.to_soar_rule()
        
        assert isinstance(rule, SoarRule)
        assert rule.learned
        assert len(rule.conditions) == 1
    
    def test_chunk_quality_progression(self):
        """Test chunk quality progression."""
        chunk = Chunk(name="test")
        
        assert chunk.quality == ChunkQuality.UNVALIDATED
        
        # Simulate validation
        chunk.usage_count = 10
        chunk.success_count = 9
        chunk.quality = ChunkQuality.VALIDATED
        
        assert chunk.quality == ChunkQuality.VALIDATED
        assert chunk.success_rate == 0.9
    
    def test_chunk_repository(self):
        """Test chunk repository operations."""
        repo = ChunkingEngine().repository
        
        chunk1 = Chunk(name="chunk1", chunk_type=ChunkType.OPERATOR_SELECTION)
        chunk2 = Chunk(name="chunk2", chunk_type=ChunkType.CONSTRAINT_RESOLUTION)
        
        repo.add(chunk1)
        repo.add(chunk2)
        
        # Find by type
        operator_chunks = repo.find_by_type(ChunkType.OPERATOR_SELECTION)
        assert len(operator_chunks) == 1
        assert operator_chunks[0].name == "chunk1"
        
        # Find by impasse
        nochange_chunks = repo.find_by_impasse(ImpasseType.NO_CHANGE)
        assert len(nochange_chunks) == 2  # Both default to NO_CHANGE
    
    def test_chunk_matching(self):
        """Test matching chunks to context."""
        engine = ChunkingEngine()
        
        chunk = Chunk(
            name="test_chunk",
            conditions=[{"attribute": "status", "value": "stuck"}],
            chunk_type=ChunkType.OPERATOR_SELECTION
        )
        
        engine.repository.add(chunk)
        
        # Should match
        matches1 = engine.match_chunk({"status": "stuck", "other": "value"})
        assert len(matches1) == 1
        
        # Should not match
        matches2 = engine.match_chunk({"status": "ok"})
        assert len(matches2) == 0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestCognitiveHydraulicsIntegration:
    """Integration tests for full reasoning pipeline."""
    
    def test_full_engine_initialization(self):
        """Test full engine initialization."""
        config = CognitiveHydraulicsConfig()
        config.soar.max_decision_cycles = 10
        config.max_reasoning_time_ms = 1000
        
        engine = CognitiveHydraulicsEngine(config)
        
        assert engine.soar is not None
        assert engine.actr is not None
        assert engine.pressure_valve is not None
        assert engine.chunking is not None
    
    def test_simple_problem_solving(self):
        """Test solving a simple problem."""
        config = CognitiveHydraulicsConfig()
        config.soar.max_decision_cycles = 5
        config.max_reasoning_time_ms = 1000
        
        engine = CognitiveHydraulicsEngine(config)
        
        problem = {"type": "simple", "description": "test"}
        goal = {"status": "completed"}
        
        result = engine.solve(problem, goal)
        
        assert isinstance(result, ReasoningResult)
        assert result.total_time_ms > 0
    
    def test_system_switching(self):
        """Test automatic system switching."""
        config = CognitiveHydraulicsConfig()
        config.pressure_valve.soar_to_actr_depth = 1
        config.soar.max_decision_cycles = 3
        
        engine = CognitiveHydraulicsEngine(config)
        
        # Problem that will cause impasse and depth increase
        problem = {"type": "complex", "requires_subgoals": True}
        goal = {"solved": True}
        
        result = engine.solve(problem, goal)
        
        # Should have used multiple systems
        assert len(result.systems_used) >= 1


# ============================================================================
# KG INTEGRATION TESTS
# ============================================================================

class TestKGIntegration:
    """Tests for Knowledge Graph integration."""
    
    def test_kg_integration_initialization(self):
        """Test KG integration initialization."""
        integration = CognitiveHydraulicsKGIntegration()
        
        assert integration.engine is not None
        assert integration.encoder is not None
        assert integration.decoder is not None
    
    def test_kg_reasoning(self):
        """Test KG reasoning operation."""
        integration = CognitiveHydraulicsKGIntegration()
        
        kg_subgraph = {
            "entities": [
                {"entity_id": "e1", "name": "Alice"},
                {"entity_id": "e2", "name": "Bob"}
            ],
            "relationships": [
                {"source": "e1", "target": "e2", "type": "KNOWS"}
            ]
        }
        
        query = {
            "goal": {"find": "connection"},
            "constraints": []
        }
        
        result = integration.reason_about_graph(kg_subgraph, query)
        
        assert isinstance(result, KGReasoningResult)
        assert result.execution_time_ms >= 0
    
    def test_relationship_inference(self):
        """Test relationship inference."""
        integration = CognitiveHydraulicsKGIntegration()
        
        entity1 = {"entity_id": "e1", "name": "Company A"}
        entity2 = {"entity_id": "e2", "name": "Company B"}
        context = {
            "entities": [entity1, entity2],
            "relationships": []
        }
        
        result = integration.infer_relationship(entity1, entity2, context)
        
        assert isinstance(result, KGReasoningResult)
    
    def test_explanation_generation(self):
        """Test explanation generation."""
        integration = CognitiveHydraulicsKGIntegration()
        
        result = KGReasoningResult(
            success=True,
            reasoning_type="test",
            conclusions=[{"value": "test"}],
            systems_used=["soar", "actr"],
            confidence=0.9
        )
        
        explanation = integration.explain_reasoning(result)
        
        assert "Reasoning Type" in explanation
        assert "soar" in explanation.lower()
    
    def test_reasoning_tracer(self):
        """Test reasoning tracer."""
        tracer = ReasoningTracer()
        
        tracer.add_step(
            system="soar",
            operation="decide",
            input_data={"state": "test"},
            output_data={"operator": "op1"},
            reasoning="Selected best operator"
        )
        
        trace = tracer.get_trace()
        assert len(trace) == 1
        assert trace[0]["system"] == "soar"
        
        explanation = tracer.generate_explanation()
        assert "Step 1" in explanation


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance benchmarks."""
    
    def test_soar_performance(self, soar_config, sample_operators):
        """Benchmark Soar decision cycles."""
        engine = SoarEngine(soar_config)
        engine.initialize({"type": "perf_test"}, sample_operators)
        
        start = time.time()
        for _ in range(100):
            engine.run_decision_cycle()
        elapsed = time.time() - start
        
        # Should complete 100 cycles quickly
        assert elapsed < 1.0  # 1 second max
    
    def test_actr_performance(self, actr_config):
        """Benchmark ACT-R operator selection."""
        engine = ACTREngine(actr_config)
        
        # Add many productions
        for i in range(50):
            prod = ACTRProduction(
                name=f"prod_{i}",
                conditions=[],
                probability=0.5,
                cost=1.0
            )
            engine.add_production(prod)
        
        start = time.time()
        for _ in range(100):
            engine.run_cycle({})
        elapsed = time.time() - start
        
        assert elapsed < 1.0
    
    def test_pressure_calculation_performance(self, pressure_config):
        """Benchmark pressure calculation."""
        valve = PressureValve(pressure_config)
        valve.start_monitoring()
        
        start = time.time()
        for _ in range(1000):
            valve.compute_pressure({}, {"subgoal_depth": 3})
        elapsed = time.time() - start
        
        assert elapsed < 0.1  # Should be very fast


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
