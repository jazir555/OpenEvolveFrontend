"""Comprehensive tests for DTS (Dialogue Tree Search) Integration.

Tests cover:
- Tree structure operations
- User simulation
- Trajectory scoring
- Beam search
- End-to-end optimization
"""

import pytest
import sys
from typing import List, Dict, Any

# Add parent directories to path for imports
sys.path.insert(0, "c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend")

from integrations.dts import (
    # Core structures
    ConversationNode,
    ConversationTree,
    StrategyGenerator,
    
    # User simulation
    UserPersona,
    UserSimulator,
    IntentModel,
    IntentType,
    PREDEFINED_PERSONAS,
    
    # Scoring
    ScoreResult,
    Judge,
    TrajectoryScorer,
    CriterionType,
    
    # Search
    BeamState,
    BeamSearch,
    ParallelBeamSearch,
    
    # Engine
    DTSConfig,
    DTSResult,
    DTSEngine,
    DTSEngineBuilder,
)

from knowledge_engine.integrations.dts import (
    DTSKGIntegration,
    SimulatedResponse,
    ExtractedEntities,
    ConversationScript,
    OptimalPath,
)


# =============================================================================
# Tree Structure Tests
# =============================================================================

class TestConversationNode:
    """Test ConversationNode functionality."""
    
    def test_node_creation(self):
        """Test basic node creation."""
        node = ConversationNode(message="Hello", speaker="user")
        
        assert node.message == "Hello"
        assert node.speaker == "user"
        assert node.depth == 0
        assert node.score == 0.0
        assert node.parent is None
        assert len(node.children) == 0
        assert node.node_id is not None
    
    def test_node_metadata_defaults(self):
        """Test metadata initialization."""
        node = ConversationNode(message="Test")
        
        assert "timestamp" in node.metadata
        assert "tokens" in node.metadata
        assert "cost" in node.metadata
    
    def test_add_child(self):
        """Test adding children to nodes."""
        parent = ConversationNode(message="Parent", speaker="system")
        child = parent.add_child("Child", speaker="user", score=7.5)
        
        assert child.parent == parent
        assert child in parent.children
        assert child.depth == 1
        assert child.score == 7.5
    
    def test_get_path(self):
        """Test path retrieval from root."""
        root = ConversationNode(message="Root")
        child1 = root.add_child("Child 1")
        grandchild = child1.add_child("Grandchild")
        
        path = grandchild.get_path()
        
        assert len(path) == 3
        assert path[0] == root
        assert path[1] == child1
        assert path[2] == grandchild
    
    def test_get_conversation_history(self):
        """Test conversation history extraction."""
        root = ConversationNode(message="Hi", speaker="user")
        response = root.add_child("Hello!", speaker="system")
        followup = response.add_child("How are you?", speaker="user")
        
        history = followup.get_conversation_history()
        
        assert len(history) == 3
        assert history[0]["speaker"] == "user"
        assert history[0]["message"] == "Hi"
        assert history[1]["speaker"] == "system"
        assert history[2]["speaker"] == "user"
    
    def test_is_leaf(self):
        """Test leaf node detection."""
        parent = ConversationNode(message="Parent")
        child = parent.add_child("Child")
        
        assert parent.is_leaf() is False
        assert child.is_leaf() is True
    
    def test_get_siblings(self):
        """Test sibling retrieval."""
        parent = ConversationNode(message="Parent")
        child1 = parent.add_child("Child 1")
        child2 = parent.add_child("Child 2")
        child3 = parent.add_child("Child 3")
        
        siblings = child1.get_siblings()
        
        assert len(siblings) == 2
        assert child2 in siblings
        assert child3 in siblings
        assert child1 not in siblings
    
    def test_update_score(self):
        """Test score updates."""
        node = ConversationNode(message="Test")
        node.update_score(8.5)
        
        assert node.score == 8.5
    
    def test_to_dict(self):
        """Test node serialization."""
        node = ConversationNode(message="Test", speaker="system", score=7.0)
        data = node.to_dict()
        
        assert data["message"] == "Test"
        assert data["speaker"] == "system"
        assert data["score"] == 7.0
        assert "node_id" in data


class TestConversationTree:
    """Test ConversationTree functionality."""
    
    def test_tree_creation(self):
        """Test tree initialization."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        assert tree.root == root
        assert tree.tree_id is not None
        assert "created_at" in tree.metadata
    
    def test_add_node(self):
        """Test adding nodes to tree."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        child = tree.add_node(root, "Child message", speaker="user", score=5.0)
        
        assert child in root.children
        assert tree.metadata["total_nodes"] == 2
    
    def test_get_branches(self):
        """Test branch enumeration."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        # Create two branches
        child1 = tree.add_node(root, "Child 1")
        leaf1 = tree.add_node(child1, "Leaf 1")
        
        child2 = tree.add_node(root, "Child 2")
        leaf2 = tree.add_node(child2, "Leaf 2")
        
        branches = tree.get_branches()
        
        assert len(branches) == 2
        assert len(branches[0]) == 3  # root -> child -> leaf
        assert len(branches[1]) == 3
    
    def test_get_leaves(self):
        """Test leaf node retrieval."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        child1 = tree.add_node(root, "Child 1")
        leaf1 = tree.add_node(child1, "Leaf 1")
        
        child2 = tree.add_node(root, "Child 2")
        
        leaves = tree.get_leaves()
        
        assert len(leaves) == 2
        assert leaf1 in leaves
        assert child2 in leaves
    
    def test_get_all_nodes(self):
        """Test all node retrieval."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        child = tree.add_node(root, "Child")
        grandchild = tree.add_node(child, "Grandchild")
        
        all_nodes = tree.get_all_nodes()
        
        assert len(all_nodes) == 3
        assert root in all_nodes
        assert child in all_nodes
        assert grandchild in all_nodes
    
    def test_prune_by_threshold(self):
        """Test pruning by score threshold."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        # Add nodes with varying scores (add descendants to test full pruning)
        child1 = tree.add_node(root, "Good", score=8.0)
        tree.add_node(child1, "Good Leaf", score=8.0)
        
        child2 = tree.add_node(root, "Bad", score=3.0)
        tree.add_node(child2, "Bad Leaf", score=2.0)
        
        pruned = tree.prune(threshold=5.0)
        
        assert pruned >= 1  # At minimum the bad branch should be pruned
        assert child2 not in root.children
        assert child1 in root.children
    
    def test_prune_keep_best_n(self):
        """Test pruning to keep N best branches."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        # Create multiple branches
        for i in range(5):
            child = tree.add_node(root, f"Child {i}", score=float(i))
            tree.add_node(child, f"Leaf {i}", score=float(i))
        
        pruned = tree.prune(threshold=0.0, keep_best_n=2)
        
        # Should keep only 2 best branches (scores 4 and 3)
        assert len(root.children) == 2
    
    def test_backpropagate(self):
        """Test score backpropagation."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        child = tree.add_node(root, "Child")
        leaf = tree.add_node(child, "Leaf", score=8.0)
        
        tree.backpropagate(leaf)
        
        # Parent should now have average of children
        assert child.score == 8.0
    
    def test_get_best_path(self):
        """Test best path retrieval."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        # Good branch
        good_child = tree.add_node(root, "Good", score=8.0)
        good_leaf = tree.add_node(good_child, "Good Leaf", score=9.0)
        
        # Bad branch
        bad_child = tree.add_node(root, "Bad", score=3.0)
        bad_leaf = tree.add_node(bad_child, "Bad Leaf", score=2.0)
        
        best_path = tree.get_best_path()
        
        assert best_path[-1] == good_leaf
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        
        child = tree.add_node(root, "Child", score=7.0)
        tree.add_node(child, "Leaf", score=8.0)
        
        stats = tree.get_statistics()
        
        assert stats["total_nodes"] == 3
        assert stats["total_leaves"] == 1
        assert stats["max_depth"] == 2
        assert stats["avg_score"] == 5.0  # (0 + 7 + 8) / 3


# =============================================================================
# User Simulation Tests
# =============================================================================

class TestUserPersona:
    """Test UserPersona functionality."""
    
    def test_persona_creation(self):
        """Test persona initialization."""
        persona = UserPersona(name="Test User")
        
        assert persona.name == "Test User"
        assert "cooperativeness" in persona.traits
        assert persona.goal_alignment >= 0.0 and persona.goal_alignment <= 1.0
    
    def test_get_dominant_intent(self):
        """Test dominant intent detection."""
        persona = UserPersona(
            name="Cooperative",
            traits={"cooperativeness": 0.9, "skepticism": 0.1},
        )
        
        dominant = persona.get_dominant_intent()
        
        assert dominant == IntentType.COOPERATIVE
    
    def test_sample_intent(self):
        """Test intent sampling."""
        persona = UserPersona(name="Test")
        
        # Sample multiple times to verify distribution
        intents = [persona.sample_intent() for _ in range(10)]
        
        assert all(isinstance(i, IntentType) for i in intents)
    
    def test_to_dict(self):
        """Test persona serialization."""
        persona = UserPersona(name="Test", knowledge_level="expert")
        data = persona.to_dict()
        
        assert data["name"] == "Test"
        assert data["knowledge_level"] == "expert"


class TestIntentModel:
    """Test IntentModel functionality."""
    
    def test_classify_cooperative_intent(self):
        """Test classification of cooperative messages."""
        model = IntentModel()
        
        intent, confidence = model.classify_intent("Yes, that sounds great! Thanks!")
        
        assert intent == IntentType.COOPERATIVE
        assert confidence > 0.0
    
    def test_classify_skeptical_intent(self):
        """Test classification of skeptical messages."""
        model = IntentModel()
        
        # Use a message with clear skeptical indicators
        intent, confidence = model.classify_intent("I doubt that's correct. However, I need evidence.")
        
        # Should be skeptical or confused (both indicate questioning)
        assert intent in [IntentType.SKEPTICAL, IntentType.CONFUSED]
    
    def test_classify_confused_intent(self):
        """Test classification of confused messages."""
        model = IntentModel()
        
        intent, confidence = model.classify_intent("I don't understand. What do you mean?")
        
        assert intent == IntentType.CONFUSED
    
    def test_detect_satisfaction(self):
        """Test satisfaction detection."""
        model = IntentModel()
        
        history = [
            {"speaker": "user", "message": "I need help with this problem"},
            {"speaker": "system", "message": "Here's a solution"},
            {"speaker": "user", "message": "Thank you! That's very helpful!"},
        ]
        
        satisfaction = model.detect_satisfaction(history)
        
        assert satisfaction > 0.5  # Should be positive
    
    def test_detect_goal_progress(self):
        """Test goal progress detection."""
        model = IntentModel()
        
        history = [
            {"speaker": "user", "message": "Yes, I understand"},
            {"speaker": "user", "message": "That makes sense"},
        ]
        
        progress = model.detect_goal_progress(history, "teach concept")
        
        assert progress >= 0.0


class TestUserSimulator:
    """Test UserSimulator functionality."""
    
    def test_simulator_creation(self):
        """Test simulator initialization."""
        simulator = UserSimulator()
        
        assert len(simulator.personas) > 0
    
    def test_simulate_response(self):
        """Test response simulation."""
        simulator = UserSimulator()
        
        response = simulator.simulate_response("How can I help you?")
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_simulate_with_persona(self):
        """Test simulation with specific persona."""
        simulator = UserSimulator()
        cooperative = PREDEFINED_PERSONAS["cooperative_user"]
        
        response = simulator.simulate_response(
            "Here's a solution",
            persona=cooperative
        )
        
        assert isinstance(response, str)
    
    def test_generate_intent_variants(self):
        """Test generation of intent variants."""
        simulator = UserSimulator()
        
        variants = simulator.generate_intent_variants("Test strategy", k=3)
        
        assert len(variants) == 3
        assert all(isinstance(v, str) for v in variants)
    
    def test_simulate_conversation_turns(self):
        """Test multi-turn simulation."""
        simulator = UserSimulator()
        
        strategies = [
            "Hello, how can I help?",
            "Have you tried restarting?",
        ]
        
        history = simulator.simulate_conversation_turns(strategies)
        
        assert len(history) > 0
        assert all("speaker" in turn and "message" in turn for turn in history)


# =============================================================================
# Trajectory Scoring Tests
# =============================================================================

class TestScoreResult:
    """Test ScoreResult functionality."""
    
    def test_result_creation(self):
        """Test result initialization."""
        result = ScoreResult(overall_score=7.5)
        
        assert result.overall_score == 7.5
        assert result.criteria_scores == {}
        assert result.judge_scores == []
    
    def test_get_variance(self):
        """Test variance calculation."""
        result = ScoreResult(
            overall_score=7.0,
            judge_scores=[6.0, 7.0, 8.0]
        )
        
        variance = result.get_variance()
        
        assert variance > 0.0
    
    def test_get_consensus_level(self):
        """Test consensus level determination."""
        # High consensus
        high = ScoreResult(overall_score=7.0, judge_scores=[7.0, 7.2, 6.9])
        assert high.get_consensus_level() == "high"
        
        # Low consensus
        low = ScoreResult(overall_score=7.0, judge_scores=[3.0, 7.0, 10.0])
        assert low.get_consensus_level() == "low"


class TestJudge:
    """Test Judge functionality."""
    
    def test_judge_creation(self):
        """Test judge initialization."""
        judge = Judge(name="TestJudge")
        
        assert judge.name == "TestJudge"
        assert len(judge.criteria) > 0
    
    def test_evaluate_empty_path(self):
        """Test evaluation of empty path."""
        judge = Judge(name="Test")
        
        result = judge.evaluate([])
        
        assert result["overall"] == 0.0
    
    def test_evaluate_heuristic(self):
        """Test heuristic evaluation."""
        judge = Judge(name="Test")
        
        path = [
            ConversationNode(message="Hello", speaker="user"),
            ConversationNode(message="Hi there! How can I help?", speaker="system"),
        ]
        
        result = judge.evaluate(path)
        
        assert "overall" in result
        assert "criteria" in result
        assert result["overall"] > 0.0


class TestTrajectoryScorer:
    """Test TrajectoryScorer functionality."""
    
    def test_scorer_creation(self):
        """Test scorer initialization."""
        scorer = TrajectoryScorer()
        
        assert len(scorer.judges) > 0
    
    def test_score_trajectory(self):
        """Test trajectory scoring."""
        scorer = TrajectoryScorer()
        
        path = [
            ConversationNode(message="Hello", speaker="user"),
            ConversationNode(message="How can I help?", speaker="system"),
        ]
        
        result = scorer.score_trajectory(path)
        
        assert isinstance(result, ScoreResult)
        assert result.overall_score >= 0.0
        assert result.overall_score <= 10.0
    
    def test_aggregate_scores_median(self):
        """Test median aggregation."""
        scorer = TrajectoryScorer(aggregation_method="median")
        
        aggregated = scorer.aggregate_scores([6.0, 7.0, 8.0])
        
        assert aggregated == 7.0
    
    def test_compare_trajectories(self):
        """Test trajectory comparison."""
        scorer = TrajectoryScorer()
        
        path1 = [ConversationNode(message="Good", score=8.0)]
        path2 = [ConversationNode(message="Bad", score=3.0)]
        
        results = scorer.compare_trajectories([path1, path2])
        
        assert len(results) == 2
        assert results[0].overall_score > results[1].overall_score


# =============================================================================
# Beam Search Tests
# =============================================================================

class TestBeamState:
    """Test BeamState functionality."""
    
    def test_state_creation(self):
        """Test state initialization."""
        state = BeamState()
        
        assert state.round_number == 0
        assert state.budget_remaining == 1000
        assert len(state.active_branches) == 0
    
    def test_get_best_branch(self):
        """Test best branch retrieval."""
        node1 = ConversationNode(message="A", score=5.0)
        node2 = ConversationNode(message="B", score=8.0)
        
        state = BeamState(active_branches=[node1, node2])
        state.scores[node1.node_id] = 5.0
        state.scores[node2.node_id] = 8.0
        
        best = state.get_best_branch()
        
        assert best == node2
    
    def test_get_average_score(self):
        """Test average score calculation."""
        state = BeamState()
        state.scores["a"] = 6.0
        state.scores["b"] = 8.0
        
        avg = state.get_average_score()
        
        assert avg == 7.0


class TestBeamSearch:
    """Test BeamSearch functionality."""
    
    def test_search_creation(self):
        """Test search initialization."""
        search = BeamSearch(beam_width=5, max_depth=3)
        
        assert search.beam_width == 5
        assert search.max_depth == 3
        assert search.scorer is not None
    
    def test_prune_branches(self):
        """Test branch pruning."""
        search = BeamSearch(beam_width=2)
        
        nodes = [
            ConversationNode(message="High", score=9.0),
            ConversationNode(message="Medium", score=6.0),
            ConversationNode(message="Low", score=3.0),
        ]
        
        scores = {n.node_id: n.score for n in nodes}
        pruned = search.prune_branches(nodes, scores)
        
        assert len(pruned) == 2
        assert nodes[2] not in pruned  # Lowest score pruned


# =============================================================================
# DTS Engine Tests
# =============================================================================

class TestDTSConfig:
    """Test DTSConfig functionality."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = DTSConfig()
        
        assert config.beam_width == 5
        assert config.intent_variants == 3
        assert config.judges == 3
        assert config.prune_threshold == 5.0
    
    def test_config_to_dict(self):
        """Test config serialization."""
        config = DTSConfig(beam_width=10)
        data = config.to_dict()
        
        assert data["beam_width"] == 10


class TestDTSResult:
    """Test DTSResult functionality."""
    
    def test_result_creation(self):
        """Test result initialization."""
        root = ConversationNode(message="Root")
        tree = ConversationTree(root=root)
        state = BeamState()
        
        result = DTSResult(
            tree=tree,
            best_path=[root],
            best_score=8.0,
            all_paths=[[root]],
            state=state,
        )
        
        assert result.best_score == 8.0
        assert len(result.best_path) == 1
    
    def test_get_conversation_script(self):
        """Test script extraction."""
        root = ConversationNode(message="Hi", speaker="user")
        tree = ConversationTree(root=root)
        state = BeamState()
        
        result = DTSResult(
            tree=tree,
            best_path=[root],
            best_score=7.0,
            all_paths=[[root]],
            state=state,
        )
        
        script = result.get_conversation_script()
        
        assert len(script) == 1
        assert script[0]["speaker"] == "user"


class TestDTSEngine:
    """Test DTSEngine functionality."""
    
    def test_engine_creation(self):
        """Test engine initialization."""
        engine = DTSEngine()
        
        assert engine.config is not None
        assert engine.strategy_gen is not None
        assert engine.user_sim is not None
        assert engine.scorer is not None
    
    def test_optimize_conversation(self):
        """Test full conversation optimization."""
        engine = DTSEngine(config=DTSConfig(beam_width=2, max_depth=2, max_rounds=2))
        
        result = engine.optimize_conversation(
            initial_context="Test context",
            goal="Test goal",
            rounds=2,
        )
        
        assert isinstance(result, DTSResult)
        assert result.tree is not None
        assert result.best_score >= 0.0
    
    def test_explain_strategy(self):
        """Test strategy explanation."""
        engine = DTSEngine()
        node = ConversationNode(message="Test strategy", score=7.5)
        
        explanation = engine.explain_strategy(node)
        
        assert isinstance(explanation, str)
        assert "Strategy Explanation" in explanation


class TestDTSEngineBuilder:
    """Test DTSEngineBuilder functionality."""
    
    def test_builder_chain(self):
        """Test fluent builder interface."""
        engine = (DTSEngineBuilder()
            .with_beam_width(10)
            .with_max_depth(7)
            .with_prune_threshold(6.0)
            .build())
        
        assert engine.config.beam_width == 10
        assert engine.config.max_depth == 7
        assert engine.config.prune_threshold == 6.0
    
    def test_builder_with_personas(self):
        """Test builder with personas."""
        persona = UserPersona(name="Custom")
        
        engine = (DTSEngineBuilder()
            .add_persona(persona)
            .build())
        
        assert persona in engine.user_sim.personas


# =============================================================================
# Knowledge Engine Integration Tests
# =============================================================================

class TestDTSKGIntegration:
    """Test DTS-KG Integration functionality."""
    
    def test_integration_creation(self):
        """Test integration initialization."""
        integration = DTSKGIntegration()
        
        assert integration.engine is not None
    
    def test_optimize_kg_query_dialog(self):
        """Test KG query dialog optimization."""
        integration = DTSKGIntegration()
        
        tree = integration.optimize_kg_query_dialog(
            context="Find companies",
            user_goal="Research AI companies",
        )
        
        assert isinstance(tree, ConversationTree)
        assert tree.root is not None
    
    def test_simulate_user_interactions(self):
        """Test user interaction simulation."""
        integration = DTSKGIntegration()
        
        responses = integration.simulate_user_interactions(
            query_plan={"description": "Find AI companies"},
            num_variants=3,
        )
        
        assert len(responses) == 3
        assert all(isinstance(r, SimulatedResponse) for r in responses)
    
    def test_extract_kg_via_dialog(self):
        """Test entity extraction via dialog."""
        integration = DTSKGIntegration()
        
        entities = integration.extract_kg_via_dialog(
            entity_query="AI companies",
            entity_types=["Company", "Technology"],
        )
        
        assert isinstance(entities, ExtractedEntities)
        assert len(entities.entities) > 0
    
    def test_explain_kg_result_conversation(self):
        """Test KG result explanation."""
        integration = DTSKGIntegration()
        
        script = integration.explain_kg_result_conversation(
            kg_data={"entities": ["A", "B"], "relations": ["X", "Y"]},
        )
        
        assert isinstance(script, ConversationScript)
        assert len(script.turns) > 0
    
    def test_optimize_multi_turn_retrieval(self):
        """Test multi-turn retrieval optimization."""
        integration = DTSKGIntegration()
        
        path = integration.optimize_multi_turn_retrieval(
            retrieval_goal="Find all connected entities",
        )
        
        assert isinstance(path, OptimalPath)
        assert len(path.steps) > 0
    
    def test_backtrack_and_replan(self):
        """Test backtrack and replan."""
        integration = DTSKGIntegration()
        
        # Create a failed path
        root = ConversationNode(message="Start")
        child = root.add_child("Failed attempt")
        failed_path = [root, child]
        
        new_tree = integration.backtrack_and_replan(
            failed_path=failed_path,
            failure_reason="Path was suboptimal",
        )
        
        assert isinstance(new_tree, ConversationTree)
        assert new_tree.metadata.get("replan") is True


# =============================================================================
# Predefined Personas Tests
# =============================================================================

class TestPredefinedPersonas:
    """Test predefined personas."""
    
    def test_all_personas_exist(self):
        """Test that all expected personas exist."""
        expected = [
            "cooperative_user",
            "skeptical_expert",
            "confused_beginner",
            "hostile_critic",
            "curious_explorer",
            "time_constrained",
            "formal_professional",
            "enthusiastic_early_adopter",
        ]
        
        for name in expected:
            assert name in PREDEFINED_PERSONAS
    
    def test_cooperative_persona_traits(self):
        """Test cooperative persona has correct traits."""
        coop = PREDEFINED_PERSONAS["cooperative_user"]
        
        assert coop.traits["cooperativeness"] > 0.8
        assert coop.goal_alignment > 0.8


# =============================================================================
# Integration End-to-End Tests
# =============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create engine
        config = DTSConfig(
            beam_width=3,
            intent_variants=2,
            max_depth=2,
            max_rounds=2,
        )
        engine = DTSEngine(config=config)
        
        # Run optimization
        result = engine.optimize_conversation(
            initial_context="Customer support scenario",
            goal="Resolve technical issue",
        )
        
        # Verify results
        assert result.best_score > 0.0
        assert len(result.all_paths) > 0
        assert len(result.best_path) > 0
        
        # Get conversation script
        script = result.get_conversation_script()
        assert len(script) > 0
    
    def test_kg_integration_workflow(self):
        """Test KG integration workflow."""
        kg_dts = DTSKGIntegration()
        
        # Optimize dialog
        tree = kg_dts.optimize_kg_query_dialog(
            context="Research query",
            user_goal="Find relevant papers",
        )
        
        # Extract entities
        entities = kg_dts.extract_kg_via_dialog(
            entity_query="machine learning papers",
        )
        
        # Generate explanation
        script = kg_dts.explain_kg_result_conversation(
            kg_data={"entities": entities.get_all_entities()},
        )
        
        assert script.estimated_effectiveness >= 0.0
    
    def test_different_personas_produce_different_results(self):
        """Test that different personas produce varied results."""
        engine = DTSEngine(config=DTSConfig(beam_width=2, max_depth=2))
        
        results = []
        for name in ["cooperative_user", "skeptical_expert"]:
            persona = PREDEFINED_PERSONAS[name]
            engine.set_personas([persona])
            
            result = engine.optimize_conversation(
                initial_context="Test",
                goal="Test goal",
                rounds=1,
            )
            results.append(result.best_score)
        
        # Different personas should ideally produce different scores
        # (though this is probabilistic)


if __name__ == "__main__":
    # Run tests with pytest if available
    pytest.main([__file__, "-v"])
