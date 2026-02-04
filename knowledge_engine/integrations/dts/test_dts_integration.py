"""
Comprehensive Tests for DTS (Dialogue Tree Search) Integration

Tests cover:
- Conversation tree structure
- User simulation
- Trajectory scoring
- Beam search
- DTS engine
- KG integration

Author: OpenEvolve
Date: 2026-02-03
"""

import pytest
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))


class TestConversationTree:
    """Test suite for conversation tree structure."""
    
    @pytest.fixture
    def sample_conversation(self):
        """Sample conversation tree for testing."""
        try:
            from integrations.dts.conversation_tree import ConversationNode, ConversationTree
            
            tree = ConversationTree()
            root = ConversationNode(
                message="Hello, how can I help you?",
                speaker="system",
                depth=0
            )
            tree.root = root
            
            # Add child nodes
            child1 = ConversationNode(
                message="I need help with my order",
                speaker="user",
                depth=1,
                parent=root
            )
            child2 = ConversationNode(
                message="What's your return policy?",
                speaker="user",
                depth=1,
                parent=root
            )
            root.children = [child1, child2]
            
            return tree
        except ImportError:
            pytest.skip("DTS not available")
    
    def test_tree_creation(self, sample_conversation):
        """Test conversation tree creation."""
        tree = sample_conversation
        assert tree.root is not None
        assert tree.root.message == "Hello, how can I help you?"
        assert len(tree.root.children) == 2
    
    def test_node_structure(self, sample_conversation):
        """Test node structure and relationships."""
        tree = sample_conversation
        root = tree.root
        
        assert root.speaker == "system"
        assert root.depth == 0
        assert root.parent is None
        
        child = root.children[0]
        assert child.speaker == "user"
        assert child.depth == 1
        assert child.parent == root
    
    def test_path_traversal(self, sample_conversation):
        """Test path traversal from node to root."""
        tree = sample_conversation
        child = tree.root.children[0]
        
        path = child.get_path_to_root()
        assert len(path) == 2
        assert path[0] == child
        assert path[1] == tree.root
    
    def test_score_backpropagation(self, sample_conversation):
        """Test score backpropagation up the tree."""
        tree = sample_conversation
        child = tree.root.children[0]
        child.score = 8.5
        
        tree.backpropagate_scores(child)
        # Root score should be updated based on child
        assert tree.root.score > 0


class TestUserSimulator:
    """Test suite for user simulation."""
    
    @pytest.fixture
    def simulator(self):
        """Create user simulator."""
        try:
            from integrations.dts.user_simulator import UserSimulator
            return UserSimulator()
        except ImportError:
            pytest.skip("DTS not available")
    
    def test_persona_creation(self, simulator):
        """Test user persona creation."""
        from integrations.dts.user_simulator import UserPersona
        
        persona = UserPersona(
            name="cooperative_user",
            traits={"cooperative": 0.9, "patient": 0.8}
        )
        
        assert persona.name == "cooperative_user"
        assert persona.traits["cooperative"] == 0.9
    
    def test_response_simulation(self, simulator):
        """Test user response simulation."""
        response = simulator.simulate_response(
            strategy="Helpful greeting",
            persona_name="cooperative_user",
            context={"topic": "customer_support"}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_intent_classification(self, simulator):
        """Test intent classification."""
        from integrations.dts.user_simulator import IntentType
        
        intent = simulator.classify_intent("I want to buy a product")
        assert isinstance(intent, IntentType)
    
    def test_satisfaction_detection(self, simulator):
        """Test satisfaction detection."""
        history = [
            {"speaker": "user", "message": "I have a problem"},
            {"speaker": "system", "message": "I'll help you"},
            {"speaker": "user", "message": "Thanks, that's solved"}
        ]
        
        satisfaction = simulator.detect_satisfaction(history)
        assert 0 <= satisfaction <= 1


class TestTrajectoryScorer:
    """Test suite for trajectory scoring."""
    
    @pytest.fixture
    def scorer(self):
        """Create trajectory scorer."""
        try:
            from integrations.dts.trajectory_scorer import TrajectoryScorer
            return TrajectoryScorer()
        except ImportError:
            pytest.skip("DTS not available")
    
    def test_judge_creation(self, scorer):
        """Test judge creation."""
        from integrations.dts.trajectory_scorer import Judge
        
        judge = Judge(name="coherence_judge", criteria=["coherence"])
        assert judge.name == "coherence_judge"
        assert "coherence" in judge.criteria
    
    def test_trajectory_scoring(self, scorer):
        """Test trajectory scoring."""
        from integrations.dts.conversation_tree import ConversationNode
        
        path = [
            ConversationNode(message="Hello", speaker="system", depth=0, score=0),
            ConversationNode(message="Hi", speaker="user", depth=1, score=0),
            ConversationNode(message="How can I help?", speaker="system", depth=2, score=0)
        ]
        
        result = scorer.score_trajectory(path)
        assert result.overall_score >= 0
        assert result.overall_score <= 10
    
    def test_multi_judge_consensus(self, scorer):
        """Test multi-judge consensus scoring."""
        scores = [7.5, 8.0, 7.0]
        consensus = scorer.aggregate_scores(scores, method='median')
        
        assert consensus == 7.5  # Median of [7.0, 7.5, 8.0]


class TestBeamSearch:
    """Test suite for beam search."""
    
    @pytest.fixture
    def beam_search(self):
        """Create beam search instance."""
        try:
            from integrations.dts.beam_search import BeamSearch
            return BeamSearch(beam_width=3, max_depth=5)
        except ImportError:
            pytest.skip("DTS not available")
    
    def test_beam_search_initialization(self, beam_search):
        """Test beam search initialization."""
        assert beam_search.beam_width == 3
        assert beam_search.max_depth == 5
    
    def test_node_expansion(self, beam_search):
        """Test node expansion."""
        from integrations.dts.conversation_tree import ConversationNode
        
        node = ConversationNode(message="Hello", speaker="system", depth=0)
        expanded = beam_search.expand_node(node, num_children=2)
        
        assert len(expanded) == 2
        assert all(child.parent == node for child in expanded)
    
    def test_branch_pruning(self, beam_search):
        """Test branch pruning."""
        from integrations.dts.conversation_tree import ConversationNode
        
        nodes = [
            ConversationNode(message="A", speaker="user", depth=1, score=9.0),
            ConversationNode(message="B", speaker="user", depth=1, score=5.0),
            ConversationNode(message="C", speaker="user", depth=1, score=7.0),
        ]
        
        pruned = beam_search.prune_branches(nodes, keep_top_n=2)
        assert len(pruned) == 2
        assert all(node.score >= 7.0 for node in pruned)


class TestDTSEngine:
    """Test suite for DTS engine."""
    
    @pytest.fixture
    def engine(self):
        """Create DTS engine."""
        try:
            from integrations.dts.dts_engine import DTSEngine, DTSConfig
            config = DTSConfig(beam_width=3, max_rounds=2)
            return DTSEngine(config=config)
        except ImportError:
            pytest.skip("DTS not available")
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.config.beam_width == 3
        assert engine.config.max_rounds == 2
    
    @pytest.mark.asyncio
    async def test_conversation_optimization(self, engine):
        """Test conversation optimization."""
        result = await engine.optimize_conversation(
            initial_context="Customer service scenario",
            goal="Resolve customer issue",
            rounds=2
        )
        
        assert result is not None
        assert hasattr(result, 'best_score')
    
    def test_best_path_selection(self, engine):
        """Test best path selection."""
        from integrations.dts.conversation_tree import ConversationTree, ConversationNode
        
        tree = ConversationTree()
        tree.root = ConversationNode(message="Hello", speaker="system", depth=0, score=8.0)
        tree.root.children = [
            ConversationNode(message="Option A", speaker="user", depth=1, score=9.0),
            ConversationNode(message="Option B", speaker="user", depth=1, score=7.0)
        ]
        
        best_path = engine.get_best_path(tree)
        assert len(best_path) > 0
        assert best_path[-1].score >= 8.0


class TestDTSKGIntegration:
    """Test suite for DTS Knowledge Engine integration."""
    
    @pytest.fixture
    def kg_integration(self):
        """Create DTS KG integration."""
        try:
            from knowledge_engine.integrations.dts.dts_integration import DTSKGIntegration
            return DTSKGIntegration()
        except ImportError:
            pytest.skip("DTS KG integration not available")
    
    def test_kg_integration_initialization(self, kg_integration):
        """Test KG integration initialization."""
        assert kg_integration is not None
    
    @pytest.mark.asyncio
    async def test_optimize_kg_query_dialog(self, kg_integration):
        """Test KG query dialog optimization."""
        result = await kg_integration.optimize_kg_query_dialog(
            context="Find information about companies",
            user_goal="Extract entity relationships"
        )
        
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_extract_kg_via_dialog(self, kg_integration):
        """Test KG extraction via dialog."""
        result = await kg_integration.extract_kg_via_dialog(
            text="Apple Inc. was founded by Steve Jobs in Cupertino.",
            entity_types=['ORG', 'PERSON', 'LOCATION']
        )
        
        assert result is not None
        assert isinstance(result, dict)
    
    def test_is_available(self, kg_integration):
        """Test availability check."""
        available = kg_integration.is_available()
        assert isinstance(available, bool)


class TestUnifiedHubIntegration:
    """Test suite for Unified Hub integration."""
    
    @pytest.mark.asyncio
    async def test_hub_initialization(self):
        """Test that DTS is in the hub."""
        try:
            from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
            
            hub = UnifiedKGIntegrationHub()
            await hub.initialize()
            
            # Check that DTS operation type exists
            from knowledge_engine.unified_kg_integration_hub import KGOperationType
            assert hasattr(KGOperationType, 'CONVERSATION_OPTIMIZATION')
            
            # Check that routing includes DTS
            assert 'dts' in hub._routing_map[KGOperationType.CONVERSATION_OPTIMIZATION]
        except ImportError:
            pytest.skip("Unified Hub not available")
    
    @pytest.mark.asyncio
    async def test_optimize_conversation_api(self):
        """Test the optimize_conversation API."""
        try:
            from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
            
            hub = UnifiedKGIntegrationHub()
            await hub.initialize()
            
            # This will fail if DTS not available, but tests the API exists
            if 'dts' in hub._integrations:
                result = await hub.optimize_conversation(
                    initial_context="Test context",
                    goal="Test goal"
                )
                assert result is not None
        except ImportError:
            pytest.skip("Unified Hub not available")


class TestMasterEngineIntegration:
    """Test suite for Master Engine integration."""
    
    def test_master_engine_has_dts(self):
        """Test that Master Engine has DTS component."""
        try:
            from knowledge_engine.master_engine import MasterKnowledgeEngine
            
            engine = MasterKnowledgeEngine()
            
            # Check DTS is in capabilities
            assert 'dts' in engine.capabilities
            assert 'conversation_optimization' in engine.capabilities['dts']
            
            # Check DTS component exists
            assert 'dts' in engine.components
        except ImportError:
            pytest.skip("Master Engine not available")


def run_all_tests():
    """Run all DTS integration tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
