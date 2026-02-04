"""
Comprehensive Tests for ICR (Iterative Contextual Refinements) Integration

Tests cover:
- Generator
- Critic
- Refiner
- Judge
- ICR Engine
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


class TestGenerator:
    """Test suite for content generation."""
    
    @pytest.fixture
    def generator(self):
        """Create generator."""
        try:
            from integrations.icr.generator import Generator
            return Generator()
        except ImportError:
            pytest.skip("ICR not available")
    
    def test_generation(self, generator):
        """Test basic generation."""
        result = generator.generate(
            prompt="Generate a summary of Apple Inc.",
            context={}
        )
        
        assert result is not None
        assert hasattr(result, 'content')
    
    def test_variant_generation(self, generator):
        """Test variant generation."""
        variants = generator.generate_variants(
            prompt="Generate entity extraction",
            n=3
        )
        
        assert len(variants) == 3
        assert all(hasattr(v, 'content') for v in variants)


class TestCritic:
    """Test suite for content critique."""
    
    @pytest.fixture
    def critic(self):
        """Create critic."""
        try:
            from integrations.icr.critic import Critic
            return Critic()
        except ImportError:
            pytest.skip("ICR not available")
    
    def test_critique(self, critic):
        """Test critique generation."""
        from integrations.icr.generator import GenerationResult
        
        output = GenerationResult(
            content="Apple was founded by Steve Jobs.",
            metadata={}
        )
        
        result = critic.critique(output, criteria=["accuracy", "completeness"])
        
        assert result is not None
        assert hasattr(result, 'issues')
    
    def test_issue_identification(self, critic):
        """Test issue identification."""
        from integrations.icr.generator import GenerationResult
        
        output = GenerationResult(content="Incomplete info", metadata={})
        issues = critic.identify_issues(output)
        
        assert isinstance(issues, list)


class TestRefiner:
    """Test suite for content refinement."""
    
    @pytest.fixture
    def refiner(self):
        """Create refiner."""
        try:
            from integrations.icr.refiner import Refiner
            return Refiner()
        except ImportError:
            pytest.skip("ICR not available")
    
    def test_refinement(self, refiner):
        """Test basic refinement."""
        from integrations.icr.generator import GenerationResult
        from integrations.icr.critic import CritiqueResult
        
        output = GenerationResult(content="Apple is a company.", metadata={})
        critique = CritiqueResult(issues=[], suggestions=[])
        
        result = refiner.refine(output, critique)
        
        assert result is not None
    
    def test_convergence_tracking(self, refiner):
        """Test convergence tracking."""
        history = [0.6, 0.7, 0.75]
        converged = refiner.check_convergence(history, threshold=0.01)
        
        assert isinstance(converged, bool)


class TestJudge:
    """Test suite for quality judgment."""
    
    @pytest.fixture
    def judge(self):
        """Create judge."""
        try:
            from integrations.icr.judge import Judge
            return Judge()
        except ImportError:
            pytest.skip("ICR not available")
    
    def test_evaluation(self, judge):
        """Test quality evaluation."""
        from integrations.icr.generator import GenerationResult
        
        output = GenerationResult(
            content="Apple Inc. was founded by Steve Jobs in 1976.",
            metadata={}
        )
        
        result = judge.evaluate(output, criteria={"accuracy": 1.0, "completeness": 1.0})
        
        assert result is not None
        assert hasattr(result, 'score')
        assert 0 <= result.score <= 1
    
    def test_meets_threshold(self, judge):
        """Test threshold checking."""
        from integrations.icr.generator import GenerationResult
        
        output = GenerationResult(content="Good content", metadata={})
        output.quality_score = 0.9
        
        meets = judge.meets_threshold(output, threshold=0.85)
        assert meets is True


class TestICREngine:
    """Test suite for ICR Engine."""
    
    @pytest.fixture
    def engine(self):
        """Create ICR engine."""
        try:
            from integrations.icr.iterative_engine import ICREngine
            return ICREngine(max_iterations=3, quality_threshold=0.85)
        except ImportError:
            pytest.skip("ICR not available")
    
    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.max_iterations == 3
        assert engine.quality_threshold == 0.85
    
    @pytest.mark.asyncio
    async def test_refinement_loop(self, engine):
        """Test refinement loop."""
        result = await engine.refine(
            prompt="Extract entities from: Apple Inc.",
            initial_output="Apple is a company.",
            goal="Extract accurate entity information"
        )
        
        assert result is not None
        assert hasattr(result, 'final_output')
    
    def test_should_continue(self, engine):
        """Test continuation logic."""
        should = engine.should_continue(
            current_score=0.7,
            iteration=1,
            max_iterations=5,
            threshold=0.9
        )
        
        assert should is True  # Should continue if below threshold


class TestICRKGIntegration:
    """Test suite for ICR KG Integration."""
    
    @pytest.fixture
    def kg_integration(self):
        """Create ICR KG integration."""
        try:
            from knowledge_engine.integrations.icr.icr_integration import ICRKGIntegration
            return ICRKGIntegration()
        except ImportError:
            pytest.skip("ICR KG integration not available")
    
    def test_kg_integration_initialization(self, kg_integration):
        """Test KG integration initialization."""
        assert kg_integration is not None
    
    @pytest.mark.asyncio
    async def test_refine_kg_extraction(self, kg_integration):
        """Test KG extraction refinement."""
        initial_extraction = {
            "content": "Apple was founded by Steve Jobs.",
            "goal": "Extract complete entity information"
        }
        
        result = await kg_integration.refine_kg_extraction(initial_extraction)
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_improve_cypher_query(self, kg_integration):
        """Test Cypher query improvement."""
        query = "MATCH (n) RETURN n"
        
        result = await kg_integration.improve_cypher_query(query)
        assert result is not None
    
    def test_is_available(self, kg_integration):
        """Test availability check."""
        available = kg_integration.is_available()
        assert isinstance(available, bool)


class TestUnifiedHubIntegration:
    """Test suite for Unified Hub integration."""
    
    @pytest.mark.asyncio
    async def test_hub_initialization(self):
        """Test that ICR is in the hub."""
        try:
            from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
            
            hub = UnifiedKGIntegrationHub()
            await hub.initialize()
            
            # Check that ITERATIVE_REFINEMENT operation type exists
            from knowledge_engine.unified_kg_integration_hub import KGOperationType
            assert hasattr(KGOperationType, 'ITERATIVE_REFINEMENT')
            
            # Check that routing includes ICR
            assert 'icr' in hub._routing_map[KGOperationType.ITERATIVE_REFINEMENT]
        except ImportError:
            pytest.skip("Unified Hub not available")


class TestMasterEngineIntegration:
    """Test suite for Master Engine integration."""
    
    def test_master_engine_has_icr(self):
        """Test that Master Engine has ICR component."""
        try:
            from knowledge_engine.master_engine import MasterKnowledgeEngine
            
            engine = MasterKnowledgeEngine()
            
            # Check ICR is in capabilities
            assert 'icr' in engine.capabilities
            assert 'iterative_refinement' in engine.capabilities['icr']
            
            # Check ICR component exists
            assert 'icr' in engine.components
        except ImportError:
            pytest.skip("Master Engine not available")


def run_all_tests():
    """Run all ICR integration tests."""
    pytest.main([__file__, '-v', '--tb=short'])


if __name__ == '__main__':
    run_all_tests()
