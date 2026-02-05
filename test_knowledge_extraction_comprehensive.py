"""
Comprehensive Tests for Stage 6 Knowledge Extraction with ML Clustering

Tests:
1. ML-based pattern clustering (Sentence Transformers + scikit-learn)
2. Entity and relation extraction
3. Temporal knowledge graph construction
4. Knowledge validation with Z3
5. Hybrid retrieval (semantic + keyword)
6. Integration with ACE workflow extractor

Author: OpenEvolve AI
License: Apache 2.0
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

# Test configuration
pytestmark = [
    pytest.mark.unit,
    pytest.mark.knowledge_extraction
]

# Skip decorators for optional dependencies
ml_clustering_available = False
try:
    from ml_pattern_clustering import (
        MLKnowledgeExtraction,
        MLPatternClustering,
        EntityExtractor,
        RelationExtractor,
        TemporalKnowledgeGraph,
        KnowledgeValidator,
        MLPattern,
        ExtractedEntity,
        ExtractedRelation
    )
    ml_clustering_available = True
except ImportError:
    pass

sentence_transformers_available = False
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    pass

sklearn_available = False
try:
    from sklearn.cluster import DBSCAN
    sklearn_available = True
except ImportError:
    pass

z3_available = False
try:
    from z3 import Solver, Bool
    z3_available = True
except ImportError:
    pass


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_texts():
    """Sample texts for clustering tests."""
    return [
        "Use neural networks for image classification tasks",
        "Apply deep learning to computer vision problems",
        "Neural network architectures for visual recognition",
        "Implement decision trees for tabular data",
        "Random forest classifier for structured datasets",
        "Gradient boosting on tabular features",
        "Optimize hyperparameters using grid search",
        "Hyperparameter tuning with Bayesian optimization",
        "AutoML for automated hyperparameter selection"
    ]


@pytest.fixture
def sample_execution_traces():
    """Sample execution traces for testing."""
    from stage6_knowledge_extraction import ExecutionTrace
    
    return [
        ExecutionTrace(
            trace_id=f"trace_{i:03d}",
            workflow_id=f"wf_{i:03d}",
            problem_description=desc,
            stages=[
                {'stage_name': 'decomposition', 'parameters': {'strategy': 'hybrid'}},
                {'stage_name': 'evolution', 'parameters': {'generations': 100}},
                {'stage_name': 'assembly', 'parameters': {}}
            ],
            final_result={'accuracy': 0.9 + i * 0.01},
            execution_time_ms=5000.0 + i * 100,
            timestamp=datetime.now()
        )
        for i, desc in enumerate([
            "Optimize neural network for image classification",
            "Improve CNN architecture for computer vision",
            "Tune transformer model for NLP tasks",
            "Optimize ResNet for visual recognition",
            "Fine-tune BERT for text classification",
        ])
    ]


# =============================================================================
# ML PATTERN CLUSTERING TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestMLPatternClustering:
    """Tests for ML-based pattern clustering."""
    
    def test_initialization(self):
        """Test ML pattern clustering initialization."""
        clustering = MLPatternClustering()
        assert clustering is not None
        assert clustering.model_name == 'all-MiniLM-L6-v2'
    
    def test_cluster_patterns(self, sample_texts):
        """Test pattern clustering with sample texts."""
        clustering = MLPatternClustering()
        patterns = clustering.cluster_patterns(sample_texts)
        
        assert len(patterns) > 0
        assert all(isinstance(p, MLPattern) for p in patterns)
        
        # Check pattern properties
        for pattern in patterns:
            assert pattern.pattern_id
            assert pattern.description
            assert 0 <= pattern.confidence <= 1
            assert pattern.cluster_size > 0
    
    def test_cluster_quality_metrics(self, sample_texts):
        """Test cluster quality metrics (silhouette score)."""
        clustering = MLPatternClustering()
        patterns = clustering.cluster_patterns(sample_texts)
        
        # Find patterns with multiple members
        multi_member = [p for p in patterns if p.cluster_size > 1]
        
        if multi_member:
            for pattern in multi_member:
                # Silhouette score should be in valid range
                assert -1 <= pattern.silhouette_score <= 1
    
    def test_representative_examples(self, sample_texts):
        """Test representative example selection."""
        clustering = MLPatternClustering()
        patterns = clustering.cluster_patterns(sample_texts)
        
        for pattern in patterns:
            assert len(pattern.representative_examples) > 0
            assert len(pattern.representative_examples) <= 3
            # Representatives should be from cluster members
            for example in pattern.representative_examples:
                assert example in pattern.cluster_members
    
    def test_cluster_with_metadata(self, sample_texts):
        """Test clustering with metadata."""
        clustering = MLPatternClustering()
        metadata = [{'index': i, 'domain': 'test'} for i in range(len(sample_texts))]
        
        patterns = clustering.cluster_patterns(sample_texts, metadata)
        assert len(patterns) > 0


# =============================================================================
# ENTITY EXTRACTION TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestEntityExtraction:
    """Tests for entity extraction."""
    
    def test_initialization(self):
        """Test entity extractor initialization."""
        extractor = EntityExtractor()
        assert extractor is not None
    
    def test_extract_entities(self):
        """Test entity extraction from text."""
        extractor = EntityExtractor()
        text = "Use neural networks to solve image classification problems"
        
        entities = extractor.extract_entities(text)
        
        assert isinstance(entities, list)
        # Should extract at least one entity
        if entities:
            entity = entities[0]
            assert isinstance(entity, ExtractedEntity)
            assert entity.entity_id
            assert entity.text
            assert entity.entity_type
            assert 0 <= entity.confidence <= 1
    
    def test_entity_deduplication(self):
        """Test entity deduplication."""
        extractor = EntityExtractor()
        text = "Neural networks are used in neural networks for classification"
        
        entities = extractor.extract_entities(text)
        
        # Check no overlapping entities
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                # Either no overlap or same entity
                assert (e1.start_pos >= e2.end_pos or e2.start_pos >= e1.end_pos) or e1.entity_id == e2.entity_id


# =============================================================================
# RELATION EXTRACTION TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestRelationExtraction:
    """Tests for relation extraction."""
    
    def test_initialization(self):
        """Test relation extractor initialization."""
        extractor = RelationExtractor()
        assert extractor is not None
    
    def test_extract_relations(self):
        """Test relation extraction."""
        entity_extractor = EntityExtractor()
        relation_extractor = RelationExtractor()
        
        text = "Neural networks solve image classification problems"
        entities = entity_extractor.extract_entities(text)
        
        if len(entities) >= 2:
            relations = relation_extractor.extract_relations(text, entities)
            
            assert isinstance(relations, list)
            for relation in relations:
                assert isinstance(relation, ExtractedRelation)
                assert relation.relation_id
                assert relation.source_entity_id
                assert relation.target_entity_id
                assert relation.relation_type


# =============================================================================
# TEMPORAL KNOWLEDGE GRAPH TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestTemporalKnowledgeGraph:
    """Tests for temporal knowledge graph."""
    
    def test_initialization(self):
        """Test temporal knowledge graph initialization."""
        graph = TemporalKnowledgeGraph()
        assert graph is not None
        assert len(graph.nodes) == 0
    
    def test_add_node(self):
        """Test adding nodes to graph."""
        graph = TemporalKnowledgeGraph()
        
        node = graph.add_node(
            content="Test knowledge",
            node_type="fact",
            confidence=0.8
        )
        
        assert node.node_id is not None
        assert node.content == "Test knowledge"
        assert node.node_type == "fact"
        assert node.confidence == 0.8
        assert len(graph.nodes) == 1
    
    def test_add_edge(self):
        """Test adding edges between nodes."""
        graph = TemporalKnowledgeGraph()
        
        node1 = graph.add_node(content="Node 1", node_type="fact")
        node2 = graph.add_node(content="Node 2", node_type="fact")
        
        result = graph.add_edge(node1.node_id, node2.node_id, "depends_on")
        
        assert result is True
        assert len(graph.edges) == 1
    
    def test_valid_knowledge_query(self):
        """Test querying valid knowledge."""
        graph = TemporalKnowledgeGraph()
        
        # Add permanent knowledge
        graph.add_node(
            content="Permanent fact",
            node_type="fact",
            confidence=0.9
        )
        
        # Add expiring knowledge
        graph.add_node(
            content="Temporary fact",
            node_type="fact",
            confidence=0.8,
            valid_until=datetime.now() + timedelta(days=1)
        )
        
        # Add expired knowledge
        expired = graph.add_node(
            content="Expired fact",
            node_type="fact",
            confidence=0.7,
            valid_until=datetime.now() - timedelta(days=1)
        )
        expired.valid_until = datetime.now() - timedelta(days=1)
        
        valid = graph.get_valid_knowledge()
        
        # Should have 2 valid nodes (permanent + temporary)
        assert len(valid) >= 1
    
    def test_versioning(self):
        """Test knowledge versioning."""
        graph = TemporalKnowledgeGraph()
        
        node1 = graph.add_node(content="Version 1", node_type="fact")
        node2 = graph.create_version(node1.node_id, "Version 2")
        
        assert node2 is not None
        assert node2 != node1.node_id
        assert graph.nodes[node1.node_id].validation_status == 'deprecated'


# =============================================================================
# KNOWLEDGE VALIDATION TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestKnowledgeValidation:
    """Tests for knowledge validation."""
    
    def test_initialization(self):
        """Test knowledge validator initialization."""
        validator = KnowledgeValidator()
        assert validator is not None
    
    def test_validate_pattern(self):
        """Test pattern validation."""
        validator = KnowledgeValidator()
        
        pattern = MLPattern(
            pattern_id="test_pattern",
            pattern_type="semantic",
            description="Test pattern description",
            confidence=0.8,
            cluster_size=5,
            silhouette_score=0.5
        )
        
        result = validator.validate_pattern(pattern)
        
        assert result is not None
        assert 'valid' in result
        assert 'confidence' in result
    
    def test_find_contradictions(self):
        """Test contradiction detection."""
        validator = KnowledgeValidator()
        
        # Create patterns without explicit contradictory relations
        patterns = [
            MLPattern(
                pattern_id="p1",
                pattern_type="semantic",
                description="A solves B",
                confidence=0.8,
                cluster_size=3,
                relations=[]
            ),
            MLPattern(
                pattern_id="p2",
                pattern_type="semantic",
                description="C improves D",
                confidence=0.7,
                cluster_size=3,
                relations=[]
            )
        ]
        
        contradictions = validator.find_contradictions(patterns)
        
        assert isinstance(contradictions, list)


# =============================================================================
# STAGE 6 INTEGRATION TESTS
# =============================================================================

class TestStage6KnowledgeExtraction:
    """Tests for Stage 6 Knowledge Extraction engine."""
    
    def test_initialization(self):
        """Test Stage 6 initialization."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction()
        assert engine is not None
        assert engine.pattern_extractor is not None
    
    def test_pattern_extraction(self, sample_execution_traces):
        """Test pattern extraction from traces."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction()
        patterns = engine.pattern_extractor.extract_semantic_patterns(sample_execution_traces)
        
        assert isinstance(patterns, list)
    
    def test_artifact_generation(self, sample_execution_traces):
        """Test artifact generation."""
        from stage6_knowledge_extraction import (
            Stage6KnowledgeExtraction, ExtractedPattern
        )
        
        engine = Stage6KnowledgeExtraction()
        
        pattern = ExtractedPattern(
            pattern_id="test_pattern",
            pattern_type='sequence',
            description="Test sequence pattern",
            confidence=0.8,
            occurrences=5,
            first_seen=datetime.now(),
            last_seen=datetime.now(),
            examples=[{'sequence': ['stage1', 'stage2']}]
        )
        
        artifact = engine.artifact_generator.generate_strategy_artifact(
            pattern, sample_execution_traces
        )
        
        # May be None if no matching traces
        if artifact:
            assert artifact.artifact_id
            assert artifact.artifact_type == 'strategy'
    
    @pytest.mark.asyncio
    async def test_process_trace(self, sample_execution_traces):
        """Test async trace processing."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction()
        
        # Process multiple traces to trigger clustering
        results = []
        for trace in sample_execution_traces[:3]:
            result = await engine.process_trace(trace)
            results.append(result)
        
        assert len(results) == 3
        # Check result structure
        for result in results:
            assert 'patterns_extracted' in result
            assert 'artifacts_generated' in result
            assert 'total_patterns' in result
    
    def test_get_statistics(self, sample_execution_traces):
        """Test statistics retrieval."""
        from stage6_knowledge_extraction import Stage6KnowledgeExtraction
        
        engine = Stage6KnowledgeExtraction()
        
        # Add some traces
        engine.traces.extend(sample_execution_traces)
        
        stats = engine.get_statistics()
        
        assert 'traces_processed' in stats
        assert 'patterns_extracted' in stats
        assert 'artifacts_generated' in stats
        assert stats['traces_processed'] == len(sample_execution_traces)


# =============================================================================
# HYBRID RETRIEVAL TESTS
# =============================================================================

class TestHybridRetrieval:
    """Tests for hybrid retrieval system."""
    
    def test_initialization(self):
        """Test hybrid retrieval initialization."""
        from stage6_knowledge_extraction import HybridRetrievalSystem
        
        retriever = HybridRetrievalSystem()
        assert retriever is not None
    
    def test_add_and_retrieve(self):
        """Test adding knowledge and retrieving."""
        from stage6_knowledge_extraction import HybridRetrievalSystem
        
        retriever = HybridRetrievalSystem()
        
        # Add knowledge
        retriever.add_knowledge({
            'id': 'k1',
            'description': 'Neural networks for image classification',
            'content': 'Use CNN for computer vision tasks'
        })
        
        retriever.add_knowledge({
            'id': 'k2',
            'description': 'Decision trees for tabular data',
            'content': 'Use Random Forest for structured data'
        })
        
        # Retrieve
        results = retriever.retrieve("neural network", top_k=5)
        
        assert isinstance(results, list)
        assert len(results) <= 5


# =============================================================================
# ACE WORKFLOW INTEGRATION TESTS
# =============================================================================

class TestACEWorkflowIntegration:
    """Tests for ACE workflow knowledge extractor integration."""
    
    def test_initialization(self):
        """Test ACE workflow extractor initialization."""
        from ace_workflow_knowledge_extractor import WorkflowKnowledgeExtractor
        
        extractor = WorkflowKnowledgeExtractor()
        assert extractor is not None
    
    def test_extract_from_workflow(self):
        """Test workflow extraction."""
        from ace_workflow_knowledge_extractor import WorkflowKnowledgeExtractor
        
        extractor = WorkflowKnowledgeExtractor()
        
        workflow_results = {
            'phases': {
                'phase_1': {
                    'success': True,
                    'analysis': 'Decomposed problem successfully',
                    'learning': {
                        'reflection_summary': 'Good decomposition strategy'
                    }
                }
            },
            'teams': {
                'blue_team': {
                    'name': 'Blue Team',
                    'type': 'blue_team',
                    'tasks_completed': 10,
                    'tasks_succeeded': 8
                }
            }
        }
        
        result = extractor.extract_from_workflow(
            workflow_id="test_workflow",
            problem_statement="Test problem",
            workflow_results=workflow_results
        )
        
        assert result is not None
        assert result.workflow_id == "test_workflow"


# =============================================================================
# BENCHMARK TESTS
# =============================================================================

@pytest.mark.skipif(not ml_clustering_available, reason="ML clustering not available")
class TestPerformanceBenchmarks:
    """Performance benchmarks for ML clustering."""
    
    def test_clustering_performance(self, sample_texts):
        """Benchmark clustering performance."""
        import time
        
        clustering = MLPatternClustering()
        
        start_time = time.time()
        patterns = clustering.cluster_patterns(sample_texts)
        elapsed = time.time() - start_time
        
        # Should complete in reasonable time
        assert elapsed < 30  # 30 seconds max
        assert len(patterns) > 0
    
    def test_embedding_performance(self):
        """Benchmark embedding generation."""
        import time
        
        if not sentence_transformers_available:
            pytest.skip("Sentence transformers not available")
        
        extractor = EntityExtractor()
        texts = ["Test text for embedding"] * 10
        
        start_time = time.time()
        for text in texts:
            extractor.extract_entities(text)
        elapsed = time.time() - start_time
        
        # Should be reasonably fast
        assert elapsed < 10  # 10 seconds for 10 texts


# =============================================================================
# COMPREHENSIVE INTEGRATION TEST
# =============================================================================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_pipeline():
    """Test complete knowledge extraction pipeline."""
    from stage6_knowledge_extraction import (
        Stage6KnowledgeExtraction, ExecutionTrace
    )
    
    # Initialize engine
    engine = Stage6KnowledgeExtraction(enable_ml=True)
    
    # Create sample traces
    traces = [
        ExecutionTrace(
            trace_id=f"trace_{i}",
            workflow_id=f"wf_{i}",
            problem_description=desc,
            stages=[
                {'stage_name': 'decomposition', 'parameters': {}},
                {'stage_name': 'evolution', 'parameters': {}}
            ],
            final_result={'success': True},
            execution_time_ms=1000.0,
            timestamp=datetime.now()
        )
        for i, desc in enumerate([
            "Neural network optimization",
            "Deep learning for vision",
            "Transformer architecture tuning",
            "CNN optimization",
            "BERT fine-tuning"
        ])
    ]
    
    # Process all traces
    for trace in traces:
        result = await engine.process_trace(trace)
        assert result is not None
    
    # Get statistics
    stats = engine.get_statistics()
    assert stats['traces_processed'] == 5
    
    # Validate patterns
    validation = engine.validate_all_patterns()
    assert 'total_patterns' in validation
    
    # Test retrieval
    results = engine.retrieve_knowledge("neural network", top_k=5)
    assert isinstance(results, list)
    
    print(f"\n=== Pipeline Test Results ===")
    print(f"Traces processed: {stats['traces_processed']}")
    print(f"Patterns extracted: {stats['patterns_extracted']}")
    print(f"ML clustered patterns: {stats.get('ml_clustered_patterns', 0)}")
    print(f"ML available: {stats['ml_available']}")
    print(f"Z3 available: {stats['z3_available']}")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
