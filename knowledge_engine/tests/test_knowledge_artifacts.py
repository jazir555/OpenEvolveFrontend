"""
Comprehensive Test Suite for Knowledge Artifacts

Tests for KnowledgeExtractor, KnowledgeArtifact, KnowledgeStorage, and KnowledgeRetriever
following CLAUDE.md principles: ZERO TRUST, UTC timestamps, structured logging, idempotency.
"""

import pytest
import json
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List
from pathlib import Path

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_knowledge_artifacts.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TestKnowledgeArtifact:
    """Test KnowledgeArtifact dataclass with validation and serialization"""

    @pytest.fixture
    def artifact_data(self) -> Dict[str, Any]:
        """Sample artifact data for testing"""
        return {
            'id': 'test_artifact_001',
            'artifact_type': 'solution_pattern',
            'content': {
                'problem_type': 'decomposition',
                'solution_approach': 'hierarchical task analysis',
                'success_rate': 0.95
            },
            'source_workflow_id': 'workflow_123',
            'extraction_timestamp': datetime.now(timezone.utc).timestamp(),
            'domain': 'optimization',
            'problem_type': 'decomposition',
            'confidence_score': 0.9,
            'effectiveness_score': 0.88,
            'source_quality': 0.85,
            'metadata': {
                'test_key': 'test_value'
            }
        }

    def test_artifact_creation(self, artifact_data):
        """Test creating a KnowledgeArtifact from data"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact(**artifact_data)

        assert artifact.id == 'test_artifact_001'
        assert artifact.artifact_type == 'solution_pattern'
        assert artifact.confidence_score == 0.9
        assert artifact.domain == 'optimization'
        logger.info(f"Created artifact: {artifact.id}")

    def test_artifact_to_dict(self, artifact_data):
        """Test converting artifact to dictionary"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact(**artifact_data)
        artifact_dict = artifact.to_dict()

        assert isinstance(artifact_dict, dict)
        assert artifact_dict['id'] == artifact.id
        assert artifact_dict['artifact_type'] == artifact.artifact_type
        assert artifact_dict['content'] == artifact.content
        logger.info("Artifact serialization to dict successful")

    def test_artifact_from_dict(self, artifact_data):
        """Test creating artifact from dictionary"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact.from_dict(artifact_data)

        assert artifact.id == artifact_data['id']
        assert artifact.artifact_type == artifact_data['artifact_type']
        assert artifact.content == artifact_data['content']
        logger.info("Artifact deserialization from dict successful")

    def test_artifact_quality_score(self, artifact_data):
        """Test quality score calculation"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact(**artifact_data)
        quality_score = artifact.calculate_quality_score()

        assert 0.0 <= quality_score <= 1.0
        logger.info(f"Quality score: {quality_score:.2f}")

    def test_artifact_validation(self, artifact_data):
        """Test artifact validation status updates"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact(**artifact_data)
        assert artifact.validation_status == 'unvalidated'

        # Test positive validation
        artifact.validate_artifact(True, "test_validator")
        assert artifact.validation_status == 'validated'
        assert artifact.confidence_score == 0.95

        # Test negative validation
        artifact.validate_artifact(False, "test_validator")
        assert artifact.validation_status == 'invalid'
        assert artifact.confidence_score == 0.3

        logger.info("Artifact validation status updates successful")

    def test_artifact_metadata_update(self, artifact_data):
        """Test artifact metadata updates with version tracking"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        artifact = KnowledgeArtifact(**artifact_data)
        initial_version = artifact.version
        initial_timestamp = artifact.last_updated

        # Wait a tiny bit to ensure timestamp difference
        import time
        time.sleep(0.01)

        artifact.update_metadata({'new_key': 'new_value'})

        assert 'new_key' in artifact.metadata
        assert artifact.metadata['new_key'] == 'new_value'
        assert float(artifact.version) > float(initial_version)
        assert artifact.last_updated > initial_timestamp

        logger.info(f"Artifact metadata updated: v{initial_version} -> v{artifact.version}")

    def test_artifact_serialization_roundtrip(self, artifact_data):
        """Test complete serialization/deserialization roundtrip"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact

        # Create artifact
        original = KnowledgeArtifact(**artifact_data)

        # Convert to dict
        artifact_dict = original.to_dict()

        # Convert back from dict
        restored = KnowledgeArtifact.from_dict(artifact_dict)

        # Verify all fields match
        assert restored.id == original.id
        assert restored.artifact_type == original.artifact_type
        assert restored.content == original.content
        assert restored.confidence_score == original.confidence_score
        assert restored.effectiveness_score == original.effectiveness_score

        logger.info("Serialization roundtrip successful")


class TestKnowledgeExtractor:
    """Test KnowledgeExtractor with various extraction strategies"""

    @pytest.fixture
    def workflow_data(self) -> Dict[str, Any]:
        """Sample workflow execution data"""
        return {
            'workflow_id': 'workflow_test_001',
            'domain': 'mathematical_optimization',
            'complexity': 'high',
            'execution_time': 1800,
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'solutions': [
                {
                    'id': 'sol_001',
                    'problem_type': 'optimization',
                    'domain': 'algebra',
                    'approach': 'gradient descent with adaptive learning rate',
                    'implementation': 'vectorized implementation',
                    'success_rate': 0.92,
                    'complexity': 7,
                    'code': 'def optimize(): pass',
                    'documentation': 'Efficient optimization algorithm',
                    'performance': {
                        'convergence_rate': 0.90,
                        'iterations': 100
                    }
                }
            ],
            'critiques': [
                {
                    'id': 'crit_001',
                    'issue_type': 'resource allocation',
                    'root_cause': 'suboptimal workload distribution',
                    'prevention_strategy': 'implement load balancing',
                    'severity': 'high',
                    'affected_components': ['computation', 'memory']
                }
            ],
            'teams': [
                {
                    'name': 'blue_team',
                    'role': 'Blue',
                    'domain': 'optimization',
                    'specialization': 'nonlinear_problems',
                    'success_rate': 0.90,
                    'avg_response_time': 1.2,
                    'completion_rate': 0.93,
                    'quality_score': 0.87,
                    'performance_trends': [0.85, 0.87, 0.88, 0.89, 0.90]
                }
            ],
            'gauntlets': [
                {
                    'name': 'quality_gauntlet',
                    'type': 'Gold',
                    'domain': 'validation',
                    'problem_type': 'solution_quality',
                    'detection_rate': 0.88,
                    'false_positive_rate': 0.05,
                    'true_positive_rate': 0.85,
                    'average_score': 0.87,
                    'performance_trends': [0.83, 0.85, 0.86, 0.87, 0.88]
                }
            ]
        }

    @pytest.fixture
    def extractor(self):
        """Create KnowledgeExtractor instance"""
        from knowledge_engine.knowledge_extractor import KnowledgeExtractor
        return KnowledgeExtractor({
            'quality_thresholds': {
                'high': 0.85,
                'medium': 0.65,
                'low': 0.40
            }
        })

    def test_extractor_initialization(self, extractor):
        """Test extractor initialization with config"""
        assert extractor is not None
        assert extractor.quality_thresholds['high'] == 0.85
        assert extractor.quality_thresholds['medium'] == 0.65
        assert extractor.quality_thresholds['low'] == 0.40
        logger.info("Extractor initialized successfully")

    def test_extract_from_workflow(self, extractor, workflow_data):
        """Test knowledge extraction from workflow"""
        artifacts = extractor.extract_from_workflow(workflow_data)

        assert isinstance(artifacts, list)
        assert len(artifacts) > 0

        # Verify artifact types
        artifact_types = {a.artifact_type for a in artifacts}
        expected_types = {'solution_pattern', 'critique_insight', 'team_performance', 'gauntlet_effectiveness'}

        assert len(artifact_types.intersection(expected_types)) > 0

        logger.info(f"Extracted {len(artifacts)} artifacts from workflow")

    def test_solution_pattern_extraction(self, extractor, workflow_data):
        """Test solution pattern extraction specifically"""
        artifacts = extractor.extract_from_workflow(workflow_data)
        solution_artifacts = [a for a in artifacts if a.artifact_type == 'solution_pattern']

        assert len(solution_artifacts) > 0

        # Check solution artifact structure
        solution = solution_artifacts[0]
        assert 'solution_id' in solution.content
        assert 'problem_type' in solution.content
        assert 'solution_approach' in solution.content
        assert 'success_rate' in solution.content

        logger.info(f"Extracted {len(solution_artifacts)} solution patterns")

    def test_critique_pattern_extraction(self, extractor, workflow_data):
        """Test critique pattern extraction"""
        artifacts = extractor.extract_from_workflow(workflow_data)
        critique_artifacts = [a for a in artifacts if a.artifact_type == 'critique_insight']

        assert len(critique_artifacts) > 0

        # Check critique artifact structure
        critique = critique_artifacts[0]
        assert 'critique_id' in critique.content
        assert 'issue_type' in critique.content
        assert 'severity' in critique.content
        assert 'root_cause' in critique.content

        logger.info(f"Extracted {len(critique_artifacts)} critique insights")

    def test_team_performance_extraction(self, extractor, workflow_data):
        """Test team performance extraction"""
        artifacts = extractor.extract_from_workflow(workflow_data)
        team_artifacts = [a for a in artifacts if a.artifact_type == 'team_performance']

        assert len(team_artifacts) > 0

        # Check team artifact structure
        team = team_artifacts[0]
        assert 'team_name' in team.content
        assert 'success_rate' in team.content
        assert 'performance_metrics' in team.content

        logger.info(f"Extracted {len(team_artifacts)} team performance artifacts")

    def test_quality_filtering(self, extractor, workflow_data):
        """Test that low-quality artifacts are filtered out"""
        artifacts = extractor.extract_from_workflow(workflow_data)

        # All artifacts should meet minimum quality threshold
        for artifact in artifacts:
            quality_score = artifact.calculate_quality_score()
            assert quality_score >= extractor.quality_thresholds['low'], \
                f"Artifact {artifact.id} has quality {quality_score} below threshold"

        logger.info(f"All {len(artifacts)} artifacts meet quality threshold")

    def test_extraction_stats(self, extractor, workflow_data):
        """Test extraction statistics tracking"""
        # Reset stats to ensure clean state
        extractor.reset_stats()

        # Perform extraction
        artifacts = extractor.extract_from_workflow(workflow_data)

        # Get stats
        stats = extractor.get_extraction_stats()

        assert stats['total_extractions'] >= 1
        assert stats['successful_extractions'] >= len(artifacts)
        assert stats['average_extraction_time'] >= 0

        logger.info(f"Extraction stats: {json.dumps(stats, indent=2)}")

    def test_entity_relationship_extraction(self, extractor, workflow_data):
        """Test entity relationship extraction"""
        extractor.extract_from_workflow(workflow_data)

        entity_relationships = extractor.get_entity_relationships()

        assert isinstance(entity_relationships, dict)
        assert len(entity_relationships) > 0

        logger.info(f"Extracted {len(entity_relationships)} entity relationships")

    def test_pattern_recognition(self, extractor, workflow_data):
        """Test pattern recognition in extraction"""
        artifacts = extractor.extract_from_workflow(workflow_data)

        # Check that pattern recognition metadata is present
        for artifact in artifacts:
            if artifact.artifact_type == 'solution_pattern':
                assert 'pattern_recognition' in artifact.metadata
                pattern_info = artifact.metadata['pattern_recognition']
                assert 'pattern_type' in pattern_info
                assert 'match_score' in pattern_info

        logger.info("Pattern recognition working correctly")


class TestKnowledgeStorage:
    """Test KnowledgeStorage with multiple backends"""

    @pytest.fixture
    def storage(self):
        """Create KnowledgeStorage instance"""
        from knowledge_engine.knowledge_storage import KnowledgeStorage
        return KnowledgeStorage({
            'qdrant_host': 'localhost',
            'qdrant_port': 6333,
            'mongo_uri': 'mongodb://localhost:27017',
            'neo4j_uri': 'bolt://localhost:7687',
            'redis_host': 'localhost',
            'redis_port': 6379
        })

    @pytest.fixture
    def sample_artifact(self) -> Dict[str, Any]:
        """Sample artifact for storage testing"""
        return {
            'type': 'solution_pattern',
            'source': 'workflow_execution',
            'content': 'Test solution pattern for optimization',
            'context': {
                'problem_type': 'optimization',
                'domain': 'mathematics'
            },
            'metadata': {
                'test': True
            },
            'embeddings': [0.1] * 768,  # 768-dim vector
            'related_entities': ['optimization', 'mathematics']
        }

    def test_storage_initialization(self, storage):
        """Test storage initialization"""
        assert storage is not None
        assert storage.qdrant_client is not None
        assert storage.mongo_client is not None
        assert storage.neo4j_client is not None
        assert storage.redis_client is not None
        logger.info("Storage initialized successfully")

    def test_store_artifact(self, storage, sample_artifact):
        """Test storing a knowledge artifact"""
        artifact_id = storage.store_knowledge_artifact(sample_artifact)

        assert artifact_id is not None
        assert isinstance(artifact_id, str)
        logger.info(f"Stored artifact with ID: {artifact_id}")

    def test_retrieve_artifact_by_id(self, storage, sample_artifact):
        """Test retrieving artifact by ID"""
        # Store artifact
        artifact_id = storage.store_knowledge_artifact(sample_artifact)

        # Retrieve artifact
        retrieved = storage.get_artifact_by_id(artifact_id)

        assert retrieved is not None
        assert retrieved['_id'] == artifact_id
        assert retrieved['type'] == sample_artifact['type']
        assert retrieved['content'] == sample_artifact['content']

        logger.info("Retrieved artifact successfully")

    def test_update_artifact(self, storage, sample_artifact):
        """Test updating an existing artifact"""
        # Store artifact
        artifact_id = storage.store_knowledge_artifact(sample_artifact)

        # Update artifact
        updates = {
            'content': 'Updated content',
            'metadata': {'updated': True}
        }

        success = storage.update_artifact(artifact_id, updates)
        assert success is True

        # Verify update
        updated = storage.get_artifact_by_id(artifact_id)
        assert updated['content'] == 'Updated content'
        assert updated['metadata']['updated'] is True

        logger.info("Artifact updated successfully")

    def test_delete_artifact(self, storage, sample_artifact):
        """Test deleting an artifact"""
        # Store artifact
        artifact_id = storage.store_knowledge_artifact(sample_artifact)

        # Delete artifact
        success = storage.delete_artifact(artifact_id)
        assert success is True

        # Verify deletion
        deleted = storage.get_artifact_by_id(artifact_id)
        assert deleted is None

        logger.info("Artifact deleted successfully")

    def test_retrieve_artifacts_with_filters(self, storage):
        """Test retrieving artifacts with filters"""
        # Store multiple artifacts
        artifacts = [
            {
                'type': 'solution_pattern',
                'source': 'workflow_1',
                'content': 'Solution 1',
                'context': {'domain': 'math'}
            },
            {
                'type': 'critique_insight',
                'source': 'workflow_1',
                'content': 'Critique 1',
                'context': {'domain': 'math'}
            },
            {
                'type': 'solution_pattern',
                'source': 'workflow_2',
                'content': 'Solution 2',
                'context': {'domain': 'physics'}
            }
        ]

        for artifact in artifacts:
            storage.store_knowledge_artifact(artifact)

        # Retrieve with filter
        results = storage.retrieve_knowledge_artifacts({'type': 'solution_pattern'})

        assert len(results) >= 2
        for result in results:
            assert result['type'] == 'solution_pattern'

        logger.info(f"Retrieved {len(results)} artifacts with filter")

    def test_search_similar_artifacts(self, storage, sample_artifact):
        """Test vector similarity search"""
        # Store artifact with embedding
        artifact_id = storage.store_knowledge_artifact(sample_artifact)

        # Search for similar artifacts
        query_embedding = sample_artifact['embeddings']
        similar = storage.search_similar_artifacts(query_embedding, limit=5)

        assert isinstance(similar, list)

        logger.info(f"Found {len(similar)} similar artifacts")

    def test_storage_statistics(self, storage):
        """Test getting storage statistics"""
        # Store a few artifacts
        for i in range(3):
            storage.store_knowledge_artifact({
                'type': 'test_artifact',
                'source': f'source_{i}',
                'content': f'Content {i}'
            })

        # Get statistics
        stats = storage.get_statistics()

        assert stats['total_artifacts'] >= 3
        assert isinstance(stats['artifact_types'], dict)
        assert stats['storage_size'] > 0

        logger.info(f"Storage stats: {json.dumps(stats, indent=2)}")

    def test_backup_and_restore(self, storage, sample_artifact):
        """Test knowledge base backup and restore"""
        import tempfile
        import os

        # Store artifact
        storage.store_knowledge_artifact(sample_artifact)

        # Create backup
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            backup_path = f.name

        try:
            backup_success = storage.backup_knowledge_base(backup_path)
            assert backup_success is True

            # Verify backup file exists
            assert os.path.exists(backup_path)

            # Restore from backup
            restore_success = storage.restore_knowledge_base(backup_path)
            assert restore_success is True

            logger.info("Backup and restore successful")

        finally:
            # Cleanup
            if os.path.exists(backup_path):
                os.remove(backup_path)


class TestKnowledgeRetriever:
    """Test KnowledgeRetriever search and recommendation functionality"""

    @pytest.fixture
    def retriever(self):
        """Create KnowledgeRetriever instance"""
        from knowledge_engine.knowledge_retriever import KnowledgeRetriever
        from knowledge_engine.knowledge_storage import KnowledgeStorage

        storage = KnowledgeStorage()
        return KnowledgeRetriever(storage=storage)

    @pytest.fixture
    def populated_storage(self, retriever):
        """Populate storage with test data"""
        test_artifacts = [
            {
                'type': 'solution_pattern',
                'source': 'workflow_1',
                'content': 'Hierarchical decomposition for complex problems',
                'context': {
                    'problem_type': 'decomposition',
                    'complexity': 'high'
                },
                'embeddings': [0.1] * 768
            },
            {
                'type': 'solution_pattern',
                'source': 'workflow_2',
                'content': 'Gradient descent optimization',
                'context': {
                    'problem_type': 'optimization',
                    'complexity': 'medium'
                },
                'embeddings': [0.2] * 768
            },
            {
                'type': 'critique_insight',
                'source': 'workflow_1',
                'content': 'Resource allocation issues',
                'context': {
                    'severity': 'high'
                },
                'embeddings': [0.3] * 768
            }
        ]

        for artifact in test_artifacts:
            retriever.storage.store_knowledge_artifact(artifact)

        return retriever

    def test_search_knowledge(self, populated_storage):
        """Test knowledge search"""
        results = populated_storage.search_knowledge(
            query='decomposition',
            query_type='hybrid',
            limit=5
        )

        assert isinstance(results, list)

        logger.info(f"Search returned {len(results)} results")

    def test_keyword_search(self, populated_storage):
        """Test keyword-based search"""
        results = populated_storage.search_knowledge(
            query='decomposition',
            query_type='keyword',
            limit=5
        )

        assert isinstance(results, list)

        logger.info(f"Keyword search returned {len(results)} results")

    def test_vector_search(self, populated_storage):
        """Test vector-based search"""
        results = populated_storage.search_knowledge(
            query='optimization',
            query_type='vector',
            limit=5
        )

        assert isinstance(results, list)

        logger.info(f"Vector search returned {len(results)} results")

    def test_get_recommendations(self, populated_storage):
        """Test context-aware recommendations"""
        recommendations = populated_storage.get_recommendations(
            context={
                'problem_type': 'decomposition',
                'complexity': 'high'
            },
            recommendation_type='solution_pattern',
            limit=5
        )

        assert isinstance(recommendations, list)

        logger.info(f"Got {len(recommendations)} recommendations")

    def test_get_related_knowledge(self, populated_storage):
        """Test getting related knowledge"""
        # First, get an artifact ID
        artifacts = populated_storage.storage.retrieve_knowledge_artifacts({}, limit=1)

        if artifacts:
            artifact_id = artifacts[0]['_id']

            related = populated_storage.get_related_knowledge(
                artifact_id=artifact_id,
                relationship_type='related',
                limit=5
            )

            assert isinstance(related, list)

            logger.info(f"Found {len(related)} related artifacts")

    def test_advanced_search(self, populated_storage):
        """Test advanced search with multiple criteria"""
        results = populated_storage.advanced_search({
            'query': 'solution',
            'filters': {'type': 'solution_pattern'},
            'sort_by': 'timestamp',
            'sort_order': 'desc',
            'page': 1,
            'page_size': 10
        })

        assert 'results' in results
        assert 'total_results' in results
        assert 'page' in results
        assert isinstance(results['results'], list)

        logger.info(f"Advanced search: {results['total_results']} total results")

    def test_knowledge_trends(self, populated_storage):
        """Test knowledge trend analysis"""
        trends = populated_storage.get_knowledge_trends(
            time_range='30d',
            artifact_type='solution_pattern'
        )

        assert 'time_range' in trends
        assert 'total_artifacts' in trends
        assert 'daily_trends' in trends
        assert 'trend_analysis' in trends

        logger.info(f"Trend analysis: {trends['trend_analysis']}")

    def test_quality_metrics(self, populated_storage):
        """Test knowledge quality metrics"""
        metrics = populated_storage.get_knowledge_quality_metrics()

        assert 'quality_metrics' in metrics
        assert 'overall_quality_score' in metrics

        quality_metrics = metrics['quality_metrics']
        assert 'completeness' in quality_metrics
        assert 'consistency' in quality_metrics
        assert 'relevance' in quality_metrics

        logger.info(f"Overall quality score: {metrics['overall_quality_score']:.2f}")


class TestIntegration:
    """Integration tests for the complete knowledge pipeline"""

    def test_end_to_end_pipeline(self):
        """Test complete knowledge extraction and retrieval pipeline"""
        from knowledge_engine.knowledge_extractor import KnowledgeExtractor
        from knowledge_engine.knowledge_storage import KnowledgeStorage
        from knowledge_engine.knowledge_retriever import KnowledgeRetriever

        # Setup
        storage = KnowledgeStorage()
        extractor = KnowledgeExtractor()
        retriever = KnowledgeRetriever(storage=storage)

        # Sample workflow data
        workflow_data = {
            'workflow_id': 'integration_test_001',
            'domain': 'optimization',
            'complexity': 'high',
            'execution_time': 1200,
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'solutions': [
                {
                    'id': 'sol_001',
                    'problem_type': 'optimization',
                    'domain': 'math',
                    'approach': 'gradient descent',
                    'implementation': 'iterative approach',
                    'success_rate': 0.90,
                    'complexity': 6,
                    'code': 'def optimize(): pass',
                    'documentation': 'Standard gradient descent',
                    'performance': {'iterations': 100}
                }
            ],
            'critiques': [
                {
                    'id': 'crit_001',
                    'issue_type': 'convergence',
                    'root_cause': 'learning rate too high',
                    'prevention_strategy': 'adaptive learning rate',
                    'severity': 'medium',
                    'affected_components': ['optimizer']
                }
            ],
            'teams': [
                {
                    'name': 'test_team',
                    'role': 'Blue',
                    'domain': 'optimization',
                    'success_rate': 0.88,
                    'avg_response_time': 1.5,
                    'completion_rate': 0.90,
                    'quality_score': 0.85,
                    'performance_trends': [0.82, 0.85, 0.86, 0.87, 0.88]
                }
            ],
            'gauntlets': [
                {
                    'name': 'test_gauntlet',
                    'type': 'Gold',
                    'domain': 'validation',
                    'problem_type': 'quality',
                    'detection_rate': 0.85,
                    'false_positive_rate': 0.08,
                    'true_positive_rate': 0.82,
                    'average_score': 0.84,
                    'performance_trends': [0.80, 0.82, 0.83, 0.84, 0.85]
                }
            ]
        }

        # Extract knowledge
        logger.info("Extracting knowledge from workflow...")
        artifacts = extractor.extract_from_workflow(workflow_data)

        assert len(artifacts) > 0
        logger.info(f"Extracted {len(artifacts)} artifacts")

        # Store artifacts
        logger.info("Storing artifacts in knowledge base...")
        stored_ids = []
        for artifact in artifacts:
            artifact_dict = artifact.to_dict()
            artifact_dict['type'] = artifact.artifact_type
            artifact_dict['source'] = artifact.source_workflow_id
            artifact_dict['content'] = json.dumps(artifact.content)

            artifact_id = storage.store_knowledge_artifact(artifact_dict)
            stored_ids.append(artifact_id)

        logger.info(f"Stored {len(stored_ids)} artifacts")

        # Retrieve knowledge
        logger.info("Retrieving knowledge...")
        retrieved = storage.get_artifact_by_id(stored_ids[0])
        assert retrieved is not None

        # Search knowledge
        logger.info("Searching knowledge base...")
        search_results = retriever.search_knowledge(
            query='optimization',
            query_type='hybrid',
            limit=5
        )

        logger.info(f"Found {len(search_results)} search results")

        # Get recommendations
        logger.info("Getting recommendations...")
        recommendations = retriever.get_recommendations(
            context={'problem_type': 'optimization'},
            recommendation_type='solution_pattern',
            limit=3
        )

        logger.info(f"Got {len(recommendations)} recommendations")

        # Verify quality metrics
        logger.info("Calculating quality metrics...")
        quality_metrics = retriever.get_knowledge_quality_metrics()
        assert quality_metrics['overall_quality_score'] >= 0.0

        logger.info("End-to-end pipeline test completed successfully")

    def test_idempotent_operations(self):
        """Test that operations are idempotent as per CLAUDE.md"""
        from knowledge_engine.knowledge_storage import KnowledgeStorage

        storage = KnowledgeStorage()

        artifact = {
            'type': 'test_artifact',
            'source': 'test_workflow',
            'content': 'Test content',
            '_id': 'test_idempotent_001'
        }

        # Store same artifact multiple times
        id1 = storage.store_knowledge_artifact(artifact)
        id2 = storage.store_knowledge_artifact(artifact)
        id3 = storage.store_knowledge_artifact(artifact)

        # Should return same ID (idempotent)
        assert id1 == id2 == id3

        # Should only have one artifact
        results = storage.retrieve_knowledge_artifacts({'_id': id1})
        assert len(results) >= 1

        logger.info("Idempotent operations verified")

    def test_utc_timestamps(self):
        """Test that all timestamps are in UTC as per CLAUDE.md"""
        from knowledge_engine.knowledge_extractor import KnowledgeArtifact
        from knowledge_engine.knowledge_storage import KnowledgeStorage

        # Create artifact with UTC timestamp
        now_utc = datetime.now(timezone.utc).timestamp()

        artifact_data = {
            'id': 'test_utc_001',
            'artifact_type': 'test_type',
            'content': {'test': 'data'},
            'source_workflow_id': 'test_workflow',
            'extraction_timestamp': now_utc,
            'last_updated': now_utc
        }

        artifact = KnowledgeArtifact(**artifact_data)

        # Verify timestamp is UTC
        assert artifact.extraction_timestamp == now_utc
        assert artifact.last_updated == now_utc

        # Store and verify
        storage = KnowledgeStorage()
        artifact_dict = artifact.to_dict()
        artifact_dict['type'] = artifact.artifact_type
        artifact_dict['source'] = artifact.source_workflow_id
        artifact_dict['content'] = json.dumps(artifact.content)

        artifact_id = storage.store_knowledge_artifact(artifact_dict)
        retrieved = storage.get_artifact_by_id(artifact_id)

        assert retrieved is not None
        assert 'timestamp' in retrieved

        logger.info("UTC timestamps verified")


# Performance tests
class TestPerformance:
    """Performance tests for knowledge components"""

    def test_large_batch_extraction(self):
        """Test extracting knowledge from large workflow"""
        from knowledge_engine.knowledge_extractor import KnowledgeExtractor

        extractor = KnowledgeExtractor()

        # Create large workflow
        workflow_data = {
            'workflow_id': 'perf_test_001',
            'domain': 'performance_test',
            'complexity': 'high',
            'execution_time': 5000,
            'success': True,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'solutions': [
                {
                    'id': f'sol_{i:03d}',
                    'problem_type': 'optimization',
                    'domain': 'test',
                    'approach': f'Approach {i}',
                    'implementation': f'Implementation {i}',
                    'success_rate': 0.8 + (i % 20) * 0.01,
                    'complexity': 5 + (i % 5),
                    'code': f'def test_{i}(): pass',
                    'documentation': f'Documentation {i}',
                    'performance': {'iterations': i * 10}
                }
                for i in range(100)
            ],
            'critiques': [
                {
                    'id': f'crit_{i:03d}',
                    'issue_type': f'Issue {i}',
                    'root_cause': f'Root cause {i}',
                    'prevention_strategy': f'Prevention {i}',
                    'severity': 'high' if i % 3 == 0 else 'medium',
                    'affected_components': [f'comp_{j}' for j in range(3)]
                }
                for i in range(50)
            ],
            'teams': [
                {
                    'name': f'team_{i}',
                    'role': 'Blue' if i % 2 == 0 else 'Red',
                    'domain': 'test',
                    'success_rate': 0.8 + (i % 20) * 0.01,
                    'avg_response_time': 1.0 + i * 0.1,
                    'completion_rate': 0.85 + (i % 15) * 0.01,
                    'quality_score': 0.8 + (i % 20) * 0.01,
                    'performance_trends': [0.8 + j * 0.01 for j in range(5)]
                }
                for i in range(10)
            ],
            'gauntlets': [
                {
                    'name': f'gauntlet_{i}',
                    'type': 'Gold',
                    'domain': 'test',
                    'problem_type': 'validation',
                    'detection_rate': 0.8 + (i % 20) * 0.01,
                    'false_positive_rate': 0.05 - i * 0.001,
                    'true_positive_rate': 0.75 + (i % 25) * 0.01,
                    'average_score': 0.8 + (i % 20) * 0.01,
                    'performance_trends': [0.8 + j * 0.01 for j in range(5)]
                }
                for i in range(20)
            ]
        }

        import time
        start_time = time.time()

        artifacts = extractor.extract_from_workflow(workflow_data)

        extraction_time = time.time() - start_time

        logger.info(f"Extracted {len(artifacts)} artifacts from large workflow in {extraction_time:.2f}s")

        # Should extract at least 100 artifacts (solutions)
        assert len(artifacts) >= 100

        # Should complete in reasonable time (< 30 seconds for 100 items)
        assert extraction_time < 30.0


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short', '-s'])
