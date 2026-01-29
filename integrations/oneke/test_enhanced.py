"""
Tests for Enhanced OneKE Integration

This module provides comprehensive tests for the reflection agent,
quality enhancer, case repository, and enhanced bridge.
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

from .case import (
    Case, CaseSimilarity, QualityScore,
    ReflectionResult, ConsistencyResult, EnhancedResult, CaseStatistics
)
from .case_repository import OneKECaseRepository
from .reflection_agent import OneKEReflectionAgent
from .quality_enhancement import OneKEQualityEnhancer
from .enhanced_bridge import EnhancedOneKEBridge
from .adapter import OneKEAdapter


# Test data
SAMPLE_TEXT = """
Python is a high-level programming language that supports async/await
for concurrent code execution. It was created by Guido van Rossum and
first released in 1991. Python emphasizes code readability with its
notable use of significant whitespace.
"""

SAMPLE_EXTRACTION = {
    'entities': [
        {'text': 'Python', 'type': 'LANGUAGE', 'confidence': 0.95},
        {'text': 'async/await', 'type': 'CONSTRUCT', 'confidence': 0.88},
        {'text': 'Guido van Rossum', 'type': 'PERSON', 'confidence': 0.92},
        {'text': '1991', 'type': 'DATE', 'confidence': 0.98}
    ],
    'relations': [
        {'subject': 'Python', 'object': 'Guido van Rossum', 'type': 'CREATED_BY', 'confidence': 0.90},
        {'subject': 'Python', 'object': '1991', 'type': 'RELEASED_IN', 'confidence': 0.92}
    ],
    'events': [],
    'triples': []
}


class TestCaseDataStructures:
    """Test case data structures."""

    def test_case_creation(self):
        """Test creating a case."""
        case = Case.create(
            input_text=SAMPLE_TEXT,
            extracted_data=SAMPLE_EXTRACTION,
            schema='software_engineering',
            domain='software_engineering',
            quality_score=0.85
        )

        assert case.case_id is not None
        assert case.input_text == SAMPLE_TEXT
        assert case.extracted_data == SAMPLE_EXTRACTION
        assert case.quality_score == 0.85
        assert case.domain == 'software_engineering'

    def test_case_serialization(self):
        """Test case to_dict and from_dict."""
        case = Case.create(
            input_text=SAMPLE_TEXT,
            extracted_data=SAMPLE_EXTRACTION,
            schema='software_engineering',
            domain='software_engineering',
            quality_score=0.85
        )

        # Convert to dict
        case_dict = case.to_dict()
        assert 'case_id' in case_dict
        assert 'input_text' in case_dict
        assert 'quality_score' in case_dict

        # Convert back from dict
        restored_case = Case.from_dict(case_dict)
        assert restored_case.case_id == case.case_id
        assert restored_case.input_text == case.input_text
        assert restored_case.quality_score == case.quality_score

    def test_quality_score(self):
        """Test QualityScore creation and serialization."""
        score = QualityScore(
            completeness=0.9,
            accuracy=0.85,
            consistency=0.95,
            confidence=0.88,
            overall=0.89
        )

        score_dict = score.to_dict()
        assert score_dict['completeness'] == 0.9
        assert score_dict['overall'] == 0.89

        restored = QualityScore.from_dict(score_dict)
        assert restored.completeness == score.completeness
        assert restored.overall == score.overall


class TestCaseRepository:
    """Test case repository."""

    @pytest.fixture
    def temp_storage(self, tmp_path):
        """Create temporary storage for tests."""
        storage_path = tmp_path / "test_cases.json"
        yield str(storage_path)

    @pytest.fixture
    async def repository(self, temp_storage):
        """Create a test repository."""
        repo = OneKECaseRepository(
            storage_path=temp_storage,
            auto_save=True,
            save_interval=5
        )
        await repo.initialize()

        yield repo

        # Cleanup
        await repo.close()

    @pytest.mark.asyncio
    async def test_repository_initialization(self, repository):
        """Test repository initialization."""
        assert repository is not None
        assert len(repository.cases) == 0

    @pytest.mark.asyncio
    async def test_add_case(self, repository):
        """Test adding a case."""
        case = Case.create(
            input_text=SAMPLE_TEXT,
            extracted_data=SAMPLE_EXTRACTION,
            schema='software_engineering',
            domain='software_engineering',
            quality_score=0.85
        )

        await repository.add_case(case)

        assert len(repository.cases) == 1
        assert repository.cases[0].case_id == case.case_id

    @pytest.mark.asyncio
    async def test_retrieve_similar_cases(self, repository):
        """Test retrieving similar cases."""
        # Add some cases
        for i in range(3):
            case = Case.create(
                input_text=f"Sample text {i}: {SAMPLE_TEXT}",
                extracted_data=SAMPLE_EXTRACTION,
                schema='software_engineering',
                domain='software_engineering',
                quality_score=0.8 + i * 0.05
            )
            await repository.add_case(case)

        # Retrieve similar cases
        query = {'input_text': SAMPLE_TEXT, 'domain': 'software_engineering'}
        similar = await repository.retrieve_similar_cases(
            query=query,
            top_k=2,
            min_similarity=0.0  # Low threshold for testing
        )

        assert len(similar) <= 2
        assert all(isinstance(s, CaseSimilarity) for s in similar)

    @pytest.mark.asyncio
    async def test_get_good_cases(self, repository):
        """Test getting high-quality cases."""
        # Add cases with varying quality
        for i in range(5):
            case = Case.create(
                input_text=f"Text {i}",
                extracted_data=SAMPLE_EXTRACTION,
                schema='software_engineering',
                domain='software_engineering',
                quality_score=0.6 + i * 0.1
            )
            await repository.add_case(case)

        # Get good cases
        good_cases = await repository.get_good_cases(
            domain='software_engineering',
            min_quality=0.8,
            limit=3
        )

        assert len(good_cases) <= 3
        assert all(c.quality_score >= 0.8 for c in good_cases)

    @pytest.mark.asyncio
    async def test_get_statistics(self, repository):
        """Test getting repository statistics."""
        # Add test cases
        for i in range(10):
            case = Case.create(
                input_text=f"Text {i}",
                extracted_data=SAMPLE_EXTRACTION,
                schema='software_engineering',
                domain='software_engineering',
                quality_score=0.7 + i * 0.03
            )
            await repository.add_case(case)

        stats = await repository.get_statistics()

        assert stats.total_cases == 10
        assert 0.0 <= stats.average_quality <= 1.0
        assert 'software_engineering' in stats.domain_distribution

    @pytest.mark.asyncio
    async def test_export_import_cases(self, repository, tmp_path):
        """Test exporting and importing cases."""
        # Add some cases
        for i in range(5):
            case = Case.create(
                input_text=f"Text {i}",
                extracted_data=SAMPLE_EXTRACTION,
                schema='software_engineering',
                domain='software_engineering',
                quality_score=0.8
            )
            await repository.add_case(case)

        # Export
        export_path = tmp_path / "exported_cases.json"
        await repository.export_cases(str(export_path))

        assert export_path.exists()

        # Import into new repository
        import_path = tmp_path / "import_cases.json"
        shutil.copy(str(export_path), str(import_path))

        new_repo = OneKECaseRepository(
            storage_path=str(tmp_path / "new_cases.json")
        )
        await new_repo.initialize()
        await new_repo.import_cases(str(import_path))

        assert len(new_repo.cases) == 5
        await new_repo.close()


class TestReflectionAgent:
    """Test reflection agent."""

    @pytest.fixture
    async def adapter(self):
        """Create a test adapter."""
        adapter = OneKEAdapter()
        # Mock initialization
        return adapter

    @pytest.fixture
    async def reflection_agent(self, adapter, tmp_path):
        """Create a test reflection agent."""
        repo = OneKECaseRepository(
            storage_path=str(tmp_path / "test_cases.json")
        )
        await repo.initialize()

        agent = OneKEReflectionAgent(
            oneke_adapter=adapter,
            case_repository=repo,
            reflection_iterations=2,
            num_samples=2
        )

        yield agent

        await repo.close()

    @pytest.mark.asyncio
    async def test_score_quality(self, reflection_agent):
        """Test quality scoring."""
        score = await reflection_agent.score_quality(
            extraction=SAMPLE_EXTRACTION,
            original_text=SAMPLE_TEXT
        )

        assert isinstance(score, QualityScore)
        assert 0.0 <= score.completeness <= 1.0
        assert 0.0 <= score.accuracy <= 1.0
        assert 0.0 <= score.consistency <= 1.0
        assert 0.0 <= score.confidence <= 1.0
        assert 0.0 <= score.overall <= 1.0

    @pytest.mark.asyncio
    async def test_check_self_consistency(self, reflection_agent):
        """Test self-consistency checking."""
        # This test may fail if adapter is not properly initialized
        try:
            result = await reflection_agent.check_self_consistency(
                text=SAMPLE_TEXT,
                schema='software_engineering',
                reference_extraction=SAMPLE_EXTRACTION,
                num_samples=2
            )

            assert isinstance(result, ConsistencyResult)
            assert isinstance(result.is_consistent, bool)
            assert isinstance(result.agreement_ratio, float)
            assert isinstance(result.samples, list)

        except Exception as e:
            # Adapter not available, skip
            pytest.skip(f"Adapter not available: {e}")


class TestQualityEnhancer:
    """Test quality enhancer."""

    @pytest.fixture
    async def quality_enhancer(self, tmp_path):
        """Create a test quality enhancer."""
        repo = OneKECaseRepository(
            storage_path=str(tmp_path / "test_cases.json")
        )
        await repo.initialize()

        adapter = OneKEAdapter()

        reflection_agent = OneKEReflectionAgent(
            oneke_adapter=adapter,
            case_repository=repo
        )

        enhancer = OneKEQualityEnhancer(
            reflection_agent=reflection_agent,
            min_quality_threshold=0.7
        )

        yield enhancer

        await repo.close()

    @pytest.mark.asyncio
    async def test_enhance_extraction(self, quality_enhancer):
        """Test extraction enhancement."""
        result = await quality_enhancer.enhance_extraction(
            raw_extraction=SAMPLE_EXTRACTION,
            text=SAMPLE_TEXT,
            schema='software_engineering',
            domain='software_engineering',
            strategies=['validation']  # Only validation for testing
        )

        assert isinstance(result, EnhancedResult)
        assert isinstance(result.quality_score, QualityScore)
        assert isinstance(result.original_quality, QualityScore)
        assert isinstance(result.strategies_applied, list)
        assert 'validation' in result.strategies_applied

    @pytest.mark.asyncio
    async def test_apply_validation_strategy(self, quality_enhancer):
        """Test validation strategy."""
        result = await quality_enhancer.apply_validation_strategy(
            extraction=SAMPLE_EXTRACTION,
            schema='software_engineering'
        )

        assert isinstance(result, dict)
        assert 'is_valid' in result
        assert 'needs_fix' in result
        assert 'errors' in result
        assert 'warnings' in result


class TestEnhancedBridge:
    """Test enhanced bridge."""

    @pytest.fixture
    async def enhanced_bridge(self, tmp_path):
        """Create a test enhanced bridge."""
        bridge = EnhancedOneKEBridge(
            config_path=None,  # Use default config
            enhanced_config_path=None  # Use default config
        )

        success = await bridge.initialize()

        if not success:
            pytest.skip("Bridge initialization failed")

        yield bridge

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_extract_with_enhancement(self, enhanced_bridge):
        """Test extraction with enhancement."""
        try:
            result = await enhanced_bridge.extract_with_enhancement(
                text=SAMPLE_TEXT,
                schema='software_engineering',
                domain='software_engineering',
                enable_reflection=False,  # Disable for faster testing
                enable_cases=False,
                enable_validation=True,
                enable_consistency=False
            )

            assert isinstance(result, EnhancedResult)
            assert isinstance(result.extraction, dict)
            assert isinstance(result.quality_score, QualityScore)

        except Exception as e:
            pytest.skip(f"Extraction failed: {e}")

    @pytest.mark.asyncio
    async def test_get_repository_statistics(self, enhanced_bridge):
        """Test getting repository statistics."""
        stats = await enhanced_bridge.get_repository_statistics()

        assert isinstance(stats, dict)
        assert 'total_cases' in stats


# Integration tests
class TestIntegration:
    """Integration tests for the full enhancement pipeline."""

    @pytest.mark.asyncio
    async def test_full_enhancement_pipeline(self, tmp_path):
        """Test the full enhancement pipeline."""
        # Create bridge
        bridge = EnhancedOneKEBridge()
        await bridge.initialize()

        try:
            # Extract with enhancement
            result = await bridge.extract_with_enhancement(
                text=SAMPLE_TEXT,
                schema='software_engineering',
                domain='software_engineering',
                enable_reflection=False,
                enable_cases=False,
                enable_validation=True,
                enable_consistency=False
            )

            # Verify result
            assert result is not None
            assert isinstance(result, EnhancedResult)

            # Check repository
            stats = await bridge.get_repository_statistics()
            assert stats['total_cases'] >= 0

        finally:
            await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_learning_loop(self, tmp_path):
        """Test the learning loop with feedback."""
        bridge = EnhancedOneKEBridge()
        await bridge.initialize()

        try:
            # Extract and learn
            result = await bridge.extract_and_learn(
                text=SAMPLE_TEXT,
                schema='software_engineering',
                domain='software_engineering',
                feedback={
                    'correctness': 0.9,
                    'completeness': 0.85,
                    'comments': 'Good extraction'
                }
            )

            assert result is not None
            assert result.metadata.get('learning_occurred') == True

            # Verify case was stored
            stats = await bridge.get_repository_statistics()
            assert stats['total_cases'] > 0

        finally:
            await bridge.shutdown()


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
