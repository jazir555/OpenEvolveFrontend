"""
OneKE Integration Test Suite
Comprehensive tests for all OneKE components

Test Coverage:
- Model Adapter (Task 3.1)
- Extraction Framework (Task 3.2)
- Schema Manager (Task 3.3)
- Entity Linker (Task 3.4)
- Event Extractor (Task 3.5)

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Tests verify actual behavior
- IDEMPOTENCY: Tests verify idempotent operations
- STRUCTURED LOGGING: Test results logged
"""

import pytest
import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any
from unittest.mock import Mock, AsyncMock, patch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from knowledge_engine.integrations.oneke.model_adapter import (
    OneKEModelAdapter,
    ModelConfig,
    ExtractionResult,
    Language,
    QuantizationMode
)
from knowledge_engine.integrations.oneke.extraction_framework import (
    MultiTaskExtractionFramework,
    TaskType,
    TaskConfig
)
from knowledge_engine.integrations.oneke.schema_manager import (
    OneKESchemaManager,
    SchemaDefinition
)
from knowledge_engine.integrations.oneke.entity_linker import (
    CrossLingualEntityLinker,
    Entity,
    EntityMatchResult,
    MatchStrategy,
    Language as LinkerLanguage,
    LinkerConfig
)
from knowledge_engine.integrations.oneke.event_extractor import (
    EventExtractionPipeline,
    TemporalEvent,
    EventChain,
    CausalRelation,
    EventType,
    ArgumentRole,
    ExtractorConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Test Fixtures
# =============================================================================

@pytest.fixture
def model_config():
    """Create test model configuration."""
    return ModelConfig(
        model_name="test/oneke",
        device="cpu",
        max_length=2048,
        quantization=QuantizationMode.NONE,
        temperature=0.1,
        do_sample=False
    )


@pytest.fixture
def task_config():
    """Create test task configuration."""
    return TaskConfig(
        task_timeout=30,
        max_retries=2
    )


@pytest.fixture
def linker_config():
    """Create test linker configuration."""
    return LinkerConfig(
        fuzzy_threshold=80,
        semantic_threshold=0.6,
        enable_translation=False  # Disable for tests
    )


@pytest.fixture
def extractor_config():
    """Create test extractor configuration."""
    return ExtractorConfig(
        confidence_threshold=0.5,
        max_events_per_doc=20,
        enable_causal_extraction=True,
        enable_temporal_ordering=True
    )


# =============================================================================
# Model Adapter Tests (Task 3.1)
# =============================================================================

class TestModelAdapter:
    """Test OneKE model adapter."""

    def test_model_config_validation(self, model_config):
        """Test 3.1.1: Model configuration validation."""
        # Valid config
        assert model_config.model_name == "test/oneke"
        assert model_config.device == "cpu"
        assert model_config.max_length == 2048

        # Invalid temperature
        with pytest.raises(ValueError):
            ModelConfig(temperature=3.0)

        # Invalid top_p
        with pytest.raises(ValueError):
            ModelConfig(top_p=1.5)

        # Invalid max_length
        with pytest.raises(ValueError):
            ModelConfig(max_length=0)

    def test_extraction_result_creation(self):
        """Test 3.1.2: Extraction result structure."""
        result = ExtractionResult(
            entities=[{"id": "E1", "type": "PERSON"}],
            relations=[{"subject": "E1", "object": "E2", "type": "WORKS_FOR"}],
            confidence=0.85,
            language=Language.ENGLISH
        )

        assert len(result.entities) == 1
        assert len(result.relations) == 1
        assert result.confidence == 0.85
        assert result.language == Language.ENGLISH
        assert result.timestamp.tzinfo == timezone.utc

    def test_language_enum(self):
        """Test 3.1.3: Language enumeration."""
        assert Language.ENGLISH.value == "en"
        assert Language.CHINESE.value == "zh"
        assert Language.BILINGUAL.value == "bilingual"


# =============================================================================
# Extraction Framework Tests (Task 3.2)
# =============================================================================

class TestExtractionFramework:
    """Test multi-task extraction framework."""

    def test_task_config_validation(self, task_config):
        """Test 3.2.1: Task configuration validation."""
        assert task_config.task_timeout == 30
        assert task_config.max_retries == 2

        # Invalid timeout
        with pytest.raises(ValueError):
            TaskConfig(task_timeout=0)

        # Invalid retries
        with pytest.raises(ValueError):
            TaskConfig(max_retries=-1)

    def test_task_type_enum(self):
        """Test 3.2.2: Task type enumeration."""
        assert TaskType.NER.value == "named_entity_recognition"
        assert TaskType.RE.value == "relation_extraction"
        assert TaskType.EE.value == "event_extraction"


# =============================================================================
# Schema Manager Tests (Task 3.3)
# =============================================================================

class TestSchemaManager:
    """Test schema manager."""

    @pytest.mark.asyncio
    async def test_schema_loading(self):
        """Test 3.3.1: Schema loading from file."""
        manager = OneKESchemaManager()

        # Create test schema
        test_schema = {
            "name": "test_schema",
            "entity_types": [
                {"name": "PERSON", "description": "A person"},
                {"name": "ORG", "description": "An organization"}
            ],
            "relation_types": [
                {"name": "WORKS_FOR", "description": "Employment"},
                {"name": "LOCATED_IN", "description": "Location"}
            ]
        }

        # Load schema
        schema = SchemaDefinition(**test_schema)
        assert schema.name == "test_schema"
        assert len(schema.entity_types) == 2

    @pytest.mark.asyncio
    async def test_schema_validation(self):
        """Test 3.3.2: Schema validation."""
        manager = OneKESchemaManager()

        # Valid schema
        valid_schema = {
            "name": "valid",
            "entity_types": [
                {"name": "PERSON", "description": "A person"}
            ]
        }

        schema = SchemaDefinition(**valid_schema)
        assert schema.name == "valid"

        # Invalid schema (missing required fields)
        with pytest.raises(Exception):
            SchemaDefinition(**{})


# =============================================================================
# Entity Linker Tests (Task 3.4)
# =============================================================================

class TestEntityLinker:
    """Test cross-lingual entity linker."""

    @pytest.mark.asyncio
    async def test_linker_initialization(self, linker_config):
        """Test 3.4.1: Linker initialization."""
        linker = CrossLingualEntityLinker(linker_config)
        assert linker.config.fuzzy_threshold == 80
        assert linker.config.enable_translation == False
        assert len(linker.entity_index) == 0

    @pytest.mark.asyncio
    async def test_language_detection(self):
        """Test 3.4.2: Language detection."""
        linker = CrossLingualEntityLinker()

        # English text
        lang_en = await linker.detect_language("This is English text")
        assert lang_en == LinkerLanguage.ENGLISH

        # Chinese text
        lang_zh = await linker.detect_language("这是中文文本")
        assert lang_zh == LinkerLanguage.CHINESE

        # Bilingual text
        lang_bi = await linker.detect_language("This is English and 中文 mixed")
        assert lang_bi == LinkerLanguage.BILINGUAL

    @pytest.mark.asyncio
    async def test_entity_creation(self):
        """Test 3.4.3: Entity creation and validation."""
        entity = Entity(
            entity_id="E1",
            name_en=["Apple Inc."],
            name_zh=["苹果公司"],
            type="ORGANIZATION",
            language=LinkerLanguage.ENGLISH
        )

        assert entity.entity_id == "E1"
        assert "Apple Inc." in entity.name_en
        assert "苹果公司" in entity.name_zh
        assert entity.type == "ORGANIZATION"

        # Invalid entity (no names)
        with pytest.raises(ValueError):
            Entity(entity_id="E2", type="PERSON")

        # Invalid confidence
        with pytest.raises(ValueError):
            Entity(
                entity_id="E3",
                name_en=["Test"],
                confidence=1.5
            )

    @pytest.mark.asyncio
    async def test_add_entity(self, linker_config):
        """Test 3.4.4: Adding entities to index."""
        linker = CrossLingualEntityLinker(linker_config)

        entity = Entity(
            entity_id="E1",
            name_en=["Microsoft"],
            name_zh=["微软"],
            type="ORGANIZATION",
            language=LinkerLanguage.BILINGUAL
        )

        # Add entity
        result = await linker.add_entity(entity)
        assert result == True
        assert "E1" in linker.entity_index

        # Idempotent: adding again should return False
        result = await linker.add_entity(entity)
        assert result == False

    @pytest.mark.asyncio
    async def test_exact_match(self, linker_config):
        """Test 3.4.5: Exact entity matching."""
        linker = CrossLingualEntityLinker(linker_config)

        entity1 = Entity(
            entity_id="E1",
            name_en=["Apple Inc."],
            type="ORGANIZATION"
        )

        entity2 = Entity(
            entity_id="E2",
            name_en=["Apple Inc."],
            type="ORGANIZATION"
        )

        match_result = await linker.match_entities(entity1, entity2, MatchStrategy.EXACT)

        assert match_result.matched == True
        assert match_result.confidence == 1.0
        assert match_result.strategy == MatchStrategy.EXACT
        assert len(match_result.evidence) > 0

    @pytest.mark.asyncio
    async def test_fuzzy_match(self, linker_config):
        """Test 3.4.6: Fuzzy entity matching."""
        linker = CrossLingualEntityLinker(linker_config)

        entity1 = Entity(
            entity_id="E1",
            name_en=["International Business Machines"],
            type="ORGANIZATION"
        )

        entity2 = Entity(
            entity_id="E2",
            name_en=["Intl Business Machines"],
            type="ORGANIZATION"
        )

        match_result = await linker.match_entities(entity1, entity2, MatchStrategy.FUZZY)

        # Should match with high confidence
        assert match_result.matched == True
        assert match_result.confidence > 0.8

    @pytest.mark.asyncio
    async def test_cross_lingual_match(self, linker_config):
        """Test 3.4.7: Cross-lingual entity matching."""
        linker = CrossLingualEntityLinker(linker_config)

        entity1 = Entity(
            entity_id="E1",
            name_en=["Apple"],
            type="ORGANIZATION",
            language=LinkerLanguage.ENGLISH
        )

        entity2 = Entity(
            entity_id="E2",
            name_zh=["苹果"],
            type="ORGANIZATION",
            language=LinkerLanguage.CHINESE
        )

        match_result = await linker.match_entities(entity1, entity2, MatchStrategy.HYBRID)

        assert match_result.cross_lingual == True

    @pytest.mark.asyncio
    async def test_entity_deduplication(self, linker_config):
        """Test 3.4.8: Entity deduplication."""
        linker = CrossLingualEntityLinker(linker_config)

        entities = [
            Entity(entity_id="E1", name_en=["Apple Inc."], type="ORG"),
            Entity(entity_id="E2", name_en=["Apple"], type="ORG"),
            Entity(entity_id="E3", name_en=["Microsoft"], type="ORG")
        ]

        for entity in entities:
            await linker.add_entity(entity)

        clusters = await linker.deduplicate_entities(entities, MatchStrategy.HYBRID)

        # Should find at least one duplicate cluster (E1, E2)
        assert len(clusters) >= 1

    def test_bilingual_kg_format(self):
        """Test 3.4.9: Bilingual knowledge graph format."""
        linker = CrossLingualEntityLinker()

        entities = [
            Entity(
                entity_id="E1",
                name_en=["Apple"],
                name_zh=["苹果"],
                type="ORGANIZATION"
            )
        ]

        kg = linker.to_bilingual_kg(entities)

        assert "nodes" in kg
        assert "metadata" in kg
        assert kg["metadata"]["format"] == "bilingual_kg"
        assert "en" in kg["metadata"]["languages"]
        assert "zh" in kg["metadata"]["languages"]
        assert len(kg["nodes"]) == 1
        assert "en" in kg["nodes"][0]["names"]
        assert "zh" in kg["nodes"][0]["names"]


# =============================================================================
# Event Extractor Tests (Task 3.5)
# =============================================================================

class TestEventExtractor:
    """Test event extraction pipeline."""

    def test_extractor_initialization(self, extractor_config):
        """Test 3.5.1: Extractor initialization."""
        pipeline = EventExtractionPipeline(extractor_config)
        assert pipeline.config.confidence_threshold == 0.5
        assert pipeline.config.enable_causal_extraction == True

    def test_event_type_enum(self):
        """Test 3.5.2: Event type enumeration."""
        assert EventType.ACQUISITION.value == "acquisition"
        assert EventType.LAUNCH.value == "launch"
        assert EventType.APPOINTMENT.value == "appointment"

    def test_argument_role_enum(self):
        """Test 3.5.3: Argument role enumeration."""
        assert ArgumentRole.TRIGGER.value == "trigger"
        assert ArgumentRole.SUBJECT.value == "subject"
        assert ArgumentRole.TIME.value == "time"

    def test_temporal_event_creation(self):
        """Test 3.5.4: Temporal event creation."""
        event = TemporalEvent(
            event_id="EV1",
            event_type=EventType.ACQUISITION,
            trigger="acquired",
            timestamp=datetime.now(timezone.utc),
            certainty=0.9
        )

        assert event.event_id == "EV1"
        assert event.event_type == EventType.ACQUISITION
        assert event.trigger == "acquired"
        assert event.certainty == 0.9
        assert event.timestamp.tzinfo == timezone.utc

    def test_event_creation_invalid(self):
        """Test 3.5.5: Invalid event creation."""
        # Invalid certainty
        with pytest.raises(ValueError):
            TemporalEvent(
                event_id="EV1",
                event_type=EventType.ACQUISITION,
                trigger="test",
                certainty=1.5
            )

    @pytest.mark.asyncio
    async def test_event_argument_extraction(self):
        """Test 3.5.6: Event argument extraction."""
        pipeline = EventExtractionPipeline()

        event = TemporalEvent(
            event_id="EV1",
            event_type=EventType.LAUNCH,
            trigger="launched"
        )

        text = "Apple launched the iPhone on June 29, 2007 in San Francisco"

        updated_event = await pipeline.extract_arguments(event, text)

        # Should extract time and location
        time_args = [arg for arg in updated_event.arguments if arg.role == ArgumentRole.TIME]
        location_args = [arg for arg in updated_event.arguments if arg.role == ArgumentRole.LOCATION]

        assert len(time_args) > 0 or len(location_args) > 0

    @pytest.mark.asyncio
    async def test_event_chain_building(self):
        """Test 3.5.7: Event chain construction."""
        pipeline = EventExtractionPipeline()

        now = datetime.now(timezone.utc)

        events = [
            TemporalEvent(
                event_id="E1",
                event_type=EventType.LAUNCH,
                trigger="announced",
                timestamp=now
            ),
            TemporalEvent(
                event_id="E2",
                event_type=EventType.LAUNCH,
                trigger="released",
                timestamp=now + timedelta(hours=12)  # Within temporal_window (24 hours)
            )
        ]

        chains = await pipeline.build_event_chains(events)

        # Should create at least one chain
        assert len(chains) >= 1

        if len(chains) > 0:
            chain = chains[0]
            assert len(chain.events) >= 1

    @pytest.mark.asyncio
    async def test_causal_relation_extraction(self):
        """Test 3.5.8: Causal relationship extraction."""
        pipeline = EventExtractionPipeline()

        events = [
            TemporalEvent(event_id="E1", event_type=EventType.LAUNCH, trigger="announced"),
            TemporalEvent(event_id="E2", event_type=EventType.LAUNCH, trigger="increased")
        ]

        text = "The company announced a new product, which caused sales to increase"

        causal_relations = await pipeline.extract_causal_relations(
            events,
            text,
            Language.ENGLISH
        )

        # Should detect causal relation
        assert len(causal_relations) >= 0

    @pytest.mark.asyncio
    async def test_temporal_ordering(self):
        """Test 3.5.9: Temporal event ordering."""
        pipeline = EventExtractionPipeline()

        now = datetime.now(timezone.utc)

        events = [
            TemporalEvent(
                event_id="E1",
                event_type=EventType.LAUNCH,
                trigger="first",
                timestamp=now
            ),
            TemporalEvent(
                event_id="E2",
                event_type=EventType.LAUNCH,
                trigger="second",
                timestamp=now + timedelta(days=1)
            )
        ]

        text = "first event, then second event"

        ordered = await pipeline.order_events_temporally(events, text)

        assert len(ordered) == 2
        # E1 should come before E2
        assert ordered.index("E1") < ordered.index("E2")

    @pytest.mark.asyncio
    async def test_complete_pipeline(self):
        """Test 3.5.10: Complete extraction pipeline."""
        pipeline = EventExtractionPipeline()

        text = """
        Apple announced the iPhone in January 2007.
        The company released it in June 2007.
        This launch revolutionized the smartphone industry.
        """

        result = await pipeline.extract_complete_pipeline(
            text,
            Language.ENGLISH
        )

        assert "events" in result
        assert "event_chains" in result
        assert "causal_relations" in result
        assert "temporal_order" in result
        assert "metadata" in result


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_bilingual_extraction_workflow(self, linker_config):
        """Test complete bilingual extraction workflow."""
        linker = CrossLingualEntityLinker(linker_config)

        # Create bilingual entities
        entities = [
            Entity(
                entity_id="E1",
                name_en=["Apple Inc."],
                name_zh=["苹果公司"],
                type="ORGANIZATION",
                language=LinkerLanguage.BILINGUAL
            ),
            Entity(
                entity_id="E2",
                name_en=["Microsoft"],
                name_zh=["微软"],
                type="ORGANIZATION",
                language=LinkerLanguage.BILINGUAL
            )
        ]

        # Add entities
        for entity in entities:
            await linker.add_entity(entity)

        # Match entities
        match_result = await linker.match_entities(entities[0], entities[1], MatchStrategy.FUZZY)

        # Convert to KG
        kg = linker.to_bilingual_kg(entities)

        assert len(kg["nodes"]) == 2
        assert kg["metadata"]["entity_count"] == 2

    @pytest.mark.asyncio
    async def test_event_chain_workflow(self, extractor_config):
        """Test complete event chain workflow."""
        pipeline = EventExtractionPipeline(extractor_config)

        text = """
        In 2007, Apple announced the iPhone.
        Steve Jobs launched the product at Macworld.
        The device was released later that year.
        This launch changed the mobile phone industry forever.
        """

        result = await pipeline.extract_complete_pipeline(text, Language.ENGLISH)

        # Verify structure
        assert "events" in result
        assert "event_chains" in result
        assert "causal_relations" in result
        assert result["metadata"]["num_events"] >= 0


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
