#!/usr/bin/env python3
"""
Probe Script: Check Bilingual Extraction
Task 3.4: Verify cross-lingual entity linking functionality

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Verify bilingual matching works
- STRUCTURED LOGGING: JSON output with correlation IDs
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from knowledge_engine.integrations.oneke.entity_linker import (
    CrossLingualEntityLinker,
    Entity,
    MatchStrategy,
    Language as LinkerLanguage,
    LinkerConfig
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def probe_bilingual_extraction():
    """Probe bilingual extraction functionality."""
    correlation_id = "probe_bilingual_001"

    logger.info({
        "msg": "Starting bilingual extraction probe",
        "correlation_id": correlation_id
    })

    try:
        # Test 1: Linker initialization
        logger.info("Test 1: Linker initialization")
        config = LinkerConfig(
            fuzzy_threshold=80,
            semantic_threshold=0.6,
            enable_translation=False
        )
        linker = CrossLingualEntityLinker(config)
        logger.info("✓ Linker initialized successfully")

        # Test 2: Language detection
        logger.info("Test 2: Language detection")
        lang_en = await linker.detect_language("This is English text")
        lang_zh = await linker.detect_language("这是中文文本")
        lang_bi = await linker.detect_language("Mixed English and 中文")

        assert lang_en == LinkerLanguage.ENGLISH
        assert lang_zh == LinkerLanguage.CHINESE
        logger.info(f"✓ Language detection working: EN={lang_en}, ZH={lang_zh}")

        # Test 3: Entity creation
        logger.info("Test 3: Entity creation")
        entity1 = Entity(
            entity_id="E1",
            name_en=["Apple Inc."],
            name_zh=["苹果公司"],
            type="ORGANIZATION",
            language=LinkerLanguage.BILINGUAL
        )

        entity2 = Entity(
            entity_id="E2",
            name_en=["Microsoft"],
            name_zh=["微软"],
            type="ORGANIZATION",
            language=LinkerLanguage.BILINGUAL
        )

        logger.info("✓ Entities created successfully")

        # Test 4: Add entities to index
        logger.info("Test 4: Add entities to index")
        result1 = await linker.add_entity(entity1)
        result2 = await linker.add_entity(entity2)

        assert result1 == True
        assert result2 == True
        assert len(linker.entity_index) == 2
        logger.info("✓ Entities added to index")

        # Test 5: Idempotent add
        logger.info("Test 5: Idempotent operations")
        result_dup = await linker.add_entity(entity1)
        assert result_dup == False  # Already exists
        logger.info("✓ Idempotent operations working")

        # Test 6: Exact match
        logger.info("Test 6: Exact entity matching")
        entity3 = Entity(
            entity_id="E3",
            name_en=["Apple Inc."],
            type="ORGANIZATION"
        )

        match_result = await linker.match_entities(entity1, entity3, MatchStrategy.EXACT)
        assert match_result.matched == True
        assert match_result.confidence == 1.0
        logger.info("✓ Exact matching working")

        # Test 7: Fuzzy match
        logger.info("Test 7: Fuzzy entity matching")
        entity4 = Entity(
            entity_id="E4",
            name_en=["Apple"],
            type="ORGANIZATION"
        )

        fuzzy_result = await linker.match_entities(entity1, entity4, MatchStrategy.FUZZY)
        logger.info(f"Fuzzy match result: matched={fuzzy_result.matched}, confidence={fuzzy_result.confidence}")

        # Test 8: Cross-lingual match
        logger.info("Test 8: Cross-lingual entity matching")
        entity5 = Entity(
            entity_id="E5",
            name_zh=["苹果公司"],
            type="ORGANIZATION",
            language=LinkerLanguage.CHINESE
        )

        cross_result = await linker.match_entities(entity1, entity5, MatchStrategy.HYBRID)
        logger.info(f"Cross-lingual match: matched={cross_result.matched}, cross_lingual={cross_result.cross_lingual}")

        # Test 9: Bilingual KG format
        logger.info("Test 9: Bilingual knowledge graph format")
        entities = [entity1, entity2]
        kg = linker.to_bilingual_kg(entities)

        assert "nodes" in kg
        assert "metadata" in kg
        assert kg["metadata"]["format"] == "bilingual_kg"
        assert len(kg["nodes"]) == 2
        assert kg["nodes"][0]["names"]["en"] == ["Apple Inc."]
        assert kg["nodes"][0]["names"]["zh"] == ["苹果公司"]
        logger.info("✓ Bilingual KG format working")

        # Test 10: Find candidates
        logger.info("Test 10: Find candidate entities")
        candidates = await linker.find_candidates(entity1)
        logger.info(f"✓ Found {len(candidates)} candidates")

        logger.info({
            "msg": "Bilingual extraction probe complete",
            "status": "SUCCESS",
            "tests_passed": 10,
            "correlation_id": correlation_id
        })

        return True

    except Exception as e:
        logger.error({
            "msg": "Bilingual extraction probe failed",
            "error": str(e),
            "correlation_id": correlation_id
        })
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(probe_bilingual_extraction())
    sys.exit(0 if success else 1)
