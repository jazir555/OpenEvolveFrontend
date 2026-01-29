#!/usr/bin/env python3
"""
Probe Script: Check Entity Linking
Task 3.4: Verify advanced entity linking features

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Verify linking operations work
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


async def probe_entity_linking():
    """Probe entity linking functionality."""
    correlation_id = "probe_entity_linking_001"

    logger.info({
        "msg": "Starting entity linking probe",
        "correlation_id": correlation_id
    })

    try:
        linker = CrossLingualEntityLinker()

        # Test 1: Entity deduplication
        logger.info("Test 1: Entity deduplication")
        entities = [
            Entity(entity_id="E1", name_en=["Apple Inc."], name_zh=["苹果公司"], type="ORG"),
            Entity(entity_id="E2", name_en=["Apple"], name_zh=["苹果"], type="ORG"),
            Entity(entity_id="E3", name_en=["Microsoft"], name_zh=["微软"], type="ORG"),
            Entity(entity_id="E4", name_en=["Google"], name_zh=["谷歌"], type="ORG")
        ]

        for entity in entities:
            await linker.add_entity(entity)

        clusters = await linker.deduplicate_entities(entities, MatchStrategy.HYBRID)
        logger.info(f"✓ Found {len(clusters)} duplicate clusters")

        # Test 2: Cross-lingual relation alignment
        logger.info("Test 2: Cross-lingual relation alignment")

        relations1 = [
            {"type": "WORKS_FOR", "head": "Steve Jobs", "tail": "Apple Inc."}
        ]

        relations2 = [
            {"type": "WORKS_FOR", "head": "史蒂夫·乔布斯", "tail": "苹果公司"}
        ]

        alignments = await linker.align_relations(relations1, relations2)
        logger.info(f"✓ Found {len(alignments)} relation alignments")

        # Test 3: Semantic matching
        logger.info("Test 3: Semantic matching")
        entity_a = Entity(
            entity_id="A",
            name_en=["International Business Machines"],
            type="ORGANIZATION"
        )

        entity_b = Entity(
            entity_id="B",
            name_en=["IBM"],
            type="ORGANIZATION"
        )

        semantic_result = await linker.match_entities(entity_a, entity_b, MatchStrategy.SEMANTIC)
        logger.info(f"✓ Semantic match: confidence={semantic_result.confidence}")

        # Test 4: Hybrid strategy
        logger.info("Test 4: Hybrid matching strategy")
        hybrid_result = await linker.match_entities(entity_a, entity_b, MatchStrategy.HYBRID)
        logger.info(f"✓ Hybrid match: matched={hybrid_result.matched}, strategy={hybrid_result.strategy.value}")

        # Test 5: Multiple candidates
        logger.info("Test 5: Find multiple candidates")
        await linker.add_entity(entity_a)
        await linker.add_entity(entity_b)

        candidates = await linker.find_candidates(entity_a, limit=10)
        logger.info(f"✓ Found {len(candidates)} candidates for entity")

        # Test 6: Entity with aliases
        logger.info("Test 6: Entity with aliases")
        entity_aliases = Entity(
            entity_id="E5",
            name_en=["Apple Inc."],
            aliases_en=["Apple", "Apple Computer"],
            name_zh=["苹果公司"],
            aliases_zh=["苹果", "苹果电脑"],
            type="ORGANIZATION"
        )

        await linker.add_entity(entity_aliases)
        all_names = entity_aliases.get_all_names()
        logger.info(f"✓ Entity has {len(all_names['en'])} English names and {len(all_names['zh'])} Chinese names")

        # Test 7: Match result serialization
        logger.info("Test 7: Match result serialization")
        match_result_dict = hybrid_result.to_dict()
        assert "entity1_id" in match_result_dict
        assert "entity2_id" in match_result_dict
        assert "matched" in match_result_dict
        assert "confidence" in match_result_dict
        assert "cross_lingual" in match_result_dict
        logger.info("✓ Match result serialization working")

        logger.info({
            "msg": "Entity linking probe complete",
            "status": "SUCCESS",
            "tests_passed": 7,
            "correlation_id": correlation_id
        })

        return True

    except Exception as e:
        logger.error({
            "msg": "Entity linking probe failed",
            "error": str(e),
            "correlation_id": correlation_id
        })
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(probe_entity_linking())
    sys.exit(0 if success else 1)
