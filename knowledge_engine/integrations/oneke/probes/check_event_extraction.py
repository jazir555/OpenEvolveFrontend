#!/usr/bin/env python3
"""
Probe Script: Check Event Extraction
Task 3.5: Verify event extraction functionality

Following CLAUDE.md Principles:
- RUNTIME TRUTH: Verify event extraction works
- STRUCTURED LOGGING: JSON output with correlation IDs
"""

import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

from knowledge_engine.integrations.oneke.event_extractor import (
    EventExtractionPipeline,
    TemporalEvent,
    EventChain,
    EventType,
    ArgumentRole,
    CausalType,
    ExtractorConfig,
    Language
)

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def probe_event_extraction():
    """Probe event extraction functionality."""
    correlation_id = "probe_event_extraction_001"

    logger.info({
        "msg": "Starting event extraction probe",
        "correlation_id": correlation_id
    })

    try:
        # Test 1: Pipeline initialization
        logger.info("Test 1: Pipeline initialization")
        config = ExtractorConfig(
            confidence_threshold=0.5,
            max_events_per_doc=20,
            enable_causal_extraction=True
        )
        pipeline = EventExtractionPipeline(config)
        logger.info("[OK] Pipeline initialized successfully")

        # Test 2: Event type enum
        logger.info("Test 2: Event type enumeration")
        assert EventType.ACQUISITION.value == "acquisition"
        assert EventType.LAUNCH.value == "launch"
        logger.info("[OK] Event types working")

        # Test 3: Argument role enum
        logger.info("Test 3: Argument role enumeration")
        assert ArgumentRole.SUBJECT.value == "subject"
        assert ArgumentRole.TIME.value == "time"
        logger.info("[OK] Argument roles working")

        # Test 4: Temporal event creation
        logger.info("Test 4: Temporal event creation")
        event = TemporalEvent(
            event_id="EV1",
            event_type=EventType.LAUNCH,
            trigger="launched",
            timestamp=datetime.now(timezone.utc),
            certainty=0.9
        )

        assert event.event_id == "EV1"
        assert event.event_type == EventType.LAUNCH
        assert event.timestamp.tzinfo == timezone.utc
        logger.info("[OK] Temporal event creation working")

        # Test 5: Event serialization
        logger.info("Test 5: Event serialization")
        event_dict = event.to_dict()
        assert "event_id" in event_dict
        assert "event_type" in event_dict
        assert "trigger" in event_dict
        assert "timestamp" in event_dict
        logger.info("[OK] Event serialization working")

        # Test 6: Event chain creation
        logger.info("Test 6: Event chain creation")
        chain = EventChain(chain_id="chain_1")

        now = datetime.now(timezone.utc)
        event1 = TemporalEvent(
            event_id="E1",
            event_type=EventType.LAUNCH,
            trigger="announced",
            timestamp=now
        )

        event2 = TemporalEvent(
            event_id="E2",
            event_type=EventType.LAUNCH,
            trigger="released",
            timestamp=now + timedelta(days=30)
        )

        chain.add_event(event1)
        chain.add_event(event2)
        assert len(chain.events) == 2
        assert len(chain.temporal_order) == 2
        logger.info("[OK] Event chain creation working")

        # Test 7: Event argument extraction
        logger.info("Test 7: Event argument extraction")
        text = "Apple launched the iPhone on June 29, 2007 in San Francisco"

        event3 = TemporalEvent(
            event_id="E3",
            event_type=EventType.LAUNCH,
            trigger="launched"
        )

        updated_event = await pipeline.extract_arguments(event3, text)
        logger.info(f"[OK] Extracted {len(updated_event.arguments)} arguments")

        # Test 8: Event chain building
        logger.info("Test 8: Event chain building")
        events = [event1, event2]
        chains = await pipeline.build_event_chains(events)
        logger.info(f"[OK] Built {len(chains)} event chains")

        # Test 9: Causal relation extraction
        logger.info("Test 9: Causal relation extraction")
        text_causal = "The announcement caused stock prices to rise significantly"

        causal_events = [
            TemporalEvent(event_id="E1", event_type=EventType.LAUNCH, trigger="announcement"),
            TemporalEvent(event_id="E2", event_type=EventType.LAUNCH, trigger="rise")
        ]

        causal_relations = await pipeline.extract_causal_relations(
            causal_events,
            text_causal,
            Language.ENGLISH
        )
        logger.info(f"[OK] Extracted {len(causal_relations)} causal relations")

        # Test 10: Temporal ordering
        logger.info("Test 10: Temporal ordering")
        text_order = "First event happened, then second event occurred"

        ordered = await pipeline.order_events_temporally(events, text_order)
        assert len(ordered) == 2
        logger.info(f"[OK] Temporal ordering: {ordered}")

        # Test 11: Complete pipeline
        logger.info("Test 11: Complete extraction pipeline")
        sample_text = """
        In January 2007, Apple announced the iPhone.
        Steve Jobs introduced the device at Macworld.
        The iPhone was released in June 2007.
        This launch revolutionized the smartphone industry.
        Sales increased dramatically after the release.
        """

        result = await pipeline.extract_complete_pipeline(sample_text, Language.ENGLISH)

        assert "events" in result
        assert "event_chains" in result
        assert "causal_relations" in result
        assert "temporal_order" in result
        assert "metadata" in result

        logger.info(f"[OK] Complete pipeline: {result['metadata']['num_events']} events, "
                   f"{result['metadata']['num_chains']} chains")

        # Test 12: Event chain serialization
        logger.info("Test 12: Event chain serialization")
        chain_dict = chain.to_dict()
        assert "chain_id" in chain_dict
        assert "events" in chain_dict
        assert "temporal_order" in chain_dict
        logger.info("[OK] Event chain serialization working")

        logger.info({
            "msg": "Event extraction probe complete",
            "status": "SUCCESS",
            "tests_passed": 12,
            "correlation_id": correlation_id
        })

        return True

    except Exception as e:
        logger.error({
            "msg": "Event extraction probe failed",
            "error": str(e),
            "correlation_id": correlation_id
        })
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(probe_event_extraction())
    sys.exit(0 if success else 1)
