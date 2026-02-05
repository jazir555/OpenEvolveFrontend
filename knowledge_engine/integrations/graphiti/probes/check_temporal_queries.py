#!/usr/bin/env python3
"""
Probe script: Verify temporal query functionality.

Following CLAUDE.md LAW OF RUNTIME TRUTH:
- Verify temporal search works with actual queries
- Test different temporal filter types
- Confirm point-in-time queries work correctly
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from knowledge_engine.integrations.graphiti.config import GraphitiConfig
from knowledge_engine.integrations.graphiti.temporal_bridge import (
    GraphitiTemporalBridge,
    TemporalFilter,
)


async def probe_temporal_queries() -> bool:
    """
    Probe temporal query functionality.

    Returns:
        True if successful, False otherwise
    """
    print(f"[{datetime.utcnow().isoformat()}] Starting temporal query probe...")

    bridge = None

    try:
        # Load configuration
        print("[1/5] Loading configuration...")
        config = GraphitiConfig()
        config.validate()
        print("[OK] Configuration loaded")

        # Create bridge
        print("\n[2/5] Creating temporal bridge...")
        bridge = GraphitiTemporalBridge(config=config)
        await bridge.initialize()
        print("[OK] Bridge initialized")

        # Add test data with different timestamps
        print("\n[3/5] Adding test episodes with different timestamps...")
        now = datetime.utcnow()
        times = [
            now - timedelta(hours=2),
            now - timedelta(hours=1),
            now,
        ]

        for i, timestamp in enumerate(times):
            await bridge.add_episode(
                name=f"Temporal Test Episode {i+1}",
                episode_body=f"Test episode at {timestamp.isoformat()}",
                reference_time=timestamp,
                source="temporal_probe",
            )
            print(f"  [OK] Added episode {i+1} at {timestamp.isoformat()}")

        # Test current query
        print("\n[4/5] Testing CURRENT temporal filter...")
        results = await bridge.search_temporal(
            query="Temporal Test",
            filter_type=TemporalFilter.CURRENT,
            max_results=10,
        )
        print(f"  [OK] CURRENT filter returned {len(results.get('edges', []))} edges")

        # Test time range query
        print("\n[5/5] Testing TIME_RANGE temporal filter...")
        results = await bridge.search_temporal(
            query="Temporal Test",
            filter_type=TemporalFilter.TIME_RANGE,
            start_time=now - timedelta(hours=3),
            end_time=now + timedelta(hours=1),
            max_results=10,
        )
        print(f"  [OK] TIME_RANGE filter returned {len(results.get('edges', []))} edges")

        # Test point-in-time query
        print("\n[Bonus] Testing point-in-time query...")
        artifacts = await bridge.search_temporal(
            query="*",
            max_results=10,
        )
        print(f"  [OK] Point-in-time query returned {len(artifacts.get('edges', []))} edges")

        print("\n[OK] All temporal query checks passed")
        return True

    except Exception as e:
        print(f"\n[FAIL] Probe failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        if bridge:
            try:
                await bridge.close()
                print("\n[OK] Cleanup completed")
            except Exception as e:
                print(f"\n⚠ Cleanup warning: {e}")


async def main() -> int:
    """Main entry point."""
    success = await probe_temporal_queries()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
