#!/usr/bin/env python3
"""
Probe script: Verify Graphiti database connection.

Following CLAUDE.md LAW OF RUNTIME TRUTH:
- Verify actual connectivity before using the integration
- Fail explicitly if the probe doesn't succeed
- This script MUST return 0 for success, non-zero for failure
"""

import sys
import os
import asyncio
import json
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from knowledge_engine.integrations.graphiti.config import GraphitiConfig
from knowledge_engine.integrations.graphiti.exceptions import ConnectionError


async def probe_graphiti_connection() -> bool:
    """
    Probe Graphiti database connection.

    Returns:
        True if connection successful, False otherwise
    """
    print(f"[{datetime.utcnow().isoformat()}] Starting Graphiti connection probe...")

    try:
        # Load configuration from environment
        print("[1/4] Loading configuration from environment...")
        config = GraphitiConfig()
        config.validate()
        print("✓ Configuration loaded and validated")
        print(f"  Provider: {config.graphiti_provider}")
        print(f"  URI: {config.graphiti_uri[:20]}...")
        print(f"  Database: {config.graphiti_database}")

        # Import temporal bridge
        print("\n[2/4] Importing temporal bridge...")
        from knowledge_engine.integrations.graphiti.temporal_bridge import GraphitiTemporalBridge
        print("✓ Temporal bridge imported")

        # Create bridge instance
        print("\n[3/4] Creating bridge instance...")
        bridge = GraphitiTemporalBridge(config=config)
        print("✓ Bridge instance created")

        # Initialize and test connection
        print("\n[4/4] Testing database connection...")
        await bridge.initialize()
        print("✓ Connection successful")

        # Test basic search
        print("\n[Bonus] Testing basic search operation...")
        results = await bridge.search_temporal(
            query="CONNECTION_TEST",
            max_results=1,
        )
        print(f"✓ Search executed successfully (returned {len(results.get('edges', []))} edges)")

        # Cleanup
        await bridge.close()
        print("\n✓ Probe completed successfully")

        return True

    except Exception as e:
        print(f"\n✗ Probe failed: {e}")
        print(f"  Error type: {type(e).__name__}")

        # Print detailed error info
        if isinstance(e, ConnectionError):
            print(f"  Provider: {e.provider}")
            print(f"  Correlation ID: {e.correlation_id}")

        return False


async def main() -> int:
    """
    Main entry point.

    Returns:
        0 for success, 1 for failure
    """
    success = await probe_graphiti_connection()
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
