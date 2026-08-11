#!/usr/bin/env python
"""
One-Click Integration Initialization

This script initializes all integration components for the OpenEvolve Frontend.
Call this during system startup to wire everything together.

Usage:
    python initialize_integrations.py
"""


import logging
import sys
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main initialization function."""
    print("\n" + "=" * 70)
    print("OPENEVOLVE FRONTEND - INTEGRATION INITIALIZATION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    try:
        # Import master integration system
        from master_integration_system import initialize_all_integrations

        # Initialize all integrations
        print("Initializing all integration systems...")
        print()

        results = initialize_all_integrations()

        # Report results
        print()
        print("=" * 70)
        print("INITIALIZATION RESULTS")
        print("=" * 70)

        for component, success in results.items():
            status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
            print(f"  {status}: {component}")

        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)

        print()
        print("=" * 70)
        print(f"SUMMARY: {successful}/{total} components initialized")
        print("=" * 70)

        if successful == total:
            print()
            print("🎉 ALL INTEGRATIONS INITIALIZED SUCCESSFULLY! 🎉")
            print()
            print("Next steps:")
            print("  1. Use master_integration_system.get_master_system() to access the system")
            print("  2. Check system health: system.get_system_health()")
            print("  3. Verify components: system.verify_component_state()")
            print("  4. Optimize components: system.optimize_component()")
            print()
            return 0
        else:
            print()
            print("[WARN]  Some components failed to initialize")
            print("   Check the logs above for details")
            print()
            return 1

    except Exception as e:
        logger.error(f"Initialization failed: {e}", exc_info=True)
        print()
        print("=" * 70)
        print(f"[FAIL] INITIALIZATION FAILED: {e}")
        print("=" * 70)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
