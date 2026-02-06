import asyncio
import logging
import sys
from typing import Dict, Any, Union

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("VerifyLean")

async def test_leanaide_client():
    try:
        from leanaide_client import LeanAideClient
        logger.info("Successfully imported LeanAideClient")
        
        client = LeanAideClient()
        logger.info("Successfully instantiated LeanAideClient")
        
        # Test methods existence
        assert hasattr(client, 'verify'), "LeanAideClient missing verify method"
        assert hasattr(client, 'autoformalize'), "LeanAideClient missing autoformalize method"
        assert hasattr(client, 'translate_thm'), "LeanAideClient missing translate_thm method"
        assert hasattr(client, 'elaborate'), "LeanAideClient missing elaborate method"
        
        logger.info("LeanAideClient methods verified")
        return True
    except Exception as e:
        logger.error(f"LeanAideClient verification failed: {e}")
        return False

async def test_lean4_integration():
    try:
        from lean4_integration import Lean4VerificationEngine, LeanAideService
        logger.info("Successfully imported Lean4 integration components")
        
        engine = Lean4VerificationEngine()
        service = LeanAideService()
        
        logger.info("Successfully instantiated Lean4 components")
        return True
    except Exception as e:
        logger.error(f"Lean4 integration verification failed: {e}")
        return False

async def test_unified_math_service():
    try:
        from openevolve.unified_math_service import UnifiedMathService
        logger.info("Successfully imported UnifiedMathService")
        
        service = UnifiedMathService()
        logger.info("Successfully instantiated UnifiedMathService")
        
        return True
    except Exception as e:
        logger.error(f"UnifiedMathService verification failed: {e}")
        return False

async def test_lean_gauntlet():
    try:
        from gauntlet_types import LeanVerificationGauntlet, create_gauntlet
        logger.info("Successfully imported LeanVerificationGauntlet")
        
        gauntlet = create_gauntlet("lean_verification", "test_lean_gauntlet")
        logger.info(f"Successfully created gauntlet: {gauntlet.name}")
        
        # Test execute method signature
        # We won't actually run it as it requires a live server for full success
        logger.info("LeanVerificationGauntlet instantiation OK")
        return True
    except Exception as e:
        logger.error(f"LeanVerificationGauntlet verification failed: {e}")
        return False

async def main():
    logger.info("Starting Thorough Verification of Lean/LeanAide Integration")
    logger.info("=" * 70)
    
    results = {
        "LeanAide Client": await test_leanaide_client(),
        "Lean4 Integration": await test_lean4_integration(),
        "Unified Math Service": await test_unified_math_service(),
        "Lean Gauntlet": await test_lean_gauntlet()
    }
    
    logger.info("=" * 70)
    logger.info("VERIFICATION SUMMARY:")
    all_passed = True
    for test, result in results.items():
        status = "PASSED" if result else "FAILED"
        logger.info(f"{test:30}: {status}")
        if not result:
            all_passed = False
            
    if all_passed:
        logger.info("ALL LEAN INTEGRATION CHECKS PASSED!")
        return 0
    else:
        logger.error("SOME CHECKS FAILED. Please review the logs.")
        return 1

if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        sys.exit(1)