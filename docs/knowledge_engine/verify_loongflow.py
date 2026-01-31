#!/usr/bin/env python3
"""
Quick verification script for LoongFlow integration
Run this to verify LoongFlow is properly installed and working
"""

def verify_installation():
    """Verify LoongFlow installation"""
    print("="*70)
    print("LoongFlow Integration Verification")
    print("="*70)
    print()

    # Test 1: Basic import
    print("[1/6] Testing basic import...")
    try:
        import loongflow
        print(f"  [OK] LoongFlow {loongflow.__version__} imported successfully")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    # Test 2: Memory components
    print("\n[2/6] Testing memory components...")
    try:
        from loongflow.agentsdk.memory.evolution import (
            EvolveMemory, InMemory, MemoryFactory
        )
        print("  [OK] Memory components imported")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    # Test 3: Message system
    print("\n[3/6] Testing message system...")
    try:
        from loongflow.agentsdk.message import Message, Role
        # Create message using from_text helper
        msg = Message.from_text("Test message", role=Role.USER)
        print("  [OK] Message system working")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    # Test 4: Model integration
    print("\n[4/6] Testing model integration...")
    try:
        from loongflow.agentsdk.models import LiteLLMModel
        print("  [OK] Model integration available")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    # Test 5: PES framework
    print("\n[5/6] Testing PES framework...")
    try:
        from loongflow.framework.pes import PESAgent, Worker
        print("  [OK] PES framework available")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    # Test 6: Tools framework
    print("\n[6/6] Testing tools framework...")
    try:
        from loongflow.agentsdk.tools import BaseTool, function_tool
        print("  [OK] Tools framework available")
    except Exception as e:
        print(f"  [FAIL] Failed: {e}")
        return False

    print()
    print("="*70)
    print("[OK] All verification checks passed!")
    print("="*70)
    print()
    print("LoongFlow is ready for use in OpenEvolve")
    print()
    print("Available PES Agents:")
    print("  - MathEvolveAgent: LoongFlow/agents/math_agent/math_evolve_agent.py")
    print("  - MLEvolveAgent:   LoongFlow/agents/ml_agent/ml_evolve_agent.py")
    print("  - GeneralEvolveAgent: LoongFlow/agents/general_agent/general_evolve_agent.py")
    print()
    print("Documentation: docs/knowledge_engine/LOONGFLOW_INTEGRATION_REPORT.md")
    print()

    return True


if __name__ == "__main__":
    import sys
    success = verify_installation()
    sys.exit(0 if success else 1)
