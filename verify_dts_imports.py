#!/usr/bin/env python3
"""
DTS Integration Import Verification Script
Tests that all imports work correctly.
"""
import sys

CHECK = "[OK]"
CROSS = "[FAIL]"

print("=" * 70)
print("DTS INTEGRATION IMPORT VERIFICATION")
print("=" * 70)

all_ok = True

# 1. Primary import
print("\n[1] Primary Import: from integrations.dts import DTSEngine")
try:
    from integrations.dts import DTSEngine
    print(f"  {CHECK} Primary import works")
except ImportError as e:
    print(f"  {CROSS} Primary import failed: {e}")
    all_ok = False

# 2. KE wrapper import
print("\n[2] KE Wrapper Import: from knowledge_engine.integrations.dts import DTSKGIntegration")
try:
    from knowledge_engine.integrations.dts import DTSKGIntegration
    print(f"  {CHECK} KE wrapper import works")
except ImportError as e:
    print(f"  {CROSS} KE wrapper import failed: {e}")
    all_ok = False

# 3. Master Engine integration
print("\n[3] Master Engine Integration")
try:
    from knowledge_engine.master_engine import MasterKnowledgeEngine
    engine = MasterKnowledgeEngine()
    assert 'dts' in engine.capabilities, "dts not in capabilities"
    assert 'dts' in engine.components, "dts not in components"
    print(f"  {CHECK} Master Engine integration works")
    print(f"      - dts in capabilities: {CHECK}")
    print(f"      - dts in components: {CHECK}")
except Exception as e:
    print(f"  {CROSS} Master Engine integration failed: {e}")
    all_ok = False

# 4. Unified Hub integration
print("\n[4] Unified Hub Integration")
try:
    from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub, KGOperationType
    assert hasattr(KGOperationType, 'CONVERSATION_OPTIMIZATION'), "CONVERSATION_OPTIMIZATION not in KGOperationType"
    hub = UnifiedKGIntegrationHub()
    # Check routing
    from knowledge_engine.unified_kg_integration_hub import KGOperationType
    assert KGOperationType.CONVERSATION_OPTIMIZATION in hub._routing_map, "CONVERSATION_OPTIMIZATION not in routing map"
    assert 'dts' in hub._routing_map[KGOperationType.CONVERSATION_OPTIMIZATION], "dts not in CONVERSATION_OPTIMIZATION routing"
    print(f"  {CHECK} Unified Hub integration works")
    print(f"      - CONVERSATION_OPTIMIZATION in KGOperationType: {CHECK}")
    print(f"      - dts in routing map: {CHECK}")
except Exception as e:
    print(f"  {CROSS} Unified Hub integration failed: {e}")
    all_ok = False

print("\n" + "=" * 70)
if all_ok:
    print("ALL IMPORT TESTS PASSED [PASS]")
else:
    print("SOME IMPORT TESTS FAILED [FAIL]")
print("=" * 70)

sys.exit(0 if all_ok else 1)
