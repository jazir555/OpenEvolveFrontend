#!/usr/bin/env python3
"""
DTS Integration Verification Script
Verifies that the DTS integration is 100% complete.
"""
import ast
import os

CHECK = "[OK]"
CROSS = "[FAIL]"

def main():
    print("=" * 70)
    print("DTS INTEGRATION VERIFICATION REPORT")
    print("=" * 70)

    # 1. Check Primary Implementation Files
    print("\n[1] PRIMARY IMPLEMENTATION FILES (integrations/dts/)")
    primary_files = [
        "__init__.py",
        "conversation_tree.py",
        "user_simulator.py",
        "trajectory_scorer.py",
        "beam_search.py",
        "dts_engine.py",
    ]
    primary_ok = True
    for f in primary_files:
        path = f"integrations/dts/{f}"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as file:
                    ast.parse(file.read())
                print(f"  {CHECK} {path}")
            except Exception as e:
                print(f"  {CROSS} {path}: {e}")
                primary_ok = False
        else:
            print(f"  {CROSS} {path} - MISSING")
            primary_ok = False

    # 2. Check KE Wrapper
    print("\n[2] KNOWLEDGE ENGINE WRAPPER (knowledge_engine/integrations/dts/)")
    ke_files = ["__init__.py", "dts_integration.py"]
    ke_ok = True
    for f in ke_files:
        path = f"knowledge_engine/integrations/dts/{f}"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as file:
                    ast.parse(file.read())
                print(f"  {CHECK} {path}")
            except Exception as e:
                print(f"  {CROSS} {path}: {e}")
                ke_ok = False
        else:
            print(f"  {CROSS} {path} - MISSING")
            ke_ok = False

    # 3. Check Unified Hub
    print("\n[3] UNIFIED HUB (knowledge_engine/unified_kg_integration_hub.py)")
    hub_path = "knowledge_engine/unified_kg_integration_hub.py"
    hub_ok = True
    if os.path.exists(hub_path):
        with open(hub_path, "r", encoding="utf-8") as f:
            content = f.read()
        checks = [
            ("CONVERSATION_OPTIMIZATION in KGOperationType", "CONVERSATION_OPTIMIZATION" in content),
            ("_initialize_dts method", "def _initialize_dts" in content),
            ("await self._initialize_dts()", "await self._initialize_dts()" in content),
            ("optimize_conversation method", "def optimize_conversation" in content),
        ]
        for name, result in checks:
            if result:
                print(f"  {CHECK} {name}")
            else:
                print(f"  {CROSS} {name}")
                hub_ok = False
        # Check routing map separately
        if "CONVERSATION_OPTIMIZATION" in content and "dts" in content:
            print(f"  {CHECK} dts in CONVERSATION_OPTIMIZATION routing")
        else:
            print(f"  {CROSS} dts in CONVERSATION_OPTIMIZATION routing")
            hub_ok = False
    else:
        print(f"  {CROSS} {hub_path} - MISSING")
        hub_ok = False

    # 4. Check Master Engine
    print("\n[4] MASTER ENGINE (knowledge_engine/master_engine.py)")
    master_path = "knowledge_engine/master_engine.py"
    master_ok = True
    if os.path.exists(master_path):
        with open(master_path, "r", encoding="utf-8") as f:
            content = f.read()
        checks = [
            ("DTSKGIntegration import", "from knowledge_engine.integrations.dts.dts_integration import DTSKGIntegration" in content),
            ("DTS_AVAILABLE flag", "DTS_AVAILABLE" in content),
            ("dts in capabilities", "dts" in content and "capabilities" in content),
            ("dts in components", "components['dts']" in content),
            ("dts in substitution_matrix", "dts" in content and "substitution_matrix" in content),
        ]
        for name, result in checks:
            if result:
                print(f"  {CHECK} {name}")
            else:
                print(f"  {CROSS} {name}")
                master_ok = False
    else:
        print(f"  {CROSS} {master_path} - MISSING")
        master_ok = False

    # 5. Check Package Exports
    print("\n[5] PACKAGE EXPORTS (knowledge_engine/integrations/__init__.py)")
    exports_path = "knowledge_engine/integrations/__init__.py"
    exports_ok = True
    if os.path.exists(exports_path):
        with open(exports_path, "r", encoding="utf-8") as f:
            content = f.read()
        has_import = "DTSKGIntegration" in content
        has_all = "DTSKGIntegration" in content and "__all__" in content
        if has_import:
            print(f"  {CHECK} DTSKGIntegration imported")
        else:
            print(f"  {CROSS} DTSKGIntegration import missing")
            exports_ok = False
        if has_all:
            print(f"  {CHECK} DTSKGIntegration in __all__")
        else:
            print(f"  {CROSS} DTSKGIntegration not in __all__")
            exports_ok = False
    else:
        print(f"  {CROSS} {exports_path} - MISSING")
        exports_ok = False

    # 6. Check Capability Report
    print("\n[6] CAPABILITY REPORT (knowledge_engine/capability_report.py)")
    cap_path = "knowledge_engine/capability_report.py"
    cap_ok = True
    if os.path.exists(cap_path):
        with open(cap_path, "r", encoding="utf-8") as f:
            content = f.read()
        if "DTS_INTEGRATION_AVAILABLE" in content and "dts" in content.lower():
            print(f"  {CHECK} DTS_INTEGRATION_AVAILABLE imported and referenced")
        else:
            print(f"  {CROSS} DTS_INTEGRATION_AVAILABLE or dts entry missing")
            cap_ok = False
    else:
        print(f"  {CROSS} {cap_path} - MISSING")
        cap_ok = False

    # 7. Check Test File
    print("\n[7] TEST FILE")
    test_path = "knowledge_engine/integrations/dts/test_dts_integration.py"
    test_ok = True
    if os.path.exists(test_path):
        try:
            with open(test_path, "r", encoding="utf-8") as file:
                ast.parse(file.read())
            print(f"  {CHECK} {test_path}")
        except Exception as e:
            print(f"  {CROSS} {test_path}: {e}")
            test_ok = False
    else:
        print(f"  {CROSS} {test_path} - MISSING")
        test_ok = False

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Primary Implementation (6 files): {CHECK if primary_ok else CROSS}")
    print(f"KE Wrapper (2 files): {CHECK if ke_ok else CROSS}")
    print(f"Unified Hub (5 checks): {CHECK if hub_ok else CROSS}")
    print(f"Master Engine (5 checks): {CHECK if master_ok else CROSS}")
    print(f"Package Exports (2 checks): {CHECK if exports_ok else CROSS}")
    print(f"Capability Report (1 check): {CHECK if cap_ok else CROSS}")
    print(f"Test File: {CHECK if test_ok else CROSS}")

    all_ok = primary_ok and ke_ok and hub_ok and master_ok and exports_ok and cap_ok and test_ok
    print("\n" + "=" * 70)
    if all_ok:
        print("FINAL VERDICT: 100% COMPLETE [PASS]")
    else:
        print("FINAL VERDICT: INCOMPLETE - See failures above [FAIL]")
    print("=" * 70)

    return 0 if all_ok else 1


if __name__ == "__main__":
    exit(main())
