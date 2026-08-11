#!/usr/bin/env python3
"""Audit script for LeanAide files"""
import os
import re

files = [
    'leanaide_adversarial.py', 'leanaide_api_routes.py', 'leanaide_autoformalization_mdap_maker.py',
    'leanaide_client.py', 'leanaide_config.py', 'leanaide_continuous_math.py', 'leanaide_continuous_mcp.py',
    'leanaide_crewai_bridge.py', 'leanaide_decomposition_integration.py', 'leanaide_evolution.py',
    'leanaide_evolution_mdap.py', 'leanaide_evolution_mdap_workflow.py', 'leanaide_evolutionary_workflow.py',
    'leanaide_hybrid_maker_enhanced.py', 'leanaide_hybrid_strategies.py', 'leanaide_maker.py',
    'leanaide_mcp_tools.py', 'leanaide_mcts.py', 'leanaide_mcts_mdap.py', 'leanaide_mcts_mdap_complete.py',
    'leanaide_mcts_mdap_workflow.py', 'leanaide_mcts_strategies.py', 'leanaide_mcts_workflow.py',
    'leanaide_mdap.py', 'leanaide_mdap_demo.py', 'leanaide_mdap_workflow.py', 'leanaide_pes_benchmark.py',
    'leanaide_pes_handler.py', 'leanaide_predictive_flagging.py', 'leanaide_redflagging.py',
    'leanaide_redflagging_system.py', 'leanaide_selfplay.py', 'leanaide_sop_integration.py',
    'leanaide_strategies.py', 'leanaide_workflow_integration.py'
]

results = []
for f in files:
    if not os.path.exists(f):
        results.append((f, False, False, False, False, False, 'NOT FOUND'))
        continue
    content = open(f, 'r', encoding='utf-8', errors='ignore').read()
    
    # Check imports from lean4_integration or leanaide_client
    imports_lean4 = bool(re.search(r'from lean4_integration|from leanaide_client|import lean4_integration|import leanaide_client', content))
    
    # Check flags
    has_lean_available = 'LEAN_AVAILABLE' in content or 'LEAN4_AVAILABLE' in content
    has_leanaide_available = 'LEANAIDE_AVAILABLE' in content
    
    # Check for mock/stub implementations
    has_mocks = bool(re.search(r'class.*Stub|# Stub|"""Stub|fallback|simulation mode|simulation', content, re.I))
    
    # Check for actual Lean usage
    uses_real_lean = bool(re.search(r'leanaide_client|LeanAideClient|lean4_integration|verify.*lean|lean_service', content, re.I))
    
    # Determine status
    if imports_lean4 or has_leanaide_available:
        if has_mocks:
            status = 'PARTIAL - Has fallback'
        else:
            status = 'INTEGRATED'
    else:
        status = 'NO LEAN INTEGRATION'
    
    results.append((f, imports_lean4, has_lean_available, has_leanaide_available, uses_real_lean, has_mocks, status))

# Print markdown table
print("# LeanAide Files Audit Report\n")
print("| # | File | Imports Lean? | LEAN_AVAILABLE? | LEANAIDE_AVAILABLE? | Uses Real Lean? | Has Mocks? | Status |")
print("|---|------|---------------|-----------------|---------------------|-----------------|------------|--------|")
for i, r in enumerate(results, 1):
    print(f"| {i} | {r[0]} | {'Yes' if r[1] else 'No'} | {'Yes' if r[2] else 'No'} | {'Yes' if r[3] else 'No'} | {'Yes' if r[4] else 'No'} | {'Yes' if r[5] else 'No'} | {r[6]} |")

# Summary
print("\n## Summary Statistics\n")
total = len(results)
with_imports = sum(1 for r in results if r[1])
with_lean_flag = sum(1 for r in results if r[2])
with_leanaide_flag = sum(1 for r in results if r[3])
with_mocks = sum(1 for r in results if r[5])
print(f"- Total files: {total}")
print(f"- Files importing Lean: {with_imports}")
print(f"- Files with LEAN_AVAILABLE flag: {with_lean_flag}")
print(f"- Files with LEANAIDE_AVAILABLE flag: {with_leanaide_flag}")
print(f"- Files with mocks/fallbacks: {with_mocks}")
