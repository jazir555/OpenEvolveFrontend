"""
Master Engine Verification Script
Checks all 31+ components are properly wired in the master engine.
"""

import sys
sys.path.insert(0, 'c:\\Users\\mmeadow\\Documents\\OpenEvolve\\Frontend')

from knowledge_engine.master_engine import ComponentRegistry

registry = ComponentRegistry()

print("=" * 60)
print("MASTER ENGINE VERIFICATION REPORT")
print("=" * 60)
print()

print(f"Total components registered: {len(registry.components)}")
print(f"Total capabilities defined: {len(registry.capabilities)}")
print(f"Substitution matrix entries: {len(registry.substitution_matrix)}")
print()

# Expected components list (29 components)
expected = [
    'graphiti', 'kggen', 'oneke', 'aikg', 'deepke', 'ragbits', 
    'crewai', 'pami', 'neuralkg', 'causal_learn', 'karateclub',
    'global_chem', 'neuromancer', 'lagrange_mapper', 'leanaide',
    'research_quest', 'agentic_context', 'agentjson', 'dspy',
    'openevolve_lib', 'mcp_gateway', 'outlines', 'lmql',
    'neuromancer_ke', 'cognitive_hydraulics', 'dts', 'guardrails',
    'icr', 'roma'
]

print("COMPONENT CHECK:")
print("-" * 40)

missing = []
present = []
for comp in expected:
    if comp in registry.components:
        present.append(comp)
        status = "[OK]"
    else:
        missing.append(comp)
        status = "[MISSING]"
    print(f"  {status} {comp}")

print()
print(f"Present: {len(present)}/{len(expected)}")
print(f"Missing: {len(missing)}")
if missing:
    print(f"  Missing components: {missing}")
print()

# Check capabilities coverage
print("CAPABILITY COVERAGE:")
print("-" * 40)
capability_issues = []
for comp in expected:
    if comp not in registry.capabilities:
        capability_issues.append(f"{comp}: no capabilities defined")

if capability_issues:
    print("  ISSUES FOUND:")
    for issue in capability_issues:
        print(f"    - {issue}")
else:
    print(f"  ✓ All {len(expected)} components have capabilities defined")
print()

# Check substitution matrix coverage
print("SUBSTITUTION MATRIX COVERAGE:")
print("-" * 40)
substitution_issues = []
for comp in expected:
    if comp not in registry.substitution_matrix:
        substitution_issues.append(f"{comp}: no substitution fallback")

if substitution_issues:
    print("  ISSUES FOUND:")
    for issue in substitution_issues:
        print(f"    - {issue}")
else:
    print(f"  ✓ All {len(expected)} components have substitution fallbacks")
print()

# Check execute handlers in MasterKnowledgeEngine
print("EXECUTE HANDLERS CHECK:")
print("-" * 40)
from knowledge_engine.master_engine import MasterKnowledgeEngine

# Create a dummy engine to check handlers
try:
    engine = MasterKnowledgeEngine(enable_learning=False, enable_healing=False)
    
    # Check which components have dedicated handlers
    expected_handlers = [
        'kggen', 'graphiti', 'oneke', 'aikg', 'deepke', 'ragbits', 
        'crewai', 'pami', 'neuralkg', 'causal_learn', 'karateclub',
        'global_chem', 'neuromancer', 'lagrange_mapper', 'leanaide',
        'research_quest', 'agentic_context', 'agentjson', 'dspy',
        'openevolve_lib', 'mcp_gateway', 'roma'
    ]
    
    # Check _execute_component handlers mapping
    handlers_defined = []
    handlers_missing = []
    
    # The handlers are defined in _execute_component method
    # Let's check which ones have methods defined
    for comp in expected:
        handler_name = f"_execute_{comp}"
        if hasattr(engine, handler_name):
            handlers_defined.append(comp)
        else:
            handlers_missing.append(comp)
    
    print(f"  Handlers defined: {len(handlers_defined)}")
    print(f"  Handlers missing: {len(handlers_missing)}")
    if handlers_missing:
        print(f"    Missing handlers for: {handlers_missing}")
    else:
        print("  [OK] All components have execute handlers")
        
except Exception as e:
    print(f"  Could not instantiate engine: {e}")
    handlers_defined = []
    handlers_missing = expected

print()

# Final summary
print("=" * 60)
print("FINAL VERDICT")
print("=" * 60)

all_checks_pass = (
    len(missing) == 0 and 
    len(capability_issues) == 0 and 
    len(substitution_issues) == 0 and
    len(handlers_missing) == 0
)

if all_checks_pass:
    print("STATUS: FULLY WIRED [OK]")
    print("All 29 components are properly configured with:")
    print("  - Imports present")
    print("  - Component initialization")
    print("  - Capabilities defined")
    print("  - Substitution fallbacks")
    print("  - Execute handlers")
else:
    print("STATUS: PARTIAL / INCOMPLETE")
    if missing:
        print(f"  - {len(missing)} components missing from registry")
    if capability_issues:
        print(f"  - {len(capability_issues)} components missing capabilities")
    if substitution_issues:
        print(f"  - {len(substitution_issues)} components missing substitutions")
    if handlers_missing:
        print(f"  - {len(handlers_missing)} components missing execute handlers")

print()
print("COMPONENT SUMMARY:")
print("-" * 40)
print(f"  Total Expected:     {len(expected)}")
print(f"  Registered:         {len(present)}")
print(f"  With Capabilities:  {len(registry.capabilities)}")
print(f"  With Substitutions: {len(registry.substitution_matrix)}")
print(f"  With Handlers:      {len(handlers_defined)}")

# Show all available capabilities
print()
print("ALL CAPABILITIES BY COMPONENT:")
print("-" * 40)
for comp in sorted(registry.capabilities.keys()):
    caps = registry.capabilities[comp]
    print(f"  {comp}: {', '.join(caps[:3])}{'...' if len(caps) > 3 else ''}")
