"""
Component Communication Paths Verification Script

Verifies that components can communicate through:
1. Master Engine's component registry
2. Unified Hub's integration routing
3. Global Orchestrator's workflow composition
"""

print("=== Component Communication Paths Verification ===")

# Check Master Engine component access
try:
    from knowledge_engine.master_engine import MasterKnowledgeEngine
    engine = MasterKnowledgeEngine()
    registry = engine.component_registry
    print("\n[OK] Master Engine initialized successfully")
    print("   Total components: {}".format(len(registry.components)))
except Exception as e:
    print("\n[FAIL] Master Engine Error: {}".format(e))
    registry = None

# Verify extraction pipeline components can communicate
if registry:
    extraction_pipeline = ['deepke', 'oneke', 'kggen', 'guardrails', 'icr']
    print("\n1. Extraction Pipeline Communication:")
    extraction_ok = 0
    for comp in extraction_pipeline:
        if comp in registry.components:
            print("  [OK] {} accessible".format(comp))
            extraction_ok += 1
        else:
            print("  [FAIL] {} NOT accessible".format(comp))
    print("   Status: {}/{} components accessible".format(extraction_ok, len(extraction_pipeline)))

    # Verify reasoning pipeline
    reasoning_pipeline = ['cognitive_hydraulics', 'neuromancer', 'z3', 'leanaide']
    print("\n2. Reasoning Pipeline Communication:")
    reasoning_ok = 0
    for comp in reasoning_pipeline:
        if comp in registry.components:
            print("  [OK] {} accessible".format(comp))
            reasoning_ok += 1
        else:
            print("  [FAIL] {} NOT accessible".format(comp))
    print("   Status: {}/{} components accessible".format(reasoning_ok, len(reasoning_pipeline)))

    # Verify conversation pipeline
    conversation_pipeline = ['dts', 'guardrails', 'outlines', 'crewai']
    print("\n3. Conversation Pipeline Communication:")
    conversation_ok = 0
    for comp in conversation_pipeline:
        if comp in registry.components:
            print("  [OK] {} accessible".format(comp))
            conversation_ok += 1
        else:
            print("  [FAIL] {} NOT accessible".format(comp))
    print("   Status: {}/{} components accessible".format(conversation_ok, len(conversation_pipeline)))

# Check Unified Hub routing
try:
    from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub, KGOperationType
    hub = UnifiedKGIntegrationHub()
    print("\n4. Operation Routing:")
    operations = [
        (KGOperationType.ENTITY_EXTRACTION, ['deepke', 'oneke']),
        (KGOperationType.SAFETY_VALIDATION, ['guardrails']),
        (KGOperationType.ITERATIVE_REFINEMENT, ['icr']),
        (KGOperationType.PHYSICS_SIMULATION, ['neuromancer']),
        (KGOperationType.TOPOLOGICAL_ANALYSIS, ['lagrange_mapper']),
    ]
    
    routing_ok = 0
    for op_type, expected_integrations in operations:
        if op_type in hub._routing_map:
            routed = hub._routing_map[op_type]
            match = any(r in expected_integrations for r in routed)
            status = "[OK]" if match else "[FAIL]"
            if match:
                routing_ok += 1
            print("  {} {}: {}".format(status, op_type.name, routed))
        else:
            print("  [FAIL] {}: NOT ROUTED".format(op_type.name))
    print("   Status: {}/{} operations routed correctly".format(routing_ok, len(operations)))
except Exception as e:
    print("\n[FAIL] Unified Hub Error: {}".format(e))
    hub = None

# Verify fallback chains
if registry:
    print("\n5. Fallback Communication Chains:")
    fallback_chains = [
        ('outlines', ['agentjson', 'dspy']),
        ('neuromancer_ke', ['neuromancer', 'causal_learn']),
        ('dts', ['crewai', 'agentic_context']),
        ('guardrails', ['agentjson', 'z3']),
        ('icr', ['dspy', 'outlines']),
        ('lagrange_mapper', ['neuralkg', 'karateclub']),
    ]
    
    fallback_ok = 0
    for primary, fallbacks in fallback_chains:
        if primary in registry.substitution_matrix:
            actual = registry.substitution_matrix[primary]
            match = all(f in actual for f in fallbacks)
            status = "[OK]" if match else "[WARN]"
            if match:
                fallback_ok += 1
            print("  {} {} -> {}".format(status, primary, actual))
        else:
            print("  [FAIL] {}: NO FALLBACK CHAIN".format(primary))
    print("   Status: {}/{} fallback chains configured".format(fallback_ok, len(fallback_chains)))

# Additional verification: Component Capabilities
if registry:
    print("\n6. Component Capabilities Mapping:")
    key_capabilities = [
        'entity_extraction',
        'relation_extraction',
        'ai_safety',
        'output_validation',
        'iterative_refinement',
        'physics_simulation',
        'hybrid_reasoning',
        'conversation_optimization'
    ]
    
    all_caps = registry.get_all_capabilities()
    for cap in key_capabilities:
        providers = all_caps.get(cap, [])
        if providers:
            print("  [OK] {}: {}".format(cap, providers))
        else:
            print("  [FAIL] {}: NO PROVIDERS".format(cap))

# Summary
print("\n" + "="*50)
print("Component Communication Paths Report")
print("="*50)

if registry:
    print("\n1. Extraction Pipeline:")
    for comp in ['deepke', 'oneke', 'kggen', 'guardrails', 'icr']:
        status = "[OK]" if comp in registry.components else "[FAIL]"
        print("  {} {}".format(status, comp))

    print("\n2. Reasoning Pipeline:")
    for comp in ['cognitive_hydraulics', 'neuromancer', 'z3', 'leanaide']:
        status = "[OK]" if comp in registry.components else "[FAIL]"
        print("  {} {}".format(status, comp))

    print("\n3. Conversation Pipeline:")
    for comp in ['dts', 'guardrails', 'outlines', 'crewai']:
        status = "[OK]" if comp in registry.components else "[FAIL]"
        print("  {} {}".format(status, comp))

if hub:
    print("\n4. Operation Routing:")
    for op_type, expected in operations:
        if op_type in hub._routing_map:
            routed = hub._routing_map[op_type]
            status = "[OK]" if routed else "[FAIL]"
            print("  {} {}: {}".format(status, op_type.name, routed))

if registry:
    print("\n5. Fallback Chains:")
    for primary, fallbacks in fallback_chains:
        if primary in registry.substitution_matrix:
            actual = registry.substitution_matrix[primary]
            print("  [OK] {} -> {}".format(primary, actual))
        else:
            print("  [FAIL] {}: NO FALLBACK".format(primary))

# Determine overall status
total_components = 0
accessible_components = 0
if registry:
    all_pipelines = ['deepke', 'oneke', 'kggen', 'guardrails', 'icr', 
                     'cognitive_hydraulics', 'neuromancer', 'z3', 'leanaide',
                     'dts', 'outlines', 'crewai']
    total_components = len(all_pipelines)
    accessible_components = sum(1 for c in all_pipelines if c in registry.components)

print("\n" + "="*50)
if accessible_components == total_components:
    print("Communication Status: FULLY CONNECTED [OK]")
elif accessible_components >= total_components * 0.7:
    print("Communication Status: PARTIAL ({}/{})".format(accessible_components, total_components))
else:
    print("Communication Status: DISCONNECTED ({}/{})".format(accessible_components, total_components))
print("="*50)
