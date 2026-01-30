#!/usr/bin/env python3
"""
Verify Knowledge Engine Integration Status
Checks all 14+ integrated projects
"""
import sys
sys.path.insert(0, '.')

def check_integration(module_name, class_name=None):
    """Check if an integration module imports successfully."""
    try:
        module = __import__(module_name, fromlist=[class_name] if class_name else [])
        if class_name:
            getattr(module, class_name)
        return True, "OK"
    except Exception as e:
        return False, str(e)

# Define all integrations
integrations = [
    # Core 14+ projects
    ("knowledge_engine.integrations.graphiti_integration", "GraphitiIntegration"),
    ("knowledge_engine.integrations.kggen_integration", "KGGenIntegration"),
    ("knowledge_engine.integrations.oneke_integration", "OneKEIntegration"),
    ("knowledge_engine.integrations.aikg_integration", "AIKGIntegration"),
    ("knowledge_engine.integrations.ragbits_integration", "RagbitsIntegration"),
    ("knowledge_engine.integrations.crewai_integration", "CrewAIIntegration"),
    ("knowledge_engine.integrations.deepke_integration", "DeepKEIntegration"),
    ("knowledge_engine.integrations.research_quest_integration", "ResearchQuestIntegration"),
    ("knowledge_engine.integrations.agentic_context_integration", "AgenticContextEngine"),
    ("knowledge_engine.integrations.agentjson_integration", "AgentJSONIntegration"),
    ("knowledge_engine.integrations.dspy_integration", "DSPyIntegration"),
    ("knowledge_engine.integrations.leanaide_integration", "LeanAideIntegration"),
    ("knowledge_engine.integrations.openevolve_integration_library", "OpenEvolveIntegrationLibrary"),
    ("knowledge_engine.integrations.mcp_gateway_integration", "MCPGatewayIntegration"),
    # Additional integrations
    ("knowledge_engine.integrations.pami_integration", "PAMIIntegration"),
    ("knowledge_engine.integrations.neuralkg_integration", "NeuralKGIntegration"),
    ("knowledge_engine.integrations.causal_learn_integration", "CausalLearnIntegration"),
    ("knowledge_engine.integrations.lagrange_mapper_integration", "LagrangeMapperIntegration"),
    ("knowledge_engine.integrations.karateclub_integration", "KarateClubIntegration"),
    ("knowledge_engine.integrations.global_chem_integration", "GlobalChemIntegration"),
    ("knowledge_engine.integrations.neuromancer_integration", "NeuromancerIntegration"),
]

# Check orchestration components
orchestration_components = [
    ("knowledge_engine.orchestration.knowledge_orchestrator", "KnowledgeOrchestrator"),
    ("knowledge_engine.orchestration.self_healing_orchestrator", "SelfHealingOrchestrator"),
    ("knowledge_engine.orchestration.learning_engine", "LearningEngine"),
    ("knowledge_engine.orchestration.global_learning_engine", "GlobalLearningEngine"),
    ("knowledge_engine.orchestration.integrated_orchestrator", "IntegratedOrchestrator"),
    ("knowledge_engine.orchestration.component_coordination", "ComponentCoordinator"),
    ("knowledge_engine.orchestration.feedback_loop", "FeedbackCollector"),
    ("knowledge_engine.orchestration.circuit_breaker", "CircuitBreaker"),
]

# Check core backends
backends = [
    ("knowledge_engine.core.backends.neo4j_backend", "Neo4jBackend"),
    ("knowledge_engine.core.backends.memory_backend", "MemoryBackend"),
    ("knowledge_engine.core.backends.qdrant_backend", "QdrantBackend"),
    ("knowledge_engine.core.backends.mongodb_backend", "MongoDBBackend"),
    ("knowledge_engine.core.backends.karateclub_backend", "KarateClubBackend"),
]

print("=" * 70)
print("KNOWLEDGE ENGINE - COMPREHENSIVE INTEGRATION STATUS")
print("=" * 70)

# Check integrations
print("\n[PROJECT INTEGRATIONS - 14+]")
print("-" * 70)
success_count = 0
failed_count = 0
failed_modules = []

for module, cls in integrations:
    success, msg = check_integration(module, cls)
    status = "OK" if success else "FAIL"
    short_name = module.split('.')[-1].replace('_integration', '')
    if success:
        success_count += 1
        print(f"[OK] {short_name:25} - {msg}")
    else:
        failed_count += 1
        failed_modules.append((short_name, msg))
        short_error = msg[:50] + "..." if len(msg) > 50 else msg
        print(f"[FAIL] {short_name:25} - {short_error}")

# Check orchestration
print("\n[ORCHESTRATION COMPONENTS]")
print("-" * 70)
orchestration_success = 0
for module, cls in orchestration_components:
    success, msg = check_integration(module, cls)
    status = "OK" if success else "FAIL"
    short_name = cls
    if success:
        orchestration_success += 1
        print(f"[OK] {short_name:25} - {msg}")
    else:
        short_error = msg[:50] + "..." if len(msg) > 50 else msg
        print(f"[FAIL] {short_name:25} - {short_error}")

# Check backends
print("\n[CORE BACKENDS]")
print("-" * 70)
backend_success = 0
for module, cls in backends:
    success, msg = check_integration(module, cls)
    status = "OK" if success else "FAIL"
    short_name = cls
    if success:
        backend_success += 1
        print(f"[OK] {short_name:25} - {msg}")
    else:
        short_error = msg[:50] + "..." if len(msg) > 50 else msg
        print(f"[FAIL] {short_name:25} - {short_error}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
total_integrations = len(integrations)
total_orchestration = len(orchestration_components)
total_backends = len(backends)

print(f"Project Integrations: {success_count}/{total_integrations} working ({100*success_count//total_integrations}%)")
print(f"Orchestration Components: {orchestration_success}/{total_orchestration} working")
print(f"Core Backends: {backend_success}/{total_backends} working")

# Failed modules details
if failed_modules:
    print("\n[FAILED MODULES DETAILS]")
    for name, error in failed_modules:
        print(f"\n{name}:")
        print(f"  Error: {error}")

# Overall status
if failed_count == 0:
    print("\n*** ALL INTEGRATIONS OPERATIONAL ***")
    sys.exit(0)
else:
    print(f"\n*** {failed_count} integration(s) need attention ***")
    sys.exit(1)
