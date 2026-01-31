"""Comprehensive test - check all methods and attributes referenced in main __init__.py"""
import sys
import traceback

def test_attribute(obj, attr_name, context=""):
    """Test if an object has an attribute."""
    if hasattr(obj, attr_name):
        return True
    else:
        print(f"  [MISSING] {context}.{attr_name}")
        return False

print("=" * 70)
print("Comprehensive Knowledge Engine Audit")
print("=" * 70)

# Import main module
print("\n1. Importing knowledge_engine...")
from knowledge_engine import OpenEvolveKnowledgeEngine, KnowledgeEngineOutput
print("   [OK] Main imports successful")

# Check learning module imports in main __init__
print("\n2. Checking learning module interface...")
from knowledge_engine.learning import AdaptationEngine, ReflectionEngine

# Check AdaptationEngine required attributes
ae = AdaptationEngine(learning_rate=0.1, memory_retention_hours=24, experience_buffer_size=100)
required_ae_attrs = ['close', 'record_experience', 'get_learning_summary', 'start', 'stop']
for attr in required_ae_attrs:
    test_attribute(ae, attr, "AdaptationEngine")

# Check ReflectionEngine required attributes
re = ReflectionEngine(reflection_frequency=10)
required_re_attrs = ['close', 'record_operation', 'reflect', 'get_reflection_summary', 'start', 'stop']
for attr in required_re_attrs:
    test_attribute(re, attr, "ReflectionEngine")

# Check orchestrator interface
print("\n3. Checking orchestrator interface...")
from knowledge_engine.orchestration import IntegratedOrchestrator, OrchestratorResult

orch = IntegratedOrchestrator()
required_orch_methods = [
    'process_knowledge_request',
    'run_comprehensive_analysis', 
    'get_system_status',
    'close',
    'process',
    'get_comprehensive_status'
]
for method in required_orch_methods:
    test_attribute(orch, method, "IntegratedOrchestrator")

# Check OpenEvolveKnowledgeEngine initialization
print("\n4. Checking OpenEvolveKnowledgeEngine initialization...")
try:
    engine = OpenEvolveKnowledgeEngine()
    print("   [OK] Engine instantiated")
    
    # Check required attributes
    required_attrs = [
        'orchestrator',
        'adaptation_engine', 
        'reflection_engine',
        'learning_memory',
        'config'
    ]
    for attr in required_attrs:
        test_attribute(engine, attr, "OpenEvolveKnowledgeEngine")
    
    # Check required methods
    required_methods = [
        'process_request',
        'run_comprehensive_analysis',
        'get_status',
        'close',
        '_update_learning_memory',
        '_perform_reflection'
    ]
    for method in required_methods:
        test_attribute(engine, method, "OpenEvolveKnowledgeEngine")
        
except Exception as e:
    print(f"   [ERROR] Failed to instantiate: {e}")
    traceback.print_exc()

# Check all submodules referenced in __init__.py
print("\n5. Checking all submodule imports from main __init__.py...")

submodules_to_check = [
    # Phase 1: Graph Foundation
    ("knowledge_engine.graph", [
        "NodeType", "EdgeType", "PropertyType",
        "NodeSchema", "EdgeSchema", "GraphSchema",
        "KnowledgeNode", "KnowledgeEdge", "KnowledgeGraph",
        "NodeProperties", "EdgeProperties",
        "GraphCRUD", "ConnectionPool", "RetryPolicy",
        "CypherQueryBuilder"
    ]),
    
    # Phase 2: DeepKE
    ("knowledge_engine.deepke", [
        "DeepKEExtractor", "EntityExtractor", "RelationExtractor",
        "ExtractedEntity", "ExtractedRelation", "ExtractionResult",
        "DeepKEPipeline", "EntityLinker", "EntityDisambiguator"
    ]),
    
    # Phase 3: Hybrid
    ("knowledge_engine.hybrid", [
        "HybridSearch", "VectorSearch", "GraphSearch",
        "SearchResult", "FusionStrategy",
        "QueryOptimizer", "ResultRanker", "ReciprocalRankFusion"
    ]),
    
    # Phase 4: Architectural gaps
    ("knowledge_engine.sandbox", [
        "SandboxManager", "SandboxType", "ExecutionResult", "SecurityPolicy"
    ]),
    ("knowledge_engine.vision", [
        "VisionLanguageMonitor", "VLMProvider", "VisualAnalysis"
    ]),
    ("knowledge_engine.browser", [
        "BrowserResearchAgent", "SearchResult", "ResearchSession"
    ]),
    ("knowledge_engine.router", [
        "ComplexityRouter", "RouteDecision", "ModelTier", "ComplexityLevel"
    ]),
    ("knowledge_engine.chronicle", [
        "Chronicle", "Episode", "ChronicleQuery", "EpisodeType", "ChronicleIntegration"
    ]),
    
    # Phase 5: Integration
    ("knowledge_engine.integrations.openevolve_integration", [
        "OpenEvolveKnowledgeEngineIntegration", "KnowledgeEngineConfig", 
        "create_knowledge_engine_integration"
    ]),
    
    # Phase 6: Query
    ("knowledge_engine.query", [
        "KnowledgeQuery", "QueryResult", "QueryEngine",
        "QueryOptimizer", "create_query_engine"
    ]),
]

failed_imports = []
for module_name, items in submodules_to_check:
    try:
        module = __import__(module_name, fromlist=items)
        for item in items:
            if not hasattr(module, item):
                print(f"   [MISSING] {module_name}.{item}")
                failed_imports.append(f"{module_name}.{item}")
    except Exception as e:
        print(f"   [ERROR] {module_name}: {e}")
        failed_imports.append(module_name)

if not failed_imports:
    print("   [OK] All submodule imports successful")
else:
    print(f"   [FAIL] {len(failed_imports)} import(s) failed")

print("\n" + "=" * 70)
print("Audit complete")
print("=" * 70)
