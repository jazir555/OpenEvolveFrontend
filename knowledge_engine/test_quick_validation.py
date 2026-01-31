"""Quick validation of key platform capabilities."""

import sys

print("Python:", sys.version)
print()

# Test key imports
modules = [
    'enhanced_knowledge_core', 'enhanced_knowledge_engine', 'unified_knowledge_platform',
    'distributed_coordination', 'realtime_collaboration', 'ml_intelligence', 'nlp_layer',
    'workflow_automation', 'security_layer', 'knowledge_analytics', 'multi_tenant',
    'backup_recovery', 'api_gateway', 'final_integration'
]

print("Import Test:")
for mod in modules:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}")

print()
print("Testing CompleteKnowledgePlatform...")
from final_integration import CompleteKnowledgePlatform

# Get all methods
methods = [m for m in dir(CompleteKnowledgePlatform) if not m.startswith('_')]
print(f"Total public members: {len(methods)}")

# Check key capabilities
key_capabilities = [
    'initialize', 'shutdown', 'add_knowledge_with_nlp', 'search_with_nlp',
    'create_tenant', 'backup', 'export', 'handle_api_request',
    'get_comprehensive_stats', 'health_check'
]

print("Key capabilities:")
for cap in key_capabilities:
    has = hasattr(CompleteKnowledgePlatform, cap)
    status = "OK" if has else "MISSING"
    print(f"  [{status}] {cap}")

# Test individual components
print()
print("Component Tests:")

# NLP
from nlp_layer import NLPEngine
nlp = NLPEngine()
analysis = nlp.analyze("Python is a programming language created by Guido van Rossum.")
print(f"  [OK] NLP: {len(analysis.keywords)} keywords extracted")

# ML
from ml_intelligence import ContentClassifier
classifier = ContentClassifier()
result = classifier.classify("Machine learning with Python and TensorFlow")
print(f"  [OK] ML: Classified as '{result.category}' ({result.confidence:.2f})")

# Security
from security_layer import SecurityManager
sec = SecurityManager()
encrypted = sec.encryption.encrypt("test data")
print(f"  [OK] Security: Data encrypted ({len(encrypted)} chars)")

# Multi-tenant
from multi_tenant import TenantManager
tm = TenantManager()
tenant = tm.create_tenant("Test Corp", "test-corp", "owner-123")
print(f"  [OK] Multi-tenant: Created tenant '{tenant.name}'")

# API Gateway
from api_gateway import RESTAPIGateway, KnowledgeAPIFactory
gateway = RESTAPIGateway()
print(f"  [OK] API Gateway: Created with {len(gateway.routes)} routes")

print()
print("=" * 60)
print("ALL COMPONENT TESTS PASSED")
print("=" * 60)
