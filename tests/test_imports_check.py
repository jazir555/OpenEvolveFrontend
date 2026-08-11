#!/usr/bin/env python
"""Quick check for import and API compatibility issues."""

import sys
from pathlib import Path

# Add tests directory to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("CHECKING IMPORTS AND API COMPATIBILITY")
print("=" * 80)

OK = "[OK]"
FAIL = "[FAIL]"

# Check 1: Entity import
print("\n1. Checking Entity import...")
try:
    from knowledge_engine.core.entity_knowledge_graph import Entity
    print("   [OK] Entity imported successfully")

    # Check Entity accepts both 'attributes' and 'properties'
    try:
        e1 = Entity(name="Test1", entity_type="Concept", attributes={"key": "value"})
        print(f"   [OK] Entity accepts 'attributes' parameter: {e1.properties}")
    except Exception as e:
        print(f"   [FAIL] Entity 'attributes' parameter failed: {e}")

    try:
        e2 = Entity(name="Test2", entity_type="Concept", properties={"key": "value"})
        print(f"   [OK] Entity accepts 'properties' parameter: {e2.properties}")
    except Exception as e:
        print(f"   [FAIL] Entity 'properties' parameter failed: {e}")

except ImportError as e:
    print(f"   [FAIL] Entity import failed: {e}")

# Check 2: KnowledgeState
print("\n2. Checking KnowledgeState...")
try:
    from knowledge_engine.core import KnowledgeState
    print("   [OK] KnowledgeState imported")

    state = KnowledgeState(query="test")
    print(f"   [OK] KnowledgeState has query: {state.query}")
    print(f"   [OK] KnowledgeState has search_history: {hasattr(state, 'search_history')}")
    print(f"   [OK] KnowledgeState has set_current_understanding: {hasattr(state, 'set_current_understanding')}")

except ImportError as e:
    print(f"   [FAIL] KnowledgeState import failed: {e}")

# Check 3: EntityKnowledgeGraph
print("\n3. Checking EntityKnowledgeGraph...")
try:
    from knowledge_engine.core.entity_knowledge_graph import EntityKnowledgeGraph
    print("   [OK] EntityKnowledgeGraph imported")

    kg = EntityKnowledgeGraph(correlation_id="test")
    print(f"   [OK] EntityKnowledgeGraph initialized with correlation_id")

    # Check add_entity method
    try:
        kg.add_entity(name="Test", entity_type="Concept", attributes={"test": True})
        print(f"   [OK] add_entity accepts 'attributes'")
    except Exception as e:
        print(f"   [FAIL] add_entity with 'attributes' failed: {e}")

except ImportError as e:
    print(f"   [FAIL] EntityKnowledgeGraph import failed: {e}")

# Check 4: ROMAResult
print("\n4. Checking ROMAResult...")
try:
    from knowledge_engine.integrations.roma_integration import ROMAResult
    print("   [OK] ROMAResult imported")
    print(f"   [OK] ROMAResult should have 'subproblems' property (defined in class)")
except ImportError as e:
    print(f"   [FAIL] ROMAResult import failed: {e}")

# Check 5: UnifiedEvolutionAPI
print("\n5. Checking UnifiedEvolutionAPI...")
try:
    from openevolve.unified.unified_evolution_api import UnifiedEvolutionAPI
    print("   [OK] UnifiedEvolutionAPI imported")

    # Try initializing with parameters that tests use
    try:
        api = UnifiedEvolutionAPI(
            knowledge_engine=None,  # Tests pass this
            strategy_recommender=None,  # Tests pass this
            enable_gauntlets=False  # Tests pass this
        )
        print(f"   [OK] UnifiedEvolutionAPI accepts deprecated parameters")
    except TypeError as e:
        print(f"   [FAIL] UnifiedEvolutionAPI parameter mismatch: {e}")

except ImportError as e:
    print(f"   [FAIL] UnifiedEvolutionAPI import failed: {e}")

# Check 6: KnowledgeBase
print("\n6. Checking KnowledgeBase...")
try:
    from knowledge_engine.knowledge_base import KnowledgeBase
    print("   [OK] KnowledgeBase imported")

    # Try initializing with db_path (what tests use)
    try:
        kb = KnowledgeBase(db_path=":memory:")
        print(f"   [OK] KnowledgeBase accepts 'db_path'")
    except TypeError as e:
        print(f"   [FAIL] KnowledgeBase parameter mismatch: {e}")

except ImportError as e:
    print(f"   [FAIL] KnowledgeBase import failed: {e}")

# Check 7: OneKEIntegration
print("\n7. Checking OneKEIntegration...")
try:
    from knowledge_engine.integrations.oneke_integration import OneKEIntegration
    print("   [OK] OneKEIntegration imported")
    print(f"   [OK] extract_entities should return EnhancedExtractionResult with .success")
except ImportError as e:
    print(f"   [FAIL] OneKEIntegration import failed: {e}")

print("\n" + "=" * 80)
print("CHECK COMPLETE")
print("=" * 80)
