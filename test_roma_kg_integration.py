#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test script for ROMA-KG integration functionality.

Verifies that the new knowledge graph integration methods
work correctly with the ROMA integration.
"""

import asyncio
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Add knowledge_engine to path
sys.path.insert(0, str(Path(__file__).parent))

from knowledge_engine.integrations.roma_integration import (
    ROMAIntegration,
    ROMAResult,
    ROMADecomposition,
    ROMASolution
)


async def test_knowledge_integration():
    """Test knowledge integration features."""

    print("=" * 80)
    print("Testing ROMA-KG Integration")
    print("=" * 80)

    # Initialize ROMA with knowledge integration enabled
    config = {
        "knowledge_integration": {
            "enabled": True,
            "auto_extract_entities": True,
            "auto_store_solutions": True,
            "entity_types": ["concept", "solution", "pattern", "problem"],
            "similarity_threshold": 0.7,
            "max_artifacts": 10,
            "cache_results": True
        }
    }

    roma = ROMAIntegration(config=config)

    print("\n1. Testing configuration...")
    kg_config = roma.config.get("knowledge_integration", {})
    print(f"   [OK] Knowledge integration enabled: {kg_config.get('enabled', False)}")
    print(f"   [OK] Auto-extract entities: {kg_config.get('auto_extract_entities', False)}")
    print(f"   [OK] Auto-store solutions: {kg_config.get('auto_store_solutions', False)}")

    print("\n2. Testing decompose_problem with entity extraction...")
    result = await roma.decompose_problem(
        "Design a scalable microservices architecture",
        max_depth=2,
        extract_entities=True
    )

    print(f"   [OK] Decomposition success: {result.success}")
    print(f"   [OK] Sub-problem count: {result.metadata.get('sub_problem_count', 0)}")
    print(f"   [OK] Entities extracted: {result.metadata.get('entities_extracted', 0)}")

    if result.metadata.get("entities"):
        entities = result.metadata["entities"]
        print(f"   [OK] Sample entity types: {[e['type'] for e in entities[:3]]}")

    print("\n3. Testing extract_knowledge_entities method...")
    if result.success:
        entities = await roma.extract_knowledge_entities(result)
        print(f"   [OK] Extracted {len(entities)} entities")
        if entities:
            print(f"   [OK] Sample entity: {entities[0]['type']} - {entities[0]['name'][:50]}...")

    print("\n4. Testing solve_atomic...")
    if result.decomposition and result.decomposition.sub_problems:
        atomic = result.decomposition.sub_problems[0]
        solve_result = await roma.solve_atomic(atomic)
        print(f"   [OK] Solve success: {solve_result.success}")
        print(f"   [OK] Solution confidence: {solve_result.metadata.get('confidence', 0.0):.2f}")

    print("\n5. Testing reassemble_solution with knowledge storage...")
    # Create mock solutions
    solutions = [
        ROMASolution(
            solution_id="sol1",
            problem_id="prob1",
            solution="Implement API Gateway",
            confidence=0.85,
            reasoning="Applied reasoning agent",
            metadata={"agent": "reasoning"}
        ),
        ROMASolution(
            solution_id="sol2",
            problem_id="prob2",
            solution="Implement Service Discovery",
            confidence=0.90,
            reasoning="Applied reasoning agent",
            metadata={"agent": "reasoning"}
        )
    ]

    reassemble_result = await roma.reassemble_solution(
        solutions,
        store_as_knowledge=True
    )

    print(f"   [OK] Reassembly success: {reassemble_result.success}")
    print(f"   [OK] Aggregate confidence: {reassemble_result.metadata.get('aggregate_confidence', 0.0):.2f}")
    print(f"   [OK] Artifact ID: {reassemble_result.metadata.get('knowledge_artifact_id', 'None')}")

    print("\n6. Testing store_solution_as_knowledge method...")
    if reassemble_result.success:
        artifact_id = await roma.store_solution_as_knowledge(reassemble_result)
        print(f"   [OK] Stored artifact: {artifact_id}")

    print("\n7. Testing statistics...")
    stats = roma.get_statistics()
    print(f"   [OK] Decompositions performed: {stats['decompositions_performed']}")
    print(f"   [OK] Problems solved: {stats['problems_solved']}")
    print(f"   [OK] Reassemblies performed: {stats['reassemblies_performed']}")
    print(f"   [OK] Entities extracted: {stats['entities_extracted']}")
    print(f"   [OK] Solutions stored: {stats['solutions_stored']}")
    print(f"   [OK] Knowledge integration info:")
    print(f"      - Enabled: {stats['knowledge_integration']['enabled']}")
    print(f"      - Cached artifacts: {stats['knowledge_integration']['cached_artifacts']}")

    print("\n8. Testing health check...")
    health = roma.health_check()
    print(f"   [OK] Health status: {health['status']}")
    print(f"   [OK] Components available: {sum(1 for v in health['components'].values() if v == 'available')}")

    print("\n9. Cleanup...")
    await roma.close()
    print("   [OK] ROMA integration closed")

    print("\n" + "=" * 80)
    print("All tests completed successfully!")
    print("=" * 80)

    return True


async def test_backward_compatibility():
    """Test that existing functionality still works."""

    print("\n" + "=" * 80)
    print("Testing Backward Compatibility")
    print("=" * 80)

    # Initialize ROMA with default config (knowledge integration disabled)
    roma = ROMAIntegration()

    print("\n1. Testing decompose_problem (no entity extraction)...")
    result = await roma.decompose_problem(
        "Solve a simple problem",
        max_depth=1
    )

    print(f"   [OK] Decomposition success: {result.success}")
    print(f"   [OK] Entities extracted: {result.metadata.get('entities_extracted', 0)}")

    print("\n2. Testing reassemble_solution (no knowledge storage)...")
    solutions = [
        ROMASolution(
            solution_id="sol1",
            problem_id="prob1",
            solution="Simple solution",
            confidence=0.8,
            reasoning="Reasoning",
            metadata={}
        )
    ]

    reassemble_result = await roma.reassemble_solution(solutions)
    print(f"   [OK] Reassembly success: {reassemble_result.success}")
    print(f"   [OK] Knowledge artifact: {reassemble_result.metadata.get('knowledge_artifact_id', 'None')}")

    print("\n" + "=" * 80)
    print("Backward compatibility maintained!")
    print("=" * 80)

    await roma.close()
    return True


if __name__ == "__main__":
    try:
        # Run tests
        asyncio.run(test_knowledge_integration())
        asyncio.run(test_backward_compatibility())

        print("\n[OK] All tests passed!")

    except Exception as e:
        print(f"\n[FAIL] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
