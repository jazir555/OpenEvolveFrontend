#!/usr/bin/env python
"""
Test script for Sprint 4 Visualization generation.

This script tests the actual generation of visualizations with pyvis.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from knowledge_engine.visualization.graph_explorer import GraphExplorer


async def test_generate():
    """Test visualization generation."""
    print("=" * 60)
    print("SPRINT 4: VISUALIZATION - GENERATION TEST")
    print("=" * 60)

    print("\n[1/4] Initializing GraphExplorer...")
    try:
        explorer = GraphExplorer()
        print("   SUCCESS: GraphExplorer initialized")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    print("\n[2/4] Creating test data...")
    try:
        triples = [
            {"subject": "Alice", "predicate": "knows", "object": "Bob"},
            {"subject": "Bob", "predicate": "knows", "object": "Charlie"},
            {"subject": "Charlie", "predicate": "works_with", "object": "Alice"},
        ]
        entities = [
            {"id": "Alice", "type": "Person", "label": "Alice"},
            {"id": "Bob", "type": "Person", "label": "Bob"},
            {"id": "Charlie", "type": "Person", "label": "Charlie"},
        ]
        print(f"   SUCCESS: Created {len(triples)} test triples and {len(entities)} entities")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    print("\n[3/4] Generating visualization...")
    try:
        result = await explorer.visualize(triples=triples, entities=entities)
        print(f"   SUCCESS: Visualization generated")
        print(f"   Output path: {result.output_path}")
        print(f"   Nodes: {result.node_count}")
        print(f"   Edges: {result.edge_count}")
        print(f"   Communities: {result.community_count}")
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n[4/4] Verifying output file...")
    try:
        if os.path.exists(result.output_path):
            file_size = os.path.getsize(result.output_path)
            print(f"   SUCCESS: File exists")
            print(f"   File size: {file_size} bytes")

            # Check if it's a valid HTML file
            with open(result.output_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if '<!DOCTYPE html>' in content or '<html' in content:
                    print(f"   SUCCESS: Valid HTML structure detected")
                else:
                    print(f"   WARNING: HTML structure not found")

            return True
        else:
            print(f"   FAILED: Output file not created")
            return False
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


async def test_interactive():
    """Test interactive visualization generation."""
    print("\n" + "=" * 60)
    print("BONUS TEST: INTERACTIVE VISUALIZATION")
    print("=" * 60)

    print("\n[1/2] Creating complex graph...")
    try:
        explorer = GraphExplorer()

        triples = [
            {"subject": "Python", "predicate": "is_a", "object": "Programming_Language"},
            {"subject": "Python", "predicate": "used_for", "object": "Data_Science"},
            {"subject": "Python", "predicate": "used_for", "object": "Web_Development"},
            {"subject": "Data_Science", "predicate": "uses", "object": "Machine_Learning"},
            {"subject": "Machine_Learning", "predicate": "requires", "object": "Statistics"},
            {"subject": "Web_Development", "predicate": "uses", "object": "HTML"},
            {"subject": "Web_Development", "predicate": "uses", "object": "CSS"},
        ]

        print(f"   Created {len(triples)} triples")
    except Exception as e:
        print(f"   FAILED: {e}")
        return False

    print("\n[2/2] Generating interactive visualization...")
    try:
        entities = [
            {"id": "Python", "type": "Language", "label": "Python"},
            {"id": "Programming_Language", "type": "Concept", "label": "Programming Language"},
            {"id": "Data_Science", "type": "Field", "label": "Data Science"},
            {"id": "Web_Development", "type": "Field", "label": "Web Development"},
            {"id": "Machine_Learning", "type": "Field", "label": "Machine Learning"},
            {"id": "Statistics", "type": "Field", "label": "Statistics"},
            {"id": "HTML", "type": "Technology", "label": "HTML"},
            {"id": "CSS", "type": "Technology", "label": "CSS"},
        ]

        result = await explorer.visualize(
            triples=triples,
            entities=entities,
            output_path="test_interactive.html"
        )

        print(f"   SUCCESS: Interactive visualization generated")
        print(f"   Output: {result.output_path}")
        print(f"   Nodes: {result.node_count}")
        print(f"   Edges: {result.edge_count}")

        if os.path.exists(result.output_path):
            print(f"   File exists: YES")
            print(f"   File size: {os.path.getsize(result.output_path)} bytes")
            return True
        else:
            print(f"   File exists: NO")
            return False
    except Exception as e:
        print(f"   FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  SPRINT 4: VISUALIZATION - COMPREHENSIVE TEST SUITE".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    # Test basic generation
    basic_success = await test_generate()

    # Test interactive generation
    interactive_success = await test_interactive()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"\nBasic Generation Test:    {'PASSED' if basic_success else 'FAILED'}")
    print(f"Interactive Generation:   {'PASSED' if interactive_success else 'FAILED'}")

    if basic_success and interactive_success:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED - SPRINT 4 IS FULLY FUNCTIONAL!")
        print("=" * 60)
        return 0
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED - REVIEW OUTPUT ABOVE")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
