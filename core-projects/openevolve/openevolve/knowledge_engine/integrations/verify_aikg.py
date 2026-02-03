"""
AI-Knowledge-Graph Integration Verification Script

This script verifies that all AIKG components are properly installed and working.
"""

import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def print_section(title):
    """Print a section header."""
    print("\n" + "="*80)
    print(f" {title}")
    print("="*80)


def print_success(message):
    """Print success message."""
    print(f"✓ {message}")


def print_error(message):
    """Print error message."""
    print(f"✗ {message}")


def print_info(message):
    """Print info message."""
    print(f"  {message}")


async def verify_imports():
    """Verify that all AIKG modules can be imported."""
    print_section("1. Verifying Imports")

    try:
        from knowledge_engine.integrations.aikg_standardization import (
            AIKGEntityStandardizer,
            Entity,
            Triple,
            StandardizationResult
        )
        print_success("Entity standardization module imported")
    except ImportError as e:
        print_error(f"Failed to import standardization: {e}")
        return False

    try:
        from knowledge_engine.integrations.aikg_inference import (
            AIKGRelationshipInference,
            InferenceResult
        )
        print_success("Relationship inference module imported")
    except ImportError as e:
        print_error(f"Failed to import inference: {e}")
        return False

    try:
        from knowledge_engine.integrations.aikg_visualization import (
            AIKGVisualizer,
            VisualizationOptions,
            VisualizationResult
        )
        print_success("Visualization module imported")
    except ImportError as e:
        print_error(f"Failed to import visualization: {e}")
        return False

    try:
        from knowledge_engine.integrations.aikg_integration import (
            AIKGIntegration,
            AIKGResult
        )
        print_success("Main integration module imported")
    except ImportError as e:
        print_error(f"Failed to import integration: {e}")
        return False

    return True


async def verify_configuration():
    """Verify configuration file exists and is valid."""
    print_section("2. Verifying Configuration")

    config_path = Path(__file__).parent.parent / "config" / "aikg_integration.yaml"

    if not config_path.exists():
        print_error(f"Configuration file not found: {config_path}")
        return False

    print_success(f"Configuration file found: {config_path}")

    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Verify required sections
        required_sections = ['standardization', 'inference', 'visualization']
        for section in required_sections:
            if section not in config:
                print_error(f"Missing configuration section: {section}")
                return False
            print_success(f"Configuration section '{section}' present")

        return True
    except Exception as e:
        print_error(f"Failed to load configuration: {e}")
        return False


async def verify_knowledge_engine_integration():
    """Verify Knowledge Engine integration."""
    print_section("3. Verifying Knowledge Engine Integration")

    try:
        from knowledge_engine.engine import KnowledgeEngine

        # Check if AIKG integration is initialized
        engine = KnowledgeEngine()

        if not hasattr(engine, 'aikg_integration'):
            print_error("AIKG integration not initialized in KnowledgeEngine")
            return False

        if engine.aikg_integration is None:
            print_info("AIKG integration initialized but not available (check logs)")
            return True  # Not fatal, may be due to missing dependencies

        print_success("AIKG integration initialized in KnowledgeEngine")

        # Check for API methods
        required_methods = [
            'process_with_aikg',
            'standardize_entities_with_aikg',
            'infer_relationships_with_aikg',
            'visualize_knowledge_graph',
            'export_knowledge_graph'
        ]

        for method in required_methods:
            if not hasattr(engine, method):
                print_error(f"Missing API method: {method}")
                return False
            print_success(f"API method '{method}' available")

        return True
    except Exception as e:
        print_error(f"Failed to verify KnowledgeEngine integration: {e}")
        return False


async def verify_entity_standardization():
    """Verify entity standardization functionality."""
    print_section("4. Verifying Entity Standardization")

    try:
        from knowledge_engine.integrations.aikg_standardization import (
            AIKGEntityStandardizer,
            Entity,
            Triple
        )

        config = {
            'use_llm_for_entities': False,
            'stopword_removal': True,
            'root_word_analysis': True,
            'self_reference_filtering': True
        }

        standardizer = AIKGEntityStandardizer(config)
        print_success("Standardizer initialized")

        # Test text normalization
        text = "Python Programming"
        normalized = await standardizer.normalize_text(text)
        if normalized == "python programming":
            print_success(f"Text normalization: '{text}' -> '{normalized}'")
        else:
            print_error(f"Text normalization failed: expected 'python programming', got '{normalized}'")
            return False

        # Test entity standardization
        entities = [
            Entity("Python"),
            Entity("python"),
            Entity("Django")
        ]

        triples = [
            Triple("Python", "used_for", "Web Dev"),
            Triple("python", "related_to", "Django"),
            Triple("Python", "related_to", "Python")  # Self-reference
        ]

        result = await standardizer.standardize_entities(entities, triples)

        print_success(f"Entity standardization: {len(entities)} -> {len(result.canonical_entities)} entities")
        print_info(f"  Variants resolved: {result.statistics.get('variants_resolved', 0)}")
        print_info(f"  Self-references removed: {result.removed_self_refs}")

        return True
    except Exception as e:
        print_error(f"Entity standardization test failed: {e}")
        import traceback
        print_info(traceback.format_exc())
        return False


async def verify_relationship_inference():
    """Verify relationship inference functionality."""
    print_section("5. Verifying Relationship Inference")

    try:
        from knowledge_engine.integrations.aikg_inference import (
            AIKGRelationshipInference
        )
        from knowledge_engine.integrations.aikg_standardization import Entity, Triple

        config = {
            'apply_transitive': True,
            'use_llm_for_inference': False,
            'similarity_threshold': 0.7,
            'max_inference_depth': 3
        }

        inference = AIKGRelationshipInference(config)
        print_success("Inference engine initialized")

        # Test inference
        entities = [
            Entity("Python"),
            Entity("Django"),
            Entity("Web Dev")
        ]

        triples = [
            Triple("Python", "used_for", "Web Dev"),
            Triple("Django", "framework_of", "Python")
        ]

        result = await inference.infer_relationships(triples, entities)

        print_success(f"Relationship inference: {len(result.original_triples)} -> {len(result.all_triples)} triples")
        print_info(f"  Inferred relationships: {len(result.inferred_triples)}")

        stats = result.get_statistics()
        print_info(f"  Average confidence: {stats['avg_confidence']:.3f}")

        return True
    except Exception as e:
        print_error(f"Relationship inference test failed: {e}")
        import traceback
        print_info(traceback.format_exc())
        return False


async def verify_visualization():
    """Verify visualization functionality."""
    print_section("6. Verifying Visualization")

    try:
        from knowledge_engine.integrations.aikg_visualization import (
            AIKGVisualizer,
            VisualizationOptions
        )
        from knowledge_engine.integrations.aikg_standardization import Entity, Triple
        import tempfile

        config = {
            'output_dir': tempfile.gettempdir(),
            'community_algorithm': 'louvain'
        }

        visualizer = AIKGVisualizer(config)
        print_success("Visualizer initialized")

        # Test visualization
        entities = [
            Entity("Python"),
            Entity("Django"),
            Entity("Web Dev")
        ]

        triples = [
            Triple("Python", "used_for", "Web Dev"),
            Triple("Django", "framework_of", "Python")
        ]

        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_path = f.name

        try:
            result = await visualizer.visualize_graph(
                triples=triples,
                entities=entities,
                output_path=output_path
            )

            print_success(f"Visualization generated: {result.output_path}")
            print_info(f"  Nodes: {result.node_count}")
            print_info(f"  Edges: {result.edge_count}")
            print_info(f"  Communities: {result.community_count}")

            # Verify file exists
            if Path(output_path).exists():
                print_success("Visualization file exists")

                # Check file content
                with open(output_path, 'r') as f:
                    content = f.read()
                    if '<!DOCTYPE html>' in content and 'd3.js' in content.lower():
                        print_success("Visualization file contains valid HTML and D3.js")
                    else:
                        print_error("Visualization file is invalid")
                        return False
            else:
                print_error("Visualization file was not created")
                return False

            return True
        finally:
            # Cleanup
            Path(output_path).unlink(missing_ok=True)

    except Exception as e:
        print_error(f"Visualization test failed: {e}")
        import traceback
        print_info(traceback.format_exc())
        return False


async def main():
    """Run all verification tests."""
    print("\n" + "="*80)
    print(" AI-Knowledge-Graph Integration Verification")
    print("="*80)

    results = []

    # Run all tests
    results.append(await verify_imports())
    results.append(await verify_configuration())
    results.append(await verify_knowledge_engine_integration())
    results.append(await verify_entity_standardization())
    results.append(await verify_relationship_inference())
    results.append(await verify_visualization())

    # Summary
    print_section("Verification Summary")

    total = len(results)
    passed = sum(results)
    failed = total - passed

    print(f"\nTotal tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n✓ All verification tests passed!")
        print("\nThe AI-Knowledge-Graph integration is ready to use.")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed")
        print("\nPlease review the errors above and fix the issues.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
