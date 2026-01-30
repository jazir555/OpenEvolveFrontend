"""
Comprehensive Import Test Suite for Knowledge Engine Module

This script tests EVERY import from the knowledge_engine module to ensure
all components are properly exported and can be imported.

Following CLAUDE.md principles:
- RUNTIME TRUTH: Actually execute all imports
- IDEMPOTENCY: Safe to run multiple times
- STRUCTURED LOGGING: JSON output format
"""

import sys
import json
from datetime import datetime
from typing import Dict, List, Any

class ImportTestResult:
    """Track individual import test results"""
    def __init__(self, import_path: str, items: List[str]):
        self.import_path = import_path
        self.items = items
        self.total_items = len(items)
        self.success = False
        self.error = None
        self.missing_items = []
        self.imported_objects = {}
        self.imported_count = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'import_path': self.import_path,
            'success': self.success,
            'error': str(self.error) if self.error else None,
            'missing_items': self.missing_items,
            'total_items': len(self.items),
            'imported_count': len(self.imported_objects)
        }


class ImportTestSuite:
    """Comprehensive import test suite"""

    def __init__(self):
        self.results = []
        self.start_time = datetime.now()

    def test_import(self, import_path: str, items: List[str]) -> ImportTestResult:
        """
        Test importing specific items from a module path

        Args:
            import_path: Module path to import from
            items: List of item names to import

        Returns:
            ImportTestResult with detailed results
        """
        result = ImportTestResult(import_path, items)

        try:
            # Dynamically import the module
            import importlib
            module = importlib.import_module(import_path)

            # Check if each item exists and can be accessed
            missing = []
            imported = {}

            for item_name in items:
                if hasattr(module, item_name):
                    obj = getattr(module, item_name)
                    imported[item_name] = type(obj).__name__
                else:
                    missing.append(item_name)

            result.imported_objects = imported
            result.missing_items = missing
            result.imported_count = len(imported)
            result.success = len(missing) == 0

            if not result.success:
                result.error = f"Missing items: {', '.join(missing)}"

        except ImportError as e:
            result.error = e
            result.success = False
        except (ImportError, AttributeError, RuntimeError) as e:
            result.error = e
            result.success = False

        self.results.append(result)
        return result

    def print_results(self):
        """Print test results in a formatted way"""
        print("\n" + "="*80)
        print("KNOWLEDGE ENGINE IMPORT TEST RESULTS")
        print("="*80)

        total_items = 0
        total_imported = 0
        failed_tests = 0

        for result in self.results:
            status = "[PASS]" if result.success else "[FAIL]"
            print(f"\n{status}: {result.import_path}")
            print(f"  Items: {result.imported_count}/{result.total_items}")

            total_items += result.total_items
            total_imported += result.imported_count

            if not result.success:
                failed_tests += 1
                print(f"  Error: {result.error}")

                if result.missing_items:
                    print(f"  Missing items:")
                    for item in result.missing_items:
                        print(f"    - {item}")

            elif result.imported_objects:
                print(f"  Imported items:")
                for item, obj_type in result.imported_objects.items():
                    print(f"    - {item}: {obj_type}")

        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total tests: {len(self.results)}")
        print(f"Passed: {len(self.results) - failed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Total items: {total_items}")
        print(f"Successfully imported: {total_imported}")
        print(f"Success rate: {(total_imported/total_items*100):.1f}%")

        duration = (datetime.now() - self.start_time).total_seconds()
        print(f"Duration: {duration:.2f}s")

        if failed_tests == 0 and total_imported == total_items:
            print("\n[SUCCESS] ALL TESTS PASSED - 100% IMPORT SUCCESS")
        else:
            print(f"\n[FAILURE] SOME TESTS FAILED - {total_items - total_imported} items missing")

    def export_json(self, filename: str = None):
        """Export results to JSON file"""
        if filename is None:
            filename = f"import_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        output = {
            'timestamp': datetime.now().isoformat(),
            'duration_seconds': (datetime.now() - self.start_time).total_seconds(),
            'total_tests': len(self.results),
            'passed_tests': sum(1 for r in self.results if r.success),
            'failed_tests': sum(1 for r in self.results if not r.success),
            'total_items': sum(r.total_items for r in self.results),
            'total_imported': sum(r.imported_count for r in self.results),
            'results': [r.to_dict() for r in self.results]
        }

        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults exported to: {filename}")


def main():
    """Run comprehensive import tests"""

    print("Starting comprehensive knowledge_engine import tests...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {sys.path[0]}")

    suite = ImportTestSuite()

    # Test 1: Main knowledge_engine imports
    print("\n[1/5] Testing main knowledge_engine imports...")
    suite.test_import(
        'knowledge_engine',
        [
            'KnowledgeEngine',
            'create_knowledge_engine',
            'ProcessingResult',
            'QueryResult',
            'KnowledgeState',
            'EntityKnowledgeGraph',
            'KnowledgeExtractor',
            'KnowledgeArtifact',
            'KnowledgeStorage',
            'KnowledgeRetriever',
            'IntegratedKnowledgeEngine'
        ]
    )

    # Test 2: Graphiti imports (Sprint 1)
    print("\n[2/5] Testing knowledge_engine.integrations.graphiti imports...")
    suite.test_import(
        'knowledge_engine.integrations.graphiti',
        [
            'GraphitiConfig',
            'validate_config',
            'GraphitiIntegrationError',
            'ConfigurationError',
            'ConnectionError',
            'ContradictionError',
            'InvalidTimestampError',
            'EpisodeProcessingError',
            'IncrementalUpdateError',
            'GraphitiTemporalBridge',
            'WorkflowArtifact',
            'WorkflowState',
            'TemporalFilter',
            'TemporalRelationship',
            'GraphitiAgentMemory',
            'AgentInteraction',
            'MemorySummary',
            'MemoryType',
            'GraphitiContradictionDetector',
            'Contradiction',
            'ContradictionReport',
            'ContradictionSeverity',
            'ResolutionAction',
            'GraphitiIncrementalUpdater',
            'GraphUpdate',
            'EntityMergeResult',
            'UpdateType',
            'UpdateStatus',
            'GraphitiHealthChecker',
            'HealthCheckResult',
            'SystemHealthReport',
            'health_check_quick'
        ]
    )

    # Test 3: KG-Gen imports (Sprint 2)
    print("\n[3/5] Testing knowledge_engine.integrations.kggen imports...")
    suite.test_import(
        'knowledge_engine.integrations.kggen',
        [
            'ExtractionPipeline',
            'ExtractionResult',
            'PipelineConfig',
            'PipelineStatus',
            'DeduplicationEngine',
            'DeduplicationResult',
            'SEMHASHStrategy',
            'LMClusterStrategy',
            'CrossDocumentResolver',
            'KGGenMCPServer',
            'MemoryManager',
            'MemoryTools',
            'ConversationAnalyzer',
            'ConversationResult',
            'SpeakerEntityExtractor',
            'GraphAggregator',
            'AggregationResult',
            'GraphVersion',
            'ConflictResolver'
        ]
    )

    # Test 4: OneKE imports (Sprint 3)
    print("\n[4/5] Testing knowledge_engine.integrations.oneke imports...")
    suite.test_import(
        'knowledge_engine.integrations.oneke',
        [
            'OneKEModelAdapter',
            'ModelConfig',
            'ExtractionResult',
            'MultiTaskExtractionFramework',
            'TaskType',
            'OneKESchemaManager',
            'SchemaDefinition',
            'CrossLingualEntityLinker',
            'EntityMatchResult',
            'EventExtractionPipeline',
            'EventChain',
            'TemporalEvent'
        ]
    )

    # Test 5: Visualization imports (Sprint 4)
    print("\n[5/5] Testing knowledge_engine.visualization imports...")
    suite.test_import(
        'knowledge_engine.visualization',
        [
            'GraphExplorer',
            'TemporalVisualizer',
            'CommunityVisualizer',
            'VisualizationOptions',
            'TemporalVisualizationOptions',
            'CommunityVisualizationOptions',
            'NodeFilter',
            'EdgeFilter',
            'VisualizationResult',
            'TemporalSnapshot',
            'TimeRange',
            'CommunityInfo',
            'ExportHandler',
            'VisualizationConfig'
        ]
    )

    # Print results
    suite.print_results()

    # Export to JSON
    suite.export_json()

    # Return exit code
    failed_tests = sum(1 for r in suite.results if not r.success)
    return 0 if failed_tests == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
