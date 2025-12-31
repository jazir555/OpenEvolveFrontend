"""
Test CRITICAL Edge Case Fixes

This file tests the fixes for the 2 CRITICAL edge cases:
1. No None check on workflow_definition in bubblelabs_hephaestus_bridge.py
2. No None check on workflow_definition in bubblelabs_typescript_export.py

These tests verify that the code handles None inputs gracefully without crashing.
"""

import sys
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Mock imports since we may not have all dependencies
class MockWorkflowDefinition:
    """Mock workflow definition for testing."""
    def __init__(self, workflow_id: str, name: str):
        self.id = workflow_id
        self.name = name
        self.description = "Test workflow"
        self.nodes = [{"id": "node1", "type": "test"}]
        self.edges = []
        self.metadata = {}


class MockBubbleLabsIntegration:
    """Mock BubbleLabs integration for testing."""
    def get_workflow_definition(self, workflow_id: str):
        """Return None to test edge case handling."""
        return None  # Simulates API returning None

    def list_workflow_definitions(self):
        """Return list with None values to test edge case handling."""
        return [
            MockWorkflowDefinition("workflow1", "Valid Workflow"),
            None,  # CRITICAL: None in list
            MockWorkflowDefinition("workflow2", "Another Valid"),
            None,  # CRITICAL: Another None
        ]


def test_edge_case_1_sync_workflow_to_ticket():
    """
    TEST 1: sync_workflow_to_ticket with None workflow

    CRITICAL EDGE CASE: No None check on workflow_definition (bubblelabs_hephaestus_bridge.py line 128)

    BEFORE FIX: Would crash with AttributeError when trying to access workflow.id
    AFTER FIX: Returns False with proper error logging

    Expected Result: False (graceful failure, no crash)
    """
    print("\n" + "="*80)
    print("TEST 1: CRITICAL - sync_workflow_to_ticket with None workflow")
    print("="*80)

    try:
        from bubblelabs_hephaestus_bridge import BubbleLabsHephaestusBridge

        # Create bridge with mock integration
        bridge = BubbleLabsHephaestusBridge(
            bubblelabs_integration=MockBubbleLabsIntegration(),
            hephaestus_client=None
        )

        # Test with workflow that returns None
        result = bridge.sync_workflow_to_ticket("test_workflow_id")

        print(f"[PASS] Test PASSED: Function returned {result} (no crash)")
        print(f"  Expected: False")
        print(f"  Got: {result}")
        assert result == False, f"Expected False, got {result}"
        print("[PASS] No AttributeError - None check is working correctly!")
        return True

    except AttributeError as e:
        print(f"[FAIL] Test FAILED: AttributeError still occurs!")
        print(f"  Error: {e}")
        print("  This means the None check is NOT working properly")
        return False
    except Exception as e:
        print(f"[FAIL] Test FAILED: Unexpected error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        return False


def test_edge_case_2_export_workflow_with_none():
    """
    TEST 2: export_workflow with None workflow_definition

    CRITICAL EDGE CASE: No None check on workflow_definition (bubblelabs_typescript_export.py line 183)

    BEFORE FIX: Would crash with AttributeError when trying to access workflow.name or workflow.id
    AFTER FIX: Returns ExportResult with error message

    Expected Result: ExportResult(success=False, error="workflow_definition is required")
    """
    print("\n" + "="*80)
    print("TEST 2: CRITICAL - export_workflow with None workflow_definition")
    print("="*80)

    try:
        from bubblelabs_typescript_export import BubbleLabsTypeScriptExporter, ExportResult

        # Create exporter
        exporter = BubbleLabsTypeScriptExporter()

        # Test with None workflow_definition
        result = exporter.export_workflow(None)

        print(f"[PASS] Test PASSED: Function returned ExportResult (no crash)")
        print(f"  Result.success: {result.success}")
        print(f"  Result.error: {result.error}")
        print(f"  Result.code: {result.code}")

        assert result.success == False, f"Expected success=False, got {result.success}"
        assert result.error is not None, "Expected error message, got None"
        assert "workflow_definition" in result.error.lower() or "required" in result.error.lower(), \
            f"Expected error about workflow_definition, got: {result.error}"
        assert result.code is None, f"Expected code=None, got {result.code}"

        print("[PASS] No AttributeError - None check is working correctly!")
        print("[PASS] Proper error response returned!")
        return True

    except AttributeError as e:
        print(f"[FAIL] Test FAILED: AttributeError still occurs!")
        print(f"  Error: {e}")
        print("  This means the None check is NOT working properly")
        return False
    except Exception as e:
        print(f"[FAIL] Test FAILED: Unexpected error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        return False


def test_edge_case_3_export_workflow_missing_attributes():
    """
    TEST 3: export_workflow with workflow missing required attributes

    CRITICAL EDGE CASE: Validate workflow has required attributes

    BEFORE FIX: Would crash when trying to access workflow.id, workflow.name, or workflow.nodes
    AFTER FIX: Returns ExportResult with specific error about missing attribute

    Expected Result: ExportResult(success=False, error="missing 'id' attribute")
    """
    print("\n" + "="*80)
    print("TEST 3: CRITICAL - export_workflow with workflow missing attributes")
    print("="*80)

    try:
        from bubblelabs_typescript_export import BubbleLabsTypeScriptExporter, ExportResult

        # Create exporter
        exporter = BubbleLabsTypeScriptExporter()

        # Create a mock workflow with missing 'id' attribute
        class IncompleteWorkflow:
            def __init__(self):
                self.name = "Test"
                self.nodes = []
                # Missing 'id' attribute!

        incomplete_workflow = IncompleteWorkflow()

        # Test with incomplete workflow
        result = exporter.export_workflow(incomplete_workflow)

        print(f"[PASS] Test PASSED: Function returned ExportResult (no crash)")
        print(f"  Result.success: {result.success}")
        print(f"  Result.error: {result.error}")

        assert result.success == False, f"Expected success=False, got {result.success}"
        assert result.error is not None, "Expected error message, got None"
        assert "id" in result.error.lower(), f"Expected error about 'id' attribute, got: {result.error}"

        print("[PASS] No AttributeError - Attribute validation is working!")
        print("[PASS] Proper error response for missing attribute!")
        return True

    except AttributeError as e:
        print(f"[FAIL] Test FAILED: AttributeError still occurs!")
        print(f"  Error: {e}")
        print("  This means the attribute validation is NOT working")
        return False
    except Exception as e:
        print(f"[FAIL] Test FAILED: Unexpected error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        return False


def test_edge_case_4_export_all_workflows_with_none_list():
    """
    TEST 4: export_all_workflows with None values in list

    CRITICAL EDGE CASE: Validate workflows list contains no None values

    BEFORE FIX: Would crash when trying to access definition.id on None
    AFTER FIX: Skips None workflows with error logging

    Expected Result: Exports valid workflows, skips None ones
    """
    print("\n" + "="*80)
    print("TEST 4: CRITICAL - export_all_workflows with None in list")
    print("="*80)

    try:
        from bubblelabs_typescript_export import export_all_workflows
        from unittest.mock import patch

        # Mock the BubbleLabsIntegration to return list with None values
        with patch('bubblelabs_typescript_export.BubbleLabsIntegration') as MockIntegration:
            mock_instance = MockIntegration.return_value
            mock_instance.list_workflow_definitions.return_value = [
                MockWorkflowDefinition("workflow1", "Valid 1"),
                None,  # CRITICAL: None in list
                MockWorkflowDefinition("workflow2", "Valid 2"),
                None,  # CRITICAL: Another None
            ]

            # Export workflows
            count, results = export_all_workflows("./test_export")

            print(f"[PASS] Test PASSED: Function handled None values in list (no crash)")
            print(f"  Total workflows: 4 (2 valid, 2 None)")
            print(f"  Successfully exported: {count}")
            print(f"  Total results: {len(results)}")
            print(f"  Failed results: {sum(1 for r in results if not r.success)}")

            # Should export 2 valid workflows and fail 2 None ones
            assert count == 2, f"Expected 2 successful exports, got {count}"
            assert len(results) == 4, f"Expected 4 results, got {len(results)}"
            assert sum(1 for r in results if not r.success) == 2, \
                f"Expected 2 failures (for None workflows), got {sum(1 for r in results if not r.success)}"

            print("[PASS] No AttributeError - None in list handling is working!")
            print("[PASS] Valid workflows exported, None workflows skipped with errors!")
            return True

    except AttributeError as e:
        print(f"[FAIL] Test FAILED: AttributeError still occurs!")
        print(f"  Error: {e}")
        print("  This means the None-in-list check is NOT working")
        return False
    except Exception as e:
        print(f"[FAIL] Test FAILED: Unexpected error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edge_case_5_sync_workflow_with_invalid_attributes():
    """
    TEST 5: sync_workflow_to_ticket with workflow having invalid attributes

    CRITICAL EDGE CASE: Validate workflow has valid 'id' and 'name' attributes

    BEFORE FIX: Would crash when trying to access workflow.id or workflow.name
    AFTER FIX: Returns False with error logging

    Expected Result: False (graceful failure)
    """
    print("\n" + "="*80)
    print("TEST 5: CRITICAL - sync_workflow_to_ticket with invalid workflow")
    print("="*80)

    try:
        from bubblelabs_hephaestus_bridge import BubbleLabsHephaestusBridge

        # Mock that returns workflow with missing 'id'
        class MockBadWorkflow:
            def __init__(self):
                self.name = "Bad Workflow"
                # Missing 'id' attribute!

        class MockBubbleLabsBad:
            def get_workflow_definition(self, workflow_id: str):
                return MockBadWorkflow()  # Returns workflow without 'id'

        # Create bridge
        bridge = BubbleLabsHephaestusBridge(
            bubblelabs_integration=MockBubbleLabsBad(),
            hephaestus_client=None
        )

        # Test with workflow missing 'id'
        result = bridge.sync_workflow_to_ticket("test_workflow_id")

        print(f"[PASS] Test PASSED: Function returned {result} (no crash)")
        print(f"  Expected: False")
        print(f"  Got: {result}")
        assert result == False, f"Expected False, got {result}"
        print("[PASS] No AttributeError - Attribute validation is working!")
        return True

    except AttributeError as e:
        print(f"[FAIL] Test FAILED: AttributeError still occurs!")
        print(f"  Error: {e}")
        print("  This means the attribute validation is NOT working")
        return False
    except Exception as e:
        print(f"[FAIL] Test FAILED: Unexpected error")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error: {e}")
        return False


def run_all_tests():
    """Run all edge case tests."""
    print("\n" + "="*80)
    print("TESTING CRITICAL EDGE CASE FIXES")
    print("="*80)
    print("\nThese tests verify that the code handles None inputs gracefully")
    print("without crashing with AttributeError.\n")

    results = []

    # Run all tests
    results.append(("Edge Case 1: sync_workflow_to_ticket with None workflow",
                   test_edge_case_1_sync_workflow_to_ticket()))

    results.append(("Edge Case 2: export_workflow with None workflow_definition",
                   test_edge_case_2_export_workflow_with_none()))

    results.append(("Edge Case 3: export_workflow with missing attributes",
                   test_edge_case_3_export_workflow_missing_attributes()))

    results.append(("Edge Case 4: export_all_workflows with None in list",
                   test_edge_case_4_export_all_workflows_with_none_list()))

    results.append(("Edge Case 5: sync_workflow_to_ticket with invalid workflow",
                   test_edge_case_5_sync_workflow_with_invalid_attributes()))

    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    passed = sum(1 for _, result in results if result)
    failed = sum(1 for _, result in results if not result)
    total = len(results)

    for test_name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {total} tests")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed == 0:
        print("\n[PASS][PASS][PASS] ALL TESTS PASSED! [PASS][PASS][PASS]")
        print("All CRITICAL edge cases are properly handled.")
        return True
    else:
        print(f"\n[FAIL][FAIL][FAIL] {failed} TESTS FAILED [FAIL][FAIL][FAIL]")
        print("Some edge cases still need fixing!")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
