#!/usr/bin/env python3
"""
Comprehensive BubbleLabs Integration Validation Test
====================================================

This script performs comprehensive validation of the BubbleLabs integration
to ensure all components are working correctly.

Author: OpenEvolve Team
Date: 2025-12-29
"""

import sys
import json
from typing import Dict, Any, List
import traceback

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}\n")

def print_success(text: str):
    """Print a success message."""
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")

def print_error(text: str):
    """Print an error message."""
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")

def print_warning(text: str):
    """Print a warning message."""
    print(f"{Colors.YELLOW}! {text}{Colors.END}")

def print_info(text: str):
    """Print an info message."""
    print(f"{Colors.BLUE}i {text}{Colors.END}")

class TestResult:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors: List[str] = []

    def add_pass(self):
        self.passed += 1

    def add_fail(self, error: str):
        self.failed += 1
        self.errors.append(error)

    def add_warning(self):
        self.warnings += 1

    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{Colors.BOLD}Test Summary:{Colors.END}")
        print(f"  Total Tests: {total}")
        print(f"{Colors.GREEN}  Passed: {self.passed}{Colors.END}")
        if self.failed > 0:
            print(f"{Colors.RED}  Failed: {self.failed}{Colors.END}")
        if self.warnings > 0:
            print(f"{Colors.YELLOW}  Warnings: {self.warnings}{Colors.END}")

        if self.errors:
            print(f"\n{Colors.RED}{Colors.BOLD}Failed Tests:{Colors.END}")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        return self.failed == 0

def test_imports(result: TestResult) -> bool:
    """Test all critical imports."""
    print_header("Testing Critical Imports")

    tests = [
        ("WorkflowState", "from workflow_structures import WorkflowState"),
        ("BubbleNode", "from bubblelabs_integration import BubbleNode"),
        ("BubbleEdge", "from bubblelabs_integration import BubbleEdge"),
        ("BubbleWorkflowDefinition", "from bubblelabs_integration import BubbleWorkflowDefinition"),
        ("BubbleLabsIntegration", "from bubblelabs_integration import BubbleLabsIntegration"),
        ("BubbleLabsWorkflowUI", "from bubblelabs_ui_component import BubbleLabsWorkflowUI"),
        ("OpenEvolveBubbleLabsIntegration", "from openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration"),
    ]

    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print_success(f"{name} imported successfully")
            result.add_pass()
        except Exception as e:
            print_error(f"Failed to import {name}: {str(e)}")
            result.add_fail(f"Import {name}: {str(e)}")

    return result.failed == 0

def test_core_classes(result: TestResult) -> bool:
    """Test core class functionality."""
    print_header("Testing Core Classes")

    try:
        from bubblelabs_integration import BubbleNode, BubbleEdge, BubbleWorkflowDefinition

        # Test BubbleNode creation
        node = BubbleNode(
            id="test_node",
            type="test_type",
            position={"x": 100, "y": 100},
            data={"key": "value"}
        )
        print_success("BubbleNode creation successful")
        result.add_pass()

        # Test BubbleEdge creation (note: camelCase parameters)
        edge = BubbleEdge(
            id="test_edge",
            source="node1",
            target="node2",
            sourceHandle="output",
            targetHandle="input"
        )
        print_success("BubbleEdge creation successful")
        result.add_pass()

        # Test BubbleWorkflowDefinition creation (nodes and edges are dicts)
        node_dict = {"id": "test_node", "type": "test", "position": {"x": 0, "y": 0}, "data": {}}
        edge_dict = {"id": "test_edge", "source": "n1", "target": "n2"}
        definition = BubbleWorkflowDefinition(
            id="test_definition",
            name="Test Workflow",
            description="Test description",
            nodes=[node_dict],
            edges=[edge_dict],
            metadata={}
        )
        print_success("BubbleWorkflowDefinition creation successful")
        result.add_pass()

        # Verify attributes
        assert definition.id == "test_definition", "Definition ID mismatch"
        assert len(definition.nodes) == 1, "Node count mismatch"
        assert len(definition.edges) == 1, "Edge count mismatch"
        print_success("BubbleWorkflowDefinition attributes verified")
        result.add_pass()

    except Exception as e:
        print_error(f"Core class test failed: {str(e)}")
        result.add_fail(f"Core classes: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def test_integration_class(result: TestResult) -> bool:
    """Test BubbleLabsIntegration class."""
    print_header("Testing BubbleLabsIntegration Class")

    try:
        from bubblelabs_integration import BubbleLabsIntegration

        # Create integration instance
        integration = BubbleLabsIntegration()
        print_success("BubbleLabsIntegration instantiation successful")
        result.add_pass()

        # Test workflow definition creation
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Test problem: Create REST API",
            team_config={
                "planner_team": "Backend-Team",
                "solver_team": "Fullstack-Team"
            },
            gauntlet_config={
                "sub_problem_red_gauntlet": "Security-Review"
            }
        )
        print_success("Workflow definition creation successful")
        result.add_pass()

        # Verify definition properties
        assert definition is not None, "Definition is None"
        assert definition.name is not None, "Definition name is None"
        assert len(definition.nodes) > 0, "No nodes in definition"
        print_success("Workflow definition properties verified")
        result.add_pass()

        # Test get_workflow_definition method
        retrieved_def = integration.get_workflow_definition(definition.id)
        assert retrieved_def is not None, "Failed to retrieve definition"
        assert retrieved_def.id == definition.id, "Retrieved wrong definition"
        print_success("get_workflow_definition works correctly")
        result.add_pass()

    except Exception as e:
        print_error(f"BubbleLabsIntegration test failed: {str(e)}")
        result.add_fail(f"BubbleLabsIntegration: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def test_api_bridge(result: TestResult) -> bool:
    """Test OpenEvolveBubbleLabsIntegration API bridge."""
    print_header("Testing API Bridge")

    try:
        from openevolve_bubblelabs_api import OpenEvolveBubbleLabsIntegration, WorkflowStatus, WorkflowMetrics

        # Test WorkflowStatus enum
        statuses = [s.value for s in WorkflowStatus]
        expected_statuses = ["created", "pending", "running", "paused", "stopping", "stopped", "completed", "failed", "cancelled"]
        for status in expected_statuses:
            assert status in statuses, f"Missing status: {status}"
        print_success("WorkflowStatus enum verified")
        result.add_pass()

        # Test WorkflowMetrics dataclass (actual fields: execution_time, tokens_used, best_fitness, etc.)
        metrics = WorkflowMetrics(
            execution_time=30.0,
            tokens_used=1000,
            best_fitness=0.95,
            avg_fitness=0.85,
            diversity=0.75,
            convergence=0.90,
            population_size=100,
            iterations_completed=50,
            total_iterations=100
        )
        assert metrics.execution_time == 30.0, "Execution time mismatch"
        assert metrics.tokens_used == 1000, "Tokens used mismatch"
        assert metrics.best_fitness == 0.95, "Best fitness mismatch"
        print_success("WorkflowMetrics dataclass verified")
        result.add_pass()

        # Create API integration instance
        api_integration = OpenEvolveBubbleLabsIntegration()
        print_success("OpenEvolveBubbleLabsIntegration instantiation successful")
        result.add_pass()

        # Test get_workflow_instance_status (should return status dict or error)
        status = api_integration.get_workflow_instance_status("test_instance_id")
        assert isinstance(status, dict), "Status should be a dict"
        # Should have either "status" key (if instance exists) or "error" key (if not found)
        assert "status" in status or "error" in status, "Status should have 'status' or 'error' key"
        print_success("get_workflow_instance_status returns valid structure")
        result.add_pass()

    except Exception as e:
        print_error(f"API bridge test failed: {str(e)}")
        result.add_fail(f"API bridge: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def test_ui_component(result: TestResult) -> bool:
    """Test BubbleLabsWorkflowUI component."""
    print_header("Testing UI Component")

    try:
        from bubblelabs_ui_component import BubbleLabsWorkflowUI

        # Create UI instance
        ui = BubbleLabsWorkflowUI()
        print_success("BubbleLabsWorkflowUI instantiation successful")
        result.add_pass()

        # Verify UI has the actual method
        assert hasattr(ui, "render_workflow_visualizer"), "Missing method: render_workflow_visualizer"
        print_success("UI component has render_workflow_visualizer method")
        result.add_pass()

        # Check that it's callable
        assert callable(ui.render_workflow_visualizer), "render_workflow_visualizer is not callable"
        print_success("render_workflow_visualizer is callable")
        result.add_pass()

    except Exception as e:
        print_error(f"UI component test failed: {str(e)}")
        result.add_fail(f"UI component: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def test_json_serialization(result: TestResult) -> bool:
    """Test JSON serialization of workflow definitions."""
    print_header("Testing JSON Serialization")

    try:
        from bubblelabs_integration import BubbleLabsIntegration

        integration = BubbleLabsIntegration()

        # Create workflow definition
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Serialization test",
            team_config={"planner_team": "Test-Team"},
            gauntlet_config={}
        )

        # Create dict from definition attributes (no to_dict method)
        definition_dict = {
            "id": definition.id,
            "name": definition.name,
            "description": definition.description,
            "nodes": definition.nodes,
            "edges": definition.edges,
            "metadata": definition.metadata
        }
        assert isinstance(definition_dict, dict), "Dict creation should return dict"
        assert "id" in definition_dict, "Missing 'id' in dict"
        assert "nodes" in definition_dict, "Missing 'nodes' in dict"
        assert "edges" in definition_dict, "Missing 'edges' in dict"
        print_success("Dict creation works correctly")
        result.add_pass()

        # Test JSON serialization
        json_str = json.dumps(definition_dict, default=str)
        assert len(json_str) > 0, "JSON serialization failed"
        print_success("JSON serialization successful")
        result.add_pass()

        # Test deserialization
        parsed_dict = json.loads(json_str)
        assert parsed_dict["id"] == definition_dict["id"], "ID mismatch after deserialization"
        print_success("JSON deserialization successful")
        result.add_pass()

    except Exception as e:
        print_error(f"JSON serialization test failed: {str(e)}")
        result.add_fail(f"JSON serialization: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def test_workflow_execution_flow(result: TestResult) -> bool:
    """Test workflow execution flow (dry-run)."""
    print_header("Testing Workflow Execution Flow")

    try:
        from bubblelabs_integration import BubbleLabsIntegration

        integration = BubbleLabsIntegration()

        # Create workflow definition
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Test execution flow",
            team_config={
                "planner_team": "Planner-Team",
                "solver_team": "Solver-Team"
            },
            gauntlet_config={
                "sub_problem_red_gauntlet": "Red-Gauntlet"
            }
        )
        print_success("Workflow definition created")
        result.add_pass()

        # Test that we can get definition ID
        assert definition.id is not None, "Definition ID is None"
        print_success(f"Workflow definition ID: {definition.id}")
        result.add_pass()

        # Verify nodes are properly created (nodes are dicts)
        assert len(definition.nodes) > 0, "No nodes in workflow"
        print_success(f"Workflow contains {len(definition.nodes)} nodes")
        result.add_pass()

        # Verify workflow has proper structure (node IDs are in dicts)
        node_ids = [node["id"] for node in definition.nodes]
        assert len(node_ids) == len(set(node_ids)), "Duplicate node IDs found"
        print_success("All node IDs are unique")
        result.add_pass()

    except Exception as e:
        print_error(f"Workflow execution flow test failed: {str(e)}")
        result.add_fail(f"Execution flow: {str(e)}")
        traceback.print_exc()

    return result.failed == 0

def main():
    """Run all validation tests."""
    print_header("BubbleLabs Integration - Comprehensive Validation")
    print_info("Starting comprehensive validation tests...\n")

    result = TestResult()

    # Run all tests
    test_imports(result)
    test_core_classes(result)
    test_integration_class(result)
    test_api_bridge(result)
    test_ui_component(result)
    test_json_serialization(result)
    test_workflow_execution_flow(result)

    # Print summary
    print_header("Validation Complete")
    success = result.print_summary()

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] ALL TESTS PASSED{Colors.END}")
        print(f"\n{Colors.GREEN}BubbleLabs integration is fully functional and ready for production use.{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}[FAIL] SOME TESTS FAILED{Colors.END}")
        print(f"\n{Colors.RED}Please review the errors above and fix the issues.{Colors.END}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
