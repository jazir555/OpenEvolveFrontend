#!/usr/bin/env python3
"""
Comprehensive Test Suite for BubbleLabs Complete Integration

Tests all newly implemented components:
1. BubbleLabs-Hephaestus Bridge
2. BubbleLabs MCP Tools
3. BubbleLabs Analytics
4. BubbleLabs TypeScript Export

Author: OpenEvolve Team
Date: 2025-12-29
"""

import sys
import os
import tempfile
import traceback
from typing import List, Dict, Any

# Test result tracking
class TestResult:
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

    def add_warning(self, warning: str):
        self.warnings += 1
        print(f"[!] WARNING: {warning}")

    def print_summary(self):
        total = self.passed + self.failed
        print("\n" + "=" * 70)
        print("Test Summary")
        print("=" * 70)
        print(f"Total Tests: {total}")
        print(f"[OK] Passed: {self.passed}")
        if self.failed > 0:
            print(f"[FAIL] Failed: {self.failed}")
        if self.warnings > 0:
            print(f"[!] Warnings: {self.warnings}")

        if self.errors:
            print("\nFailed Tests:")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")

        return self.failed == 0


# Color codes
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(70)}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'=' * 70}{Colors.END}\n")


def print_success(text: str):
    print(f"{Colors.GREEN}[OK] {text}{Colors.END}")


def print_error(text: str):
    print(f"{Colors.RED}[FAIL] {text}{Colors.END}")


def print_info(text: str):
    print(f"{Colors.BLUE}[i] {text}{Colors.END}")


# =============================================================================
# HEPHAESTUS BRIDGE TESTS
# =============================================================================

def test_hephaestus_bridge(result: TestResult):
    """Test BubbleLabs-Hephaestus bridge."""
    print_header("Testing BubbleLabs-Hephaestus Bridge")

    try:
        from bubblelabs_hephaestus_bridge import (
            BubbleLabsHephaestusBridge,
            BubbleLabsTicketConfig,
            create_bridge
        )

        print_success("Hephaestus bridge imports successful")
        result.add_pass()

        # Test config creation
        config = BubbleLabsTicketConfig(
            auto_create_tickets=True,
            auto_update_progress=True,
            auto_close_on_completion=True,
            ticket_prefix="TEST-",
            ticket_type="story"
        )
        print_success("TicketConfig created")
        result.add_pass()

        # Test bridge creation
        bridge = create_bridge(
            hephaestus_api_base="http://localhost:8000",
            hephaestus_api_key="test-key",
            hephaestus_project_id="test-project",
            config=config
        )
        print_success("Bridge created successfully")
        result.add_pass()

        # Verify bridge attributes
        assert hasattr(bridge, 'bubblelabs'), "Missing bubblelabs attribute"
        assert hasattr(bridge, 'config'), "Missing config attribute"
        assert hasattr(bridge, 'mappings'), "Missing mappings attribute"
        print_success("Bridge has all required attributes")
        result.add_pass()

        # Test workflow creation
        from bubblelabs_integration import BubbleLabsIntegration
        integration = BubbleLabsIntegration()
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Test workflow for Hephaestus integration",
            team_config={"planner_team": "Test-Team"},
            gauntlet_config={}
        )
        print_success("Test workflow created")
        result.add_pass()

        # Test ticket creation (mock mode)
        ticket_id = bridge.create_ticket_from_workflow(definition)
        assert ticket_id is not None, "Ticket ID should not be None"
        print_success(f"Mock ticket created: {ticket_id}")
        result.add_pass()

        # Test mapping retrieval
        mapping_ticket_id = bridge.get_ticket_for_workflow(definition.id)
        assert mapping_ticket_id == ticket_id, "Ticket ID mismatch in mapping"
        print_success("Workflow-to-ticket mapping verified")
        result.add_pass()

        # Test progress update (mock mode)
        from openevolve_bubblelabs_api import WorkflowStatus
        success = bridge.update_ticket_progress(
            workflow_instance_id="test-instance",
            progress=0.5,
            status=WorkflowStatus.RUNNING
        )
        assert success, "Progress update failed"
        print_success("Progress update successful")
        result.add_pass()

        # Test get all mappings
        mappings = bridge.get_all_mappings()
        assert isinstance(mappings, dict), "Mappings should be a dict"
        assert definition.id in mappings, "Workflow ID not in mappings"
        print_success(f"Retrieved all mappings: {len(mappings)} workflow(s)")
        result.add_pass()

    except Exception as e:
        print_error(f"Hephaestus bridge test failed: {str(e)}")
        result.add_fail(f"Hephaestus bridge: {str(e)}")
        traceback.print_exc()


# =============================================================================
# MCP TOOLS TESTS
# =============================================================================

def test_mcp_tools(result: TestResult):
    """Test BubbleLabs MCP tools."""
    print_header("Testing BubbleLabs MCP Tools")

    try:
        from bubblelabs_mcp_tools import (
            create_bubblelabs_workflow,
            execute_bubblelabs_workflow,
            get_bubblelabs_workflow_status,
            control_bubblelabs_workflow,
            list_bubblelabs_workflows,
            get_bubblelabs_workflow_results,
            list_mcp_tools,
            get_mcp_tool
        )

        print_success("MCP tools imports successful")
        result.add_pass()

        # Test tool listing
        tools = list_mcp_tools()
        assert len(tools) > 0, "No MCP tools registered"
        print_success(f"Registered MCP tools: {len(tools)}")
        result.add_pass()

        # Verify expected tools exist
        expected_tools = [
            "create_bubblelabs_workflow",
            "execute_bubblelabs_workflow",
            "get_bubblelabs_workflow_status",
            "control_bubblelabs_workflow",
            "list_bubblelabs_workflows",
            "get_bubblelabs_workflow_results"
        ]
        for tool in expected_tools:
            assert tool in tools, f"Missing tool: {tool}"
        print_success("All expected MCP tools registered")
        result.add_pass()

        # Test create workflow tool
        create_result = create_bubblelabs_workflow(
            problem_statement="Test MCP workflow creation",
            team_config={"planner_team": "MCP-Test-Team"},
            workflow_name="MCP Test Workflow"
        )
        assert create_result["success"], f"Workflow creation failed: {create_result.get('error')}"
        workflow_id = create_result["workflow_id"]
        print_success(f"Created workflow via MCP tool: {workflow_id}")
        result.add_pass()

        # Test list workflows tool
        list_result = list_bubblelabs_workflows()
        assert list_result["success"], "List workflows failed"
        assert len(list_result["definitions"]) > 0, "No workflows found"
        print_success(f"Listed {len(list_result['definitions'])} workflow(s)")
        result.add_pass()

        # Test get workflow status tool
        status_result = get_bubblelabs_workflow_status("test-instance-id")
        assert status_result["success"], "Get status failed"
        assert "instance_id" in status_result, "Missing instance_id in result"
        print_success("Workflow status retrieval successful")
        result.add_pass()

        # Test control workflow tool
        control_result = control_bubblelabs_workflow(
            instance_id="test-instance-id",
            action="pause"
        )
        # Note: This will fail with mock instance, but we test the API structure
        assert "success" in control_result, "Missing success in control result"
        print_success("Workflow control tool structure verified")
        result.add_pass()

    except Exception as e:
        print_error(f"MCP tools test failed: {str(e)}")
        result.add_fail(f"MCP tools: {str(e)}")
        traceback.print_exc()


# =============================================================================
# ANALYTICS TESTS
# =============================================================================

def test_analytics(result: TestResult):
    """Test BubbleLabs analytics."""
    print_header("Testing BubbleLabs Analytics")

    try:
        from bubblelabs_analytics import (
            BubbleLabsAnalytics,
            ProviderCostConfig,
            create_analytics_tracker
        )

        print_success("Analytics imports successful")
        result.add_pass()

        # Create temporary database
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            # Create analytics tracker
            analytics = create_analytics_tracker(db_path)
            print_success("Analytics tracker created")
            result.add_pass()

            # Start workflow tracking
            workflow_id = "test-workflow-analytics"
            success = analytics.start_workflow_tracking(
                workflow_id=workflow_id,
                workflow_name="Test Analytics Workflow",
                instance_id="instance-analytics-123"
            )
            assert success, "Failed to start workflow tracking"
            print_success("Started workflow tracking")
            result.add_pass()

            # Track node executions
            success = analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id="node-1",
                node_type="decomposer",
                tokens_used=1000,
                execution_time=5.2,
                provider="openai",
                input_tokens=500,
                output_tokens=500
            )
            assert success, "Failed to track node execution"
            print_success("Tracked node execution")
            result.add_pass()

            success = analytics.track_node_execution(
                workflow_id=workflow_id,
                node_id="node-2",
                node_type="solver",
                tokens_used=1500,
                execution_time=8.7,
                provider="anthropic",
                input_tokens=750,
                output_tokens=750
            )
            assert success, "Failed to track second node execution"
            print_success("Tracked second node execution")
            result.add_pass()

            # End workflow tracking
            success = analytics.end_workflow_tracking(
                workflow_id=workflow_id,
                status="completed"
            )
            assert success, "Failed to end workflow tracking"
            print_success("Ended workflow tracking")
            result.add_pass()

            # Get workflow analytics
            workflow_analytics = analytics.get_workflow_analytics(workflow_id)
            assert workflow_analytics is not None, "Failed to get workflow analytics"
            assert workflow_analytics.total_tokens == 2500, "Token count mismatch"
            assert workflow_analytics.total_cost > 0, "Cost should be > 0"
            assert len(workflow_analytics.node_metrics) == 2, "Node metrics count mismatch"
            print_success(f"Retrieved analytics: {workflow_analytics.total_tokens} tokens, ${workflow_analytics.total_cost:.6f}")
            result.add_pass()

            # Get analytics summary
            summary = analytics.get_analytics_summary()
            assert summary["total_workflows"] == 1, "Workflow count mismatch"
            assert summary["total_tokens"] == 2500, "Total tokens mismatch"
            assert summary["completed_workflows"] == 1, "Completed workflows count mismatch"
            print_success(f"Analytics summary: {summary['total_workflows']} workflow(s), ${summary['total_cost']:.6f}")
            result.add_pass()

            # Get cost breakdown
            breakdown = analytics.get_cost_breakdown(workflow_id)
            assert breakdown["total_cost"] > 0, "Total cost should be > 0"
            assert "openai" in breakdown["providers"], "OpenAI not in breakdown"
            assert "anthropic" in breakdown["providers"], "Anthropic not in breakdown"
            print_success(f"Cost breakdown: {len(breakdown['providers'])} provider(s)")
            result.add_pass()

            # Test export report
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
                report_path = tmp.name

            success = analytics.export_analytics_report(report_path, format="json")
            assert success, "Failed to export analytics report"
            assert os.path.exists(report_path), "Report file not created"
            print_success(f"Exported analytics report: {report_path}")
            result.add_pass()

            # Clean up report file
            os.unlink(report_path)

        finally:
            # Clean up database
            if os.path.exists(db_path):
                os.unlink(db_path)

    except Exception as e:
        print_error(f"Analytics test failed: {str(e)}")
        result.add_fail(f"Analytics: {str(e)}")
        traceback.print_exc()


# =============================================================================
# TYPESCRIPT EXPORT TESTS
# =============================================================================

def test_typescript_export(result: TestResult):
    """Test BubbleLabs TypeScript export."""
    print_header("Testing BubbleLabs TypeScript Export")

    try:
        from bubblelabs_typescript_export import (
            BubbleLabsTypeScriptExporter,
            TypeScriptExportConfig,
            export_workflow_to_typescript,
            export_all_workflows
        )

        print_success("TypeScript export imports successful")
        result.add_pass()

        # Create test workflow
        from bubblelabs_integration import BubbleLabsIntegration
        integration = BubbleLabsIntegration()
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Test TypeScript export",
            team_config={"planner_team": "Export-Test-Team"},
            gauntlet_config={}
        )
        print_success("Created test workflow for export")
        result.add_pass()

        # Test module export
        config = TypeScriptExportConfig(
            include_comments=True,
            include_error_handling=True,
            include_logging=True,
            export_format="module"
        )
        exporter = BubbleLabsTypeScriptExporter(config)
        export_result = exporter.export_workflow(definition)

        assert export_result.success, f"Export failed: {export_result.error}"
        assert export_result.code is not None, "No code generated"
        assert len(export_result.code) > 0, "Empty code generated"
        print_success("Module export successful")
        result.add_pass()

        # Verify code contains expected elements
        code = export_result.code
        assert "workflowDefinition" in code, "Missing workflowDefinition"
        assert "executeWorkflow" in code, "Missing executeWorkflow function"
        assert definition.name in code, "Missing workflow name"
        assert definition.id in code, "Missing workflow ID"
        print_success("Generated code contains all expected elements")
        result.add_pass()

        # Test standalone export
        config.export_format = "standalone"
        exporter = BubbleLabsTypeScriptExporter(config)
        export_result = exporter.export_workflow(definition)

        assert export_result.success, f"Standalone export failed: {export_result.error}"
        assert "#!/usr/bin/env ts-node" in export_result.code, "Missing shebang"
        assert "async function main()" in export_result.code, "Missing main function"
        print_success("Standalone export successful")
        result.add_pass()

        # Test class export
        config.export_format = "class"
        exporter = BubbleLabsTypeScriptExporter(config)
        export_result = exporter.export_workflow(definition)

        assert export_result.success, f"Class export failed: {export_result.error}"
        assert "export class" in export_result.code, "Missing class declaration"
        assert "constructor()" in export_result.code, "Missing constructor"
        print_success("Class export successful")
        result.add_pass()

        # Test file export
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, f"{definition.id}.ts")
            export_result = exporter.export_workflow(definition, filepath)

            assert export_result.success, f"File export failed: {export_result.error}"
            assert os.path.exists(filepath), "Export file not created"
            with open(filepath, 'r') as f:
                file_content = f.read()
            assert len(file_content) > 0, "Export file is empty"
            print_success(f"File export successful: {filepath}")
            result.add_pass()

    except Exception as e:
        print_error(f"TypeScript export test failed: {str(e)}")
        result.add_fail(f"TypeScript export: {str(e)}")
        traceback.print_exc()


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

def test_full_integration(result: TestResult):
    """Test full integration of all components."""
    print_header("Testing Full Integration")

    try:
        # Create workflow
        from bubblelabs_integration import BubbleLabsIntegration
        integration = BubbleLabsIntegration()
        definition = integration.create_workflow_definition_from_openevolve(
            problem_statement="Full integration test workflow",
            team_config={
                "planner_team": "Integration-Test-Team",
                "solver_team": "Integration-Test-Team"
            },
            gauntlet_config={
                "sub_problem_red_gauntlet": "Test-Gauntlet"
            }
        )
        print_success("Created workflow for integration test")
        result.add_pass()

        # Create Hephaestus ticket
        from bubblelabs_hephaestus_bridge import create_bridge
        bridge = create_bridge(
            hephaestus_api_base="http://localhost:8000",
            hephaestus_api_key="test-key",
            hephaestus_project_id="test-project"
        )
        ticket_id = bridge.create_ticket_from_workflow(definition)
        assert ticket_id is not None, "Failed to create ticket"
        print_success(f"Hephaestus ticket created: {ticket_id}")
        result.add_pass()

        # Create analytics tracker
        from bubblelabs_analytics import create_analytics_tracker
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
            db_path = tmp.name

        try:
            analytics = create_analytics_tracker(db_path)
            analytics.start_workflow_tracking(
                workflow_id=definition.id,
                workflow_name=definition.name,
                instance_id="integration-test-instance"
            )
            print_success("Analytics tracking started")
            result.add_pass()

            # Track node execution
            analytics.track_node_execution(
                workflow_id=definition.id,
                node_id="test-node",
                node_type="test",
                tokens_used=500,
                execution_time=2.5,
                provider="openai",
                input_tokens=250,
                output_tokens=250
            )
            print_success("Node execution tracked")
            result.add_pass()

            # Export to TypeScript
            from bubblelabs_typescript_export import export_workflow_to_typescript
            with tempfile.NamedTemporaryFile(mode="w", suffix=".ts", delete=False) as tmp:
                ts_path = tmp.name

            try:
                export_result = export_workflow_to_typescript(definition.id, ts_path)
                assert export_result.success, f"Export failed: {export_result.error}"
                print_success(f"TypeScript export successful: {ts_path}")
                result.add_pass()

            finally:
                if os.path.exists(ts_path):
                    os.unlink(ts_path)

            # End tracking
            analytics.end_workflow_tracking(definition.id, "completed")
            print_success("Analytics tracking completed")
            result.add_pass()

        finally:
            if os.path.exists(db_path):
                os.unlink(db_path)

        print_success("Full integration test completed successfully")
        result.add_pass()

    except Exception as e:
        print_error(f"Full integration test failed: {str(e)}")
        result.add_fail(f"Full integration: {str(e)}")
        traceback.print_exc()


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

def main():
    """Run all tests."""
    print_header("BubbleLabs Complete Integration Test Suite")
    print_info("Starting comprehensive tests...\n")

    result = TestResult()

    # Run all test suites
    test_hephaestus_bridge(result)
    test_mcp_tools(result)
    test_analytics(result)
    test_typescript_export(result)
    test_full_integration(result)

    # Print summary
    print_header("Test Suite Complete")
    success = result.print_summary()

    if success:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[OK] ALL TESTS PASSED{Colors.END}")
        print(f"\n{Colors.GREEN}BubbleLabs complete integration is fully functional.{Colors.END}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}[FAIL] SOME TESTS FAILED{Colors.END}")
        print(f"\n{Colors.RED}Please review the errors above.{Colors.END}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
