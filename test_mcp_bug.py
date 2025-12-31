#!/usr/bin/env python3
"""
Test to verify the MCP tools bug with integration instances
"""

print("Testing MCP Tools Integration Instance Bug")
print("=" * 70)

# Test the bug
from bubblelabs_mcp_tools import create_bubblelabs_workflow, list_bubblelabs_workflows

# Create a workflow
result1 = create_bubblelabs_workflow(
    problem_statement="Test workflow",
    team_config={"planner_team": "Test-Team"}
)

print(f"\n1. Created workflow:")
print(f"   Success: {result1['success']}")
print(f"   Workflow ID: {result1['workflow_id']}")

# Try to list workflows
result2 = list_bubblelabs_workflows()

print(f"\n2. Listed workflows:")
print(f"   Success: {result2['success']}")
print(f"   Definitions count: {len(result2['definitions'])}")
print(f"   Message: {result2['message']}")

# This shows the bug - the newly created workflow won't appear in the list
# because list_bubblelabs_workflows creates a NEW integration instance

if len(result2['definitions']) == 0:
    print("\n   [BUG CONFIRMED] Workflows not being shared between MCP tool calls!")
    print("   Each MCP tool creates a new OpenEvolveBubbleLabsIntegration instance.")
    print("   This means workflows created in one call won't appear in list_bubblelabs_workflows.")
else:
    print("\n   [OK] Workflows are properly shared")

print("\n" + "=" * 70)
