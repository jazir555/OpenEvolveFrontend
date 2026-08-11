"""
OpenEvolve + BubbleLabs Integration Demo

This script demonstrates the complete integration between OpenEvolve workflows
and the BubbleLabs visual workflow designer.

Usage:
    python demo_openevolve_bubblelabs.py
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openevolve_workflow_manager import (
    OpenEvolveWorkflowManager,
    WorkflowTemplate,
    WorkflowConfig
)

print("=" * 70)
print("OpenEvolve + BubbleLabs Integration Demo")
print("=" * 70)

# Initialize the workflow manager
print("\n1. Initializing OpenEvolve Workflow Manager...")
manager = OpenEvolveWorkflowManager(
    analytics_db_path='demo_openevolve_analytics.db',
    enable_CREWAI=False  # Disable for demo
)
print("   [OK] Manager initialized")

# Create a workflow from template
print("\n2. Creating workflow from template...")
workflow_id = manager.create_workflow_from_template(
    template=WorkflowTemplate.SOVEREIGN_DECOMPOSITION,
    name="Demo Optimization Workflow",
    description="Demonstrates OpenEvolve + BubbleLabs integration",
    parameters={
        'max_refinement_loops': 3,
        'team_size': 2
    }
)
print(f"   [OK] Created workflow: {workflow_id}")

# List all workflows
print("\n3. Listing all workflows...")
workflows = manager.list_workflows()
print(f"   Total workflows: {len(workflows)}")
for wf in workflows:
    print(f"   - {wf['name']} ({wf['type']})")

# Get workflow status
print("\n4. Checking workflow status...")
status = manager.get_workflow_status(workflow_id)
if status:
    print(f"   Status: {status['status']}")
    print(f"   Progress: {status['progress']*100:.1f}%")
else:
    print("   (No active execution)")

# Execute workflow
print("\n5. Executing workflow...")
print("   Problem statement: 'Optimize database query performance'")
result = manager.execute_workflow(
    workflow_id=workflow_id,
    problem_statement="How can we optimize database query performance for large datasets?"
)

if result.success:
    print(f"   [OK] Execution successful!")
    print(f"   - Status: {result.status}")
    print(f"   - Execution time: {result.execution_time:.2f}s")
    print(f"   - Tokens used: {result.tokens_used}")
    print(f"   - Iterations: {result.iterations_completed}")
    if result.result:
        print(f"   - Result: {result.result}")
else:
    print(f"   [FAIL] Execution failed: {result.error}")

# Get analytics
print("\n6. Getting workflow analytics...")
metrics = manager.get_workflow_metrics(workflow_id)
if metrics:
    print(f"   [OK] Analytics available:")
    print(f"   - Total workflows: {metrics.get('total_workflows', 0)}")
    print(f"   - Total tokens: {metrics.get('total_tokens', 0)}")
    print(f"   - Total cost: ${metrics.get('total_cost', 0.0):.4f}")
else:
    print("   (No analytics available)")

# Demonstrate workflow templates
print("\n7. Available workflow templates:")
from openevolve_workflow_mcp_tools import get_workflow_templates
templates_result = get_workflow_templates()
if templates_result['success']:
    for template_name, template_info in templates_result['templates'].items():
        print(f"\n   {template_name}:")
        print(f"     {template_info['description']}")

# Demonstrate control operations
print("\n8. Demonstrating workflow control...")

# Note: These operations would work on actual running workflows
print("   - pause_workflow(): Pause a running workflow")
print("   - resume_workflow(): Resume a paused workflow")
print("   - cancel_workflow(): Cancel a workflow")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nNext Steps:")
print("1. Enable CREWAI integration for project management")
print("2. Use MCP tools for external agent control")
print("3. Integrate with BubbleLabs UI for visual workflow design")
print("4. Explore other workflow templates")
print("\nFor more information, see: OPENEVOLVE_BUBBLELABS_INTEGRATION.md")
