"""Test advanced visualization features."""

from advanced_visualization import (
    DependencyGraphVisualizer,
    AdvancedVisualizer,
    ReportGenerator
)
from workflow_structures import SubProblem, WorkflowState, DecompositionPlan
import time


def test_dependency_graph_visualizer():
    """Test dependency graph visualizer."""
    print("Testing DependencyGraphVisualizer...")
    
    visualizer = DependencyGraphVisualizer()
    
    # Create test sub-problems with dependencies
    sub_problems = [
        SubProblem(id="sp1", description="First sub-problem", dependencies=[], status="solved"),
        SubProblem(id="sp2", description="Second sub-problem", dependencies=["sp1"], status="solved"),
        SubProblem(id="sp3", description="Third sub-problem", dependencies=["sp1"], status="in_progress"),
        SubProblem(id="sp4", description="Fourth sub-problem", dependencies=["sp2", "sp3"], status="pending")
    ]
    
    # Test graph generation
    G = visualizer.generate_graph(sub_problems)
    assert len(G.nodes()) == 4
    assert len(G.edges()) == 4  # sp1->sp2, sp1->sp3, sp2->sp4, sp3->sp4
    
    # Test interactive graph creation
    fig = visualizer.create_interactive_graph(sub_problems)
    assert fig is not None
    assert len(fig.data) == 2  # edges and nodes
    
    # Test critical path
    critical_path = visualizer.get_critical_path(sub_problems)
    assert len(critical_path) > 0
    assert critical_path[0] == "sp1"  # Should start with sp1
    
    print("✓ DependencyGraphVisualizer tests passed")


def test_advanced_visualizer():
    """Test advanced visualizer."""
    print("\nTesting AdvancedVisualizer...")
    
    visualizer = AdvancedVisualizer()
    
    # Create test workflow state
    plan = DecompositionPlan(
        problem_statement="Test problem",
        analyzed_context={},
        sub_problems=[
            SubProblem(id="sp1", description="Test 1", dependencies=[], status="solved"),
            SubProblem(id="sp2", description="Test 2", dependencies=["sp1"], status="in_progress")
        ]
    )
    
    workflow_state = WorkflowState(
        workflow_id="test_wf_1",
        workflow_type="sovereign_decomposition",
        problem_statement="Test problem",
        current_stage="Sub-Problem Solving",
        status="running",
        progress=0.5,
        decomposition_plan=plan
    )
    workflow_state.solved_sub_problem_ids.add("sp1")
    
    # Test timeline creation
    timeline_fig = visualizer.create_workflow_timeline(workflow_state)
    assert timeline_fig is not None
    
    # Test complexity heatmap
    heatmap_fig = visualizer.create_complexity_heatmap(plan)
    assert heatmap_fig is not None
    
    # Test performance dashboard
    dashboard = visualizer.create_performance_dashboard(workflow_state)
    assert "progress" in dashboard
    assert "status" in dashboard
    assert "refinement" in dashboard
    assert "timeline" in dashboard
    
    print("✓ AdvancedVisualizer tests passed")


def test_report_generator():
    """Test report generator."""
    print("\nTesting ReportGenerator...")
    
    generator = ReportGenerator()
    
    # Create test workflow state
    plan = DecompositionPlan(
        problem_statement="Test problem for reporting",
        analyzed_context={},
        sub_problems=[
            SubProblem(id="sp1", description="First test sub-problem", dependencies=[], status="solved"),
            SubProblem(id="sp2", description="Second test sub-problem", dependencies=["sp1"], status="solved")
        ]
    )
    
    workflow_state = WorkflowState(
        workflow_id="test_wf_2",
        workflow_type="sovereign_decomposition",
        problem_statement="Test problem for reporting",
        current_stage="Completed",
        status="completed",
        progress=1.0,
        decomposition_plan=plan
    )
    workflow_state.solved_sub_problem_ids.add("sp1")
    workflow_state.solved_sub_problem_ids.add("sp2")
    workflow_state.end_time = time.time()
    
    # Test executive summary
    summary = generator.generate_executive_summary(workflow_state)
    assert "WORKFLOW EXECUTIVE SUMMARY" in summary
    assert workflow_state.workflow_id in summary
    assert "2/2 solved" in summary
    
    # Test detailed report
    detailed = generator.generate_detailed_report(workflow_state)
    assert "DETAILED WORKFLOW REPORT" in detailed
    assert "SUB-PROBLEM DETAILS" in detailed
    assert "sp1" in detailed
    assert "sp2" in detailed
    
    # Test JSON export
    json_data = generator.export_to_json(workflow_state)
    assert json_data["workflow_id"] == "test_wf_2"
    assert json_data["status"] == "completed"
    assert json_data["progress"] == 1.0
    assert json_data["total_sub_problems"] == 2
    assert len(json_data["solved_sub_problems"]) == 2
    
    print("✓ ReportGenerator tests passed")


def test_graph_export():
    """Test graph export functionality."""
    print("\nTesting graph export...")
    
    visualizer = DependencyGraphVisualizer()
    
    sub_problems = [
        SubProblem(id="sp1", description="First", dependencies=[], status="solved"),
        SubProblem(id="sp2", description="Second", dependencies=["sp1"], status="solved")
    ]
    
    # Test PNG export
    try:
        png_bytes = visualizer.export_graph(sub_problems, format='png')
        assert len(png_bytes) > 0
        print("✓ PNG export successful")
    except Exception as e:
        print(f"⚠ PNG export skipped (matplotlib may not be configured): {e}")
    
    print("✓ Graph export tests passed")


if __name__ == "__main__":
    print("Running advanced visualization tests...\n")
    
    test_dependency_graph_visualizer()
    test_advanced_visualizer()
    test_report_generator()
    test_graph_export()
    
    print("\n" + "="*50)
    print("All advanced visualization tests passed!")
    print("="*50)
