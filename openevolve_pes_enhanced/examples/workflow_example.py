"""Example usage of Workflow PES Adapter.

This example demonstrates how to use the PES Enhanced adapter with the
Workflow Engine for cost-aware evolution and workflow execution.
"""

import asyncio
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_workflow_state():
    """Create a sample workflow state for demonstration."""
    try:
        from workflow_structures import WorkflowState, DecompositionPlan, SubProblem, Team, ModelConfig
        
        # Create sample subproblems
        subproblems = [
            SubProblem(
                id="sp_1",
                description="Design the database schema",
                dependencies=[],
                ai_suggested_complexity_score=6,
                ai_suggested_evolution_mode="standard"
            ),
            SubProblem(
                id="sp_2",
                description="Implement API endpoints",
                dependencies=["sp_1"],
                ai_suggested_complexity_score=8,
                ai_suggested_evolution_mode="qd"  # Quality Diversity for exploration
            ),
            SubProblem(
                id="sp_3",
                description="Create frontend components",
                dependencies=[],
                ai_suggested_complexity_score=5,
                ai_suggested_evolution_mode="standard"
            ),
            SubProblem(
                id="sp_4",
                description="Optimize performance",
                dependencies=["sp_2", "sp_3"],
                ai_suggested_complexity_score=9,
                ai_suggested_evolution_mode="mo"  # Multi-objective for tradeoffs
            ),
        ]
        
        # Create decomposition plan
        decomposition_plan = DecompositionPlan(
            problem_statement="Build a full-stack web application",
            analyzed_context={
                "domain": "Software Development",
                "keywords": ["web", "api", "database", "frontend"],
                "estimated_complexity": 7
            },
            sub_problems=subproblems,
            max_refinement_loops=3
        )
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id="demo_workflow_001",
            workflow_type="sovereign_decomposition",
            problem_statement="Build a full-stack web application",
            current_stage="AI-Assisted Decomposition",
            decomposition_plan=decomposition_plan
        )
        
        return workflow_state
        
    except ImportError as e:
        logger.warning(f"Could not import workflow_structures: {e}")
        # Return a mock object for demonstration
        class MockWorkflowState:
            def __init__(self):
                self.workflow_id = "demo_workflow_001"
                self.problem_statement = "Build a full-stack web application"
                self.current_stage = "AI-Assisted Decomposition"
                self.status = "running"
                self.metadata = {}
                
                class MockPlan:
                    def __init__(self):
                        self.sub_problems = [
                            MockSubProblem("sp_1", "Design database", 6),
                            MockSubProblem("sp_2", "Implement API", 8),
                            MockSubProblem("sp_3", "Frontend", 5),
                            MockSubProblem("sp_4", "Optimization", 9),
                        ]
                
                class MockSubProblem:
                    def __init__(self, id, desc, complexity):
                        self.id = id
                        self.description = desc
                        self.ai_suggested_complexity_score = complexity
                        self.dependencies = []
                        
                self.decomposition_plan = MockPlan()
        
        return MockWorkflowState()


def example_1_basic_adapter_usage():
    """Example 1: Basic adapter usage with cost tracking."""
    print("\n" + "="*60)
    print("Example 1: Basic Adapter Usage")
    print("="*60 + "\n")
    
    from openevolve_pes_enhanced.workflow_adapter import (
        WorkflowPESAdapter, WorkflowStatePESExtension
    )
    from openevolve_pes_enhanced.config import PESEnhancedConfig
    
    # Create workflow state
    workflow_state = create_sample_workflow_state()
    
    # Create PES config with cost awareness
    pes_config = PESEnhancedConfig.cost_aware(max_cost_usd=10.0)
    
    # Extend workflow state with PES tracking
    WorkflowStatePESExtension.extend(workflow_state, pes_config)
    
    # Create adapter
    adapter = WorkflowPESAdapter(pes_config)
    
    # Initialize budget
    adapter.initialize_budget(max_cost_usd=10.0)
    
    print(f"Workflow ID: {workflow_state.workflow_id}")
    print(f"Problem: {workflow_state.problem_statement}")
    print(f"Budget: $10.00")
    print(f"PES Config: cost_aware mode enabled")
    
    # Enhance decomposition with PES allocation
    if hasattr(workflow_state, 'decomposition_plan') and workflow_state.decomposition_plan:
        subproblems = workflow_state.decomposition_plan.sub_problems
        budget_per_problem = 10.0 / len(subproblems)
        
        allocations = adapter.enhance_decomposition_with_pes(
            subproblems, budget_per_problem
        )
        
        print(f"\nResource Allocations for {len(allocations)} subproblems:")
        print("-" * 60)
        
        for sp, allocation in allocations:
            print(f"\n  SubProblem: {sp.id}")
            print(f"    Description: {sp.description if hasattr(sp, 'description') else 'N/A'}")
            print(f"    Complexity: {sp.ai_suggested_complexity_score if hasattr(sp, 'ai_suggested_complexity_score') else 'N/A'}")
            print(f"    Decision: {allocation.decision.value}")
            print(f"    Budget: ${allocation.budget_usd:.2f}")
            print(f"    Max Iterations: {allocation.max_iterations}")
            print(f"    Priority: {allocation.priority}")
            print(f"    Reason: {allocation.reason}")
    
    print("\n[OK] Example 1 completed successfully")


async def example_2_full_workflow_execution():
    """Example 2: Full workflow execution with PES (requires full environment)."""
    print("\n" + "="*60)
    print("Example 2: Full Workflow with PES Tracking")
    print("="*60 + "\n")
    
    try:
        from openevolve_pes_enhanced.workflow_adapter import (
            run_sovereign_workflow_with_pes,
            create_cost_aware_workflow_config
        )
        from workflow_structures import WorkflowState, Team, GauntletDefinition, GauntletRoundRule
        
        # Create workflow state
        workflow_state = create_sample_workflow_state()
        
        # Create PES config
        pes_config = create_cost_aware_workflow_config(
            max_cost_usd=15.0,
            enable_early_stopping=True,
            enable_cost_optimization=True
        )
        
        # Create sample teams (in real usage, these come from team_manager)
        content_analyzer_team = Team(
            name="ContentAnalyzer",
            role="Blue",
            members=[]
        )
        planner_team = Team(
            name="Planner",
            role="Blue",
            members=[]
        )
        solver_team = Team(
            name="Solver",
            role="Blue",
            members=[]
        )
        patcher_team = Team(
            name="Patcher",
            role="Blue",
            members=[]
        )
        assembler_team = Team(
            name="Assembler",
            role="Blue",
            members=[]
        )
        
        # Create sample gauntlets
        round_rule = GauntletRoundRule(
            round_number=1,
            quorum_required_approvals=1,
            quorum_from_panel_size=1
        )
        
        sub_problem_red_gauntlet = GauntletDefinition(
            name="SubProblemRed",
            team_name="RedTeam",
            rounds=[round_rule]
        )
        sub_problem_gold_gauntlet = GauntletDefinition(
            name="SubProblemGold",
            team_name="GoldTeam",
            rounds=[round_rule]
        )
        final_red_gauntlet = GauntletDefinition(
            name="FinalRed",
            team_name="RedTeam",
            rounds=[round_rule]
        )
        final_gold_gauntlet = GauntletDefinition(
            name="FinalGold",
            team_name="GoldTeam",
            rounds=[round_rule]
        )
        solver_generation_gauntlet = GauntletDefinition(
            name="SolverGen",
            team_name="SolverTeam",
            rounds=[round_rule]
        )
        
        print("Starting workflow with PES Enhanced...")
        print(f"Budget: $15.00")
        print(f"Early stopping: enabled")
        print(f"Cost optimization: enabled")
        
        # Note: This would normally run the full workflow
        # For demonstration, we show the function signature
        print("\nFunction call:")
        print("  result = await run_sovereign_workflow_with_pes(")
        print("      workflow_state=workflow_state,")
        print("      content_analyzer_team=content_analyzer_team,")
        print("      planner_team=planner_team,")
        print("      solver_team=solver_team,")
        print("      patcher_team=patcher_team,")
        print("      assembler_team=assembler_team,")
        print("      sub_problem_red_gauntlet=sub_problem_red_gauntlet,")
        print("      sub_problem_gold_gauntlet=sub_problem_gold_gauntlet,")
        print("      final_red_gauntlet=final_red_gauntlet,")
        print("      final_gold_gauntlet=final_gold_gauntlet,")
        print("      solver_generation_gauntlet=solver_generation_gauntlet,")
        print("      pes_config=pes_config,")
        print("      max_cost_usd=15.0,")
        print("      enable_cost_tracking=True")
        print("  )")
        
        print("\n[OK] Example 2 setup completed")
        print("  (Full execution requires configured teams and gauntlets)")
        
    except ImportError as e:
        print(f"Skipping full workflow example - missing dependencies: {e}")
        print("  (This is expected if running without full OpenEvolve environment)")


def example_3_budget_enforcement():
    """Example 3: Demonstrate budget enforcement."""
    print("\n" + "="*60)
    print("Example 3: Budget Enforcement")
    print("="*60 + "\n")
    
    from openevolve_pes_enhanced.workflow_adapter import (
        WorkflowPESAdapter, BudgetExceededError
    )
    from openevolve_pes_enhanced.config import PESEnhancedConfig
    
    # Create adapter with very tight budget
    pes_config = PESEnhancedConfig.cost_aware(max_cost_usd=1.0)
    adapter = WorkflowPESAdapter(pes_config)
    adapter.initialize_budget(max_cost_usd=1.0)
    
    workflow_state = create_sample_workflow_state()
    
    print("Budget: $1.00 (very tight)")
    print(f"Subproblems: {len(workflow_state.decomposition_plan.sub_problems)}")
    
    # Try to allocate resources
    budget_per_problem = 1.0 / len(workflow_state.decomposition_plan.sub_problems)
    allocations = adapter.enhance_decomposition_with_pes(
        workflow_state.decomposition_plan.sub_problems,
        budget_per_problem
    )
    
    print("\nAllocations with tight budget:")
    print("-" * 60)
    
    for sp, allocation in allocations:
        status = "[OK]" if allocation.decision.value != "defer" else "⚠ DEFERRED"
        print(f"  {status} {sp.id}: ${allocation.budget_usd:.2f} - {allocation.reason}")
    
    # Check budget status
    status = adapter.cost_tracker.get_status()
    print(f"\nBudget Status:")
    print(f"  Total cost: ${status['total_cost_usd']:.2f}")
    print(f"  Budget remaining: ${status['budget_remaining']:.2f}")
    print(f"  Budget % used: {status['budget_pct_used']:.1f}%")
    
    print("\n[OK] Example 3 completed - budget enforcement working")


def example_4_stage_tracking():
    """Example 4: Track costs across workflow stages."""
    print("\n" + "="*60)
    print("Example 4: Stage-by-Stage Cost Tracking")
    print("="*60 + "\n")
    
    from openevolve_pes_enhanced.workflow_adapter import (
        WorkflowPESAdapter, WorkflowCostMetrics
    )
    from openevolve_pes_enhanced.config import PESEnhancedConfig
    
    pes_config = PESEnhancedConfig.cost_aware(max_cost_usd=50.0)
    adapter = WorkflowPESAdapter(pes_config)
    adapter.initialize_budget(max_cost_usd=50.0)
    
    # Simulate workflow stages
    stages = [
        ("Content Analysis", 0.50),
        ("Decomposition", 1.20),
        ("Solution Generation (SP1)", 3.50),
        ("Solution Generation (SP2)", 5.80),
        ("Solution Generation (SP3)", 2.30),
        ("Verification", 1.80),
        ("Assembly", 0.90),
    ]
    
    print("Simulating workflow execution with cost tracking:")
    print("-" * 60)
    
    for stage_name, cost in stages:
        adapter.cost_tracker.start_stage(stage_name)
        
        # Simulate work
        import time
        time.sleep(0.1)
        
        # Record cost
        adapter.cost_tracker.record_cost(cost, stage=stage_name)
        adapter.cost_tracker.end_stage(stage_name)
        
        # Check budget
        should_continue, stop_reason = adapter.check_and_enforce_budget()
        status = "[OK]" if should_continue else "✗ STOPPED"
        
        print(f"  {status} {stage_name:30s} ${cost:6.2f}")
        
        if not should_continue:
            print(f"\n  BUDGET EXCEEDED: {stop_reason}")
            break
    
    # Final summary
    metrics = adapter.cost_tracker.metrics
    status = adapter.cost_tracker.get_status()
    
    print("\n" + "-" * 60)
    print("FINAL COST SUMMARY:")
    print(f"  Total Cost: ${metrics.total_cost_usd:.2f}")
    print(f"  Total Time: {metrics.total_time_seconds:.1f}s")
    print(f"  Budget Remaining: ${status['budget_remaining']:.2f}")
    print(f"  Budget Used: {status['budget_pct_used']:.1f}%")
    
    print("\n  Stage Breakdown:")
    for stage, cost in status['stage_breakdown'].items():
        pct = (cost / metrics.total_cost_usd * 100) if metrics.total_cost_usd > 0 else 0
        print(f"    {stage:30s} ${cost:6.2f} ({pct:5.1f}%)")
    
    print("\n[OK] Example 4 completed - stage tracking working")


def example_5_integration_with_existing_workflow():
    """Example 5: Show how to integrate with existing workflow code."""
    print("\n" + "="*60)
    print("Example 5: Integration with Existing Workflow")
    print("="*60 + "\n")
    
    print("""
# Existing workflow code (unchanged):
# -----------------------------------
from workflow_engine import run_sovereign_workflow

workflow_state = WorkflowState(...)
result = await run_sovereign_workflow(
    workflow_state,
    content_analyzer_team=...,
    planner_team=...,
    ...
)

# With PES Enhanced (opt-in):
# ---------------------------
from openevolve_pes_enhanced.workflow_adapter import (
    run_sovereign_workflow_with_pes,
    create_cost_aware_workflow_config
)

# Create PES config
pes_config = create_cost_aware_workflow_config(max_cost_usd=20.0)

# Use drop-in replacement
result = await run_sovereign_workflow_with_pes(
    workflow_state,
    content_analyzer_team=...,
    planner_team=...,
    ...,
    pes_config=pes_config,
    max_cost_usd=20.0  # Enable cost tracking
)

# Access cost metrics
if 'pes_cost_metrics' in result.metadata:
    metrics = result.metadata['pes_cost_metrics']
    print(f"Total cost: ${metrics['total_cost_usd']:.2f}")
    print(f"Efficiency gain: {metrics['efficiency_gain']*100:.1f}%")
""")

    print("\n[OK] Example 5 completed - integration pattern shown")


async def run_all_examples():
    """Run all examples."""
    print("\n" + "="*60)
    print("OpenEvolve PES Enhanced - Workflow Adapter Examples")
    print("="*60)
    
    example_1_basic_adapter_usage()
    await example_2_full_workflow_execution()
    example_3_budget_enforcement()
    example_4_stage_tracking()
    example_5_integration_with_existing_workflow()
    
    print("\n" + "="*60)
    print("All Examples Completed!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(run_all_examples())
