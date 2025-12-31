"""
Example usage of Hephaestus integration with OpenEvolve workflows
"""
import os
import time
from typing import Optional

from workflow_structures import (WorkflowState, DecompositionPlan, SubProblem, 
                               SolutionAttempt, CritiqueReport, VerificationReport)
from hephaestus_integration import setup_hephaestus_integration, TicketStatus

def create_sample_workflow_state() -> WorkflowState:
    """Create a sample workflow state for demonstration"""
    sub_problems = [
        SubProblem(
            id="sub_1.1",
            description="Implement user authentication system",
            dependencies=[],
            ai_suggested_complexity_score=8,
            ai_suggested_evolution_mode="standard"
        ),
        SubProblem(
            id="sub_1.2", 
            description="Design database schema",
            dependencies=["sub_1.1"],
            ai_suggested_complexity_score=7,
            ai_suggested_evolution_mode="standard"
        ),
        SubProblem(
            id="sub_1.3",
            description="Create API endpoints",
            dependencies=["sub_1.2"],
            ai_suggested_complexity_score=6,
            ai_suggested_evolution_mode="standard"
        )
    ]
    
    plan = DecompositionPlan(
        problem_statement="Build a secure e-commerce platform",
        analyzed_context={"domain": "e-commerce", "requirements": ["security", "scalability"]},
        sub_problems=sub_problems
    )
    
    return WorkflowState(
        workflow_id="test-workflow-001",
        workflow_type="sovereign_grade_decomposition",
        problem_statement="Build a secure e-commerce platform",
        current_stage="Initialization",
        decomposition_plan=plan
    )

def run_hephaestus_integration_example():
    """Example of how to use the Hephaestus integration"""
    
    print("=== Hephaestus Integration Example ===")
    
    # Create a sample workflow
    workflow_state = create_sample_workflow_state()
    print(f"Created workflow: {workflow_state.workflow_id}")
    print(f"Problem: {workflow_state.problem_statement}")
    print(f"Sub-problems: {len(workflow_state.decomposition_plan.sub_problems)}")
    
    # Get credentials from environment variables (or use placeholders for demo)
    hephaestus_api_base = os.getenv("HEPHAESTUS_API_BASE", "https://hephaestus.example.com/api/v1")
    hephaestus_api_key = os.getenv("HEPHAESTUS_API_KEY", "demo_key")
    hephaestus_project_id = os.getenv("HEPHAESTUS_PROJECT_ID", "demo_project")
    
    # For demo purposes, we'll show what would happen without making real API calls
    print(f"\nAPI Base: {hephaestus_api_base}")
    print(f"Project ID: {hephaestus_project_id}")
    
    # Initialize Hephaestus integration
    print("\n--- Initializing Hephaestus Integration ---")
    integration_manager = setup_hephaestus_integration(
        workflow_state, 
        hephaestus_api_base, 
        hephaestus_api_key, 
        hephaestus_project_id
    )
    
    if integration_manager:
        print("✓ Hephaestus integration initialized successfully")
        
        # Simulate workflow progress
        print("\n--- Simulating Workflow Progress ---")
        
        # Update workflow stage
        workflow_state.current_stage = "Sub-Problem Solving Loop"
        
        # Process each sub-problem with Hephaestus sync
        for i, sub_problem in enumerate(workflow_state.decomposition_plan.sub_problems):
            print(f"\nProcessing sub-problem: {sub_problem.id}")
            
            # Update Hephaestus status to in-progress
            success = workflow_state.sync_subproblem_status_to_hephaestus(
                integration_manager, 
                sub_problem.id, 
                "in_progress"
            )
            print(f"  Updated status in Hephaestus: {'✓' if success else '✗'}")
            
            # Simulate solution generation
            solution = SolutionAttempt(
                sub_problem_id=sub_problem.id,
                content=f"Implementation of {sub_problem.description}",
                generated_by_model="gpt-4o",
                timestamp=time.time(),
                status="generated"
            )
            
            # Sync solution to Hephaestus
            success = workflow_state.sync_solution_to_hephaestus_ticket(
                integration_manager,
                sub_problem.id,
                solution
            )
            print(f"  Synced solution to Hephaestus: {'✓' if success else '✗'}")
            
            # Simulate critique
            critique = CritiqueReport(
                solution_attempt_id=sub_problem.id,
                gauntlet_name="Security Review Gauntlet",
                is_approved=(i % 2 == 0),  # Alternate approval for demo
                reports_by_judge=[{"judge": "security_bot", "score": 0.8, "feedback": "Good implementation with some minor issues"}],
                summary="Security review completed",
                overall_score=0.8
            )
            
            # Sync critique to Hephaestus
            success = workflow_state.sync_critique_to_hephaestus_ticket(
                integration_manager,
                sub_problem.id,
                critique
            )
            print(f"  Synced critique to Hephaestus: {'✓' if success else '✗'}")
            
            # Simulate verification
            verification = VerificationReport(
                solution_attempt_id=sub_problem.id,
                gauntlet_name="Quality Verification Gauntlet",
                is_approved=True,
                reports_by_judge=[{"judge": "quality_bot", "score": 0.9, "feedback": "Meets all requirements"}],
                average_score=0.9,
                summary="Quality verification completed",
                criteria_met=["functional", "performance", "security"],
                criteria_not_met=[]
            )
            
            # Sync verification to Hephaestus
            success = workflow_state.sync_verification_to_hephaestus_ticket(
                integration_manager,
                sub_problem.id,
                verification
            )
            print(f"  Synced verification to Hephaestus: {'✓' if success else '✗'}")
            
            # Mark as solved
            success = workflow_state.sync_subproblem_status_to_hephaestus(
                integration_manager,
                sub_problem.id,
                "solved",
                solution.content
            )
            print(f"  Marked as solved in Hephaestus: {'✓' if success else '✗'}")
            
            # Add to solved problems
            workflow_state.solved_sub_problem_ids.add(sub_problem.id)
        
        # Close the workflow sync
        workflow_state.status = "completed"
        workflow_state.end_time = time.time()
        
        print(f"\n--- Closing Workflow Sync ---")
        success = integration_manager.close_workflow_sync(workflow_state)
        print(f"  Closed workflow sync: {'✓' if success else '✗'}")
        
        # Get sync metrics
        metrics = integration_manager.get_workflow_sync_status(workflow_state)
        print(f"\n--- Sync Metrics ---")
        print(f"  Workflow ID: {metrics['workflow_id']}")
        print(f"  Total Sub-problems: {metrics['total_subproblems']}")
        print(f"  Synced Sub-problems: {metrics['synced_subproblems']}")
        print(f"  Sync Percentage: {metrics['sync_percentage']:.1f}%")
        print(f"  Hephaestus Workflow ID: {metrics['hephaestus_workflow_id']}")
    
    else:
        print("! Failed to initialize Hephaestus integration")
        print("  Make sure you have the required environment variables set:")
        print("  - HEPHAESTUS_API_BASE: Base URL for Hephaestus API")
        print("  - HEPHAESTUS_API_KEY: API key for authentication") 
        print("  - HEPHAESTUS_PROJECT_ID: Project ID in Hephaestus")

def demonstrate_mapping_functionality():
    """Demonstrate the ID mapping functionality"""
    print("\n=== ID Mapping Demonstration ===")
    
    # Create workflow with sample mappings
    workflow_state = create_sample_workflow_state()
    
    # Simulate some mappings (normally created during workflow initialization)
    workflow_state.id_to_ticket_id_map = {
        "sub_1.1": "HEP-101",
        "sub_1.2": "HEP-102", 
        "sub_1.3": "HEP-103"
    }
    
    workflow_state.ticket_id_to_subproblem_id_map = {
        "HEP-101": "sub_1.1",
        "HEP-102": "sub_1.2",
        "HEP-103": "sub_1.3"
    }
    
    print("Forward mapping (sub-problem ID to ticket ID):")
    for sub_id, ticket_id in workflow_state.id_to_ticket_id_map.items():
        print(f"  {sub_id} → {ticket_id}")
    
    print("\nReverse mapping (ticket ID to sub-problem ID):")
    for ticket_id, sub_id in workflow_state.ticket_id_to_subproblem_id_map.items():
        print(f"  {ticket_id} → {sub_id}")
    
    # Test lookup
    sample_sub_id = "sub_1.2"
    ticket_id = workflow_state.id_to_ticket_id_map.get(sample_sub_id)
    print(f"\nLooking up ticket for {sample_sub_id}: {ticket_id}")
    
    sample_ticket_id = "HEP-102"
    sub_id = workflow_state.ticket_id_to_subproblem_id_map.get(sample_ticket_id)
    print(f"Looking up sub-problem for {sample_ticket_id}: {sub_id}")

if __name__ == "__main__":
    run_hephaestus_integration_example()
    demonstrate_mapping_functionality()
    
    print("\n=== Integration Complete ===")
    print("The Hephaestus integration provides:")
    print("- Automatic creation of workflow epic and sub-problem tickets")
    print("- Status synchronization between OpenEvolve and Hephaestus")
    print("- Solution, critique, and verification syncing to tickets")
    print("- Bidirectional ID mapping between systems")
    print("- Workflow lifecycle management with Hephaestus")