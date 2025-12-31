"""
End-to-End Test: OpenEvolve-Hephaestus Integration

This script demonstrates the complete integration of OpenEvolve and Hephaestus,
showing how agents use MCP tools to execute the workflow.

This is a simulation that shows what would happen in a real Hephaestus workflow
without actually starting the Hephaestus services.

Usage:
    python test_hephaestus_end_to_end.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import the workflow bridge
from hephaestus_workflow_bridge import (
    execute_phase_1_decomposition,
    execute_phase_2_solving,
    execute_phase_3_critique,
    execute_phase_4_verification,
    execute_phase_5_reassembly,
    execute_phase_6_final_verification,
    get_workflow_bridge,
)


class WorkflowSimulator:
    """
    Simulates a Hephaestus workflow execution.

    This shows what would happen when Hephaestus orchestrates the workflow,
    with agents being spawned and calling MCP tools.
    """

    def __init__(self, problem_statement: str):
        self.problem_statement = problem_statement
        self.workflow_state = {
            "problem_statement": problem_statement,
            "phase": 0,
            "sub_problems": [],
            "solutions": {},
            "critiques": {},
            "verifications": {},
            "verified_solutions": [],
        }
        self.bridge = get_workflow_bridge()

    def print_phase_header(self, phase: int, title: str):
        """Print a formatted phase header"""
        print("\n" + "="*80)
        print(f"  PHASE {phase}: {title}")
        print("="*80 + "\n")

    def print_result(self, result: Dict[str, Any]):
        """Print phase execution result"""
        status = result.get("status", "unknown")
        status_emoji = "✓" if status == "completed" else "✗"
        print(f"\n{status_emoji} Status: {status}")
        print(f"  Message: {result.get('message', 'No message')}")

        if result.get("status") == "completed":
            next_phase = result.get("next_phase")
            if next_phase:
                print(f"  Next Phase: {next_phase}")

    def simulate_phase_1(self):
        """Simulate Phase 1: Decomposition"""
        self.print_phase_header(1, "PROBLEM DECOMPOSITION")

        print("Agent: Phase 1 Agent (Decomposition Specialist)")
        print(f"Task: Decompose problem into solvable sub-problems\n")

        print("Agent: Using analyze_problem_context MCP tool...")
        print(f"  Problem: {self.problem_statement[:100]}...")

        result = execute_phase_1_decomposition(
            problem_statement=self.problem_statement,
            domain="Software Development",
            complexity_estimate=7,
            max_sub_problems=5,
        )

        self.print_result(result)

        if result["status"] == "completed":
            self.workflow_state["sub_problems"] = result["decomposition"]["sub_problems"]
            self.workflow_state["phase"] = 1

            print(f"\n  Created {len(result['phase_2_tasks'])} Phase 2 tasks:")
            for i, task in enumerate(result["phase_2_tasks"][:3], 1):
                print(f"    {i}. {task['metadata']['sub_problem_id']} - {task['metadata']['solver_team']}")
            if len(result["phase_2_tasks"]) > 3:
                print(f"    ... and {len(result['phase_2_tasks']) - 3} more")

            return result["phase_2_tasks"]

        return None

    def simulate_phase_2(self, task: Dict[str, Any]):
        """Simulate Phase 2: Solving a sub-problem"""
        self.print_phase_header(2, f"SUB-PROBLEM SOLVING - {task['metadata']['sub_problem_id']}")

        print(f"Agent: Phase 2 Agent ({task['metadata']['solver_team']})")
        print(f"Task: Solve sub-problem {task['metadata']['sub_problem_id']}\n")

        print("Agent: Using solve_sub_problem MCP tool...")

        result = execute_phase_2_solving(
            sub_problem_id=task['metadata']['sub_problem_id'],
            sub_problem_description=task['description'],
            constraints=[],
            requirements=[],
            context={},
            solver_team=task['metadata']['solver_team'],
        )

        self.print_result(result)

        if result["status"] == "completed":
            sub_problem_id = task['metadata']['sub_problem_id']
            self.workflow_state["solutions"][sub_problem_id] = result["solution"]
            return result["solution"], task['metadata']

        return None, None

    def simulate_phase_3(self, solution: Dict[str, Any], metadata: Dict[str, Any]):
        """Simulate Phase 3: Critique"""
        self.print_phase_header(3, f"SOLUTION CRITIQUE - {metadata['sub_problem_id']}")

        print(f"Agent: Phase 3 Agent (Red Team - {metadata['red_gauntlet']})")
        print(f"Task: Critique solution for {metadata['sub_problem_id']}\n")

        print("Agent: Using critique_solution MCP tool...")

        result = execute_phase_3_critique(
            solution=solution,
            sub_problem_id=metadata['sub_problem_id'],
            red_team_gauntlet=metadata['red_gauntlet'],
        )

        self.print_result(result)

        if result["status"] == "completed":
            self.workflow_state["critiques"][metadata['sub_problem_id']] = result["critique"]

            if result["approved"]:
                print(f"\n  ✓ Solution APPROVED - proceeding to Phase 4")
                return result["critique"], True
            else:
                print(f"\n  ✗ Solution NEEDS REWORK - would go back to Phase 2")
                return result["critique"], False

        return None, False

    def simulate_phase_4(self, solution: Dict[str, Any], critique: Dict[str, Any], metadata: Dict[str, Any]):
        """Simulate Phase 4: Verification"""
        self.print_phase_header(4, f"SOLUTION VERIFICATION - {metadata['sub_problem_id']}")

        print(f"Agent: Phase 4 Agent (Gold Team - {metadata['gold_gauntlet']})")
        print(f"Task: Verify solution for {metadata['sub_problem_id']}\n")

        print("Agent: Using verify_solution MCP tool...")

        result = execute_phase_4_verification(
            solution=solution,
            critique=critique,
            sub_problem_id=metadata['sub_problem_id'],
            gold_team_gauntlet=metadata['gold_gauntlet'],
            requirements=[],
        )

        self.print_result(result)

        if result["status"] == "completed":
            self.workflow_state["verifications"][metadata['sub_problem_id']] = result["verification"]

            if result["approved"]:
                print(f"\n  ✓ Solution VERIFIED - adding to verified solutions")
                return True
            else:
                print(f"\n  ✗ Solution NOT VERIFIED")
                return False

        return False

    def simulate_phase_5(self):
        """Simulate Phase 5: Reassembly"""
        self.print_phase_header(5, "SOLUTION REASSEMBLY")

        print("Agent: Phase 5 Agent (Integration Specialist)")
        print(f"Task: Integrate {len(self.workflow_state['verified_solutions'])} verified solutions\n")

        print("Agent: Using reassemble_solution MCP tool...")

        # Build dependencies
        dependencies = {}
        for sp in self.workflow_state["sub_problems"]:
            dependencies[sp["id"]] = sp.get("dependencies", [])

        result = execute_phase_5_reassembly(
            verified_solutions=self.workflow_state["verified_solutions"],
            dependencies=dependencies,
            original_problem=self.problem_statement,
        )

        self.print_result(result)

        if result["status"] == "completed":
            return result["integrated_solution"]

        return None

    def simulate_phase_6(self, integrated_solution: Dict[str, Any]):
        """Simulate Phase 6: Final Verification"""
        self.print_phase_header(6, "FINAL VERIFICATION")

        print("Agent: Phase 6 Agent (Quality Assurance)")
        print("Task: Perform final comprehensive verification\n")

        print("Agent: Using final_verification MCP tool...")

        result = execute_phase_6_final_verification(
            integrated_solution=integrated_solution,
            original_problem=self.problem_statement,
            requirements=[],
        )

        self.print_result(result)

        if result["status"] == "completed" and result["approved"]:
            print("\n" + "="*80)
            print("  ✓✓✓ WORKFLOW COMPLETE - SOLUTION APPROVED ✓✓✓")
            print("="*80 + "\n")

            # Print summary
            print("FINAL SUMMARY:")
            print(f"  Problem: {self.problem_statement[:80]}...")
            print(f"  Sub-problems: {len(self.workflow_state['sub_problems'])}")
            print(f"  Solutions Generated: {len(self.workflow_state['solutions'])}")
            print(f"  Critiques Performed: {len(self.workflow_state['critiques'])}")
            print(f"  Verifications: {len(self.workflow_state['verifications'])}")
            print(f"  Final Score: {result['final_verification'].get('overall_score', 0):.2f}")

            return True

        return False

    def run_workflow(self):
        """Run the complete simulated workflow"""
        print("\n" + "="*80)
        print("  OPENEVOLVE-HEPHAEUSTUS INTEGRATION: END-TO-END SIMULATION")
        print("="*80)
        print(f"\nProblem: {self.problem_statement}\n")

        try:
            # Phase 1: Decomposition
            phase_2_tasks = self.simulate_phase_1()
            if not phase_2_tasks:
                print("\n✗ Phase 1 failed - workflow aborted")
                return False

            # Phase 2-4: Solve, critique, verify each sub-problem
            print("\n" + "="*80)
            print("  PHASES 2-4: SOLVING, CRITIQUE, VERIFICATION")
            print("="*80 + "\n")

            for task in phase_2_tasks:
                # Phase 2: Solve
                solution, metadata = self.simulate_phase_2(task)
                if not solution:
                    continue

                # Phase 3: Critique
                critique, approved = self.simulate_phase_3(solution, metadata)
                if not approved:
                    continue

                # Phase 4: Verify
                verified = self.simulate_phase_4(solution, critique, metadata)
                if verified:
                    self.workflow_state["verified_solutions"].append(solution)

            if not self.workflow_state["verified_solutions"]:
                print("\n✗ No solutions verified - workflow aborted")
                return False

            # Phase 5: Reassembly
            integrated_solution = self.simulate_phase_5()
            if not integrated_solution:
                print("\n✗ Reassembly failed - workflow aborted")
                return False

            # Phase 6: Final Verification
            success = self.simulate_phase_6(integrated_solution)

            return success

        except Exception as e:
            logger.error(f"Workflow simulation failed: {e}", exc_info=True)
            print(f"\n✗ Workflow failed with error: {e}")
            return False


def main():
    """Run the end-to-end simulation"""

    # Example problem statements
    examples = [
        "Implement a binary search tree in Python with insert, delete, and search operations",
        "Design a RESTful API for a task management system",
        "Solve the N-Queens problem using backtracking",
    ]

    print("\n" + "="*80)
    print("  OPENEVOLVE-HEPHAEUSTUS INTEGRATION TEST")
    print("="*80)
    print("\nAvailable Examples:")
    for i, example in enumerate(examples, 1):
        print(f"  {i}. {example[:70]}...")
    print("  0. Run all examples")

    try:
        choice = input("\nSelect example (0-3): ").strip()
        choice = int(choice)
    except (ValueError, KeyboardInterrupt):
        print("\nExiting...")
        return

    if choice == 0:
        # Run all examples
        for i, problem_statement in enumerate(examples, 1):
            print(f"\n\n{'#'*80}")
            print(f"  EXAMPLE {i}/{len(examples)}")
            print(f"{'#'*80}\n")

            simulator = WorkflowSimulator(problem_statement)
            success = simulator.run_workflow()

            if success:
                print(f"\n✓ Example {i} completed successfully")
            else:
                print(f"\n✗ Example {i} failed")

            if i < len(examples):
                input("\nPress Enter to continue to next example...")

    elif 1 <= choice <= len(examples):
        problem_statement = examples[choice - 1]
        simulator = WorkflowSimulator(problem_statement)
        success = simulator.run_workflow()

        if success:
            print("\n✓ Simulation completed successfully")
        else:
            print("\n✗ Simulation failed")

    else:
        print("Invalid choice")

    print("\n" + "="*80)
    print("  SIMULATION COMPLETE")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
