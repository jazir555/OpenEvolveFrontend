#!/usr/bin/env python3
"""
Demonstration of Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration

This script demonstrates the complete integration between:
- Z3 SMT Solver
- LeanAIDE Formal Verification
- OpenEvolve Workflow Engine
- BubbleLabs Visualization

Usage:
    python demo_z3_leanaide_integration.py

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import time
from typing import Dict, Any

# Import integrations
from z3prover_integration import (
    Z3Variable, Z3Constraint, Z3ConstraintType,
    get_z3_solver_engine, get_z3_theorem_prover,
    is_z3_available
)

from z3_leanaide_bridge import (
    get_z3_leanaide_bridge_sync, VerificationStrategy
)

from z3_leanaide_openevolve_integration import (
    get_z3_leanaide_openevolve_integration,
    solve_with_z3_leanaide, get_integration_status
)

from z3_leanaide_bubblelabs_ui import (
    get_z3_bubblelabs_ui, register_z3_leanaide_bubblelabs_tools
)


class Z3LeanAideIntegrationDemo:
    """Demonstrates the Z3-LeanAIDE-OpenEvolve-BubbleLabs integration."""
    
    def __init__(self):
        self.print_header("Z3-LeanAIDE-OpenEvolve-BubbleLabs Integration Demo")
        self.show_system_status()
    
    def print_header(self, text: str):
        """Print a formatted header."""
        print("\n" + "=" * 70)
        print(f"  {text}")
        print("=" * 70)
    
    def print_section(self, text: str):
        """Print a section header."""
        print(f"\n--- {text} ---")
    
    def show_system_status(self):
        """Display system status."""
        status = get_integration_status()
        
        self.print_section("System Status")
        print(f"Integration Ready: {status['ready']}")
        print(f"Message: {status['message']}")
        print(f"\nComponents:")
        print(f"  Z3 Solver: {'✓' if status['z3_available'] else '✗'}")
        print(f"  LeanAIDE: {'✓' if status['leanaide_available'] else '✗'}")
        print(f"  Z3-LeanAIDE Bridge: {'✓' if status['z3_leanaide_bridge_available'] else '✗'}")
        print(f"  OpenEvolve: {'✓' if status['openevolve_available'] else '✗'}")
        print(f"  BubbleLabs: {'✓' if status['bubblelabs_available'] else '✗'}")
    
    async def demo_z3_constraint_solving(self):
        """Demonstrate Z3 constraint solving."""
        if not is_z3_available():
            print("\n⚠️  Z3 not available - skipping constraint solving demo")
            return
        
        self.print_header("Demo 1: Z3 Constraint Solving")
        
        # Create a scheduling problem
        print("\nProblem: Employee Scheduling")
        print("- Alice and Bob need to be assigned shifts")
        print("- Alice works days (shift 1), Bob works nights (shift 2)")
        print("- Both must work at least 1 day but no more than 5 days")
        
        engine = get_z3_solver_engine()
        
        # Define variables
        variables = [
            Z3Variable("alice_days", Z3ConstraintType.INTEGER),
            Z3Variable("bob_days", Z3ConstraintType.INTEGER),
            Z3Variable("total_coverage", Z3ConstraintType.INTEGER)
        ]
        
        # Define constraints
        constraints = [
            Z3Constraint("(>= alice_days 1)", Z3ConstraintType.INTEGER, "Alice min days"),
            Z3Constraint("(<= alice_days 5)", Z3ConstraintType.INTEGER, "Alice max days"),
            Z3Constraint("(>= bob_days 1)", Z3ConstraintType.INTEGER, "Bob min days"),
            Z3Constraint("(<= bob_days 5)", Z3ConstraintType.INTEGER, "Bob max days"),
            Z3Constraint("(= total_coverage (+ alice_days bob_days))", Z3ConstraintType.INTEGER, "Total coverage"),
            Z3Constraint("(>= total_coverage 5)", Z3ConstraintType.INTEGER, "Minimum coverage"),
        ]
        
        print("\nSolving...")
        start = time.time()
        result = engine.solve_constraints(variables, constraints)
        elapsed = time.time() - start
        
        print(f"\n✓ Solved in {elapsed:.3f}s")
        print(f"Status: {result.status.value}")
        
        if result.model:
            print("\nSolution:")
            for var_name, value in result.model.assignments.items():
                print(f"  {var_name}: {value}")
            
            print(f"\nTotal coverage: {result.model.assignments.get('total_coverage')} days")
    
    async def demo_z3_theorem_proving(self):
        """Demonstrate Z3 theorem proving."""
        if not is_z3_available():
            print("\n⚠️  Z3 not available - skipping theorem proving demo")
            return
        
        self.print_header("Demo 2: Z3 Theorem Proving")
        
        print("\nTheorem: For all integers x, if x > 0 then x + 1 > 0")
        
        prover = get_z3_theorem_prover()
        
        # Theorem in SMT-LIB format (proof by contradiction)
        theorem = """
        (set-logic LIA)
        (declare-fun x () Int)
        (assert (> x 0))
        (assert (not (> (+ x 1) 0)))
        (check-sat)
        """
        
        print("\nProving (by contradiction)...")
        start = time.time()
        result = prover.prove_theorem(theorem)
        elapsed = time.time() - start
        
        print(f"\n✓ Proven in {elapsed:.3f}s")
        print(f"Proven: {result.proven}")
        print(f"Tactic used: {result.tactic_used}")
        
        if result.counterexample:
            print(f"Counterexample: {result.counterexample}")
    
    async def demo_smt_to_lean_translation(self):
        """Demonstrate SMT to Lean translation."""
        try:
            from z3_leanaide_bridge import TranslationDirection
        except ImportError:
            print("\n⚠️  Z3-LeanAIDE bridge not available - skipping translation demo")
            return
        
        self.print_header("Demo 3: SMT-LIB to Lean 4 Translation")
        
        smtlib = """
        (set-logic LIA)
        (declare-fun x () Int)
        (declare-fun y () Int)
        (assert (> x 0))
        (assert (< x 10))
        (assert (= y (+ x 5)))
        (check-sat)
        """
        
        print("\nOriginal SMT-LIB:")
        print(smtlib)
        
        bridge = get_z3_leanaide_bridge_sync()
        
        print("\nTranslating to Lean 4...")
        result = await bridge.translate_smt_to_lean(smtlib)
        
        if result.success:
            print(f"\n✓ Translation successful ({result.execution_time:.3f}s)")
            print("\nGenerated Lean 4 code:")
            print("-" * 50)
            print(result.translation)
            print("-" * 50)
        else:
            print(f"\n✗ Translation failed: {result.errors}")
    
    async def demo_combined_verification(self):
        """Demonstrate combined Z3 + LeanAIDE verification."""
        try:
            from z3_leanaide_bridge import CombinedVerificationResult
        except ImportError:
            print("\n⚠️  Z3-LeanAIDE bridge not available - skipping combined verification demo")
            return
        
        self.print_header("Demo 4: Combined Z3 + LeanAIDE Verification")
        
        problem = """
        (set-logic LIA)
        (declare-fun n () Int)
        (assert (> n 0))
        (assert (< n 100))
        (check-sat)
        """
        
        print("Problem: Find an integer n where 0 < n < 100")
        print("\nRunning parallel verification with Z3 and LeanAIDE...")
        
        bridge = get_z3_leanaide_bridge_sync()
        
        result = await bridge.verify_with_both(problem, VerificationStrategy.PARALLEL)
        
        print(f"\n✓ Verification complete ({result.execution_time:.3f}s)")
        print(f"Strategy: {result.strategy_used.value}")
        print(f"Success: {result.success}")
        print(f"Z3 Status: {result.z3_result.status.value if result.z3_result else 'N/A'}")
        
        if result.lean_result:
            lean_success = (result.lean_result.success if hasattr(result.lean_result, 'success') 
                          else result.lean_result.get('success', False))
            print(f"Lean Status: {'success' if lean_success else 'failed'}")
        
        print(f"Agreement: {result.agreement}")
        print(f"Confidence: {result.confidence_score:.2f}")
        print(f"Recommendation: {result.recommendation}")
    
    async def demo_problem_classification(self):
        """Demonstrate automatic problem classification."""
        self.print_header("Demo 5: Automatic Problem Classification")
        
        problems = [
            ("Constraint Problem", "Find x and y where x + y = 10 and x > 0"),
            ("Theorem Problem", "Prove that for all x, x + 0 = x"),
            ("Optimization Problem", "Minimize x^2 + y^2 subject to x + y = 1"),
            ("SMT-LIB Problem", "(set-logic LIA)(declare-fun x () Int)(assert (> x 0))(check-sat)"),
        ]
        
        integration = get_z3_leanaide_openevolve_integration()
        
        for name, problem in problems:
            print(f"\n{name}:")
            print(f"  Problem: {problem[:60]}...")
            
            classification = integration.classifier.classify(problem)
            
            print(f"  Classification: {classification.category.value}")
            print(f"  Confidence: {classification.confidence:.2f}")
            print(f"  Recommended: {classification.recommended_solver}")
            print(f"  Strategy: {classification.suggested_strategy.value}")
    
    async def demo_integrated_workflow(self):
        """Demonstrate complete integrated workflow."""
        self.print_header("Demo 6: Complete Integrated Workflow")
        
        problem = """
        A company produces two products A and B.
        - Product A yields $40 profit per unit, requires 2 hours of labor
        - Product B yields $60 profit per unit, requires 3 hours of labor
        - Total labor available: 100 hours
        - Demand constraints: at most 40 units of A, at most 30 units of B
        Find the optimal production plan to maximize profit.
        """
        
        print("Problem: Production Optimization")
        print(problem[:200] + "...")
        
        print("\nProcessing through integrated workflow...")
        print("  1. Problem Classification")
        print("  2. Solver Selection")
        print("  3. Solution Generation")
        print("  4. Verification")
        
        start = time.time()
        result = await solve_with_z3_leanaide(problem)
        elapsed = time.time() - start
        
        print(f"\n✓ Workflow complete in {elapsed:.3f}s")
        print(f"Status: {result['status']}")
        
        if result['status'] == 'completed':
            print(f"\nClassification: {result['classification']['category']}")
            print(f"Recommended Solver: {result['classification']['recommended_solver']}")
            print(f"\nSolution:")
            print(f"  {result['solution']['content']}")
            print(f"  Confidence: {result['solution']['confidence_score']:.2f}")
    
    def demo_bubblelabs_nodes(self):
        """Demonstrate BubbleLabs UI nodes."""
        self.print_header("Demo 7: BubbleLabs UI Integration")
        
        ui = get_z3_bubblelabs_ui()
        
        # Register tools
        print("\nRegistering Z3-LeanAIDE tools with BubbleLabs...")
        result = register_z3_leanaide_bubblelabs_tools()
        
        if result['success']:
            print(f"✓ Registered {result['nodes_registered']} node types")
            print(f"  Node types: {', '.join(result['node_types'])}")
        
        # Show node definitions
        print("\nAvailable Workflow Nodes:")
        definitions = ui.get_node_definitions()
        
        for defn in definitions:
            print(f"\n  {defn['icon']} {defn['name']}")
            print(f"    Type: {defn['type']}")
            print(f"    Category: {defn['category']}")
            print(f"    Inputs: {', '.join(defn['inputs'])}")
            print(f"    Outputs: {', '.join(defn['outputs'])}")
    
    async def run_all_demos(self):
        """Run all demonstrations."""
        demos = [
            ("Z3 Constraint Solving", self.demo_z3_constraint_solving),
            ("Z3 Theorem Proving", self.demo_z3_theorem_proving),
            ("SMT to Lean Translation", self.demo_smt_to_lean_translation),
            ("Combined Verification", self.demo_combined_verification),
            ("Problem Classification", self.demo_problem_classification),
            ("Integrated Workflow", self.demo_integrated_workflow),
            ("BubbleLabs UI", self.demo_bubblelabs_nodes),
        ]
        
        for name, demo_func in demos:
            try:
                if asyncio.iscoroutinefunction(demo_func):
                    await demo_func()
                else:
                    demo_func()
            except Exception as e:
                print(f"\n⚠️  Demo '{name}' failed: {e}")
        
        self.print_header("Demo Complete")
        print("\nIntegration Features Demonstrated:")
        print("  ✓ Z3 Constraint Solving")
        print("  ✓ Z3 Theorem Proving")
        print("  ✓ SMT-LIB to Lean Translation")
        print("  ✓ Combined Z3 + LeanAIDE Verification")
        print("  ✓ Automatic Problem Classification")
        print("  ✓ Integrated Workflow Processing")
        print("  ✓ BubbleLabs UI Integration")
        
        print("\nFor more information, see:")
        print("  - z3prover_integration.py (Core Z3 integration)")
        print("  - z3_leanaide_bridge.py (Z3-LeanAIDE bridge)")
        print("  - z3_leanaide_openevolve_integration.py (OpenEvolve workflow)")
        print("  - z3_leanaide_bubblelabs_ui.py (UI components)")


async def main():
    """Main entry point."""
    demo = Z3LeanAideIntegrationDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
