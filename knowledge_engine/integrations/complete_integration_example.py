"""
Complete Mathematical Knowledge Integration Example

This example demonstrates the full integration between:
- Z3 SMT solver (constraint solving)
- LeanAIDE (theorem proving)
- Knowledge extraction and learning
- Pattern matching and strategy recommendation
- OpenEvolve workflow integration
- BubbleLabs UI components

Usage:
    python complete_integration_example.py

Author: OpenEvolve
Created: 2026-01-31
"""

import asyncio
import json
import logging
from typing import Any, Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MathKnowledgeIntegrationDemo:
    """
    Complete demonstration of mathematical knowledge integration.
    
    This class shows how all components work together to provide
    a unified mathematical problem-solving experience.
    """
    
    def __init__(self):
        self.z3_connector = None
        self.leanaide_connector = None
        self.unified_bridge = None
        self.knowledge_manager = None
    
    async def initialize(self):
        """Initialize all components."""
        logger.info("Initializing mathematical knowledge integration...")
        
        # Initialize Z3 connector
        try:
            from z3_solver_connector import get_z3_connector
            self.z3_connector = get_z3_connector()
            logger.info("✓ Z3 connector initialized")
        except Exception as e:
            logger.warning(f"✗ Z3 connector failed: {e}")
        
        # Initialize LeanAIDE connector
        try:
            from leanaide_production_connector import get_leanaide_connector
            self.leanaide_connector = await get_leanaide_connector()
            logger.info("✓ LeanAIDE connector initialized")
        except Exception as e:
            logger.warning(f"✗ LeanAIDE connector failed: {e}")
        
        # Initialize unified bridge
        try:
            from unified_math_bridge_complete import get_unified_bridge_complete
            self.unified_bridge = await get_unified_bridge_complete()
            logger.info("✓ Unified bridge initialized")
        except Exception as e:
            logger.warning(f"✗ Unified bridge failed: {e}")
        
        # Initialize knowledge manager
        try:
            from z3_knowledge_complete import get_z3_knowledge_manager
            self.knowledge_manager = await get_z3_knowledge_manager()
            logger.info("✓ Knowledge manager initialized")
        except Exception as e:
            logger.warning(f"✗ Knowledge manager failed: {e}")
        
        logger.info("Initialization complete")
    
    async def demo_1_basic_solving(self):
        """
        Demo 1: Basic Problem Solving
        
        Shows how to solve problems using individual solvers.
        """
        print("\n" + "="*70)
        print("DEMO 1: Basic Problem Solving")
        print("="*70)
        
        # Example: Linear constraints
        problem = """
        Find x, y such that:
        - x + 2*y = 10
        - 3*x - y = 5
        - x >= 0, y >= 0
        """
        
        print(f"\nProblem: {problem}")
        
        # Solve with Z3
        if self.z3_connector:
            smtlib = """
            (declare-fun x () Int)
            (declare-fun y () Int)
            (assert (= (+ x (* 2 y)) 10))
            (assert (= (- (* 3 x) y) 5))
            (assert (>= x 0))
            (assert (>= y 0))
            (check-sat)
            (get-model)
            """
            
            from z3_solver_connector import Z3SolverConfig
            result = await self.z3_connector.solve_smtlib(
                smtlib, 
                Z3SolverConfig(timeout_ms=10000, model_generation=True)
            )
            
            print(f"\nZ3 Result:")
            print(f"  Status: {result.status.value}")
            print(f"  Model: {result.model}")
            print(f"  Time: {result.solving_time_ms:.2f}ms")
        
        # Solve with Lean
        if self.leanaide_connector:
            theorem = """
            theorem algebra_001 :
              ∃ x y : ℤ,
                x + 2*y = 10 ∧
                3*x - y = 5 ∧
                x ≥ 0 ∧
                y ≥ 0 := by
            """
            
            result = await self.leanaide_connector.prove_theorem(
                theorem,
                auto_tactics=["use 20 / 7", "norm_num", "linarith"]
            )
            
            print(f"\nLeanAIDE Result:")
            print(f"  Success: {result.get('success')}")
            print(f"  Proof: {result.get('proof', 'N/A')[:100]}...")
    
    async def demo_2_unified_solving(self):
        """
        Demo 2: Unified Problem Solving
        
        Shows how the unified bridge selects the optimal solver
        and validates results across systems.
        """
        print("\n" + "="*70)
        print("DEMO 2: Unified Problem Solving with Consensus")
        print("="*70)
        
        if not self.unified_bridge:
            print("Unified bridge not available, skipping demo")
            return
        
        problems = [
            {
                "name": "Linear System",
                "description": "x + y = 5, x - y = 1",
                "expected": "sat"
            },
            {
                "name": "Nonlinear Equation",
                "description": "x^2 + y^2 = 25, x + y = 7",
                "expected": "sat"
            },
            {
                "name": "Unsatisfiable",
                "description": "x > 5 AND x < 3",
                "expected": "unsat"
            }
        ]
        
        for problem in problems:
            print(f"\nProblem: {problem['name']}")
            print(f"Description: {problem['description']}")
            
            from unified_math_bridge_complete import SolverSystem
            
            result = await self.unified_bridge.solve(
                problem=problem['description'],
                preferred_solver=SolverSystem.HYBRID,
                timeout=60
            )
            
            print(f"  Result: {result.get('result_status')}")
            print(f"  Primary Solver: {result.get('primary_solver')}")
            print(f"  Consensus: {result.get('consensus_status')}")
            print(f"  Verified: {result.get('verified', False)}")
    
    async def demo_3_knowledge_extraction(self):
        """
        Demo 3: Knowledge Extraction and Learning
        
        Shows how the system learns from solved problems.
        """
        print("\n" + "="*70)
        print("DEMO 3: Knowledge Extraction and Learning")
        print("="*70)
        
        if not self.knowledge_manager:
            print("Knowledge manager not available, skipping demo")
            return
        
        # Example problems to learn from
        examples = [
            {
                "problem": "Linear system with 2 variables",
                "constraints": ["x + y = 10", "2x - y = 5"],
                "result": "success",
                "strategy": "substitution",
                "time_ms": 45.2
            },
            {
                "problem": "Quadratic optimization",
                "constraints": ["x^2 + y^2 <= 100", "x >= 0", "y >= 0"],
                "result": "success",
                "strategy": "gradient_descent",
                "time_ms": 120.5
            },
            {
                "problem": "Boolean SAT",
                "constraints": ["a OR b", "NOT a OR c", "b AND c"],
                "result": "success",
                "strategy": "dpll",
                "time_ms": 15.3
            }
        ]
        
        print("\nLearning from example problems...")
        
        for i, example in enumerate(examples, 1):
            # Extract knowledge
            await self.knowledge_manager.learn_from_solution(
                problem_statement=example["problem"],
                constraints=example["constraints"],
                result=example["result"],
                metadata={
                    "strategy": example["strategy"],
                    "solving_time_ms": example["time_ms"]
                }
            )
            
            print(f"  [{i}/3] Learned: {example['problem']}")
            print(f"         Strategy: {example['strategy']}")
            print(f"         Time: {example['time_ms']}ms")
        
        print("\nKnowledge base statistics:")
        stats = self.knowledge_manager.get_statistics()
        print(f"  Total records: {stats.get('total_records', 0)}")
        print(f"  Pattern count: {stats.get('pattern_count', 0)}")
        print(f"  Concept count: {stats.get('concept_count', 0)}")
    
    async def demo_4_pattern_matching(self):
        """
        Demo 4: Pattern Matching and Strategy Recommendation
        
        Shows how to find similar problems and get strategy recommendations.
        """
        print("\n" + "="*70)
        print("DEMO 4: Pattern Matching and Strategy Recommendation")
        print("="*70)
        
        if not self.knowledge_manager:
            print("Knowledge manager not available, skipping demo")
            return
        
        # New problem to solve
        new_problem = "System with two linear equations and positivity constraints"
        new_constraints = ["3x + 2y = 15", "x - y = 1", "x >= 0", "y >= 0"]
        
        print(f"\nNew problem: {new_problem}")
        print(f"Constraints: {new_constraints}")
        
        # Find similar solutions
        similar = await self.knowledge_manager.find_similar_solutions(
            problem_statement=new_problem,
            constraints=new_constraints,
            top_k=3
        )
        
        print(f"\nFound {len(similar)} similar solutions:")
        for i, sol in enumerate(similar, 1):
            print(f"  [{i}] Similarity: {sol.get('similarity', 0):.2f}")
            print(f"      Problem: {sol.get('problem', 'N/A')[:50]}...")
            print(f"      Strategy: {sol.get('metadata', {}).get('strategy', 'unknown')}")
        
        # Get strategy recommendation
        strategy = await self.knowledge_manager.get_recommended_strategy(
            problem_statement=new_problem,
            constraints=new_constraints
        )
        
        print(f"\nRecommended strategy:")
        print(f"  Strategy: {strategy.get('strategy', 'unknown')}")
        print(f"  Confidence: {strategy.get('confidence', 0):.2f}")
        print(f"  Expected time: {strategy.get('expected_time_ms', 0):.1f}ms")
    
    async def demo_5_cross_system_learning(self):
        """
        Demo 5: Cross-System Knowledge Transfer
        
        Shows how knowledge transfers between Z3 and LeanAIDE.
        """
        print("\n" + "="*70)
        print("DEMO 5: Cross-System Knowledge Transfer")
        print("="*70)
        
        if not self.unified_bridge:
            print("Unified bridge not available, skipping demo")
            return
        
        print("\nAnalyzing cross-system knowledge transfer...")
        
        transfer_report = await self.unified_bridge.analyze_cross_system_transfer()
        
        print("\nTransfer Analysis:")
        print(f"  Total patterns analyzed: {transfer_report.get('total_patterns', 0)}")
        print(f"  Successful transfers: {transfer_report.get('successful_transfers', 0)}")
        print(f"  Success rate: {transfer_report.get('success_rate', 0)*100:.1f}%")
        print(f"  Average adaptation time: {transfer_report.get('avg_adaptation_time', 0):.1f}ms")
        
        # Show example transfer
        if transfer_report.get('transfer_examples'):
            example = transfer_report['transfer_examples'][0]
            print(f"\nExample Transfer:")
            print(f"  Source: {example.get('source_system')}")
            print(f"  Target: {example.get('target_system')}")
            print(f"  Pattern: {example.get('pattern', 'N/A')[:50]}...")
            print(f"  Success: {example.get('success', False)}")
    
    async def demo_6_workflow_integration(self):
        """
        Demo 6: OpenEvolve Workflow Integration
        
        Shows how mathematical knowledge integrates with OpenEvolve workflows.
        """
        print("\n" + "="*70)
        print("DEMO 6: OpenEvolve Workflow Integration")
        print("="*70)
        
        print("\nSimulating OpenEvolve workflow...")
        
        # Simulate workflow stages
        workflow_stages = [
            {
                "stage": "problem_analysis",
                "description": "Analyze problem type and complexity",
                "action": self._analyze_problem,
                "input": "Find integer solutions to x^2 + y^2 = z^2 with x,y,z < 100"
            },
            {
                "stage": "solver_selection",
                "description": "Select optimal solver",
                "action": self._select_solver,
                "input": None
            },
            {
                "stage": "solution_attempt",
                "description": "Attempt to solve",
                "action": self._attempt_solution,
                "input": None
            },
            {
                "stage": "knowledge_extraction",
                "description": "Extract and store knowledge",
                "action": self._extract_knowledge,
                "input": None
            },
            {
                "stage": "result_integration",
                "description": "Integrate result into workflow",
                "action": self._integrate_result,
                "input": None
            }
        ]
        
        context = {}
        
        for stage in workflow_stages:
            print(f"\n  Stage: {stage['stage']}")
            print(f"  Description: {stage['description']}")
            
            try:
                result = await stage['action'](stage['input'], context)
                context[stage['stage']] = result
                print(f"  Status: ✓ Success")
                print(f"  Result: {json.dumps(result, indent=4)[:150]}...")
            except Exception as e:
                print(f"  Status: ✗ Failed - {e}")
    
    async def _analyze_problem(self, input_data, context):
        """Analyze problem in workflow."""
        # Extract features
        return {
            "problem_type": "diophantine",
            "complexity": "medium",
            "estimated_time_ms": 500,
            "recommended_solver": "hybrid"
        }
    
    async def _select_solver(self, input_data, context):
        """Select solver in workflow."""
        return {
            "primary_solver": "z3",
            "backup_solver": "lean",
            "strategy": "incremental"
        }
    
    async def _attempt_solution(self, input_data, context):
        """Attempt solution in workflow."""
        if self.unified_bridge:
            result = await self.unified_bridge.solve(
                problem="x^2 + y^2 = z^2, x,y,z < 100",
                preferred_solver=0  # AUTO
            )
            return {
                "success": result.get('result_status') == 'sat',
                "solver_used": result.get('primary_solver'),
                "time_ms": result.get('metadata', {}).get('solving_time_ms', 0)
            }
        return {"success": True, "mock": True}
    
    async def _extract_knowledge(self, input_data, context):
        """Extract knowledge in workflow."""
        if self.knowledge_manager:
            await self.knowledge_manager.learn_from_solution(
                problem_statement="Pythagorean triples search",
                constraints=["x^2 + y^2 = z^2", "x<100", "y<100", "z<100"],
                result="success"
            )
        return {"knowledge_extracted": True}
    
    async def _integrate_result(self, input_data, context):
        """Integrate result in workflow."""
        return {
            "integrated": True,
            "next_steps": ["verify_solution", "update_metrics"]
        }
    
    async def demo_7_bubblelabs_ui(self):
        """
        Demo 7: BubbleLabs UI Integration
        
        Shows how the system integrates with BubbleLabs UI components.
        """
        print("\n" + "="*70)
        print("DEMO 7: BubbleLabs UI Integration")
        print("="*70)
        
        print("\nBubbleLabs UI Components:")
        
        # Problem input component
        print("\n  1. Problem Input Component")
        print("     - Natural language input: 'Find all integer solutions to x^2 + y^2 = 25'")
        print("     - SMT-LIB editor with syntax highlighting")
        print("     - Lean theorem statement builder")
        print("     - File upload (SMT2, Lean, TPTP)")
        
        # Solver selection component
        print("\n  2. Solver Selection Component")
        print("     - Auto-detect (recommended)")
        print("     - Z3 SMT Solver")
        print("     - Lean 4 Theorem Prover")
        print("     - Hybrid (both with consensus)")
        
        # Results visualization component
        print("\n  3. Results Visualization Component")
        print("     - Solution display (model/proof)")
        print("     - Execution trace")
        print("     - Performance metrics")
        print("     - Verification status")
        
        # Knowledge base explorer
        print("\n  4. Knowledge Base Explorer")
        print("     - Pattern browser")
        print("     - Strategy statistics")
        print("     - Similar problem finder")
        print("     - Learning progress dashboard")
        
        # Integration workflow
        print("\n  5. OpenEvolve Workflow Integration")
        print("     - Drag-and-drop problem nodes")
        print("     - Visual solver pipeline")
        print("     - Real-time collaboration")
        print("     - Version control for proofs")
        
        # MCP tool invocation
        print("\n  6. MCP Tool Invocation")
        print("     - Claude/Cursor integration")
        print("     - Natural language problem solving")
        print("     - Automated proof suggestions")
        print("     - Code generation from proofs")
    
    async def run_all_demos(self):
        """Run all demonstrations."""
        print("\n" + "="*70)
        print("MATHEMATICAL KNOWLEDGE INTEGRATION - COMPLETE DEMO")
        print("="*70)
        print("\nThis demo showcases the full integration between:")
        print("  • Z3 SMT Solver (constraint satisfaction)")
        print("  • LeanAIDE (theorem proving)")
        print("  • Knowledge extraction and learning")
        print("  • Pattern matching and strategy recommendation")
        print("  • OpenEvolve workflow integration")
        print("  • BubbleLabs UI components")
        
        # Initialize
        await self.initialize()
        
        # Run demos
        demos = [
            ("Basic Solving", self.demo_1_basic_solving),
            ("Unified Solving", self.demo_2_unified_solving),
            ("Knowledge Extraction", self.demo_3_knowledge_extraction),
            ("Pattern Matching", self.demo_4_pattern_matching),
            ("Cross-System Learning", self.demo_5_cross_system_learning),
            ("Workflow Integration", self.demo_6_workflow_integration),
            ("BubbleLabs UI", self.demo_7_bubblelabs_ui),
        ]
        
        for name, demo_func in demos:
            try:
                await demo_func()
            except Exception as e:
                logger.error(f"Demo '{name}' failed: {e}")
        
        # Summary
        print("\n" + "="*70)
        print("DEMO COMPLETE")
        print("="*70)
        print("\nSummary:")
        print(f"  • Total demos: {len(demos)}")
        print(f"  • Z3 available: {self.z3_connector is not None}")
        print(f"  • LeanAIDE available: {self.leanaide_connector is not None}")
        print(f"  • Unified bridge available: {self.unified_bridge is not None}")
        print(f"  • Knowledge manager available: {self.knowledge_manager is not None}")
        print("\nNext steps:")
        print("  1. Deploy with Docker: docker-compose up -d")
        print("  2. Access API at: http://localhost:8765")
        print("  3. View metrics at: http://localhost:9090")
        print("  4. Explore knowledge base via MCP tools")


async def main():
    """Main entry point."""
    demo = MathKnowledgeIntegrationDemo()
    await demo.run_all_demos()


if __name__ == "__main__":
    asyncio.run(main())
