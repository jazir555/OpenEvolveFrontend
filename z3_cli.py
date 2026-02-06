"""
Z3 Integration Command Line Interface

Unified CLI for the Z3-LeanAIDE-OpenEvolve-BubbleLabs integration.

Commands:
- solve: Solve constraint problems [--use-cav-nlp]
- solve-batch: Batch problem solving from JSON file
- solve-portfolio: Portfolio/multi-strategy solving
- solve-incremental: Interactive incremental solving with push/pop/add/check
- optimize: Run single-objective optimization
- optimize-multi: Multi-objective optimization from JSON file
- prove: Prove theorems
- translate: Translate between formats
- server: Run API server
- monitor: Show performance metrics
- config: Manage configuration
- knowledge: Query knowledge base

CAV-NLP Commands:
- formalize: Formalize natural language to Lean/Z3
- verify: Verify constraint with optional hybrid Z3+Lean
- canonicalize: Canonicalize constraint to standard form

Web3 Formal Commands:
- web3-translate-invariant: Translate Solidity updates to Z3/Lean invariants
- web3-solve-witness: Solve symbolic exploit witness predicates
- web3-audit-exploit-verification: Combined invariant + witness exploit verification

Author: OpenEvolve
Created: 2026-01-31
"""


import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# CAV-NLP Integration
try:
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    from openevolve.unified_math_service import UnifiedMathService
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False

# CLI framework
try:
    import click
    from click import echo, style
    CLICK_AVAILABLE = True
except ImportError:
    CLICK_AVAILABLE = False
    # Create dummy click module
    class click:
        @staticmethod
        def command(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def option(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def argument(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def group(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def pass_context(*args, **kwargs):
            return lambda f: f
        @staticmethod
        def echo(message, *args, **kwargs):
            print(message)
    echo = print

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# CLI Group
# =============================================================================

if CLICK_AVAILABLE:
    @click.group()
    @click.version_option(version="2.0.0")
    @click.option('--config', '-c', help='Configuration file path')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
    @click.pass_context
    def cli(ctx, config, verbose):
        """Z3-LeanAIDE-OpenEvolve Integration CLI"""
        ctx.ensure_object(dict)
        ctx.obj['config_path'] = config
        ctx.obj['verbose'] = verbose
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)


    # =============================================================================
    # Solve Command
    # =============================================================================

    @cli.command()
    @click.argument('problem', type=str)
    @click.option('--variables', '-v', help='Variables JSON')
    @click.option('--constraints', '-c', help='Constraints JSON')
    @click.option('--timeout', '-t', default=60.0, help='Timeout in seconds')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    @click.option('--use-cav-nlp', is_flag=True, help='Use CAV-NLP enhancement for natural language problems')
    def solve(problem, variables, constraints, timeout, output, output_format, use_cav_nlp):
        """Solve a constraint satisfaction problem."""
        try:
            from z3prover_integration import get_z3_solver_engine, Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style("Solving constraint problem...", fg='blue'))
            echo(f"Problem: {problem[:100]}...")
            
            # Use CAV-NLP enhanced solver if requested and available
            if use_cav_nlp and CAV_NLP_AVAILABLE:
                echo(style("Using CAV-NLP enhancement...", fg='cyan'))
                solver = EnhancedZ3Solver()
            else:
                if use_cav_nlp and not CAV_NLP_AVAILABLE:
                    echo(style("Warning: CAV-NLP not available, using standard solver", fg='yellow'))
                solver = get_z3_solver_engine()
            
            # Parse inputs
            vars_list = json.loads(variables) if variables else []
            constraints_list = json.loads(constraints) if constraints else []
            
            z3_vars = [
                Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()])
                for v in vars_list
            ]
            
            z3_constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in constraints_list
            ]
            
            # If problem is provided and not empty, add it as a constraint 
            # or treat as SMT-LIB
            is_smtlib = any(kw in problem for kw in ['(assert', '(declare-fun', '(check-sat)'])
            
            # Solve
            import time
            start = time.time()
            
            if is_smtlib:
                result = solver.solve_smtlib(problem)
            else:
                if problem and problem.strip():
                    z3_constraints.append(Z3Constraint(problem, Z3ConstraintType.INTEGER))
                result = solver.solve_constraints(z3_vars, z3_constraints)
            
            elapsed = (time.time() - start) * 1000
            
            # Format output
            output_data = {
                "success": True,
                "status": result.status.value,
                "satisfiable": result.is_sat(),
                "model": result.model.assignments if result.model else None,
                "execution_time_ms": elapsed
            }
            
            _output_result(output_data, output, output_format)
            
            if result.is_sat():
                echo(style(f"[OK] SATISFIABLE", fg='green'))
                if result.model:
                    echo("Solution:")
                    for var, val in result.model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style(f"[FAIL] {result.status.value.upper()}", fg='red'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # CAV-NLP Commands
    # =============================================================================

    @cli.command()
    @click.argument('text', type=str)
    @click.option('--elaborate', is_flag=True, help='Elaborate with LeanAide before formalization')
    @click.option('--output', '-o', help='Output file for formalized code')
    def formalize(text, elaborate, output):
        """Formalize natural language to Lean/Z3 using CAV-NLP.
        
        Examples:
            z3 formalize "x is greater than 0 and less than 100"
            z3 formalize "forall integers x, x squared is non-negative" --elaborate
            z3 formalize "sum of two even numbers is even" -o output.lean
        """
        if not CAV_NLP_AVAILABLE:
            echo(style("Error: CAV-NLP not available. Install required dependencies.", fg='red'), err=True)
            sys.exit(1)
        
        try:
            echo(style("Formalizing natural language...", fg='blue'))
            echo(f"Input: {text}")
            
            service = UnifiedMathService()
            result = asyncio.run(service.formalize(text, elaborate=elaborate))
            
            output_data = {
                "success": result.success,
                "code": result.code,
                "language": result.language,
                "confidence": result.confidence,
                "elaborated": result.elaborated,
                "errors": result.errors
            }
            
            if output:
                Path(output).write_text(result.code)
                echo(style(f"Formalized code written to {output}", fg='green'))
            
            if result.success:
                echo(style(f"[OK] Formalized to {result.language}", fg='green'))
                echo(f"Confidence: {result.confidence:.1%}")
                if result.elaborated:
                    echo(style("(Elaborated with LeanAide)", fg='cyan'))
                echo("\nFormalized code:")
                echo(result.code)
            else:
                echo(style("[FAIL] Formalization failed", fg='red'))
                if result.errors:
                    echo("Errors:")
                    for error in result.errors:
                        echo(f"  - {error}")
            
            if not output:
                echo(json.dumps(output_data, indent=2))
                
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    @cli.command()
    @click.argument('constraint', type=str)
    @click.option('--hybrid', is_flag=True, help='Use hybrid Z3+Lean verification')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def verify(constraint, hybrid, output, output_format):
        """Verify constraint with optional hybrid verification.
        
        Examples:
            z3 verify "x > 0 and x < 100"
            z3 verify "forall x: x * x >= 0" --hybrid
            z3 verify "n > 2 implies n^2 > 4" --hybrid -o result.json
        """
        try:
            echo(style("Verifying constraint...", fg='blue'))
            echo(f"Constraint: {constraint}")
            
            if hybrid and CAV_NLP_AVAILABLE:
                echo(style("Using hybrid Z3+Lean verification...", fg='cyan'))
                solver = EnhancedZ3Solver()
                result = asyncio.run(solver.verify_with_lean(constraint))
                
                output_data = {
                    "success": result.success,
                    "verified": result.verified,
                    "confidence": result.confidence,
                    "method": result.method,
                    "z3_result": result.z3_result,
                    "lean_result": result.lean_result,
                    "errors": result.errors
                }
                
                if result.verified:
                    echo(style(f"[OK] Verified (Confidence: {result.confidence:.1%})", fg='green'))
                    echo(f"Method: {result.method}")
                else:
                    echo(style("[FAIL] Verification failed", fg='red'))
                    if result.errors:
                        echo("Errors:")
                        for error in result.errors:
                            echo(f"  - {error}")
                
                _output_result(output_data, output, output_format)
                
            else:
                if hybrid and not CAV_NLP_AVAILABLE:
                    echo(style("Warning: CAV-NLP not available, using standard Z3 verification", fg='yellow'))
                
                # Standard Z3 verification
                from z3prover_integration import get_z3_solver_engine, Z3Constraint, Z3ConstraintType
                
                solver = get_z3_solver_engine()
                z3_constraint = Z3Constraint(constraint, Z3ConstraintType.BOOLEAN)
                
                import time
                start = time.time()
                result = solver.solve_constraints([], [z3_constraint])
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": result.status.value == 'sat',
                    "verified": result.is_sat(),
                    "status": result.status.value,
                    "method": "z3",
                    "execution_time_ms": elapsed
                }
                
                if result.is_sat():
                    echo(style("[OK] Constraint is satisfiable", fg='green'))
                elif result.is_unsat():
                    echo(style("[OK] Constraint is valid (unsatisfiable negation)", fg='green'))
                else:
                    echo(style("[FAIL] Verification inconclusive", fg='yellow'))
                
                _output_result(output_data, output, output_format)
                
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    @cli.command()
    @click.argument('constraint', type=str)
    @click.option('--output', '-o', help='Output file')
    def canonicalize(constraint, output):
        """Canonicalize constraint using CAV-NLP.
        
        Converts constraints to a standardized canonical form for easier
        comparison and pattern matching.
        
        Examples:
            z3 canonicalize "x > 0 and x < 100"
            z3 canonicalize "y + x = 10"
            z3 canonicalize "a implies b" -o canonical.txt
        """
        if not CAV_NLP_AVAILABLE:
            echo(style("Error: CAV-NLP not available. Install required dependencies.", fg='red'), err=True)
            sys.exit(1)
        
        try:
            echo(style("Canonicalizing constraint...", fg='blue'))
            echo(f"Input: {constraint}")
            
            solver = EnhancedZ3Solver()
            canonical = solver.canonical_manager.canonicalize(constraint)
            
            output_data = {
                "success": True,
                "input": constraint,
                "canonical": canonical
            }
            
            if output:
                Path(output).write_text(canonical)
                echo(style(f"Canonical form written to {output}", fg='green'))
            
            echo(style("[OK] Canonicalized", fg='green'))
            echo(f"Input:    {constraint}")
            echo(f"Canonical: {canonical}")
            
            if not output:
                echo(json.dumps(output_data, indent=2))
                
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    @cli.command('web3-translate-invariant')
    @click.argument('statement', type=str)
    @click.option('--non-negative-target/--allow-negative-target', default=True)
    @click.option('--max-withdraw-expr', help='Optional withdrawal upper-bound expression')
    @click.option('--verify/--no-verify', default=True)
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def web3_translate_invariant(statement, non_negative_target, max_withdraw_expr, verify, output, output_format):
        """Translate Solidity assignment/update semantics into Z3/Lean invariants."""
        try:
            from z3prover_integration import (
                translate_solidity_assignment_to_z3,
                verify_solidity_invariant_translation,
            )

            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
            )
            result = {"success": True, "translation": translation}
            if verify:
                result["verification"] = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=True,
                )

            _output_result(result, output, output_format)
            echo(style("[OK] Web3 invariant translation complete", fg='green'))
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    @cli.command('web3-solve-witness')
    @click.option('--constraints', '-c', help='JSON array of additional constraints')
    @click.option('--timeout', '-t', default=10.0, type=float)
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def web3_solve_witness(constraints, timeout, output, output_format):
        """Solve symbolic exploit witness query for smart-contract balance drain predicates."""
        try:
            from z3prover_integration import solve_smart_contract_exploit_witness

            parsed_constraints = []
            if constraints:
                parsed_constraints = json.loads(constraints)
                if not isinstance(parsed_constraints, list):
                    raise ValueError("--constraints must be a JSON list of strings")

            result = solve_smart_contract_exploit_witness(
                additional_constraints=parsed_constraints,
                timeout=timeout,
            )
            wrapped = {"success": True, "result": result}
            _output_result(wrapped, output, output_format)
            if result.get("satisfiable"):
                echo(style("[OK] Exploit witness found (SAT)", fg='green'))
            else:
                echo(style("[FAIL] No exploit witness found", fg='yellow'))
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    @cli.command('web3-audit-exploit-verification')
    @click.argument('statement', type=str, required=False, default='balance[msg.sender] -= amount;')
    @click.option('--non-negative-target/--allow-negative-target', default=True)
    @click.option('--max-withdraw-expr', help='Optional withdrawal upper-bound expression')
    @click.option('--verify/--no-verify', default=True)
    @click.option('--constraints', '-c', help='JSON array of additional constraints')
    @click.option('--timeout', '-t', default=10.0, type=float)
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def web3_audit_exploit_verification(
        statement,
        non_negative_target,
        max_withdraw_expr,
        verify,
        constraints,
        timeout,
        output,
        output_format,
    ):
        """Run combined Web3 exploit verification (translate + verify + witness)."""
        try:
            from z3prover_integration import (
                solve_smart_contract_exploit_witness,
                translate_solidity_assignment_to_z3,
                verify_solidity_invariant_translation,
            )

            parsed_constraints = []
            if constraints:
                parsed_constraints = json.loads(constraints)
                if not isinstance(parsed_constraints, list):
                    raise ValueError("--constraints must be a JSON list of strings")

            translation = translate_solidity_assignment_to_z3(
                statement=statement,
                non_negative_target=non_negative_target,
                max_withdraw_expr=max_withdraw_expr,
            )
            verification = None
            if verify:
                verification = verify_solidity_invariant_translation(
                    translation=translation,
                    assume_non_negative_amount=True,
                )

            witness = solve_smart_contract_exploit_witness(
                additional_constraints=parsed_constraints,
                timeout=timeout,
            )

            verified_exploit = bool(witness.get("satisfiable", False))
            if verify and isinstance(verification, dict):
                verified_exploit = verified_exploit and bool(verification.get("proven", False))

            result = {
                "success": True,
                "translation": translation,
                "verification": verification,
                "exploit_witness": witness,
                "verified_exploit": verified_exploit,
            }
            _output_result(result, output, output_format)
            echo(style("[OK] Web3 exploit verification complete", fg='green'))
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Optimize Command
    # =============================================================================

    @cli.command()
    @click.argument('objective', type=str)
    @click.option('--variables', '-v', required=True, help='Variables JSON')
    @click.option('--constraints', '-c', required=True, help='Constraints JSON')
    @click.option('--direction', '-d', default='minimize', type=click.Choice(['minimize', 'maximize']))
    @click.option('--output', '-o', help='Output file')
    def optimize(objective, variables, constraints, direction, output):
        """Run optimization."""
        try:
            from z3prover_advanced import get_z3_advanced_solver, OptimizationObjective
            from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style("Running optimization...", fg='blue'))
            
            solver = get_z3_advanced_solver()
            
            vars_list = json.loads(variables)
            constraints_list = json.loads(constraints)
            
            z3_vars = [Z3Variable(v['name'], Z3ConstraintType[v.get('type', 'INTEGER').upper()]) for v in vars_list]
            z3_constraints = [Z3Constraint(c, Z3ConstraintType.INTEGER) for c in constraints_list]
            
            obj_type = OptimizationObjective.MINIMIZE if direction == 'minimize' else OptimizationObjective.MAXIMIZE
            
            result = solver.optimize(z3_vars, z3_constraints, [(objective, obj_type)])
            
            if result.success:
                echo(style(f"[OK] Optimal value: {result.optimal_value}", fg='green'))
                if result.optimal_model:
                    echo("Optimal solution:")
                    for var, val in result.optimal_model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style("[FAIL] Optimization failed", fg='red'))
            
            if output:
                with open(output, 'w') as f:
                    json.dump(result.to_dict(), f, indent=2)
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Multi-Objective Optimize Command
    # =============================================================================

    @cli.command('optimize-multi')
    @click.argument('input_file', type=click.Path(exists=True))
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def optimize_multi(input_file, output, output_format):
        """Run multi-objective optimization from a JSON input file.

        Input file format:
        {
          "variables": [
            {"name": "x", "type": "INTEGER"},
            {"name": "y", "type": "INTEGER"}
          ],
          "constraints": ["x > 0", "y > 0", "x + y < 100"],
          "objectives": [
            {"expression": "x + y", "direction": "maximize"},
            {"expression": "x - y", "direction": "minimize"}
          ]
        }

        Examples:
            z3 optimize-multi objectives.json
            z3 optimize-multi objectives.json -o results.json
            z3 optimize-multi objectives.json --format yaml
        """
        try:
            from z3prover_advanced import get_z3_advanced_solver, OptimizationObjective
            from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style("Running multi-objective optimization...", fg='blue'))
            
            # Load input file
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            solver = get_z3_advanced_solver()
            
            # Parse variables
            vars_list = data.get('variables', [])
            z3_vars = [
                Z3Variable(
                    v['name'], 
                    Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                    bit_width=v.get('bit_width')
                )
                for v in vars_list
            ]
            
            # Parse constraints
            constraints_list = data.get('constraints', [])
            z3_constraints = [
                Z3Constraint(c, Z3ConstraintType.INTEGER)
                for c in constraints_list
            ]
            
            # Parse multiple objectives
            objectives_data = data.get('objectives', [])
            if not objectives_data:
                echo(style("Error: No objectives specified in input file", fg='red'), err=True)
                sys.exit(1)
            
            objectives = []
            for obj in objectives_data:
                direction = obj.get('direction', 'minimize')
                obj_type = OptimizationObjective.MINIMIZE if direction == 'minimize' else OptimizationObjective.MAXIMIZE
                objectives.append((obj['expression'], obj_type))
            
            import time
            start = time.time()
            
            result = solver.optimize(z3_vars, z3_constraints, objectives)
            
            elapsed = (time.time() - start) * 1000
            
            # Format output
            output_data = {
                "success": result.success,
                "optimal_value": result.optimal_value,
                "model": result.optimal_model.assignments if result.optimal_model else None,
                "is_pareto": result.is_pareto,
                "pareto_front_size": len(result.pareto_front),
                "execution_time_ms": elapsed
            }
            
            if result.is_pareto and result.pareto_front:
                output_data["pareto_front"] = [
                    {"value": p.value, "assignments": p.assignments}
                    for p in result.pareto_front
                ]
            
            _output_result(output_data, output, output_format)
            
            if result.success:
                echo(style(f"[OK] Multi-objective optimization complete", fg='green'))
                if result.is_pareto:
                    echo(f"Pareto front size: {len(result.pareto_front)}")
                if result.optimal_model:
                    echo("Optimal solution:")
                    for var, val in result.optimal_model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style("[FAIL] Optimization failed", fg='red'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Batch Solve Command
    # =============================================================================

    @cli.command('solve-batch')
    @click.argument('input_file', type=click.Path(exists=True))
    @click.option('--parallel/--sequential', default=True, help='Run problems in parallel')
    @click.option('--max-workers', '-w', default=4, help='Maximum parallel workers')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def solve_batch(input_file, parallel, max_workers, output, output_format):
        """Solve multiple constraint problems in batch from a JSON input file.

        Input file format:
        {
          "problems": [
            {
              "problem": "x + y = 10",
              "variables": [{"name": "x", "type": "INTEGER"}, {"name": "y", "type": "INTEGER"}],
              "constraints": ["x > 0", "y > 0"]
            },
            ...
          ]
        }

        Or for SMT-LIB problems:
        {
          "problems": [
            {"problem": "(declare-fun x () Int) (assert (> x 0)) (check-sat)"},
            ...
          ]
        }

        Examples:
            z3 solve-batch problems.json
            z3 solve-batch problems.json --parallel -w 8
            z3 solve-batch problems.json --sequential -o results.json
        """
        try:
            from z3prover_integration import get_z3_solver_engine, Z3Variable, Z3Constraint, Z3ConstraintType
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            echo(style(f"Loading batch problems from {input_file}...", fg='blue'))
            
            # Load input file
            with open(input_file, 'r') as f:
                data = json.load(f)
            
            problems = data.get('problems', [])
            if not problems:
                echo(style("Error: No problems found in input file", fg='red'), err=True)
                sys.exit(1)
            
            echo(f"Solving {len(problems)} problems ({('parallel' if parallel else 'sequential')}, workers={max_workers})...")
            
            solver = get_z3_solver_engine()
            results = []
            completed = 0
            failed = 0
            
            import time
            start = time.time()
            
            def solve_single_problem(problem_data, idx):
                """Solve a single problem."""
                try:
                    prob_text = problem_data.get('problem', '')
                    vars_list = problem_data.get('variables', [])
                    constraints_list = problem_data.get('constraints', [])
                    timeout = problem_data.get('timeout', 60.0)
                    
                    # Check if SMT-LIB
                    is_smtlib = any(kw in prob_text for kw in ['(assert', '(declare-fun', '(check-sat)'])
                    
                    p_start = time.time()
                    
                    if is_smtlib:
                        result = solver.solve_smtlib(prob_text)
                    else:
                        z3_vars = [
                            Z3Variable(
                                v['name'], 
                                Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                                bit_width=v.get('bit_width')
                            )
                            for v in vars_list
                        ]
                        z3_constraints = [
                            Z3Constraint(c, Z3ConstraintType.INTEGER)
                            for c in constraints_list
                        ]
                        if prob_text and prob_text.strip():
                            z3_constraints.append(Z3Constraint(prob_text, Z3ConstraintType.INTEGER))
                        result = solver.solve_constraints(z3_vars, z3_constraints)
                    
                    p_elapsed = (time.time() - p_start) * 1000
                    
                    return {
                        "index": idx,
                        "success": True,
                        "status": result.status.value,
                        "satisfiable": result.is_sat(),
                        "model": result.model.assignments if result.model else None,
                        "execution_time_ms": p_elapsed
                    }
                except Exception as e:
                    return {
                        "index": idx,
                        "success": False,
                        "status": "error",
                        "error": str(e),
                        "execution_time_ms": 0
                    }
            
            if parallel:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {executor.submit(solve_single_problem, p, i): i for i, p in enumerate(problems)}
                    for future in as_completed(futures):
                        result = future.result()
                        results.append(result)
                        if result['success']:
                            completed += 1
                        else:
                            failed += 1
                        echo(f"  Completed {completed + failed}/{len(problems)}...", nl=False)
                        echo("\r", nl=False)
            else:
                for i, problem in enumerate(problems):
                    result = solve_single_problem(problem, i)
                    results.append(result)
                    if result['success']:
                        completed += 1
                    else:
                        failed += 1
                    echo(f"  Completed {completed + failed}/{len(problems)}...", nl=False)
                    echo("\r", nl=False)
            
            # Sort results by index
            results.sort(key=lambda x: x['index'])
            
            total_elapsed = (time.time() - start) * 1000
            
            echo(f"\nCompleted: {completed}, Failed: {failed}, Total time: {total_elapsed:.0f}ms")
            
            # Format output
            output_data = {
                "success": True,
                "completed": completed,
                "failed": failed,
                "total": len(problems),
                "total_time_ms": total_elapsed,
                "results": results
            }
            
            _output_result(output_data, output, output_format)
            
            if completed == len(problems):
                echo(style(f"[OK] All {completed} problems solved successfully", fg='green'))
            else:
                echo(style(f"[WARN] {completed} succeeded, {failed} failed", fg='yellow'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Portfolio Solve Command
    # =============================================================================

    @cli.command('solve-portfolio')
    @click.argument('input_file', type=click.Path(exists=True))
    @click.option('--strategies', '-s', help='Comma-separated list of strategies (default: auto)')
    @click.option('--timeout', '-t', default=30.0, help='Timeout per strategy in seconds')
    @click.option('--sequential/--parallel', default=True, help='Run strategies in parallel')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def solve_portfolio(input_file, strategies, timeout, sequential, output, output_format):
        """Solve using multiple strategies in parallel (portfolio solving).

        Available strategies: simplify, solve-eqs, bit-blast, smt, qfbv, qflia,
        qfnra, qfuf, auto-config, default

        Input file should contain SMT-LIB problem:
            (declare-fun x () Int)
            (assert (> x 0))
            (check-sat)

        Examples:
            z3 solve-portfolio problem.smt2
            z3 solve-portfolio problem.smt2 -s "smt,qfbv,qflia"
            z3 solve-portfolio problem.smt2 --sequential -t 60
        """
        try:
            from z3prover_advanced import get_z3_advanced_solver
            
            echo(style("Running portfolio solve...", fg='blue'))
            
            # Load SMT-LIB problem
            smtlib = Path(input_file).read_text()
            
            # Parse strategies
            strategy_list = None
            if strategies:
                strategy_list = [s.strip() for s in strategies.split(',')]
                echo(f"Using strategies: {strategy_list}")
            else:
                echo("Using default strategy portfolio")
            
            solver = get_z3_advanced_solver()
            
            import time
            start = time.time()
            
            result = solver.solve_portfolio(
                smtlib=smtlib,
                strategies=strategy_list,
                parallel=sequential
            )
            
            elapsed = (time.time() - start) * 1000
            
            # Format output
            output_data = {
                "success": result.success,
                "winner_strategy": result.winner_strategy,
                "execution_time_ms": elapsed,
                "parallel_speedup": result.parallel_speedup,
                "strategies_tried": len(result.all_results),
                "status": result.best_result.status.value if result.best_result else None,
                "model": result.best_result.model.assignments if result.best_result and result.best_result.model else None
            }
            
            if result.all_results:
                output_data["all_results"] = [
                    {
                        "strategy": r.strategy,
                        "status": r.status.value if hasattr(r.status, 'value') else str(r.status),
                        "execution_time_ms": r.execution_time_ms
                    }
                    for r in result.all_results
                ]
            
            _output_result(output_data, output, output_format)
            
            if result.success:
                echo(style(f"[OK] Portfolio solve complete", fg='green'))
                if result.winner_strategy:
                    echo(f"Winner strategy: {result.winner_strategy}")
                echo(f"Parallel speedup: {result.parallel_speedup:.2f}x")
                echo(f"Strategies tried: {len(result.all_results)}")
                if result.best_result and result.best_result.model:
                    echo("Solution:")
                    for var, val in result.best_result.model.assignments.items():
                        echo(f"  {var} = {val}")
            else:
                echo(style("[FAIL] Portfolio solve failed", fg='red'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Incremental Solve Command
    # =============================================================================

    @cli.command('solve-incremental')
    @click.option('--state-id', help='Incremental state ID (omit to create new state)')
    @click.option('--operation', '-op', 
                  type=click.Choice(['create', 'push', 'pop', 'add', 'check', 'reset']),
                  default='create', help='Incremental operation')
    @click.option('--variables', '-v', help='Variables JSON (for create operation)')
    @click.option('--constraints', '-c', help='Constraints JSON (for create operation)')
    @click.option('--constraint', help='Single constraint to add (for add operation)')
    @click.option('--output', '-o', help='Output file')
    @click.option('--format', 'output_format', default='json', type=click.Choice(['json', 'yaml', 'text']))
    def solve_incremental(state_id, operation, variables, constraints, constraint, output, output_format):
        """Interactive incremental constraint solving with push/pop/add/check.

        Operations:
          create: Create a new incremental solver state
          push:   Push a new scope onto the assertion stack
          pop:    Pop a scope from the assertion stack
          add:    Add a constraint to the current scope
          check:  Check satisfiability of current constraints
          reset:  Reset the solver state

        Examples:
            # Create new incremental state
            z3 solve-incremental --operation create -v '[{"name":"x","type":"INTEGER"}]'

            # Add constraint (use state-id from create output)
            z3 solve-incremental --state-id <id> --operation add --constraint "x > 0"

            # Check satisfiability
            z3 solve-incremental --state-id <id> --operation check

            # Push scope and add more constraints
            z3 solve-incremental --state-id <id> --operation push
            z3 solve-incremental --state-id <id> --operation add --constraint "x < 10"
            z3 solve-incremental --state-id <id> --operation check

            # Pop scope (removes x < 10 constraint)
            z3 solve-incremental --state-id <id> --operation pop

        For interactive mode, use state-id from the 'create' operation output.
        """
        try:
            from z3prover_advanced import get_z3_advanced_solver
            from z3prover_integration import Z3Variable, Z3Constraint, Z3ConstraintType
            
            echo(style(f"Incremental solve: {operation}...", fg='blue'))
            
            solver = get_z3_advanced_solver()
            
            import time
            start = time.time()
            
            if operation == 'create':
                # Parse variables and constraints
                vars_list = json.loads(variables) if variables else []
                constraints_list = json.loads(constraints) if constraints else []
                
                z3_vars = [
                    Z3Variable(
                        v['name'], 
                        Z3ConstraintType[v.get('type', 'INTEGER').upper()],
                        bit_width=v.get('bit_width')
                    )
                    for v in vars_list
                ]
                z3_constraints = [
                    Z3Constraint(c, Z3ConstraintType.INTEGER)
                    for c in constraints_list
                ]
                
                new_state_id = solver.create_incremental_state(z3_vars, z3_constraints, state_id)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": True,
                    "state_id": new_state_id,
                    "operation": operation,
                    "message": "Incremental state created successfully",
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                echo(style(f"[OK] Created incremental state: {new_state_id}", fg='green'))
                echo(style("\nUse this state-id for subsequent operations:", fg='yellow'))
                echo(f"  z3 solve-incremental --state-id {new_state_id} --operation push")
                echo(f"  z3 solve-incremental --state-id {new_state_id} --operation add --constraint '...'")
                echo(f"  z3 solve-incremental --state-id {new_state_id} --operation check")
            
            elif operation == 'push':
                if not state_id:
                    echo(style("Error: --state-id required for push operation", fg='red'), err=True)
                    sys.exit(1)
                
                success = solver.push_scope(state_id)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": success,
                    "state_id": state_id,
                    "operation": operation,
                    "message": "Scope pushed" if success else "Failed to push scope",
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                if success:
                    echo(style(f"[OK] Scope pushed for state: {state_id}", fg='green'))
                else:
                    echo(style(f"[FAIL] Failed to push scope", fg='red'))
            
            elif operation == 'pop':
                if not state_id:
                    echo(style("Error: --state-id required for pop operation", fg='red'), err=True)
                    sys.exit(1)
                
                success = solver.pop_scope(state_id)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": success,
                    "state_id": state_id,
                    "operation": operation,
                    "message": "Scope popped" if success else "Failed to pop scope",
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                if success:
                    echo(style(f"[OK] Scope popped for state: {state_id}", fg='green'))
                else:
                    echo(style(f"[FAIL] Failed to pop scope", fg='red'))
            
            elif operation == 'add':
                if not state_id:
                    echo(style("Error: --state-id required for add operation", fg='red'), err=True)
                    sys.exit(1)
                if not constraint:
                    echo(style("Error: --constraint required for add operation", fg='red'), err=True)
                    sys.exit(1)
                
                z3_constraint = Z3Constraint(constraint, Z3ConstraintType.INTEGER)
                success = solver.add_constraint_incremental(state_id, z3_constraint)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": success,
                    "state_id": state_id,
                    "operation": operation,
                    "constraint": constraint,
                    "message": "Constraint added" if success else "Failed to add constraint",
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                if success:
                    echo(style(f"[OK] Constraint added to state: {state_id}", fg='green'))
                else:
                    echo(style(f"[FAIL] Failed to add constraint", fg='red'))
            
            elif operation == 'check':
                if not state_id:
                    echo(style("Error: --state-id required for check operation", fg='red'), err=True)
                    sys.exit(1)
                
                result = solver.check_incremental(state_id)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": True,
                    "state_id": state_id,
                    "operation": operation,
                    "status": result.status.value,
                    "satisfiable": result.is_sat(),
                    "model": result.model.assignments if result.model else None,
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                if result.is_sat():
                    echo(style(f"[OK] SATISFIABLE", fg='green'))
                    if result.model:
                        echo("Model:")
                        for var, val in result.model.assignments.items():
                            echo(f"  {var} = {val}")
                elif result.is_unsat():
                    echo(style(f"[OK] UNSATISFIABLE", fg='yellow'))
                else:
                    echo(style(f"[FAIL] UNKNOWN", fg='red'))
            
            elif operation == 'reset':
                if not state_id:
                    echo(style("Error: --state-id required for reset operation", fg='red'), err=True)
                    sys.exit(1)
                
                success = solver.reset_incremental_state(state_id)
                
                elapsed = (time.time() - start) * 1000
                
                output_data = {
                    "success": success,
                    "state_id": state_id,
                    "operation": operation,
                    "message": "State reset" if success else "Failed to reset state",
                    "execution_time_ms": elapsed
                }
                
                _output_result(output_data, output, output_format)
                
                if success:
                    echo(style(f"[OK] State reset: {state_id}", fg='green'))
                else:
                    echo(style(f"[FAIL] Failed to reset state", fg='red'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Prove Command
    # =============================================================================

    @cli.command()
    @click.argument('theorem_file', type=click.Path(exists=True))
    @click.option('--extract-proof', is_flag=True, help='Extract detailed proof')
    @click.option('--timeout', '-t', default=300.0, help='Timeout in seconds')
    def prove(theorem_file, extract_proof, timeout):
        """Prove a theorem from file."""
        try:
            from z3prover_integration import get_z3_theorem_prover
            
            echo(style(f"Proving theorem from {theorem_file}...", fg='blue'))
            
            theorem = Path(theorem_file).read_text()
            
            prover = get_z3_theorem_prover()
            result = prover.prove_theorem(theorem)
            
            if result.proven:
                echo(style("[OK] Theorem PROVEN", fg='green'))
                echo(f"Tactic used: {result.tactic_used}")
                if result.proof and extract_proof:
                    echo("\nProof:")
                    echo(result.proof[:500] + "..." if len(result.proof) > 500 else result.proof)
            else:
                echo(style("[FAIL] Could not prove theorem", fg='red'))
                if result.counterexample:
                    echo("Counterexample found:")
                    echo(json.dumps(result.counterexample, indent=2))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Server Command
    # =============================================================================

    @cli.command()
    @click.option('--host', default='0.0.0.0', help='Host to bind to')
    @click.option('--port', '-p', default=8765, help='Port to bind to')
    @click.option('--reload', is_flag=True, help='Enable auto-reload')
    def server(host, port, reload):
        """Run the API server."""
        try:
            import uvicorn
            
            echo(style(f"Starting API server on {host}:{port}...", fg='blue'))
            echo(f"Documentation: http://{host}:{port}/docs")
            
            uvicorn.run(
                "z3_api_server:app",
                host=host,
                port=port,
                reload=reload
            )
        
        except ImportError:
            echo(style("Error: uvicorn not installed. Run: pip install uvicorn", fg='red'), err=True)
            sys.exit(1)
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)
            sys.exit(1)


    # =============================================================================
    # Monitor Command
    # =============================================================================

    @cli.command()
    @click.option('--watch', '-w', is_flag=True, help='Continuous monitoring')
    @click.option('--interval', '-i', default=5.0, help='Update interval')
    def monitor(watch, interval):
        """Show performance metrics."""
        try:
            from z3_performance_monitor import get_z3_performance_monitor
            
            monitor = get_z3_performance_monitor()
            
            if watch:
                echo(style("Monitoring (press Ctrl+C to stop)...", fg='blue'))
                try:
                    while True:
                        _print_metrics(monitor)
                        echo("\n" + "-" * 50)
                        import time
                        time.sleep(interval)
                except KeyboardInterrupt:
                    echo(style("\nMonitoring stopped.", fg='yellow'))
            else:
                _print_metrics(monitor)
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    def _print_metrics(monitor):
        """Print current metrics."""
        dashboard = monitor.get_dashboard_data()
        
        echo(style("\n=== Performance Metrics ===", fg='blue', bold=True))
        
        summary = dashboard.get('summary', {})
        echo(f"Total Operations: {summary.get('total_operations', 0)}")
        echo(f"Total Calls: {summary.get('total_calls', 0)}")
        echo(f"Success Rate: {summary.get('overall_success_rate', 'N/A')}")
        echo(f"Active Alerts: {summary.get('active_alerts', 0)}")
        
        bottlenecks = dashboard.get('top_bottlenecks', [])
        if bottlenecks:
            echo(style("\nTop Bottlenecks:", fg='yellow'))
            for b in bottlenecks[:5]:
                echo(f"  {b['operation']}: {b['avg_time_s']:.3f}s")


    # =============================================================================
    # Config Command
    # =============================================================================

    @cli.group()
    def config():
        """Manage configuration."""
        pass

    @config.command('show')
    def config_show():
        """Show current configuration."""
        try:
            from z3_config_manager import get_config_manager
            
            cfg = get_config_manager()
            echo(json.dumps(cfg.to_dict(), indent=2))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    @config.command('validate')
    def config_validate():
        """Validate configuration."""
        try:
            from z3_config_manager import get_config_manager
            
            cfg = get_config_manager()
            errors = cfg.validate()
            
            if errors:
                echo(style("Validation errors:", fg='red'))
                for error in errors:
                    echo(f"  - {error}")
                sys.exit(1)
            else:
                echo(style("[OK] Configuration is valid", fg='green'))
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)


    # =============================================================================
    # Knowledge Command
    # =============================================================================

    @cli.group()
    def knowledge():
        """Query knowledge base."""
        pass

    @knowledge.command('patterns')
    @click.option('--domain', '-d', help='Filter by domain')
    @click.option('--limit', '-l', default=10, help='Number of results')
    def knowledge_patterns(domain, limit):
        """Show learned proof patterns."""
        try:
            from z3_knowledge_extraction import get_z3_knowledge_extractor
            
            extractor = get_z3_knowledge_extractor()
            summary = extractor.get_knowledge_summary()
            
            echo(style(f"\n=== Proof Patterns ({summary['proof_patterns']['count']} total) ===", fg='blue'))
            
            patterns = summary['proof_patterns'].get('top_patterns', [])
            for p in patterns[:limit]:
                echo(f"\n  {p['name']}")
                echo(f"    Success rate: {p['success_rate']}")
                echo(f"    Usage count: {p['usage_count']}")
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)

    @knowledge.command('insights')
    @click.option('--category', '-c', help='Filter by category')
    def knowledge_insights(category):
        """Show mathematical insights."""
        try:
            from z3_knowledge_extraction import get_z3_knowledge_extractor
            
            extractor = get_z3_knowledge_extractor()
            
            insights = extractor.find_related_insights(category=category)
            
            echo(style(f"\n=== Mathematical Insights ({len(insights)} found) ===", fg='blue'))
            
            for i in insights[:10]:
                echo(f"\n  [{i.category}] {i.statement[:80]}...")
                echo(f"    Confidence: {i.confidence:.1%}")
        
        except Exception as e:
            echo(style(f"Error: {e}", fg='red'), err=True)


    # =============================================================================
    # Utility Functions
    # =============================================================================

    def _output_result(data: dict, output_file: Optional[str], output_format: str):
        """Output result to file or stdout."""
        if output_format == 'json':
            content = json.dumps(data, indent=2)
        elif output_format == 'yaml':
            try:
                import yaml
                content = yaml.dump(data, default_flow_style=False)
            except ImportError:
                content = json.dumps(data, indent=2)
        else:  # text
            content = str(data)
        
        if output_file:
            Path(output_file).write_text(content)
            echo(style(f"Output written to {output_file}", fg='green'))
        else:
            echo(content)


    # =============================================================================
    # Main Entry Point
    # =============================================================================

    def main():
        """Run the CLI."""
        cli()

else:
    # Fallback if click not available
    def main():
        print("Click is required for CLI. Install with: pip install click")
        print("\nAvailable commands would be:")
        print("  z3 solve <problem> [--use-cav-nlp]")
        print("  z3 solve-batch <input-file>")
        print("  z3 solve-portfolio <input-file>")
        print("  z3 solve-incremental --operation <op>")
        print("  z3 optimize <objective>")
        print("  z3 optimize-multi <input-file>")
        print("  z3 prove <theorem-file>")
        print("  z3 server")
        print("  z3 monitor")
        print("  z3 config show")
        print("  z3 knowledge patterns")
        print("\nCAV-NLP Commands:")
        print("  z3 formalize <text> [--elaborate]")
        print("  z3 verify <constraint> [--hybrid]")
        print("  z3 canonicalize <constraint>")
        print("\nWeb3 Formal Commands:")
        print("  z3 web3-translate-invariant <statement> [--verify]")
        print("  z3 web3-solve-witness [--constraints '[]'] [--timeout 10]")
        print("  z3 web3-audit-exploit-verification [statement] [--verify] [--constraints '[]']")


if __name__ == "__main__":
    main()
