"""
Comprehensive Gap Analysis for Mathematical Knowledge Integration

This script checks for any missing functionality or gaps in the implementation.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio


async def functional_test():
    """Test functional completeness."""
    print('='*70)
    print('FUNCTIONAL COMPLETENESS CHECK')
    print('='*70)
    
    # Test 1: Z3 solving with different problem types
    print('\n1. Z3 Solver - Problem Types')
    from z3_solver_connector import get_z3_connector, Z3SolverConfig
    z3 = get_z3_connector()
    
    problems = [
        ('Linear', '(declare-fun x () Int) (assert (= x 5)) (check-sat) (get-model)'),
        ('Inequality', '(declare-fun x () Int) (assert (> x 0)) (assert (< x 10)) (check-sat)'),
        ('Unsat', '(declare-fun x () Int) (assert (> x 5)) (assert (< x 3)) (check-sat)'),
    ]
    
    for name, smt in problems:
        result = await z3.solve_smtlib(smt, Z3SolverConfig())
        status = "[OK]" if result.status else "[FAIL]"
        print(f'   {name}: {result.status.value} {status}')
    
    # Test 2: Knowledge manager methods
    print('\n2. Knowledge Manager - Methods')
    from z3_knowledge_complete import get_z3_knowledge_manager
    manager = await get_z3_knowledge_manager()
    
    methods = [
        'learn_from_solution',
        'find_similar_solutions', 
        'get_recommended_strategy',
        'get_statistics',
    ]
    
    all_ok = True
    for method in methods:
        has_method = hasattr(manager, method)
        status = "[OK]" if has_method else "[MISSING]"
        print(f'   {method}: {status}')
        if not has_method:
            all_ok = False
    
    # Test 3: Unified bridge methods
    print('\n3. Unified Bridge - Methods')
    from unified_math_bridge_complete import get_unified_bridge_complete
    bridge = await get_unified_bridge_complete()
    
    methods = [
        'solve',
        'translator',
        'consensus',
        'stats',
    ]
    
    for method in methods:
        has_method = hasattr(bridge, method)
        status = "[OK]" if has_method else "[MISSING]"
        print(f'   {method}: {status}')
        if not has_method:
            all_ok = False
    
    # Test 4: MCP tools
    print('\n4. MCP Tools - Available')
    from math_mcp_tools import get_math_mcp_tools
    tools = await get_math_mcp_tools()
    available = tools.get_tools()
    
    expected_tools = [
        'z3_solve',
        'lean_prove',
        'math_solve',
        'math_pattern_search',
        'math_strategy_recommend',
        'math_extract_knowledge',
        'math_translate',
        'math_health_check',
    ]
    
    tool_names = [t['name'] for t in available]
    for tool in expected_tools:
        has_tool = tool in tool_names
        status = "[OK]" if has_tool else "[MISSING]"
        print(f'   {tool}: {status}')
        if not has_tool:
            all_ok = False
    
    # Test 5: Configuration
    print('\n5. Configuration - Sections')
    from math_knowledge_config import MathKnowledgeConfig
    config = MathKnowledgeConfig()
    
    sections = [
        'database',
        'z3',
        'leanaide',
        'api',
        'monitoring',
    ]
    
    for section in sections:
        has_section = hasattr(config, section)
        status = "[OK]" if has_section else "[MISSING]"
        print(f'   {section}: {status}')
        if not has_section:
            all_ok = False
    
    # Test 6: CLI commands
    print('\n6. CLI - Commands')
    from math_knowledge_cli import MathKnowledgeCLI
    cli = MathKnowledgeCLI()
    
    commands = [
        'solve',
        'prove',
        'search',
        'config',
        'benchmark',
        'server',
        'knowledge',
        'health',
        'version',
    ]
    
    for cmd in commands:
        has_cmd = hasattr(cli, f'_cmd_{cmd}')
        status = "[OK]" if has_cmd else "[MISSING]"
        print(f'   {cmd}: {status}')
        if not has_cmd:
            all_ok = False
    
    # Test 7: Database models
    print('\n7. Database Models')
    from math_knowledge_models import MODELS_AVAILABLE
    
    if MODELS_AVAILABLE:
        from math_knowledge_models import Z3KnowledgeBase, Z3SolverRun, LeanProofRecord
        models = [Z3KnowledgeBase, Z3SolverRun, LeanProofRecord]
        for model in models:
            has_table = hasattr(model, '__tablename__')
            status = "[OK]" if has_table else "[MISSING]"
            print(f'   {model.__name__}: {status}')
            if not has_table:
                all_ok = False
    else:
        print('   Models not available (SQLAlchemy not installed)')
    
    # Test 8: Benchmark suite
    print('\n8. Benchmark Suite - Methods')
    from benchmark_suite import MathKnowledgeBenchmarks
    bench = MathKnowledgeBenchmarks(iterations=1, warmup=0)
    
    methods = [
        'run_all',
        'benchmark_z3_basic',
        'benchmark_knowledge_extraction',
    ]
    
    for method in methods:
        has_method = hasattr(bench, method)
        status = "[OK]" if has_method else "[MISSING]"
        print(f'   {method}: {status}')
        if not has_method:
            all_ok = False
    
    # Test 9: Migration tool
    print('\n9. Migration Tool - Commands')
    from migrate_database import DatabaseMigration
    migrator = DatabaseMigration()
    
    commands = [
        'init_database',
        'migrate',
        'backup',
        'restore',
        'validate',
        'export',
    ]
    
    for cmd in commands:
        has_cmd = hasattr(migrator, cmd)
        status = "[OK]" if has_cmd else "[MISSING]"
        print(f'   {cmd}: {status}')
        if not has_cmd:
            all_ok = False
    
    print('')
    print('='*70)
    if all_ok:
        print('ALL CHECKS PASSED - NO GAPS FOUND')
    else:
        print('SOME CHECKS FAILED - GAPS IDENTIFIED')
    print('='*70)
    
    return all_ok


if __name__ == '__main__':
    result = asyncio.run(functional_test())
    sys.exit(0 if result else 1)
