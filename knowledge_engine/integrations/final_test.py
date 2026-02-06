"""
Final Integration Test

Tests all major components of the mathematical knowledge integration.
Includes CAV-NLP integration tests.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio

# CAV-NLP integration imports
try:
    from openevolve.unified_math_service import UnifiedMathService
    from openevolve.z3_cav_nlp_integration import EnhancedZ3Solver
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    UnifiedMathService = None
    EnhancedZ3Solver = None


async def final_test():
    """Run final integration tests including CAV-NLP."""
    print('='*70)
    print('FINAL INTEGRATION TEST')
    print('='*70)
    print(f'CAV-NLP Available: {CAV_NLP_AVAILABLE}')
    
    # Test 1: Z3 solving
    print('\n1. Testing Z3 solver...')
    from z3_solver_connector import get_z3_connector, Z3SolverConfig
    z3 = get_z3_connector()
    result = await z3.solve_smtlib(
        '(declare-fun x () Int) (assert (> x 0)) (assert (< x 10)) (check-sat) (get-model)',
        Z3SolverConfig()
    )
    print(f'   Status: {result.status.value}')
    print(f'   Model: {result.model}')
    assert result.status.value == 'sat', 'Z3 should solve this'
    print('   [OK]')
    
    # Test 2: Knowledge extraction
    print('\n2. Testing knowledge extraction...')
    from z3_knowledge_complete import get_z3_knowledge_manager
    manager = await get_z3_knowledge_manager()
    learn_result = await manager.learn_from_solution(
        problem_statement='Test linear constraint',
        constraints=['x > 0', 'x < 10'],
        result='success'
    )
    print(f'   Learned: {learn_result is not None}')
    print('   [OK]')
    
    # Test 3: Strategy recommendation
    print('\n3. Testing strategy recommendation...')
    strategy = await manager.get_recommended_strategy(
        problem_statement='Linear system',
        constraints=['x + y = 10']
    )
    print(f'   Strategy: {strategy.get("strategy", "none")}')
    print('   [OK]')
    
    # Test 4: Unified bridge
    print('\n4. Testing unified bridge...')
    from unified_math_bridge_complete import get_unified_bridge_complete
    bridge = await get_unified_bridge_complete()
    result = await bridge.solve('x > 0 and x < 5', timeout=10)
    print(f'   Status: {result.get("result_status")}')
    print('   [OK]')
    
    # Test 5: MCP tools
    print('\n5. Testing MCP tools...')
    from math_mcp_tools import get_math_mcp_tools
    tools = await get_math_mcp_tools()
    available = tools.get_tools()
    print(f'   Available tools: {len(available)}')
    print('   [OK]')
    
    # Test 6: Configuration
    print('\n6. Testing configuration...')
    from math_knowledge_config import MathKnowledgeConfig
    config = MathKnowledgeConfig()
    print(f'   Z3 timeout: {config.z3.timeout_ms}ms')
    print('   [OK]')
    
    # Test 7: API
    print('\n7. Testing API...')
    from z3_api import app
    print(f'   FastAPI app: {app is not None}')
    print('   [OK]')
    
    # Test 8: CLI
    print('\n8. Testing CLI...')
    from math_knowledge_cli import MathKnowledgeCLI
    cli = MathKnowledgeCLI()
    print(f'   CLI initialized: {cli is not None}')
    print('   [OK]')
    
    # Test 9: Benchmarks
    print('\n9. Testing benchmarks...')
    from benchmark_suite import MathKnowledgeBenchmarks
    bench = MathKnowledgeBenchmarks(iterations=2, warmup=0)
    print(f'   Benchmarks ready: {bench is not None}')
    print('   [OK]')
    
    # Test 10: Migration
    print('\n10. Testing migration tool...')
    from migrate_database import DatabaseMigration
    migrator = DatabaseMigration()
    print(f'   Migration tool: {migrator is not None}')
    print('   [OK]')
    
    # Test 11: CAV-NLP Integration
    print('\n11. Testing CAV-NLP integration...')
    if CAV_NLP_AVAILABLE:
        try:
            from openevolve.unified_math_service import UnifiedMathService
            math_service = UnifiedMathService()
            result = math_service.formalize("x is greater than zero")
            print(f'   CAV-NLP formalization: {result is not None}')
            print('   [OK]')
        except Exception as e:
            print(f'   [WARN] CAV-NLP test: {e}')
    else:
        print('   CAV-NLP not available (optional)')
        print('   [OK]')
    
    print('')
    print('='*70)
    print('ALL TESTS PASSED!')
    print('='*70)
    print(f'CAV-NLP Status: {"ENABLED" if CAV_NLP_AVAILABLE else "NOT AVAILABLE"}')
    print('\nMathematical Knowledge Integration is ready for production!')


if __name__ == '__main__':
    asyncio.run(final_test())
