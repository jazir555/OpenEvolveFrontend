"""
Quick integration test for LoongFlow gauntlet adapter.
"""
import asyncio
import sys
sys.path.insert(0, '.')

from enhanced_gauntlet_manager import create_enhanced_gauntlet_system

async def test_gauntlet_system():
    print('Testing Enhanced Gauntlet System...')

    # Create system
    llm_config = {
        'model': 'claude-3-5-sonnet-20241022',
        'api_key': 'test-key',
        'url': 'http://localhost:8001'
    }

    system = create_enhanced_gauntlet_system(
        llm_config=llm_config,
        enable_loongflow=False
    )

    print('[OK] System created')

    # Test gauntlet creation
    gauntlet = system.create_enhanced_gauntlet(
        problem_type='engineering',
        strictness='standard'
    )

    print(f'[OK] Gauntlet created: {gauntlet.name}')
    print(f'  - Rounds: {len(gauntlet.rounds)}')

    # Create mock solution
    class MockSolution:
        def __init__(self):
            self.id = 'test_solution'
            self.content = "Engineering solution with proper approach and validation"

    # Execute gauntlet
    execution = await system.execute_gauntlet(
        gauntlet=gauntlet,
        solution=MockSolution(),
        context={'problem': 'Design a component'}
    )

    print(f'[OK] Gauntlet executed')
    print(f'  - Overall Passed: {execution.overall_passed}')
    print(f'  - Final Score: {execution.final_score:.3f}')
    print(f'  - Rounds Passed: {len(execution.rounds_passed)}/{len(execution.rounds_results)}')
    print(f'  - Execution Time: {execution.execution_time:.2f}s')

    for i, round_result in enumerate(execution.rounds_results, 1):
        print(f'  - Round {i}: {round_result.rule_id} ({round_result.status.value}) - Score: {round_result.score:.3f}')

    print('')
    print('Gauntlet system test PASSED!')

if __name__ == '__main__':
    asyncio.run(test_gauntlet_system())
