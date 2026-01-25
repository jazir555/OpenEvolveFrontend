"""
LeanAide MDAP Integration Demo

This file demonstrates the usage of the Lean MDAP integration for Lean 4 proof generation.
Shows multi-agent, voting-based proof generation with various strategies.
"""

import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import Lean MDAP components
try:
    from leanaide_mdap import (
        ProofStrategy,
        LeanDomain,
        VotingStrategy,
        LeanMDAPConfig,
        LeanMDAPStep,
        LeanMDAPTask,
        LeanProof,
        LeanProofAgent,
        LeanAgentSelector,
        LeanMDAPOrchestrator,
        LeanMDAPResult,
        CheckpointManager,
        create_lean_mdap_config,
        get_lean_mdap_status
    )
    LEAN_MDAP_AVAILABLE = True
except ImportError as e:
    logger.error(f'Lean MDAP not available: {e}')
    LEAN_MDAP_AVAILABLE = False


def main():
    """Run demonstration"""
    print('
' + '=' * 80)
    print('LeanAide MDAP Integration Demonstration')
    print('=' * 80)
    
    if not LEAN_MDAP_AVAILABLE:
        print('
ERROR: Lean MDAP not available. Please check imports.')
        return
    
    # System status
    print('
' + '-' * 80)
    print('1. System Status')
    print('-' * 80)
    status = get_lean_mdap_status()
    print(f'MDAP Available: {status["mdap_available"]}')
    print(f'Available Strategies: {status["available_strategies"]}')
    print(f'Available Domains: {status["available_domains"]}')
    
    # Create configuration
    print('
' + '-' * 80)
    print('2. Configuration')
    print('-' * 80)
    config = create_lean_mdap_config(
        available_agents=['evolution', 'mcts', 'direct'],
        default_parallel_agents=3,
        voting_strategy='first_k_ahead',
        k_ahead_threshold=3
    )
    print(f'Parallel agents: {config.default_parallel_agents}')
    print(f'Voting strategy: {config.voting_strategy.value}')
    print(f'K-ahead threshold: {config.k_ahead_threshold}')
    
    # Create task
    print('
' + '-' * 80)
    print('3. Task Creation')
    print('-' * 80)
    task = LeanMDAPTask(
        task_id='demo_task',
        description='Prove addition commutativity',
        theorem_statement='theorem add_comm (a b : Nat) : a + b = b + a',
        domain=LeanDomain.ALGEBRA
    )
    strategies = [ProofStrategy.EVOLUTION, ProofStrategy.MCTS, ProofStrategy.DIRECT]
    task.create_default_steps(strategies, parallel=True)
    print(f'Task: {task.task_id}')
    print(f'Theorem: {task.theorem_statement}')
    print(f'Domain: {task.domain.value}')
    print(f'Steps: {len(task.get_execution_plan())}')
    
    # Initialize orchestrator
    print('
' + '-' * 80)
    print('4. Orchestrator Initialization')
    print('-' * 80)
    orchestrator = LeanMDAPOrchestrator(config=config)
    print(f'Agents registered: {len(orchestrator.agent_selector.agents)}')
    for agent_id in orchestrator.agent_selector.agents:
        print(f'  - {agent_id}')
    
    print('
' + '=' * 80)
    print('Demonstration Complete!')
    print('=' * 80)
    print('
To execute proof generation:')
    print('  1. Set OPENAI_API_KEY environment variable')
    print('  2. Run: result = orchestrator.orchestrate_proof_generation(task)')
    print('  3. Access: result.best_proof, result.voting_statistics')


if __name__ == '__main__':
    main()
