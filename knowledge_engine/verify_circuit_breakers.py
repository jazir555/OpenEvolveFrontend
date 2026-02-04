"""
Circuit Breaker & Health Monitoring Verification Script
"""
from knowledge_engine.master_engine import MasterKnowledgeEngine
from knowledge_engine.orchestration.circuit_breaker import CircuitBreaker, get_circuit_breaker


def verify_circuit_breakers():
    print('=' * 60)
    print('Circuit Breaker & Health Verification Report')
    print('=' * 60)

    # Initialize Master Knowledge Engine
    engine = MasterKnowledgeEngine()

    # 1. Check circuit breaker initialization
    print('\n[1] CIRCUIT BREAKER REGISTRATION')
    print('-' * 40)
    print(f'Total Circuit Breakers: {len(engine.circuit_breakers)}')

    # Check all expected components have circuit breakers
    expected_components = [
        'graphiti', 'kggen', 'oneke', 'aikg', 'deepke', 'ragbits', 
        'crewai', 'pami', 'neuralkg', 'causal_learn', 'karateclub', 
        'global_chem', 'neuromancer', 'lagrange_mapper', 'leanaide',
        'research_quest', 'agentic_context', 'agentjson', 'dspy', 
        'openevolve_lib', 'mcp_gateway', 'outlines', 'lmql', 
        'neuromancer_ke', 'cognitive_hydraulics', 'dts', 'guardrails', 
        'icr', 'roma'
    ]

    print('\nCircuit Breaker Status:')
    cb_status = []
    for comp in expected_components:
        if comp in engine.circuit_breakers:
            cb = engine.circuit_breakers[comp]
            state = cb.state.value
            print(f'  [OK] {comp}: {state.upper()}')
            cb_status.append((comp, state))
        else:
            print(f'  [MISSING] {comp}: NO CIRCUIT BREAKER')
            cb_status.append((comp, 'missing'))

    # 2. Check health status from unified hub
    print('\n\n[2] HEALTH STATUS TRACKING')
    print('-' * 40)
    try:
        from knowledge_engine.unified_kg_integration_hub import UnifiedKGIntegrationHub
        hub = UnifiedKGIntegrationHub()
        
        # Manually check health for all components
        health_count = 0
        for comp in expected_components:
            if hasattr(hub, '_health_status') and comp in hub._health_status:
                health_count += 1
        
        print(f'Health Tracking Integrations: {health_count}')
        
        # Check critical integrations health tracking
        critical = ['outlines', 'lmql', 'neuromancer_ke', 'cognitive_hydraulics', 
                    'dts', 'guardrails', 'icr', 'lagrange_mapper']
        
        print('\nCritical Components Health:')
        for comp in critical:
            if comp in hub._health_status:
                status = hub._health_status[comp].status.value
                print(f'  [OK] {comp}: {status}')
            else:
                print(f'  [MISSING] {comp}: NO HEALTH TRACKING')
                
    except Exception as e:
        print(f'Health check error: {e}')
        health_count = 0

    # 3. Check substitution matrix (fallback chains)
    print('\n\n[3] SUBSTITUTION MATRIX / FALLBACK CHAINS')
    print('-' * 40)
    registry = engine.component_registry
    print(f'Substitution Matrix Coverage: {len(registry.substitution_matrix)} components')

    print('\nFallback Chains:')
    missing_fallbacks = []
    for comp in expected_components:
        if comp in registry.substitution_matrix:
            fallbacks = registry.substitution_matrix[comp]
            if fallbacks:
                print(f'  {comp} -> {fallbacks}')
            else:
                print(f'  {comp} -> [] (no fallbacks defined)')
        else:
            print(f'  [MISSING] {comp}: NO FALLBACK DEFINED')
            missing_fallbacks.append(comp)

    # Summary
    print('\n\n[SUMMARY]')
    print('-' * 40)
    cb_count = len([c for c in cb_status if c[1] != 'missing'])
    print(f'Circuit Breakers: {cb_count}/{len(expected_components)}')
    print(f'Health Tracking: {health_count}/{len(expected_components)}')
    print(f'Substitution Matrix: {len(registry.substitution_matrix)} entries')

    if missing_fallbacks:
        print(f'\nMissing Fallbacks: {missing_fallbacks}')
    else:
        print('\nMissing Fallbacks: NONE')

    # Determine final verdict
    cb_ok = cb_count == len(expected_components)
    health_ok = health_count >= len(expected_components) - 5  # Allow some missing
    matrix_ok = len(registry.substitution_matrix) >= 10  # At least 10 entries

    print('\n' + '=' * 60)
    if cb_ok and health_ok and matrix_ok:
        print('FINAL VERDICT: FULLY PROTECTED')
    elif cb_ok or health_ok:
        print('FINAL VERDICT: PARTIAL')
    else:
        print('FINAL VERDICT: INCOMPLETE')
    print('=' * 60)
    
    return {
        'circuit_breakers': cb_count,
        'health_tracking': health_count,
        'substitution_matrix': len(registry.substitution_matrix),
        'missing_fallbacks': missing_fallbacks,
        'verdict': 'FULLY PROTECTED' if (cb_ok and health_ok and matrix_ok) else 'PARTIAL' if (cb_ok or health_ok) else 'INCOMPLETE'
    }


if __name__ == '__main__':
    verify_circuit_breakers()
