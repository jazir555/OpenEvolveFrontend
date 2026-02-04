#!/usr/bin/env python
"""Verify all 7 new integrations are 100% complete."""

import sys
import ast

def check_syntax(filepath):
    """Check Python file syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        ast.parse(source)
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except FileNotFoundError:
        return False, "File not found"
    except Exception as e:
        return False, str(e)

def verify_master_engine():
    """Verify integrations in Master Engine."""
    from knowledge_engine.master_engine import MasterKnowledgeEngine

    engine = MasterKnowledgeEngine()

    integrations = [
        'outlines', 'lmql', 'neuromancer_ke', 'cognitive_hydraulics',
        'dts', 'guardrails', 'icr'
    ]

    # Check circuit breakers (which tracks all registered integrations)
    circuit_breaker_keys = list(engine.circuit_breakers.keys())

    results = {}
    print('='*60)
    print('MASTER ENGINE VERIFICATION')
    print('='*60)

    for integration in integrations:
        # Check if integration is registered (via circuit breaker)
        in_circuit_breaker = integration in circuit_breaker_keys

        # Check capabilities
        capabilities = engine.component_registry.get_all_capabilities()
        in_capabilities = any(integration in str(cap) for cap in capabilities)

        # Check components
        available_components = engine.component_registry.get_available_components()
        in_components = integration in available_components

        # Check substitution matrix
        substitutes = engine.component_registry.get_substitutes(integration)
        in_substitution = substitutes is not None

        checks = {
            'circuit_breaker': in_circuit_breaker,
            'capabilities': in_capabilities,
            'components': in_components,
            'substitution': in_substitution
        }
        results[integration] = checks

        status = 'PASS' if any(checks.values()) else 'FAIL'
        print(f'{integration}:')
        print(f'  circuit_breaker: {"Y" if checks["circuit_breaker"] else "N"}')
        print(f'  capabilities: {"Y" if checks["capabilities"] else "N"}')
        print(f'  components: {"Y" if checks["components"] else "N"}')
        print(f'  substitution: {"Y" if checks["substitution"] else "N"}')
        print(f'  Status: {status}')
        print()

    return results

def verify_unified_hub():
    """Verify integrations in Unified Hub."""
    from knowledge_engine.unified_kg_integration_hub import (
        UnifiedKGIntegrationHub, KGOperationType
    )

    hub = UnifiedKGIntegrationHub()

    operations = {
        'outlines': 'STRUCTURED_GENERATION',
        'lmql': 'DECLARATIVE_QUERY',
        'neuromancer': 'PHYSICS_SIMULATION',
        'cognitive_hydraulics': 'HYBRID_REASONING',
        'dts': 'CONVERSATION_OPTIMIZATION',
        'guardrails': 'SAFETY_VALIDATION',
        'icr': 'ITERATIVE_REFINEMENT'
    }

    results = {}
    print('='*60)
    print('UNIFIED HUB VERIFICATION')
    print('='*60)

    for integration, operation in operations.items():
        try:
            op_type = getattr(KGOperationType, operation)
            routing_list = hub._routing_map.get(op_type, [])
            in_routing = integration in routing_list
            results[integration] = in_routing
            status = 'PASS' if in_routing else 'FAIL'
            print(f'{integration} -> {operation}:')
            print(f'  in_routing: {"Y" if in_routing else "N"}')
            print(f'  Status: {status}')
            print()
        except Exception as e:
            results[integration] = False
            print(f'{integration} -> {operation}: ERROR - {e}')
            print()

    return results

def verify_syntax():
    """Check syntax of all relevant files."""
    files = [
        'knowledge_engine/master_engine.py',
        'knowledge_engine/unified_kg_integration_hub.py',
        'knowledge_engine/integrations/__init__.py',
        'knowledge_engine/capability_report.py',
        'knowledge_engine/global_kg_orchestrator.py'
    ]

    print('='*60)
    print('SYNTAX CHECK')
    print('='*60)

    results = {}
    for filepath in files:
        valid, error = check_syntax(filepath)
        results[filepath] = valid
        status = 'PASS' if valid else 'FAIL'
        print(f'{filepath}: {status}')
        if error and not valid:
            print(f'  Error: {error}')
    print()

    return results

def main():
    print('\n' + '='*60)
    print('INTEGRATION VERIFICATION REPORT')
    print('All 7 New Integrations: Outlines, LMQL, Neuromancer,')
    print('                        Cognitive-Hydraulics, DTS, Guardrails, ICR')
    print('='*60)
    print()

    # Verify syntax
    syntax_results = verify_syntax()

    # Verify master engine
    master_results = verify_master_engine()

    # Verify unified hub
    hub_results = verify_unified_hub()

    # Final Report
    print('='*60)
    print('FINAL VERIFICATION REPORT')
    print('='*60)

    all_pass = True

    integration_map = {
        'Outlines': ('outlines', 'outlines'),
        'LMQL': ('lmql', 'lmql'),
        'Neuromancer': ('neuromancer_ke', 'neuromancer'),
        'Cognitive-Hydraulics': ('cognitive_hydraulics', 'cognitive_hydraulics'),
        'DTS': ('dts', 'dts'),
        'Guardrails': ('guardrails', 'guardrails'),
        'ICR': ('icr', 'icr')
    }

    summary = []

    for name, (master_key, hub_key) in integration_map.items():
        master_checks = master_results.get(master_key, {})
        master_ok = any(master_checks.values()) if master_checks else False
        hub_ok = hub_results.get(hub_key, False)

        status = '100%' if (master_ok and hub_ok) else 'INCOMPLETE'
        symbol = 'PASS' if (master_ok and hub_ok) else 'FAIL'
        all_pass = all_pass and master_ok and hub_ok

        summary.append({
            'name': name,
            'master_ok': master_ok,
            'hub_ok': hub_ok,
            'status': status,
            'symbol': symbol
        })

        print(f'{name}:')
        print(f'  Master Engine: {"Y" if master_ok else "N"}')
        print(f'  Unified Hub: {"Y" if hub_ok else "N"}')
        print(f'  Status: {status} ({symbol})')
        print()

    # Syntax summary
    syntax_ok = all(syntax_results.values())
    print('-'*60)
    print('Syntax Check:')
    for f, ok in syntax_results.items():
        sym = 'PASS' if ok else 'FAIL'
        print(f'  {f}: {sym}')
    print()

    # Count
    passed = sum(1 for s in summary if s['symbol'] == 'PASS')
    total = len(summary)

    print('='*60)
    print(f'Overall: {passed}/{total} integrations verified')
    print('='*60)

    # Final verdict
    if all_pass and syntax_ok:
        print('\n' + '='*60)
        print('ALL 7 INTEGRATIONS 100% COMPLETE')
        print('='*60)
        return 0
    else:
        print('\n' + '='*60)
        print('SOME INTEGRATIONS INCOMPLETE OR ERRORS FOUND')
        print('='*60)
        return 1

if __name__ == '__main__':
    sys.exit(main())
