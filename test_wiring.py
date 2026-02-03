#!/usr/bin/env python3
"""Test Adaptive MDAP wiring throughout the codebase."""

import sys
print('Python:', sys.version)
print()

# Test key integrations
results = []

# 1. Test adaptive_mdap package
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    results.append('[PASS] adaptive_mdap package imports OK')
except Exception as e:
    results.append(f'[FAIL] adaptive_mdap: {e}')

# 2. Test workflow_engine integration
try:
    from workflow_engine import ADAPTIVE_MDAP_AVAILABLE, get_adaptive_mdap_status
    results.append(f'[PASS] workflow_engine ADAPTIVE_MDAP_AVAILABLE={ADAPTIVE_MDAP_AVAILABLE}')
except Exception as e:
    results.append(f'[FAIL] workflow_engine: {e}')

# 3. Test evolution integration  
try:
    from evolution import ADAPTIVE_MDAP_AVAILABLE
    results.append(f'[PASS] evolution ADAPTIVE_MDAP_AVAILABLE={ADAPTIVE_MDAP_AVAILABLE}')
except Exception as e:
    results.append(f'[FAIL] evolution: {e}')

# 4. Test openevolve_orchestrator integration
try:
    from openevolve_orchestrator import ADAPTIVE_MDAP_AVAILABLE
    results.append(f'[PASS] openevolve_orchestrator ADAPTIVE_MDAP_AVAILABLE={ADAPTIVE_MDAP_AVAILABLE}')
except Exception as e:
    results.append(f'[FAIL] openevolve_orchestrator: {e}')

# 5. Test config_loader integration
try:
    from config_loader import AdaptiveMDAPConfig
    cfg = AdaptiveMDAPConfig()
    results.append(f'[PASS] config_loader AdaptiveMDAPConfig enabled={cfg.enabled}')
except Exception as e:
    results.append(f'[FAIL] config_loader: {e}')

# 6. Test api_server integration (just check file parses)
try:
    import ast
    with open('api_server.py', 'r') as f:
        ast.parse(f.read())
    results.append('[PASS] api_server.py parses OK')
except Exception as e:
    results.append(f'[FAIL] api_server.py: {e}')

# 7. Test sidebar integration (just check file parses)
try:
    import ast
    with open('sidebar.py', 'r') as f:
        ast.parse(f.read())
    results.append('[PASS] sidebar.py parses OK')
except Exception as e:
    results.append(f'[FAIL] sidebar.py: {e}')

# 8. Test app.py integration
try:
    import ast
    with open('app.py', 'r') as f:
        ast.parse(f.read())
    results.append('[PASS] app.py parses OK')
except Exception as e:
    results.append(f'[FAIL] app.py: {e}')

# Print results
print('\n'.join(results))
passed = len([r for r in results if r.startswith('[PASS]')])
print(f'\nTotal: {passed}/{len(results)} checks passed')

# Return 0 if all passed
sys.exit(0 if passed == len(results) else 1)
