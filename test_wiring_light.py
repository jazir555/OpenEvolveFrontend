#!/usr/bin/env python3
"""Lightweight test for Adaptive MDAP wiring (avoids heavy imports)."""

import sys
import ast

print('Python:', sys.version)
print()

results = []

# 1. Test adaptive_mdap package
try:
    from adaptive_mdap import TaskComplexityClassifier, AdaptiveMDAPAllocator
    results.append('[PASS] adaptive_mdap package imports OK')
except Exception as e:
    results.append(f'[FAIL] adaptive_mdap: {e}')

# 2. Test config_loader integration
try:
    from config_loader import AdaptiveMDAPConfig
    cfg = AdaptiveMDAPConfig()
    results.append(f'[PASS] config_loader AdaptiveMDAPConfig enabled={cfg.enabled}')
except Exception as e:
    results.append(f'[FAIL] config_loader: {e}')

# 3. Check workflow_engine.py has adaptive imports
try:
    with open('workflow_engine.py', 'r') as f:
        content = f.read()
    if 'ADAPTIVE_MDAP_AVAILABLE' in content and 'get_adaptive_workflow' in content:
        results.append('[PASS] workflow_engine.py has adaptive imports')
    else:
        results.append('[FAIL] workflow_engine.py missing adaptive imports')
except Exception as e:
    results.append(f'[FAIL] workflow_engine.py check: {e}')

# 4. Check evolution.py has adaptive imports
try:
    with open('evolution.py', 'r') as f:
        content = f.read()
    if 'ADAPTIVE_MDAP_AVAILABLE' in content and 'enable_adaptive_mdap' in content:
        results.append('[PASS] evolution.py has adaptive parameters')
    else:
        results.append('[FAIL] evolution.py missing adaptive parameters')
except Exception as e:
    results.append(f'[FAIL] evolution.py check: {e}')

# 5. Check openevolve_orchestrator.py has adaptive imports
try:
    with open('openevolve_orchestrator.py', 'r') as f:
        content = f.read()
    if 'ADAPTIVE_MDAP_AVAILABLE' in content and 'adaptive_mdap_config' in content:
        results.append('[PASS] openevolve_orchestrator.py has adaptive config')
    else:
        results.append('[FAIL] openevolve_orchestrator.py missing adaptive config')
except Exception as e:
    results.append(f'[FAIL] openevolve_orchestrator.py check: {e}')

# 6. Check sidebar.py has adaptive UI
try:
    with open('sidebar.py', 'r') as f:
        content = f.read()
    if 'enable_adaptive_mdap' in content and 'adaptive_profile' in content:
        results.append('[PASS] sidebar.py has adaptive UI controls')
    else:
        results.append('[FAIL] sidebar.py missing adaptive UI controls')
except Exception as e:
    results.append(f'[FAIL] sidebar.py check: {e}')

# 7. Check api_server.py has adaptive endpoints
try:
    with open('api_server.py', 'r') as f:
        content = f.read()
    if '/adaptive-mdap/' in content:
        results.append('[PASS] api_server.py has adaptive endpoints')
    else:
        results.append('[FAIL] api_server.py missing adaptive endpoints')
except Exception as e:
    results.append(f'[FAIL] api_server.py check: {e}')

# 8. Check app.py has adaptive demo
try:
    with open('app.py', 'r') as f:
        content = f.read()
    if 'TaskComplexityClassifier' in content and 'AdaptiveMDAPAllocator' in content:
        results.append('[PASS] app.py has adaptive demo')
    else:
        results.append('[FAIL] app.py missing adaptive demo')
except Exception as e:
    results.append(f'[FAIL] app.py check: {e}')

# Print results
print('\n'.join(results))
passed = len([r for r in results if r.startswith('[PASS]')])
print(f'\nTotal: {passed}/{len(results)} checks passed')

# Return 0 if all passed
sys.exit(0 if passed == len(results) else 1)
