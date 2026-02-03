#!/usr/bin/env python3
"""Test ROMA with proper path setup."""

import sys
sys.path.insert(0, 'core-projects/ROMA/src')

print('Testing ROMA availability with path setup...')
print()

try:
    # Direct ROMA import
    from roma_dspy import RecursiveSolver
    print('[OK] Direct ROMA import works')
except Exception as e:
    print(f'[FAIL] Direct import failed: {e}')

try:
    # Integration import
    from knowledge_engine.integrations import ROMAIntegration, ROMA_INTEGRATION_AVAILABLE
    print(f'[OK] ROMA_INTEGRATION_AVAILABLE: {ROMA_INTEGRATION_AVAILABLE}')

    if ROMA_INTEGRATION_AVAILABLE:
        roma = ROMAIntegration()
        print('[OK] ROMA initialized successfully')
    else:
        print('[FAIL] ROMA still not available through integration')
except Exception as e:
    print(f'[FAIL] Integration import failed: {e}')
    import traceback
    traceback.print_exc()
