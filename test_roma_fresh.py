#!/usr/bin/env python3
"""Test ROMA with fresh imports."""

import sys

# Clear any cached modules
if 'knowledge_engine.integrations.roma_integration' in sys.modules:
    del sys.modules['knowledge_engine.integrations.roma_integration']
if 'knowledge_engine.integrations' in sys.modules:
    del sys.modules['knowledge_engine.integrations']

# Add ROMA to path
sys.path.insert(0, 'core-projects/ROMA/src')

print('Testing ROMA with fresh imports...')
print()

try:
    from knowledge_engine.integrations import ROMAIntegration, ROMA_INTEGRATION_AVAILABLE
    print(f'[INFO] ROMA_INTEGRATION_AVAILABLE: {ROMA_INTEGRATION_AVAILABLE}')

    roma = ROMAIntegration()
    print(f'[INFO] ROMA initialized')
    print(f'[INFO] _roma_available: {roma._roma_available}')
    print(f'[INFO] decomposer: {roma.decomposer}')

    if roma._roma_available:
        print()
        print('[SUCCESS] ROMA IS NOW FULLY FUNCTIONAL!')
        print('[SUCCESS] ROMA core components are loaded and ready!')
    else:
        print()
        print('[FAIL] ROMA still in mock mode')

except Exception as e:
    print(f'[FAIL] Error: {e}')
    import traceback
    traceback.print_exc()
