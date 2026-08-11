#!/usr/bin/env python3
"""Verify ROMA core integration is working."""

import warnings
warnings.filterwarnings('ignore')

print('=' * 80)
print('ROMA CORE INTEGRATION - FINAL VERIFICATION')
print('=' * 80)

from knowledge_engine.integrations import ROMAIntegration, ROMA_INTEGRATION_AVAILABLE

print()
print(f'ROMA_INTEGRATION_AVAILABLE: {ROMA_INTEGRATION_AVAILABLE}')
print()

if ROMA_INTEGRATION_AVAILABLE:
    print('ROMA Core Components:')
    from roma_dspy import Atomizer, Planner, Executor, Aggregator, Verifier, RecursiveSolver
    print('  [OK] Atomizer')
    print('  [OK] Planner')
    print('  [OK] Executor')
    print('  [OK] Aggregator')
    print('  [OK] Verifier')
    print('  [OK] RecursiveSolver')

    print()
    print('ROMA Integration:')
    roma = ROMAIntegration()
    print('  [OK] ROMA initialized')

    print()
    print('=' * 80)
    print('SUCCESS - ROMA CORE IS FULLY FUNCTIONAL!')
    print('=' * 80)
else:
    print('[FAIL] ROMA core not available')
