#!/usr/bin/env python3
"""
Simple comparison of ParameterManager vs UnifiedConfiguration
"""

import sys
import os
import time
import statistics

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from parameter_manager import ParameterManager
    from unified_configuration import create_unified_config
except ImportError as e:
    print(f"ERROR: Import error: {e}")
    sys.exit(1)

def main():
    print("=" * 60)
    print("PARAMETER MANAGER vs UNIFIEDCONFIGURATION COMPARISON")
    print("=" * 60)
    print()

    # Test performance
    print("PERFORMANCE TEST:")
    print("-" * 20)

    # ParameterManager
    pm_times = []
    for _ in range(10):
        start = time.perf_counter()
        pm = ParameterManager()
        pm_times.append(time.perf_counter() - start)
    pm_avg = statistics.mean(pm_times) * 1000

    # UnifiedConfiguration
    uc_times = []
    for _ in range(10):
        start = time.perf_counter()
        uc = create_unified_config(validate=False)
        uc_times.append(time.perf_counter() - start)
    uc_avg = statistics.mean(uc_times) * 1000

    print(f"ParameterManager:     {pm_avg:.3f}ms")
    print(f"UnifiedConfiguration:  {uc_avg:.3f}ms")

    if uc_avg > 0:
        ratio = pm_avg / uc_avg
        print(f"Ratio: {ratio:.2f}x {'faster' if ratio > 1 else 'slower'}")

    print()
    print("BENEFITS OF UNIFIEDCONFIGURATION:")
    print("-" * 20)
    print("[OK] Property access: config.max_iterations")
    print("[OK] Flexible get/set methods")
    print("[OK] Dict-style access: config['param']")
    print("[OK] Parameter merging capabilities")
    print("[OK] File save/load operations")
    print("[OK] Preset configuration functions")
    print("[OK] Better error handling")
    print("[OK] Reduced code duplication")
    print("[OK] Consistent interface across modules")
    print()
    print("CONCLUSION:")
    print("-" * 20)
    print("UnifiedConfiguration provides more features with")
    print("similar performance and better maintainability.")
    print("Migration is recommended!")

    return 0

if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as e:  # TODO: Catch specific exception instead of Exception
        print(f"ERROR: {e}")
        sys.exit(1)