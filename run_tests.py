#!/usr/bin/env python3
"""
Quick test runner for BubbleLabs OpenEvolve Plugin comprehensive tests.
"""

import sys
import subprocess
import argparse


def run_pytest(args):
    """Run pytest with the provided arguments"""
    pytest_args = ['pytest', 'test_bubblelabs_comprehensive.py']

    if args.verbose:
        pytest_args.append('-vv')
    elif args.v:
        pytest_args.append('-v')
    else:
        pytest_args.append('-v')

    if args.coverage:
        pytest_args.extend([
            '--cov=.',
            '--cov-report=html',
            '--cov-report=term-missing'
        ])

    if args.unit:
        pytest_args.extend(['-m', 'unit'])
    elif args.integration:
        pytest_args.extend(['-m', 'integration'])
    elif args.fast:
        pytest_args.extend(['-m', 'not slow'])
    elif args.e2e:
        pytest_args.extend(['-m', 'e2e'])

    if args.parallel:
        pytest_args.extend(['-n', 'auto'])

    if args.failfast:
        pytest_args.append('-x')

    if args.s:
        pytest_args.append('-s')

    print(f"\n{'='*70}")
    print("Running: " + ' '.join(pytest_args))
    print(f"{'='*70}\n")

    result = subprocess.run(pytest_args)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(description='Run BubbleLabs OpenEvolve Plugin tests')
    
    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument('--unit', action='store_true', help='Run only unit tests')
    test_group.add_argument('--integration', action='store_true', help='Run only integration tests')
    test_group.add_argument('--e2e', action='store_true', help='Run only end-to-end tests')
    test_group.add_argument('--fast', action='store_true', help='Run only fast tests')

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    parser.add_argument('-vv', action='store_true', help='Very verbose output')
    parser.add_argument('-s', '--capture', action='store_true', help='Show print statements')
    parser.add_argument('--coverage', action='store_true', help='Generate coverage report')
    parser.add_argument('--parallel', action='store_true', help='Run tests in parallel')
    parser.add_argument('--failfast', '-x', action='store_true', help='Stop on first failure')

    args = parser.parse_args()

    if args.vv:
        args.verbose = True

    sys.exit(run_pytest(args))


if __name__ == '__main__':
    main()
