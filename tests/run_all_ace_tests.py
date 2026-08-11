"""
MASTER TEST RUNNER - ACE Integration Comprehensive Tests
Runs all test suites with proper encoding
"""

import sys
import io

# Force UTF-8 encoding for output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

print('=' * 80)
print(' ACE INTEGRATION - COMPREHENSIVE TEST SUITE')
print('=' * 80)

test_results = {}

# Test 1: Security Attacks
print('\n[1/4] Running Security Attack Tests...')
print('-' * 80)
try:
    import subprocess
    result = subprocess.run(
        [sys.executable, 'test_ace_security_attacks.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    test_results['security_attacks'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }
    # Print output safely
    for line in result.stdout.split('\n')[:50]:
        try:
            print(line)
        except:
            print(line.encode('ascii', errors='replace').decode('ascii'))
except Exception as e:
    print(f'Error running security tests: {e}')
    test_results['security_attacks'] = {'error': str(e)}

# Test 2: Thread Safety
print('\n[2/4] Running Thread Safety Tests...')
print('-' * 80)
try:
    result = subprocess.run(
        [sys.executable, 'test_ace_thread_safety.py'],
        capture_output=True,
        text=True,
        timeout=120
    )
    test_results['thread_safety'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }
    for line in result.stdout.split('\n')[:50]:
        try:
            print(line)
        except:
            print(line.encode('ascii', errors='replace').decode('ascii'))
except Exception as e:
    print(f'Error running thread safety tests: {e}')
    test_results['thread_safety'] = {'error': str(e)}

# Test 3: Resource Leaks
print('\n[3/4] Running Resource Leak Tests...')
print('-' * 80)
try:
    result = subprocess.run(
        [sys.executable, 'test_ace_resource_leaks.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    test_results['resource_leaks'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }
    for line in result.stdout.split('\n')[:50]:
        try:
            print(line)
        except:
            print(line.encode('ascii', errors='replace').decode('ascii'))
except Exception as e:
    print(f'Error running resource leak tests: {e}')
    test_results['resource_leaks'] = {'error': str(e)}

# Test 4: Edge Cases
print('\n[4/4] Running Edge Case Tests...')
print('-' * 80)
try:
    result = subprocess.run(
        [sys.executable, 'test_ace_edge_cases.py'],
        capture_output=True,
        text=True,
        timeout=60
    )
    test_results['edge_cases'] = {
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }
    for line in result.stdout.split('\n')[:50]:
        try:
            print(line)
        except:
            print(line.encode('ascii', errors='replace').decode('ascii'))
except Exception as e:
    print(f'Error running edge case tests: {e}')
    test_results['edge_cases'] = {'error': str(e)}

# Summary
print('\n' + '=' * 80)
print(' COMPREHENSIVE TEST SUMMARY')
print('=' * 80)

for test_name, result in test_results.items():
    returncode = result.get('returncode', -1)
    status = 'PASS' if returncode == 0 else 'FAIL'
    print(f'{test_name.replace("_", " ").title()}: {status}')

print('\n' + '=' * 80)
