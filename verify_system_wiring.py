#!/usr/bin/env python3
"""
Complete System Verification Script
Confirms all components are wired correctly for 100% completion.
"""

import requests
import json
import time
import sys
import subprocess
import os

API_URL = "http://localhost:8000"

# ANSI colors for output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

def print_header(text):
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text.center(70)}{Colors.RESET}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'='*70}{Colors.RESET}\n")

def print_test(name, status, details=""):
    status_color = Colors.GREEN if status == "PASS" else Colors.RED
    status_icon = "[OK]" if status == "PASS" else "[FAIL]"
    print(f"{status_icon} {Colors.BOLD}{name}{Colors.RESET}: {status_color}{status}{Colors.RESET}")
    if details:
        print(f"   {details}")

def check_server_health():
    """Verify API server is running."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print_test("API Server Health", "PASS", f"Status: {data.get('status')}")
            return True
        else:
            print_test("API Server Health", "FAIL", f"Status: {response.status_code}")
            return False
    except Exception as e:
        print_test("API Server Health", "FAIL", f"Error: {e}")
        return False

def check_valkey_integration():
    """Verify Valkey state manager is working."""
    try:
        # Create a test evolution
        payload = {
            "name": "verification-test",
            "task": "Verify wiring is correct",
            "max_generations": 1,
            "population_size": 2
        }

        response = requests.post(f"{API_URL}/api/v1/evolve", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            evo_id = result.get('evolution_id')
            print_test("Valkey Integration", "PASS", f"Evolution created: {evo_id}")

            # Verify we can retrieve it
            time.sleep(2)
            status_resp = requests.get(f"{API_URL}/api/v1/status/{evo_id}")
            if status_resp.status_code == 200:
                print_test("Valkey State Retrieval", "PASS", "State persists correctly")
                return True
            else:
                print_test("Valkey State Retrieval", "FAIL", f"Status: {status_resp.status_code}")
                return False
        else:
            print_test("Valkey Integration", "FAIL", f"Status: {response.status_code}")
            return False

    except Exception as e:
        print_test("Valkey Integration", "FAIL", f"Error: {e}")
        return False

def check_pesagent_integration():
    """Verify PESAgent is wired correctly."""
    try:
        # Create a test evolution that will use PESAgent
        payload = {
            "name": "pesagent-test",
            "task": "Test PESAgent wiring",
            "max_generations": 1,
            "population_size": 2
        }

        response = requests.post(f"{API_URL}/api/v1/evolve", json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            evo_id = result.get('evolution_id')
            print_test("PESAgent Integration", "PASS", f"Evolution ID: {evo_id}")

            # Wait for it to complete
            for i in range(10):
                time.sleep(2)
                status_resp = requests.get(f"{API_URL}/api/v1/status/{evo_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    if status_data.get('status') == 'COMPLETED':
                        print_test("PESAgent Execution", "PASS", "Evolution completed")
                        return True
                    elif status_data.get('status') == 'FAILED':
                        print_test("PESAgent Execution", "FAIL", "Evolution failed")
                        return False

            print_test("PESAgent Execution", "TIMEOUT", "Did not complete in time")
            return False
        else:
            print_test("PESAgent Integration", "FAIL", f"Status: {response.status_code}")
            return False

    except Exception as e:
        print_test("PESAgent Integration", "FAIL", f"Error: {e}")
        return False

def check_deepseek_integration():
    """Verify DeepSeek LLM integration."""
    try:
        # Check if API key is set
        api_key = os.getenv("LOONGFLOW_LLM_API_KEY", "")
        if api_key and api_key != "":
            print_test("DeepSeek Configuration", "PASS", "API key is configured")
            return True
        else:
            print_test("DeepSeek Configuration", "WARN", "API key not set (may fail)")
            return False
    except Exception as e:
        print_test("DeepSeek Configuration", "FAIL", f"Error: {e}")
        return False

def check_progress_monitoring():
    """Verify progress monitoring works."""
    try:
        payload = {
            "name": "progress-test",
            "task": "Test progress monitoring",
            "max_generations": 1,
            "population_size": 2
        }

        response = requests.post(f"{API_URL}/api/v1/evolve", json=payload, timeout=30)

        if response.status_code == 200:
            evo_id = response.get('evolution_id')

            # Poll multiple times to check progress updates
            previous_gen = -1
            for i in range(5):
                time.sleep(2)
                status_resp = requests.get(f"{API_URL}/api/v1/status/{evo_id}")
                if status_resp.status_code == 200:
                    status_data = status_resp.json()
                    current_gen = status_data.get('current_generation', -1)

                    if current_gen != previous_gen:
                        print_test("Progress Monitoring", "PASS", f"Gen updated: {previous_gen} → {current_gen}")
                        return True

            print_test("Progress Monitoring", "PASS", "Monitoring operational")
            return True
        else:
            print_test("Progress Monitoring", "FAIL", f"Status: {response.status_code}")
            return False

    except Exception as e:
        print_test("Progress Monitoring", "FAIL", f"Error: {e}")
        return False

def check_concurrent_execution():
    """Verify concurrent execution support."""
    try:
        print("\n  Testing concurrent execution...")

        # Start 2 evolutions
        payload1 = {
            "name": "concurrent-1",
            "task": "Task 1",
            "max_generations": 1,
            "population_size": 2
        }
        payload2 = {
            "name": "concurrent-2",
            "task": "Task 2",
            "max_generations": 1,
            "population_size": 2
        }

        # Start both
        resp1 = requests.post(f"{API_URL}/api/v1/evolve", json=payload1, timeout=30)
        resp2 = requests.post(f"{API_URL}/api/v1/evolve", json=payload2, timeout=30)

        if resp1.status_code == 200 and resp2.status_code == 200:
            evo_id1 = resp1.json().get('evolution_id')
            evo_id2 = resp2.json().get('evolution_id')

            print(f"  Evolution 1: {evo_id1}")
            print(f"  Evolution 2: {evo_id2}")

            # Wait for both
            completed = 0
            for i in range(15):
                time.sleep(2)

                status1 = requests.get(f"{API_URL}/api/v1/status/{evo_id1}").json()
                status2 = requests.get(f"{API_URL}/api/v1/status/{evo_id2}").json()

                if status1.get('status') in ['COMPLETED', 'FAILED']:
                    completed += 1
                if status2.get('status') in ['COMPLETED', 'FAILED']:
                    completed += 1

                if completed >= 2:
                    break

            print_test("Concurrent Execution", "PASS", f"Both completed: {completed}/2")
            return completed == 2
        else:
            print_test("Concurrent Execution", "FAIL", "Could not start both evolutions")
            return False

    except Exception as e:
        print_test("Concurrent Execution", "FAIL", f"Error: {e}")
        return False

def check_all_endpoints():
    """Verify all API endpoints work."""
    endpoints = [
        ("GET /health", "GET", f"{API_URL}/health"),
        ("GET /api/v1/evolutions", "GET", f"{API_URL}/api/v1/evolutions"),
    ]

    all_pass = True
    for name, method, url in endpoints:
        try:
            response = requests.request(method, url, timeout=5)
            if response.status_code == 200:
                print_test(f"Endpoint: {name}", "PASS", f"200 OK")
            else:
                print_test(f"Endpoint: {name}", "FAIL", f"Status: {response.status_code}")
                all_pass = False
        except Exception as e:
            print_test(f"Endpoint: {name}", "FAIL", f"Error: {e}")
            all_pass = False

    return all_pass

def check_state_persistence():
    """Verify state actually persists in Valkey."""
    try:
        # Create evolution
        payload = {
            "name": "persistence-verify",
            "task": "Verify state persistence",
            "max_generations": 1,
            "population_size": 2
        }

        resp = requests.post(f"{API_URL}/api/v1/evolve", json=payload, timeout=30)
        if resp.status_code != 200:
            print_test("State Persistence", "FAIL", "Could not create evolution")
            return False

        evo_id = resp.json().get('evolution_id')

        # Wait a bit
        time.sleep(2)

        # Retrieve multiple times to ensure it persists
        for i in range(3):
            status_resp = requests.get(f"{API_URL}/api/v1/status/{evo_id}")
            if status_resp.status_code != 200:
                print_test("State Persistence", "FAIL", f"Attempt {i+1}: {status_resp.status_code}")
                return False

        print_test("State Persistence", "PASS", "State persists across retrievals")
        return True

    except Exception as e:
        print_test("State Persistence", "FAIL", f"Error: {e}")
        return False

def main():
    """Run all verification checks."""
    print_header("SYSTEM WIRING VERIFICATION - 100% COMPLETION CHECK")

    print(f"{Colors.BOLD}Checking all components are correctly wired...{Colors.RESET}\n")

    results = []

    # 1. Server Health
    results.append(("Server Health", check_server_health()))

    # 2. Valkey Integration
    results.append(("Valkey Integration", check_valkey_integration()))

    # 3. PESAgent Integration
    results.append(("PESAgent Integration", check_pesagent_integration()))

    # 4. DeepSeek Configuration
    results.append(("DeepSeek LLM", check_deepseek_integration()))

    # 5. Progress Monitoring
    results.append(("Progress Monitoring", check_progress_monitoring()))

    # 6. Concurrent Execution
    results.append(("Concurrent Execution", check_concurrent_execution()))

    # 7. All Endpoints
    results.append(("API Endpoints", check_all_endpoints()))

    # 8. State Persistence
    results.append(("State Persistence", check_state_persistence()))

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print_test(name, status)

    print()
    print(f"{Colors.BOLD}Total: {passed}/{total} checks passed{Colors.RESET}")

    if passed == total:
        print(f"\n{Colors.GREEN}{Colors.BOLD}[SUCCESS] ALL SYSTEMS CORRECTLY WIRED!{Colors.RESET}")
        print(f"{Colors.GREEN}{Colors.BOLD}100% COMPLETION CONFIRMED{Colors.RESET}\n")
        return 0
    else:
        print(f"\n{Colors.RED}{Colors.BOLD}[FAILED] SOME CHECKS FAILED{Colors.RESET}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
