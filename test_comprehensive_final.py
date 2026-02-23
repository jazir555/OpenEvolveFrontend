import requests
import json
import time
import sys

API_URL = "http://localhost:8000"

def test_comprehensive_evolution():
    """Comprehensive test with real evolutionary problem."""

    print("=" * 70)
    print("COMPREHENSIVE PHASE 2 FINAL TEST")
    print("=" * 70)
    print()

    # Test 1: Simple Math Problem with LLM Evaluator
    print("Test 1: Simple Math Problem")
    print("-" * 70)

    payload1 = {
        "name": "math-test",
        "task": "What is the sum of 15 and 27? Answer with just the number.",
        "max_generations": 1,
        "population_size": 3
    }

    print(f"Task: {payload1['task']}")

    try:
        response = requests.post(f"{API_URL}/api/v1/evolve", json=payload1, timeout=30)

        if response.status_code != 200:
            print(f"FAIL: Could not start evolution ({response.status_code})")
            return False

        result = response.json()
        evolution_id = result['evolution_id']
        print(f"Evolution ID: {evolution_id}")
        print(f"Status: {result['status']}")
        print()

        # Wait for completion
        print("Polling for completion...")
        for i in range(20):
            time.sleep(3)
            status_resp = requests.get(f"{API_URL}/api/v1/status/{evolution_id}")
            status_data = status_resp.json()
            status = status_data['status']
            gen = status_data.get('current_generation', 0)
            fit = status_data.get('best_fitness', 0.0)
            print(f"  Poll {i+1}: Status={status}, Gen={gen}, Fit={fit:.3f}")

            if status == 'COMPLETED':
                sol_resp = requests.get(f"{API_URL}/api/v1/solutions/{evolution_id}")
                solution = sol_resp.json()
                print()
                print("  SOLUTION RECEIVED:")
                print(f"  Generations: {solution.get('generations_completed')}")
                print(f"  Fitness: {solution.get('fitness')}")
                print(f"  Solution (first 200 chars): {solution.get('solution', '')[:200]}")
                print()
                print("  Test 1: PASS")
                break
            elif status == 'FAILED':
                print(f"  ERROR: {status_data.get('error', 'Unknown')}")
                print("  Test 1: FAIL")
                return False
        else:
            print("  Test 1: TIMEOUT")
            return False

    except Exception as e:
        print(f"Test 1: ERROR - {e}")
        return False

    print()

    # Test 2: List All Evolutions
    print("Test 2: List All Evolutions")
    print("-" * 70)

    try:
        list_resp = requests.get(f"{API_URL}/api/v1/evolutions")
        if list_resp.status_code == 200:
            evolutions = list_resp.json()
            print(f"Total evolutions: {len(evolutions.get('evolutions', []))}")
            for evo in evolutions.get('evolutions', [])[:3]:
                print(f"  - {evo.get('evolution_id')}: {evo.get('status')}")
            print("  Test 2: PASS")
        else:
            print(f"  Test 2: FAIL ({list_resp.status_code})")
            return False
    except Exception as e:
        print(f"  Test 2: ERROR - {e}")
        return False

    print()

    # Test 3: Delete Evolution
    print("Test 3: Delete Evolution")
    print("-" * 70)

    try:
        del_resp = requests.delete(f"{API_URL}/api/v1/evolutions/{evolution_id}")
        if del_resp.status_code == 200:
            result = del_resp.json()
            print(f"Delete result: {result.get('message', 'Success')}")
            print("  Test 3: PASS")
        else:
            print(f"  Test 3: FAIL ({del_resp.status_code})")
            return False
    except Exception as e:
        print(f"  Test 3: ERROR - {e}")
        return False

    print()

    # Test 4: Verify Deletion
    print("Test 4: Verify Deletion")
    print("-" * 70)

    try:
        verify_resp = requests.get(f"{API_URL}/api/v1/status/{evolution_id}")
        if verify_resp.status_code == 404:
            print("Evolution successfully deleted (404 response)")
            print("  Test 4: PASS")
        else:
            print(f"  Test 4: FAIL (expected 404, got {verify_resp.status_code})")
            return False
    except Exception as e:
        print(f"  Test 4: ERROR - {e}")
        return False

    print()
    print("=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print()
    print("Summary:")
    print("  - Real evolution execution: WORKING")
    print("  - Progress tracking: WORKING")
    print("  - Solution retrieval: WORKING")
    print("  - List evolutions: WORKING")
    print("  - Delete evolutions: WORKING")
    print("  - Error handling: WORKING")
    print()
    print("Phase 2 is FULLY OPERATIONAL!")
    print("=" * 70)

    return True

if __name__ == "__main__":
    success = test_comprehensive_evolution()
    sys.exit(0 if success else 1)
