import requests
import json
import time
import sys

API_URL = "http://localhost:8000"

def test_phase2_evolution():
    """Test Phase 2: Real LoongFlow Evolution"""

    # Prepare a simple task
    payload = {
        "name": "phase2-test",
        "task": "What is 2+2? Answer with just the number.",
        "max_generations": 2,
        "population_size": 5
    }

    print("=" * 60)
    print("PHASE 2: REAL LOONGFLOW EVOLUTION TEST")
    print("=" * 60)
    print(f"API URL: {API_URL}")
    print(f"Task: {payload['task']}")
    print(f"Max Generations: {payload['max_generations']}")
    print()

    try:
        # Start evolution
        print("Starting real evolution...")
        response = requests.post(
            f"{API_URL}/api/v1/evolve",
            json=payload,
            timeout=30
        )

        print(f"Status Code: {response.status_code}")

        if response.status_code != 200:
            print(f"Failed to start evolution")
            print(f"Error: {response.text}")
            return False

        result = response.json()
        evolution_id = result.get('evolution_id')
        print(f"Evolution ID: {evolution_id}")
        print(f"Initial Status: {result.get('status')}")
        print()

        # Poll for completion
        print("Polling for completion (this will take time for real evolution)...")
        max_wait = 300  # 5 minutes max
        start_time = time.time()

        while time.time() - start_time < max_wait:
            time.sleep(5)  # Poll every 5 seconds

            status_response = requests.get(f"{API_URL}/api/v1/status/{evolution_id}")
            if status_response.status_code != 200:
                print(f"Error checking status: {status_response.status_code}")
                continue

            status_data = status_response.json()
            status = status_data.get('status')
            generation = status_data.get('current_generation', 0)
            fitness = status_data.get('best_fitness', 0.0)

            elapsed = int(time.time() - start_time)
            print(f"[{elapsed}s] Status: {status}, Generation: {generation}, Fitness: {fitness:.3f}")

            if status == 'COMPLETED':
                print()
                print("=" * 60)
                print("EVOLUTION COMPLETED!")
                print("=" * 60)

                # Get solution
                solution_response = requests.get(f"{API_URL}/api/v1/solutions/{evolution_id}")
                if solution_response.status_code == 200:
                    solution = solution_response.json()
                    print("SOLUTION:")
                    print("-" * 60)
                    print(solution.get('solution', 'No solution text')[:500])
                    print("-" * 60)
                    print(f"Fitness: {solution.get('fitness')}")
                    print(f"Generations: {solution.get('generations_completed')}")
                    print()
                    print("=" * 60)
                    print("PHASE 2 TEST: PASSED!")
                    print("=" * 60)
                    print("Real LoongFlow evolution is working!")
                    return True

            elif status == 'FAILED':
                print()
                print("EVOLUTION FAILED!")
                error = status_data.get('error', 'Unknown error')
                print(f"Error: {error}")
                return False

        print()
        print("TIMEOUT: Evolution taking too long")
        return False

    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_phase2_evolution()
    sys.exit(0 if success else 1)
