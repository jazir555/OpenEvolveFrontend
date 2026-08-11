import requests
import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
import sys

API_URL = "http://localhost:8000"

def run_evolution(task_id: int, task: str, max_gen: int = 2):
    """Run a single evolution and return results."""
    payload = {
        "name": f"concurrent-test-{task_id}",
        "task": task,
        "max_generations": max_gen,
        "population_size": 3
    }

    print(f"[Task {task_id}] Starting evolution...")
    start_time = time.time()

    try:
        # Start evolution
        response = requests.post(
            f"{API_URL}/api/v1/evolve",
            json=payload,
            timeout=30
        )

        if response.status_code != 200:
            print(f"[Task {task_id}] FAILED to start: {response.status_code}")
            return {"task_id": task_id, "status": "failed", "error": response.text}

        result = response.json()
        evolution_id = result.get('evolution_id')
        print(f"[Task {task_id}] Started: {evolution_id}")

        # Poll for completion
        while time.time() - start_time < 120:  # 2 min timeout
            time.sleep(2)

            status_response = requests.get(f"{API_URL}/api/v1/status/{evolution_id}")
            if status_response.status_code != 200:
                continue

            status_data = status_response.json()
            status = status_data.get('status')
            generation = status_data.get('current_generation', 0)
            fitness = status_data.get('best_fitness', 0.0)

            elapsed = int(time.time() - start_time)
            print(f"[Task {task_id}] T+{elapsed}s: Gen={generation}, Fit={fitness:.3f}, Status={status}")

            if status in ['COMPLETED', 'FAILED']:
                # Get solution
                solution_response = requests.get(f"{API_URL}/api/v1/solutions/{evolution_id}")
                solution_data = solution_response.json() if solution_response.status_code == 200 else {}

                elapsed_total = int(time.time() - start_time)
                print(f"[Task {task_id}] COMPLETE in {elapsed_total}s - Status: {status}, Fitness: {fitness}")

                return {
                    "task_id": task_id,
                    "evolution_id": evolution_id,
                    "status": status,
                    "generations": generation,
                    "fitness": fitness,
                    "duration": elapsed_total,
                    "solution": solution_data.get('solution', '')[:100]
                }

        print(f"[Task {task_id}] TIMEOUT")
        return {"task_id": task_id, "status": "timeout"}

    except Exception as e:
        print(f"[Task {task_id}] ERROR: {e}")
        return {"task_id": task_id, "status": "error", "error": str(e)}

def test_concurrent_evolutions():
    """Test multiple concurrent evolutions."""
    print("=" * 70)
    print("PHASE 2.3: CONCURRENT EVOLUTION TEST")
    print("=" * 70)
    print(f"API URL: {API_URL}")
    print()

    # Define test tasks
    tasks = [
        (1, "What is 2+2? Answer with the number."),
        (2, "What is 3+3? Answer with the number."),
        (3, "What is 5+5? Answer with the number."),
    ]

    print(f"Running {len(tasks)} concurrent evolutions...")
    print()

    start_time = time.time()

    # Run evolutions concurrently
    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
        futures = [executor.submit(run_evolution, task_id, task, 2) for task_id, task in tasks]

        results = []
        for future in futures:
            try:
                result = future.result(timeout=150)
                results.append(result)
            except Exception as e:
                print(f"Future failed: {e}")
                results.append({"status": "error", "error": str(e)})

    total_time = int(time.time() - start_time)

    print()
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Time: {total_time}s")
    print(f"Total Evolutions: {len(results)}")
    print()

    completed = sum(1 for r in results if r.get('status') == 'COMPLETED')
    failed = sum(1 for r in results if r.get('status') == 'FAILED')
    errors = sum(1 for r in results if r.get('status') in ['error', 'timeout'])

    print(f"[OK] Completed: {completed}")
    print(f"[FAIL] Failed: {failed}")
    print(f"[WARN] Errors: {errors}")
    print()

    for result in results:
        task_id = result.get('task_id')
        status = result.get('status')
        fitness = result.get('fitness', 0)
        duration = result.get('duration', 0)
        print(f"Task {task_id}: {status:12} | Fitness: {fitness:.3f} | Duration: {duration}s")

    print()
    print("=" * 70)

    # Success criteria:
    # 1. All tasks should complete without errors
    # 2. Total time should be roughly the time of one task (concurrent)
    # 3. No database corruption or mixing of results

    if completed == len(tasks) and failed == 0 and errors == 0:
        print("PHASE 2.3 TEST: PASSED")
        print("Concurrent evolutions work correctly!")
        print("=" * 70)
        return True
    else:
        print("PHASE 2.3 TEST: FAILED")
        print("=" * 70)
        return False

if __name__ == "__main__":
    success = test_concurrent_evolutions()
    sys.exit(0 if success else 1)
