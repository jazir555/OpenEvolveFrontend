import requests
import json
import sys

API_URL = "http://localhost:8000"

def test_evolution():
    """Test evolution with DeepSeek"""
    
    # Prepare the request
    payload = {
        "name": "deepseek-test",
        "task": "What is 2+2?",
        "max_generations": 1,
        "population_size": 1
    }
    
    print("Testing LoongFlow API with DeepSeek...")
    print(f"API URL: {API_URL}")
    print(f"Task: {payload['task']}")
    print()
    
    try:
        # Start evolution
        print("Starting evolution...")
        response = requests.post(
            f"{API_URL}/api/v1/evolve",
            json=payload,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("Evolution started successfully!")
            print(f"Evolution ID: {result.get('evolution_id')}")
            print(f"Status: {result.get('status')}")
            
            # Wait for evolution to complete
            import time
            evolution_id = result.get('evolution_id')
            
            for i in range(10):  # Wait up to 30 seconds
                time.sleep(3)
                
                # Check status
                status_response = requests.get(f"{API_URL}/api/v1/status/{evolution_id}")
                
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    status = status_data.get('status')
                    generation = status_data.get('current_generation', 0)
                    
                    print(f"Status: {status}, Generation: {generation}")
                    
                    if status == 'COMPLETED':
                        # Get solution
                        solution_response = requests.get(f"{API_URL}/api/v1/solutions/{evolution_id}")
                        if solution_response.status_code == 200:
                            solution = solution_response.json()
                            print("SOLUTION FOUND:")
                            print(f"  Actions: {solution.get('actions')}")
                            print(f"  Fitness: {solution.get('fitness')}")
                            return True
                    elif status == 'FAILED':
                        print("Evolution failed")
                        return False
                        
                elif i == 9:
                    print("Timeout - evolution taking too long")
                    return False
                    
        else:
            print(f"Failed to start evolution")
            print(f"Status: {response.status_code}")
            print(f"Error: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return False

if __name__ == "__main__":
    success = test_evolution()
    print()
    if success:
        print("TEST PASSED: LoongFlow API works with DeepSeek!")
    else:
        print("TEST FAILED: Could not complete evolution")
    sys.exit(0 if success else 1)
