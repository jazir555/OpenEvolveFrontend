"""
Test MDAP/MAKER-Gauntlet Integration with DeepSeek API

Actually executes gauntlets using your DeepSeek API key.
"""

import os
import sys
from dotenv import load_dotenv

load_dotenv()

from mdap_maker_gauntlet_integration import (
    MDAPMakerGauntletIntegration,
    MDAPMakerGauntletConfig,
    MDAPMakerGauntletMode
)
from gauntlet_types import AdversarialGauntlet, StatisticalGauntlet

print("\n" + "="*80)
print("MDAP/MAKER-GAUNTLET INTEGRATION TEST WITH DEEPSEEK")
print("="*80)

# Check API key
deepseek_key = os.getenv('DEEPSEEK_API_KEY')
if not deepseek_key:
    print("[FAIL] DEEPSEEK_API_KEY not found in .env")
    sys.exit(1)

print(f"[OK] DeepSeek API key loaded: {deepseek_key[:10]}...")
print()

# Create integration
print("Creating MDAP/MAKER-Gauntlet integration...")
config = MDAPMakerGauntletConfig(
    mode=MDAPMakerGauntletMode.HYBRID,
    use_complexity_adaptation=True,
    use_maker_voting=True,
    use_red_flagging=True,
    maker_k_min=2,
    maker_k_max=5,
    maker_max_votes=10  # Keep it small for testing
)

integration = MDAPMakerGauntletIntegration(config=config)
print("[OK] Integration created")
print()

# Test 1: Simple problem
print("="*80)
print("TEST 1: Simple Math Problem")
print("="*80)

problem1 = "What is 2 + 2?"
solution1 = {"answer": 4, "code": "def add(): return 2 + 2"}

print(f"Problem: {problem1}")
print(f"Solution: {solution1}")
print()

gauntlet1, result1 = integration.create_mdap_adaptive_gauntlet(
    problem_description=problem1,
    solution=solution1,
    context={"domain": "math"}
)

print(f"Gauntlet Type: {gauntlet1.gauntlet_type.value}")
print(f"Gauntlet Name: {gauntlet1.name}")
print(f"Complexity Score: {result1.complexity_score.overall_score:.3f}")
print(f"MDAP Strategy: {result1.mdap_strategy}")
print(f"Gauntlet Passed: {result1.gauntlet_result.passed}")
print(f"Gauntlet Score: {result1.gauntlet_result.score:.3f}")
print(f"Consensus Score: {result1.consensus_score:.3f}")
print()

# Test 2: Medium complexity problem
print("="*80)
print("TEST 2: Medium Complexity - Sorting Algorithm")
print("="*80)

problem2 = "Implement a quicksort algorithm in Python"
solution2 = {
    "code": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
"""
}

print(f"Problem: {problem2[:50]}...")
print(f"Solution: {len(solution2['code'])} characters")
print()

gauntlet2, result2 = integration.create_mdap_adaptive_gauntlet(
    problem_description=problem2,
    solution=solution2,
    context={"domain": "algorithms", "difficulty": "medium"}
)

print(f"Gauntlet Type: {gauntlet2.gauntlet_type.value}")
print(f"Complexity Score: {result2.complexity_score.overall_score:.3f}")
print(f"MDAP Strategy: {result2.mdap_strategy}")
print(f"Gauntlet Passed: {result2.gauntlet_result.passed}")
print(f"Gauntlet Score: {result2.gauntlet_result.score:.3f}")
print(f"Agent Votes: {len(result2.agent_votes)}")
print(f"Consensus Reached: {result2.consensus_reached}")
print()

# Test 3: Execute gauntlet directly
print("="*80)
print("TEST 3: Direct Gauntlet Execution with MAKER")
print("="*80)

gauntlet3 = AdversarialGauntlet("deepseek_test", config={'timeout': 30})
solution3 = {"data": [1, 2, 3, 4, 5], "operation": "sum"}

print(f"Gauntlet: {gauntlet3.name}")
print(f"Solution: {solution3}")
print()

result3 = integration.execute_with_mdap_maker(
    gauntlet=gauntlet3,
    solution=solution3,
    problem_description="Calculate the sum of a list",
    context={"expected": 15}
)

print(f"Gauntlet Passed: {result3.gauntlet_result.passed}")
print(f"Gauntlet Score: {result3.gauntlet_result.score:.3f}")
print(f"Complexity: {result3.complexity_score.overall_score if result3.complexity_score else 'N/A'}")
print(f"Agent Votes: {len(result3.agent_votes)}")
print(f"Red Flags: {len(result3.red_flags)}")
print(f"Consensus: {result3.consensus_score:.3f}")
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print(f"Test 1 (Simple): {'PASS' if result1.gauntlet_result.passed else 'FAIL'}")
print(f"Test 2 (Medium): {'PASS' if result2.gauntlet_result.passed else 'FAIL'}")
print(f"Test 3 (Direct): {'PASS' if result3.gauntlet_result.passed else 'FAIL'}")
print()

passed = sum([
    result1.gauntlet_result.passed,
    result2.gauntlet_result.passed,
    result3.gauntlet_result.passed
])

print(f"Total: {passed}/3 tests passed ({100*passed/3:.1f}%)")
print("="*80)

if passed >= 2:
    print("\n[SUCCESS] DeepSeek integration is working!")
else:
    print("\n[WARN] Some tests failed. Check the output above.")
