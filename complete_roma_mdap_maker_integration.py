"""
Complete ROMA-MDAP-MAKER Integration Verification

Verifies the end-to-end integration of:
1. ROMA Hierarchical Decomposition
2. Associative Recomposition
3. MDAP Multi-Agent Validation
4. Evaluator Team Assessment
5. Adaptive Gauntlet System (Placeholder)

Usage:
    python complete_roma_mdap_maker_integration.py
"""

import sys
import logging
import time

sys.path.insert(0, '.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def verify_integration():
    print("\n" + "="*80)
    print("COMPLETE ROMA-MDAP-MAKER INTEGRATION VERIFICATION")
    print("="*80 + "\n")

    # 1. Check Imports
    print("1. Checking Imports...")
    try:
        from roma_mdap_maker_associative_integration import (
            create_romamdapmaker_associative_config,
            ROMAMDAPMakerAssociativeEngine,
            get_romamdapmaker_associative_status,
            EVALUATOR_TEAM_AVAILABLE,
            GAUNTLET_SYSTEM_AVAILABLE
        )
        print("   [OK] roma_mdap_maker_associative_integration imported")
    except ImportError as e:
        print(f"   [FAIL] Import failed: {e}")
        return

    # 2. Check Component Status
    print("\n2. Checking Component Status...")
    status = get_romamdapmaker_associative_status()
    print(f"   ROMA-MDAP-MAKER Available: {status['roma_mdap_maker_available']}")
    print(f"   Associative Recomposition Available: {status['associative_available']}")
    print(f"   Evaluator Team Available: {EVALUATOR_TEAM_AVAILABLE}")
    print(f"   Gauntlet System Available: {GAUNTLET_SYSTEM_AVAILABLE}")

    if not status['roma_mdap_maker_available']:
        print("\n   [WARN] Core ROMA/MDAP components missing. Running in mock/fallback mode if possible.")

    # 3. Creating Configuration
    from roma_mdap_maker_reliability_ssot import get_reliability_config
    config = get_reliability_config(
        preset="fast", # Base on fast preset
        use_evaluator_team=True, # Force enable for mock test
        use_gauntlet_system=True, # Force enable for mock test
        roma_enable_logging=True
    )
    print("   [OK] Configuration created (Mock Full Pipeline)")

    # 4. Initialize Engine
    print("\n4. Initializing Engine...")
    engine = ROMAMDAPMakerAssociativeEngine(config)
    print("   [OK] Engine initialized")
    
    if engine.evaluator_team:
        print("   [OK] Evaluator Team attached")
    else:
        print("   [WARN] Evaluator Team NOT attached")

    if engine.gauntlet_system:
        print("   [OK] Gauntlet System attached")
    else:
        print("   [WARN] Gauntlet System NOT attached")

    # 5. Run Test Problem (Recursive)
    print("\n5. Running Test Problem (Recursive)...")
    problem = """
    Create a Python function to calculate the Fibonacci sequence up to n terms.
    Include error handling for negative inputs and a docstring.
    """
    print(f"   Problem: {problem.strip()}")

    start_time = time.time()
    result = engine.solve_problem_recursive(
        problem=problem,
        context={"requirements": ["Python", "Error handling", "Docstring"]}
    )
    duration = time.time() - start_time

    # 6. Analyze Results
    print("\n6. Analyzing Results...")
    if result.get("error"):
        print(f"   [FAIL] Execution error: {result['error']}")
    else:
        print(f"   [OK] Execution successful in {duration:.2f}s")
        print(f"   Final Attempt: {result.get('final_attempt', 1)}")
        print(f"   Confidence: {result.get('confidence', 0.0):.2f}")

        
        # Check Evaluator Assessment
        eval_assessment = result.get("evaluator_assessment", {})
        print("\n   Evaluator Assessment:")
        print(f"     Verdict: {eval_assessment.get('final_verdict') or eval_assessment.get('verdict') or 'N/A'}")
        print(f"     Consensus Score: {eval_assessment.get('consensus_score', 'N/A')}")
        
        # Check MDAP Validation
        mdap_val = result.get("mdap_validation", {})
        print("\n   MDAP Validation:")
        print(f"     Validated: {mdap_val.get('validated', False)}")
        print(f"     Red Flags: {mdap_val.get('red_flags', 'N/A')}")
        
        # Check Gauntlet
        gauntlet_res = result.get("gauntlet_result", {})
        print("\n   Gauntlet Result:")
        if gauntlet_res:
            print(f"     Passed: {gauntlet_res.get('passed', False)}")
            print(f"     Score: {gauntlet_res.get('score', 0.0):.2f}")
            print(f"     Rounds: {len(gauntlet_res.get('rounds', []))}")
            for feedback in gauntlet_res.get("feedback", []):
                print(f"       - {feedback}")
        else:
            print("     Status: Not Run/Empty")

        # Check Solution
        solution = result.get("solution", "")
        print(f"\n   Solution Length: {len(solution)} chars")
        if len(solution) < 500:
            print(f"   Solution Preview:\n{solution}")

    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    verify_integration()
