"""
MDAP/MAKER-Gauntlet Integration Test with DeepSeek Multi-Agent

Uses multiple DeepSeek instances for different gauntlet roles:
- Red Team: 2 agents (attackers)
- Blue Team: 2 agents (defenders)  
- Judge: 1 agent (evaluator)
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
from gauntlet_types import AdversarialGauntlet

print("="*80)
print("DEEPSEEK MULTI-AGENT GAUNTLET TEST")
print("="*80)

# Check API key
deepseek_key = os.getenv('DEEPSEEK_API_KEY')
if not deepseek_key:
    print("[FAIL] DEEPSEEK_API_KEY not found in .env")
    sys.exit(1)

print(f"[OK] DeepSeek API key loaded")
print(f"[INFO] Using DeepSeek for ALL gauntlet roles (multi-instance)")
print()

# Create integration with multi-agent config
config = MDAPMakerGauntletConfig(
    mode=MDAPMakerGauntletMode.HYBRID,
    use_complexity_adaptation=True,
    use_maker_voting=True,
    use_red_flagging=True,
    maker_k_min=2,
    maker_k_max=5,
    maker_max_votes=15  # Allow more votes for multi-agent
)

print("Creating MDAP/MAKER-Gauntlet integration...")
integration = MDAPMakerGauntletIntegration(config=config)
print("[OK] Integration created")
print()

# Test Problem: Code Review
print("="*80)
print("TEST: Code Review Gauntlet with DeepSeek Multi-Agent")
print("="*80)

problem = "Review this Python code for security vulnerabilities and suggest improvements:"
solution = {
    "code": """
def authenticate_user(username, password):
    # Connect to database
    query = f"SELECT * FROM users WHERE username='{username}' AND password='{password}'"
    result = db.execute(query)
    
    if result:
        # Store password in session for 'convenience'
        session['password'] = password
        return True
    return False
"""
}

print(f"Problem: {problem[:60]}...")
print(f"Solution: {len(solution['code'])} characters")
print()

# Execute gauntlet
print("Executing gauntlet with DeepSeek multi-agent...")
print("(This may take 30-60 seconds for multiple API calls)")
print()

gauntlet, result = integration.create_mdap_adaptive_gauntlet(
    problem_description=problem,
    solution=solution,
    context={"domain": "security", "type": "code_review"}
)

# Display results
print("="*80)
print("RESULTS")
print("="*80)

print(f"\nGauntlet Type: {gauntlet.gauntlet_type.value}")
print(f"Gauntlet Name: {gauntlet.name}")

print(f"\n--- MDAP Analysis ---")
if result.complexity_score:
    print(f"Complexity Score: {result.complexity_score.overall_score:.3f}")
    print(f"  - Text: {result.complexity_score.text_length_score:.3f}")
    print(f"  - Depth: {result.complexity_score.depth_score:.3f}")
print(f"MDAP Strategy: {result.mdap_strategy}")

print(f"\n--- Gauntlet Result ---")
print(f"Passed: {result.gauntlet_result.passed}")
print(f"Score: {result.gauntlet_result.score:.3f}")
print(f"Confidence: {result.gauntlet_result.confidence:.3f}")
if result.gauntlet_result.feedback:
    print(f"Feedback: {result.gauntlet_result.feedback[:200]}...")

print(f"\n--- MAKER Multi-Agent ---")
print(f"Agent Votes: {len(result.agent_votes)}")
print(f"Red Flags: {len(result.red_flags)}")
print(f"Consensus Score: {result.consensus_score:.3f}")
print(f"Consensus Reached: {result.consensus_reached}")

if result.agent_votes:
    print(f"\n--- Sample Agent Votes ---")
    for i, vote in enumerate(result.agent_votes[:3], 1):
        if isinstance(vote, dict):
            print(f"  Agent {i}: Score={vote.get('score', 'N/A')}")

if result.red_flags:
    print(f"\n--- Red Flags ---")
    for flag in result.red_flags[:3]:
        print(f"  - {flag.get('message', 'Unknown issue')}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

status = "PASS" if result.gauntlet_result.passed else "FAIL"
print(f"Gauntlet: {status}")
print(f"DeepSeek Multi-Agent: {'WORKING' if len(result.agent_votes) > 0 else 'NO VOTES'}")
print(f"Consensus: {'REACHED' if result.consensus_reached else 'NOT REACHED'}")

if result.gauntlet_result.passed and len(result.agent_votes) > 0:
    print("\n[SUCCESS] DeepSeek multi-agent gauntlet is fully functional!")
else:
    print("\n[INFO] Gauntlet executed. Check results above.")

print("="*80)
