"""
Integration Test for Red-Flagging System with MCTS MDAP
"""

import asyncio
from leanaide_redflagging_system import (
    IntegratedRedFlaggingSystem,
    RedFlagConfig,
    RedFlagType
)


async def test_integration():
    """Test red-flagging system integration with MCTS MDAP."""
    print("Testing Red-Flagging System Integration with MCTS MDAP...")
    
    # Create configuration
    config = RedFlagConfig(
        confidence_threshold=0.4,
        max_proof_length=100,
        blocked_patterns=["sorry", "admit", "classical.choice"],
        enable_detailed_analysis=True
    )
    
    # Create integrated system
    system = IntegratedRedFlaggingSystem(config)
    
    # Test 1: Flag low-confidence action
    print("\n1. Testing low-confidence action flagging:")
    is_flagged, flags = system.flag_mdap_mcts_item(
        item="simp",
        item_type="action",
        context={"agent_id": "test_agent", "confidence": 0.2}
    )
    print(f"   Flagged: {is_flagged}")
    for flag in flags:
        print(f"   - {flag.flag_type.value}: {flag.reason}")
    
    # Test 2: Flag proof with blocked pattern
    print("\n2. Testing proof with blocked pattern:")
    bad_proof = "theorem test : True := by sorry  -- This uses sorry which is blocked"
    is_flagged, flags = system.flag_mdap_mcts_item(
        item=bad_proof,
        item_type="proof"
    )
    print(f"   Flagged: {is_flagged}")
    for flag in flags:
        print(f"   - {flag.flag_type.value}: {flag.reason}")
    
    # Test 3: Test normal proof (should not be flagged)
    print("\n3. Testing normal proof (should not be flagged):")
    good_proof = "theorem test : True := by trivial"
    is_flagged, flags = system.flag_mdap_mcts_item(
        item=good_proof,
        item_type="proof"
    )
    print(f"   Flagged: {is_flagged}")
    if flags:
        for flag in flags:
            print(f"   - {flag.flag_type.value}: {flag.reason}")
    else:
        print("   No flags (as expected)")
    
    # Test 4: Test vote aggregation with low agreement
    print("\n4. Testing vote aggregation with low agreement:")
    votes = [{"tactic": "simp"}, {"tactic": "intros"}, {"tactic": "rw"}, {"tactic": "apply"}]
    is_flagged, flags = system.flag_mdap_mcts_item(
        item="selected_tactic",
        item_type="aggregation",
        context={"votes": votes}
    )
    print(f"   Flagged: {is_flagged}")
    for flag in flags:
        print(f"   - {flag.flag_type.value}: {flag.reason}")
    
    # Test 5: Test node with high visit count but low reward
    print("\n5. Testing MCTS node analysis:")
    node = {
        "N": 50,  # High visit count
        "W": 2,   # Low total reward
        "Q": 0.04  # Low average reward
    }
    is_flagged, flags = system.flag_mdap_mcts_item(
        item=node,
        item_type="node",
        context={"system": "mcts"}
    )
    print(f"   Flagged: {is_flagged}")
    for flag in flags:
        print(f"   - {flag.flag_type.value}: {flag.reason}")
    
    # Test 6: System analysis
    print("\n6. Testing system analysis:")
    all_flags = []
    # Add some flags for analysis
    all_flags.extend(flags)  # From previous test
    
    analysis = system.analyze_system_flags(all_flags)
    print(f"   Total flags analyzed: {analysis['total_flags']}")
    print(f"   MDAP flags: {analysis['mdap_flags']}")
    print(f"   MCTS flags: {analysis['mcts_flags']}")
    print(f"   MAKER flags: {analysis['maker_flags']}")
    
    # Test 7: Recommendations
    print("\n7. Testing system recommendations:")
    recommendations = system.get_system_recommendations(all_flags)
    print(f"   Recommendations: {recommendations}")
    
    print("\nIntegration test completed successfully!")


if __name__ == "__main__":
    asyncio.run(test_integration())