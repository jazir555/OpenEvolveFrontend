"""
Test Enhanced Red-Flagging Functionality
"""

import asyncio
from leanaide_mcts_mdap import MDAPMCTSNode, MDAPMCTSConfig, ActionVote, ProofState


async def test_enhanced_redflagging():
    """Test the enhanced red-flagging functionality."""
    print("Testing Enhanced Red-Flagging Functionality...")
    
    # Create a mock ProofState
    from leanaide_mcts import ProofState
    state = ProofState(goals=["forall n, n + 0 = n"])
    
    # Create a node
    node = MDAPMCTSNode(state=state)
    
    # Add some low-performance agent votes
    node.add_agent_vote(
        agent_id="poor_agent",
        action="simp",
        confidence=0.1,
        rationale="This approach often fails",
        agent_type="test"
    )
    
    node.add_agent_vote(
        agent_id="poor_agent",
        action="simp",
        confidence=0.05,
        rationale="This approach often fails",
        agent_type="test"
    )
    
    # Update agent performance to reflect poor performance
    node.update_agent_performance("simp", success=False, confidence=0.1)
    node.update_agent_performance("simp", success=False, confidence=0.05)
    
    # Add some votes with suspicious rationales
    node.add_agent_vote(
        agent_id="another_agent",
        action="intros",
        confidence=0.2,
        rationale="This will probably error out",
        agent_type="test"
    )
    
    node.add_agent_vote(
        agent_id="another_agent",
        action="intros",
        confidence=0.15,
        rationale="Unable to proceed with this approach",
        agent_type="test"
    )
    
    # Manually set some properties to trigger red flags
    node.depth = 60  # Too deep
    node.children = {f"child_{i}": None for i in range(25)}  # Too many children
    
    # Test the enhanced red-flagging
    is_flagged, reasons = node.compute_comprehensive_red_flags()
    
    print(f"Node flagged: {is_flagged}")
    print(f"Reasons: {reasons}")
    
    # Test basic red-flagging
    node.set_red_flag(True, ["Test reason"])
    print(f"Manual flag status: {node.is_red_flagged()}")
    print(f"Manual flag reasons: {node.red_flag_reasons}")
    
    print("\nEnhanced red-flagging test completed!")


if __name__ == "__main__":
    asyncio.run(test_enhanced_redflagging())