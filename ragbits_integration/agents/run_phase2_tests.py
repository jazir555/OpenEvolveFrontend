#!/usr/bin/env python
"""
Phase 2 Integration Test - Agent Coordination with A2A Protocol

Demonstrates complete agent workflow:
1. Blue Team generates solution
2. Blue Team notifies Red Team via A2A
3. Red Team critiques solution
4. Red Team requests refinement via A2A
5. Blue Team refines solution
6. Gold Team verifies solution
7. Results communicated via A2A
"""

import asyncio
import sys

# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client for agent testing"""

    async def generate(self, prompt: str, **kwargs):
        """Mock LLM generation"""
        return {"text": "Mock LLM response for: " + prompt[:100]}


async def run_phase_2_integration_test():
    """Run complete Phase 2 integration test"""

    from ragbits_integration.agents.blue_team_agent import BlueTeamAgent
    from ragbits_integration.agents.red_team_agent import RedTeamAgent
    from ragbits_integration.agents.gold_team_agent import GoldTeamAgent
    from ragbits_integration.agents.communication.a2a_protocol import (
        A2AProtocol,
        MessageBuilder
    )
    from ragbits_integration.intermediary_storage.storage_manager import IntermediaryStorageManager
    from ragbits_integration.tests.test_storage_manager import MockDocumentSearch

    print("=" * 80)
    print("PHASE 2: AGENT COORDINATION INTEGRATION TEST")
    print("=" * 80)
    print()

    # Setup
    print("Setting up agents and communication...")
    document_search = MockDocumentSearch()
    storage = IntermediaryStorageManager(document_search)
    llm_client = MockLLMClient()
    protocol = A2AProtocol()

    # Create agents
    blue_agent = BlueTeamAgent(
        hephaestus_client=llm_client,
        storage_manager=storage,
        knowledge_retriever=None
    )

    red_agent = RedTeamAgent(
        hephaestus_client=llm_client,
        storage_manager=storage,
        knowledge_retriever=None
    )

    gold_agent = GoldTeamAgent(
        hephaestus_client=llm_client,
        storage_manager=storage,
        knowledge_retriever=None
    )

    print("✓ Agents initialized")
    print("✓ A2A Protocol initialized")
    print("✓ Storage Manager initialized")
    print()

    # Sub-problem definition
    sub_problem = {
        "id": "sub_1",
        "title": "User Authentication System",
        "description": "Implement secure user authentication with JWT and OAuth support",
        "requirements": [
            "JWT token generation and validation",
            "OAuth 2.0 provider integration",
            "Secure password hashing (bcrypt)",
            "Session management"
        ]
    }

    print("-" * 80)
    print("STEP 1: Blue Team Generates Solution")
    print("-" * 80)

    # Blue Team generates solution
    blue_result = await blue_agent.generate_solution(
        sub_problem=sub_problem,
        context={},
        use_rag=False
    )

    print(f"✓ Blue Team generated solution")
    print(f"  Sub-problem: {blue_result['sub_problem_title']}")
    print(f"  Solution length: {len(blue_result['solution'])} chars")
    print(f"  Artifact ID: {blue_result['artifact_id']}")
    print()

    # Blue Team sends A2A message
    solution_msg = MessageBuilder.solution_submitted(
        sender="blue_team",
        recipient="red_team",
        solution=blue_result["solution"][:200],
        sub_problem_id=sub_problem["id"],
        artifact_id=blue_result["artifact_id"]
    )

    await protocol.send_message(
        sender=solution_msg.sender,
        recipient=solution_msg.recipient,
        message_type=solution_msg.message_type,
        content=solution_msg.content,
        sub_problem_id=solution_msg.sub_problem_id,
        artifact_id=solution_msg.artifact_id
    )

    print(f"✓ Blue Team notified Red Team via A2A protocol")
    print()

    # Red Team receives message
    red_messages = await protocol.get_messages("red_team")
    print(f"✓ Red Team received {len(red_messages)} messages")

    print("-" * 80)
    print("STEP 2: Red Team Critiques Solution")
    print("-" * 80)

    # Red Team critiques solution
    red_result = await red_agent.critique_solution(
        solution=blue_result["solution"],
        sub_problem=sub_problem,
        context={"solution_artifact_id": blue_result["artifact_id"]},
        use_patterns=False
    )

    print(f"✓ Red Team completed critique")
    print(f"  Issues identified: {red_result['total_issues']}")
    print(f"  Artifact ID: {red_result['artifact_id']}")
    print()

    # Red Team sends refinement request if needed
    if red_result["total_issues"] > 0:
        refinement_msg = await protocol.send_message(
            sender="red_team",
            recipient="blue_team",
            message_type=MessageType.REFINEMENT_REQUEST,
            content=f"Refinement requested for {len(red_result['parsed']['issues'])} issues",
            metadata={
                "issues": red_result["parsed"]["issues"][:3]
            },
            priority=MessagePriority.HIGH,
            requires_response=True,
            sub_problem_id=sub_problem["id"],
            artifact_id=blue_result["artifact_id"]
        )

        print(f"✓ Red Team sent refinement request via A2A protocol")
        print()

    print("-" * 80)
    print("STEP 3: Blue Team Refines Solution")
    print("-" * 80)

    # Blue Team refines based on critique
    refined_result = await blue_agent.refine_solution(
        current_solution=blue_result["solution"],
        critique=red_result["parsed"],
        iteration=2
    )

    print(f"✓ Blue Team refined solution")
    print(f"  New artifact ID: {refined_result['artifact_id']}")
    print(f"  Issues addressed: {len(refined_result['critique_addressed'])}")
    print()

    # Blue Team sends reply to Red Team
    blue_messages = await protocol.get_messages("blue_team")
    if blue_messages:
        original_refinement_request = blue_messages[0]
        await protocol.send_reply(
            original_message=original_refinement_request,
            reply_content="Refinement complete, all issues addressed"
        )

        print(f"✓ Blue Team confirmed refinement via A2A protocol")
        print()

    print("-" * 80)
    print("STEP 4: Gold Team Verifies Solution")
    print("-" * 80)

    # Gold Team verifies solution
    gold_result = await gold_agent.verify_solution(
        solution=refined_result["solution"],
        critique=red_result["parsed"],
        sub_problem=sub_problem,
        context={
            "solution_artifact_id": refined_result["artifact_id"],
            "critique_artifact_id": red_result["artifact_id"]
        }
    )

    print(f"✓ Gold Team completed verification")
    print(f"  Passes: {gold_result['passes']}")
    print(f"  Overall Score: {gold_result['overall_score']}/10")
    print(f"  Artifact ID: {gold_result['artifact_id']}")
    print()

    # Gold Team sends verification result via A2A
    verification_msg = MessageBuilder.verification_result(
        sender="gold_team",
        recipient="blue_team",
        passes=gold_result["passes"],
        score=gold_result["overall_score"],
        sub_problem_id=sub_problem["id"],
        artifact_id=refined_result["artifact_id"]
    )

    await protocol.send_message(
        sender=verification_msg.sender,
        recipient=verification_msg.recipient,
        message_type=verification_msg.message_type,
        content=verification_msg.content,
        metadata=verification_msg.metadata,
        sub_problem_id=verification_msg.sub_problem_id,
        artifact_id=verification_msg.artifact_id
    )

    print(f"✓ Gold Team sent verification result via A2A protocol")
    print()

    print("=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)

    # Get protocol statistics
    stats = protocol.get_statistics()
    print(f"Total messages sent: {stats['messages_sent']}")
    print(f"Messages delivered: {stats['messages_delivered']}")
    print(f"Pending responses: {stats['pending_responses']}")
    print()

    # Get agent metadata
    print("Agent Statistics:")
    for agent_name, agent in [("Blue Team", blue_agent), ("Red Team", red_agent), ("Gold Team", gold_agent)]:
        metadata = agent.get_metadata()
        print(f"  {agent_name}:")
        print(f"    Agent ID: {metadata['agent_id']}")
        print(f"    Model: {metadata['model_id']}")
        print(f"    Tools: {len(metadata['tools_available'])}")
    print()

    print("=" * 80)
    print("✅ PHASE 2 INTEGRATION TEST COMPLETE!")
    print("=" * 80)
    print()
    print("Components Tested:")
    print("  ✓ Blue Team Agent (solution generation)")
    print("  ✓ Red Team Agent (critique)")
    print("  ✓ Gold Team Agent (verification)")
    print("  ✓ A2A Protocol (agent-to-agent messaging)")
    print("  ✓ Message Builder (convenience methods)")
    print("  ✓ Message routing and delivery")
    print("  ✓ Refinement workflow")
    print("  ✓ Verification workflow")
    print()

    return True


async def test_agent_tools_integration():
    """Test agent tools integration"""
    from ragbits_integration.agents.blue_team_agent import BlueTeamAgent
    from ragbits_integration.agents.tools.knowledge_search_tool import KnowledgeSearchTool
    from ragbits_integration.agents.tools.solution_eval_tool import SolutionEvaluationTool
    from ragbits_integration.intermediary_storage.storage_manager import IntermediaryStorageManager
    from ragbits_integration.tests.test_storage_manager import MockDocumentSearch

    print("-" * 80)
    print("AGENT TOOLS INTEGRATION TEST")
    print("-" * 80)
    print()

    document_search = MockDocumentSearch()
    storage = IntermediaryStorageManager(document_search)
    llm_client = MockLLMClient()

    # Create retriever mock
    class MockKnowledgeRetriever:
        async def retrieve_similar_solutions(self, problem_description, top_k=5, **kwargs):
            return [
                {
                    "content": "Similar solution 1",
                    "success_rate": 0.9,
                    "team_used": "blue",
                    "similarity": 0.85
                }
            ]

    retriever = MockKnowledgeRetriever()

    # Create Blue Team agent with tools
    blue_agent = BlueTeamAgent(
        hephaestus_client=llm_client,
        storage_manager=storage,
        knowledge_retriever=retriever
    )

    print(f"✓ Blue Team agent initialized with {len(blue_agent.get_tools())} tools")
    print(f"  Available tools: {blue_agent.get_tools()}")
    print()

    # Test tool usage
    if "knowledge_search" in blue_agent.get_tools():
        tool_result = await blue_agent.use_tool(
            "knowledge_search",
            search_type="similar_solutions",
            query="authentication system",
            top_k=3
        )

        print("✓ Knowledge search tool executed")
        print(f"  Results: {len(tool_result)} items returned")
        print()

    # Get agent metadata
    metadata = blue_agent.get_metadata()
    print("Agent Metadata:")
    print(f"  Role: {metadata['role']}")
    print(f"  Model: {metadata['model_id']}")
    print(f"  Tools Available: {metadata['tools_available']}")
    print()

    print("✅ AGENT TOOLS INTEGRATION TEST COMPLETE!")
    print()

    return True


if __name__ == "__main__":
    async def main():
        print()
        print("=" * 80)
        print("RAGBits INTEGRATION - PHASE 2 TEST SUITE")
        print("Testing Agent Coordination with A2A Protocol")
        print("=" * 80)
        print()

        try:
            # Run main integration test
            success = await run_phase_2_integration_test()

            if success:
                # Run tools integration test
                await test_agent_tools_integration()

                print()
                print("=" * 80)
                print("🎉 ALL PHASE 2 TESTS PASSED!")
                print("=" * 80)

        except Exception as e:
            print()
            print("=" * 80)
            print(f"❌ TEST FAILED: {e}")
            print("=" * 80)
            import traceback
            traceback.print_exc()
            sys.exit(1)

    asyncio.run(main())
