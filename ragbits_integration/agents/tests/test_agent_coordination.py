"""
Tests for Agent Coordination with RAGBits
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from ragbits_integration.agents.base_agent import BaseWorkflowAgent, AgentTool
from ragbits_integration.agents.blue_team_agent import BlueTeamAgent
from ragbits_integration.agents.red_team_agent import RedTeamAgent
from ragbits_integration.agents.gold_team_agent import GoldTeamAgent
from ragbits_integration.agents.communication.a2a_protocol import (
    A2AProtocol,
    A2AMessage,
    MessageType,
    MessageBuilder
)


# Mock LLM Client
class MockLLMClient:
    """Mock LLM client for testing"""

    async def generate(self, prompt: str, **kwargs):
        """Mock LLM generation"""
        role = kwargs.get("role", "assistant")

        if "solution" in prompt.lower() and "generate" in prompt.lower():
            return {
                "text": """## Overview
Implement JWT-based user authentication system with OAuth support.

## Implementation Details
1. Setup JWT token generation/validation
2. Implement bcrypt password hashing
3. Add OAuth provider integration
4. Create session management

## Key Components
- AuthService: Handles authentication logic
- TokenManager: Manages JWT tokens
- OAuthProvider: Integrates OAuth providers

## Considerations
- Security: Use bcrypt with salt rounds >= 12
- Performance: Cache JWT validation results
- Scalability: Stateless JWT design scales horizontally

## Testing Recommendations
- Unit tests for each component
- Integration tests for OAuth flow
- Security testing for authentication bypass
"""
            }
        elif "critique" in prompt.lower():
            return {
                "text": """## Overall Assessment
Score: 6/10
The solution covers basic requirements but misses some important aspects.

## Requirements Coverage
MET: JWT implementation
MET: Password hashing
NOT_MET: No rate limiting mentioned
NOT_MET: No account verification expiration

## Issues Identified
1. Missing rate limiting (severity: medium)
2. No input sanitization mentioned (severity: high)
3. Lacks password strength requirements (severity: medium)

## Strengths
- Good use of JWT
- Bcrypt for passwords
- Stateless design

## Recommendations
- Add rate limiting (10 requests/minute)
- Implement input validation
- Require 8+ char passwords with mixed case
"""
            }
        elif "verif" in prompt.lower():
            return {
                "text": """## Overall Assessment
Score: 8/10

## Requirements Verification
MET: JWT implementation
MET: Password hashing with bcrypt
MET: OAuth support
CONDITIONAL: Rate limiting mentioned in refinement

## Issue Resolution
Most Red Team concerns have been addressed in the refined solution.

## Quality Assessment
Completeness: 8/10
Correctness: 9/10
Efficiency: 8/10
Clarity: 9/10

## Final Verdict
VERIFIED_CONDITIONAL

## Recommendations
Solution is ready with minor recommendations for production deployment.
"""
            }
        else:
            return {"text": "Mock LLM response"}


@pytest.fixture
def mock_llm():
    """Create mock LLM client"""
    return MockLLMClient()


@pytest.fixture
def mock_storage():
    """Create mock storage manager"""
    storage = Mock()
    storage.store_artifact = AsyncMock(return_value="artifact_123")
    return storage


@pytest.fixture
def mock_retriever():
    """Create mock knowledge retriever"""
    retriever = Mock()
    retriever.retrieve_similar_solutions = AsyncMock(return_value=[])
    return retriever


@pytest.fixture
def blue_agent(mock_llm, mock_storage, mock_retriever):
    """Create Blue Team agent"""
    return BlueTeamAgent(
        crewai_client=mock_llm,
        storage_manager=mock_storage,
        knowledge_retriever=mock_retriever
    )


@pytest.fixture
def red_agent(mock_llm, mock_storage, mock_retriever):
    """Create Red Team agent"""
    return RedTeamAgent(
        crewai_client=mock_llm,
        storage_manager=mock_storage,
        knowledge_retriever=mock_retriever
    )


@pytest.fixture
def gold_agent(mock_llm, mock_storage, mock_retriever):
    """Create Gold Team agent"""
    return GoldTeamAgent(
        crewai_client=mock_llm,
        storage_manager=mock_storage,
        knowledge_retriever=mock_retriever
    )


@pytest.mark.asyncio
async def test_blue_agent_generate_solution(blue_agent):
    """Test Blue Team agent solution generation"""
    sub_problem = {
        "id": "sub_1",
        "title": "User Authentication",
        "description": "Implement secure user authentication",
        "requirements": ["JWT tokens", "OAuth support", "bcrypt hashing"]
    }

    result = await blue_agent.generate_solution(
        sub_problem=sub_problem,
        context={},
        use_rag=False
    )

    assert "solution" in result
    assert len(result["solution"]) > 0
    assert result["sub_problem_id"] == "sub_1"
    assert result["sub_problem_title"] == "User Authentication"


@pytest.mark.asyncio
async def test_red_agent_critique_solution(red_agent):
    """Test Red Team agent critique"""
    solution = "Implement JWT authentication with bcrypt passwords"

    sub_problem = {
        "id": "sub_1",
        "title": "User Authentication",
        "description": "Implement secure authentication",
        "requirements": ["JWT", "OAuth", "bcrypt"]
    }

    result = await red_agent.critique_solution(
        solution=solution,
        sub_problem=sub_problem,
        context={},
        use_patterns=False
    )

    assert "critique" in result
    assert len(result["critique"]) > 0
    assert result["total_issues"] >= 0


@pytest.mark.asyncio
async def test_gold_agent_verify_solution(gold_agent):
    """Test Gold Team agent verification"""
    solution = "Implement JWT with bcrypt and OAuth"

    sub_problem = {
        "id": "sub_1",
        "title": "User Authentication",
        "description": "Implement secure authentication",
        "requirements": ["JWT", "OAuth", "bcrypt"]
    }

    critique = {
        "issues": ["missing rate limiting"],
        "concerns": ["security"]
    }

    result = await gold_agent.verify_solution(
        solution=solution,
        critique=critique,
        sub_problem=sub_problem,
        context={}
    )

    assert "verification" in result
    assert "passes" in result
    assert result["sub_problem_id"] == "sub_1"


@pytest.mark.asyncio
async def test_agent_tool_usage():
    """Test agent using tools"""
    # Create mock tool
    mock_tool = Mock(spec=AgentTool)
    mock_tool.name = "test_tool"
    mock_tool.execute = AsyncMock(return_value="tool result")

    agent = BlueTeamAgent(
        crewai_client=None,
        tools=[mock_tool]
    )

    result = await agent.use_tool("test_tool", arg1="value1")

    assert result == "tool result"
    mock_tool.execute.assert_called_once_with(arg1="value1")


@pytest.mark.asyncio
async def test_a2a_protocol_send_message():
    """Test A2A protocol message sending"""
    protocol = A2AProtocol()

    message = await protocol.send_message(
        sender="blue_team",
        recipient="red_team",
        message_type=MessageType.SOLUTION_SUBMITTED,
        content="Solution submitted",
        sub_problem_id="sub_1",
        artifact_id="artifact_123"
    )

    assert message.sender == "blue_team"
    assert message.recipient == "red_team"
    assert message.message_type == MessageType.SOLUTION_SUBMITTED
    assert message.sub_problem_id == "sub_1"

    # Check message was queued
    messages = await protocol.get_messages("red_team", wait=False)
    assert len(messages) == 1
    assert messages[0].sender == "blue_team"


@pytest.mark.asyncio
async def test_a2a_protocol_send_reply():
    """Test A2A protocol reply sending"""
    protocol = A2AProtocol()

    # Send original message
    original = await protocol.send_message(
        sender="red_team",
        recipient="blue_team",
        message_type=MessageType.CRITIQUE_SUBMITTED,
        content="Critique",
        requires_response=True,
        sub_problem_id="sub_1"
    )

    # Send reply
    reply = await protocol.send_reply(
        original_message=original,
        reply_content="I've addressed the issues"
    )

    assert reply.sender == "blue_team"
    assert reply.recipient == "red_team"
    assert reply.reply_to == original.message_id


@pytest.mark.asyncio
async def test_a2a_protocol_broadcast():
    """Test A2A protocol broadcast"""
    protocol = A2AProtocol()

    messages = await protocol.broadcast(
        sender="orchestrator",
        recipients=["blue_team", "red_team", "gold_team"],
        message_type=MessageType.STATUS_UPDATE,
        content="Status update"
    )

    assert len(messages) == 3

    # Check each recipient got the message
    for agent in ["blue_team", "red_team", "gold_team"]:
        agent_messages = await protocol.get_messages(agent, wait=False)
        assert len(agent_messages) == 1


@pytest.mark.asyncio
async def test_message_builder():
    """Test MessageBuilder convenience methods"""
    # Test solution submitted message
    solution_msg = MessageBuilder.solution_submitted(
        sender="blue_team",
        recipient="red_team",
        solution="Solution content",
        sub_problem_id="sub_1",
        artifact_id="artifact_123"
    )

    assert solution_msg.message_type == MessageType.SOLUTION_SUBMITTED
    assert solution_msg.sub_problem_id == "sub_1"

    # Test critique submitted message
    critique_msg = MessageBuilder.critique_submitted(
        sender="red_team",
        recipient="blue_team",
        critique="Critique content",
        issues=["issue1", "issue2"],
        sub_problem_id="sub_1",
        artifact_id="artifact_123"
    )

    assert critique_msg.message_type == MessageType.CRITIQUE_SUBMITTED
    assert critique_msg.metadata["issues_count"] == 2

    # Test refinement request
    refinement_msg = MessageBuilder.refinement_request(
        sender="red_team",
        recipient="blue_team",
        issues=["security", "performance"],
        sub_problem_id="sub_1",
        artifact_id="artifact_123"
    )

    assert refinement_msg.message_type == MessageType.REFINEMENT_REQUEST
    assert refinement_msg.requires_response is True


@pytest.mark.asyncio
async def test_agent_metadata():
    """Test agent metadata"""
    agent = BlueTeamAgent(
        crewai_client=None
    )

    metadata = agent.get_metadata()

    assert metadata["role"] == "blue_team"
    assert "agent_id" in metadata
    assert "model_id" in metadata
    assert "tools_available" in metadata


@pytest.mark.asyncio
async def test_agent_conversation_history():
    """Test conversation history tracking"""
    agent = BlueTeamAgent(
        crewai_client=None
    )

    # Add some conversation turns
    agent.conversation_history.append({
        "role": "user",
        "content": "Task 1"
    })
    agent.conversation_history.append({
        "role": "assistant",
        "content": "Response 1"
    })

    history = agent.get_conversation_history()

    assert len(history) == 2
    assert history[0]["role"] == "user"


@pytest.mark.asyncio
async def test_message_serialization():
    """Test message to_dict and from_dict"""
    original = A2AMessage(
        sender="blue_team",
        recipient="red_team",
        message_type=MessageType.SOLUTION_SUBMITTED,
        content="Test message",
        sub_problem_id="sub_1"
    )

    # Convert to dict
    data = original.to_dict()

    # Convert back to message
    restored = A2AMessage.from_dict(data)

    assert restored.sender == original.sender
    assert restored.recipient == original.recipient
    assert restored.message_type == original.message_type
    assert restored.content == original.content


if __name__ == "__main__":
    # Run tests manually
    import sys

    async def run_tests():
        print("Running agent coordination tests...\n")

        tests = [
            ("Blue Agent Generate Solution", test_blue_agent_generate_solution),
            ("Red Agent Critique", test_red_agent_critique_solution),
            ("Gold Agent Verify", test_gold_agent_verify_solution),
            ("A2A Send Message", test_a2a_protocol_send_message),
            ("A2A Reply", test_a2a_protocol_send_reply),
            ("A2A Broadcast", test_a2a_protocol_broadcast),
            ("Message Builder", test_message_builder),
        ]

        passed = 0
        failed = 0

        for name, test_func in tests:
            try:
                # Create fixtures manually for standalone execution
                mock_llm = MockLLMClient()
                mock_storage = Mock()
                mock_storage.store_artifact = AsyncMock(return_value="artifact_123")
                mock_retriever = Mock()
                mock_retriever.retrieve_similar_solutions = AsyncMock(return_value=[])

                if "blue" in name.lower():
                    await test_func(BlueTeamAgent(mock_llm, mock_storage, mock_retriever))
                elif "red" in name.lower():
                    await test_func(RedTeamAgent(mock_llm, mock_storage, mock_retriever))
                elif "gold" in name.lower():
                    await test_func(GoldTeamAgent(mock_llm, mock_storage, mock_retriever))
                else:
                    await test_func()

                passed += 1
                print(f"✅ PASSED: {name}")
            except Exception as e:
                failed += 1
                print(f"❌ FAILED: {name}")
                print(f"   Error: {e}")

        print(f"\n{'='*70}")
        print(f"Passed: {passed}/{passed + failed}")
        print('='*70)

        if failed > 0:
            sys.exit(1)

    asyncio.run(run_tests())
