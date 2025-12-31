"""
LeanAide Client Test Suite

Comprehensive tests for the LeanAide async client.
Tests connection, all task types, error handling, and edge cases.
"""

import asyncio
import pytest
import json
from unittest.mock import AsyncMock, MagicMock, patch

from leanaide_client import (
    LeanAideClient,
    LeanAideConfig,
    LeanAideResult,
    TaskType,
    ConnectionError,
    TimeoutError,
    TaskExecutionError,
    ValidationError,
    create_client
)


# ========== Fixtures ==========

@pytest.fixture
async def client():
    """Create a client instance for testing."""
    config = LeanAideConfig(
        host="localhost",
        port=7654,
        timeout=10.0,
        max_retries=2
    )
    return LeanAideClient(config=config)


@pytest.fixture
def mock_response():
    """Create a mock response object."""
    response = AsyncMock()
    response.status = 200
    response.headers = {"Content-Type": "application/json"}
    return response


# ========== Connection Tests ==========

@pytest.mark.asyncio
async def test_client_context_manager():
    """Test that client works as async context manager."""
    async with LeanAideClient() as client:
        assert client is not None
        assert not client._closed
    assert client._closed


@pytest.mark.asyncio
async def test_health_check_success(client, mock_response):
    """Test successful health check."""
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="OK")

    with patch.object(client.session, 'get', return_value=mock_response):
        result = await client.health_check()
        assert result is True


@pytest.mark.asyncio
async def test_health_check_failure(client, mock_response):
    """Test failed health check."""
    with patch.object(client.session, 'get', side_effect=Exception("Connection refused")):
        result = await client.health_check()
        assert result is False


# ========== Task Execution Tests ==========

@pytest.mark.asyncio
async def test_translate_thm(client, mock_response):
    """Test translate_thm task."""
    mock_response.json = AsyncMock(return_value={
        "result": "theorem infinitely_many_primes : Infinite {p : Nat | Prime p}",
        "logs": "[info] Translation complete"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_thm("There are infinitely many primes")

        assert result.success is True
        assert result.task == "translate_thm"
        assert result.data is not None
        assert result.response_time > 0


@pytest.mark.asyncio
async def test_translate_thm_detailed(client, mock_response):
    """Test translate_thm_detailed task."""
    mock_response.json = AsyncMock(return_value={
        "name": "infinitely_many_primes",
        "type": "Infinite {p : Nat | Prime p}",
        "statement": "theorem infinitely_many_primes : Infinite {p : Nat | Prime p} := by sorry"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_thm_detailed(
            "There are infinitely many primes",
            theorem_name="infinitely_many_primes"
        )

        assert result.success is True
        assert result.task == "translate_thm_detailed"


@pytest.mark.asyncio
async def test_translate_def(client, mock_response):
    """Test translate_def task."""
    mock_response.json = AsyncMock(return_value={
        "result": "def cube_free (n : Nat) : Prop := ¬∃ p : Nat, Prime p ∧ p^3 ∣ n"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_def(
            "A number is cube-free if it is not divisible by the cube of any prime"
        )

        assert result.success is True
        assert result.task == "translate_def"


@pytest.mark.asyncio
async def test_theorem_doc(client, mock_response):
    """Test theorem_doc task."""
    mock_response.json = AsyncMock(return_value={
        "result": "This theorem states that there are infinitely many prime numbers..."
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.theorem_doc(
            theorem_name="infinitely_many_primes",
            theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
        )

        assert result.success is True
        assert result.task == "theorem_doc"


@pytest.mark.asyncio
async def test_def_doc(client, mock_response):
    """Test def_doc task."""
    mock_response.json = AsyncMock(return_value={
        "result": "A natural number is cube-free if no prime cubed divides it..."
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.def_doc(
            definition_name="cube_free",
            definition_code="def cube_free (n : Nat) : Prop := ¬∃ p : Nat, Prime p ∧ p^3 ∣ n"
        )

        assert result.success is True
        assert result.task == "def_doc"


@pytest.mark.asyncio
async def test_theorem_name(client, mock_response):
    """Test theorem_name task."""
    mock_response.json = AsyncMock(return_value={
        "result": "infinitely_many_primes"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.theorem_name("There are infinitely many primes")

        assert result.success is True
        assert result.task == "theorem_name"


@pytest.mark.asyncio
async def test_prove_for_formalization(client, mock_response):
    """Test prove_for_formalization task."""
    mock_response.json = AsyncMock(return_value={
        "result": "Proof: We proceed by contradiction. Assume there are finitely many primes..."
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.prove_for_formalization(
            theorem_text="There are infinitely many primes",
            theorem_code="Infinite {p : Nat | Prime p}",
            theorem_statement="theorem infinitely_many_primes : Infinite {p : Nat | Prime p}"
        )

        assert result.success is True
        assert result.task == "prove_for_formalization"


@pytest.mark.asyncio
async def test_json_structured(client, mock_response):
    """Test json_structured task."""
    mock_response.json = AsyncMock(return_value={
        "result": {
            "title": "Infinite Primes Theorem",
            "statements": [...],
            "proofs": [...]
        }
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.json_structured(
            "Theorem: There are infinitely many primes. Proof: ..."
        )

        assert result.success is True
        assert result.task == "json_structured"


@pytest.mark.asyncio
async def test_lean_from_json_structured(client, mock_response):
    """Test lean_from_json_structured task."""
    mock_response.json = AsyncMock(return_value={
        "result": "theorem infinitely_many_primes : Infinite {p : Nat | Prime p} := by..."
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.lean_from_json_structured({
            "title": "Infinite Primes",
            "content": "..."
        })

        assert result.success is True
        assert result.task == "lean_from_json_structured"


@pytest.mark.asyncio
async def test_elaborate(client, mock_response):
    """Test elaborate task."""
    mock_response.json = AsyncMock(return_value={
        "declarations": ["infinitely_many_primes"],
        "logs": ["Elaborating...", "Done"],
        "sorries": [],
        "sorriesAfterPurge": []
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.elaborate(
            "theorem infinitely_many_primes : Infinite {p : Nat | Prime p} := by sorry"
        )

        assert result.success is True
        assert result.task == "elaborate"


@pytest.mark.asyncio
async def test_math_query(client, mock_response):
    """Test math_query task."""
    mock_response.json = AsyncMock(return_value={
        "result": [
            "The fundamental theorem of algebra states that...",
            "Every non-constant polynomial has a root in complex numbers..."
        ]
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.math_query(
            "What is the fundamental theorem of algebra?",
            n=2
        )

        assert result.success is True
        assert result.task == "math_query"


@pytest.mark.asyncio
async def test_math_query_with_history(client, mock_response):
    """Test math_query task with conversation history."""
    mock_response.json = AsyncMock(return_value={
        "result": ["Based on our previous discussion..."]
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.math_query(
            "Can you explain more?",
            history=[
                {"role": "user", "content": "What is a group?"},
                {"role": "assistant", "content": "A group is..."}
            ],
            n=1
        )

        assert result.success is True
        assert result.task == "math_query"


# ========== Error Handling Tests ==========

@pytest.mark.asyncio
async def test_task_execution_error(client, mock_response):
    """Test handling of task execution errors."""
    mock_response.status = 500
    mock_response.json = AsyncMock(return_value={
        "error": "Internal server error",
        "logs": "[error] Process crashed"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_thm("test")

        assert result.success is False
        assert result.error is not None


@pytest.mark.asyncio
async def test_validation_error(client, mock_response):
    """Test handling of validation errors."""
    mock_response.status = 400
    mock_response.json = AsyncMock(return_value={
        "error": "Missing required field: theorem_text"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_thm("")

        assert result.success is False
        assert "error" in str(result.error).lower()


@pytest.mark.asyncio
async def test_timeout_error(client, mock_response):
    """Test handling of timeout errors."""
    mock_response.status = 504
    mock_response.json = AsyncMock(return_value={
        "error": "Process timed out"
    })

    with patch.object(client.session, 'post', return_value=mock_response):
        result = await client.translate_thm("test")

        assert result.success is False
        assert "timeout" in str(result.error).lower()


@pytest.mark.asyncio
async def test_retry_logic(client, mock_response):
    """Test that client retries on failure."""
    # First two attempts fail, third succeeds
    call_count = [0]

    async def side_effect(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] < 3:
            raise Exception("Temporary error")
        mock_response.json = AsyncMock(return_value={"result": "success"})
        return mock_response

    with patch.object(client.session, 'post', side_effect=side_effect):
        result = await client.translate_thm("test")

        assert call_count[0] == 3  # Should have retried
        assert result.success is True


@pytest.mark.asyncio
async def test_max_retries_exceeded(client, mock_response):
    """Test that client gives up after max retries."""
    # Always fail
    async def side_effect(*args, **kwargs):
        raise Exception("Persistent error")

    with patch.object(client.session, 'post', side_effect=side_effect):
        result = await client.translate_thm("test")

        assert result.success is False
        assert "retry" in str(result.error).lower() or "exceeded" in str(result.error).lower()


# ========== Batch Operations Tests ==========

@pytest.mark.asyncio
async def test_batch_translate_theorems(client, mock_response):
    """Test batch theorem translation."""
    mock_response.json = AsyncMock(return_value={"result": "success"})

    with patch.object(client.session, 'post', return_value=mock_response):
        theorems = [
            "There are infinitely many primes",
            "The square root of 2 is irrational"
        ]
        results = await client.batch_translate_theorems(theorems)

        assert len(results) == 2
        assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_batch_translate_definitions(client, mock_response):
    """Test batch definition translation."""
    mock_response.json = AsyncMock(return_value={"result": "success"})

    with patch.object(client.session, 'post', return_value=mock_response):
        definitions = [
            "A prime number is...",
            "A perfect square is..."
        ]
        results = await client.batch_translate_definitions(definitions)

        assert len(results) == 2
        assert all(r.success for r in results)


@pytest.mark.asyncio
async def test_execute_parallel_tasks(client, mock_response):
    """Test parallel task execution."""
    mock_response.json = AsyncMock(return_value={"result": "success"})

    with patch.object(client.session, 'post', return_value=mock_response):
        tasks = [
            {"task": "translate_thm", "theorem_text": "Test 1"},
            {"task": "translate_thm", "theorem_text": "Test 2"},
            {"task": "translate_def", "definition_text": "Test 3"}
        ]
        results = await client.execute_parallel_tasks(tasks)

        assert len(results) == 3
        assert all(r.success for r in results)


# ========== Result Data Class Tests ==========

def test_lean_aide_result_to_dict():
    """Test LeanAideResult serialization."""
    result = LeanAideResult(
        success=True,
        task="translate_thm",
        data={"result": "test"},
        logs="test logs",
        response_time=1.5
    )

    result_dict = result.to_dict()

    assert result_dict["success"] is True
    assert result_dict["task"] == "translate_thm"
    assert result_dict["data"]["result"] == "test"
    assert result_dict["logs"] == "test logs"
    assert result_dict["response_time"] == 1.5
    assert "timestamp" in result_dict


# ========== Configuration Tests ==========

def test_config_base_url():
    """Test LeanAideConfig base URL generation."""
    config = LeanAideConfig(host="localhost", port=7654)
    assert config.base_url == "http://localhost:7654"

    config_https = LeanAideConfig(
        host="example.com",
        port=443,
        verify_ssl=True
    )
    assert config_https.base_url == "https://example.com:443"


def test_config_defaults():
    """Test LeanAideConfig default values."""
    config = LeanAideConfig()
    assert config.host == "localhost"
    assert config.port == 7654
    assert config.timeout == 6000.0
    assert config.max_retries == 3
    assert config.max_connections == 100


# ========== Factory Function Tests ==========

@pytest.mark.asyncio
async def test_create_client_factory():
    """Test create_client factory function."""
    client = await create_client(host="localhost", port=7654)
    assert client.config.host == "localhost"
    assert client.config.port == 7654
    await client.close()


# ========== Integration Tests ==========

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_workflow_integration():
    """
    Integration test: Full workflow with actual server.
    This test requires a running LeanAide server.
    Run with: pytest test_leanaide_client.py -m integration
    """
    async with LeanAideClient() as client:
        # Health check
        is_healthy = await client.health_check()
        if not is_healthy:
            pytest.skip("LeanAide server not running")

        # Translate theorem
        result = await client.translate_thm(
            "There are infinitely many prime numbers"
        )
        assert result.success or result.error is not None

        # Elaborate code
        if result.success and result.data:
            code_result = await client.elaborate(
                result.data.get("result", "")
            )
            assert code_result.success or code_result.error is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
