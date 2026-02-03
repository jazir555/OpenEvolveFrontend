"""
Mock LLM Client for testing without API calls

This module provides a mock implementation of the LLM client that returns
sensible responses for testing purposes, avoiding expensive API calls.
"""

import asyncio
import json
import random
from typing import Any, AsyncIterator, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class MockLLMResponse:
    """Mock response from LLM"""
    content: str
    model: str = "mock-model"
    finish_reason: str = "stop"
    usage: Optional[Dict[str, int]] = None

    def __post_init__(self):
        if self.usage is None:
            self.usage = {
                "prompt_tokens": 100,
                "completion_tokens": 200,
                "total_tokens": 300
            }


class MockLLMClient:
    """
    Mock LLM client for testing without API calls

    Returns sensible mock responses based on the prompt content, allowing
    tests to run without expensive API calls while still exercising the
    evolution logic.

    Example:
        >>> client = MockLLMClient()
        >>> response = await client.generate("Improve this function")
        >>> assert response.content is not None
    """

    def __init__(
        self,
        model_name: str = "mock-model",
        latency_ms: int = 0,
        error_rate: float = 0.0,
        response_quality: str = "good"  # "good", "medium", "poor"
    ):
        """
        Initialize mock LLM client

        Args:
            model_name: Name of the mock model
            latency_ms: Simulated latency in milliseconds
            error_rate: Probability of returning an error (0.0 to 1.0)
            response_quality: Quality of mock responses ("good", "medium", "poor")
        """
        self.model_name = model_name
        self.latency_ms = latency_ms
        self.error_rate = error_rate
        self.response_quality = response_quality
        self.call_count = 0

    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> MockLLMResponse:
        """
        Generate a mock response based on prompt content

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (ignored in mock)
            max_tokens: Maximum tokens (ignored in mock)
            **kwargs: Additional parameters

        Returns:
            MockLLMResponse with sensible content
        """
        self.call_count += 1

        # Simulate latency
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000.0)

        # Simulate errors
        if random.random() < self.error_rate:
            raise Exception(f"Mock LLM error (simulated error rate: {self.error_rate})")

        # Generate response based on prompt content
        content = self._generate_mock_response(prompt)

        return MockLLMResponse(
            content=content,
            model=self.model_name
        )

    async def generate_stream(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Generate a streaming mock response

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Yields:
            Chunks of the response
        """
        response = await self.generate(prompt, temperature, max_tokens, **kwargs)

        # Split response into chunks
        chunk_size = 50
        for i in range(0, len(response.content), chunk_size):
            chunk = response.content[i:i + chunk_size]
            yield chunk
            await asyncio.sleep(0.01)  # Small delay between chunks

    def _generate_mock_response(self, prompt: str) -> str:
        """
        Generate a sensible mock response based on prompt content

        Args:
            prompt: The input prompt

        Returns:
            Mock response content
        """
        prompt_lower = prompt.lower()

        # Code improvement prompts
        if any(word in prompt_lower for word in ["improve", "optimize", "better", "enhance"]):
            return self._generate_improvement_response(prompt)

        # Planning prompts
        if any(word in prompt_lower for word in ["plan", "strategy", "approach"]):
            return self._generate_planning_response(prompt)

        # Analysis prompts
        if any(word in prompt_lower for word in ["analyze", "why", "explain"]):
            return self._generate_analysis_response(prompt)

        # Default: code evolution response
        return self._generate_code_response(prompt)

    def _generate_improvement_response(self, prompt: str) -> str:
        """Generate mock code improvement response"""
        if self.response_quality == "good":
            return '''```python
def improved_function(x, y):
    """
    Improved version with better error handling and optimization
    """
    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
        raise ValueError("Both x and y must be numeric")

    # Optimized computation
    result = x * x + y * y
    return result

# Test cases
assert improved_function(3, 4) == 25
assert improved_function(0, 0) == 0
```'''
        elif self.response_quality == "medium":
            return '''```python
def improved_function(x, y):
    result = x * x + y * y
    return result
```'''
        else:  # poor
            return '```python\ndef func(x, y):\n    return x * x\n```'

    def _generate_planning_response(self, prompt: str) -> str:
        """Generate mock planning response"""
        return '''## Optimization Strategy

1. **Initial Analysis**: Understand the problem constraints and objective function
2. **Parameter Exploration**: Systematically explore the parameter space
3. **Gradient-Based Optimization**: Use gradient information for convergence
4. **Fine-Tuning**: Refine solution with local search
5. **Validation**: Verify solution quality on test cases

Key improvements:
- Better initial parameter selection
- Adaptive learning rate
- Early stopping based on convergence
- Robust error handling'''

    def _generate_analysis_response(self, prompt: str) -> str:
        """Generate mock analysis response"""
        return '''## Analysis

**Current Performance**: The function achieves a score of 0.85

**Key Findings**:
1. The algorithm performs well on simple cases but struggles with edge cases
2. Time complexity is O(n²) which could be optimized
3. Memory usage is acceptable
4. Numerical stability issues at extreme values

**Recommendations**:
- Add input validation
- Optimize the inner loop
- Use more numerically stable operations
- Add comprehensive test coverage'''

    def _generate_code_response(self, prompt: str) -> str:
        """Generate mock code evolution response"""
        # Try to extract existing code from prompt and improve it
        return '''```python
def evolved_solution(param1, param2):
    """
    Evolved solution with improved algorithm
    """
    # Better parameter handling
    validated_params = validate_inputs(param1, param2)

    # Optimized core algorithm
    result = compute_optimized(validated_params)

    return result

def validate_inputs(p1, p2):
    """Validate and normalize inputs"""
    return max(0, min(1, p1)), max(0, min(1, p2))

def compute_optimized(params):
    """Optimized computation"""
    p1, p2 = params
    return p1 * p1 + p2 * p2 - 0.5 * p1 * p2
```'''

    def reset(self):
        """Reset call counter"""
        self.call_count = 0


class MockLLMEnsemble:
    """
    Mock ensemble of LLM clients for testing

    Simulates multiple LLM instances with different characteristics
    """

    def __init__(self, num_models: int = 3):
        """
        Initialize mock ensemble

        Args:
            num_models: Number of mock models in ensemble
        """
        self.models = [
            MockLLMClient(
                model_name=f"mock-model-{i}",
                response_quality=["good", "medium", "poor"][i % 3]
            )
            for i in range(num_models)
        ]

    async def generate_ensemble(
        self,
        prompt: str,
        num_responses: int = 1,
        **kwargs
    ) -> List[MockLLMResponse]:
        """
        Generate responses from multiple models

        Args:
            prompt: Input prompt
            num_responses: Number of responses per model
            **kwargs: Additional parameters

        Returns:
            List of responses from ensemble
        """
        responses = []
        for model in self.models:
            for _ in range(num_responses):
                response = await model.generate(prompt, **kwargs)
                responses.append(response)
        return responses

    def get_model(self, index: int) -> MockLLMClient:
        """Get specific model from ensemble"""
        return self.models[index % len(self.models)]
