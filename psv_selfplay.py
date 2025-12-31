"""
PSV (Propose-Solve-Verify) Self-Play System

Production-ready implementation of mathematical self-play learning where:
- Proposer Agent: Generates new mathematical problems
- Solver Agent: Attempts to solve the problems
- Verifier Agent: Verifies the correctness of solutions

All agents use real LLM API calls with complete working logic.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
from datetime import datetime
import re
import threading
import queue
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MathematicalProblem:
    """A mathematical problem in the PSV system"""
    id: str
    statement: str
    domain: str  # "algebra", "geometry", "number_theory", "calculus", etc.
    difficulty: float  # 0.0 to 1.0
    expected_difficulty: float = 0.5
    created_at: float = field(default_factory=time.time)
    proposed_by: str = "system"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionAttempt:
    """A solution attempt for a mathematical problem"""
    problem_id: str
    solution: str
    solver_id: str
    timestamp: float = field(default_factory=time.time)
    solving_time_seconds: float = 0.0
    approach: str = "unknown"
    confidence: float = 0.5


@dataclass
class VerificationResult:
    """Result of verifying a solution"""
    problem_id: str
    solution_id: str
    is_correct: bool
    confidence: float
    feedback: str
    verification_time: float = 0.0
    specific_errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)
    verifier_id: str = "system"


@dataclass
class PSVEpisode:
    """A complete PSV episode (problem -> solution -> verification)"""
    episode_id: str
    problem: MathematicalProblem
    solution: SolutionAttempt
    verification: VerificationResult
    timestamp: float = field(default_factory=time.time)
    learning_outcome: str = "unknown"  # "success", "failure", "partial"


class LLMProvider(Enum):
    """Available LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class PSVConfig:
    """Configuration for PSV system"""

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        anthropic_api_key: Optional[str] = None,
        custom_api_base: Optional[str] = None,
        custom_api_key: Optional[str] = None,
        default_provider: LLMProvider = LLMProvider.OPENAI,
        default_model: str = "gpt-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 120
    ):
        self.openai_api_key = openai_api_key
        self.anthropic_api_key = anthropic_api_key
        self.custom_api_base = custom_api_base
        self.custom_api_key = custom_api_key
        self.default_provider = default_provider
        self.default_model = default_model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout


class LLMClient:
    """Real LLM API client for making actual requests"""

    def __init__(self, config: PSVConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)

    async def generate_completion(
        self,
        prompt: str,
        provider: Optional[LLMProvider] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a completion using the configured LLM provider"""
        provider = provider or self.config.default_provider
        model = model or self.config.default_model
        temperature = temperature if temperature is not None else self.config.temperature
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens

        if provider == LLMProvider.OPENAI:
            return await self._generate_openai(prompt, model, temperature, max_tokens)
        elif provider == LLMProvider.ANTHROPIC:
            return await self._generate_anthropic(prompt, model, temperature, max_tokens)
        elif provider == LLMProvider.OPENAI_COMPATIBLE:
            return await self._generate_custom(prompt, model, temperature, max_tokens)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _generate_openai(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate completion using OpenAI API"""
        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not configured")

        headers = {
            "Authorization": f"Bearer {self.config.openai_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a mathematical expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    async def _generate_anthropic(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate completion using Anthropic API"""
        if not self.config.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")

        headers = {
            "x-api-key": self.config.anthropic_api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }

        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        data = response.json()

        return data["content"][0]["text"]

    async def _generate_custom(self, prompt: str, model: str, temperature: float, max_tokens: int) -> str:
        """Generate completion using custom OpenAI-compatible API"""
        if not self.config.custom_api_base or not self.config.custom_api_key:
            raise ValueError("Custom API configuration incomplete")

        headers = {
            "Authorization": f"Bearer {self.config.custom_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a mathematical expert."},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        response = await self.client.post(
            f"{self.config.custom_api_base}/chat/completions",
            headers=headers,
            json=payload
        )

        response.raise_for_status()
        data = response.json()

        return data["choices"][0]["message"]["content"]

    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()


class MathematicalProblemProposer:
    """Agent that proposes new mathematical problems"""

    def __init__(self, llm_client: LLMClient, config: PSVConfig):
        self.llm_client = llm_client
        self.config = config
        self.domains = [
            "algebra", "geometry", "number_theory", "calculus",
            "linear_algebra", "probability", "statistics", "combinatorics"
        ]
        self.difficulty_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

    async def propose_problem(
        self,
        domain: Optional[str] = None,
        target_difficulty: Optional[float] = None,
        previous_problems: Optional[List[MathematicalProblem]] = None
    ) -> MathematicalProblem:
        """Propose a new mathematical problem"""
        # Select domain if not specified
        if domain is None:
            domain = self._select_domain(previous_problems)

        # Select difficulty if not specified
        if target_difficulty is None:
            target_difficulty = self._select_difficulty(previous_problems)

        # Generate problem statement
        problem_statement = await self._generate_problem_statement(domain, target_difficulty, previous_problems)

        # Create problem object
        problem = MathematicalProblem(
            id=str(uuid.uuid4()),
            statement=problem_statement,
            domain=domain,
            difficulty=target_difficulty,
            expected_difficulty=target_difficulty,
            proposed_by="MathematicalProblemProposer",
            tags=[domain],
            metadata={
                "generation_timestamp": time.time(),
                "target_difficulty": target_difficulty
            }
        )

        logger.info(f"Proposed problem {problem.id} in {domain} at difficulty {target_difficulty}")
        return problem

    def _select_domain(self, previous_problems: Optional[List[MathematicalProblem]]) -> str:
        """Select a domain using diversity-aware sampling"""
        if not previous_problems:
            return "algebra"

        # Count domain usage
        domain_counts = defaultdict(int)
        for problem in previous_problems:
            domain_counts[problem.domain] += 1

        # Select least used domain
        least_used = min(domain_counts.keys(), key=lambda d: domain_counts[d])
        return least_used

    def _select_difficulty(self, previous_problems: Optional[List[MathematicalProblem]]) -> float:
        """Select difficulty using adaptive curriculum"""
        if not previous_problems:
            return 0.5

        # Calculate success rate of recent problems
        recent_problems = previous_problems[-10:]
        success_count = sum(1 for p in recent_problems if p.metadata.get("solved", False))
        success_rate = success_count / len(recent_problems)

        # Adjust difficulty based on success rate
        if success_rate > 0.8:
            # Increase difficulty
            return min(0.9, 0.5 + 0.1)
        elif success_rate < 0.4:
            # Decrease difficulty
            return max(0.1, 0.5 - 0.1)
        else:
            return 0.5

    async def _generate_problem_statement(
        self,
        domain: str,
        difficulty: float,
        previous_problems: Optional[List[MathematicalProblem]]
    ) -> str:
        """Generate the actual problem statement using LLM"""
        difficulty_description = self._describe_difficulty(difficulty)

        prompt = f"""Generate a {difficulty_description} mathematical problem in the domain of {domain}.

Requirements:
1. The problem should be clearly stated and unambiguous
2. It should require mathematical reasoning to solve
3. It should have a definite correct answer
4. The difficulty should be approximately {difficulty:.1f} on a scale of 0.0 to 1.0
5. Provide only the problem statement, no solution

Example format:
"Find all integer solutions (x, y) to the equation x^2 + y^2 = 25 where x and y are positive."

Generate a problem:"""

        problem_statement = await self.llm_client.generate_completion(prompt)

        # Clean up the response
        problem_statement = problem_statement.strip()

        # Remove common prefixes
        for prefix in ["Problem:", "Question:", "Here is a problem:", "Here's a problem:"]:
            if problem_statement.startswith(prefix):
                problem_statement = problem_statement[len(prefix):].strip()

        return problem_statement

    def _describe_difficulty(self, difficulty: float) -> str:
        """Convert difficulty score to description"""
        if difficulty < 0.2:
            return "very easy"
        elif difficulty < 0.4:
            return "easy"
        elif difficulty < 0.6:
            return "medium"
        elif difficulty < 0.8:
            return "hard"
        else:
            return "very hard"


class MathematicalProblemSolver:
    """Agent that solves mathematical problems"""

    def __init__(self, llm_client: LLMClient, config: PSVConfig):
        self.llm_client = llm_client
        self.config = config

    async def solve_problem(
        self,
        problem: MathematicalProblem,
        max_attempts: int = 3
    ) -> SolutionAttempt:
        """Attempt to solve a mathematical problem"""
        logger.info(f"Solving problem {problem.id} in {problem.domain}")

        start_time = time.time()

        # Generate solution
        prompt = self._build_solution_prompt(problem)

        solution_text = await self.llm_client.generate_completion(prompt)

        solving_time = time.time() - start_time

        # Parse solution
        solution = self._parse_solution(solution_text)

        # Create solution attempt
        attempt = SolutionAttempt(
            problem_id=problem.id,
            solution=solution,
            solver_id="MathematicalProblemSolver",
            solving_time_seconds=solving_time,
            approach=self._detect_approach(solution_text),
            confidence=self._estimate_confidence(solution_text)
        )

        logger.info(f"Generated solution for problem {problem.id} in {solving_time:.2f}s")
        return attempt

    def _build_solution_prompt(self, problem: MathematicalProblem) -> str:
        """Build the prompt for solution generation"""
        prompt = f"""Solve the following mathematical problem:

Domain: {problem.domain}
Difficulty: {problem.difficulty:.1f}/1.0

Problem Statement:
{problem.statement}

Instructions:
1. Show your work step by step
2. Explain your reasoning clearly
3. Provide the final answer
4. Be thorough and precise

Solution:"""

        return prompt

    def _parse_solution(self, solution_text: str) -> str:
        """Parse and clean the solution text"""
        # Remove common prefixes
        prefixes = ["Solution:", "Answer:", "Here is the solution:", "The solution is:"]
        for prefix in prefixes:
            if solution_text.startswith(prefix):
                solution_text = solution_text[len(prefix):].strip()

        return solution_text.strip()

    def _detect_approach(self, solution_text: str) -> str:
        """Detect the approach used in the solution"""
        solution_lower = solution_text.lower()

        approaches = {
            "algebraic_manipulation": ["substitute", "rearrange", "solve for", "equation"],
            "geometric_reasoning": ["triangle", "circle", "angle", "prove", "geometric"],
            "induction": ["base case", "inductive step", "induction", "assume for k"],
            "contradiction": ["assume opposite", "contradiction", "leads to contradiction"],
            "calculus": ["derivative", "integral", "limit", "differentiate", "integrate"],
            "combinatorial": ["count", "combination", "permutation", "binomial"]
        }

        best_approach = "general"
        max_matches = 0

        for approach, keywords in approaches.items():
            matches = sum(1 for kw in keywords if kw in solution_lower)
            if matches > max_matches:
                max_matches = matches
                best_approach = approach

        return best_approach

    def _estimate_confidence(self, solution_text: str) -> float:
        """Estimate confidence in the solution based on textual indicators"""
        solution_lower = solution_text.lower()

        confidence_indicators = {
            "high": ["therefore", "we conclude", "the answer is", "solution is", "final answer"],
            "medium": ["thus", "so", "hence", "we get"],
            "low": ["maybe", "probably", "might be", "could be", "not sure"]
        }

        high_count = sum(1 for ind in confidence_indicators["high"] if ind in solution_lower)
        low_count = sum(1 for ind in confidence_indicators["low"] if ind in solution_lower)

        if high_count > 0:
            return min(0.95, 0.7 + high_count * 0.1)
        elif low_count > 0:
            return max(0.3, 0.6 - low_count * 0.1)
        else:
            return 0.6


class MathematicalProblemVerifier:
    """Agent that verifies solutions to mathematical problems"""

    def __init__(self, llm_client: LLMClient, config: PSVConfig):
        self.llm_client = llm_client
        self.config = config

    async def verify_solution(
        self,
        problem: MathematicalProblem,
        solution: SolutionAttempt
    ) -> VerificationResult:
        """Verify a solution to a problem"""
        logger.info(f"Verifying solution for problem {problem.id}")

        start_time = time.time()

        # Build verification prompt
        prompt = self._build_verification_prompt(problem, solution)

        # Get verification response
        verification_text = await self.llm_client.generate_completion(prompt)

        verification_time = time.time() - start_time

        # Parse verification result
        result = self._parse_verification_result(problem.id, solution, verification_text, verification_time)

        logger.info(f"Verification {'passed' if result.is_correct else 'failed'} for problem {problem.id}")
        return result

    def _build_verification_prompt(self, problem: MathematicalProblem, solution: SolutionAttempt) -> str:
        """Build the verification prompt"""
        prompt = f"""Verify the following solution to a mathematical problem:

Problem Statement:
{problem.statement}

Proposed Solution:
{solution.solution}

Instructions:
1. Carefully check the solution step by step
2. Identify any logical errors or mistakes
3. Verify the final answer
4. Provide specific feedback on any issues

Respond in the following format:

CORRECTNESS: [correct/incorrect/partially_correct]
CONFIDENCE: [0.0 to 1.0]
FEEDBACK: [Your detailed feedback]
ERRORS: [List any specific errors found, or "none"]
SUGGESTIONS: [List any improvement suggestions, or "none"]

Verification:"""

        return prompt

    def _parse_verification_result(
        self,
        problem_id: str,
        solution: SolutionAttempt,
        verification_text: str,
        verification_time: float
    ) -> VerificationResult:
        """Parse the verification response"""
        # Extract components
        correctness = self._extract_field(verification_text, "CORRECTNESS")
        confidence_str = self._extract_field(verification_text, "CONFIDENCE")
        feedback = self._extract_field(verification_text, "FEEDBACK")
        errors_str = self._extract_field(verification_text, "ERRORS")
        suggestions_str = self._extract_field(verification_text, "SUGGESTIONS")

        # Parse correctness
        is_correct = correctness.lower() in ["correct", "correctness: correct", "correct: true"]
        is_partial = "partial" in correctness.lower()

        # Parse confidence
        try:
            confidence = float(confidence_str.strip())
        except (ValueError, AttributeError):
            confidence = 0.6

        # Parse errors
        specific_errors = []
        if errors_str and errors_str.lower() != "none":
            # Split by common delimiters
            for delimiter in ["\n", ";", ","]:
                if delimiter in errors_str:
                    specific_errors = [e.strip() for e in errors_str.split(delimiter) if e.strip()]
                    break

        # Parse suggestions
        suggestions = []
        if suggestions_str and suggestions_str.lower() != "none":
            for delimiter in ["\n", ";", ","]:
                if delimiter in suggestions_str:
                    suggestions = [s.strip() for s in suggestions_str.split(delimiter) if s.strip()]
                    break

        return VerificationResult(
            problem_id=problem_id,
            solution_id=solution.solver_id,
            is_correct=is_correct or is_partial,
            confidence=confidence,
            feedback=feedback,
            verification_time=verification_time,
            specific_errors=specific_errors,
            suggestions=suggestions,
            verifier_id="MathematicalProblemVerifier"
        )

    def _extract_field(self, text: str, field_name: str) -> str:
        """Extract a field from structured text"""
        # Try different patterns
        patterns = [
            f"{field_name}:",
            f"{field_name} =",
            f"{field_name} is"
        ]

        for pattern in patterns:
            if pattern.lower() in text.lower():
                # Find the field
                idx = text.lower().find(pattern.lower())
                start = idx + len(pattern)

                # Find the end (next field or end of text)
                end = len(text)
                for next_field in ["CORRECTNESS:", "CONFIDENCE:", "FEEDBACK:", "ERRORS:", "SUGGESTIONS:"]:
                    if next_field.lower() in text[start:].lower():
                        end = start + text[start:].lower().find(next_field.lower())
                        break

                return text[start:end].strip()

        return ""


class PSVManager:
    """Manager for the complete PSV self-play system"""

    def __init__(self, config: PSVConfig):
        self.config = config
        self.llm_client = LLMClient(config)
        self.proposer = MathematicalProblemProposer(self.llm_client, config)
        self.solver = MathematicalProblemSolver(self.llm_client, config)
        self.verifier = MathematicalProblemVerifier(self.llm_client, config)

        self.episode_history: List[PSVEpisode] = []
        self.problem_database: Dict[str, MathematicalProblem] = {}

        # Performance tracking
        self.metrics = {
            "total_episodes": 0,
            "successful_episodes": 0,
            "failed_episodes": 0,
            "partial_episodes": 0,
            "average_verification_time": 0.0,
            "average_solving_time": 0.0,
            "difficulty_distribution": defaultdict(int)
        }

    async def run_self_play_episode(
        self,
        domain: Optional[str] = None,
        target_difficulty: Optional[float] = None
    ) -> PSVEpisode:
        """Run a complete PSV episode"""
        logger.info("Starting new PSV self-play episode")

        # 1. Propose problem
        problem = await self.proposer.propose_problem(
            domain=domain,
            target_difficulty=target_difficulty,
            previous_problems=list(self.problem_database.values())
        )

        # 2. Solve problem
        solution = await self.solver.solve_problem(problem)

        # 3. Verify solution
        verification = await self.verifier.verify_solution(problem, solution)

        # 4. Create episode
        episode = PSVEpisode(
            episode_id=str(uuid.uuid4()),
            problem=problem,
            solution=solution,
            verification=verification
        )

        # 5. Determine learning outcome
        if verification.is_correct and verification.confidence > 0.8:
            episode.learning_outcome = "success"
            self.metrics["successful_episodes"] += 1
            problem.metadata["solved"] = True
        elif verification.is_correct:
            episode.learning_outcome = "partial"
            self.metrics["partial_episodes"] += 1
            problem.metadata["solved"] = True
        else:
            episode.learning_outcome = "failure"
            self.metrics["failed_episodes"] += 1
            problem.metadata["solved"] = False

        # 6. Store episode and problem
        self.episode_history.append(episode)
        self.problem_database[problem.id] = problem
        self.metrics["total_episodes"] += 1
        self.metrics["difficulty_distribution"][problem.domain] += 1

        # Update timing metrics
        self.metrics["average_verification_time"] = (
            (self.metrics["average_verification_time"] * (self.metrics["total_episodes"] - 1) +
             verification.verification_time) / self.metrics["total_episodes"]
        )
        self.metrics["average_solving_time"] = (
            (self.metrics["average_solving_time"] * (self.metrics["total_episodes"] - 1) +
             solution.solving_time_seconds) / self.metrics["total_episodes"]
        )

        logger.info(f"Episode {episode.episode_id} completed with outcome: {episode.learning_outcome}")
        return episode

    async def run_batch_episodes(
        self,
        num_episodes: int,
        domains: Optional[List[str]] = None
    ) -> List[PSVEpisode]:
        """Run multiple PSV episodes"""
        episodes = []

        for i in range(num_episodes):
            domain = domains[i % len(domains)] if domains else None
            episode = await self.run_self_play_episode(domain=domain)
            episodes.append(episode)

            logger.info(f"Completed {i+1}/{num_episodes} episodes")

        return episodes

    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return {
            **self.metrics,
            "success_rate": (
                self.metrics["successful_episodes"] / self.metrics["total_episodes"]
                if self.metrics["total_episodes"] > 0 else 0.0
            ),
            "total_problems": len(self.problem_database),
            "domain_distribution": dict(self.metrics["difficulty_distribution"])
        }

    async def close(self):
        """Clean up resources"""
        await self.llm_client.close()


# Factory function
def create_psv_manager(
    openai_api_key: Optional[str] = None,
    anthropic_api_key: Optional[str] = None,
    custom_api_base: Optional[str] = None,
    custom_api_key: Optional[str] = None,
    default_provider: LLMProvider = LLMProvider.OPENAI
) -> PSVManager:
    """Create a PSV manager with the specified configuration"""
    config = PSVConfig(
        openai_api_key=openai_api_key,
        anthropic_api_key=anthropic_api_key,
        custom_api_base=custom_api_base,
        custom_api_key=custom_api_key,
        default_provider=default_provider
    )

    return PSVManager(config)


# Export main classes
__all__ = [
    'PSVManager',
    'MathematicalProblemProposer',
    'MathematicalProblemSolver',
    'MathematicalProblemVerifier',
    'LLMClient',
    'PSVConfig',
    'LLMProvider',
    'MathematicalProblem',
    'SolutionAttempt',
    'VerificationResult',
    'PSVEpisode',
    'create_psv_manager'
]
