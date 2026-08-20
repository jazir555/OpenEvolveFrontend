"""
Deterministic mock LLM backend for offline runs and tests.

This backend implements the same :class:`~openevolve.llm.base.LLMInterface`
contract as :class:`~openevolve.llm.openai.OpenAILLM`, but never touches the
network. It inspects the evolution prompt, recovers the "Current Program" that
the prompt sampler embedded in it, and returns a locally generated mutation in
whichever format the prompt asked for (SEARCH/REPLACE diff or full rewrite).

Determinism
-----------
The mutation is derived from a stable hash of the parent code (plus an optional
``random_seed`` from the model config), so a given parent program always yields
the same child regardless of process, worker or call ordering. Successive
generations still make progress because every mutation stamps an incrementing
``# [mock-evolve pass N]`` marker into the code, which changes the hash input
for the next generation.

Selection
---------
Enabled via config, either by provider or by model name::

    from openevolve.config import Config, LLMModelConfig

    config = Config()
    config.llm.models = [LLMModelConfig(name="mock", api_key="not-needed")]

Any model whose ``provider`` is ``"mock"`` or whose ``name`` starts with
``"mock"`` (e.g. ``"mock"``, ``"mock-gpt"``, ``"mock-llm"``) is routed here by
:func:`openevolve.llm.ensemble._create_model`.
"""

import asyncio
import hashlib
import json
import logging
import re
from typing import Dict, List, Optional, Tuple

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)

# Global counters so tests can assert the mock was actually exercised during a
# run, even when it is instantiated inside worker processes. Use the ``get()``
# helper (or read the list) rather than importing the integer by value, since
# ints are immutable and an ``import X`` snapshot would not see increments.
TOTAL_CALLS = [0]


def total_calls() -> int:
    """Return the number of times any MockLLM produced a response."""
    return TOTAL_CALLS[0]

# Model names that select this backend without an explicit provider.
MOCK_MODEL_PREFIX = "mock"

# Marker stamped into evolved code so generations remain distinguishable.
_MARKER_PATTERN = re.compile(r"#\s*\[mock-evolve pass (\d+)\]")

# "# Current Program\n```python\n...\n```" as emitted by the prompt templates.
_CURRENT_PROGRAM_PATTERN = re.compile(
    r"#\s*Current Program\s*\n```[^\n]*\n(.*?)\n```",
    re.DOTALL,
)

# Any fenced code block, used as a fallback when the header is missing.
_CODE_FENCE_PATTERN = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)

_EVOLVE_START = "# EVOLVE-BLOCK-START"
_EVOLVE_END = "# EVOLVE-BLOCK-END"

# Integer literal not attached to an identifier, name or dot.
_INT_LITERAL_PATTERN = re.compile(r"(?<![\w.])(\d+)(?![\w.])")


def is_mock_model(model_cfg) -> bool:
    """Return True if the given model config should use the mock backend."""
    provider = getattr(model_cfg, "provider", None)
    if provider and str(provider).strip().lower() == MOCK_MODEL_PREFIX:
        return True
    name = getattr(model_cfg, "name", None)
    if name and str(name).strip().lower().startswith(MOCK_MODEL_PREFIX):
        return True
    return False


class MockLLM(LLMInterface):
    """Offline, deterministic stand-in for a real LLM backend.

    Example:
        >>> import asyncio
        >>> from openevolve.config import LLMModelConfig
        >>> llm = MockLLM(LLMModelConfig(name="mock"))
        >>> out = asyncio.run(llm.generate("# Current Program\\n```python\\nx = 1\\n```"))
        >>> "REPLACE" in out or "```" in out
        True
    """

    def __init__(self, model_cfg=None):
        self.model = getattr(model_cfg, "name", None) or "mock"
        self.system_message = getattr(model_cfg, "system_message", None)
        self.temperature = getattr(model_cfg, "temperature", None)
        self.top_p = getattr(model_cfg, "top_p", None)
        self.max_tokens = getattr(model_cfg, "max_tokens", None)
        self.timeout = getattr(model_cfg, "timeout", None)
        self.retries = getattr(model_cfg, "retries", None)
        self.retry_delay = getattr(model_cfg, "retry_delay", None)
        self.api_base = getattr(model_cfg, "api_base", None)
        self.api_key = getattr(model_cfg, "api_key", None)
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Simulated latency in seconds (0 keeps tests fast).
        self.latency = float(getattr(model_cfg, "mock_latency", 0.0) or 0.0)

        # Number of generate calls served, for observability in tests.
        self.call_count = 0

        logger.info(f"Initialized MockLLM backend (model={self.model}, offline)")

    # ------------------------------------------------------------------
    # LLMInterface
    # ------------------------------------------------------------------

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response for a single prompt."""
        return await self.generate_with_context(
            system_message=self.system_message or "",
            messages=[{"role": "user", "content": prompt or ""}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate a response from a system message and conversation."""
        self.call_count += 1
        TOTAL_CALLS[0] += 1

        if self.latency > 0:
            await asyncio.sleep(self.latency)

        user_text = "\n".join(
            str(m.get("content", ""))
            for m in (messages or [])
            if str(m.get("role", "user")) != "system"
        )
        full_prompt = f"{system_message or ''}\n{user_text}"

        # Evaluator prompts ask for a JSON verdict, not a code change.
        if self._wants_json(full_prompt):
            return self._build_json_feedback(full_prompt)

        code = self._extract_current_program(user_text) or self._extract_current_program(
            full_prompt
        )

        if not code or not code.strip():
            logger.debug("MockLLM: no program found in prompt; returning no-op response")
            return (
                "No current program was provided, so there is nothing to change.\n"
            )

        wants_diff = self._wants_diff(full_prompt)
        mutated, description = self._mutate(code)

        if wants_diff:
            return self._build_diff_response(code, mutated, description)
        return self._build_rewrite_response(mutated, description, full_prompt)

    # ------------------------------------------------------------------
    # Prompt inspection
    # ------------------------------------------------------------------

    @staticmethod
    def _wants_diff(prompt: str) -> bool:
        """Detect whether the prompt asked for SEARCH/REPLACE diffs."""
        return "SEARCH" in prompt and "REPLACE" in prompt

    @staticmethod
    def _wants_json(prompt: str) -> bool:
        """Detect an evaluator-style prompt expecting a JSON object."""
        lowered = prompt.lower()
        return "json" in lowered and (
            "readability" in lowered or "evaluate" in lowered or "score" in lowered
        ) and "SEARCH" not in prompt

    @staticmethod
    def _extract_current_program(prompt: str) -> Optional[str]:
        """Recover the current program from a rendered evolution prompt."""
        if not prompt:
            return None

        match = _CURRENT_PROGRAM_PATTERN.search(prompt)
        if match:
            return match.group(1)

        # Fallback: the largest fenced block is almost certainly the program.
        blocks = _CODE_FENCE_PATTERN.findall(prompt)
        if blocks:
            return max(blocks, key=len).rstrip("\n")

        return None

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def _seed_for(self, code: str) -> int:
        """Stable integer seed derived from the code and configured seed."""
        payload = f"{self.random_seed if self.random_seed is not None else ''}\n{code}"
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return int(digest[:16], 16)

    @staticmethod
    def _pass_number(code: str) -> int:
        """Next mock-evolve pass number for the given code."""
        passes = [int(n) for n in _MARKER_PATTERN.findall(code)]
        return (max(passes) + 1) if passes else 1

    @staticmethod
    def _evolve_region(lines: List[str]) -> Tuple[int, int]:
        """Return the [start, end) line range the mock is allowed to touch."""
        start, end = 0, len(lines)
        for idx, line in enumerate(lines):
            if _EVOLVE_START in line:
                start = idx + 1
            elif _EVOLVE_END in line:
                end = idx
                break
        if start >= end:
            return 0, len(lines)
        return start, end

    @classmethod
    def _candidate_lines(cls, lines: List[str]) -> List[int]:
        """Indices of lines that are safe and useful to mutate."""
        start, end = cls._evolve_region(lines)
        candidates = []
        for idx in range(start, end):
            stripped = lines[idx].strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if _EVOLVE_START in lines[idx] or _EVOLVE_END in lines[idx]:
                continue
            candidates.append(idx)
        return candidates

    def _mutate(self, code: str) -> Tuple[str, str]:
        """Produce a deterministic mutation of ``code``.

        Returns:
            Tuple of (mutated_code, human-readable description).
        """
        lines = code.split("\n")
        candidates = self._candidate_lines(lines)
        pass_no = self._pass_number(code)

        if not candidates:
            # Nothing structured to change: stamp a marker so the child differs.
            marker = f"# [mock-evolve pass {pass_no}] no mutable statements found"
            return code.rstrip("\n") + "\n" + marker, "Annotated program with mock marker"

        seed = self._seed_for(code)
        target = candidates[seed % len(candidates)]
        line = lines[target]
        indent = line[: len(line) - len(line.lstrip())]

        # Strategy 1: tune an integer literal (a real, score-affecting change).
        literal_match = _INT_LITERAL_PATTERN.search(line)
        if literal_match and (seed // 7) % 2 == 0:
            original = int(literal_match.group(1))
            delta = 1 + (seed // 13) % 3
            updated = original + delta if (seed // 3) % 2 == 0 else max(0, original - delta)
            if updated != original:
                new_line = (
                    line[: literal_match.start(1)]
                    + str(updated)
                    + line[literal_match.end(1) :]
                )
                comment = (
                    f"{indent}# [mock-evolve pass {pass_no}] tuned constant "
                    f"{original} -> {updated}"
                )
                new_lines = list(lines)
                new_lines[target : target + 1] = [comment, new_line]
                return (
                    "\n".join(new_lines),
                    f"Tuned constant {original} -> {updated}",
                )

        # Strategy 2: annotate the selected statement (always valid Python).
        comment = (
            f"{indent}# [mock-evolve pass {pass_no}] reviewed hot path; "
            f"kept behavior identical"
        )
        new_lines = list(lines)
        new_lines[target : target + 1] = [comment, line]
        return "\n".join(new_lines), "Documented the selected statement"

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _build_diff_response(original: str, mutated: str, description: str) -> str:
        """Render the mutation as one or more SEARCH/REPLACE blocks."""
        original_lines = original.split("\n")
        mutated_lines = mutated.split("\n")

        # Find the first and last differing positions to build a tight hunk.
        prefix = 0
        while (
            prefix < len(original_lines)
            and prefix < len(mutated_lines)
            and original_lines[prefix] == mutated_lines[prefix]
        ):
            prefix += 1

        suffix = 0
        while (
            suffix < len(original_lines) - prefix
            and suffix < len(mutated_lines) - prefix
            and original_lines[len(original_lines) - 1 - suffix]
            == mutated_lines[len(mutated_lines) - 1 - suffix]
        ):
            suffix += 1

        search_lines = original_lines[prefix : len(original_lines) - suffix]
        replace_lines = mutated_lines[prefix : len(mutated_lines) - suffix]

        # An empty SEARCH section cannot be matched; anchor on the previous line.
        if not search_lines:
            anchor = prefix - 1
            if anchor < 0:
                anchor = 0
            search_lines = [original_lines[anchor]]
            replace_lines = [original_lines[anchor]] + replace_lines

        search_text = "\n".join(search_lines)
        replace_text = "\n".join(replace_lines)

        return (
            f"Analysis: {description}. Applying a targeted change to the "
            "evolve block while preserving the program's interface.\n\n"
            "<<<<<<< SEARCH\n"
            f"{search_text}\n"
            "=======\n"
            f"{replace_text}\n"
            ">>>>>>> REPLACE\n"
        )

    @staticmethod
    def _build_rewrite_response(mutated: str, description: str, prompt: str) -> str:
        """Render the mutation as a full-rewrite fenced code block."""
        language = "python"
        fence_match = re.search(r"```([a-zA-Z0-9_+-]+)\n", prompt or "")
        if fence_match:
            language = fence_match.group(1)
        return (
            f"Analysis: {description}.\n\n"
            f"```{language}\n{mutated}\n```\n"
        )

    @staticmethod
    def _build_json_feedback(prompt: str) -> str:
        """Return a deterministic JSON verdict for evaluator-style prompts."""
        digest = hashlib.sha256((prompt or "").encode("utf-8")).hexdigest()
        # Map the digest into a stable, plausible 0.5-0.9 band.
        base = 0.5 + (int(digest[:8], 16) % 41) / 100.0
        payload = {
            "readability": round(base, 2),
            "maintainability": round(min(1.0, base + 0.05), 2),
            "efficiency": round(max(0.0, base - 0.05), 2),
            "reasoning": "Deterministic offline mock evaluation (no LLM call).",
        }
        return f"```json\n{json.dumps(payload, indent=2)}\n```\n"
