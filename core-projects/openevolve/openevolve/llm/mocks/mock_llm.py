"""
Offline mock LLM that satisfies :class:`openevolve.llm.base.LLMInterface`

``MockLLMClient`` (in ``mock_client.py``) returns rich ``MockLLMResponse`` objects and is
meant for unit tests of client behaviour. The evolution pipeline, however, needs an object
that implements ``LLMInterface`` and returns plain strings, so this module provides a thin
adapter plus a module-level factory that can be used as ``LLMModelConfig.init_client``.

The factory is defined at module level on purpose: ``openevolve.process_parallel`` pickles
the config (including ``init_client``) when it hands work to worker processes, and only
module-level callables are picklable by reference.

Typical use (see ``openevolve.server``)::

    from openevolve.config import Config, LLMModelConfig
    from openevolve.llm.mocks.mock_llm import create_mock_llm

    config = Config()
    config.diff_based_evolution = False  # the mock emits full rewrites
    config.llm.models = [LLMModelConfig(name="mock-model", init_client=create_mock_llm)]
"""

import itertools
import logging
import re
from typing import Any, Dict, List, Optional

from openevolve.llm.base import LLMInterface
from openevolve.llm.mocks.mock_client import MockLLMClient

logger = logging.getLogger(__name__)

# Matches the "# Current Program" section emitted by the built-in prompt templates
_CURRENT_PROGRAM_RE = re.compile(
    r"#\s*Current Program\s*```[A-Za-z0-9_+#-]*\n(.*?)```", re.DOTALL
)
_ANY_CODE_BLOCK_RE = re.compile(r"```[A-Za-z0-9_+#-]*\n(.*?)```", re.DOTALL)


class MockLLM(LLMInterface):
    """LLMInterface-compatible mock used for offline/no-API-key runs.

    The mock echoes the current program back as a "full rewrite" with an extra marker
    comment. That keeps evolution runs end-to-end functional without any network access:
    children are valid programs, the real evaluator still runs, and metrics are real.
    """

    def __init__(self, model_cfg: Optional[Any] = None) -> None:
        self.model = getattr(model_cfg, "name", None) or "mock-model"
        self.model_cfg = model_cfg
        self.client = MockLLMClient(model_name=self.model)
        self._counter = itertools.count(1)

    async def generate(self, prompt: str, **kwargs: Any) -> str:
        return await self.generate_with_context(
            system_message="", messages=[{"role": "user", "content": prompt}], **kwargs
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs: Any
    ) -> str:
        user_text = "\n\n".join(str(m.get("content", "")) for m in messages or [])
        variant = next(self._counter)

        code = self._extract_current_program(user_text)
        if code is None:
            # No program found in the prompt (e.g. an evaluation/analysis prompt):
            # fall back to the canned MockLLMClient response.
            response = await self.client.generate(user_text or system_message)
            return response.content

        marker = f"# [mock-llm] variant {variant} from {self.model}"
        mutated = f"{marker}\n{code.strip()}\n"
        return f"```python\n{mutated}```"

    @staticmethod
    def _extract_current_program(text: str) -> Optional[str]:
        match = _CURRENT_PROGRAM_RE.search(text)
        if match:
            return match.group(1)
        blocks = _ANY_CODE_BLOCK_RE.findall(text)
        if blocks:
            # The current program is normally the last code block of the prompt
            return blocks[-1]
        return None


def create_mock_llm(model_cfg: Optional[Any] = None) -> MockLLM:
    """Factory usable as ``LLMModelConfig.init_client`` (picklable by reference)."""
    return MockLLM(model_cfg)
