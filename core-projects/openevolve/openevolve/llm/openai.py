"""
OpenAI API interface for LLMs

This module also supports a "manual mode" (human-in-the-loop) where prompts are written
to a task queue directory and the system waits for a corresponding *.answer.json file
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import openai

from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


def _iso_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _build_display_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Render messages into a single plain-text prompt for the manual UI.
    """
    chunks: List[str] = []
    for m in messages:
        role = str(m.get("role", "user")).upper()
        content = m.get("content", "")
        chunks.append(f"### {role}\n{content}\n")
    return "\n".join(chunks).rstrip() + "\n"


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.tmp"
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def _is_responses_model(model_name: str) -> bool:
    """Check if model uses OpenAI Responses API (like muse-spark models)"""
    model_lower = str(model_name).lower()
    return model_lower.startswith("muse-spark-")


def _get_opencode_session_id(extra_headers: Optional[Dict[str, str]]) -> str:
    """Get or generate OpenCode session ID from headers"""
    if extra_headers and "X-Session-ID" in extra_headers:
        return extra_headers["X-Session-ID"]
    return f"openevolve-{uuid.uuid4().hex[:12]}"


class OpenAILLM(LLMInterface):
    """LLM interface using OpenAI-compatible APIs"""

    def __init__(
        self,
        model_cfg: Optional[dict] = None,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)
        self.reasoning_effort = getattr(model_cfg, "reasoning_effort", None)

        # Manual mode: enabled via llm.manual_mode in config.yaml
        self.manual_mode = (getattr(model_cfg, "manual_mode", False) is True)
        self.manual_queue_dir: Optional[Path] = None

        if self.manual_mode:
            qdir = getattr(model_cfg, "_manual_queue_dir", None)
            if not qdir:
                raise ValueError(
                    "Manual mode is enabled but manual_queue_dir is missing. "
                    "This should be injected by the OpenEvolve controller."
                )
            self.manual_queue_dir = Path(str(qdir)).expanduser().resolve()
            self.manual_queue_dir.mkdir(parents=True, exist_ok=True)
            self.client = None
        else:
            # Set up API client (normal mode)
            # OpenAI client requires max_retries to be int, not None
            max_retries = self.retries if self.retries is not None else 0
            extra_headers = getattr(model_cfg, "extra_headers", None)
            
            # Auto-generate OpenCode session ID for Responses API models if not provided
            if _is_responses_model(self.model) and extra_headers is not None:
                extra_headers = dict(extra_headers)  # Copy to avoid mutating original
                if "X-Session-ID" not in extra_headers:
                    extra_headers["X-Session-ID"] = _get_opencode_session_id(None)
            
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.api_base,
                timeout=self.timeout,
                max_retries=max_retries,
                default_headers=extra_headers,
            )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized OpenAI LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Check if this model uses the Responses API
        use_responses_api = _is_responses_model(self.model)

        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Convert messages to Responses API format if needed
        if use_responses_api:
            # Responses API uses a single input string with instructions
            # Combine system message and user messages
            input_parts = [system_message]
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    input_parts.append(content)
            input_text = "\n\n".join(input_parts)

            # Set up generation parameters for Responses API
            params = {
                "model": self.model,
                "input": input_text,
                "max_output_tokens": kwargs.get("max_tokens", self.max_tokens),
            }
            # Temperature is not supported in Responses API for some models
            temperature = kwargs.get("temperature", self.temperature)
            if temperature is not None:
                params["temperature"] = temperature

            # Handle reasoning_effort
            reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
            if reasoning_effort is not None:
                params["reasoning"] = {"effort": reasoning_effort}
        else:
            # Standard parameters for Chat Completions API
            # Define OpenAI reasoning models that require max_completion_tokens
            OPENAI_REASONING_MODEL_PREFIXES = (
                "o1-", "o1", "o3-", "o3", "o4-", "gpt-5-", "gpt-5", "gpt-oss-120b", "gpt-oss-20b",
            )
            model_lower = str(self.model).lower()
            is_openai_reasoning_model = model_lower.startswith(OPENAI_REASONING_MODEL_PREFIXES)

            if is_openai_reasoning_model:
                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "max_completion_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
                if reasoning_effort is not None:
                    params["reasoning_effort"] = reasoning_effort
                if "verbosity" in kwargs:
                    params["verbosity"] = kwargs["verbosity"]
            else:
                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "temperature": kwargs.get("temperature", self.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                }
                top_p = kwargs.get("top_p", self.top_p)
                if top_p is not None:
                    params["top_p"] = top_p
                reasoning_effort = kwargs.get("reasoning_effort", self.reasoning_effort)
                if reasoning_effort is not None:
                    params["reasoning_effort"] = reasoning_effort

            # Add seed parameter for reproducibility if configured
            seed = kwargs.get("seed", self.random_seed)
            if seed is not None:
                api_base = (self.api_base or "").rstrip("/")
                if api_base == "https://generativelanguage.googleapis.com/v1beta/openai":
                    logger.warning(
                        "Skipping seed parameter as Google AI Studio endpoint doesn't support it. "
                        "Reproducibility may be limited."
                    )
                else:
                    params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)

        # Manual mode: no timeout unless explicitly passed by the caller
        if self.manual_mode:
            timeout = kwargs.get("timeout", None)
            return await self._manual_wait_for_answer(params, timeout=timeout)

        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                response = await asyncio.wait_for(
                    self._call_api(params), timeout=timeout
                )
                return response
            except asyncio.TimeoutError:
                if attempt < retries:
                    logger.warning(
                        f"Timeout on attempt {attempt + 1}/{retries + 1}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"All {retries + 1} attempts failed with timeout")
                    raise
            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"Error on attempt {attempt + 1}/{retries + 1}: {str(e)}. Retrying..."
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(
                        f"All {retries + 1} attempts failed with error: {str(e)}"
                    )
                    raise

    async def _call_api(self, params: Dict[str, Any]) -> str:
        """Make the actual API call"""
        if self.client is None:
            raise RuntimeError("OpenAI client is not initialized (manual_mode enabled?)")

        # Use asyncio to run the blocking API call in a thread pool
        loop = asyncio.get_event_loop()
        
        # Check if this model uses the Responses API
        use_responses_api = _is_responses_model(self.model)
        
        if use_responses_api:
            # Use the Responses API for muse-spark models
            response = await loop.run_in_executor(
                None, lambda: self.client.responses.create(**params)
            )
            # Extract text from Responses API response
            return response.output_text
        else:
            # Use standard Chat Completions API
            response = await loop.run_in_executor(
                None, lambda: self.client.chat.completions.create(**params)
            )
            # Logging of system prompt, user message and response content
            logger = logging.getLogger(__name__)
            logger.debug(f"API parameters: {params}")
            logger.debug(f"API response: {response.choices[0].message.content}")
            return response.choices[0].message.content

    async def _manual_wait_for_answer(
        self, params: Dict[str, Any], timeout: Optional[Union[int, float]]
    ) -> str:
        """
        Manual mode: write a task JSON file and poll for *.answer.json
        If timeout is provided, we respect it; otherwise we wait indefinitely
        """

        if self.manual_queue_dir is None:
            raise RuntimeError("manual_queue_dir is not initialized")

        task_id = str(uuid.uuid4())
        messages = params.get("messages", [])
        display_prompt = _build_display_prompt(messages)

        task_payload: Dict[str, Any] = {
            "id": task_id,
            "created_at": _iso_now(),
            "model": params.get("model"),
            "display_prompt": display_prompt,
            "messages": messages,
            "meta": {
                "max_tokens": params.get("max_tokens"),
                "max_completion_tokens": params.get("max_completion_tokens"),
                "temperature": params.get("temperature"),
                "top_p": params.get("top_p"),
                "reasoning_effort": params.get("reasoning_effort"),
                "verbosity": params.get("verbosity"),
            },
        }

        task_path = self.manual_queue_dir / f"{task_id}.json"
        answer_path = self.manual_queue_dir / f"{task_id}.answer.json"

        _atomic_write_json(task_path, task_payload)
        logger.info(f"[manual_mode] Task enqueued: {task_path}")

        start = time.time()
        poll_interval = 0.5

        while True:
            if answer_path.exists():
                try:
                    data = json.loads(answer_path.read_text(encoding="utf-8"))
                except Exception as e:
                    logger.warning(f"[manual_mode] Failed to parse answer JSON for {task_id}: {e}")
                    await asyncio.sleep(poll_interval)
                    continue

                answer = str(data.get("answer") or "")
                logger.info(f"[manual_mode] Answer received for {task_id}")
                return answer

            if timeout is not None and (time.time() - start) > float(timeout):
                raise asyncio.TimeoutError(
                    f"Manual mode timed out after {timeout} seconds waiting for answer of task {task_id}"
                )

            await asyncio.sleep(poll_interval)
