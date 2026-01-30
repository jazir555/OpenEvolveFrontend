import hashlib
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from llm_utils import _compose_messages, _request_openai_compatible_chat
from workflow_structures import ModelConfig, Team


# SECURITY: Input validation helpers to prevent injection attacks
def _validate_string_input(value: Any, name: str, max_length: int = 10000) -> str:
    """
    Validate string input to prevent injection attacks.
    
    Args:
        value: The input value to validate
        name: Name of the field for error messages
        max_length: Maximum allowed length
        
    Returns:
        str: Validated string value
        
    Raises:
        TypeError: If value is not a string
        ValueError: If value contains suspicious patterns
    """
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string, got {type(value).__name__}")
    
    if len(value) > max_length:
        raise ValueError(f"{name} exceeds maximum length of {max_length} characters")
    
    # Check for null bytes
    if '\x00' in value:
        raise ValueError(f"{name} contains null bytes")
    
    return value


def _validate_id_string(value: str, name: str) -> str:
    """
    Validate ID string to ensure it only contains safe characters.
    
    Args:
        value: The ID string to validate
        name: Name of the field for error messages
        
    Returns:
        str: Validated ID string
    """
    value = _validate_string_input(value, name, max_length=256)
    
    # IDs should only contain alphanumeric, underscore, hyphen, and dot
    if not re.match(r'^[a-zA-Z0-9_.\-:]+$', value):
        raise ValueError(f"{name} contains invalid characters. Only alphanumeric, underscore, hyphen, colon, and dot are allowed: {value[:50]}")
    
    return value

logger = logging.getLogger(__name__)


def _approx_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _safe_json_loads(payload: str) -> Tuple[Optional[Any], Optional[str]]:
    try:
        return json.loads(payload), None
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return None, str(exc)


def validate_schema(candidate: Any, schema: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    if schema is None:
        return True, []

    errors: List[str] = []
    schema_type = schema.get("type")

    if schema_type == "object":
        if not isinstance(candidate, dict):
            return False, ["candidate is not an object"]
        required = schema.get("required", [])
        for key in required:
            if key not in candidate:
                errors.append(f"missing required key: {key}")
        properties = schema.get("properties", {})
        for key, prop in properties.items():
            if key in candidate and "type" in prop:
                if not _matches_type(candidate[key], prop["type"]):
                    errors.append(f"key {key} expected {prop['type']}")
    elif schema_type == "array":
        if not isinstance(candidate, list):
            errors.append("candidate is not an array")
    elif schema_type and not _matches_type(candidate, schema_type):
        errors.append(f"candidate expected type {schema_type}")

    return len(errors) == 0, errors


def _matches_type(value: Any, schema_type: str) -> bool:
    if schema_type == "string":
        return isinstance(value, str)
    if schema_type == "number":
        return isinstance(value, (int, float))
    if schema_type == "integer":
        return isinstance(value, int)
    if schema_type == "boolean":
        return isinstance(value, bool)
    if schema_type == "object":
        return isinstance(value, dict)
    if schema_type == "array":
        return isinstance(value, list)
    return True


def canonicalize_candidate(candidate: Any) -> str:
    if isinstance(candidate, (dict, list)):
        return json.dumps(candidate, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return str(candidate).strip()


def candidate_confidence(candidate: Any, default: float = 0.5) -> float:
    if isinstance(candidate, dict):
        value = candidate.get("confidence")
        if isinstance(value, (int, float)):
            return float(value)
    return default


@dataclass
class RedFlagRules:
    max_tokens: int = 750
    max_characters: Optional[int] = 6000
    blocked_patterns: List[str] = field(default_factory=list)
    min_confidence: float = 0.2
    require_schema_match: bool = True


class RedFlagger:
    def __init__(self, rules: RedFlagRules):
        self.rules = rules

    def is_flagged(self, raw_text: str, candidate: Any, schema: Optional[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        reasons: List[str] = []

        if raw_text is None or raw_text.strip() == "":
            reasons.append("empty_response")
            return True, reasons

        if self.rules.max_characters and len(raw_text) > self.rules.max_characters:
            reasons.append("response_too_long")

        if self.rules.max_tokens and _approx_token_count(raw_text) > self.rules.max_tokens:
            reasons.append("token_limit_exceeded")

        for pattern in self.rules.blocked_patterns:
            if re.search(pattern, raw_text, re.IGNORECASE):
                reasons.append(f"blocked_pattern:{pattern}")

        if schema is not None and self.rules.require_schema_match:
            is_valid, errors = validate_schema(candidate, schema)
            if not is_valid:
                reasons.extend(errors)

        if candidate_confidence(candidate) < self.rules.min_confidence:
            reasons.append("low_confidence")

        return len(reasons) > 0, reasons


@dataclass
class MDAPStep:
    def __post_init__(self):
        """Validate inputs after initialization to prevent injection attacks."""
        # Validate step_id
        self.step_id = _validate_id_string(self.step_id, "step_id")
        
        # Validate prompt (allow more characters but check length and null bytes)
        self.prompt = _validate_string_input(self.prompt, "prompt", max_length=50000)
        
        # Validate task_type
        self.task_type = _validate_string_input(self.task_type, "task_type", max_length=100)
        
        # Validate system_prompt if provided
        if self.system_prompt is not None:
            self.system_prompt = _validate_string_input(self.system_prompt, "system_prompt", max_length=50000)
    
    step_id: str
    prompt: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    temperature_override: Optional[float] = None
    max_tokens_override: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPTask:
    def __post_init__(self):
        """Validate inputs after initialization to prevent injection attacks."""
        # Validate task_id
        self.task_id = _validate_id_string(self.task_id, "task_id")
        
        # Validate description
        self.description = _validate_string_input(self.description, "description", max_length=10000)
        
        # Validate steps is a list
        if not isinstance(self.steps, list):
            raise TypeError(f"steps must be a list, got {type(self.steps).__name__}")
    
    task_id: str
    description: str
    steps: List[MDAPStep]
    max_retries: int = 2
    target_success_rate: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MDAPConfig:
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 50
    timeout_seconds: int = 60
    red_flag_rules: RedFlagRules = field(default_factory=RedFlagRules)
    fallback_policy: str = "escalate_then_best_effort"
    cache_ttl_seconds: Optional[int] = None
    cache_max_size: int = 5000


@dataclass
class MDAPVoteResult:
    winner: Optional[Any]
    votes: Dict[str, int]
    red_flags: int
    confidence: float
    attempts: int
    duration_seconds: float
    flagged_reasons: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


@dataclass
class MDAPStepResult:
    step_id: str
    vote_result: MDAPVoteResult
    status: str
    retries: int


@dataclass
class MDAPRunResult:
    task_id: str
    step_results: Dict[str, MDAPStepResult]
    metrics: Dict[str, Any]


class MDAPCache:
    def __init__(self, max_size: int, ttl_seconds: int):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access: Dict[str, float] = {}

    def get(self, key: str) -> Optional[Any]:
        now = time.time()
        entry = self._cache.get(key)
        if not entry:
            return None
        if now - entry["timestamp"] > self.ttl_seconds:
            self._cache.pop(key, None)
            self._access.pop(key, None)
            return None
        self._access[key] = now
        return entry["value"]

    def set(self, key: str, value: Any):
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        self._cache[key] = {"value": value, "timestamp": time.time()}
        self._access[key] = time.time()

    def _evict_lru(self):
        if not self._access:
            return
        # Safe min() with check for empty dict
        try:
            lru_key = min(self._access, key=self._access.get)
        except ValueError:
            return  # Empty access dict, nothing to evict
        self._cache.pop(lru_key, None)
        self._access.pop(lru_key, None)


class AgentSelector:
    def __init__(self, team: Team, rng: Optional[random.Random] = None):
        self.team = team
        self.rng = rng or random.Random()

    def select(self, step: MDAPStep) -> ModelConfig:
        members = self.team.members
        if not members:
            raise ValueError("Team has no members")

        weights: List[float] = []
        for member in members:
            weight = 1.0
            if step.task_type and member.problem_type_specialization:
                if step.task_type in member.problem_type_specialization:
                    weight += 1.0
            if member.performance_metrics and "success_rate" in member.performance_metrics:
                weight *= max(0.1, member.performance_metrics["success_rate"])
            weights.append(weight)

        total = sum(weights)
        if total <= 0:
            # Fix: Check if members list is empty before accessing [0]
            if not members:
                raise ValueError("Cannot select agent from empty team: no members available")
            return members[0]

        pick = self.rng.uniform(0, total)
        cumulative = 0.0
        for member, weight in zip(members, weights):
            cumulative += weight
            if cumulative >= pick:
                return member
        return members[-1]


class MDAPOrchestrator:
    def __init__(self, team: Team, config: MDAPConfig):
        self.team = team
        self.config = config
        self.selector = AgentSelector(team)
        self.red_flagger = RedFlagger(config.red_flag_rules)
        self.cache = None
        if config.cache_ttl_seconds:
            self.cache = MDAPCache(config.cache_max_size, config.cache_ttl_seconds)
        self.metrics = {
            "steps_completed": 0,
            "steps_failed": 0,
            "red_flags": 0,
            "votes_cast": 0
        }

    def execute_task(self, task: MDAPTask) -> MDAPRunResult:
        step_results: Dict[str, MDAPStepResult] = {}
        for step in task.steps:
            result = self._execute_step_with_retries(step, task)
            step_results[step.step_id] = result
            if result.status == "success":
                self.metrics["steps_completed"] += 1
            else:
                self.metrics["steps_failed"] += 1

        return MDAPRunResult(task_id=task.task_id, step_results=step_results, metrics=self.metrics.copy())

    def _execute_step_with_retries(self, step: MDAPStep, task: MDAPTask) -> MDAPStepResult:
        retries = 0
        vote_result = self._execute_step(step, task)
        while vote_result.winner is None and retries < task.max_retries:
            retries += 1
            vote_result = self._execute_step(step, task)

        status = "success" if vote_result.winner is not None else "failure"
        return MDAPStepResult(step_id=step.step_id, vote_result=vote_result, status=status, retries=retries)

    def _execute_step(self, step: MDAPStep, task: MDAPTask) -> MDAPVoteResult:
        start = time.time()
        votes: Dict[str, int] = {}
        red_flags = 0
        attempts = 0
        flagged_reasons: List[str] = []
        errors: List[str] = []
        k_value = self._compute_k_value(step, task)
        cache_key = self._cache_key(step, task)

        if self.cache:
            cached = self.cache.get(cache_key)
            if cached is not None:
                return MDAPVoteResult(
                    winner=cached,
                    votes={canonicalize_candidate(cached): 1},
                    red_flags=0,
                    confidence=1.0,
                    attempts=0,
                    duration_seconds=0.0
                )

        while attempts < self.config.max_votes_per_step:
            if time.time() - start > self.config.timeout_seconds:
                errors.append("timeout")
                break

            raw_text, candidate = self._sample_candidate(step)
            attempts += 1
            self.metrics["votes_cast"] += 1

            is_flagged, reasons = self.red_flagger.is_flagged(raw_text, candidate, step.expected_schema)
            if is_flagged:
                red_flags += 1
                self.metrics["red_flags"] += 1
                flagged_reasons.extend(reasons)
                continue

            candidate_key = canonicalize_candidate(candidate)
            votes[candidate_key] = votes.get(candidate_key, 0) + 1

            if self._has_k_ahead(votes, k_value):
                winner_key = max(votes, key=votes.get)
                winner = self._decode_vote(winner_key)
                confidence = votes[winner_key] / max(1, sum(votes.values()))
                if self.cache:
                    self.cache.set(cache_key, winner)
                return MDAPVoteResult(
                    winner=winner,
                    votes=votes,
                    red_flags=red_flags,
                    confidence=confidence,
                    attempts=attempts,
                    duration_seconds=time.time() - start,
                    flagged_reasons=flagged_reasons,
                    errors=errors
                )

        return self._fallback_result(step, task, votes, red_flags, attempts, start, flagged_reasons, errors)

    def _fallback_result(
        self,
        step: MDAPStep,
        task: MDAPTask,
        votes: Dict[str, int],
        red_flags: int,
        attempts: int,
        start: float,
        flagged_reasons: List[str],
        errors: List[str]
    ) -> MDAPVoteResult:
        if self.config.fallback_policy != "escalate_then_best_effort":
            return MDAPVoteResult(
                winner=None,
                votes=votes,
                red_flags=red_flags,
                confidence=0.0,
                attempts=attempts,
                duration_seconds=time.time() - start,
                flagged_reasons=flagged_reasons,
                errors=errors
            )

        if votes:
            winner_key = max(votes, key=votes.get)
            winner = self._decode_vote(winner_key)
            confidence = votes[winner_key] / max(1, sum(votes.values()))
            return MDAPVoteResult(
                winner=winner,
                votes=votes,
                red_flags=red_flags,
                confidence=confidence,
                attempts=attempts,
                duration_seconds=time.time() - start,
                flagged_reasons=flagged_reasons,
                errors=errors
            )

        return MDAPVoteResult(
            winner=None,
            votes=votes,
            red_flags=red_flags,
            confidence=0.0,
            attempts=attempts,
            duration_seconds=time.time() - start,
            flagged_reasons=flagged_reasons,
            errors=errors
        )

    def _sample_candidate(self, step: MDAPStep) -> Tuple[str, Any]:
        agent = self.selector.select(step)
        system_prompt = self._system_prompt_for_step(step)
        user_prompt = self._user_prompt_for_step(step)
        messages = _compose_messages(system_prompt, user_prompt)

        response = _request_openai_compatible_chat(
            api_key=agent.api_key,
            base_url=agent.api_base,
            model=agent.model_id,
            messages=messages,
            temperature=step.temperature_override or agent.temperature,
            top_p=agent.top_p,
            frequency_penalty=agent.frequency_penalty,
            presence_penalty=agent.presence_penalty,
            max_tokens=step.max_tokens_override or agent.max_tokens,
            seed=agent.seed,
            stop_sequences=step.stop_sequences or agent.stop_sequences,
            logprobs=agent.logprobs,
            top_logprobs=agent.top_logprobs,
            response_format=agent.response_format,
            stream=agent.stream,
            user=agent.user,
            reasoning_effort=agent.reasoning_effort,
            max_retries=agent.max_retries,
            timeout=agent.timeout,
            organization=agent.organization,
            response_model=agent.response_model,
            tools=agent.tools,
            tool_choice=agent.tool_choice,
            system_fingerprint=agent.system_fingerprint,
            deployment_id=agent.deployment_id,
            encoding_format=agent.encoding_format,
            max_input_tokens=agent.max_input_tokens,
            stop_token=agent.stop_token,
            best_of=agent.best_of,
            logprobs_offset=agent.logprobs_offset,
            suffix=agent.suffix,
            presence_penalty_range=agent.presence_penalty_range,
            frequency_penalty_range=agent.frequency_penalty_range,
            stop_token_id=agent.stop_token_id,
            response_json_format=agent.response_json_format,
            max_output_tokens=agent.max_output_tokens,
            stream_options=agent.stream_options,
            logprobs_type=agent.logprobs_type,
            top_k=agent.top_k,
            repetition_penalty=agent.repetition_penalty,
            length_penalty=agent.length_penalty,
            early_stopping=agent.early_stopping,
            num_beams=agent.num_beams,
            do_sample=agent.do_sample,
            temperature_fallback=agent.temperature_fallback,
            top_p_fallback=agent.top_p_fallback,
            max_time=agent.max_time,
            return_full_text=agent.return_full_text,
            tokenizer_config=agent.tokenizer_config,
            model_kwargs=agent.model_kwargs
        )

        raw_text = response or ""
        candidate = self._parse_candidate(raw_text, step.expected_schema)
        return raw_text, candidate

    def _parse_candidate(self, raw_text: str, schema: Optional[Dict[str, Any]]) -> Any:
        stripped = raw_text.strip()
        if not stripped:
            return {}

        expects_json = schema is not None and schema.get("type") in ("object", "array")
        looks_like_json = stripped.startswith("{") or stripped.startswith("[")
        if expects_json or looks_like_json:
            parsed, error = _safe_json_loads(stripped)
            if parsed is not None:
                return parsed
            logger.warning("Failed to parse JSON candidate: %s", error)
            return {"raw": raw_text, "parse_error": error}
        return stripped

    def _system_prompt_for_step(self, step: MDAPStep) -> str:
        if step.system_prompt:
            return step.system_prompt

        system_map = {
            "content_analysis": self.team.content_analysis_system_prompt,
            "decomposition": self.team.decomposition_system_prompt,
            "solve": self.team.solver_system_prompt,
            "patch": self.team.patcher_system_prompt,
            "assemble": self.team.assembler_system_prompt,
            "red_team": self.team.red_team_system_prompt,
            "gold_team": self.team.gold_team_system_prompt
        }
        return system_map.get(step.task_type) or "You are a specialized AI agent. Follow the instructions precisely."

    def _user_prompt_for_step(self, step: MDAPStep) -> str:
        template_map = {
            "content_analysis": self.team.content_analysis_user_prompt_template,
            "decomposition": self.team.decomposition_user_prompt_template,
            "solve": self.team.solver_user_prompt_template,
            "patch": self.team.patcher_user_prompt_template,
            "assemble": self.team.assembler_user_prompt_template,
            "red_team": self.team.red_team_user_prompt_template,
            "gold_team": self.team.gold_team_user_prompt_template
        }
        template = template_map.get(step.task_type)
        if template and "{prompt}" in template:
            return template.format(prompt=step.prompt)
        return step.prompt

    def _has_k_ahead(self, votes: Dict[str, int], k_value: int) -> bool:
        if not votes:
            return False
        winner = max(votes, key=votes.get)
        winner_count = votes[winner]
        max_other = max((count for key, count in votes.items() if key != winner), default=0)
        return winner_count >= max_other + k_value

    def _compute_k_value(self, step: MDAPStep, task: MDAPTask) -> int:
        base_k = max(self.config.k_min, min(self.config.k_max, int(1 + step.priority)))
        if task.target_success_rate >= 0.98:
            base_k = min(self.config.k_max, base_k + 1)
        return base_k

    def _decode_vote(self, vote_key: str) -> Any:
        candidate, error = _safe_json_loads(vote_key)
        if candidate is not None:
            return candidate
        return vote_key

    def _cache_key(self, step: MDAPStep, task: MDAPTask) -> str:
        payload = f"{task.task_id}:{step.step_id}:{step.prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
