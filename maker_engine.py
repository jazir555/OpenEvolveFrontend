import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_utils import _compose_messages, _request_openai_compatible_chat
from workflow_structures import ModelConfig, Team
from mdap_engine import (
    RedFlagRules,
    RedFlagger,
    canonicalize_candidate
)

logger = logging.getLogger(__name__)


@dataclass
class MakerStep:
    step_id: str
    prompt_template: str
    expected_schema: Optional[Dict[str, Any]] = None
    task_type: str = "general"
    priority: int = 0
    system_prompt: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, state: Any, history: List[Dict[str, Any]]) -> str:
        state_payload = json.dumps(state, ensure_ascii=True)
        history_payload = json.dumps(history, ensure_ascii=True)
        return self.prompt_template.format(state=state_payload, history=history_payload)


@dataclass
class MakerConfig:
    k_min: int = 2
    k_max: int = 8
    max_votes_per_step: int = 60
    max_steps: int = 1000
    timeout_seconds: int = 90
    checkpoint_interval: int = 25
    red_flag_rules: RedFlagRules = field(default_factory=RedFlagRules)


@dataclass
class MakerState:
    step_index: int = 0
    current_state: Any = None
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_action: Optional[Any] = None


@dataclass
class MakerRunResult:
    state: MakerState
    metrics: Dict[str, Any]
    terminated_reason: str


class CheckpointStore:
    def save(self, state: MakerState):
        raise NotImplementedError

    def load(self) -> Optional[MakerState]:
        raise NotImplementedError


class FileCheckpointStore(CheckpointStore):
    def __init__(self, path: str):
        self.path = path

    def save(self, state: MakerState):
        payload = {
            "step_index": state.step_index,
            "current_state": state.current_state,
            "history": state.history,
            "last_action": state.last_action
        }
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def load(self) -> Optional[MakerState]:
        try:
            with open(self.path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except FileNotFoundError:
            return None

        return MakerState(
            step_index=payload.get("step_index", 0),
            current_state=payload.get("current_state"),
            history=payload.get("history", []),
            last_action=payload.get("last_action")
        )


class MakerEngine:
    def __init__(self, team: Team, config: MakerConfig):
        self.team = team
        self.config = config
        self.red_flagger = RedFlagger(config.red_flag_rules)
        self.metrics = {
            "steps": 0,
            "votes_cast": 0,
            "red_flags": 0,
            "escalations": 0,
            "errors": 0
        }

    def solve(
        self,
        initial_state: Any,
        step_builder: Callable[[Any, List[Dict[str, Any]]], MakerStep],
        apply_action: Callable[[Any, Any], Any],
        checkpoint_store: Optional[CheckpointStore] = None,
        stop_condition: Optional[Callable[[MakerState], bool]] = None
    ) -> MakerRunResult:
        state = MakerState(current_state=initial_state)
        terminated_reason = "max_steps_reached"

        for _ in range(self.config.max_steps):
            step = step_builder(state.current_state, state.history)
            action = self._maker_step(step, state.current_state, state.history)
            if action is None:
                terminated_reason = "no_action_selected"
                break

            try:
                next_state = apply_action(state.current_state, action)
            except Exception as exc:
                self.metrics["errors"] += 1
                terminated_reason = f"apply_action_failed:{exc}"
                break

            state.history.append({"action": action, "state": next_state})
            state.last_action = action
            state.step_index += 1
            self.metrics["steps"] += 1
            state.current_state = next_state

            if checkpoint_store and state.step_index % self.config.checkpoint_interval == 0:
                checkpoint_store.save(state)
            if stop_condition and stop_condition(state):
                terminated_reason = "stop_condition_met"
                break

        return MakerRunResult(state=state, metrics=self.metrics.copy(), terminated_reason=terminated_reason)

    def _maker_step(self, step: MakerStep, current_state: Any,
                    history: List[Dict[str, Any]]) -> Optional[Any]:
        start = time.time()
        votes: Dict[str, int] = {}
        attempts = 0
        k_value = self._compute_k_value(step)

        while attempts < self.config.max_votes_per_step:
            if time.time() - start > self.config.timeout_seconds:
                break

            raw_text, candidate = self._collect_vote(step, current_state, history)
            attempts += 1
            self.metrics["votes_cast"] += 1

            if not self._candidate_has_action(candidate):
                self.metrics["red_flags"] += 1
                continue

            is_flagged, _ = self.red_flagger.is_flagged(raw_text, candidate, step.expected_schema)
            if is_flagged:
                self.metrics["red_flags"] += 1
                continue

            key = canonicalize_candidate(candidate)
            votes[key] = votes.get(key, 0) + 1

            if self._has_k_ahead(votes, k_value):
                winner_key = max(votes, key=votes.get)
                winner = self._decode_vote(winner_key)
                if isinstance(winner, dict):
                    return winner.get("action")
                return winner

        self.metrics["escalations"] += 1
        return self._best_effort_action(votes)

    def _collect_vote(self, step: MakerStep, current_state: Any,
                      history: List[Dict[str, Any]]) -> Tuple[str, Any]:
        agent = self._select_agent(step)
        system_prompt = step.system_prompt or "You are a specialized AI agent. Follow the instructions precisely."
        prompt = step.render_prompt(current_state, history)
        messages = _compose_messages(system_prompt, prompt)

        response = _request_openai_compatible_chat(
            api_key=agent.api_key,
            base_url=agent.api_base,
            model=agent.model_id,
            messages=messages,
            temperature=agent.temperature,
            top_p=agent.top_p,
            frequency_penalty=agent.frequency_penalty,
            presence_penalty=agent.presence_penalty,
            max_tokens=agent.max_tokens,
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
        if schema is not None and schema.get("type") in ("object", "array"):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, TypeError) as exc:
                logger.warning("Failed to parse JSON candidate: %s", exc)
                return {"raw": raw_text, "parse_error": str(exc)}
        if stripped.startswith("{") or stripped.startswith("["):
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                return stripped
        return stripped

    def _candidate_has_action(self, candidate: Any) -> bool:
        if isinstance(candidate, dict):
            return "action" in candidate
        return isinstance(candidate, str)

    def _select_agent(self, step: MakerStep) -> ModelConfig:
        if not self.team.members:
            raise ValueError("Team has no members")
        return self.team.members[step.priority % len(self.team.members)]

    def _has_k_ahead(self, votes: Dict[str, int], k_value: int) -> bool:
        if not votes:
            return False
        leader = max(votes, key=votes.get)
        leader_count = votes[leader]
        max_other = max((count for key, count in votes.items() if key != leader), default=0)
        return leader_count >= max_other + k_value

    def _compute_k_value(self, step: MakerStep) -> int:
        base_k = max(self.config.k_min, min(self.config.k_max, 1 + step.priority))
        if step.task_type in ["critical", "security", "safety"]:
            base_k = min(self.config.k_max, base_k + 1)
        return base_k

    def _decode_vote(self, vote_key: str) -> Any:
        try:
            return json.loads(vote_key)
        except (json.JSONDecodeError, TypeError):
            return vote_key

    def _best_effort_action(self, votes: Dict[str, int]) -> Optional[Any]:
        if not votes:
            return None
        winner_key = max(votes, key=votes.get)
        winner = self._decode_vote(winner_key)
        if isinstance(winner, dict):
            return winner.get("action")
        return winner
