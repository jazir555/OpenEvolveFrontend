from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

from llm_utils import _compose_messages, _request_openai_compatible_chat
from workflow_structures import ModelConfig, Team
from mdap_engine import (
    RedFlagRules,
    RedFlagger,
    canonicalize_candidate
)

# Scaling-law analytics (pure, offline). Import guarded so the engine is usable
# even if maker_scaling is not on sys.path (it lives in the same flat package).
try:
    from maker_scaling import (
        step_success_probability,
        full_task_success_probability,
        required_k_for_reliability,
        expected_votes_per_step,
        expected_cost,
        parallelization_factor,
    )
    SCALING_AVAILABLE = True
except ImportError:  # pragma: no cover - only when module missing
    SCALING_AVAILABLE = False

# Optional richer correlated-error detection via the project's red-flagging system.
# Guarded so maker_engine never hard-depends on reliability/*.
try:
    from reliability.enhanced_redflagger import EnhancedRedflagger as _EnhancedRedflagger
    ENHANCED_REDFLAG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _EnhancedRedflagger = None
    ENHANCED_REDFLAG_AVAILABLE = False

# A voter is a backend-agnostic callable that, given a rendered prompt and a
# system prompt, returns (raw_text, candidate). The default voter calls the LLM
# (OpenAI-compatible). Injecting a voter makes the engine generic and lets tests
# run fully offline with a deterministic/mock voter.
Voter = Callable[[str, Optional[str], Optional[Dict[str, Any]], "MakerStep"], Tuple[str, Any]]

# **ACTUAL INTEGRATION**: Alerting and knowledge for Maker operations
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False

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
    use_enhanced_redflag: bool = False


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
    def __init__(self, team: Team, config: MakerConfig, voter: Optional[Voter] = None):
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
        # Backend-agnostic voter (paper: "relatively small non-reasoning models
        # suffice"). Default = OpenAI-compatible LLM call. Injected voters let the
        # engine run offline (mock/deterministic) and remain fully generic.
        self.voter: Voter = voter if voter is not None else self._default_voter

        # Optional richer correlated-error detection (exceeds paper: integrates the
        # project's EnhancedRedflagger in addition to the core RedFlagger).
        self._enhanced_redflagger = None
        if config.use_enhanced_redflag and ENHANCED_REDFLAG_AVAILABLE:
            try:
                self._enhanced_redflagger = _EnhancedRedflagger()
            except Exception:
                self._enhanced_redflagger = None

    def solve(
        self,
        initial_state: Any,
        step_builder: Callable[[Any, List[Dict[str, Any]]], MakerStep],
        apply_action: Callable[[Any, Any], Any],
        checkpoint_store: Optional[CheckpointStore] = None,
        stop_condition: Optional[Callable[[MakerState], bool]] = None
    ) -> MakerRunResult:
        import time
        start_time = time.time()
        state = MakerState(current_state=initial_state)
        terminated_reason = "max_steps_reached"
        success = False

        try:
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
                    success = True
                    break

            duration = time.time() - start_time
            result = MakerRunResult(state=state, metrics=self.metrics.copy(), terminated_reason=terminated_reason)

            # **ACTUAL INTEGRATION**: Extract knowledge and track performance for successful solve
            if success or "steps" in result.metrics and result.metrics["steps"] > 0:
                self._extract_maker_knowledge("solve", result)
                self._track_maker_performance("solve", True, duration, result.metrics["steps"])
            else:
                self._track_maker_performance("solve", False, duration, 0)

            return result

        except Exception as e:
            logger.error(f"Error in maker engine solve: {e}", exc_info=True)
            duration = time.time() - start_time

            # **ACTUAL INTEGRATION**: Trigger alert and track failure
            self._trigger_maker_alerts("solve", False, str(e))
            self._track_maker_performance("solve", False, duration, 0)

            raise

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

            # (EXCEED) Optional richer correlated-error detection via the project's
            # EnhancedRedflagger. Paper core signal: malformed/structurally
            # inconsistent outputs correlate with deeper reasoning errors and must
            # be discarded BEFORE voting to decorrelate errors.
            if self._enhanced_flag(raw_text):
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
        """Obtain one candidate via the (injectable) voter.

        The voter callable returns ``(raw_text, candidate)``. The default voter
        performs the OpenAI-compatible LLM call and parses the response. Injected
        voters (mock/deterministic) bypass the LLM entirely, making MAKER generic
        and testable offline.
        """
        system_prompt = step.system_prompt or "You are a specialized AI agent. Follow the instructions precisely."
        prompt = step.render_prompt(current_state, history)
        raw_text, candidate = self.voter(prompt, system_prompt, step.expected_schema, step)
        return raw_text, candidate

    def _default_voter(self, prompt: str, system_prompt: str | None,
                       expected_schema: Optional[Dict[str, Any]], step: MakerStep) -> Tuple[str, Any]:
        """Default voter: OpenAI-compatible LLM call + response parsing."""
        agent = self._select_agent(step)
        sys_prompt = system_prompt or step.system_prompt or "You are a specialized AI agent. Follow the instructions precisely."
        messages = _compose_messages(sys_prompt, prompt)

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
        candidate = self._parse_candidate(raw_text, expected_schema)
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

    def _enhanced_flag(self, raw_text: str) -> bool:
        """Return True if the EnhancedRedflagger flags this response.

        Used only when ``config.use_enhanced_redflag`` is set and the module is
        importable. Always returns False otherwise (guarded, never raises).
        """
        if self._enhanced_redflagger is None or not raw_text:
            return False
        try:
            flags = self._enhanced_redflagger.scan(raw_text)
            return bool(flags)
        except Exception:
            return False

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

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Maker
    # =========================================================================

    def _trigger_maker_alerts(
        self,
        run_id: str,
        success: bool,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for Maker failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            if not success:
                severity = AlertSeverity.HIGH

                alert_manager.create_alert(
                    title=f"Maker Run Failed: {run_id}",
                    description=f"Maker run '{run_id}' failed. " + (f"Error: {error}" if error else ""),
                    severity=severity.value,
                    source="maker_engine",
                    component="maker",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger Maker alert: {e}")

    def _extract_maker_knowledge(
        self,
        run_id: str,
        result: 'MakerRunResult'
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract Maker execution knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"maker_{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="maker_execution",
                source_component="maker_engine",
                title=f"Maker Execution: {run_id}",
                content={
                    "run_id": run_id,
                    "metrics": result.metrics,
                    "terminated_reason": result.terminated_reason,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "final_step": result.state.step_index
                },
                tags=["maker", "workflow", "multi_step"]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted Maker knowledge for {run_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract Maker knowledge: {e}")
            return False

    def _track_maker_performance(
        self,
        run_id: str,
        success: bool,
        duration: float,
        steps_completed: int
    ):
        """**ACTUAL INTEGRATION**: Track Maker performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            performance_data = StrategyPerformanceData(
                strategy_name=f"maker_{self.team.name}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=1.0 if success else 0.0,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={"steps_completed": steps_completed, "duration": duration}
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked Maker performance: {run_id}")

        except Exception as e:
            logger.error(f"Failed to track Maker performance: {e}")


# -----------------------------------------------------------------------------
# Generic, backend-agnostic runner (exceeds the paper: any voter backend works)
# -----------------------------------------------------------------------------

def _deterministic_mock_voter(
    prompt: str,
    system_prompt: Optional[str],
    expected_schema: Optional[Dict[str, Any]],
    step: "MakerStep",
) -> Tuple[str, Any]:
    """Built-in offline voter used when no voter/LLM is supplied.

    Returns a consistent, valid action for each step (keyed by step_id) so the
    workflow can be exercised without any LLM. Deterministic -> zero errors, which
    is the point of the demonstration that the machinery converges.
    """
    import re
    idx = 0
    match = re.search(r"(\d+)", step.step_id or "")
    if match:
        idx = int(match.group(1))
    candidate = {
        "action": {"kind": "noop", "step": idx},
        "next_state": {"step": idx},
    }
    return json.dumps(candidate), candidate


class _StubTeam:
    """Minimal Team stand-in so the engine can run without a real model config.

    The default (LLM) voter is the only code path that touches team members; when
    an injectable voter is supplied the team is never dereferenced.
    """

    name = "generic"
    members: List[Any] = []


def run_generic_maker(
    initial_state: Any,
    num_steps: int,
    step_prompt_template: str,
    expected_schema: Optional[Dict[str, Any]] = None,
    config: Optional[MakerConfig] = None,
    voter: Optional[Voter] = None,
    apply_action: Optional[Callable[[Any, Any], Any]] = None,
    target_reliability: float = 0.95,
    estimated_p: float = 0.9,
    checkpoint_store: Optional[CheckpointStore] = None,
) -> Dict[str, Any]:
    """Run the generic MAKER generate_solution workflow end-to-end.

    This is the central wiring used by API routes: it drives ``MakerEngine`` with
    an injectable voter (offline mock by default) and returns the action sequence,
    final state, run metrics, and scaling-law predictions.

    Args:
        initial_state: starting task state.
        num_steps: number of dependent steps to execute (s in the paper).
        step_prompt_template: prompt template rendered per step; may use the
            ``{state}`` and ``{history}`` placeholders.
        expected_schema: optional JSON schema used for red-flagging/validation.
        config: optional ``MakerConfig``; sensible defaults applied otherwise.
        voter: injectable voter callable; if None a deterministic offline mock is
            used so the workflow runs without any API key.
        apply_action: maps (state, action) -> next_state; defaults to using the
            action's ``next_state`` when present.
        target_reliability: target full-task zero-error probability used for the
            scaling-law predictions (and auto-tuned k).
        estimated_p: estimated per-sample correctness used for the projections.
        checkpoint_store: optional checkpoint store.

    Returns:
        dict with keys: ``actions``, ``final_state``, ``metrics``, ``scaling_laws``.
    """
    config = config or MakerConfig(max_steps=num_steps)
    config.max_steps = max(config.max_steps, num_steps)
    if voter is None:
        voter = _deterministic_mock_voter

    engine = MakerEngine(_StubTeam(), config, voter=voter)

    def step_builder(state: Any, history: List[Dict[str, Any]]) -> MakerStep:
        return MakerStep(
            step_id=f"step_{len(history) + 1}",
            prompt_template=step_prompt_template,
            expected_schema=expected_schema,
            task_type="solve",
            priority=0,
            system_prompt="You are a specialized micro-agent executing one step.",
        )

    def default_apply_action(current_state: Any, action: Any) -> Any:
        if isinstance(action, dict) and "next_state" in action:
            return action["next_state"]
        return current_state

    used_apply = apply_action or default_apply_action

    result = engine.solve(
        initial_state=initial_state,
        step_builder=step_builder,
        apply_action=used_apply,
        checkpoint_store=checkpoint_store,
    )

    actions = [entry.get("action") for entry in result.state.history]

    scaling_laws: Dict[str, Any] = {}
    if SCALING_AVAILABLE:
        k_used = config.k_min
        scaling_laws = {
            "estimated_p": estimated_p,
            "target_reliability": target_reliability,
            "num_steps": num_steps,
            "k_used": k_used,
            "step_success_probability": step_success_probability(estimated_p, k_used),
            "full_task_success_probability": full_task_success_probability(
                estimated_p, k_used, num_steps
            ),
            "required_k_for_reliability": required_k_for_reliability(
                estimated_p, num_steps, target_reliability
            ),
            "expected_cost": expected_cost(
                estimated_p, num_steps, k=k_used, target=target_reliability
            ),
            "parallelization_factor": parallelization_factor(
                num_steps, target=target_reliability, p=estimated_p
            ),
        }

    return {
        "actions": actions,
        "final_state": result.state.current_state,
        "metrics": result.metrics,
        "scaling_laws": scaling_laws,
    }

