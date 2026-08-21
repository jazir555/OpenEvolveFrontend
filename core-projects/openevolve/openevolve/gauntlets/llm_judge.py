"""
Shared LLM judging helpers for the gauntlet rounds
==================================================

Round 2 (Red Team) and Round 3 (Gold Team) both need to (a) render a prompt that
describes the candidate solution, (b) send it to an LLM, and (c) turn the reply
into a numeric score plus feedback. Both rounds reuse the same plumbing that
:class:`openevolve.evaluator.Evaluator` already uses for LLM feedback:

- :class:`openevolve.prompt.sampler.PromptSampler` renders the ``red_team`` /
  ``gold_team`` templates shipped in ``openevolve/prompts/defaults``.
- :class:`openevolve.llm.ensemble.LLMEnsemble` performs the call, so the offline
  :class:`openevolve.llm.mock.MockLLM` backend is selected automatically when no
  real model/API key is configured. That keeps the gauntlet runnable, offline and
  deterministic (the mock derives its verdict from a hash of the prompt).

The module also provides the deterministic, dependency-free static analysis the
two rounds combine with the LLM verdict:

- :func:`probe_solution` executes concrete adversarial probes (the Red Team's
  attack vectors) and reports which ones succeeded.
- :func:`verify_solution` performs static verification of the candidate and is
  used in place of a Lean 4 proof (Lean is not available offline).

Author: OpenEvolve Gauntlet System
"""

import ast
import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from openevolve.config import LLMConfig, LLMModelConfig, PromptConfig
from openevolve.llm.ensemble import LLMEnsemble
from openevolve.llm.mock import is_mock_model
from openevolve.prompt.sampler import PromptSampler

logger = logging.getLogger(__name__)

# Offline/deterministic backend used when no judge model is configured
OFFLINE_JUDGE_MODEL = "mock"

# Template keys (see openevolve/prompts/defaults/*.txt)
RED_TEAM_TEMPLATE = "red_team"
RED_TEAM_SYSTEM_TEMPLATE = "red_team_system_message"
GOLD_TEAM_TEMPLATE = "gold_team"
GOLD_TEAM_SYSTEM_TEMPLATE = "gold_team_system_message"

# Keys that directly carry the judge's verdict, in priority order
_SCORE_KEYS = (
    "overall_score",
    "score",
    "robustness_score",
    "verification_score",
    "quality_score",
)

# Numeric keys that must never be averaged into a fallback score
_NON_SCORE_KEYS = (
    "severity",
    "count",
    "attempts",
    "attempted",
    "successful",
    "vulnerabilities",
    "iterations",
)

# Keys carrying prose feedback
_FEEDBACK_KEYS = (
    "reasoning",
    "feedback",
    "summary",
    "verdict",
    "explanation",
    "recommendation",
)

# Keys carrying lists of findings
_FINDING_KEYS = (
    "vulnerabilities",
    "findings",
    "issues",
    "weaknesses",
    "risks",
    "attacks",
    "failures",
    "concerns",
    "outstanding_risks",
)

# Severity weights used to turn probe outcomes into a robustness score
_SEVERITY_WEIGHTS = {"high": 1.0, "medium": 0.6, "low": 0.3}

_SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*['\"][^'\"]{4,}['\"]"
)
_UNSAFE_CALL_PATTERN = re.compile(
    r"\beval\s*\(|\bexec\s*\(|os\.system\s*\(|pickle\.loads\s*\(|shell\s*=\s*True"
)
_DIVISION_PATTERN = re.compile(r"[^/*\s]\s*/\s*[A-Za-z_(]|%\s*[A-Za-z_(]")
_ZERO_GUARD_PATTERN = re.compile(
    r"(!=\s*0|==\s*0|>\s*0|<\s*0|ZeroDivision|or\s+1\b|abs\(|if\s+not\s|assert\s)"
)
_MUTABLE_DEFAULT_PATTERN = re.compile(r"def\s+\w+\s*\([^)]*=\s*(\[\]|\{\}|set\(\))")
_VALIDATION_TOKENS = (
    "if not ",
    "assert ",
    "isinstance(",
    "raise valueerror",
    "raise typeerror",
    "is none",
    "validate",
)


# ============================================================================
# VERDICTS
# ============================================================================


@dataclass
class JudgeVerdict:
    """
    Parsed verdict returned by a single judge model.

    Attributes:
        score: Normalized verdict score (0.0-1.0)
        feedback: Human-readable feedback from the judge
        metrics: All numeric fields the judge returned
        findings: Issues/vulnerabilities the judge reported
        model: Name of the model that produced the verdict
        parsed: Whether a structured verdict could be parsed from the response
        raw_response: Raw model response (truncated)
    """

    score: float
    feedback: str
    metrics: Dict[str, float] = field(default_factory=dict)
    findings: List[str] = field(default_factory=list)
    model: str = "unknown"
    parsed: bool = True
    raw_response: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model": self.model,
            "score": self.score,
            "feedback": self.feedback,
            "metrics": self.metrics,
            "findings": self.findings,
            "parsed": self.parsed,
        }


def _extract_payload(response: str) -> Dict[str, Any]:
    """
    Extract the JSON verdict object from a model response.

    Mirrors the extraction logic in ``Evaluator._llm_evaluate``: prefer a
    ```json fenced block, otherwise take the first ``{...}`` object.

    Args:
        response: Raw model response

    Returns:
        Parsed dict (empty when no JSON object could be recovered)
    """
    if not response:
        return {}

    json_match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
    if json_match:
        json_str = json_match.group(1)
    else:
        json_str = response
        start_idx = json_str.find("{")
        end_idx = json_str.rfind("}") + 1
        if start_idx >= 0 and end_idx > start_idx:
            json_str = json_str[start_idx:end_idx]

    try:
        payload = json.loads(json_str)
    except (json.JSONDecodeError, TypeError) as exc:
        logger.debug(f"Judge response is not valid JSON: {exc}")
        return {}

    return payload if isinstance(payload, dict) else {}


def normalize_score(value: Any) -> Optional[float]:
    """
    Coerce a judge-reported number into the 0.0-1.0 range.

    Judges sometimes answer on a 0-10 or 0-100 scale; both are rescaled.

    Args:
        value: Raw value from the model response

    Returns:
        Normalized score, or None if the value is not usable
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if math.isnan(number) or math.isinf(number):
        return None

    if number > 1.0:
        number = number / 10.0 if number <= 10.0 else number / 100.0

    return max(0.0, min(1.0, number))


def parse_verdict(response: str, model: str = "unknown") -> JudgeVerdict:
    """
    Parse an LLM judge response into a :class:`JudgeVerdict`.

    Uses the same JSON extraction convention as ``Evaluator._llm_evaluate``:
    a ```json fenced block, or the first ``{...}`` object in the reply.

    Args:
        response: Raw model response
        model: Name of the model that produced the response

    Returns:
        JudgeVerdict (``parsed=False`` when no JSON verdict was found)
    """
    payload = _extract_payload(response)

    if not isinstance(payload, dict) or not payload:
        return JudgeVerdict(
            score=0.0,
            feedback=(response or "").strip()[:500] or "Judge returned no response",
            model=model,
            parsed=False,
            raw_response=(response or "")[:2000],
        )

    metrics: Dict[str, float] = {}
    findings: List[str] = []
    texts: List[str] = []

    for key, value in payload.items():
        lowered = str(key).lower()

        if isinstance(value, bool):
            metrics[lowered] = 1.0 if value else 0.0
        elif isinstance(value, (int, float)):
            metrics[lowered] = float(value)
        elif isinstance(value, str):
            if lowered in _FEEDBACK_KEYS:
                texts.append(value.strip())
            elif lowered in _FINDING_KEYS:
                findings.append(value.strip())
            else:
                texts.append(f"{key}: {value.strip()}")
        elif isinstance(value, (list, tuple)):
            for item in value:
                if isinstance(item, dict):
                    findings.append(
                        "; ".join(f"{k}={v}" for k, v in item.items())
                    )
                elif item is not None:
                    findings.append(str(item).strip())
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    metrics[f"{lowered}.{str(sub_key).lower()}"] = float(sub_value)

    # Prefer an explicit verdict key, otherwise average the usable numbers
    score: Optional[float] = None
    for key in _SCORE_KEYS:
        if key in metrics:
            score = normalize_score(metrics[key])
            if score is not None:
                break

    if score is None:
        candidates = [
            normalize_score(value)
            for key, value in metrics.items()
            if not any(token in key for token in _NON_SCORE_KEYS)
        ]
        candidates = [value for value in candidates if value is not None]
        if candidates:
            score = sum(candidates) / len(candidates)

    if score is None:
        return JudgeVerdict(
            score=0.0,
            feedback=" ".join(texts)[:500] or "Judge returned no numeric verdict",
            metrics=metrics,
            findings=[f for f in findings if f],
            model=model,
            parsed=False,
            raw_response=(response or "")[:2000],
        )

    return JudgeVerdict(
        score=score,
        feedback=" ".join(texts).strip()[:2000] or "No reasoning provided",
        metrics=metrics,
        findings=[f for f in findings if f],
        model=model,
        parsed=True,
        raw_response=(response or "")[:2000],
    )


def aggregate_verdicts(verdicts: Sequence[JudgeVerdict]) -> JudgeVerdict:
    """
    Merge several judge verdicts into one (mean score, unioned findings).

    Args:
        verdicts: Verdicts from individual models

    Returns:
        Aggregated JudgeVerdict
    """
    usable = [v for v in verdicts if v.parsed]

    if not usable:
        return JudgeVerdict(
            score=0.0,
            feedback="; ".join(v.feedback for v in verdicts)[:2000]
            or "No usable judge verdict",
            model="+".join(v.model for v in verdicts) or "unknown",
            parsed=False,
        )

    metrics: Dict[str, float] = {}
    for verdict in usable:
        for key, value in verdict.metrics.items():
            metrics[key] = metrics.get(key, 0.0) + value / len(usable)

    findings: List[str] = []
    for verdict in usable:
        for finding in verdict.findings:
            if finding not in findings:
                findings.append(finding)

    return JudgeVerdict(
        score=sum(v.score for v in usable) / len(usable),
        feedback=" | ".join(v.feedback for v in usable)[:2000],
        metrics=metrics,
        findings=findings,
        model="+".join(v.model for v in usable),
        parsed=True,
    )


def consensus_score(verdicts: Sequence[JudgeVerdict]) -> float:
    """
    Measure agreement between judges.

    With multiple judges, consensus is the inverse of the score spread. With a
    single judge, agreement cannot be observed, so its own verdict is used.

    Args:
        verdicts: Verdicts from individual models

    Returns:
        Consensus score (0.0-1.0)
    """
    scores = [v.score for v in verdicts if v.parsed]

    if not scores:
        return 0.0
    if len(scores) == 1:
        return scores[0]

    spread = max(scores) - min(scores)
    return max(0.0, min(1.0, 1.0 - spread))


# ============================================================================
# LLM ENSEMBLE CONSTRUCTION
# ============================================================================


def _offline_model_config(random_seed: Optional[int] = 42) -> LLMModelConfig:
    """Build the offline mock model config used when no judge model is set"""
    return LLMModelConfig(
        name=OFFLINE_JUDGE_MODEL,
        provider=OFFLINE_JUDGE_MODEL,
        api_key="not-needed",
        temperature=0.0,
        max_tokens=1024,
        timeout=30,
        retries=1,
        retry_delay=1,
        random_seed=random_seed,
    )


def _coerce_model_configs(
    llm_config: Union[None, Dict[str, Any], LLMConfig, LLMModelConfig, Sequence[Any]],
) -> List[LLMModelConfig]:
    """
    Normalize the many accepted judge configurations into model configs.

    Accepts ``None``, a dict (as used by ``ThreeRoundConfig.roundN_config``),
    an :class:`~openevolve.config.LLMConfig`, a single
    :class:`~openevolve.config.LLMModelConfig`, or a sequence of either.
    """
    if llm_config is None:
        return []

    if isinstance(llm_config, LLMConfig):
        return list(llm_config.evaluator_models or llm_config.models)

    if isinstance(llm_config, LLMModelConfig):
        return [llm_config]

    if isinstance(llm_config, dict):
        if not llm_config:
            return []
        if "models" in llm_config or "evaluator_models" in llm_config:
            raw_models = llm_config.get("evaluator_models") or llm_config.get("models")
            return _coerce_model_configs(list(raw_models or []))

        params = dict(llm_config)
        name = params.pop("name", None) or params.pop("model", None) or params.pop(
            "primary_model", None
        )
        allowed = {f for f in LLMModelConfig.__dataclass_fields__}
        params = {k: v for k, v in params.items() if k in allowed}
        return [LLMModelConfig(name=name, **params)]

    if isinstance(llm_config, (list, tuple)):
        models: List[LLMModelConfig] = []
        for entry in llm_config:
            models.extend(_coerce_model_configs(entry))
        return models

    logger.warning(f"Unsupported judge llm_config type: {type(llm_config)!r}; using offline mock")
    return []


def _has_credentials(model_cfg: LLMModelConfig) -> bool:
    """Whether a real (non-mock) model has a usable API key"""
    if is_mock_model(model_cfg):
        return True
    if model_cfg.api_key:
        return True
    return bool(os.environ.get("OPENAI_API_KEY"))


def build_judge_ensemble(
    llm_config: Union[None, Dict[str, Any], LLMConfig, LLMModelConfig, Sequence[Any]] = None,
) -> LLMEnsemble:
    """
    Build the ensemble used to judge gauntlet rounds.

    Falls back to the offline deterministic mock backend when no model is
    configured or when a real model has no API key available, so the gauntlet
    always has a working evaluator.

    Args:
        llm_config: Judge model configuration (see :func:`_coerce_model_configs`)

    Returns:
        LLMEnsemble ready for ``generate_all_with_context``
    """
    models = _coerce_model_configs(llm_config)
    models = [m for m in models if m.name]

    if models and not all(_has_credentials(m) for m in models):
        logger.warning(
            "No API key available for judge models "
            f"{[m.name for m in models]}; falling back to the offline mock backend"
        )
        models = []

    if not models:
        models = [_offline_model_config()]

    return LLMEnsemble(models)


# ============================================================================
# GAUNTLET JUDGE
# ============================================================================


class GauntletJudge:
    """
    LLM judge shared by the adversarial (Round 2) and gold (Round 3) rounds.

    Example:
        ```python
        judge = GauntletJudge()  # offline mock backend
        verdict = await judge.red_team(
            solution="def solve(x): return x * 2",
            problem="Double the input",
            domain="general",
        )
        print(verdict.score, verdict.feedback)
        ```
    """

    def __init__(
        self,
        llm_config: Union[None, Dict[str, Any], LLMConfig, LLMModelConfig, Sequence[Any]] = None,
        prompt_config: Optional[PromptConfig] = None,
        llm_ensemble: Optional[LLMEnsemble] = None,
    ):
        """
        Initialize the judge.

        Args:
            llm_config: Judge model configuration (offline mock when omitted)
            prompt_config: Optional prompt configuration override
            llm_ensemble: Pre-built ensemble (skips configuration handling)
        """
        self.llm_ensemble = llm_ensemble or build_judge_ensemble(llm_config)
        # Template stochasticity would break determinism of offline runs
        self.prompt_config = prompt_config or PromptConfig(
            use_template_stochasticity=False, include_artifacts=False
        )
        self._samplers: Dict[str, PromptSampler] = {}

    @property
    def model_names(self) -> List[str]:
        """Names of the models backing this judge"""
        return [getattr(model, "model", "unknown") for model in self.llm_ensemble.models]

    def _sampler(self, user_template: str, system_template: str) -> PromptSampler:
        """Get (and cache) a prompt sampler bound to a template pair"""
        key = f"{system_template}|{user_template}"
        sampler = self._samplers.get(key)
        if sampler is None:
            sampler = PromptSampler(self.prompt_config)
            sampler.set_templates(
                system_template=system_template, user_template=user_template
            )
            self._samplers[key] = sampler
        return sampler

    async def judge(
        self,
        template_key: str,
        system_template_key: str,
        solution: str,
        problem: str,
        domain: str,
        language: str = "python",
        prior_findings: Optional[Sequence[str]] = None,
        **extra: Any,
    ) -> List[JudgeVerdict]:
        """
        Render a judging prompt and collect one verdict per judge model.

        Args:
            template_key: User template key (e.g. ``red_team``)
            system_template_key: System message template key
            solution: Candidate solution
            problem: Problem statement
            domain: Problem domain
            language: Language of the candidate solution
            prior_findings: Findings from earlier rounds/probes
            **extra: Additional template placeholders

        Returns:
            One JudgeVerdict per model in the ensemble
        """
        sampler = self._sampler(template_key, system_template_key)

        prompt = sampler.build_prompt(
            current_program=solution,
            template_key=template_key,
            language=language,
            problem=problem,
            domain=domain,
            prior_findings=format_findings(prior_findings),
            **extra,
        )

        responses = await self.llm_ensemble.generate_all_with_context(
            prompt["system"], [{"role": "user", "content": prompt["user"]}]
        )

        model_names = self.model_names
        verdicts: List[JudgeVerdict] = []
        for index, response in enumerate(responses):
            model = model_names[index] if index < len(model_names) else f"model_{index}"
            verdicts.append(parse_verdict(response, model=model))

        return verdicts

    async def red_team(
        self,
        solution: str,
        problem: str,
        domain: str,
        language: str = "python",
        prior_findings: Optional[Sequence[str]] = None,
    ) -> JudgeVerdict:
        """
        Ask the judge to adversarially critique the candidate (Round 2).

        Returns:
            Aggregated JudgeVerdict (higher score == survived more attacks)
        """
        verdicts = await self.judge(
            template_key=RED_TEAM_TEMPLATE,
            system_template_key=RED_TEAM_SYSTEM_TEMPLATE,
            solution=solution,
            problem=problem,
            domain=domain,
            language=language,
            prior_findings=prior_findings,
        )
        return aggregate_verdicts(verdicts)

    async def gold_team(
        self,
        solution: str,
        problem: str,
        domain: str,
        language: str = "python",
        prior_findings: Optional[Sequence[str]] = None,
    ) -> List[JudgeVerdict]:
        """
        Ask the judge(s) to verify/certify the candidate (Round 3).

        Returns:
            One verdict per judge model so consensus can be measured
        """
        return await self.judge(
            template_key=GOLD_TEAM_TEMPLATE,
            system_template_key=GOLD_TEAM_SYSTEM_TEMPLATE,
            solution=solution,
            problem=problem,
            domain=domain,
            language=language,
            prior_findings=prior_findings,
        )


def format_findings(findings: Optional[Sequence[str]]) -> str:
    """Render findings as a bullet list for prompt injection"""
    if not findings:
        return "None reported yet."
    return "\n".join(f"- {str(finding).strip()}" for finding in findings if finding)


# ============================================================================
# DETERMINISTIC STATIC PROBES (RED TEAM ATTACK VECTORS)
# ============================================================================


def looks_like_python(solution: str) -> bool:
    """Heuristic check whether the candidate is Python source"""
    text = solution or ""
    if re.search(r"<\s*(html|div|body|section|form)\b", text, re.IGNORECASE):
        return False
    return bool(re.search(r"^\s*(def|class|import|from|return|async def)\b", text, re.MULTILINE))


def _probe(name: str, description: str, severity: str, successful: bool, evidence: str = "") -> Dict[str, Any]:
    """Build a single attack-probe record"""
    return {
        "name": name,
        "description": description,
        "severity": severity,
        "successful": bool(successful),
        "evidence": evidence,
    }


def _python_probes(solution: str) -> List[Dict[str, Any]]:
    """Adversarial probes for Python candidates"""
    text = solution or ""
    lowered = text.lower()
    probes: List[Dict[str, Any]] = []

    # 1. Malformed program: does it even parse?
    syntax_error = ""
    try:
        ast.parse(text)
    except SyntaxError as exc:
        syntax_error = f"{exc.msg} (line {exc.lineno})"
    probes.append(
        _probe(
            "syntax_integrity",
            "Parse the candidate to confirm it is executable code",
            "high",
            bool(syntax_error),
            syntax_error,
        )
    )

    # 2. Failure handling stripped away
    has_handling = "except" in lowered or "raise " in lowered
    probes.append(
        _probe(
            "error_path_removal",
            "Trigger a failing branch with no error handling",
            "medium",
            not has_handling,
            "No except/raise statement found" if not has_handling else "",
        )
    )

    # 3. Silent failure: swallowed exceptions hide the attack
    swallows = bool(re.search(r"except\s*:?[^\n]*:\s*\n\s*pass\b", text)) or bool(
        re.search(r"except\s*:", text)
    )
    probes.append(
        _probe(
            "silent_failure_injection",
            "Hide a fault behind a bare or swallowed exception handler",
            "medium",
            swallows,
            "Bare/swallowed except found" if swallows else "",
        )
    )

    # 4. Hostile inputs with no validation
    validated = any(token in lowered for token in _VALIDATION_TOKENS)
    probes.append(
        _probe(
            "input_validation_bypass",
            "Feed out-of-domain inputs to unvalidated parameters",
            "high",
            not validated,
            "No validation/guard statements found" if not validated else "",
        )
    )

    # 5. Degenerate numeric input
    divides = bool(_DIVISION_PATTERN.search(text))
    guarded = bool(_ZERO_GUARD_PATTERN.search(text))
    probes.append(
        _probe(
            "degenerate_numeric_input",
            "Drive a division/modulo with a zero or empty denominator",
            "high",
            divides and not guarded,
            "Division without a zero guard" if divides and not guarded else "",
        )
    )

    # 6. Arbitrary code / command execution
    unsafe = _UNSAFE_CALL_PATTERN.search(text)
    probes.append(
        _probe(
            "unsafe_execution",
            "Smuggle a payload into a dynamic execution sink",
            "high",
            bool(unsafe),
            unsafe.group(0) if unsafe else "",
        )
    )

    # 7. Credential exposure
    secret = _SECRET_PATTERN.search(text)
    probes.append(
        _probe(
            "credential_exposure",
            "Extract hardcoded credentials from the source",
            "high",
            bool(secret),
            secret.group(1) if secret else "",
        )
    )

    # 8. Resource exhaustion
    unbounded = "while true" in lowered and "break" not in lowered
    probes.append(
        _probe(
            "resource_exhaustion",
            "Keep an unbounded loop running to exhaust the budget",
            "medium",
            unbounded,
            "while True without break" if unbounded else "",
        )
    )

    # 9. Shared mutable state between calls
    mutable_default = _MUTABLE_DEFAULT_PATTERN.search(text)
    probes.append(
        _probe(
            "shared_state_poisoning",
            "Poison a mutable default argument across calls",
            "low",
            bool(mutable_default),
            mutable_default.group(0) if mutable_default else "",
        )
    )

    # 10. Leaked file handles
    leaks = "open(" in lowered and "with " not in lowered and ".close()" not in lowered
    probes.append(
        _probe(
            "resource_leak",
            "Exhaust file handles via unmanaged open() calls",
            "low",
            leaks,
            "open() without context manager or close()" if leaks else "",
        )
    )

    return probes


def _markup_probes(solution: str) -> List[Dict[str, Any]]:
    """Adversarial probes for markup/front-end candidates"""
    text = solution or ""
    lowered = text.lower()
    probes: List[Dict[str, Any]] = []

    injection = "innerhtml" in lowered or "document.write" in lowered
    probes.append(
        _probe(
            "dom_injection",
            "Inject markup through an unescaped DOM sink",
            "high",
            injection,
            "innerHTML/document.write sink" if injection else "",
        )
    )

    has_form = "<form" in lowered
    unvalidated_form = has_form and not any(
        token in lowered for token in ("required", "pattern=", "type=\"email\"", "novalidate")
    )
    probes.append(
        _probe(
            "form_validation_bypass",
            "Submit malformed data to an unvalidated form",
            "high",
            unvalidated_form,
            "Form without validation attributes" if unvalidated_form else "",
        )
    )

    tabnabbing = "_blank" in lowered and "noopener" not in lowered
    probes.append(
        _probe(
            "reverse_tabnabbing",
            "Hijack the opener of a target=_blank link",
            "medium",
            tabnabbing,
            "target=_blank without rel=noopener" if tabnabbing else "",
        )
    )

    secret = _SECRET_PATTERN.search(text)
    probes.append(
        _probe(
            "credential_exposure",
            "Extract hardcoded credentials from the markup",
            "high",
            bool(secret),
            secret.group(1) if secret else "",
        )
    )

    images = len(re.findall(r"<img", lowered))
    alts = len(re.findall(r"<img[^>]*\balt\s*=", lowered))
    probes.append(
        _probe(
            "accessibility_regression",
            "Disable images and check the textual fallback",
            "medium",
            images > alts,
            f"{images - alts} image(s) without alt text" if images > alts else "",
        )
    )

    inline_script = bool(re.search(r"<script(?![^>]*src=)", lowered))
    probes.append(
        _probe(
            "csp_violation",
            "Block inline scripts (strict CSP) and check degradation",
            "low",
            inline_script,
            "Inline <script> block" if inline_script else "",
        )
    )

    return probes


def probe_solution(solution: str, language: str = "python") -> List[Dict[str, Any]]:
    """
    Run deterministic adversarial probes against a candidate solution.

    These are the Red Team's concrete attack vectors: each probe is "attempted"
    and reports whether the attack "succeeded" (i.e. the candidate is missing the
    corresponding defense). No external service is required.

    Args:
        solution: Candidate solution source
        language: Candidate language hint

    Returns:
        List of probe records with ``name``, ``severity``, ``successful``
    """
    if language.lower() in ("html", "css", "javascript", "js", "markup") or not looks_like_python(
        solution
    ):
        return _markup_probes(solution)
    return _python_probes(solution)


def robustness_from_probes(probes: Sequence[Dict[str, Any]]) -> float:
    """
    Convert probe outcomes into a severity-weighted robustness score.

    Args:
        probes: Probe records from :func:`probe_solution`

    Returns:
        Robustness score (0.0-1.0, 1.0 == every attack was repelled)
    """
    if not probes:
        return 0.0

    total = sum(_SEVERITY_WEIGHTS.get(p.get("severity", "medium"), 0.6) for p in probes)
    if total <= 0:
        return 0.0

    lost = sum(
        _SEVERITY_WEIGHTS.get(p.get("severity", "medium"), 0.6)
        for p in probes
        if p.get("successful")
    )
    return max(0.0, min(1.0, 1.0 - lost / total))


def successful_attacks(probes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Probes whose attack succeeded"""
    return [p for p in probes if p.get("successful")]


def describe_attacks(probes: Sequence[Dict[str, Any]]) -> List[str]:
    """Render successful probes as short findings for prompts/reports"""
    descriptions = []
    for probe in successful_attacks(probes):
        evidence = probe.get("evidence") or probe.get("description", "")
        descriptions.append(f"[{probe.get('severity', 'medium')}] {probe.get('name')}: {evidence}")
    return descriptions


# ============================================================================
# STATIC VERIFICATION (GOLD TEAM)
# ============================================================================


def verify_solution(solution: str, language: str = "python") -> Dict[str, Any]:
    """
    Statically verify a candidate solution.

    This replaces the Lean 4 formal verification step for offline runs: instead
    of a proof, the candidate must compile, expose an interface, assert its
    invariants and avoid unsafe sinks. Every check is deterministic and local.

    Args:
        solution: Candidate solution source
        language: Candidate language hint

    Returns:
        Dict with ``passed``, ``checks`` (bool per check) and ``detail``
    """
    text = solution or ""
    checks: Dict[str, bool] = {}

    if language.lower() == "python" and looks_like_python(text):
        detail = ""
        try:
            tree = ast.parse(text)
            checks["parses"] = True
        except SyntaxError as exc:
            tree = None
            checks["parses"] = False
            detail = f"SyntaxError: {exc.msg} (line {exc.lineno})"

        if tree is not None:
            try:
                compile(text, "<gauntlet-candidate>", "exec")
                checks["compiles"] = True
            except Exception as exc:  # pragma: no cover - compile after parse rarely fails
                checks["compiles"] = False
                detail = f"{type(exc).__name__}: {exc}"

            definitions = [
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            ]
            checks["exposes_interface"] = bool(definitions)
            checks["documented"] = any(ast.get_docstring(node) for node in definitions) or bool(
                ast.get_docstring(tree)
            )
            checks["asserts_invariants"] = any(
                isinstance(node, (ast.Assert, ast.Raise)) for node in ast.walk(tree)
            )
            checks["handles_failures"] = any(
                isinstance(node, (ast.Try, ast.Raise)) for node in ast.walk(tree)
            )
        else:
            checks["compiles"] = False
            checks["exposes_interface"] = False
            checks["documented"] = False
            checks["asserts_invariants"] = False
            checks["handles_failures"] = False

        checks["no_unsafe_sinks"] = not bool(_UNSAFE_CALL_PATTERN.search(text))
        checks["no_hardcoded_secrets"] = not bool(_SECRET_PATTERN.search(text))

        required = (
            "parses",
            "compiles",
            "exposes_interface",
            "asserts_invariants",
            "no_unsafe_sinks",
            "no_hardcoded_secrets",
        )
    else:
        # Markup / non-Python candidates: structural checks only
        lowered = text.lower()
        checks["non_empty"] = bool(text.strip())
        checks["balanced_tags"] = lowered.count("<") == lowered.count(">")
        checks["declares_structure"] = any(
            token in lowered for token in ("<html", "<body", "<section", "<main", "<div")
        )
        checks["no_unsafe_sinks"] = not bool(_UNSAFE_CALL_PATTERN.search(text))
        checks["no_hardcoded_secrets"] = not bool(_SECRET_PATTERN.search(text))
        detail = ""
        required = (
            "non_empty",
            "balanced_tags",
            "no_unsafe_sinks",
            "no_hardcoded_secrets",
        )

    passed = all(checks.get(name, False) for name in required)
    failed = [name for name in checks if not checks[name]]

    if not detail:
        detail = (
            "All static verification checks passed"
            if passed
            else f"Failed checks: {', '.join(failed)}"
        )

    return {"passed": passed, "checks": checks, "detail": detail}
