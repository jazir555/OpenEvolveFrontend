"""
Self-contained SOP document model and renderer.

This module provides REAL, importable SOP generation logic that does NOT depend
on external services (no MAKER / LLM / network). It renders Standard Operating
Procedure documents from `SOPParameter` values plus templates, supports
parameter substitution and section generation, and validates parameters.

`SOPParameter` is provided externally (by another agent) as
`engines/other/sop_parameter.py` and is imported here. It is never redefined.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from sop_parameter import SOPParameter


# ============================================================================
# SOP document model
# ============================================================================


@dataclass
class SOPStep:
    """A single protocol step."""

    step_number: int
    action: str
    duration: Optional[float] = None  # seconds
    duration_tolerance: Optional[float] = None
    verification_method: str = ""
    acceptance_criteria: str = ""
    contingency_action: str = ""
    substeps: List[str] = field(default_factory=list)

    def format_step(self) -> str:
        result = f"**Step {self.step_number}:** {self.action}\n\n"
        if self.duration is not None:
            dur = self._format_duration(self.duration)
            if self.duration_tolerance is not None:
                dur += f" +/- {self._format_duration(self.duration_tolerance)}"
            result += f"- Duration: {dur}\n"
        if self.verification_method:
            result += f"- Verification: {self.verification_method}\n"
        if self.acceptance_criteria:
            result += f"- Acceptance: {self.acceptance_criteria}\n"
        if self.contingency_action:
            result += f"- Contingency: {self.contingency_action}\n"
        for i, sub in enumerate(self.substeps, 1):
            result += f"  - Sub-step {self.step_number}.{i}: {sub}\n"
        return result

    @staticmethod
    def _format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f} s"
        if seconds < 3600:
            return f"{seconds / 60:.1f} min"
        return f"{seconds / 3600:.1f} h"


@dataclass
class StandardOperatingProcedure:
    """A complete, renderable SOP document."""

    title: str
    version: str = "1.0"
    status: str = "DRAFT"
    effective_date: str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d"))
    description: str = ""
    classification: str = "TURNKEY"

    preconditions: List[str] = field(default_factory=list)
    environmental_conditions: Dict[str, SOPParameter] = field(default_factory=dict)
    equipment: List[Dict[str, str]] = field(default_factory=list)
    materials: List[Dict[str, Any]] = field(default_factory=list)
    protocols: List[SOPStep] = field(default_factory=list)
    quality_control: List[str] = field(default_factory=list)
    safety_protocols: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    scaling_info: List[str] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    revision_history: List[Dict[str, str]] = field(default_factory=list)

    # ----- helpers -----

    def parameter_spec(self, name: str) -> str:
        """Human-readable specification for one parameter."""
        param = self.environmental_conditions.get(name)
        if param is None:
            return name
        return format_parameter(param)

    def validate(self) -> List[str]:
        """Validate parameters; return list of issue strings (empty == OK)."""
        issues: List[str] = []
        for name, param in self.environmental_conditions.items():
            if not _is_finite(getattr(param, "value", None)):
                issues.append(f"Parameter '{name}' has a non-numeric value")
            if getattr(param, "tolerance", None) is None:
                issues.append(f"Parameter '{name}' is missing a tolerance")
            elif getattr(param, "tolerance", 0) < 0:
                issues.append(f"Parameter '{name}' has a negative tolerance")
            if not getattr(param, "unit", ""):
                issues.append(f"Parameter '{name}' is missing a unit")
        if not self.protocols:
            issues.append("SOP has no protocol steps")
        return issues

    def to_markdown(self) -> str:
        out: List[str] = []
        out.append(f"# {self.title}\n")
        out.append(f"**Version:** {self.version}")
        out.append(f"**Status:** {self.status}")
        out.append(f"**Effective Date:** {self.effective_date}")
        out.append(f"**Classification:** {self.classification}\n")
        if self.description:
            out.append(self.description + "\n")

        if self.preconditions:
            out.append("## Preconditions\n")
            for p in self.preconditions:
                out.append(f"- {p}")
            out.append("")

        if self.environmental_conditions:
            out.append("## Environmental Conditions\n")
            for name, param in self.environmental_conditions.items():
                out.append(f"### {name}\n")
                out.append(f"- Target: {format_parameter(param)}")
                vm = getattr(param, "verification_method", "")
                if vm:
                    out.append(f"- Verification: {vm}")
                rationale = getattr(param, "rationale", "")
                if rationale:
                    out.append(f"- Rationale: {rationale}")
                out.append("")

        if self.equipment:
            out.append("## Equipment Specifications\n")
            for eq in self.equipment:
                out.append(f"### {eq.get('name', 'Unknown')}\n")
                for k, v in eq.items():
                    if k != "name":
                        out.append(f"- **{k}:** {v}")
                out.append("")

        if self.materials:
            out.append("## Materials\n")
            for mat in self.materials:
                out.append(f"### {mat.get('name', 'Unknown')}\n")
                for k, v in mat.items():
                    if k != "name":
                        out.append(f"- **{k}:** {v}")
                out.append("")

        if self.protocols:
            out.append("## Detailed Execution Protocols\n")
            for step in self.protocols:
                out.append(step.format_step())
                out.append("")

        if self.quality_control:
            out.append("## Quality Control\n")
            for qc in self.quality_control:
                out.append(f"- {qc}")
            out.append("")

        if self.safety_protocols:
            out.append("## Safety Protocols\n")
            for s in self.safety_protocols:
                out.append(f"- {s}")
            out.append("")

        if self.validation_criteria:
            out.append("## Validation\n")
            for v in self.validation_criteria:
                out.append(f"- {v}")
            out.append("")

        if self.scaling_info:
            out.append("## Scaling\n")
            for s in self.scaling_info:
                out.append(f"- {s}")
            out.append("")

        if self.metadata:
            out.append("---\n")
            out.append("## Metadata\n")
            for k, v in self.metadata.items():
                out.append(f"- **{k}:** {v}")
            out.append("")

        return "\n".join(out)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "version": self.version,
            "status": self.status,
            "effective_date": self.effective_date,
            "description": self.description,
            "classification": self.classification,
            "preconditions": self.preconditions,
            "environmental_conditions": {
                n: {
                    "name": getattr(p, "name", n),
                    "value": getattr(p, "value", None),
                    "unit": getattr(p, "unit", ""),
                    "tolerance": getattr(p, "tolerance", None),
                    "verification_method": getattr(p, "verification_method", ""),
                    "critical": getattr(p, "critical", True),
                    "rationale": getattr(p, "rationale", ""),
                }
                for n, p in self.environmental_conditions.items()
            },
            "equipment": self.equipment,
            "materials": self.materials,
            "protocols": [
                {
                    "step_number": s.step_number,
                    "action": s.action,
                    "duration": s.duration,
                    "duration_tolerance": s.duration_tolerance,
                    "verification_method": s.verification_method,
                    "acceptance_criteria": s.acceptance_criteria,
                    "contingency_action": s.contingency_action,
                    "substeps": s.substeps,
                }
                for s in self.protocols
            ],
            "quality_control": self.quality_control,
            "safety_protocols": self.safety_protocols,
            "validation_criteria": self.validation_criteria,
            "scaling_info": self.scaling_info,
            "metadata": self.metadata,
            "revision_history": self.revision_history,
        }


# ============================================================================
# Formatting / validation helpers
# ============================================================================


def format_parameter(param: SOPParameter) -> str:
    """Format a parameter as a specification string."""
    if hasattr(param, "format_spec") and callable(param.format_spec):
        try:
            return param.format_spec()
        except Exception:
            pass
    value = getattr(param, "value", 0)
    unit = getattr(param, "unit", "")
    tol = getattr(param, "tolerance", 0)
    if tol >= (abs(value) if value else 0) and value:
        pct = (tol / value) * 100 if value else 0
        return f"{value} {unit} +/- {pct:.1f}%"
    return f"{value} {unit} +/- {tol} {unit}"


def _is_finite(x: Any) -> bool:
    try:
        return x is not None and float(x) == float(x)
    except (TypeError, ValueError):
        return False


# Token patterns supported by substitute():
#   {{ name }}   or   ${name}   or   {name}
_TOKEN_RE = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}|\$\{([A-Za-z0-9_]+)\}|\{([A-Za-z0-9_]+)\}")


def substitute(template: str, parameters: Dict[str, Any]) -> str:
    """
    Substitute parameter tokens in a template string.

    Tokens reference a key in ``parameters``. A value may be an ``SOPParameter``
    (rendered via :func:`format_parameter`) or any object (rendered via str()).
    Unknown tokens are left untouched so partial templates stay readable.
    """

    def repl(match: "re.Match[str]") -> str:
        name = match.group(1) or match.group(2) or match.group(3)
        if name not in parameters:
            return match.group(0)
        value = parameters[name]
        if isinstance(value, SOPParameter):
            return format_parameter(value)
        return str(value)

    return _TOKEN_RE.sub(repl, template)


# ============================================================================
# Renderer
# ============================================================================


class SOPRenderer:
    """
    Builds SOP documents from parameters, steps and section content.

    Provides section generation and parameter substitution without any external
    service. A template is an optional dict that may contain default section
    text with parameter tokens.
    """

    def __init__(self, template: Optional[Dict[str, Any]] = None):
        self.template = template or {}

    def render(
        self,
        *,
        title: str,
        parameters: Optional[Dict[str, SOPParameter]] = None,
        steps: Optional[List[SOPStep]] = None,
        preconditions: Optional[List[str]] = None,
        equipment: Optional[List[Dict[str, str]]] = None,
        materials: Optional[List[Dict[str, Any]]] = None,
        quality_control: Optional[List[str]] = None,
        safety_protocols: Optional[List[str]] = None,
        validation_criteria: Optional[List[str]] = None,
        scaling_info: Optional[List[str]] = None,
        status: str = "DRAFT",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StandardOperatingProcedure:
        # Substitute parameter tokens inside any free-text section content.
        all_params: Dict[str, Any] = dict(parameters or {})

        def _fill(items: Optional[List[str]]) -> List[str]:
            if not items:
                return []
            return [substitute(text, all_params) for text in items]

        sop = StandardOperatingProcedure(
            title=title,
            status=status,
            description=substitute(description, all_params),
            environmental_conditions={k: v for k, v in (parameters or {}).items()},
            preconditions=_fill(preconditions),
            equipment=equipment or [],
            materials=materials or [],
            protocols=steps or [],
            quality_control=_fill(quality_control),
            safety_protocols=_fill(safety_protocols),
            validation_criteria=_fill(validation_criteria),
            scaling_info=_fill(scaling_info),
            metadata=metadata or {},
        )
        return sop

    def from_template(
        self, title: str, parameters: Dict[str, SOPParameter], domain: str = "general"
    ) -> StandardOperatingProcedure:
        """Render an SOP from this renderer's template plus the given parameters."""
        t = self.template
        steps = [
            SOPStep(
                step_number=i + 1,
                action=substitute(a, parameters),
                verification_method=substitute(t.get("step_verification", ""), parameters),
            )
            for i, a in enumerate(t.get("steps", []))
        ]
        return self.render(
            title=title,
            parameters=parameters,
            steps=steps,
            preconditions=t.get("preconditions"),
            quality_control=t.get("quality_control"),
            safety_protocols=t.get("safety_protocols"),
            validation_criteria=t.get("validation_criteria"),
            scaling_info=t.get("scaling_info"),
            description=t.get("description", ""),
            metadata={"domain": domain, "generator": "SOPRenderer"},
        )


# ============================================================================
# Quality evaluator (no external service)
# ============================================================================


class SOPEvaluator:
    """Heuristic SOP quality evaluator based on completeness and specificity."""

    def __init__(self, domain: str = "general", constraints: Optional[List[str]] = None,
                 equipment: Optional[List[str]] = None):
        self.domain = domain
        self.constraints = constraints or []
        self.equipment = equipment or []

    def evaluate(self, sop: StandardOperatingProcedure) -> float:
        score = 0.0
        # Completeness
        if sop.environmental_conditions:
            score += 0.25
        if sop.protocols:
            score += 0.25
        if sop.quality_control:
            score += 0.15
        if sop.safety_protocols:
            score += 0.15
        if sop.materials or sop.equipment:
            score += 0.10
        # Specificity: every parameter must carry a tolerance + unit.
        if sop.environmental_conditions:
            specific = sum(
                1
                for p in sop.environmental_conditions.values()
                if getattr(p, "tolerance", None) is not None
                and getattr(p, "unit", "")
            )
            score += 0.10 * (specific / len(sop.environmental_conditions))
        issues = sop.validate()
        score -= 0.05 * len(issues)
        return max(0.0, min(1.0, score))


# ============================================================================
# Local generator (no external service)
# ============================================================================


class SOPGenerator:
    """
    Generate a StandardOperatingProcedure from a natural-language requirement.

    This is a deterministic, template-driven generator (no LLM / MAKER). It
    extracts parameter tokens and structured hints from the requirement and
    produces a complete, valid SOP document.
    """

    def __init__(self, config: Any = None):
        self.config = config
        self.renderer = SOPRenderer()

    async def generate_sop(
        self,
        requirement_description: str,
        domain: str = "general",
        constraints: Optional[List[str]] = None,
        equipment_available: Optional[List[str]] = None,
        existing_sop: Optional[StandardOperatingProcedure] = None,
    ) -> StandardOperatingProcedure:
        if existing_sop is not None:
            existing_sop.revision_history.append(
                {
                    "date": datetime.now().isoformat(),
                    "change": f"Refined based on: {requirement_description}",
                }
            )
            return existing_sop

        params = self._extract_parameters(requirement_description)

        # Build a generic but complete protocol from the requirement sentences.
        sentences = [s.strip() for s in re.split(r"[.\n]", requirement_description) if s.strip()]
        steps = [
            SOPStep(
                step_number=i + 1,
                action=sentence[0].upper() + sentence[1:],
                verification_method="Operator confirmation / measurement",
                acceptance_criteria="Step completed as specified",
                contingency_action="Stop and reassess if outcome deviates",
            )
            for i, sentence in enumerate(sentences[:12])
        ]
        if not steps:
            steps = [
                SOPStep(
                    step_number=1,
                    action="Execute the procedure described in the requirement",
                    verification_method="Review against requirement",
                    acceptance_criteria="Requirement satisfied",
                )
            ]

        safety = list(constraints or [])
        safety.append("Follow standard laboratory / workplace safety procedures")
        sop = self.renderer.render(
            title=self._title(requirement_description),
            parameters=params,
            steps=steps,
            preconditions=["Personnel trained for the procedure",
                           "All equipment calibrated and available"],
            equipment=[{"name": e} for e in (equipment_available or [])],
            quality_control=[
                "Record all measured parameters against their tolerances",
                "Quarantine and investigate any out-of-tolerance result",
            ],
            safety_protocols=safety,
            validation_criteria=[
                "Procedure reproduced successfully with consistent results",
            ],
            scaling_info=["Linear scaling assumed unless stated otherwise"],
            description=requirement_description,
            metadata={"domain": domain, "generator": "SOPGenerator"},
        )
        return sop

    @staticmethod
    def _title(requirement: str) -> str:
        first = requirement.strip().split("\n")[0].strip()
        first = re.sub(r"^[#*\-\s]+", "", first)
        return (first[:80] if first else "Standard Operating Procedure").title()

    @staticmethod
    def _extract_parameters(requirement: str) -> Dict[str, SOPParameter]:
        """Extract simple `name: value unit` specifications from text."""
        params: Dict[str, SOPParameter] = {}
        pattern = re.compile(
            r"([A-Za-z][A-Za-z0-9 _-]{1,30}?)\s*[:=]\s*"
            r"(\d+(?:\.\d+)?)\s*([°%A-Za-z/]+)?",
            re.IGNORECASE,
        )
        for m in pattern.finditer(requirement):
            name = m.group(1).strip().rstrip(":").strip()
            if len(name) < 2 or name.lower() in ("the", "a", "an", "to", "of"):
                continue
            value = float(m.group(2))
            unit = (m.group(3) or "").strip()
            key = name.lower().replace(" ", "_")
            if key in params:
                continue
            params[key] = SOPParameter(
                name=name,
                value=value,
                unit=unit,
                tolerance=max(value * 0.05, 0.0),
                verification_method="Measured with calibrated instrument",
                critical=False,
                rationale=f"Extracted from requirement: {name}",
            )
        return params


__all__ = [
    "SOPStep",
    "StandardOperatingProcedure",
    "SOPRenderer",
    "SOPEvaluator",
    "SOPGenerator",
    "format_parameter",
    "substitute",
]
