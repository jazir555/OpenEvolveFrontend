"""
Parameter synchronisation manager (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full manager pushes parameters over the BubbleLabs
REST API. This stub implements the parts that need no backend for real:
parameter specs, range/type validation, change history and sync bookkeeping
against the headless :mod:`.ui_shim` session state. The "push to BubbleLabs"
step is local bookkeeping only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ._stub_support import STUB
from .ui_shim import ui as st

logger = logging.getLogger(__name__)

__all__ = ["STUB", "ParameterChange", "ParameterSpec", "PARAMETER_SPECS", "ParameterSyncManager"]


def _utcnow() -> datetime:
    """Return an aware UTC timestamp."""
    return datetime.now(timezone.utc)


@dataclass
class ParameterChange:
    """
    One recorded parameter mutation.

    Attributes:
        name: Parameter name.
        old_value: Value before the change.
        new_value: Value after the change.
        source_ui: Which side originated the change (e.g. ``"ui"``,
            ``"bubblelabs"``).
        timestamp: When the change was recorded.
    """

    name: str
    old_value: Any
    new_value: Any
    source_ui: str
    timestamp: datetime = field(default_factory=_utcnow)


@dataclass(frozen=True)
class ParameterSpec:
    """
    Type and range contract for a synchronised parameter.

    Attributes:
        name: Parameter name.
        kind: Expected Python type (``bool``, ``int``, ``float`` or ``str``).
        minimum: Inclusive lower bound for numeric parameters.
        maximum: Inclusive upper bound for numeric parameters.
        description: Human-readable purpose of the parameter.
    """

    name: str
    kind: type
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    description: str = ""

    def validate(self, value: Any) -> bool:
        """
        Check ``value`` against this spec.

        Args:
            value: Candidate value.

        Returns:
            ``True`` when the value satisfies the type and range contract.
        """
        if self.kind is bool:
            return isinstance(value, bool)

        # bool is a subclass of int - reject it for numeric specs.
        if isinstance(value, bool):
            return False

        if self.kind is int:
            if not isinstance(value, int):
                return False
        elif self.kind is float:
            if not isinstance(value, (int, float)):
                return False
        elif not isinstance(value, self.kind):
            return False

        if self.minimum is not None and float(value) < self.minimum:
            return False
        if self.maximum is not None and float(value) > self.maximum:
            return False
        return True


#: Parameters this manager knows how to synchronise. Unknown names are rejected.
PARAMETER_SPECS: Dict[str, ParameterSpec] = {
    spec.name: spec
    for spec in (
        ParameterSpec("temperature", float, 0.0, 2.0, "LLM sampling temperature"),
        ParameterSpec("top_p", float, 0.0, 1.0, "Nucleus sampling cutoff"),
        ParameterSpec("max_iterations", int, 1, 1_000_000, "Evolution iteration budget"),
        ParameterSpec("population_size", int, 1, 100_000, "Candidates per generation"),
        ParameterSpec("num_islands", int, 1, 1_000, "Island-model subpopulation count"),
        ParameterSpec("elite_ratio", float, 0.0, 1.0, "Fraction of elites carried over"),
        ParameterSpec("exploration_ratio", float, 0.0, 1.0, "Exploration/exploitation balance"),
        ParameterSpec("max_tokens", int, 1, 1_000_000, "Per-request token ceiling"),
        ParameterSpec("enable_qd_evolution", bool, description="Enable quality-diversity evolution"),
        ParameterSpec("enable_adversarial", bool, description="Enable adversarial evaluation"),
        ParameterSpec("problem_statement", str, description="Problem being solved"),
    )
}


class ParameterSyncManager:
    """
    Track and reconcile parameters between the UI and BubbleLabs.

    Attributes:
        change_history: Every recorded :class:`ParameterChange`, oldest first.
        conflicts: Parameters whose UI and BubbleLabs values disagree.
        last_full_sync: Timestamp of the most recent full sync, if any.
    """

    def __init__(self) -> None:
        self.change_history: List[ParameterChange] = []
        self.conflicts: List[str] = []
        self.last_full_sync: Optional[datetime] = None
        self._to_bubblelabs: Dict[str, Any] = {}
        self._from_bubblelabs: Dict[str, Any] = {}
        self._synced_count: int = 0

    # -- validation / history -------------------------------------------------

    def _validate_parameter(self, name: str, value: Any) -> bool:
        """
        Validate one parameter against :data:`PARAMETER_SPECS`.

        Unknown parameter names are rejected so typos cannot silently propagate
        to the backend.

        Args:
            name: Parameter name.
            value: Candidate value.

        Returns:
            ``True`` when the parameter is known and the value is in range.
        """
        spec = PARAMETER_SPECS.get(name)
        if spec is None:
            logger.debug("Rejecting unknown parameter %r", name)
            return False
        return spec.validate(value)

    def _record_parameter_change(
        self,
        name: str,
        old_value: Any,
        new_value: Any,
        source_ui: str,
    ) -> ParameterChange:
        """
        Append a parameter change to the history.

        Args:
            name: Parameter name.
            old_value: Value before the change.
            new_value: Value after the change.
            source_ui: Originating side of the change.

        Returns:
            The recorded :class:`ParameterChange`.
        """
        change = ParameterChange(name=name, old_value=old_value, new_value=new_value, source_ui=source_ui)
        self.change_history.append(change)
        return change

    # -- synchronisation ------------------------------------------------------

    def _collect_from_session(self) -> Tuple[Dict[str, Any], List[str]]:
        """
        Read known parameters out of the headless UI session state.

        Returns:
            Tuple of ``(valid values, names present but invalid)``.
        """
        valid: Dict[str, Any] = {}
        invalid: List[str] = []
        for name in PARAMETER_SPECS:
            if name not in st.session_state:
                continue
            value = st.session_state[name]
            if self._validate_parameter(name, value):
                valid[name] = value
            else:
                invalid.append(name)
        return valid, invalid

    def sync_from_ui_to_bubblelabs(self) -> Dict[str, Any]:
        """
        Push every valid UI parameter towards BubbleLabs.

        Returns:
            Mapping with:

            * ``status`` - ``"success"`` (all present params valid),
              ``"partial"`` (some invalid) or ``"empty"`` (nothing to sync).
            * ``synced`` - number of parameters accepted.
            * ``invalid`` - names present in the UI but failing validation.
            * ``parameters`` - the accepted name/value pairs.
        """
        valid, invalid = self._collect_from_session()

        for name, value in valid.items():
            previous = self._to_bubblelabs.get(name)
            if previous != value:
                self._record_parameter_change(name, previous, value, "ui")
            self._to_bubblelabs[name] = value

        self._synced_count += len(valid)
        self.last_full_sync = _utcnow()

        if not valid:
            status = "empty"
        elif invalid:
            status = "partial"
        else:
            status = "success"

        return {
            "status": status,
            "synced": len(valid),
            "invalid": invalid,
            "parameters": dict(valid),
            "timestamp": self.last_full_sync.isoformat(),
        }

    def sync_from_bubblelabs_to_ui(self, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Pull parameters from BubbleLabs into the UI session state.

        Args:
            parameters: Values reported by BubbleLabs. When omitted, the last
                values pushed from the UI are echoed back.

        Returns:
            Mapping with ``status``, ``synced`` and ``invalid`` keys.
        """
        incoming = dict(parameters if parameters is not None else self._to_bubblelabs)
        invalid: List[str] = []
        synced = 0

        for name, value in incoming.items():
            if not self._validate_parameter(name, value):
                invalid.append(name)
                continue
            previous = st.session_state.get(name)
            if previous != value:
                self._record_parameter_change(name, previous, value, "bubblelabs")
            st.session_state[name] = value
            self._from_bubblelabs[name] = value
            synced += 1

        self._synced_count += synced
        self.last_full_sync = _utcnow()
        status = "empty" if not incoming else ("partial" if invalid else "success")
        return {"status": status, "synced": synced, "invalid": invalid}

    # -- reporting ------------------------------------------------------------

    def get_parameter_sync_status(self) -> Dict[str, Any]:
        """
        Report per-parameter synchronisation state.

        Returns:
            Mapping with ``last_full_sync``, ``params_synced_to_bubblelabs``,
            ``params_synced_from_bubblelabs``, ``parameter_statuses`` (one entry
            per known parameter) and ``conflicts``.
        """
        parameter_statuses: Dict[str, Dict[str, Any]] = {}
        for name, spec in PARAMETER_SPECS.items():
            ui_value = st.session_state.get(name)
            pushed = self._to_bubblelabs.get(name)
            parameter_statuses[name] = {
                "name": name,
                "type": spec.kind.__name__,
                "minimum": spec.minimum,
                "maximum": spec.maximum,
                "ui_value": ui_value,
                "bubblelabs_value": pushed,
                "in_sync": ui_value == pushed,
                "description": spec.description,
            }

        return {
            "last_full_sync": self.last_full_sync.isoformat() if self.last_full_sync else None,
            "params_synced_to_bubblelabs": dict(self._to_bubblelabs),
            "params_synced_from_bubblelabs": dict(self._from_bubblelabs),
            "parameter_statuses": parameter_statuses,
            "conflicts": list(self.conflicts),
        }

    def get_sync_metrics(self) -> Dict[str, Any]:
        """
        Report aggregate synchronisation counters.

        Returns:
            Mapping with ``synced_parameters``, ``known_parameters``,
            ``recorded_changes``, ``conflicts`` and ``last_full_sync``.
        """
        return {
            "synced_parameters": self._synced_count,
            "known_parameters": len(PARAMETER_SPECS),
            "recorded_changes": len(self.change_history),
            "conflicts": len(self.conflicts),
            "last_full_sync": self.last_full_sync.isoformat() if self.last_full_sync else None,
        }
