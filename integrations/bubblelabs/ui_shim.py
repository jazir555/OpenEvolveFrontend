"""
Headless UI shim (stub) for ``integrations.bubblelabs``.

``stub: implement`` - in the full product this module bridges to the BubbleLab
headless UI or to the ``bubble-studio`` front-end. This stub provides a
*headless*, side-effect-free stand-in so that the many modules doing
``from .ui_shim import ui`` remain importable and exercisable in tests, in
CI, and from the CLI without any UI runtime installed.

Nothing is rendered. Widget calls are recorded on :attr:`HeadlessUI.calls` for
assertions, and ``session_state`` is a real dict so parameter-synchronisation
code round-trips correctly.

Example:
    >>> from integrations.bubblelabs.ui_shim import ui
    >>> ui.session_state["temperature"] = 0.8
    >>> left, right = ui.columns(2)
    >>> with left:
    ...     ui.metric("Temperature", 0.8)          # inert, recorded
    >>> ui.session_state["temperature"]
    0.8
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence, Tuple

try:
    try:
        from ._stub_support import STUB
    except ImportError:
        from _stub_support import STUB
except ImportError:
    STUB = False

__all__ = ["STUB", "SessionState", "HeadlessUI", "ui"]


class SessionState(Dict[str, Any]):
    """
    Dict that also supports attribute access, mirroring a headless UI
    ``session_state`` so both ``state["k"]`` and ``state.k`` work.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - mirrors headless UI semantics
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:  # pragma: no cover - mirrors headless UI semantics
            raise AttributeError(name) from exc


class _Widget:
    """
    Inert stand-in for any UI element.

    It is callable, iterable, and usable as a context manager so the common
    headless UI idioms all work without a UI runtime::

        with ui.expander("Details"):
            ui.write("hello")

    Args:
        name: Dotted name of the widget, used for ``repr`` and call recording.
        sink: Optional shared list that records ``(name, args, kwargs)`` calls.
    """

    __slots__ = ("_name", "_sink")

    def __init__(self, name: str = "widget", sink: List[Tuple[str, tuple, dict]] | None = None) -> None:
        self._name = name
        self._sink = sink

    def __call__(self, *args: Any, **kwargs: Any) -> "_Widget":
        if self._sink is not None:
            self._sink.append((self._name, args, kwargs))
        return self

    def __getattr__(self, name: str) -> "_Widget":
        if name.startswith("__"):  # keep dunder lookups honest
            raise AttributeError(name)
        return _Widget(f"{self._name}.{name}", self._sink)

    def __enter__(self) -> "_Widget":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def __iter__(self) -> Iterator["_Widget"]:
        return iter(())

    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"<headless {self._name}>"


class HeadlessUI:
    """
    Minimal headless implementation of the subset of the BubbleLab headless UI
    API that the BubbleLab modules in this package use.

    Attributes:
        session_state: Mutable key/value store shared across modules.
        calls: Ordered record of every widget call, as ``(name, args, kwargs)``.
    """

    def __init__(self) -> None:
        self.session_state: SessionState = SessionState()
        self.calls: List[Tuple[str, tuple, dict]] = []

    # -- layout helpers that must return fixed-length sequences ---------------

    def columns(self, spec: Any = 2, **kwargs: Any) -> List[_Widget]:
        """
        Return one inert column per requested column.

        Args:
            spec: Column count, or a sequence of relative widths.
            **kwargs: Ignored; accepted for headless UI signature compatibility.

        Returns:
            A list of inert column widgets, so ``a, b = ui.columns(2)`` works.
        """
        count = spec if isinstance(spec, int) else len(spec)
        self.calls.append(("columns", (spec,), kwargs))
        return [_Widget(f"column[{i}]", self.calls) for i in range(max(1, int(count)))]

    def tabs(self, labels: Sequence[Any], **kwargs: Any) -> List[_Widget]:
        """
        Return one inert tab per label.

        Args:
            labels: Tab labels.
            **kwargs: Ignored; accepted for headless UI signature compatibility.

        Returns:
            A list of inert tab widgets, one per label.
        """
        self.calls.append(("tabs", (tuple(labels),), kwargs))
        return [_Widget(f"tab[{label}]", self.calls) for label in labels]

    # -- everything else is inert --------------------------------------------

    def __getattr__(self, name: str) -> _Widget:
        """Return an inert, recording widget for any other headless UI call."""
        if name.startswith("__"):
            raise AttributeError(name)
        return _Widget(name, self.calls)

    def reset(self) -> None:
        """Clear recorded calls and session state (useful between tests)."""
        self.calls.clear()
        self.session_state.clear()


#: Process-wide headless UI singleton, mirroring a headless UI session import.
ui: HeadlessUI = HeadlessUI()
