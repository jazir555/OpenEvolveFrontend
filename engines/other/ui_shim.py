"""
UI shim to replace UI in non-UI runtime.
Provides a minimal, no-op interface plus session_state storage.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


class SessionState(dict):
    """Dictionary-like session state with attribute access."""

    def __getattr__(self, key: str) -> Any:
        return self.get(key)

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


class _UIContainer:
    """Context manager + no-op widget container."""

    def __init__(self, ui: "UIShim") -> None:
        self._ui = ui

    def __enter__(self) -> "UIShim":
        return self._ui

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ui, name)


class UIShim:
    """Minimal UI-like API surface for non-UI execution."""

    def __init__(self) -> None:
        self.session_state = SessionState()
        if "thread_lock" not in self.session_state:
            self.session_state.thread_lock = threading.Lock()
        self.sidebar = self

    def __enter__(self) -> "UIShim":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    # --- caching ---
    def cache_data(self, *args, **kwargs):
        def _decorator(func: Callable):
            return func

        return _decorator

    def cache_resource(self, *args, **kwargs):
        def _decorator(func: Callable):
            return func

        return _decorator

    # --- control flow ---
    def stop(self) -> None:
        return None

    def rerun(self) -> None:
        return None

    # --- display / logging ---
    def write(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def table(self, *args, **kwargs):
        return None

    def code(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def pyplot(self, *args, **kwargs):
        return None

    def image(self, *args, **kwargs):
        return None

    def metric(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def title(self, *args, **kwargs):
        return None

    def set_page_config(self, *args, **kwargs):
        return None

    # --- layout helpers ---
    def columns(self, spec: Any) -> List[_UIContainer]:
        count = 1
        if isinstance(spec, int):
            count = spec
        elif isinstance(spec, (list, tuple)):
            count = len(spec)
        return [_UIContainer(self) for _ in range(max(1, count))]

    def tabs(self, labels: Sequence[str]) -> List[_UIContainer]:
        return [_UIContainer(self) for _ in labels]

    def expander(self, *args, **kwargs) -> _UIContainer:
        return _UIContainer(self)

    def container(self, *args, **kwargs) -> _UIContainer:
        return _UIContainer(self)

    def empty(self) -> "UIShim":
        return self

    # --- widgets ---
    def button(self, *args, **kwargs) -> bool:
        return False

    def checkbox(self, *args, **kwargs) -> bool:
        return bool(kwargs.get("value", False))

    def toggle(self, *args, **kwargs) -> bool:
        return bool(kwargs.get("value", False))

    def radio(self, *args, **kwargs):
        options = kwargs.get("options") or []
        index = kwargs.get("index", 0)
        if options:
            return options[index if index < len(options) else 0]
        return None

    def selectbox(self, *args, **kwargs):
        options = kwargs.get("options") or []
        index = kwargs.get("index", 0)
        if options:
            return options[index if index < len(options) else 0]
        return None

    def multiselect(self, *args, **kwargs) -> List[Any]:
        return list(kwargs.get("default", []))

    def select_slider(self, *args, **kwargs):
        if "value" in kwargs:
            return kwargs["value"]
        options = kwargs.get("options") or []
        return options[0] if options else None

    def number_input(self, *args, **kwargs):
        return kwargs.get("value", kwargs.get("min_value", 0))

    def text_input(self, *args, **kwargs) -> str:
        return str(kwargs.get("value", ""))

    def text_area(self, *args, **kwargs) -> str:
        return str(kwargs.get("value", ""))

    def date_input(self, *args, **kwargs):
        return kwargs.get("value")

    def file_uploader(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None

    def progress(self, *args, **kwargs):
        return self

    def form(self, *args, **kwargs) -> _UIContainer:
        return _UIContainer(self)

    def form_submit_button(self, *args, **kwargs) -> bool:
        return False

    # --- misc ---
    def experimental_get_query_params(self) -> Dict[str, List[str]]:
        return {}

    def experimental_set_query_params(self, **kwargs) -> None:
        return None


ui = UIShim()


def html(*args, **kwargs):
    return None


components = SimpleNamespace(v1=SimpleNamespace(html=html))


def st_autorefresh(*args, **kwargs):
    return None


def st_tags(*args, **kwargs):
    return []

