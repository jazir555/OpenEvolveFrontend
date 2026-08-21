"""
Shared support for the thin stub modules in ``integrations.bubblelabs``.

Several modules in this package were written against sibling modules that live
elsewhere in the repository (``engines/other/``, ``engines/workflow/``, ...) and
are only importable through the repo's legacy flat ``sys.path`` layout. To make
this directory a self-contained, importable package, those references are
satisfied by *thin stubs* that live next to their consumers.

Stub policy
-----------
1. A stub exposes exactly the names its consumers import - no invented subsystems.
2. Where a safe, obvious behaviour exists (validation, in-memory bookkeeping,
   headless rendering) the stub implements it for real.
3. Where behaviour requires a live backend (theorem provers, solvers, remote
   auth) the stub raises :class:`StubNotImplementedError` with a
   ``"stub: implement ..."`` message instead of silently returning a wrong value.
4. Every stub module is annotated with :data:`STUB` so stubs are greppable.

Replacing a stub means dropping in the real module and deleting the stub file.
"""

from __future__ import annotations

from typing import NoReturn

__all__ = ["STUB", "StubNotImplementedError", "stub_todo", "raise_stub"]

#: Marker attribute set by every stub module in this package.
STUB: bool = True


class StubNotImplementedError(NotImplementedError):
    """
    Raised by stub callables that have no safe default behaviour.

    Subclasses :class:`NotImplementedError` so existing ``except
    NotImplementedError`` handlers keep working.
    """


def stub_todo(what: str, *, hint: str = "") -> StubNotImplementedError:
    """
    Build a :class:`StubNotImplementedError` with a consistent message.

    Args:
        what: Description of the missing behaviour, e.g.
            ``"LeanAideClient.verify (needs a running LeanAide server)"``.
        hint: Optional extra guidance for whoever implements it.

    Returns:
        The exception instance, ready to ``raise``.
    """
    message = f"stub: implement {what}"
    if hint:
        message = f"{message} - {hint}"
    return StubNotImplementedError(message)


def raise_stub(what: str, *, hint: str = "") -> NoReturn:
    """
    Raise :class:`StubNotImplementedError` for missing behaviour.

    Args:
        what: Description of the missing behaviour.
        hint: Optional extra guidance for whoever implements it.

    Raises:
        StubNotImplementedError: Always.
    """
    raise stub_todo(what, hint=hint)
