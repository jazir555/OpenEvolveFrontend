"""
BubbleLabs security primitives (stub) for ``integrations.bubblelabs``.

``stub: implement`` - the full module implements API-key issuance, token
verification and audited authorisation against a real credential store.

Security posture of this stub
-----------------------------
Input *validation* is implemented for real (UUIDs, workflow types, workflow
actions) because it needs no backend and failing it open would be worse.

*Authentication* is deliberately fail-closed: :meth:`AuthManager.verify_token`
and :meth:`AuthManager.verify_api_key` raise
:class:`~._stub_support.StubNotImplementedError` rather than returning a
plausible-looking context. The :func:`require_auth` and :func:`validate_input`
decorators therefore pass calls through **without enforcing anything** and log a
loud warning once, so no caller can mistake this stub for a security layer.
"""

from __future__ import annotations

import functools
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, FrozenSet, Optional, TypeVar

from ._stub_support import STUB, raise_stub

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

__all__ = [
    "STUB",
    "SecurityContext",
    "AuthManager",
    "auth_manager",
    "validate_uuid",
    "validate_workflow_type",
    "validate_workflow_action",
    "require_auth",
    "validate_input",
    "verify_token",
]

#: Workflow families accepted by :func:`validate_workflow_type`.
VALID_WORKFLOW_TYPES: FrozenSet[str] = frozenset(
    {"evolution", "adversarial", "qd", "quality_diversity", "rag", "validation", "custom"}
)

#: Lifecycle actions accepted by :func:`validate_workflow_action`.
VALID_WORKFLOW_ACTIONS: FrozenSet[str] = frozenset({"start", "pause", "resume", "stop", "cancel", "status"})

_AUTH_WARNING_EMITTED = False


def _warn_auth_not_enforced(where: str) -> None:
    """Log once that authentication is not enforced by this stub."""
    global _AUTH_WARNING_EMITTED
    if not _AUTH_WARNING_EMITTED:
        logger.warning(
            "bubblelabs_security is a STUB: %s does not enforce authentication. "
            "Replace this module before exposing BubbleLabs endpoints.",
            where,
        )
        _AUTH_WARNING_EMITTED = True


@dataclass
class SecurityContext:
    """
    Identity and permissions associated with a request.

    Attributes:
        subject: Authenticated principal identifier.
        scopes: Granted permission scopes.
        api_key_id: Identifier of the presented API key, when applicable.
        authenticated: Whether authentication actually succeeded.
        issued_at: When the context was created.
    """

    subject: str = "anonymous"
    scopes: frozenset = field(default_factory=frozenset)
    api_key_id: Optional[str] = None
    authenticated: bool = False
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def has_scope(self, scope: str) -> bool:
        """
        Check whether a scope is granted.

        Args:
            scope: Scope name to test.

        Returns:
            ``True`` when the context is authenticated and holds ``scope``.
        """
        return self.authenticated and scope in self.scopes


def validate_uuid(value: Any) -> bool:
    """
    Check that ``value`` is a well-formed UUID string.

    Args:
        value: Candidate identifier.

    Returns:
        ``True`` when ``value`` parses as a UUID.
    """
    if isinstance(value, uuid.UUID):
        return True
    if not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
    except (ValueError, AttributeError, TypeError):
        return False
    return True


def validate_workflow_type(value: Any) -> bool:
    """
    Check that ``value`` is a recognised workflow type.

    Args:
        value: Candidate workflow type.

    Returns:
        ``True`` when ``value`` is in :data:`VALID_WORKFLOW_TYPES`.
    """
    return isinstance(value, str) and value.lower() in VALID_WORKFLOW_TYPES


def validate_workflow_action(value: Any) -> bool:
    """
    Check that ``value`` is a recognised lifecycle action.

    Args:
        value: Candidate action name.

    Returns:
        ``True`` when ``value`` is in :data:`VALID_WORKFLOW_ACTIONS`.
    """
    return isinstance(value, str) and value.lower() in VALID_WORKFLOW_ACTIONS


class AuthManager:
    """
    Credential verifier.

    Verification requires a real credential store, so the verification methods
    are fail-closed stubs rather than permissive defaults.
    """

    def verify_token(self, token: str) -> SecurityContext:
        """
        Verify a bearer token.

        Args:
            token: Bearer token presented by the caller.

        Returns:
            The caller's :class:`SecurityContext`.

        Raises:
            StubNotImplementedError: Always - no credential store is wired up.
        """
        raise_stub(
            "AuthManager.verify_token",
            hint="validate the bearer token against the real credential store and return a SecurityContext",
        )

    def verify_api_key(self, api_key: str) -> SecurityContext:
        """
        Verify an API key.

        Args:
            api_key: API key presented by the caller.

        Returns:
            The caller's :class:`SecurityContext`.

        Raises:
            StubNotImplementedError: Always - no credential store is wired up.
        """
        raise_stub(
            "AuthManager.verify_api_key",
            hint="look the key up in the real credential store and return a SecurityContext",
        )

    def create_api_key(self, subject: str, scopes: Optional[frozenset] = None) -> str:
        """
        Issue a new API key.

        Args:
            subject: Principal the key is issued to.
            scopes: Scopes to grant.

        Returns:
            The newly issued API key.

        Raises:
            StubNotImplementedError: Always - key issuance must be persisted.
        """
        raise_stub(
            "AuthManager.create_api_key",
            hint="persist a hashed key for the subject and return the plaintext once",
        )


#: Process-wide auth manager, mirroring the real module's singleton.
auth_manager: AuthManager = AuthManager()


def verify_token(token: str) -> SecurityContext:
    """
    Module-level convenience wrapper around :meth:`AuthManager.verify_token`.

    Args:
        token: Bearer token presented by the caller.

    Returns:
        The caller's :class:`SecurityContext`.

    Raises:
        StubNotImplementedError: Always - no credential store is wired up.
    """
    return auth_manager.verify_token(token)


def require_auth(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[F], F]:
    """
    Authorisation decorator.

    This stub does **not** enforce authentication; it passes calls straight
    through and warns once. It exists so decorated modules remain importable.

    Args:
        *decorator_args: Accepted for signature compatibility.
        **decorator_kwargs: Accepted for signature compatibility.

    Returns:
        A decorator that returns the wrapped function unchanged in behaviour.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _warn_auth_not_enforced(f"require_auth({func.__name__})")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    # Support both @require_auth and @require_auth(scope="...")
    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorate(decorator_args[0])  # type: ignore[arg-type,return-value]
    return decorate


def validate_input(*decorator_args: Any, **decorator_kwargs: Any) -> Callable[[F], F]:
    """
    Input-validation decorator.

    This stub performs no schema validation; it passes calls through so
    decorated modules remain importable.

    Args:
        *decorator_args: Accepted for signature compatibility.
        **decorator_kwargs: Accepted for signature compatibility.

    Returns:
        A decorator that returns the wrapped function unchanged in behaviour.
    """

    def decorate(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    if len(decorator_args) == 1 and callable(decorator_args[0]) and not decorator_kwargs:
        return decorate(decorator_args[0])  # type: ignore[arg-type,return-value]
    return decorate
