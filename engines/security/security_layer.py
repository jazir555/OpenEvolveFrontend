"""Security layer module

Real, dependency-light (stdlib only) access control, secret scanning and
input validation for OpenEvolve.

Public names preserved: EncryptionLevel, Permission, User, AuditEvent,
AccessPolicy, SecurityManager, AccessControlManager.
New: SecretScanner (regex + entropy based), InputValidator.
"""
from __future__ import annotations


import base64
import hashlib
import math
import os
import re
import secrets
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple


class EncryptionLevel(Enum):
    """Encryption level."""
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"


@dataclass
class Permission:
    """A single permission (resource:action)."""
    resource: str
    action: str

    def __str__(self) -> str:
        return f"{self.resource}:{self.action}"

    @classmethod
    def parse(cls, value: str) -> "Permission":
        res, _, act = value.partition(":")
        return cls(res, act or "*")


@dataclass
class User:
    """A user with roles and optional direct permissions."""
    user_id: str
    roles: Set[str] = field(default_factory=set)
    permissions: Set[str] = field(default_factory=set)

    def has_permission(self, perm: str) -> bool:
        if perm in self.permissions:
            return True
        resource, _, action = perm.partition(":")
        for r in self.roles:
            if r == perm or r == f"{resource}:*":
                return True
        return False


@dataclass
class AuditEvent:
    """An auditable security event."""
    actor: str
    action: str
    target: str
    allowed: bool
    timestamp: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class AccessPolicy:
    """Maps roles to the permissions they grant."""
    role_permissions: Dict[str, Set[str]] = field(default_factory=dict)

    def grant(self, role: str, perm: str) -> None:
        self.role_permissions.setdefault(role, set()).add(perm)

    def permissions_for(self, role: str) -> Set[str]:
        return self.role_permissions.get(role, set())


# Default secret-detection patterns (regex + entropy based).
_SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "aws_access_key_id"),
    (r"aws_secret_access_key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{40}", "aws_secret_key"),
    (r"sk-(live|test)-[A-Za-z0-9]{24,}", "stripe_api_key"),
    (r"sk-[A-Za-z0-9]{20,}", "openai_api_key"),
    (r"AIza[0-9A-Za-z_\-]{35}", "google_api_key"),
    (r"ghp_[A-Za-z0-9]{36}", "github_pat"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack_token"),
    (r"-----BEGIN (RSA|EC|OPENSSH|DSA)? ?PRIVATE KEY-----", "private_key"),
    (r"(?i)password\s*[:=]\s*['\"]?[^\s'\"]{6,}", "password_assignment"),
    (r"(?i)api[_-]?key\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}", "api_key_assignment"),
    (r"(?i)secret\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{12,}", "secret_assignment"),
    (r"(?i)token\s*[:=]\s*['\"]?[A-Za-z0-9_\-\.]{20,}", "token_assignment"),
]


class SecretScanner:
    """
    Regex + Shannon-entropy secret scanner.

    Detects well-known credential formats and falls back to flagging
    high-entropy base64/hex blobs that look like leaked secrets.
    """

    def __init__(self, patterns: Optional[List[Tuple[str, str]]] = None,
                 min_entropy: float = 3.5, min_token_len: int = 16):
        self._compiled = [(re.compile(p), label) for p, label in (patterns or _SECRET_PATTERNS)]
        self.min_entropy = min_entropy
        self.min_token_len = min_token_len

    @staticmethod
    def entropy(text: str) -> float:
        """Shannon entropy (bits/char) of a string."""
        if not text:
            return 0.0
        freq: Dict[str, int] = {}
        for ch in text:
            freq[ch] = freq.get(ch, 0) + 1
        n = len(text)
        return -sum((c / n) * math.log2(c / n) for c in freq.values())

    def scan(self, text: str) -> List[Dict[str, str]]:
        findings: List[Dict[str, str]] = []
        seen: Set[str] = set()
        for regex, label in self._compiled:
            for m in regex.finditer(text):
                key = f"{label}:{m.start()}"
                if key in seen:
                    continue
                seen.add(key)
                findings.append({
                    "type": label,
                    "redacted": self._redact(m.group(0)),
                    "line": text.count("\n", 0, m.start()) + 1,
                })
        # Entropy-based pass over candidate tokens (b64 / hex).
        for tok in re.findall(r"[A-Za-z0-9_\-+/=]{%d,}" % self.min_token_len, text):
            if any(f["redacted"] and tok in f["redacted"] or tok == f.get("type") for f in findings):
                continue
            if self._looks_secret(tok):
                findings.append({
                    "type": "high_entropy",
                    "redacted": self._redact(tok),
                    "line": text.count("\n", 0, text.find(tok)) + 1,
                })
        return findings

    def _looks_secret(self, tok: str) -> bool:
        stripped = tok.rstrip("=")
        if self.entropy(stripped) >= self.min_entropy:
            return True
        return False

    @staticmethod
    def _redact(value: str) -> str:
        if len(value) <= 6:
            return "******"
        return value[:3] + "******" + value[-3:]


class InputValidator:
    """Validates user-supplied input against simple constraints."""

    def validate_string(self, value: str, min_len: int = 0, max_len: int = 10000,
                        allow_chars: str = r"[\s\S]*") -> Tuple[bool, Optional[str]]:
        if not isinstance(value, str):
            return False, "not a string"
        if len(value) < min_len:
            return False, f"shorter than {min_len}"
        if len(value) > max_len:
            return False, f"longer than {max_len}"
        if not re.fullmatch(allow_chars, value):
            return False, "contains disallowed characters"
        return True, None

    def validate_email(self, value: str) -> bool:
        return bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value or ""))

    def validate_id(self, value: str) -> bool:
        return bool(re.fullmatch(r"[A-Za-z0-9_\-]{1,128}", value or ""))


class AccessControlManager:
    """Role/permission based access control with audit trail."""

    def __init__(self, policy: Optional[AccessPolicy] = None):
        self.policy = policy or AccessPolicy()
        self.audit_log: List[AuditEvent] = []

    def grant(self, role: str, perm: str) -> None:
        self.policy.grant(role, perm)

    def check_access(self, user: User, perm: str) -> bool:
        allowed = user.has_permission(perm) or any(
            perm in self.policy.permissions_for(r) for r in user.roles
        )
        self.audit_log.append(AuditEvent(actor=user.user_id, action=perm, target="*", allowed=allowed))
        return allowed

    def check_access_detailed(self, user: User, perm: str) -> Dict[str, Any]:
        allowed = self.check_access(user, perm)
        return {"user": user.user_id, "permission": perm, "allowed": allowed}


class SecurityManager:
    """
    Higher-level security facade combining RBAC, secret scanning and
    input validation. Passwords are hashed with PBKDF2 (no plaintext storage).
    """

    def __init__(self, policy: Optional[AccessPolicy] = None):
        self.acm = AccessControlManager(policy)
        self.secret_scanner = SecretScanner()
        self.input_validator = InputValidator()

    # --- RBAC ---
    def grant(self, role: str, perm: str) -> None:
        self.acm.grant(role, perm)

    def check_access(self, user: User, perm: str) -> bool:
        return self.acm.check_access(user, perm)

    # --- Secret scanning ---
    def scan_secrets(self, text: str) -> List[Dict[str, str]]:
        return self.secret_scanner.scan(text)

    # --- Input validation ---
    def validate_input(self, value: str, **kwargs) -> Tuple[bool, Optional[str]]:
        return self.input_validator.validate_string(value, **kwargs)

    # --- Credential hashing ---
    @staticmethod
    def hash_password(password: str, iterations: int = 100_000) -> str:
        salt = secrets.token_hex(16)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), iterations)
        return f"pbkdf2${iterations}${salt}${dk.hex()}"

    @staticmethod
    def verify_password(password: str, stored: str) -> bool:
        try:
            algo, iters, salt, digest = stored.split("$")
            if algo != "pbkdf2":
                return False
            dk = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iters))
            return secrets.compare_digest(dk.hex(), digest)
        except (ValueError, AttributeError):
            return False
