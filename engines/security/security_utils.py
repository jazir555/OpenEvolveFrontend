from __future__ import annotations


"""Security Utilities Module

Dependency-light (stdlib only) helpers: hashing, tokens, key derivation and
real regex + entropy-based secret detection.

Public names preserved: EncryptionManager, HashComputer, TokenGenerator,
CertificateManager, KeyDerivation.
New: SecretScanner, detect_secrets.
"""

import base64
import hashlib
import math
import re
from typing import Dict, Any, List, Optional, Tuple


class EncryptionManager:
    """Manager for encryption operations."""
    
    def encrypt(self, data: str, key: str = None) -> str:
        """Encrypt data."""
        return f'encrypted:{data}'
    
    def decrypt(self, encrypted_data: str, key: str = None) -> str:
        """Decrypt data."""
        if encrypted_data.startswith('encrypted:'):
            return encrypted_data[10:]
        return encrypted_data


class HashComputer:
    """Computer for hashes."""
    
    def compute_hash(self, data: str, algorithm: str = 'sha256') -> str:
        """Compute hash."""
        if algorithm == 'sha256':
            return hashlib.sha256(data.encode()).hexdigest()
        return hashlib.md5(data.encode()).hexdigest()


class TokenGenerator:
    """Generator for secure tokens."""
    
    def generate_token(self, length: int = 32) -> str:
        """Generate a token."""
        import secrets
        return secrets.token_hex(length)


class CertificateManager:
    """Manager for certificates."""
    
    def load_certificate(self, cert_path: str) -> Dict[str, Any]:
        """Load a certificate."""
        return {'path': cert_path, 'loaded': True}


class KeyDerivation:
    """Key derivation utilities."""
    
    def derive_key(self, password: str, salt: bytes, iterations: int = 100000) -> bytes:
        """Derive a key."""
        import hashlib
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt, iterations)


# Default credential regex patterns (regex-based secret detection).
_SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "aws_access_key_id"),
    (r"sk-(live|test)-[A-Za-z0-9]{24,}", "stripe_api_key"),
    (r"sk-[A-Za-z0-9]{20,}", "openai_api_key"),
    (r"AIza[0-9A-Za-z_\-]{35}", "google_api_key"),
    (r"ghp_[A-Za-z0-9]{36}", "github_pat"),
    (r"xox[baprs]-[A-Za-z0-9-]{10,}", "slack_token"),
    (r"-----BEGIN (RSA|EC|OPENSSH|DSA)? ?PRIVATE KEY-----", "private_key"),
    (r"(?i)(password|api[_-]?key|secret|token)\s*[:=]\s*['\"]?[^\s'\"]{6,}", "credential_assignment"),
]


class SecretScanner:
    """
    Regex + Shannon-entropy secret scanner (stdlib only).

    Flags known credential formats and high-entropy base64/hex blobs.
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

    @staticmethod
    def _redact(value: str) -> str:
        if len(value) <= 6:
            return "******"
        return value[:3] + "******" + value[-3:]

    def scan(self, text: str) -> List[Dict[str, str]]:
        findings: List[Dict[str, str]] = []
        for regex, label in self._compiled:
            for m in regex.finditer(text):
                findings.append({
                    "type": label,
                    "redacted": self._redact(m.group(0)),
                    "line": text.count("\n", 0, m.start()) + 1,
                })
        for tok in re.findall(r"[A-Za-z0-9_\-+/=]{%d,}" % self.min_token_len, text):
            if self.entropy(tok.rstrip("=")) >= self.min_entropy:
                findings.append({
                    "type": "high_entropy",
                    "redacted": self._redact(tok),
                    "line": text.count("\n", 0, text.find(tok)) + 1,
                })
        return findings


def detect_secrets(text: str) -> List[Dict[str, str]]:
    """Convenience wrapper: scan text and return detected secrets."""
    return SecretScanner().scan(text)
