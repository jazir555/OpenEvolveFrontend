"""
Security Helper Functions

Provides encryption, decryption, and secure handling of sensitive data
such as API keys, passwords, and other credentials.
"""

import os
import base64
import logging
import hashlib
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from env_helpers import env_var_str, is_production, get_or_generate_secret_key

logger = logging.getLogger(__name__)


class SecurityError(Exception):
    """Raised when security operations fail."""
    pass


class EncryptionManager:
    """
    Manages encryption and decryption of sensitive data.

    Uses Fernet symmetric encryption with PBKDF2 key derivation.
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize encryption manager.

        Args:
            encryption_key: Optional encryption key. If not provided, will use
                          KEY_ENCRYPTION_KEY environment variable or generate
                          a temporary key for development.

        Raises:
            SecurityError: If encryption fails to initialize
        """
        self.encryption_key = encryption_key or env_var_str("KEY_ENCRYPTION_KEY")

        if not self.encryption_key:
            if is_production():
                raise SecurityError(
                    "KEY_ENCRYPTION_KEY environment variable must be set in production"
                )
            else:
                # Generate a temporary key for development
                self.encryption_key = get_or_generate_secret_key("KEY_ENCRYPTION_KEY")
                logger.warning(
                    "Using auto-generated encryption key for development. "
                    "This will change on restart! Set KEY_ENCRYPTION_KEY environment variable."
                )

        try:
            self.cipher = self._create_cipher()
        except Exception as e:
            raise SecurityError(f"Failed to initialize encryption cipher: {e}")

    def _create_cipher(self) -> Fernet:
        """Create Fernet cipher from encryption key."""
        # Use PBKDF2HMAC to derive a proper Fernet key from the encryption key
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'openevolve_encryption_salt',  # In production, use a random salt
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.encryption_key.encode()))
        return Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt plaintext string.

        Args:
            plaintext: String to encrypt

        Returns:
            Base64-encoded encrypted string

        Raises:
            SecurityError: If encryption fails
        """
        if not plaintext:
            return plaintext

        try:
            encrypted_bytes = self.cipher.encrypt(plaintext.encode())
            return base64.urlsafe_b64encode(encrypted_bytes).decode()
        except Exception as e:
            raise SecurityError(f"Encryption failed: {e}")

    def decrypt(self, ciphertext: str) -> str:
        """
        Decrypt ciphertext string.

        Args:
            ciphertext: Base64-encoded encrypted string

        Returns:
            Decrypted plaintext string

        Raises:
            SecurityError: If decryption fails
        """
        if not ciphertext:
            return ciphertext

        try:
            encrypted_bytes = base64.urlsafe_b64decode(ciphertext.encode())
            decrypted_bytes = self.cipher.decrypt(encrypted_bytes)
            return decrypted_bytes.decode()
        except Exception as e:
            raise SecurityError(f"Decryption failed: {e}")

    def encrypt_dict(self, data: Dict[str, Any], sensitive_keys: Optional[list] = None) -> Dict[str, Any]:
        """
        Encrypt sensitive values in a dictionary.

        Args:
            data: Dictionary to encrypt
            sensitive_keys: List of keys whose values should be encrypted.
                          If None, will auto-detect sensitive keys.

        Returns:
            Dictionary with sensitive values encrypted
        """
        if sensitive_keys is None:
            # Auto-detect sensitive keys
            sensitive_keys = self._detect_sensitive_keys(data)

        encrypted_data = data.copy()
        for key in sensitive_keys:
            if key in encrypted_data and isinstance(encrypted_data[key], str):
                encrypted_data[key] = self.encrypt(encrypted_data[key])

        return encrypted_data

    def decrypt_dict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decrypt all values in a dictionary that were encrypted by this manager.

        Args:
            data: Dictionary with potentially encrypted values

        Returns:
            Dictionary with decrypted values
        """
        decrypted_data = data.copy()
        for key, value in decrypted_data.items():
            if isinstance(value, str):
                try:
                    decrypted_data[key] = self.decrypt(value)
                except SecurityError:
                    # If decryption fails, assume the value wasn't encrypted
                    pass

        return decrypted_data

    def _detect_sensitive_keys(self, data: Dict[str, Any]) -> list:
        """
        Auto-detect sensitive keys in a dictionary.

        Args:
            data: Dictionary to scan

        Returns:
            List of keys that likely contain sensitive data
        """
        sensitive_patterns = [
            'api_key', 'apikey', 'api-key',
            'secret', 'password', 'token',
            'credential', 'auth', 'private',
            'key', 'session'
        ]

        sensitive_keys = []
        for key in data.keys():
            key_lower = key.lower()
            for pattern in sensitive_patterns:
                if pattern in key_lower:
                    sensitive_keys.append(key)
                    break

        return sensitive_keys


class APIKeyManager:
    """
    Manages secure storage and retrieval of API keys.

    Provides encryption for keys stored on disk and secure retrieval
    from environment variables.
    """

    def __init__(self, encryption_key: Optional[str] = None):
        """
        Initialize API key manager.

        Args:
            encryption_key: Optional encryption key for stored keys
        """
        self.encryption_manager = EncryptionManager(encryption_key)
        self._cache: Dict[str, str] = {}

    def get_api_key(self, provider: str, env_var_name: Optional[str] = None) -> Optional[str]:
        """
        Get API key for a provider, checking in this order:
        1. Environment variable
        2. Encrypted storage (if previously saved)
        3. Return None if not found

        Args:
            provider: Provider name (e.g., "openai", "anthropic")
            env_var_name: Environment variable name (e.g., "OPENAI_API_KEY")

        Returns:
            API key or None if not found
        """
        # Check cache first
        if provider in self._cache:
            return self._cache[provider]

        # Check environment variable
        if env_var_name:
            api_key = os.getenv(env_var_name)
            if api_key:
                self._cache[provider] = api_key
                return api_key

        # Check encrypted storage (file-based)
        api_key = self._load_encrypted_key(provider)
        if api_key:
            self._cache[provider] = api_key
            return api_key

        return None

    def save_api_key(self, provider: str, api_key: str) -> None:
        """
        Securely save API key to encrypted storage.

        Args:
            provider: Provider name
            api_key: API key to save

        Raises:
            SecurityError: If save fails
        """
        try:
            # Encrypt the key
            encrypted_key = self.encryption_manager.encrypt(api_key)

            # Save to file
            key_file = self._get_key_file_path(provider)
            with open(key_file, 'w') as f:
                f.write(encrypted_key)

            # Update cache
            self._cache[provider] = api_key

            logger.info(f"API key for {provider} saved securely")
        except Exception as e:
            raise SecurityError(f"Failed to save API key for {provider}: {e}")

    def _load_encrypted_key(self, provider: str) -> Optional[str]:
        """
        Load encrypted API key from storage.

        Args:
            provider: Provider name

        Returns:
            Decrypted API key or None if not found
        """
        key_file = self._get_key_file_path(provider)

        if not os.path.exists(key_file):
            return None

        try:
            with open(key_file, 'r') as f:
                encrypted_key = f.read().strip()

            # Decrypt the key
            api_key = self.encryption_manager.decrypt(encrypted_key)
            return api_key
        except Exception as e:
            logger.error(f"Failed to load encrypted API key for {provider}: {e}")
            return None

    def _get_key_file_path(self, provider: str) -> str:
        """Get file path for storing encrypted API key."""
        # Create secure directory in user home
        key_dir = os.path.expanduser("~/.openevolve/keys")
        os.makedirs(key_dir, mode=0o700, exist_ok=True)

        return os.path.join(key_dir, f"{provider}.key")

    def clear_cache(self) -> None:
        """Clear cached API keys from memory."""
        self._cache.clear()
        logger.info("API key cache cleared")

    def delete_api_key(self, provider: str) -> None:
        """
        Delete stored API key.

        Args:
            provider: Provider name
        """
        # Remove from cache
        if provider in self._cache:
            del self._cache[provider]

        # Remove from storage
        key_file = self._get_key_file_path(provider)
        if os.path.exists(key_file):
            os.remove(key_file)
            logger.info(f"API key for {provider} deleted")


def redact_sensitive_data(data: str, patterns: Optional[list] = None) -> str:
    """
    Redact sensitive data from strings for logging.

    Args:
        data: String to redact
        patterns: List of regex patterns to redact. If None, uses defaults.

    Returns:
        String with sensitive data redacted
    """
    if patterns is None:
        # Default patterns for common sensitive data
        patterns = [
            (r'sk-[A-Za-z0-9]{48}', '<REDACTED_API_KEY>'),  # OpenAI-style
            (r'sk-ant-[A-Za-z0-9]{95}', '<REDACTED_API_KEY>'),  # Anthropic-style
            (r'Bearer [A-Za-z0-9\-._~+/]+=*', '<REDACTED_TOKEN>'),  # Bearer tokens
            (r'password["\']?\s*[:=]\s*["\']?[^"\'\s]+', 'password=<REDACTED>'),  # Passwords
            (r'api_key["\']?\s*[:=]\s*["\']?[^"\'\s]+', 'api_key=<REDACTED>'),  # API keys
        ]

    import re

    redacted = data
    for pattern, replacement in patterns:
        redacted = re.sub(pattern, replacement, redacted, flags=re.IGNORECASE)

    return redacted


def hash_sensitive_data(data: str, salt: Optional[str] = None) -> str:
    """
    Hash sensitive data for comparison without storing the actual value.

    Args:
        data: Data to hash
        salt: Optional salt for hashing

    Returns:
        Hex-encoded hash
    """
    if salt is None:
        salt = "openevolve_default_salt"

    combined = f"{salt}{data}".encode()
    return hashlib.sha256(combined).hexdigest()


def validate_api_key_format(api_key: str, provider: str) -> bool:
    """
    Validate API key format for specific providers.

    Args:
        api_key: API key to validate
        provider: Provider name (openai, anthropic, google, etc.)

    Returns:
        True if format is valid, False otherwise
    """
    import re

    provider = provider.lower()

    if provider == "openai":
        # OpenAI keys start with 'sk-' and are 51 characters total
        return bool(re.match(r'^sk-[A-Za-z0-9]{48}$', api_key))

    elif provider == "anthropic":
        # Anthropic keys start with 'sk-ant-' and are longer
        return bool(re.match(r'^sk-ant-[A-Za-z0-9]{95}$', api_key))

    elif provider == "google":
        # Google API keys are alphanumeric, typically 39 characters
        return bool(re.match(r'^[A-Za-z0-9_-]{39}$', api_key))

    elif provider == "CREWAI":
        # CREWAI keys - flexible format
        return len(api_key) >= 20

    else:
        # Generic validation: at least 20 characters
        return len(api_key) >= 20


# Global instances
_encryption_manager: Optional[EncryptionManager] = None
_api_key_manager: Optional[APIKeyManager] = None


def get_encryption_manager() -> EncryptionManager:
    """Get or create global encryption manager."""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_api_key_manager() -> APIKeyManager:
    """Get or create global API key manager."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager
