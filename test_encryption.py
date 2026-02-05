"""
Comprehensive Encryption Security Tests
Tests for data encryption, decryption, key management, and secure storage.
"""

import pytest
import os
import json
import base64
import tempfile
from pathlib import Path
from typing import Dict, Any

# Test imports with fallback for missing dependencies
try:
    from secure_api import (
        DataEncryption, SecureAPIClient, SecureStorage, CertificateManager,
        SecureCommunicationManager, get_secure_comm_manager
    )
    SECURE_API_AVAILABLE = True
except ImportError:
    SECURE_API_AVAILABLE = False

try:
    from security_helpers import (
        EncryptionManager, APIKeyManager, get_encryption_manager, get_api_key_manager,
        redact_sensitive_data, hash_sensitive_data, validate_api_key_format
    )
    SECURITY_HELPERS_AVAILABLE = True
except ImportError:
    SECURITY_HELPERS_AVAILABLE = False


@pytest.mark.skipif(not SECURE_API_AVAILABLE, reason="secure_api module not available")
class TestDataEncryption:
    """Test DataEncryption class."""
    
    @staticmethod
    def generate_test_key():
        """Generate a test encryption key."""
        from cryptography.fernet import Fernet
        return Fernet.generate_key()
    
    @pytest.fixture
    def encryption(self):
        """Create a DataEncryption instance with a test key."""
        return DataEncryption(encryption_key=self.generate_test_key())
    
    def test_key_generation(self):
        """Test encryption key generation."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        assert isinstance(key, bytes)
        assert len(key) > 0
    
    def test_string_encryption_decryption(self, encryption):
        """Test encryption and decryption of strings."""
        plaintext = "This is a secret message"
        encrypted = encryption.encrypt_data(plaintext)
        
        # Encrypted data should be different from plaintext
        assert encrypted != plaintext
        assert isinstance(encrypted, str)
        
        # Decrypt and verify
        decrypted = encryption.decrypt_data(encrypted)
        assert decrypted == plaintext
    
    def test_dict_encryption_decryption(self, encryption):
        """Test encryption and decryption of dictionaries."""
        data = {
            "api_key": "secret-key-123",
            "credentials": "sensitive-info",
            "nested": {
                "password": "super-secret"
            }
        }
        
        encrypted = encryption.encrypt_dict(data)
        assert isinstance(encrypted, str)
        
        decrypted = encryption.decrypt_to_dict(encrypted)
        assert decrypted == data
    
    def test_bytes_encryption(self, encryption):
        """Test encryption of bytes."""
        plaintext = b"Binary data for encryption"
        encrypted = encryption.encrypt_data(plaintext)
        decrypted = encryption.decrypt_data(encrypted)
        assert decrypted == plaintext.decode('utf-8')
    
    def test_empty_data_encryption(self, encryption):
        """Test encryption of empty data."""
        assert encryption.encrypt_data("") == ""
        assert encryption.decrypt_data("") == ""
    
    def test_unicode_encryption(self, encryption):
        """Test encryption of Unicode characters."""
        plaintext = "Hello 世界 🌍 ñoño émojis"
        encrypted = encryption.encrypt_data(plaintext)
        decrypted = encryption.decrypt_data(encrypted)
        assert decrypted == plaintext
    
    def test_large_data_encryption(self, encryption):
        """Test encryption of large data."""
        plaintext = "x" * 1000000  # 1MB of data
        encrypted = encryption.encrypt_data(plaintext)
        decrypted = encryption.decrypt_data(encrypted)
        assert decrypted == plaintext
    
    def test_different_keys_produce_different_ciphertexts(self):
        """Test that different keys produce different ciphertexts."""
        from cryptography.fernet import Fernet
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        
        enc1 = DataEncryption(encryption_key=key1)
        enc2 = DataEncryption(encryption_key=key2)
        
        plaintext = "Test message"
        cipher1 = enc1.encrypt_data(plaintext)
        cipher2 = enc2.encrypt_data(plaintext)
        
        assert cipher1 != cipher2
    
    def test_wrong_key_decryption_fails(self):
        """Test that decryption with wrong key fails."""
        from cryptography.fernet import Fernet
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        
        enc1 = DataEncryption(encryption_key=key1)
        enc2 = DataEncryption(encryption_key=key2)
        
        plaintext = "Test message"
        encrypted = enc1.encrypt_data(plaintext)
        
        # Decryption with wrong key should fail
        with pytest.raises(Exception):
            enc2.decrypt_data(encrypted)


@pytest.mark.skipif(not SECURE_API_AVAILABLE, reason="secure_api module not available")
class TestSecureStorage:
    """Test SecureStorage class."""
    
    @pytest.fixture
    def secure_storage(self):
        """Create a SecureStorage instance with a temporary file."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        encryption = DataEncryption(encryption_key=key)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            temp_path = f.name
        storage = SecureStorage(encryption, storage_path=temp_path)
        yield storage
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_store_and_retrieve_string(self, secure_storage):
        """Test storing and retrieving a string."""
        key = "test_key"
        value = "sensitive_value"
        
        secure_storage.store(key, value)
        retrieved = secure_storage.retrieve(key)
        assert retrieved == value
    
    def test_store_and_retrieve_dict(self, secure_storage):
        """Test storing and retrieving a dictionary."""
        key = "config"
        value = {
            "api_key": "sk-test-123",
            "endpoint": "https://api.example.com",
            "timeout": 30
        }
        
        secure_storage.store(key, value)
        retrieved = secure_storage.retrieve(key)
        assert retrieved == value
    
    def test_retrieve_nonexistent_key(self, secure_storage):
        """Test retrieving a non-existent key."""
        assert secure_storage.retrieve("nonexistent") is None
    
    def test_delete_key(self, secure_storage):
        """Test deleting a key."""
        key = "temp_key"
        value = "temp_value"
        
        secure_storage.store(key, value)
        assert secure_storage.retrieve(key) == value
        
        secure_storage.delete(key)
        assert secure_storage.retrieve(key) is None
    
    def test_list_keys(self, secure_storage):
        """Test listing all stored keys."""
        secure_storage.store("key1", "value1")
        secure_storage.store("key2", "value2")
        secure_storage.store("key3", "value3")
        
        keys = secure_storage.list_keys()
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys
    
    def test_persistence(self, secure_storage):
        """Test that data persists across storage instances."""
        key = DataEncryption.generate_key()
        encryption = DataEncryption(encryption_key=key)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as f:
            temp_path = f.name
        
        try:
            # Store data
            storage1 = SecureStorage(encryption, storage_path=temp_path)
            storage1.store("persistent_key", "persistent_value")
            
            # Create new instance with same file
            storage2 = SecureStorage(encryption, storage_path=temp_path)
            retrieved = storage2.retrieve("persistent_key")
            assert retrieved == "persistent_value"
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


@pytest.mark.skipif(not SECURE_API_AVAILABLE, reason="secure_api module not available")
class TestCertificateManager:
    """Test CertificateManager class."""
    
    @pytest.fixture
    def cert_manager(self):
        return CertificateManager()
    
    def test_self_signed_cert_generation(self, cert_manager, tmp_path):
        """Test generation of self-signed certificates."""
        cert_path, key_path = cert_manager.generate_self_signed_cert("test.local")
        
        # Verify files were created
        assert os.path.exists(cert_path)
        assert os.path.exists(key_path)
        
        # Verify certificate content
        with open(cert_path, 'rb') as f:
            cert_content = f.read()
            assert b"BEGIN CERTIFICATE" in cert_content
        
        with open(key_path, 'rb') as f:
            key_content = f.read()
            assert b"BEGIN PRIVATE KEY" in key_content
        
        # Cleanup
        os.unlink(cert_path)
        os.unlink(key_path)
    
    def test_certificate_loading(self, cert_manager, tmp_path):
        """Test loading certificates from files."""
        cert_path, key_path = cert_manager.generate_self_signed_cert("test.local")
        
        # Create new manager and load certificates
        manager2 = CertificateManager(cert_file=cert_path, key_file=key_path)
        assert manager2._cert is not None
        assert manager2._private_key is not None
        
        # Cleanup
        os.unlink(cert_path)
        os.unlink(key_path)
    
    def test_certificate_verification(self, cert_manager):
        """Test certificate hostname verification."""
        cert_path, key_path = cert_manager.generate_self_signed_cert("test.local")
        cert_manager.load_certificates()
        
        # Should match the common name
        assert cert_manager.verify_certificate("test.local") == True
        
        # Cleanup
        os.unlink(cert_path)
        os.unlink(key_path)


@pytest.mark.skipif(not SECURITY_HELPERS_AVAILABLE, reason="security_helpers module not available")
class TestEncryptionManager:
    """Test EncryptionManager class from security_helpers."""
    
    @pytest.fixture
    def enc_manager(self):
        return EncryptionManager(encryption_key="test-encryption-key-12345678901234567890")
    
    def test_encryption_manager_initialization(self):
        """Test EncryptionManager initialization."""
        manager = EncryptionManager(encryption_key="test-key-1234567890")
        assert manager is not None
        assert manager.encryption_key == "test-key-1234567890"
    
    def test_encrypt_decrypt(self, enc_manager):
        """Test encryption and decryption."""
        plaintext = "Secret message"
        encrypted = enc_manager.encrypt(plaintext)
        
        # Should be base64 encoded
        assert isinstance(encrypted, str)
        
        decrypted = enc_manager.decrypt(encrypted)
        assert decrypted == plaintext
    
    def test_empty_string_handling(self, enc_manager):
        """Test handling of empty strings."""
        assert enc_manager.encrypt("") == ""
        assert enc_manager.decrypt("") == ""
    
    def test_dict_encryption(self, enc_manager):
        """Test dictionary encryption."""
        data = {
            "password": "secret123",
            "api_key": "sk-test-456",
            "config": {"timeout": 30}
        }
        
        encrypted = enc_manager.encrypt_dict(data)
        
        # Sensitive keys should be encrypted
        assert encrypted["password"] != data["password"]
        assert encrypted["api_key"] != data["api_key"]
        # Non-sensitive keys should remain
        assert encrypted["config"] == data["config"]
    
    def test_dict_decryption(self, enc_manager):
        """Test dictionary decryption."""
        data = {
            "password": "secret123",
            "api_key": "sk-test-456",
            "normal": "value"
        }
        
        encrypted = enc_manager.encrypt_dict(data)
        decrypted = enc_manager.decrypt_dict(encrypted)
        
        assert decrypted["password"] == data["password"]
        assert decrypted["api_key"] == data["api_key"]
        assert decrypted["normal"] == data["normal"]
    
    def test_sensitive_key_detection(self, enc_manager):
        """Test auto-detection of sensitive keys."""
        data = {
            "username": "test",
            "password": "secret",
            "apiKey": "key123",
            "secret_token": "token456",
            "normal_field": "value"
        }
        
        sensitive_keys = enc_manager._detect_sensitive_keys(data)
        
        assert "password" in sensitive_keys
        assert "apiKey" in sensitive_keys
        assert "secret_token" in sensitive_keys
        assert "username" not in sensitive_keys
        assert "normal_field" not in sensitive_keys


@pytest.mark.skipif(not SECURITY_HELPERS_AVAILABLE, reason="security_helpers module not available")
class TestAPIKeyManager:
    """Test APIKeyManager class."""
    
    @pytest.fixture
    def key_manager(self):
        return APIKeyManager(encryption_key="test-encryption-key-12345678901234567890")
    
    def test_api_key_format_validation(self):
        """Test API key format validation."""
        # OpenAI format
        assert validate_api_key_format("sk-123456789012345678901234567890123456789012345678", "openai") == True
        assert validate_api_key_format("invalid-key", "openai") == False
        
        # Anthropic format
        assert validate_api_key_format("sk-ant-12345678901234567890123456789012345678901234567890123456789012345678901234567890", "anthropic") == True
        
        # Generic
        assert validate_api_key_format("valid-api-key-1234567890", "unknown") == True
        assert validate_api_key_format("short", "unknown") == False
    
    def test_get_api_key_from_environment(self, key_manager, monkeypatch):
        """Test getting API key from environment variable."""
        monkeypatch.setenv("TEST_API_KEY", "test-key-from-env")
        
        api_key = key_manager.get_api_key("test", env_var_name="TEST_API_KEY")
        assert api_key == "test-key-from-env"
    
    def test_save_and_load_api_key(self, key_manager, tmp_path):
        """Test saving and loading API keys."""
        # Mock the key file path
        key_manager._get_key_file_path = lambda p: str(tmp_path / f"{p}.key")
        
        # Save a key
        key_manager.save_api_key("test_provider", "test-api-key-12345")
        
        # Load the key (from cache)
        loaded = key_manager.get_api_key("test_provider")
        assert loaded == "test-api-key-12345"
        
        # Clear cache and load from file
        key_manager.clear_cache()
        loaded = key_manager.get_api_key("test_provider")
        assert loaded == "test-api-key-12345"
    
    def test_delete_api_key(self, key_manager, tmp_path):
        """Test deleting API keys."""
        key_manager._get_key_file_path = lambda p: str(tmp_path / f"{p}.key")
        
        key_manager.save_api_key("temp", "temp-key")
        assert key_manager.get_api_key("temp") == "temp-key"
        
        key_manager.delete_api_key("temp")
        assert key_manager.get_api_key("temp") is None
    
    def test_cache_clearing(self, key_manager):
        """Test clearing API key cache."""
        key_manager._cache["test"] = "value"
        assert "test" in key_manager._cache
        
        key_manager.clear_cache()
        assert len(key_manager._cache) == 0


@pytest.mark.skipif(not SECURITY_HELPERS_AVAILABLE, reason="security_helpers module not available")
class TestSensitiveDataRedaction:
    """Test sensitive data redaction functions."""
    
    def test_api_key_redaction(self):
        """Test API key redaction."""
        log_message = "Request with API key: sk-123456789012345678901234567890123456789012345678"
        redacted = redact_sensitive_data(log_message)
        assert "sk-1234567890" not in redacted
        assert "<REDACTED_API_KEY>" in redacted
    
    def test_bearer_token_redaction(self):
        """Test Bearer token redaction."""
        log_message = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
        redacted = redact_sensitive_data(log_message)
        assert "eyJhbGc" not in redacted
        assert "<REDACTED_TOKEN>" in redacted
    
    def test_password_redaction(self):
        """Test password redaction."""
        log_message = 'user login with password: "super-secret-123"'
        redacted = redact_sensitive_data(log_message)
        assert "super-secret-123" not in redacted
        assert "<REDACTED>" in redacted
    
    def test_anthropic_key_redaction(self):
        """Test Anthropic API key redaction."""
        log_message = "sk-ant-api03-test1234567890123456789012345678901234567890123456789012345678901234567890"
        redacted = redact_sensitive_data(log_message)
        # Anthropic keys should be redacted
        assert "<REDACTED" in redacted or "sk-ant" not in redacted
    
    def test_custom_patterns(self):
        """Test custom redaction patterns."""
        patterns = [
            (r'\bsecret_\w+\b', '[SECRET]'),
        ]
        message = "The secret_token was used"
        redacted = redact_sensitive_data(message, patterns=patterns)
        assert "secret_token" not in redacted
        assert "[SECRET]" in redacted


@pytest.mark.skipif(not SECURITY_HELPERS_AVAILABLE, reason="security_helpers module not available")
class TestHashing:
    """Test hashing functions."""
    
    def test_hash_sensitive_data(self):
        """Test sensitive data hashing."""
        data = "sensitive information"
        hash1 = hash_sensitive_data(data)
        hash2 = hash_sensitive_data(data)
        
        # Same data should produce same hash
        assert hash1 == hash2
        
        # Should be hex string
        assert isinstance(hash1, str)
        assert all(c in '0123456789abcdef' for c in hash1)
    
    def test_hash_with_salt(self):
        """Test hashing with custom salt."""
        data = "sensitive information"
        hash1 = hash_sensitive_data(data, salt="salt1")
        hash2 = hash_sensitive_data(data, salt="salt2")
        hash3 = hash_sensitive_data(data, salt="salt1")
        
        # Different salts should produce different hashes
        assert hash1 != hash2
        # Same salt should produce same hash
        assert hash1 == hash3
    
    def test_different_data_produces_different_hashes(self):
        """Test that different data produces different hashes."""
        hash1 = hash_sensitive_data("data1")
        hash2 = hash_sensitive_data("data2")
        
        assert hash1 != hash2


class TestEncryptionEdgeCases:
    """Test edge cases in encryption."""
    
    @pytest.mark.skipif(not SECURE_API_AVAILABLE, reason="secure_api module not available")
    def test_encryption_with_special_characters(self):
        """Test encryption of special characters."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        encryption = DataEncryption(encryption_key=key)
        
        special_chars = [
            "!@#$%^&*()_+-=[]{}|;':\",./<>?",
            "\n\r\t\0",
            "Hello\x00World",
            "🔐🔒🗝️",  # Emojis
        ]
        
        for char in special_chars:
            encrypted = encryption.encrypt_data(char)
            decrypted = encryption.decrypt_data(encrypted)
            assert decrypted == char
    
    @pytest.mark.skipif(not SECURE_API_AVAILABLE, reason="secure_api module not available")
    def test_multiple_encryption_operations(self):
        """Test multiple consecutive encryption operations."""
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        encryption = DataEncryption(encryption_key=key)
        
        data = [f"message_{i}" for i in range(100)]
        encrypted = [encryption.encrypt_data(d) for d in data]
        decrypted = [encryption.decrypt_data(e) for e in encrypted]
        
        assert decrypted == data
    
    @pytest.mark.skipif(not SECURITY_HELPERS_AVAILABLE, reason="security_helpers module not available")
    def test_corrupted_ciphertext(self):
        """Test handling of corrupted ciphertext."""
        manager = EncryptionManager(encryption_key="test-key-1234567890")
        
        plaintext = "Test message"
        encrypted = manager.encrypt(plaintext)
        
        # Corrupt the ciphertext
        corrupted = encrypted[:-5] + "XXXXX"
        
        # Decryption should fail
        with pytest.raises(Exception):
            manager.decrypt(corrupted)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
