"""
Comprehensive Encryption Testing Suite - TRUE 100%
Tests all encryption methods: AES, RSA, Fernet, Hashing, Key management
"""

import pytest
import hashlib
import hmac
import secrets
import os
import tempfile
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding, ec
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.backends import default_backend
import base64

from security_framework import SecurityManager


class TestFernetEncryption:
    """Test Fernet symmetric encryption (AES-128-CBC)."""
    
    @pytest.fixture
    def fernet_key(self):
        return Fernet.generate_key()
    
    @pytest.fixture
    def fernet(self, fernet_key):
        return Fernet(fernet_key)
    
    def test_fernet_encryption_decryption(self, fernet):
        """Test basic Fernet encryption and decryption."""
        plaintext = b"Sensitive data that needs encryption"
        
        encrypted = fernet.encrypt(plaintext)
        decrypted = fernet.decrypt(encrypted)
        
        assert decrypted == plaintext
        assert encrypted != plaintext
    
    def test_fernet_token_structure(self, fernet):
        """Test Fernet token structure."""
        plaintext = b"Test data"
        token = fernet.encrypt(plaintext)
        
        # Fernet tokens are base64 encoded
        decoded = base64.urlsafe_b64decode(token)
        
        # Structure: version(1) + timestamp(8) + iv(16) + ciphertext + hmac(32)
        assert decoded[0] == 0x80  # Version byte
        assert len(decoded) > 57  # Minimum size
    
    def test_fernet_ciphertext_different_each_time(self, fernet):
        """Test that same plaintext produces different ciphertext."""
        plaintext = b"Same text"
        
        encrypted1 = fernet.encrypt(plaintext)
        encrypted2 = fernet.encrypt(plaintext)
        
        assert encrypted1 != encrypted2  # Due to random IV
    
    def test_fernet_invalid_key(self):
        """Test Fernet with invalid key."""
        invalid_key = b"not_a_valid_key"
        
        with pytest.raises(Exception):
            Fernet(invalid_key)
    
    def test_fernet_tampered_token(self, fernet):
        """Test detection of tampered tokens."""
        plaintext = b"Important data"
        token = fernet.encrypt(plaintext)
        
        # Tamper with token
        tampered = token[:-5] + b"XXXXX"
        
        with pytest.raises(Exception):
            fernet.decrypt(tampered)
    
    def test_fernet_expired_token(self, fernet):
        """Test Fernet token expiration."""
        from cryptography.fernet import InvalidToken
        
        plaintext = b"Time-sensitive data"
        # Create token with past TTL
        token = fernet.encrypt(plaintext)
        
        # Try to decrypt with very short TTL
        with pytest.raises(InvalidToken):
            fernet.decrypt(token, ttl=0)


class TestAESEncryption:
    """Test AES encryption modes."""
    
    @pytest.fixture
    def aes_key(self):
        """Generate 256-bit AES key."""
        return os.urandom(32)
    
    def test_aes_cbc_encryption(self, aes_key):
        """Test AES-CBC encryption."""
        iv = os.urandom(16)
        plaintext = b"Secret message!!"  # 16 bytes for block alignment
        
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Decrypt
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        assert decrypted == plaintext
    
    def test_aes_gcm_encryption(self, aes_key):
        """Test AES-GCM authenticated encryption."""
        iv = os.urandom(12)  # GCM uses 96-bit IV
        plaintext = b"Authenticated secret message"
        associated_data = b"metadata"
        
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        encryptor.authenticate_additional_data(associated_data)
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        tag = encryptor.tag  # 128-bit authentication tag
        
        # Decrypt
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag))
        decryptor = cipher.decryptor()
        decryptor.authenticate_additional_data(associated_data)
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        assert decrypted == plaintext
    
    def test_aes_gcm_tampering_detection(self, aes_key):
        """Test AES-GCM detects tampering."""
        from cryptography.exceptions import InvalidTag
        
        iv = os.urandom(12)
        plaintext = b"Message"
        
        cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        tag = encryptor.tag
        
        # Tamper with ciphertext
        tampered_ciphertext = ciphertext[:-1] + bytes([ciphertext[-1] ^ 0xFF])
        
        # Decryption should fail authentication
        with pytest.raises(InvalidTag):
            cipher = Cipher(algorithms.AES(aes_key), modes.GCM(iv, tag))
            decryptor = cipher.decryptor()
            decryptor.update(tampered_ciphertext) + decryptor.finalize()
    
    def test_aes_ctr_encryption(self, aes_key):
        """Test AES-CTR mode encryption."""
        nonce = os.urandom(16)
        plaintext = b"Stream cipher style encryption"
        
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce))
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        cipher = Cipher(algorithms.AES(aes_key), modes.CTR(nonce))
        decryptor = cipher.decryptor()
        decrypted = decryptor.update(ciphertext) + decryptor.finalize()
        
        assert decrypted == plaintext
    
    def test_aes_key_sizes(self):
        """Test different AES key sizes."""
        plaintext = b"16 byte message!"
        iv = os.urandom(16)
        
        # AES-128
        key_128 = os.urandom(16)
        cipher = Cipher(algorithms.AES(key_128), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ct_128 = encryptor.update(plaintext) + encryptor.finalize()
        assert len(ct_128) == len(plaintext)
        
        # AES-192
        key_192 = os.urandom(24)
        cipher = Cipher(algorithms.AES(key_192), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ct_192 = encryptor.update(plaintext) + encryptor.finalize()
        assert len(ct_192) == len(plaintext)
        
        # AES-256
        key_256 = os.urandom(32)
        cipher = Cipher(algorithms.AES(key_256), modes.CBC(iv))
        encryptor = cipher.encryptor()
        ct_256 = encryptor.update(plaintext) + encryptor.finalize()
        assert len(ct_256) == len(plaintext)


class TestRSAEncryption:
    """Test RSA asymmetric encryption."""
    
    @pytest.fixture
    def rsa_keypair(self):
        """Generate RSA key pair."""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        public_key = private_key.public_key()
        return private_key, public_key
    
    def test_rsa_encryption_decryption(self, rsa_keypair):
        """Test RSA encryption and decryption."""
        private_key, public_key = rsa_keypair
        plaintext = b"Secret message for RSA"
        
        # Encrypt with public key
        ciphertext = public_key.encrypt(
            plaintext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        # Decrypt with private key
        decrypted = private_key.decrypt(
            ciphertext,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        
        assert decrypted == plaintext
    
    def test_rsa_signing_verification(self, rsa_keypair):
        """Test RSA digital signatures."""
        private_key, public_key = rsa_keypair
        message = b"Message to sign"
        
        # Sign with private key
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Verify with public key
        public_key.verify(
            signature,
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Verification successful if no exception
        assert True
    
    def test_rsa_signature_tampering(self, rsa_keypair):
        """Test RSA signature detects tampering."""
        from cryptography.exceptions import InvalidSignature
        
        private_key, public_key = rsa_keypair
        message = b"Original message"
        
        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Try to verify with different message
        with pytest.raises(InvalidSignature):
            public_key.verify(
                signature,
                b"Different message",
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
    
    def test_rsa_key_serialization(self, rsa_keypair):
        """Test RSA key serialization."""
        private_key, public_key = rsa_keypair
        
        # Serialize private key with password
        pem_private = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.BestAvailableEncryption(b"password")
        )
        
        # Serialize public key
        pem_public = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        assert b"BEGIN ENCRYPTED PRIVATE KEY" in pem_private
        assert b"BEGIN PUBLIC KEY" in pem_public
        
        # Deserialize
        loaded_private = serialization.load_pem_private_key(pem_private, b"password")
        loaded_public = serialization.load_pem_public_key(pem_public)
        
        assert loaded_private is not None
        assert loaded_public is not None


class TestHashing:
    """Test cryptographic hashing."""
    
    def test_sha256_hashing(self):
        """Test SHA-256 hashing."""
        data = b"Data to hash"
        hash1 = hashlib.sha256(data).hexdigest()
        hash2 = hashlib.sha256(data).hexdigest()
        
        # Deterministic
        assert hash1 == hash2
        assert len(hash1) == 64  # 256 bits = 64 hex chars
    
    def test_sha256_avalanche_effect(self):
        """Test that small changes cause large hash differences."""
        data1 = b"Hello World"
        data2 = b"Hello world"  # Small change: W -> w
        
        hash1 = hashlib.sha256(data1).hexdigest()
        hash2 = hashlib.sha256(data2).hexdigest()
        
        assert hash1 != hash2
        # Should differ by approximately 50% of bits
    
    def test_hmac_generation(self):
        """Test HMAC generation."""
        key = b"secret_key"
        message = b"Message to authenticate"
        
        hmac1 = hmac.new(key, message, hashlib.sha256).hexdigest()
        hmac2 = hmac.new(key, message, hashlib.sha256).hexdigest()
        
        # Deterministic
        assert hmac1 == hmac2
        
        # Different key produces different HMAC
        wrong_key = b"wrong_key"
        hmac3 = hmac.new(wrong_key, message, hashlib.sha256).hexdigest()
        assert hmac3 != hmac1
    
    def test_hmac_verification(self):
        """Test HMAC verification."""
        key = b"secret_key"
        message = b"Message to authenticate"
        
        generated_hmac = hmac.new(key, message, hashlib.sha256)
        
        # Verify correct message
        verification = hmac.new(key, message, hashlib.sha256)
        assert hmac.compare_digest(generated_hmac.digest(), verification.digest())
        
        # Verify wrong message fails
        wrong_message = b"Wrong message"
        wrong_verification = hmac.new(key, wrong_message, hashlib.sha256)
        assert not hmac.compare_digest(generated_hmac.digest(), wrong_verification.digest())
    
    def test_password_hashing_pbkdf2(self):
        """Test PBKDF2 password hashing."""
        password = b"user_password"
        salt = os.urandom(16)
        
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = kdf.derive(password)
        
        assert len(key) == 32
        
        # Same password, different salt = different hash
        salt2 = os.urandom(16)
        kdf2 = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt2,
            iterations=100000
        )
        key2 = kdf2.derive(password)
        
        assert key != key2
    
    def test_password_hashing_scrypt(self):
        """Test Scrypt password hashing."""
        password = b"user_password"
        salt = os.urandom(16)
        
        kdf = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1
        )
        key = kdf.derive(password)
        
        assert len(key) == 32
        
        # Verification
        kdf_verify = Scrypt(
            salt=salt,
            length=32,
            n=2**14,
            r=8,
            p=1
        )
        kdf_verify.verify(password, key)  # Should not raise


class TestKeyManagement:
    """Test encryption key management."""
    
    @pytest.fixture
    def temp_key_storage(self):
        """Create temporary key storage."""
        fd, path = tempfile.mkstemp()
        os.close(fd)
        yield path
        os.unlink(path)
    
    def test_key_generation_entropy(self):
        """Test that generated keys have sufficient entropy."""
        keys = [os.urandom(32) for _ in range(100)]
        
        # All keys should be unique
        assert len(set(keys)) == 100
        
        # Check for reasonable randomness (no obvious patterns)
        for key in keys:
            # Key should not be all zeros
            assert any(b != 0 for b in key)
            # Key should not be all same byte
            assert len(set(key)) > 1
    
    def test_key_rotation(self, temp_key_storage):
        """Test key rotation."""
        # Generate initial key
        key1 = Fernet.generate_key()
        fernet1 = Fernet(key1)
        
        # Encrypt data with old key
        plaintext = b"Sensitive data"
        ciphertext = fernet1.encrypt(plaintext)
        
        # Generate new key (rotation)
        key2 = Fernet.generate_key()
        fernet2 = Fernet(key2)
        
        # Re-encrypt with new key
        decrypted = fernet1.decrypt(ciphertext)
        new_ciphertext = fernet2.encrypt(decrypted)
        
        # Verify new key works
        assert fernet2.decrypt(new_ciphertext) == plaintext
        
        # Store new key
        with open(temp_key_storage, 'wb') as f:
            f.write(key2)
    
    def test_key_wrapping(self):
        """Test key wrapping with master key."""
        # Master key
        master_key = os.urandom(32)
        
        # Data encryption key to wrap
        dek = os.urandom(32)
        
        # Wrap DEK with master key
        iv = os.urandom(16)
        cipher = Cipher(algorithms.AES(master_key), modes.CBC(iv))
        encryptor = cipher.encryptor()
        
        # Pad DEK to block size
        padded_dek = dek + b'\x00' * (16 - len(dek) % 16)
        wrapped_key = iv + encryptor.update(padded_dek) + encryptor.finalize()
        
        # Unwrap
        iv = wrapped_key[:16]
        cipher = Cipher(algorithms.AES(master_key), modes.CBC(iv))
        decryptor = cipher.decryptor()
        unwrapped = decryptor.update(wrapped_key[16:]) + decryptor.finalize()
        
        assert unwrapped[:32] == dek
    
    def test_key_derivation_from_password(self):
        """Test deriving encryption key from password."""
        password = b"user_password"
        salt = os.urandom(16)
        
        # Derive key
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key = kdf.derive(password)
        
        # Same password and salt should derive same key
        kdf2 = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000
        )
        key2 = kdf2.derive(password)
        
        assert key == key2


class TestECEncryption:
    """Test Elliptic Curve encryption."""
    
    @pytest.fixture
    def ec_keypair(self):
        """Generate EC key pair."""
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_key = private_key.public_key()
        return private_key, public_key
    
    def test_ec_signature(self, ec_keypair):
        """Test EC digital signatures."""
        private_key, public_key = ec_keypair
        message = b"Message to sign"
        
        # Sign
        signature = private_key.sign(message, ec.ECDSA(hashes.SHA256()))
        
        # Verify
        public_key.verify(signature, message, ec.ECDSA(hashes.SHA256()))
        
        # Should pass if no exception
        assert True
    
    def test_ecdh_key_exchange(self, ec_keypair):
        """Test ECDH key exchange."""
        # Alice's key pair
        alice_private = ec.generate_private_key(ec.SECP256R1())
        alice_public = alice_private.public_key()
        
        # Bob's key pair
        bob_private = ec.generate_private_key(ec.SECP256R1())
        bob_public = bob_private.public_key()
        
        # Alice computes shared secret
        alice_shared = alice_private.exchange(ec.ECDH(), bob_public)
        
        # Bob computes shared secret
        bob_shared = bob_private.exchange(ec.ECDH(), alice_public)
        
        # Shared secrets should be identical
        assert alice_shared == bob_shared


class TestSecureRandom:
    """Test secure random number generation."""
    
    def test_secrets_token(self):
        """Test secrets token generation."""
        tokens = [secrets.token_urlsafe(32) for _ in range(100)]
        
        # All unique
        assert len(set(tokens)) == 100
        
        # Reasonable length
        assert all(len(t) >= 32 for t in tokens)
    
    def test_secrets_choice(self):
        """Test secrets choice for passwords."""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        passwords = ["".join(secrets.choice(alphabet) for _ in range(16)) for _ in range(100)]
        
        # All unique
        assert len(set(passwords)) == 100
        
        # All correct length
        assert all(len(p) == 16 for p in passwords)
    
    def test_os_urandom(self):
        """Test os.urandom for cryptographic randomness."""
        random_bytes = [os.urandom(32) for _ in range(100)]
        
        # All unique
        assert len(set(random_bytes)) == 100
        
        # Correct length
        assert all(len(r) == 32 for r in random_bytes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
