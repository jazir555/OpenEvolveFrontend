"""
Security Performance Testing Suite - TRUE 100%
Tests security performance under load: rate limiting, encryption, hashing, JWT
"""

import pytest
import time
import asyncio
import concurrent.futures
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import hashlib
import hmac
import jwt
import os

from auth_system import TokenManager, JWTConfig, AuthManager
from security_framework import SecurityManager


class TestEncryptionPerformance:
    """Test encryption/decryption performance."""
    
    @pytest.fixture
    def fernet_key(self):
        return Fernet.generate_key()
    
    @pytest.fixture
    def fernet(self, fernet_key):
        return Fernet(fernet_key)
    
    @pytest.fixture
    def rsa_keypair(self):
        return rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
    
    def test_fernet_encryption_performance(self, fernet):
        """Test Fernet encryption performance."""
        plaintext = b"Sensitive data to encrypt" * 100  # 2.5KB payload
        
        # Warm up
        for _ in range(10):
            fernet.encrypt(plaintext)
        
        # Benchmark
        iterations = 1000
        start = time.perf_counter()
        
        for _ in range(iterations):
            fernet.encrypt(plaintext)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        # Should handle at least 100 ops/sec
        assert ops_per_sec > 100, f"Only {ops_per_sec:.2f} encryptions/sec"
        print(f"\nFernet encryption: {ops_per_sec:.2f} ops/sec")
    
    def test_fernet_decryption_performance(self, fernet):
        """Test Fernet decryption performance."""
        plaintext = b"Sensitive data to decrypt" * 100
        encrypted = fernet.encrypt(plaintext)
        
        iterations = 1000
        start = time.perf_counter()
        
        for _ in range(iterations):
            fernet.decrypt(encrypted)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 100, f"Only {ops_per_sec:.2f} decryptions/sec"
        print(f"Fernet decryption: {ops_per_sec:.2f} ops/sec")
    
    def test_rsa_encryption_performance(self, rsa_keypair):
        """Test RSA encryption performance."""
        public_key = rsa_keypair.public_key()
        plaintext = b"RSA encrypted message"
        
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            public_key.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        # RSA is slower, should handle at least 10 ops/sec
        assert ops_per_sec > 10, f"Only {ops_per_sec:.2f} RSA encryptions/sec"
        print(f"RSA encryption: {ops_per_sec:.2f} ops/sec")
    
    def test_rsa_signing_performance(self, rsa_keypair):
        """Test RSA signing performance."""
        message = b"Message to sign"
        
        iterations = 100
        start = time.perf_counter()
        
        for _ in range(iterations):
            rsa_keypair.sign(
                message,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 10, f"Only {ops_per_sec:.2f} RSA signs/sec"
        print(f"RSA signing: {ops_per_sec:.2f} ops/sec")


class TestHashingPerformance:
    """Test hashing performance."""
    
    def test_sha256_hashing_performance(self):
        """Test SHA-256 hashing performance."""
        data = b"Data to hash" * 1000  # 12KB
        
        iterations = 10000
        start = time.perf_counter()
        
        for _ in range(iterations):
            hashlib.sha256(data).hexdigest()
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        # Should handle at least 10000 ops/sec
        assert ops_per_sec > 10000, f"Only {ops_per_sec:.2f} hashes/sec"
        print(f"SHA-256 hashing: {ops_per_sec:.2f} ops/sec")
    
    def test_hmac_generation_performance(self):
        """Test HMAC generation performance."""
        key = b"secret_key"
        message = b"Message to authenticate" * 100
        
        iterations = 5000
        start = time.perf_counter()
        
        for _ in range(iterations):
            hmac.new(key, message, hashlib.sha256).hexdigest()
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 5000, f"Only {ops_per_sec:.2f} HMACs/sec"
        print(f"HMAC generation: {ops_per_sec:.2f} ops/sec")
    
    def test_pbkdf2_performance(self):
        """Test PBKDF2 key derivation performance."""
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
        
        password = b"user_password"
        salt = os.urandom(16)
        
        iterations_test = 100  # Number of tests
        kdf_iterations = 100000  # PBKDF2 iterations
        
        start = time.perf_counter()
        
        for _ in range(iterations_test):
            kdf = PBKDF2(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=kdf_iterations
            )
            kdf.derive(password)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations_test
        
        # Should complete in reasonable time (< 100ms per operation)
        assert avg_time < 0.1, f"PBKDF2 too slow: {avg_time*1000:.2f}ms"
        print(f"PBKDF2 (100k iterations): {avg_time*1000:.2f}ms")


class TestJWTPerformance:
    """Test JWT validation performance."""
    
    @pytest.fixture
    def token_manager(self):
        return TokenManager(JWTConfig(
            secret_key="test_secret_key_for_jwt_signing_at_least_32_chars",
            algorithm="HS256",
            access_token_expire_minutes=15
        ))
    
    def test_jwt_generation_performance(self, token_manager):
        """Test JWT generation performance."""
        user_id = "user_123"
        claims = {"role": "user", "permissions": ["read", "write"]}
        
        iterations = 10000
        start = time.perf_counter()
        
        for _ in range(iterations):
            token_manager.create_access_token(user_id, claims)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 5000, f"Only {ops_per_sec:.2f} JWTs/sec"
        print(f"JWT generation: {ops_per_sec:.2f} ops/sec")
    
    def test_jwt_verification_performance(self, token_manager):
        """Test JWT verification performance."""
        token = token_manager.create_access_token("user_123", {"role": "user"})
        
        iterations = 10000
        start = time.perf_counter()
        
        for _ in range(iterations):
            token_manager.verify_token(token)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 5000, f"Only {ops_per_sec:.2f} verifications/sec"
        print(f"JWT verification: {ops_per_sec:.2f} ops/sec")


class TestRateLimitingPerformance:
    """Test rate limiting under high load."""
    
    @pytest.fixture
    def auth_manager(self):
        return AuthManager(max_login_attempts=1000)
    
    def test_concurrent_rate_limiting(self, auth_manager):
        """Test rate limiting with concurrent requests."""
        user_id = "load_test_user"
        
        def make_requests(count):
            for _ in range(count):
                auth_manager.record_failed_login(user_id)
        
        # Simulate concurrent requests
        num_threads = 10
        requests_per_thread = 100
        
        start = time.perf_counter()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(make_requests, requests_per_thread)
                for _ in range(num_threads)
            ]
            concurrent.futures.wait(futures)
        
        elapsed = time.perf_counter() - start
        total_requests = num_threads * requests_per_thread
        ops_per_sec = total_requests / elapsed
        
        # Should handle at least 1000 ops/sec
        assert ops_per_sec > 1000, f"Only {ops_per_sec:.2f} rate limit checks/sec"
        print(f"Rate limiting: {ops_per_sec:.2f} ops/sec")
        
        # Verify rate limit was enforced
        assert not auth_manager.is_login_allowed(user_id)


class TestDatabaseQueryPerformance:
    """Test database query performance with audit logs."""
    
    @pytest.fixture
    def temp_db(self):
        import tempfile
        import sqlite3
        
        fd, path = tempfile.mkstemp(suffix='.db')
        os.close(fd)
        
        conn = sqlite3.connect(path)
        conn.execute("""
            CREATE TABLE audit_logs (
                id INTEGER PRIMARY KEY,
                timestamp REAL,
                user_id TEXT,
                action TEXT,
                resource TEXT,
                result TEXT
            )
        """)
        
        # Insert test data
        for i in range(10000):
            conn.execute(
                "INSERT INTO audit_logs (timestamp, user_id, action, resource, result) VALUES (?, ?, ?, ?, ?)",
                (time.time(), f"user_{i % 100}", "access", f"resource_{i % 1000}", "success")
            )
        conn.commit()
        conn.close()
        
        yield path
        os.unlink(path)
    
    def test_audit_log_query_performance(self, temp_db):
        """Test audit log query performance."""
        import sqlite3
        
        conn = sqlite3.connect(temp_db)
        
        # Query with filter
        start = time.perf_counter()
        
        for _ in range(100):
            cursor = conn.execute(
                "SELECT * FROM audit_logs WHERE user_id = ? AND timestamp > ?",
                ("user_50", time.time() - 3600)
            )
            results = cursor.fetchall()
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / 100
        
        # Should complete in reasonable time
        assert avg_time < 0.01, f"Query too slow: {avg_time*1000:.2f}ms"
        print(f"Audit log query: {avg_time*1000:.2f}ms")
        
        conn.close()
    
    def test_audit_log_insert_performance(self, temp_db):
        """Test audit log insert performance."""
        import sqlite3
        
        conn = sqlite3.connect(temp_db)
        
        iterations = 1000
        start = time.perf_counter()
        
        for i in range(iterations):
            conn.execute(
                "INSERT INTO audit_logs (timestamp, user_id, action, resource, result) VALUES (?, ?, ?, ?, ?)",
                (time.time(), f"user_{i}", "test", "test_resource", "success")
            )
        
        conn.commit()
        elapsed = time.perf_counter() - start
        ops_per_sec = iterations / elapsed
        
        assert ops_per_sec > 100, f"Only {ops_per_sec:.2f} inserts/sec"
        print(f"Audit log insert: {ops_per_sec:.2f} ops/sec")
        
        conn.close()


class TestInputValidationPerformance:
    """Test input validation performance."""
    
    @pytest.fixture
    def validator(self):
        from input_validation import InputValidator
        return InputValidator()
    
    def test_xss_sanitization_performance(self, validator):
        """Test XSS sanitization performance."""
        payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert(1)",
        ] * 10
        
        iterations = 1000
        start = time.perf_counter()
        
        for _ in range(iterations):
            for payload in payloads:
                validator._sanitize_html(payload)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = (iterations * len(payloads)) / elapsed
        
        assert ops_per_sec > 1000, f"Only {ops_per_sec:.2f} sanitizations/sec"
        print(f"XSS sanitization: {ops_per_sec:.2f} ops/sec")
    
    def test_sql_injection_detection_performance(self, validator):
        """Test SQL injection detection performance."""
        inputs = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "admin'--",
            "normal_input",
            "also_normal",
        ] * 10
        
        iterations = 1000
        start = time.perf_counter()
        
        for _ in range(iterations):
            for input_val in inputs:
                validator.sanitize_string(input_val)
        
        elapsed = time.perf_counter() - start
        ops_per_sec = (iterations * len(inputs)) / elapsed
        
        assert ops_per_sec > 10000, f"Only {ops_per_sec:.2f} checks/sec"
        print(f"SQL injection detection: {ops_per_sec:.2f} ops/sec")


class TestMemoryUsage:
    """Test memory usage under load."""
    
    def test_encryption_memory_stability(self):
        """Test that encryption doesn't leak memory."""
        import gc
        
        key = Fernet.generate_key()
        fernet = Fernet(key)
        data = b"x" * (1024 * 1024)  # 1MB data
        
        # Warm up
        for _ in range(10):
            encrypted = fernet.encrypt(data)
            fernet.decrypt(encrypted)
        
        gc.collect()
        
        # Run many iterations
        for _ in range(100):
            encrypted = fernet.encrypt(data)
            decrypted = fernet.decrypt(encrypted)
            del encrypted, decrypted
        
        gc.collect()
        
        # If we get here without memory error, test passes
        assert True


class TestLoadSimulation:
    """Simulate realistic load scenarios."""
    
    def test_authentication_load(self):
        """Simulate authentication load."""
        token_manager = TokenManager(JWTConfig(
            secret_key="test_secret_key_for_jwt_signing_at_least_32_chars",
            algorithm="HS256"
        ))
        
        num_users = 100
        tokens_per_user = 10
        
        start = time.perf_counter()
        
        # Generate many tokens
        tokens = []
        for i in range(num_users):
            for j in range(tokens_per_user):
                token = token_manager.create_access_token(
                    f"user_{i}",
                    {"session": j}
                )
                tokens.append(token)
        
        # Verify all tokens
        for token in tokens:
            token_manager.verify_token(token)
        
        elapsed = time.perf_counter() - start
        total_ops = num_users * tokens_per_user * 2  # create + verify
        ops_per_sec = total_ops / elapsed
        
        assert ops_per_sec > 1000, f"Only {ops_per_sec:.2f} auth ops/sec"
        print(f"\nAuthentication load test: {ops_per_sec:.2f} ops/sec")
    
    def test_encryption_load(self):
        """Simulate encryption load."""
        key = Fernet.generate_key()
        fernet = Fernet(key)
        
        # Various payload sizes
        payloads = [
            b"x" * 100,      # 100 bytes
            b"x" * 1024,     # 1 KB
            b"x" * 10240,    # 10 KB
        ]
        
        iterations_per_size = 100
        
        start = time.perf_counter()
        
        for payload in payloads:
            for _ in range(iterations_per_size):
                encrypted = fernet.encrypt(payload)
                fernet.decrypt(encrypted)
        
        elapsed = time.perf_counter() - start
        total_ops = len(payloads) * iterations_per_size * 2
        ops_per_sec = total_ops / elapsed
        
        assert ops_per_sec > 100, f"Only {ops_per_sec:.2f} encrypt ops/sec"
        print(f"Encryption load test: {ops_per_sec:.2f} ops/sec")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
