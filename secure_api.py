"""
Sovereign-Grade Problem Decomposition System - Secure API Communication
Implements encryption for sensitive data in transit and at rest.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
from enum import Enum
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import hashlib
import secrets

logger = logging.getLogger(__name__)


class DataEncryption:
    """Handles encryption/decryption of sensitive data"""
    
    def __init__(self, encryption_key: Optional[bytes] = None, password: Optional[str] = None):
        """
        Initialize encryption system.
        
        Args:
            encryption_key: Encryption key (if None, will be generated or retrieved from environment)
            password: Password to derive key from (alternative to encryption_key)
        """
        if encryption_key is not None:
            self.encryption_key = encryption_key
        elif password is not None:
            # Derive key from password with random salt
            # Generate a random 16-byte salt for each encryption
            salt = os.urandom(16)
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
            # Store salt with encryption key for later decryption
            self._encryption_salt = salt
            self.encryption_key = key
        else:
            # Get encryption key from environment or generate new one
            env_key = os.getenv('DATA_ENCRYPTION_KEY')
            if env_key:
                self.encryption_key = base64.urlsafe_b64decode(env_key.encode())
            else:
                # Generate a new key and store it in environment for future use
                self.encryption_key = Fernet.generate_key()
                encoded_key = base64.urlsafe_b64encode(self.encryption_key).decode()
                logger.info(f"Generated encryption key. Set DATA_ENCRYPTION_KEY={encoded_key} in your environment for future use.")
        
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_data(self, data: Union[str, bytes, Dict[str, Any]]) -> str:
        """Encrypt data and return as base64 encoded string."""
        if isinstance(data, dict):
            data = json.dumps(data)
        elif isinstance(data, bytes):
            data = data.decode('utf-8')
        
        encrypted = self.cipher.encrypt(data.encode('utf-8'))
        return base64.urlsafe_b64encode(encrypted).decode('utf-8')
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data from base64 encoded string."""
        encrypted_bytes = base64.urlsafe_b64decode(encrypted_data.encode('utf-8'))
        decrypted = self.cipher.decrypt(encrypted_bytes)
        return decrypted.decode('utf-8')
    
    def encrypt_dict(self, data: Dict[str, Any]) -> str:
        """Encrypt a dictionary of data."""
        return self.encrypt_data(data)
    
    def decrypt_to_dict(self, encrypted_data: str) -> Dict[str, Any]:
        """Decrypt data to a dictionary."""
        decrypted_str = self.decrypt_data(encrypted_data)
        return json.loads(decrypted_str)


class SecureAPIClient:
    """Secure API client for encrypted communication"""
    
    def __init__(self, base_url: str, encryption: DataEncryption, api_key_manager=None):
        self.base_url = base_url
        self.encryption = encryption
        self.api_key_manager = api_key_manager
        self.session = self._create_secure_session()
        logger.info(f"Initialized secure API client for {base_url}")
    
    def _create_secure_session(self):
        """Create a secure session with proper SSL/TLS settings."""
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Ensure secure connections
        session.verify = True  # Verify SSL certificates
        
        return session
    
    def make_secure_request(
        self, 
        endpoint: str, 
        method: str = 'GET', 
        data: Optional[Union[Dict[str, Any], str]] = None,
        headers: Optional[Dict[str, str]] = None,
        encrypt_payload: bool = True,
        decrypt_response: bool = True
    ) -> Union[Dict[str, Any], str]:
        """
        Make a secure API request with optional encryption.
        
        Args:
            endpoint: API endpoint
            method: HTTP method
            data: Request data
            headers: Additional headers
            encrypt_payload: Whether to encrypt the request payload
            decrypt_response: Whether to decrypt the response
            
        Returns:
            Decrypted response data or raw response
        """
        import requests
        
        full_url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        request_headers = headers or {}
        
        # Set content type
        if data:
            request_headers['Content-Type'] = 'application/json'
        
        # Encrypt payload if requested
        if data and encrypt_payload:
            encrypted_data = self.encryption.encrypt_data(data)
            payload = {'encrypted_data': encrypted_data}
        else:
            payload = data
        
        try:
            # Make the request
            response = self.session.request(
                method=method.upper(),
                url=full_url,
                json=payload,
                headers=request_headers
            )
            
            # Check for successful response
            response.raise_for_status()
            
            # Handle response
            response_data = response.json()
            
            # Decrypt response if requested
            if decrypt_response and 'encrypted_data' in response_data:
                decrypted_response = self.encryption.decrypt_to_dict(
                    response_data['encrypted_data']
                )
                return decrypted_response
            elif decrypt_response and 'encrypted_response' in response_data:
                decrypted_response = self.encryption.decrypt_data(
                    response_data['encrypted_response']
                )
                return json.loads(decrypted_response)
            else:
                return response_data
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"Secure API communication error: {e}")
            raise


class SecureStorage:
    """Secure storage for sensitive data at rest"""
    
    def __init__(self, encryption: DataEncryption, storage_path: str = "secure_storage.json"):
        self.encryption = encryption
        self.storage_path = storage_path
        self.data = self._load_storage()
    
    def _load_storage(self) -> Dict[str, str]:
        """Load encrypted data from storage."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                encrypted_data = f.read()
            try:
                decrypted_data = self.encryption.decrypt_data(encrypted_data)
                return json.loads(decrypted_data)
            except (OSError, IOError, ValueError, TypeError) as e:
                logger.error(f"Failed to decrypt storage: {e}")
                return {}
        return {}
    
    def _save_storage(self):
        """Save encrypted data to storage."""
        try:
            encrypted_data = self.encryption.encrypt_data(self.data)
            with open(self.storage_path, 'w') as f:
                f.write(encrypted_data)
        except (OSError, IOError, TypeError, ValueError) as e:
            logger.error(f"Failed to encrypt and save storage: {e}")
            raise
    
    def store(self, key: str, value: Union[str, Dict[str, Any], Any]):
        """Store sensitive data securely."""
        # Convert to JSON string if it's not already
        if not isinstance(value, str):
            value = json.dumps(value)
        
        # Encrypt the value
        encrypted_value = self.encryption.encrypt_data(value)
        
        # Store in memory
        self.data[key] = encrypted_value
        
        # Persist to disk
        self._save_storage()
        logger.info(f"Securely stored data for key: {key}")
    
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve and decrypt sensitive data."""
        if key not in self.data:
            return None
        
        encrypted_value = self.data[key]
        
        try:
            decrypted_value = self.encryption.decrypt_data(encrypted_value)
            # Try to parse as JSON, return as string if not valid JSON
            try:
                return json.loads(decrypted_value)
            except json.JSONDecodeError:
                return decrypted_value
        except (ValueError, TypeError) as e:
            logger.error(f"Failed to decrypt data for key {key}: {e}")
            return None
    
    def delete(self, key: str):
        """Delete sensitive data."""
        if key in self.data:
            del self.data[key]
            self._save_storage()
            logger.info(f"Deleted secure data for key: {key}")
    
    def list_keys(self) -> list:
        """List all stored keys."""
        return list(self.data.keys())


class CertificateManager:
    """Manage SSL/TLS certificates for secure communication"""
    
    def __init__(self, cert_file: Optional[str] = None, key_file: Optional[str] = None):
        self.cert_file = cert_file
        self.key_file = key_file
        self._cert = None
        self._private_key = None
        
        if cert_file and key_file:
            self.load_certificates()
    
    def generate_self_signed_cert(self, common_name: str = "localhost") -> tuple[str, str]:
        """Generate a self-signed certificate for testing."""
        # Generate private key
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        
        # Generate certificate
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        import ipaddress
        
        subject = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, common_name),
        ])
        
        # Set certificate validity
        valid_from = datetime.utcnow()
        valid_until = valid_from + timedelta(days=365)
        
        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            subject
        ).public_key(
            private_key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            valid_from
        ).not_valid_after(
            valid_until
        ).add_extension(
            x509.SubjectAlternativeName([
                x509.DNSName(common_name),
                x509.DNSName("localhost"),
                x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
            ]),
            critical=False,
        ).sign(private_key, hashes.SHA256())
        
        # Serialize to PEM format
        cert_pem = cert.public_bytes(serialization.Encoding.PEM)
        key_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        # Save to files
        cert_path = f"{common_name}_cert.pem"
        key_path = f"{common_name}_key.pem"
        
        with open(cert_path, 'wb') as f:
            f.write(cert_pem)
        
        with open(key_path, 'wb') as f:
            f.write(key_pem)
        
        logger.info(f"Generated self-signed certificate: {cert_path}")
        logger.info(f"Generated private key: {key_path}")
        
        return cert_path, key_path
    
    def load_certificates(self):
        """Load certificates from files."""
        if self.cert_file and self.key_file:
            with open(self.cert_file, 'rb') as f:
                self._cert = serialization.load_pem_x509_certificate(f.read())
            
            with open(self.key_file, 'rb') as f:
                self._private_key = serialization.load_pem_private_key(f.read(), password=None)
            
            logger.info(f"Loaded certificate from {self.cert_file}")
    
    def verify_certificate(self, hostname: str) -> bool:
        """Verify that the certificate is valid for the given hostname."""
        if not self._cert:
            return False
        
        try:
            from cryptography.x509.oid import NameOID
            
            # Check common name
            common_name = self._cert.subject.get_attributes_for_oid(NameOID.COMMON_NAME)[0].value
            if common_name == hostname:
                return True
            
            # Check subject alternative names
            try:
                san_ext = self._cert.extensions.get_extension_for_oid(x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME)
                san_names = san_ext.value.get_values_for_type(x509.DNSName)
                if hostname in san_names:
                    return True
            except x509.ExtensionNotFound:
                logger.debug("Certificate has no subject alternative names extension")
            
            return False
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Certificate verification failed: {e}")
            return False


class SecureCommunicationManager:
    """Main manager for secure communication features"""
    
    def __init__(self, encryption_password: Optional[str] = None):
        self.encryption = DataEncryption(password=encryption_password)
        self.secure_storage = SecureStorage(self.encryption)
        self.certificate_manager = CertificateManager()
    
    def encrypt_sensitive_data(self, data: Union[str, Dict[str, Any]]) -> str:
        """Encrypt sensitive data."""
        return self.encryption.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> Union[Dict[str, Any], str]:
        """Decrypt sensitive data."""
        return self.encryption.decrypt_data(encrypted_data)
    
    def store_securely(self, key: str, value: Any):
        """Store sensitive data securely."""
        self.secure_storage.store(key, value)
    
    def retrieve_securely(self, key: str) -> Optional[Any]:
        """Retrieve securely stored data."""
        return self.secure_storage.retrieve(key)
    
    def make_secure_api_request(
        self,
        base_url: str,
        endpoint: str,
        method: str = 'GET',
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ) -> Union[Dict[str, Any], str]:
        """Make a secure API request."""
        client = SecureAPIClient(base_url, self.encryption)
        return client.make_secure_request(endpoint, method, data, headers)
    
    def generate_certificates(self, common_name: str = "localhost") -> tuple[str, str]:
        """Generate SSL/TLS certificates."""
        return self.certificate_manager.generate_self_signed_cert(common_name)


# Global secure communication manager instance
_secure_comm_manager = None


def get_secure_comm_manager() -> SecureCommunicationManager:
    """Get the secure communication manager instance."""
    global _secure_comm_manager
    if _secure_comm_manager is None:
        password = os.getenv('SECURE_COMM_PASSWORD')
        _secure_comm_manager = SecureCommunicationManager(encryption_password=password)
    return _secure_comm_manager


# Example usage
if __name__ == "__main__":
    # Initialize secure communication manager
    sec_comm = get_secure_comm_manager()
    
    # Example: Encrypt sensitive data
    sensitive_data = {
        "api_key": "secret-key-12345",
        "credentials": "sensitive-info",
        "config": {
            "endpoint": "https://api.example.com",
            "timeout": 30
        }
    }
    
    encrypted = sec_comm.encrypt_sensitive_data(sensitive_data)
    print(f"Encrypted data: {encrypted[:50]}...")
    
    decrypted = sec_comm.decrypt_sensitive_data(encrypted)
    print(f"Decrypted data: {decrypted}")
    
    # Example: Securely store data
    sec_comm.store_securely("api_config", sensitive_data)
    retrieved = sec_comm.retrieve_securely("api_config")
    print(f"Retrieved data: {retrieved}")
    
    # Example: Generate certificates (for testing)
    cert_path, key_path = sec_comm.generate_certificates("test-server")
    print(f"Certificates generated: {cert_path}, {key_path}")
