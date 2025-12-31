"""
Sovereign-Grade Problem Decomposition System - API Key Management
Implements secure storage and management of LLM and external API keys.
"""

import os
import json
import secrets
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
import sqlite3
from contextlib import contextmanager
import base64
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


logger = logging.getLogger(__name__)


class APIKeyType(Enum):
    """Types of API keys"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENEVOLVE = "openevolve"
    CUSTOM = "custom"


class APIKeyStatus(Enum):
    """Status of API keys"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    EXPIRED = "expired"
    REVOKED = "revoked"


@dataclass
class APIKey:
    """API key model"""
    id: str
    key_hash: str  # Hashed version stored in DB
    key_prefix: str  # First few characters for identification
    name: str
    key_type: APIKeyType
    user_id: str
    created_by: str
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    usage_count: int = 0
    status: APIKeyStatus = APIKeyStatus.ACTIVE
    metadata: Dict[str, Any] = field(default_factory=dict)
    permissions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'key_prefix': self.key_prefix,
            'name': self.name,
            'key_type': self.key_type.value,
            'user_id': self.user_id,
            'created_by': self.created_by,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_count': self.usage_count,
            'status': self.status.value,
            'metadata': self.metadata,
            'permissions': self.permissions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'APIKey':
        return cls(
            id=data['id'],
            key_hash=data['key_hash'],
            key_prefix=data['key_prefix'],
            name=data['name'],
            key_type=APIKeyType(data['key_type']),
            user_id=data['user_id'],
            created_by=data['created_by'],
            created_at=datetime.fromisoformat(data['created_at']),
            expires_at=datetime.fromisoformat(data['expires_at']) if data['expires_at'] else None,
            last_used=datetime.fromisoformat(data['last_used']) if data['last_used'] else None,
            usage_count=data['usage_count'],
            status=APIKeyStatus(data['status']),
            metadata=data['metadata'],
            permissions=data['permissions']
        )


class APIKeyEncryption:
    """Handles encryption/decryption of API keys"""
    
    def __init__(self, encryption_key: Optional[bytes] = None):
        """
        Initialize encryption system.
        
        Args:
            encryption_key: Encryption key (if None, will be generated or retrieved from environment)
        """
        if encryption_key is None:
            # Get encryption key from environment or generate new one
            env_key = os.getenv('API_KEY_ENCRYPTION_KEY')
            if env_key:
                self.encryption_key = base64.urlsafe_b64decode(env_key)
            else:
                # Generate a new key and store it in environment for future use
                self.encryption_key = Fernet.generate_key()
                print(f"Generated encryption key. Set API_KEY_ENCRYPTION_KEY={base64.urlsafe_b64decode(self.encryption_key).decode()} in your environment for future use.")
        else:
            self.encryption_key = encryption_key
        
        self.cipher = Fernet(self.encryption_key)
    
    def encrypt_key(self, api_key: str) -> str:
        """Encrypt an API key."""
        encrypted = self.cipher.encrypt(api_key.encode())
        return base64.urlsafe_b64encode(encrypted).decode()
    
    def decrypt_key(self, encrypted_key: str) -> str:
        """Decrypt an API key."""
        encrypted_data = base64.urlsafe_b64decode(encrypted_key.encode())
        decrypted = self.cipher.decrypt(encrypted_data)
        return decrypted.decode()


class APIKeyDatabase:
    """Database for API key management"""
    
    def __init__(self, db_path: str = "sovereign_decomposition.db"):
        self.db_path = db_path
        self.encryption = APIKeyEncryption()
        self.init_database()
    
    def init_database(self):
        """Initialize database schema for API keys."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # API Keys table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_keys (
                    id TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    key_prefix TEXT NOT NULL,
                    name TEXT NOT NULL,
                    key_type TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT,
                    last_used TEXT,
                    usage_count INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    encrypted_key TEXT  -- Only for temporary purposes, not for storage
                )
            """)
            
            # API Key usage logs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS api_key_usage_logs (
                    id TEXT PRIMARY KEY,
                    key_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    endpoint TEXT,
                    success INTEGER NOT NULL,
                    response_time REAL,
                    metadata TEXT,
                    FOREIGN KEY (key_id) REFERENCES api_keys(id)
                )
            """)
            
            # Create indexes
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_user_id 
                ON api_keys(user_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_type 
                ON api_keys(key_type)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_status 
                ON api_keys(status)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_keys_last_used 
                ON api_keys(last_used)
            """)
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def create_api_key(
        self, 
        name: str, 
        key_type: APIKeyType, 
        user_id: str, 
        created_by: str,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> APIKey:
        """Create a new API key."""
        from sovereign_data_models import generate_id
        
        # Generate a unique ID for the key
        key_id = generate_id("api_key")
        
        # Create a random API key
        raw_key = f"sk-{secrets.token_urlsafe(32)}"
        
        # Hash the key for storage (we don't store the actual key for security)
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        
        # Store first 8 characters of the key for identification
        key_prefix = raw_key[:8]
        
        # Calculate expiry date if specified
        expires_at = None
        if expires_in_days:
            expires_at = datetime.now() + timedelta(days=expires_in_days)
        
        # Create API key object
        api_key = APIKey(
            id=key_id,
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            key_type=key_type,
            user_id=user_id,
            created_by=created_by,
            created_at=datetime.now(),
            expires_at=expires_at,
            status=APIKeyStatus.ACTIVE,
            permissions=permissions or [],
            metadata=metadata or {}
        )
        
        # Store in database
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_keys (
                    id, key_hash, key_prefix, name, key_type, user_id, 
                    created_by, created_at, expires_at, status, metadata, permissions
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                api_key.id, api_key.key_hash, api_key.key_prefix, api_key.name,
                api_key.key_type.value, api_key.user_id, api_key.created_by,
                api_key.created_at.isoformat(), 
                api_key.expires_at.isoformat() if api_key.expires_at else None,
                api_key.status.value, json.dumps(api_key.metadata), 
                json.dumps(api_key.permissions)
            ))
        
        logger.info(f"Created new API key: {api_key.id} for user: {user_id}")
        
        # Return the raw key along with the APIKey object so the user can use it
        # The raw key is returned but not stored in the database for security
        return api_key, raw_key
    
    def get_api_key(self, key_id: str) -> Optional[APIKey]:
        """Retrieve an API key by ID."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys WHERE id = ?", (key_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = dict(row)
            data['metadata'] = json.loads(data['metadata'])
            data['permissions'] = json.loads(data['permissions'])
            return APIKey.from_dict(data)
    
    def get_api_key_by_hash(self, key_hash: str) -> Optional[APIKey]:
        """Retrieve an API key by its hash (for validation)."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM api_keys WHERE key_hash = ?", (key_hash,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            data = dict(row)
            data['metadata'] = json.loads(data['metadata'])
            data['permissions'] = json.loads(data['permissions'])
            return APIKey.from_dict(data)
    
    def list_api_keys(self, user_id: Optional[str] = None, key_type: Optional[APIKeyType] = None) -> List[APIKey]:
        """List API keys with optional filters."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            query = "SELECT * FROM api_keys WHERE 1=1"
            params = []
            
            if user_id:
                query += " AND user_id = ?"
                params.append(user_id)
            
            if key_type:
                query += " AND key_type = ?"
                params.append(key_type.value)
            
            query += " ORDER BY created_at DESC"
            
            cursor.execute(query, params)
            api_keys = []
            
            for row in cursor.fetchall():
                data = dict(row)
                data['metadata'] = json.loads(data['metadata'])
                data['permissions'] = json.loads(data['permissions'])
                api_keys.append(APIKey.from_dict(data))
            
            return api_keys
    
    def update_api_key(self, api_key: APIKey) -> bool:
        """Update an existing API key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE api_keys SET
                    name = ?, status = ?, metadata = ?, permissions = ?,
                    expires_at = ?
                WHERE id = ?
            """, (
                api_key.name, api_key.status.value, json.dumps(api_key.metadata),
                json.dumps(api_key.permissions),
                api_key.expires_at.isoformat() if api_key.expires_at else None,
                api_key.id
            ))
            return cursor.rowcount > 0
    
    def delete_api_key(self, key_id: str, user_id: str) -> bool:
        """Delete an API key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM api_keys 
                WHERE id = ? AND user_id = ?
            """, (key_id, user_id))
            return cursor.rowcount > 0
    
    def revoke_api_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        api_key = self.get_api_key(key_id)
        if not api_key or api_key.user_id != user_id:
            return False
        
        api_key.status = APIKeyStatus.REVOKED
        return self.update_api_key(api_key)
    
    def log_api_key_usage(
        self, 
        key_id: str, 
        user_id: str, 
        endpoint: str, 
        success: bool, 
        response_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log API key usage."""
        from sovereign_data_models import generate_id
        
        log_id = generate_id("usage_log")
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO api_key_usage_logs (
                    id, key_id, user_id, timestamp, endpoint, success, response_time, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log_id, key_id, user_id, datetime.now().isoformat(),
                endpoint, success, response_time, json.dumps(metadata or {})
            ))
        
        # Update usage count for the key
        api_key = self.get_api_key(key_id)
        if api_key:
            api_key.usage_count += 1
            api_key.last_used = datetime.now()
            self.update_api_key(api_key)
    
    def get_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get total usage count
            cursor.execute("""
                SELECT COUNT(*) as total_requests, 
                       AVG(response_time) as avg_response_time,
                       SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful_requests
                FROM api_key_usage_logs 
                WHERE key_id = ?
            """, (key_id,))
            result = cursor.fetchone()
            
            return {
                'total_requests': result['total_requests'] or 0,
                'successful_requests': result['successful_requests'] or 0,
                'failed_requests': (result['total_requests'] or 0) - (result['successful_requests'] or 0),
                'average_response_time': result['avg_response_time'] or 0.0
            }


class APIKeyManager:
    """Main API key management class"""
    
    def __init__(self, db: APIKeyDatabase):
        self.db = db
    
    def create_key(
        self, 
        name: str, 
        key_type: APIKeyType, 
        user_id: str, 
        created_by: str,
        permissions: Optional[List[str]] = None,
        expires_in_days: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> tuple[APIKey, str]:
        """
        Create a new API key.
        
        Returns:
            tuple of (APIKey object, raw API key string)
        """
        return self.db.create_api_key(
            name=name,
            key_type=key_type,
            user_id=user_id,
            created_by=created_by,
            permissions=permissions,
            expires_in_days=expires_in_days,
            metadata=metadata
        )
    
    def validate_key(self, raw_key: str) -> Optional[APIKey]:
        """
        Validate an API key and return the APIKey object if valid.
        
        Args:
            raw_key: The raw API key string provided by the user
            
        Returns:
            APIKey object if valid, None otherwise
        """
        key_hash = hashlib.sha256(raw_key.encode()).hexdigest()
        api_key = self.db.get_api_key_by_hash(key_hash)
        
        if not api_key:
            logger.warning("Invalid API key provided")
            return None
        
        # Check if key is active
        if api_key.status != APIKeyStatus.ACTIVE:
            logger.warning(f"API key {api_key.id} is not active (status: {api_key.status.value})")
            return None
        
        # Check if key has expired
        if api_key.expires_at and api_key.expires_at < datetime.now():
            logger.warning(f"API key {api_key.id} has expired")
            api_key.status = APIKeyStatus.EXPIRED
            self.db.update_api_key(api_key)
            return None
        
        return api_key
    
    def list_keys(self, user_id: Optional[str] = None, key_type: Optional[APIKeyType] = None) -> List[APIKey]:
        """List API keys with optional filters."""
        return self.db.list_api_keys(user_id, key_type)
    
    def revoke_key(self, key_id: str, user_id: str) -> bool:
        """Revoke an API key."""
        return self.db.revoke_api_key(key_id, user_id)
    
    def delete_key(self, key_id: str, user_id: str) -> bool:
        """Delete an API key."""
        return self.db.delete_api_key(key_id, user_id)
    
    def get_key_usage_stats(self, key_id: str) -> Dict[str, Any]:
        """Get usage statistics for an API key."""
        return self.db.get_usage_stats(key_id)
    
    def log_key_usage(
        self, 
        key_id: str, 
        user_id: str, 
        endpoint: str, 
        success: bool, 
        response_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Log API key usage."""
        self.db.log_api_key_usage(key_id, user_id, endpoint, success, response_time, metadata)


class APIKeyRateLimiter:
    """Rate limiting for API keys"""
    
    def __init__(self, db: APIKeyDatabase):
        self.db = db
        self.limits = {
            APIKeyType.OPENAI: {'requests_per_minute': 1000, 'tokens_per_minute': 60000},
            APIKeyType.ANTHROPIC: {'requests_per_minute': 500, 'tokens_per_minute': 40000},
            APIKeyType.GOOGLE: {'requests_per_minute': 1200, 'tokens_per_minute': 50000},
            APIKeyType.OPENEVOLVE: {'requests_per_minute': 2000, 'tokens_per_minute': 100000},
            APIKeyType.CUSTOM: {'requests_per_minute': 100, 'tokens_per_minute': 10000}
        }
    
    def check_rate_limit(self, key_id: str, request_type: str = "request") -> tuple[bool, str]:
        """
        Check if the API key is within its rate limits.
        
        Args:
            key_id: The API key ID
            request_type: Type of request (request, token, etc.)
            
        Returns:
            tuple of (is_allowed, reason)
        """
        import time
        from datetime import datetime, timedelta
        
        # Get the API key
        api_key = self.db.get_api_key(key_id)
        if not api_key:
            return False, "Invalid API key"
        
        # Get the rate limits for this key type
        limits = self.limits.get(api_key.key_type, self.limits[APIKeyType.CUSTOM])
        
        # Check requests in the last minute
        one_minute_ago = datetime.now() - timedelta(minutes=1)
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) as recent_requests
                FROM api_key_usage_logs
                WHERE key_id = ? AND timestamp > ? AND success = 1
            """, (key_id, one_minute_ago.isoformat()))
            result = cursor.fetchone()
            recent_requests = result['recent_requests'] or 0
        
        if recent_requests >= limits['requests_per_minute']:
            return False, f"Rate limit exceeded: {recent_requests}/{limits['requests_per_minute']} requests per minute"
        
        return True, "OK"


# Global API key manager instance
_api_key_manager = None
_api_key_database = None


def get_api_key_manager() -> APIKeyManager:
    """Get the API key manager instance."""
    global _api_key_manager, _api_key_database
    if _api_key_manager is None:
        _api_key_database = APIKeyDatabase()
        _api_key_manager = APIKeyManager(_api_key_database)
    return _api_key_manager


def get_api_key_database() -> APIKeyDatabase:
    """Get the API key database instance."""
    global _api_key_database
    if _api_key_database is None:
        _api_key_database = APIKeyDatabase()
    return _api_key_database


# Example usage
if __name__ == "__main__":
    # Initialize the API key manager
    api_key_manager = get_api_key_manager()
    
    # Create a new API key for OpenAI
    api_key_obj, raw_key = api_key_manager.create_key(
        name="OpenAI Key for Research",
        key_type=APIKeyType.OPENAI,
        user_id="user_123",
        created_by="admin_456",
        permissions=["read", "write"],
        expires_in_days=30,
        metadata={"project": "research"}
    )
    
    print(f"Created API key: {api_key_obj.id}")
    print(f"Raw key (store securely): {raw_key}")
    print(f"Key prefix: {api_key_obj.key_prefix}")
    
    # Validate the key
    validated_key = api_key_manager.validate_key(raw_key)
    if validated_key:
        print(f"Key validation successful: {validated_key.id}")
    
    # List keys for the user
    user_keys = api_key_manager.list_keys(user_id="user_123")
    print(f"User has {len(user_keys)} API keys")
    
    # Log usage
    api_key_manager.log_key_usage(
        key_id=api_key_obj.id,
        user_id="user_123",
        endpoint="/v1/chat/completions",
        success=True,
        response_time=0.45
    )
    
    # Get usage stats
    stats = api_key_manager.get_key_usage_stats(api_key_obj.id)
    print(f"Usage stats: {stats}")
    
    # Check rate limits
    rate_limiter = APIKeyRateLimiter(api_key_manager.db)
    allowed, reason = rate_limiter.check_rate_limit(api_key_obj.id)
    print(f"Rate limit check: {allowed}, {reason}")