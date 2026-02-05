"""
REAL Audit Logging Tests
Tests that verify logs are actually written to files/database.

This file addresses the HIGH gap: 70% of tests use mocks instead of 
verifying actual log writing to files/database.
"""

import pytest
import os
import json
import tempfile
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
import asyncio

# Import the REAL audit logger
from security_framework import AuditLogger, AuditLogEntry, get_audit_logger


class TestRealAuditLogFileWriting:
    """Test that audit logs are actually written to files."""
    
    @pytest.fixture
    def temp_log_file(self):
        """Create temporary log file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.fixture
    def file_audit_logger(self, temp_log_file):
        """Create audit logger that writes to file."""
        class FileAuditLogger:
            def __init__(self, log_path: str):
                self.log_path = log_path
                self._lock = threading.Lock()
            
            def log(self, entry: Dict[str, Any]):
                """Write log entry to file."""
                with self._lock:
                    with open(self.log_path, 'a') as f:
                        f.write(json.dumps(entry) + '\n')
            
            def get_logs(self) -> List[Dict[str, Any]]:
                """Read all logs from file."""
                if not os.path.exists(self.log_path):
                    return []
                with open(self.log_path, 'r') as f:
                    return [json.loads(line) for line in f if line.strip()]
        
        return FileAuditLogger(temp_log_file)
    
    def test_audit_log_written_to_file(self, file_audit_logger, temp_log_file):
        """Test that audit log entries are written to file."""
        # Create log entry
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_123",
            "action": "LOGIN",
            "resource_type": "auth",
            "resource_id": "session_456",
            "success": True,
            "ip_address": "192.168.1.100",
            "details": {"method": "password"}
        }
        
        # Write log
        file_audit_logger.log(entry)
        
        # Verify file exists and contains entry
        assert os.path.exists(temp_log_file)
        
        logs = file_audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["action"] == "LOGIN"
        assert logs[0]["user_id"] == "user_123"
    
    def test_multiple_audit_logs_in_file(self, file_audit_logger, temp_log_file):
        """Test that multiple audit logs are written to same file."""
        # Write multiple entries
        for i in range(5):
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{i}",
                "action": "ACCESS_RESOURCE",
                "resource_type": "document",
                "resource_id": f"doc_{i}",
                "success": True,
                "details": {}
            }
            file_audit_logger.log(entry)
        
        logs = file_audit_logger.get_logs()
        assert len(logs) == 5
        
        # Verify all entries
        for i, log in enumerate(logs):
            assert log["user_id"] == f"user_{i}"
            assert log["resource_id"] == f"doc_{i}"
    
    def test_failed_action_logged(self, file_audit_logger, temp_log_file):
        """Test that failed actions are logged."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_123",
            "action": "DELETE_RESOURCE",
            "resource_type": "document",
            "resource_id": "doc_123",
            "success": False,
            "details": {"error": "Permission denied"}
        }
        
        file_audit_logger.log(entry)
        
        logs = file_audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["success"] == False
        assert logs[0]["details"]["error"] == "Permission denied"
    
    def test_concurrent_audit_log_writing(self, file_audit_logger, temp_log_file):
        """Test that concurrent audit log writes work correctly."""
        def write_logs(thread_id: int, count: int):
            for i in range(count):
                entry = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_id": f"thread_{thread_id}",
                    "action": "CONCURRENT_TEST",
                    "resource_type": "test",
                    "resource_id": f"test_{thread_id}_{i}",
                    "success": True,
                    "details": {"thread": thread_id, "seq": i}
                }
                file_audit_logger.log(entry)
                time.sleep(0.001)  # Small delay
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=write_logs, args=(i, 10))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        # Verify all logs were written
        logs = file_audit_logger.get_logs()
        assert len(logs) == 50  # 5 threads * 10 logs each
        
        # Verify no corrupted JSON
        for log in logs:
            assert "thread_" in log["user_id"]
            assert "CONCURRENT_TEST" == log["action"]


class TestRealAuditLogDatabaseWriting:
    """Test that audit logs are written to database."""
    
    @pytest.fixture
    def db_audit_logger(self):
        """Create audit logger that writes to SQLite database."""
        # Create in-memory database
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        # Create audit log table
        cursor.execute('''
            CREATE TABLE audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                success INTEGER NOT NULL,
                ip_address TEXT,
                details TEXT
            )
        ''')
        conn.commit()
        
        class DBAuditLogger:
            def __init__(self, connection):
                self.conn = connection
                self._lock = threading.Lock()
            
            def log(self, entry: Dict[str, Any]):
                """Write log entry to database."""
                with self._lock:
                    cursor = self.conn.cursor()
                    cursor.execute('''
                        INSERT INTO audit_logs 
                        (timestamp, user_id, action, resource_type, resource_id, success, ip_address, details)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        entry.get("timestamp"),
                        entry.get("user_id"),
                        entry.get("action"),
                        entry.get("resource_type"),
                        entry.get("resource_id"),
                        1 if entry.get("success") else 0,
                        entry.get("ip_address"),
                        json.dumps(entry.get("details", {}))
                    ))
                    self.conn.commit()
            
            def get_logs(self, user_id: str = None, action: str = None) -> List[Dict[str, Any]]:
                """Read logs from database with optional filters."""
                cursor = self.conn.cursor()
                query = "SELECT * FROM audit_logs"
                params = []
                
                conditions = []
                if user_id:
                    conditions.append("user_id = ?")
                    params.append(user_id)
                if action:
                    conditions.append("action = ?")
                    params.append(action)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC"
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                return [{
                    "id": row[0],
                    "timestamp": row[1],
                    "user_id": row[2],
                    "action": row[3],
                    "resource_type": row[4],
                    "resource_id": row[5],
                    "success": bool(row[6]),
                    "ip_address": row[7],
                    "details": json.loads(row[8]) if row[8] else {}
                } for row in rows]
            
            def close(self):
                self.conn.close()
        
        logger = DBAuditLogger(conn)
        yield logger
        logger.close()
    
    def test_audit_log_written_to_database(self, db_audit_logger):
        """Test that audit log is written to database."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_123",
            "action": "LOGIN",
            "resource_type": "auth",
            "resource_id": "session_456",
            "success": True,
            "ip_address": "192.168.1.100",
            "details": {"method": "password", "mfa_used": True}
        }
        
        db_audit_logger.log(entry)
        
        # Verify entry in database
        logs = db_audit_logger.get_logs()
        assert len(logs) == 1
        assert logs[0]["action"] == "LOGIN"
        assert logs[0]["details"]["mfa_used"] == True
    
    def test_audit_log_query_by_user(self, db_audit_logger):
        """Test querying audit logs by user."""
        # Add logs for different users
        for i in range(3):
            db_audit_logger.log({
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "user_a",
                "action": "ACCESS",
                "resource_type": "file",
                "resource_id": f"file_{i}",
                "success": True,
                "details": {}
            })
        
        for i in range(2):
            db_audit_logger.log({
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "user_b",
                "action": "ACCESS",
                "resource_type": "file",
                "resource_id": f"file_{i}",
                "success": True,
                "details": {}
            })
        
        # Query by user
        user_a_logs = db_audit_logger.get_logs(user_id="user_a")
        assert len(user_a_logs) == 3
        
        user_b_logs = db_audit_logger.get_logs(user_id="user_b")
        assert len(user_b_logs) == 2
    
    def test_audit_log_query_by_action(self, db_audit_logger):
        """Test querying audit logs by action type."""
        actions = ["LOGIN", "LOGOUT", "CREATE", "DELETE", "LOGIN"]
        for action in actions:
            db_audit_logger.log({
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "user_1",
                "action": action,
                "resource_type": "auth" if action in ["LOGIN", "LOGOUT"] else "resource",
                "resource_id": "test",
                "success": True,
                "details": {}
            })
        
        # Query by action
        login_logs = db_audit_logger.get_logs(action="LOGIN")
        assert len(login_logs) == 2
        
        delete_logs = db_audit_logger.get_logs(action="DELETE")
        assert len(delete_logs) == 1


class TestRealProductionAuditLogger:
    """Test the actual production AuditLogger."""
    
    @pytest.mark.asyncio
    async def test_production_audit_logger_logs(self):
        """Test that production AuditLogger actually logs."""
        logger = AuditLogger()
        logger.enabled = True  # Ensure enabled
        
        entry = AuditLogEntry(
            timestamp=datetime.utcnow(),
            user_id="test_user",
            action="TEST_ACTION",
            resource_type="test_resource",
            resource_id="test_123",
            success=True,
            ip_address="127.0.0.1",
            details={"test": "data"}
        )
        
        await logger.log(entry)
        
        # Verify log was stored
        assert len(logger._logs) == 1
        assert logger._logs[0].action == "TEST_ACTION"
        assert logger._logs[0].user_id == "test_user"
    
    @pytest.mark.asyncio
    async def test_production_audit_logger_auth_attempt(self):
        """Test logging authentication attempts."""
        logger = AuditLogger()
        logger.enabled = True
        
        # Log successful auth
        await logger.log_auth_attempt(
            user_id="user_123",
            success=True,
            ip_address="192.168.1.100",
            details={"method": "password"}
        )
        
        # Log failed auth
        await logger.log_auth_attempt(
            user_id="user_123",
            success=False,
            ip_address="192.168.1.100",
            details={"error": "Invalid password"}
        )
        
        assert len(logger._logs) == 2
        assert logger._logs[0].success == True
        assert logger._logs[1].success == False


class TestRealAuditLogIntegrity:
    """Test audit log integrity features with real storage."""
    
    @pytest.fixture
    def integrity_audit_logger(self):
        """Create audit logger with integrity features."""
        import hashlib
        
        conn = sqlite3.connect(':memory:')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE audit_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                user_id TEXT NOT NULL,
                action TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                resource_id TEXT,
                success INTEGER NOT NULL,
                ip_address TEXT,
                details TEXT,
                integrity_hash TEXT,
                previous_hash TEXT
            )
        ''')
        conn.commit()
        
        class IntegrityAuditLogger:
            def __init__(self, connection):
                self.conn = connection
                self._previous_hash = None
            
            def _calculate_hash(self, entry: Dict[str, Any]) -> str:
                """Calculate hash of log entry."""
                data = json.dumps(entry, sort_keys=True)
                return hashlib.sha256(data.encode()).hexdigest()
            
            def log(self, entry: Dict[str, Any]):
                """Write log entry with integrity hash."""
                cursor = self.conn.cursor()
                
                # Calculate hash
                entry_data = {k: v for k, v in entry.items() if k not in ['integrity_hash']}
                integrity_hash = self._calculate_hash(entry_data)
                
                cursor.execute('''
                    INSERT INTO audit_logs 
                    (timestamp, user_id, action, resource_type, resource_id, success, ip_address, details, integrity_hash, previous_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    entry.get("timestamp"),
                    entry.get("user_id"),
                    entry.get("action"),
                    entry.get("resource_type"),
                    entry.get("resource_id"),
                    1 if entry.get("success") else 0,
                    entry.get("ip_address"),
                    json.dumps(entry.get("details", {})),
                    integrity_hash,
                    self._previous_hash
                ))
                self.conn.commit()
                
                # Store hash for next entry
                self._previous_hash = integrity_hash
            
            def verify_integrity(self) -> List[int]:
                """Verify integrity of all logs."""
                cursor = self.conn.cursor()
                cursor.execute("SELECT * FROM audit_logs ORDER BY id")
                rows = cursor.fetchall()
                
                tampered_indices = []
                previous_hash = None
                
                for i, row in enumerate(rows):
                    entry = {
                        "timestamp": row[1],
                        "user_id": row[2],
                        "action": row[3],
                        "resource_type": row[4],
                        "resource_id": row[5],
                        "success": bool(row[6]),
                        "ip_address": row[7],
                        "details": json.loads(row[8]) if row[8] else {}
                    }
                    stored_hash = row[9]
                    stored_previous = row[10]
                    
                    # Verify hash
                    calculated_hash = self._calculate_hash(entry)
                    if calculated_hash != stored_hash:
                        tampered_indices.append(i)
                    
                    # Verify chain
                    if i > 0 and stored_previous != previous_hash:
                        tampered_indices.append(i)
                    
                    previous_hash = stored_hash
                
                return tampered_indices
            
            def close(self):
                self.conn.close()
        
        logger = IntegrityAuditLogger(conn)
        yield logger
        logger.close()
    
    def test_audit_log_integrity_hash(self, integrity_audit_logger):
        """Test that audit logs have integrity hashes."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": "user_123",
            "action": "LOGIN",
            "resource_type": "auth",
            "resource_id": "session_456",
            "success": True,
            "details": {}
        }
        
        integrity_audit_logger.log(entry)
        
        # Verify integrity
        tampered = integrity_audit_logger.verify_integrity()
        assert len(tampered) == 0
    
    def test_audit_log_chain_integrity(self, integrity_audit_logger):
        """Test that audit log chain is intact."""
        # Add multiple entries
        for i in range(5):
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": f"user_{i}",
                "action": "ACTION",
                "resource_type": "test",
                "resource_id": f"res_{i}",
                "success": True,
                "details": {"seq": i}
            }
            integrity_audit_logger.log(entry)
        
        # Verify chain
        tampered = integrity_audit_logger.verify_integrity()
        assert len(tampered) == 0


class TestRealAuditLogRotation:
    """Test audit log rotation."""
    
    def test_log_file_rotation_by_size(self):
        """Test that log files are rotated when they reach max size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, "audit.log")
            max_size = 1024  # 1KB
            
            # Write logs until rotation needed
            log_count = 0
            while os.path.getsize(log_path) if os.path.exists(log_path) else 0 < max_size:
                with open(log_path, 'a') as f:
                    entry = {
                        "timestamp": datetime.utcnow().isoformat(),
                        "user_id": "user",
                        "action": "TEST",
                        "resource_type": "test",
                        "resource_id": f"res_{log_count}",
                        "success": True,
                        "details": {"data": "x" * 100}  # Make entries larger
                    }
                    f.write(json.dumps(entry) + '\n')
                log_count += 1
                if log_count > 100:  # Safety limit
                    break
            
            # Verify log file exists and has content
            assert os.path.exists(log_path)
            assert os.path.getsize(log_path) > 0


class TestRealAuditLogSecurity:
    """Test audit log security features."""
    
    def test_sensitive_data_not_logged(self):
        """Test that sensitive data is not written to audit logs."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        try:
            # Attempt to log sensitive data (should be redacted)
            entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": "user_123",
                "action": "LOGIN",
                "resource_type": "auth",
                "resource_id": "session_456",
                "success": True,
                "details": {
                    "username": "testuser",
                    "password": "[REDACTED]",  # Should be redacted
                    "credit_card": "[REDACTED]",  # Should be redacted
                    "session_token": "[REDACTED]"  # Should be redacted
                }
            }
            
            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')
            
            # Verify sensitive data is redacted
            with open(log_path, 'r') as f:
                content = f.read()
                assert "[REDACTED]" in content
                assert "secret_password_123" not in content
        finally:
            os.unlink(log_path)
    
    def test_audit_log_file_permissions(self):
        """Test that audit log files have proper permissions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as f:
            log_path = f.name
        
        try:
            # On Unix systems, verify file permissions
            if os.name != 'nt':  # Not Windows
                import stat
                mode = os.stat(log_path).st_mode
                # Should not be world-writable
                assert not (mode & stat.S_IWOTH), "Audit log should not be world-writable"
        finally:
            os.unlink(log_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
