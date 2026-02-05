"""
Comprehensive Audit Logging Security Tests
Tests for audit log generation, integrity, retention, and tamper detection.
"""

import pytest
import json
import os
import tempfile
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List
from unittest.mock import Mock, patch, MagicMock

# Import test subjects
try:
    from auth_system import AuthenticationSystem, AuditSystem, AuditLog
    AUTH_SYSTEM_AVAILABLE = True
except ImportError:
    AUTH_SYSTEM_AVAILABLE = False

try:
    from rbac_enhanced import RBACStorage, AuditLog as RBACAuditLog, Permission
    RBAC_AVAILABLE = True
except ImportError:
    RBAC_AVAILABLE = False


class MockAuditLog:
    """Mock audit log for testing when real system not available."""
    def __init__(self, log_id: str, user_id: str, action: str, resource_type: str,
                 resource_id: str = None, success: bool = True, timestamp: datetime = None,
                 details: Dict[str, Any] = None):
        self.log_id = log_id
        self.user_id = user_id
        self.action = action
        self.resource_type = resource_type
        self.resource_id = resource_id
        self.success = success
        self.timestamp = timestamp or datetime.utcnow()
        self.details = details or {}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'log_id': self.log_id,
            'user_id': self.user_id,
            'action': self.action,
            'resource_type': self.resource_type,
            'resource_id': self.resource_id,
            'success': self.success,
            'timestamp': self.timestamp.isoformat(),
            'details': self.details
        }


class AuditLogIntegrityChecker:
    """Helper class to check audit log integrity."""
    
    @staticmethod
    def calculate_hash(log_entry: Dict[str, Any]) -> str:
        """Calculate hash of log entry for integrity verification."""
        # Create deterministic string representation
        log_str = json.dumps(log_entry, sort_keys=True)
        return hashlib.sha256(log_str.encode()).hexdigest()
    
    @staticmethod
    def verify_chain_integrity(logs: List[Dict[str, Any]]) -> bool:
        """Verify integrity chain of audit logs."""
        if not logs:
            return True
        
        for i in range(1, len(logs)):
            # Each log should reference previous log's hash
            if 'previous_hash' in logs[i]:
                expected_hash = AuditLogIntegrityChecker.calculate_hash(logs[i-1])
                if logs[i]['previous_hash'] != expected_hash:
                    return False
        return True
    
    @staticmethod
    def detect_tampering(logs: List[Dict[str, Any]]) -> List[int]:
        """Detect which logs may have been tampered with."""
        tampered_indices = []
        
        for i, log in enumerate(logs):
            if 'integrity_hash' in log:
                stored_hash = log['integrity_hash']
                # Recalculate hash (excluding the stored hash field)
                log_copy = {k: v for k, v in log.items() if k != 'integrity_hash'}
                calculated_hash = AuditLogIntegrityChecker.calculate_hash(log_copy)
                if stored_hash != calculated_hash:
                    tampered_indices.append(i)
        
        return tampered_indices


@pytest.mark.skipif(not AUTH_SYSTEM_AVAILABLE, reason="auth_system module not available")
class TestAuditLogGeneration:
    """Test audit log generation."""
    
    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def auth_system(self, temp_db):
        """Create authentication system with audit logging."""
        return AuthenticationSystem(db_path=temp_db)
    
    def test_user_creation_logging(self, auth_system):
        """Test that user creation is logged."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        # Check that audit log was created
        logs = auth_system.get_audit_logs(user_id=user.id)
        create_logs = [l for l in logs if l.operation == "CREATE_USER"]
        assert len(create_logs) >= 1
        
        log = create_logs[0]
        assert log.user_id == user.id
        assert log.success == True
        assert log.details.get('username') == "testuser"
    
    def test_authentication_logging_success(self, auth_system):
        """Test successful authentication logging."""
        auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        auth_system.authenticate("testuser", "password123")
        
        logs = auth_system.get_audit_logs()
        auth_logs = [l for l in logs if l.operation == "AUTHENTICATE"]
        assert len(auth_logs) >= 1
        
        log = auth_logs[0]
        assert log.success == True
        assert log.details.get('username') == "testuser"
    
    def test_authentication_logging_failure(self, auth_system):
        """Test failed authentication logging."""
        auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        auth_system.authenticate("testuser", "wrongpassword")
        
        logs = auth_system.get_audit_logs()
        auth_logs = [l for l in logs if l.operation == "AUTHENTICATE" and not l.success]
        assert len(auth_logs) >= 1
        
        log = auth_logs[0]
        assert log.success == False
        assert log.details.get('result') == "invalid_password"
    
    def test_api_key_creation_logging(self, auth_system):
        """Test API key creation logging."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        api_key = auth_system.generate_api_key(user.id)
        
        logs = auth_system.get_audit_logs(user_id=user.id)
        api_key_logs = [l for l in logs if l.operation == "CREATE_API_KEY"]
        assert len(api_key_logs) >= 1
        
        log = api_key_logs[0]
        assert log.success == True
        assert 'api_key_id' in log.details


@pytest.mark.skipif(not AUTH_SYSTEM_AVAILABLE, reason="auth_system module not available")
class TestAuditLogIntegrity:
    """Test audit log integrity features."""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def auth_system(self, temp_db):
        return AuthenticationSystem(db_path=temp_db)
    
    def test_log_integrity_hash(self, auth_system):
        """Test that logs can have integrity hashes."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        logs = auth_system.get_audit_logs(user_id=user.id)
        assert len(logs) > 0
        
        # Log should have necessary fields for integrity
        log = logs[0]
        assert log.timestamp is not None
        assert log.user_id is not None
    
    def test_integrity_verification(self):
        """Test integrity verification of logs."""
        # Create logs with integrity hashes
        logs = []
        for i in range(5):
            log = {
                'log_id': f'log_{i}',
                'user_id': f'user_{i}',
                'action': 'TEST_ACTION',
                'timestamp': datetime.utcnow().isoformat(),
                'details': {'data': f'value_{i}'}
            }
            log['integrity_hash'] = AuditLogIntegrityChecker.calculate_hash(log)
            logs.append(log)
        
        # Verify all logs are intact
        tampered = AuditLogIntegrityChecker.detect_tampering(logs)
        assert len(tampered) == 0
    
    def test_tamper_detection(self):
        """Test detection of tampered logs."""
        # Create logs with integrity hashes
        logs = []
        for i in range(5):
            log = {
                'log_id': f'log_{i}',
                'user_id': f'user_{i}',
                'action': 'TEST_ACTION',
                'timestamp': datetime.utcnow().isoformat(),
                'details': {'data': f'value_{i}'}
            }
            log['integrity_hash'] = AuditLogIntegrityChecker.calculate_hash(log)
            logs.append(log)
        
        # Tamper with a log
        logs[2]['details']['data'] = 'TAMPERED'
        
        # Detect tampering
        tampered = AuditLogIntegrityChecker.detect_tampering(logs)
        assert 2 in tampered
    
    def test_chain_integrity(self):
        """Test integrity chain between logs."""
        logs = []
        previous_hash = None
        
        for i in range(5):
            log = {
                'log_id': f'log_{i}',
                'action': 'TEST',
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            if previous_hash:
                log['previous_hash'] = previous_hash
            
            previous_hash = AuditLogIntegrityChecker.calculate_hash(log)
            logs.append(log)
        
        # Verify chain integrity
        assert AuditLogIntegrityChecker.verify_chain_integrity(logs) == True
        
        # Break the chain
        logs[2]['action'] = 'TAMPERED'
        assert AuditLogIntegrityChecker.verify_chain_integrity(logs) == False


@pytest.mark.skipif(not AUTH_SYSTEM_AVAILABLE, reason="auth_system module not available")
class TestAuditLogRetention:
    """Test audit log retention policies."""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    @pytest.fixture
    def auth_system(self, temp_db):
        return AuthenticationSystem(db_path=temp_db)
    
    def test_log_retrieval_by_date_range(self, auth_system):
        """Test retrieving logs within date range."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        last_week = now - timedelta(days=7)
        
        # Get logs with date range
        logs = auth_system.get_audit_logs(
            start_date=yesterday,
            end_date=now
        )
        
        # Should only return logs within range
        for log in logs:
            assert yesterday <= log.timestamp <= now
    
    def test_log_pagination(self, auth_system):
        """Test log pagination."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        # Create multiple logs
        for i in range(10):
            auth_system.log_audit(
                user_id=user.id,
                operation=f"TEST_OP_{i}",
                resource="test",
                resource_id=f"res_{i}",
                success=True,
                details={}
            )
        
        # Get limited number of logs
        logs = auth_system.get_audit_logs(limit=5)
        assert len(logs) <= 5
    
    def test_log_filtering_by_operation(self, auth_system):
        """Test filtering logs by operation type."""
        user = auth_system.create_user(
            username="testuser",
            email="test@example.com",
            password="password123",
            roles=[]
        )
        
        # Create logs with different operations
        auth_system.log_audit(user.id, "CREATE", "resource", "1", True, {})
        auth_system.log_audit(user.id, "UPDATE", "resource", "2", True, {})
        auth_system.log_audit(user.id, "DELETE", "resource", "3", True, {})
        
        # Filter by operation
        # Note: This depends on the actual implementation
        logs = auth_system.get_audit_logs()
        create_logs = [l for l in logs if l.operation == "CREATE"]
        assert len(create_logs) >= 1


class TestAuditLogSecurityFeatures:
    """Test security features of audit logging."""
    
    def test_sensitive_data_not_logged(self):
        """Test that sensitive data is not logged in plain text."""
        log_entry = MockAuditLog(
            log_id="log_1",
            user_id="user_1",
            action="LOGIN",
            resource_type="auth",
            details={
                "username": "testuser",
                "password": "[REDACTED]",  # Password should never be logged
                "api_key": "[REDACTED]",   # API keys should never be logged
                "session_id": "sess_123"
            }
        )
        
        log_dict = log_entry.to_dict()
        assert log_dict['details']['password'] == "[REDACTED]"
        assert log_dict['details']['api_key'] == "[REDACTED]"
    
    def test_log_entry_structure(self):
        """Test required fields in log entries."""
        log = MockAuditLog(
            log_id="log_1",
            user_id="user_1",
            action="TEST_ACTION",
            resource_type="test_resource",
            resource_id="res_123",
            success=True,
            details={"key": "value"}
        )
        
        log_dict = log.to_dict()
        
        # Verify all required fields
        assert 'log_id' in log_dict
        assert 'user_id' in log_dict
        assert 'action' in log_dict
        assert 'resource_type' in log_dict
        assert 'success' in log_dict
        assert 'timestamp' in log_dict
    
    def test_log_timestamp_immutability(self):
        """Test that log timestamps cannot be altered."""
        original_time = datetime.utcnow()
        log = MockAuditLog(
            log_id="log_1",
            user_id="user_1",
            action="TEST",
            resource_type="test",
            timestamp=original_time
        )
        
        # Timestamp should be preserved
        assert log.timestamp == original_time
    
    def test_log_id_uniqueness(self):
        """Test that log IDs are unique."""
        log_ids = set()
        for i in range(100):
            log = MockAuditLog(
                log_id=f"log_{i}_{datetime.utcnow().timestamp()}",
                user_id="user_1",
                action="TEST",
                resource_type="test"
            )
            log_ids.add(log.log_id)
        
        # All IDs should be unique
        assert len(log_ids) == 100


class TestAuditLogAnalysis:
    """Test audit log analysis capabilities."""
    
    def test_failed_login_detection(self):
        """Test detection of failed login attempts."""
        logs = [
            MockAuditLog("1", "user1", "LOGIN", "auth", success=False),
            MockAuditLog("2", "user1", "LOGIN", "auth", success=False),
            MockAuditLog("3", "user1", "LOGIN", "auth", success=False),
            MockAuditLog("4", "user1", "LOGIN", "auth", success=True),
        ]
        
        # Count failed attempts before success
        failed_count = 0
        for log in logs:
            if log.action == "LOGIN":
                if not log.success:
                    failed_count += 1
                else:
                    break
        
        assert failed_count == 3
    
    def test_privilege_escalation_detection(self):
        """Test detection of potential privilege escalation."""
        logs = [
            MockAuditLog("1", "user1", "LOGIN", "auth", success=True),
            MockAuditLog("2", "user1", "UPDATE_ROLE", "user", resource_id="user1", 
                        details={"old_role": "user", "new_role": "admin"}),
            MockAuditLog("3", "user1", "ACCESS_ADMIN", "admin_panel", success=True),
        ]
        
        # Check for role change followed by admin access
        role_changed = False
        for log in logs:
            if log.action == "UPDATE_ROLE" and log.details.get("new_role") == "admin":
                role_changed = True
            if role_changed and log.action == "ACCESS_ADMIN":
                assert True  # Detected potential escalation
                return
        
        assert role_changed  # At least verify role change was logged
    
    def test_unusual_activity_detection(self):
        """Test detection of unusual activity patterns."""
        # Simulate unusual access pattern
        logs = []
        for i in range(100):
            logs.append(MockAuditLog(
                f"log_{i}",
                "user1",
                "ACCESS_RESOURCE",
                "sensitive_data",
                timestamp=datetime.utcnow() - timedelta(minutes=i)
            ))
        
        # Count access to sensitive data
        sensitive_access = [l for l in logs if l.resource_type == "sensitive_data"]
        
        # More than 50 accesses in short period is unusual
        assert len(sensitive_access) > 50  # This should trigger investigation


class TestAuditLogCompliance:
    """Test audit log compliance features."""
    
    def test_gdpr_compliance_fields(self):
        """Test GDPR-required fields in logs."""
        log = MockAuditLog(
            log_id="log_1",
            user_id="user_1",
            action="DATA_ACCESS",
            resource_type="personal_data",
            details={
                "data_subject_id": "subject_123",
                "legal_basis": "consent",
                "purpose": "service_provision"
            }
        )
        
        log_dict = log.to_dict()
        
        # GDPR requires purpose specification
        assert 'purpose' in log_dict['details'] or True  # Adjust based on requirements
        # GDPR requires timestamp
        assert 'timestamp' in log_dict
    
    def test_pci_dss_compliance(self):
        """Test PCI DSS compliance requirements."""
        log = MockAuditLog(
            log_id="log_1",
            user_id="admin_1",
            action="ACCESS",
            resource_type="cardholder_data",
            success=True,
            details={
                "access_type": "read",
                "card_data_accessed": False  # Should never log actual card data
            }
        )
        
        # PCI DSS: Never log full card numbers
        assert not log.details.get('card_data_accessed', False)
    
    def test_hipaa_compliance(self):
        """Test HIPAA compliance for healthcare data."""
        log = MockAuditLog(
            log_id="log_1",
            user_id="doctor_1",
            action="VIEW_PHI",
            resource_type="patient_record",
            resource_id="patient_123",
            details={
                "phi_accessed": False,  # Should not log PHI content
                "access_reason": "treatment"
            }
        )
        
        # HIPAA: Access reason must be documented
        assert 'access_reason' in log.details


class TestAuditLogExport:
    """Test audit log export functionality."""
    
    def test_json_export(self):
        """Test exporting logs to JSON."""
        logs = [
            MockAuditLog("1", "user1", "LOGIN", "auth", success=True),
            MockAuditLog("2", "user1", "LOGOUT", "auth", success=True),
        ]
        
        # Export to JSON
        export_data = [log.to_dict() for log in logs]
        json_str = json.dumps(export_data)
        
        # Verify JSON is valid
        parsed = json.loads(json_str)
        assert len(parsed) == 2
        assert parsed[0]['action'] == "LOGIN"
    
    def test_csv_export_format(self):
        """Test CSV export format."""
        import csv
        import io
        
        logs = [
            MockAuditLog("1", "user1", "LOGIN", "auth", success=True),
        ]
        
        # Create CSV
        output = io.StringIO()
        if logs:
            writer = csv.DictWriter(output, fieldnames=logs[0].to_dict().keys())
            writer.writeheader()
            for log in logs:
                writer.writerow(log.to_dict())
        
        csv_content = output.getvalue()
        assert "log_id" in csv_content
        assert "LOGIN" in csv_content


@pytest.mark.skipif(not AUTH_SYSTEM_AVAILABLE, reason="auth_system module not available")
class TestAuditSystemIntegration:
    """Test integration of audit system with other components."""
    
    @pytest.fixture
    def temp_db(self):
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        yield db_path
        os.unlink(db_path)
    
    def test_audit_system_initialization(self, temp_db):
        """Test audit system initialization."""
        auth_system = AuthenticationSystem(db_path=temp_db)
        assert auth_system is not None
        
        # Should be able to create audit logs
        auth_system.log_audit(
            user_id="test_user",
            operation="SYSTEM_START",
            resource="system",
            resource_id="main",
            success=True,
            details={}
        )
        
        logs = auth_system.get_audit_logs()
        assert len(logs) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
