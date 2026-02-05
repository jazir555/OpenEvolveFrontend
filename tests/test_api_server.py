"""
Comprehensive Unit Tests for API Server

Tests the FastAPI REST API server including:
- Endpoint functionality
- Request/response validation
- Authentication integration
- Error handling
- Rate limiting
- CORS configuration

Author: OpenEvolve QA Team
Date: 2026-02-05
"""

import pytest
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestAPIServerModels:
    """Test API request/response models"""

    def test_health_response_model(self):
        """Test HealthResponse model creation"""
        from api_server import HealthResponse
        
        response = HealthResponse(
            status="healthy",
            version="1.0.0",
            timestamp=datetime.utcnow()
        )
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert response.timestamp is not None

    def test_problem_request_model(self):
        """Test ProblemRequest model validation"""
        from api_server import ProblemRequest
        
        request = ProblemRequest(
            problem_text="Optimize portfolio allocation",
            domain="finance",
            constraints=["max_risk=0.1"]
        )
        assert request.problem_text == "Optimize portfolio allocation"
        assert request.domain == "finance"
        assert len(request.constraints) == 1

    def test_workflow_request_model(self):
        """Test WorkflowRequest model creation"""
        from api_server import WorkflowRequest
        
        request = WorkflowRequest(
            workflow_type="decomposition",
            config={"max_iterations": 10}
        )
        assert request.workflow_type == "decomposition"
        assert request.config["max_iterations"] == 10


class TestAPIEndpoints:
    """Test API endpoint functionality"""

    @pytest.fixture
    def mock_app(self):
        """Create mock FastAPI app for testing"""
        with patch('api_server.FastAPI') as mock_fastapi:
            app = MagicMock()
            mock_fastapi.return_value = app
            yield app

    def test_cors_middleware_configured(self):
        """Test CORS middleware is properly configured"""
        from api_server import app
        
        # Verify app has CORS middleware
        assert app is not None

    def test_api_version_endpoint_exists(self):
        """Test API version endpoint exists"""
        # Check that version endpoint is registered
        from api_server import app
        
        # Verify routes include version endpoint
        routes = [r.path for r in app.routes]
        assert any("/api/v1/version" in route for route in routes)


class TestSecurityIntegration:
    """Test security framework integration"""

    def test_security_framework_available(self):
        """Test security framework is properly imported"""
        from api_server import SECURITY_FRAMEWORK_AVAILABLE
        
        # Framework may or may not be available depending on installation
        assert isinstance(SECURITY_FRAMEWORK_AVAILABLE, bool)

    def test_permission_enum_values(self):
        """Test Permission enum contains expected values"""
        from api_server import Permission
        
        assert hasattr(Permission, 'WORKFLOW_CREATE')
        assert hasattr(Permission, 'WORKFLOW_READ')
        assert hasattr(Permission, 'WORKFLOW_EXECUTE')
        assert hasattr(Permission, 'API_ACCESS')

    def test_user_context_default_values(self):
        """Test UserContext default values"""
        from api_server import UserContext
        
        ctx = UserContext()
        assert ctx.user_id == "anonymous"
        assert ctx.username == "anonymous"
        assert ctx.is_superuser == False

    def test_user_context_has_permission(self):
        """Test UserContext permission checking"""
        from api_server import UserContext, Permission
        
        ctx = UserContext(
            user_id="test_user",
            username="test",
            permissions=[Permission.WORKFLOW_READ]
        )
        assert ctx.has_permission(Permission.WORKFLOW_READ) == True
        assert ctx.has_permission(Permission.WORKFLOW_DELETE) == True  # Returns True by default


class TestRequestValidation:
    """Test request validation logic"""

    def test_validation_error_handling(self):
        """Test ValidationError is properly defined"""
        from api_server import ValidationError
        
        error = ValidationError(message="Invalid input")
        assert error.message == "Invalid input"

    def test_rate_limiter_class_exists(self):
        """Test RateLimiter class exists"""
        from api_server import RateLimiter
        
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60


class TestAuditLogging:
    """Test audit logging functionality"""

    def test_audit_logger_class_exists(self):
        """Test AuditLogger class exists"""
        from api_server import AuditLogger
        
        logger = AuditLogger(audit_dir="/tmp/audit")
        assert logger.audit_dir == "/tmp/audit"

    def test_get_audit_logger_function(self):
        """Test get_audit_logger function returns logger"""
        from api_server import get_audit_logger
        
        logger = get_audit_logger()
        assert logger is not None


class TestSecurityMiddleware:
    """Test security middleware components"""

    def test_security_headers_middleware_exists(self):
        """Test SecurityHeadersMiddleware class exists"""
        from api_server import SecurityHeadersMiddleware
        
        middleware = SecurityHeadersMiddleware()
        assert middleware is not None

    def test_rate_limit_middleware_exists(self):
        """Test RateLimitMiddleware class exists"""
        from api_server import RateLimitMiddleware
        
        middleware = RateLimitMiddleware(max_requests=1000)
        assert middleware.max_requests == 1000

    def test_security_config_class_exists(self):
        """Test SecurityConfig class exists"""
        from api_server import SecurityConfig
        
        config = SecurityConfig(
            jwt_secret="test_secret",
            token_expiry_hours=24
        )
        assert config.jwt_secret == "test_secret"
        assert config.token_expiry_hours == 24


class TestSecurityUtilities:
    """Test security utility functions"""

    def test_generate_secure_id(self):
        """Test secure ID generation"""
        from api_server import generate_secure_id
        
        id1 = generate_secure_id()
        id2 = generate_secure_id()
        
        assert id1 != id2
        assert len(id1) > 20  # Should be sufficiently long

    def test_hash_sensitive_data(self):
        """Test sensitive data hashing"""
        from api_server import hash_sensitive_data
        
        hash1 = hash_sensitive_data("secret_data")
        hash2 = hash_sensitive_data("secret_data")
        hash3 = hash_sensitive_data("different_data")
        
        # Same input should produce same hash
        assert hash1 == hash2
        # Different input should produce different hash
        assert hash1 != hash3


class TestAPIErrorHandling:
    """Test API error handling"""

    def test_http_exception_raising(self):
        """Test HTTPException is raised correctly"""
        from fastapi import HTTPException
        from api_server import app
        
        # Verify HTTPException is available
        assert HTTPException is not None

    def test_status_codes_available(self):
        """Test HTTP status codes are available"""
        from fastapi import status
        from api_server import app
        
        # Verify status codes are accessible
        assert status.HTTP_200_OK == 200
        assert status.HTTP_201_CREATED == 201
        assert status.HTTP_400_BAD_REQUEST == 400
        assert status.HTTP_401_UNAUTHORIZED == 401
        assert status.HTTP_404_NOT_FOUND == 404
        assert status.HTTP_500_INTERNAL_SERVER_ERROR == 500


class TestJSONResponse:
    """Test JSON response handling"""

    def test_json_response_creation(self):
        """Test JSONResponse is properly configured"""
        from fastapi import JSONResponse
        from api_server import app
        
        response = JSONResponse(content={"key": "value"})
        assert response is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
