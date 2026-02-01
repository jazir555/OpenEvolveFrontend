"""
Comprehensive tests for API Gateway

Tests cover:
- REST API endpoints (all CRUD operations)
- GraphQL schema, queries, and mutations
- Rate limiting functionality
- Authentication/authorization
- Error handling
- Request/response validation

Following CLAUDE.md principles:
- All tests are idempotent
- Use pytest.mark.asyncio for async tests
- Proper fixtures for setup
- Mock clients where needed
- Skip decorators for unavailable dependencies
"""

import pytest
import json
import logging
import sys
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch, call

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "knowledge_engine"))

# Import the API Gateway module
GATEWAY_AVAILABLE = False
try:
    # First try direct import
    from api_gateway import (
        RESTAPIGateway, GraphQLSchema, APIRequest, APIResponse,
        Route, RateLimiter, KnowledgeAPIFactory, APIDocumentation, HTTPMethod
    )
    GATEWAY_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Successfully imported api_gateway module via direct import")
except ImportError as e:
    # Fall back to importlib
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "api_gateway",
            project_root / "knowledge_engine" / "api_gateway.py"
        )
        if spec and spec.loader:
            api_gateway_module = importlib.util.module_from_spec(spec)
            sys.modules["api_gateway"] = api_gateway_module
            spec.loader.exec_module(api_gateway_module)
            
            # Import all classes
            RESTAPIGateway = api_gateway_module.RESTAPIGateway
            GraphQLSchema = api_gateway_module.GraphQLSchema
            APIRequest = api_gateway_module.APIRequest
            APIResponse = api_gateway_module.APIResponse
            Route = api_gateway_module.Route
            RateLimiter = api_gateway_module.RateLimiter
            KnowledgeAPIFactory = api_gateway_module.KnowledgeAPIFactory
            APIDocumentation = api_gateway_module.APIDocumentation
            HTTPMethod = api_gateway_module.HTTPMethod
            
            GATEWAY_AVAILABLE = True
            logger = logging.getLogger(__name__)
            logger.info("Successfully imported api_gateway module via importlib")
    except Exception as e2:
        GATEWAY_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.warning(f"Could not import api_gateway module: {e2}")


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mock_platform():
    """Mock knowledge platform for testing."""
    platform = MagicMock()
    platform.health_check.return_value = {"status": "healthy", "version": "1.0.0"}
    platform.health_check = MagicMock(return_value={"status": "healthy", "version": "1.0.0"})
    platform.add_knowledge = AsyncMock(return_value=(MagicMock(id="test-id-123"), {}))
    platform.get_knowledge = AsyncMock(return_value={"id": "test-id-123", "content": "test content"})
    platform.list_knowledge = AsyncMock(return_value={"items": [], "total": 0})
    platform.search = AsyncMock(return_value={"query": "test", "results": []})
    platform.get_recommendations = AsyncMock(return_value={"recommendations": []})
    return platform


@pytest.fixture
def api_gateway():
    """Create a fresh REST API Gateway for each test."""
    if not GATEWAY_AVAILABLE:
        pytest.skip("API Gateway module not available")
    return RESTAPIGateway()


@pytest.fixture
def rate_limiter():
    """Create a fresh RateLimiter for each test."""
    if not GATEWAY_AVAILABLE:
        pytest.skip("API Gateway module not available")
    return RateLimiter()


@pytest.fixture
def graphql_schema():
    """Create a fresh GraphQLSchema for each test."""
    if not GATEWAY_AVAILABLE:
        pytest.skip("API Gateway module not available")
    return GraphQLSchema()


@pytest.fixture
def sample_routes(mock_platform):
    """Create sample routes for testing."""
    def handler(req: APIRequest) -> APIResponse:
        return APIResponse(status_code=200, data={"message": "success"})
    
    return [
        Route(path="/test", method=HTTPMethod.GET, handler=handler, requires_auth=False),
        Route(path="/test/{id}", method=HTTPMethod.GET, handler=handler, requires_auth=False),
        Route(path="/protected", method=HTTPMethod.GET, handler=handler, requires_auth=True),
        Route(path="/limited", method=HTTPMethod.GET, handler=handler, requires_auth=False, rate_limit=5),
    ]


@pytest.fixture
def authenticated_request():
    """Create an authenticated API request."""
    return APIRequest(
        method=HTTPMethod.GET,
        path="/protected",
        user_id="user_123",
        tenant_id="tenant_456",
        headers={"Authorization": "Bearer test_token"}
    )


@pytest.fixture
def anonymous_request():
    """Create an anonymous API request."""
    return APIRequest(
        method=HTTPMethod.GET,
        path="/public",
        headers={}
    )


@pytest.fixture
def knowledge_api_factory(mock_platform):
    """Create a KnowledgeAPIFactory with mocked platform."""
    if not GATEWAY_AVAILABLE:
        pytest.skip("API Gateway module not available")
    return KnowledgeAPIFactory.create_rest_api(mock_platform)


# ============================================================================
# Test Classes
# ============================================================================

@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestRateLimiter:
    """Tests for RateLimiter functionality."""

    @pytest.mark.asyncio
    async def test_rate_limit_allows_requests_within_limit(self, rate_limiter):
        """Test that requests within rate limit are allowed."""
        key = "test_user:/api/test"
        limit = 5
        
        # All requests within limit should be allowed
        for i in range(limit):
            assert rate_limiter.check_rate_limit(key, limit), f"Request {i+1} should be allowed"
    
    @pytest.mark.asyncio
    async def test_rate_limit_blocks_excess_requests(self, rate_limiter):
        """Test that requests exceeding rate limit are blocked."""
        key = "test_user:/api/test"
        limit = 3
        
        # Make requests up to limit
        for i in range(limit):
            assert rate_limiter.check_rate_limit(key, limit), f"Request {i+1} should be allowed"
        
        # Next request should be blocked
        assert not rate_limiter.check_rate_limit(key, limit), "Request over limit should be blocked"
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_key_isolation(self, rate_limiter):
        """Test that rate limits are isolated per key."""
        limit = 3
        
        # Use up limit for user1
        key1 = "user1:/api/test"
        for _ in range(limit):
            rate_limiter.check_rate_limit(key1, limit)
        
        # user1 should be blocked
        assert not rate_limiter.check_rate_limit(key1, limit)
        
        # user2 should still be allowed
        key2 = "user2:/api/test"
        assert rate_limiter.check_rate_limit(key2, limit)
    
    @pytest.mark.asyncio
    async def test_rate_limit_remaining_calculation(self, rate_limiter):
        """Test that remaining requests are calculated correctly."""
        key = "test_user:/api/test"
        limit = 10
        
        # Initially all requests remaining
        assert rate_limiter.get_remaining(key, limit) == limit
        
        # Use 3 requests
        for _ in range(3):
            rate_limiter.check_rate_limit(key, limit)
        
        # Should have 7 remaining
        assert rate_limiter.get_remaining(key, limit) == 7
    
    @pytest.mark.asyncio
    async def test_rate_limit_window_expiration(self, rate_limiter):
        """Test that rate limit window expires correctly."""
        key = "test_user:/api/test"
        limit = 2
        window = 1  # 1 second window
        
        # Use up the limit
        for _ in range(limit):
            rate_limiter.check_rate_limit(key, limit, window)
        
        # Should be blocked
        assert not rate_limiter.check_rate_limit(key, limit, window)
        
        # Wait for window to expire
        await asyncio.sleep(window + 0.1)
        
        # Should be allowed again
        assert rate_limiter.check_rate_limit(key, limit, window)


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestRESTAPIRoutes:
    """Tests for REST API routing and CRUD operations."""

    @pytest.mark.asyncio
    async def test_route_registration(self, api_gateway, sample_routes):
        """Test that routes can be registered."""
        for route in sample_routes:
            api_gateway.register_route(route)
        
        assert len(api_gateway.routes) == len(sample_routes)
    
    @pytest.mark.asyncio
    async def test_path_matching_exact(self, api_gateway):
        """Test exact path matching."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/exact/path",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/exact/path")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_path_matching_with_parameters(self, api_gateway):
        """Test path matching with path parameters."""
        def handler(req): return APIResponse(status_code=200, data={"id": req.path.split("/")[-1]})
        
        api_gateway.register_route(Route(
            path="/items/{id}",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/items/12345")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_path_not_found(self, api_gateway):
        """Test 404 response for non-matching path."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/existing",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/nonexistent")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 404
        assert "Not Found" in response.error
    
    @pytest.mark.asyncio
    async def test_method_not_allowed(self, api_gateway):
        """Test that wrong method returns 404."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.POST, path="/resource")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 404


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestAuthentication:
    """Tests for authentication and authorization."""

    @pytest.mark.asyncio
    async def test_protected_route_requires_auth(self, api_gateway):
        """Test that protected routes require authentication."""
        def handler(req): return APIResponse(status_code=200, data={"secret": "data"})
        
        api_gateway.register_route(Route(
            path="/protected/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True
        ))
        
        # Anonymous request should be rejected
        request = APIRequest(method=HTTPMethod.GET, path="/protected/resource")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 401
        assert "Unauthorized" in response.error
    
    @pytest.mark.asyncio
    async def test_authenticated_request_allowed(self, api_gateway):
        """Test that authenticated requests are allowed."""
        def handler(req): return APIResponse(status_code=200, data={"secret": "data"})
        
        api_gateway.register_route(Route(
            path="/protected/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True
        ))
        
        # Authenticated request should succeed
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/protected/resource",
            user_id="user_123"
        )
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_public_route_no_auth_required(self, api_gateway):
        """Test that public routes don't require authentication."""
        def handler(req): return APIResponse(status_code=200, data={"public": "data"})
        
        api_gateway.register_route(Route(
            path="/public/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        # Anonymous request should succeed
        request = APIRequest(method=HTTPMethod.GET, path="/public/resource")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 200


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestRateLimitingIntegration:
    """Tests for rate limiting integration with REST API."""

    @pytest.mark.asyncio
    async def test_rate_limited_route_enforces_limit(self, api_gateway):
        """Test that rate-limited routes enforce rate limits."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/limited/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False,
            rate_limit=2
        ))
        
        # First 2 requests should succeed
        for i in range(2):
            request = APIRequest(method=HTTPMethod.GET, path="/limited/resource")
            response = await api_gateway.handle_request(request)
            assert response.status_code == 200, f"Request {i+1} should succeed"
        
        # 3rd request should be rate limited
        request = APIRequest(method=HTTPMethod.GET, path="/limited/resource")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.error
        assert "X-RateLimit-Remaining" in response.headers
        assert "Retry-After" in response.headers
    
    @pytest.mark.asyncio
    async def test_rate_limit_per_user(self, api_gateway):
        """Test that rate limits are applied per user."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/limited/resource",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True,
            rate_limit=2
        ))
        
        # Use up limit for user1
        for _ in range(2):
            request = APIRequest(
                method=HTTPMethod.GET,
                path="/limited/resource",
                user_id="user1"
            )
            response = await api_gateway.handle_request(request)
            assert response.status_code == 200
        
        # user1 should be rate limited
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/limited/resource",
            user_id="user1"
        )
        response = await api_gateway.handle_request(request)
        assert response.status_code == 429
        
        # user2 should still be allowed
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/limited/resource",
            user_id="user2"
        )
        response = await api_gateway.handle_request(request)
        assert response.status_code == 200


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.asyncio
    async def test_handler_exception_returns_500(self, api_gateway):
        """Test that handler exceptions return 500."""
        def failing_handler(req):
            raise ValueError("Something went wrong")
        
        api_gateway.register_route(Route(
            path="/failing",
            method=HTTPMethod.GET,
            handler=failing_handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/failing")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 500
        assert "Internal Server Error" in response.error
    
    @pytest.mark.asyncio
    async def test_async_handler_exception(self, api_gateway):
        """Test that async handler exceptions are caught."""
        async def async_failing_handler(req):
            raise RuntimeError("Async error")
        
        api_gateway.register_route(Route(
            path="/async-failing",
            method=HTTPMethod.GET,
            handler=async_failing_handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/async-failing")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 500
        assert "Internal Server Error" in response.error


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestRequestResponse:
    """Tests for request/response handling."""

    @pytest.mark.asyncio
    async def test_request_to_dict(self):
        """Test APIRequest to_dict method."""
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/test/path",
            headers={"Content-Type": "application/json"},
            query_params={"page": "1"},
            body={"key": "value"},
            user_id="user123",
            tenant_id="tenant456"
        )
        
        d = request.to_dict()
        
        assert d["method"] == "POST"
        assert d["path"] == "/test/path"
        assert d["user_id"] == "user123"
        assert d["tenant_id"] == "tenant456"
        assert d["body"] == {"key": "value"}
        assert "request_id" in d
        assert "timestamp" in d
    
    @pytest.mark.asyncio
    async def test_response_to_dict_with_data(self):
        """Test APIResponse to_dict with data."""
        response = APIResponse(
            status_code=200,
            data={"result": "success"},
            request_id="req-123"
        )
        
        d = response.to_dict()
        
        assert d["status_code"] == 200
        assert d["data"] == {"result": "success"}
        assert d["request_id"] == "req-123"
        assert "error" not in d
    
    @pytest.mark.asyncio
    async def test_response_to_dict_with_error(self):
        """Test APIResponse to_dict with error."""
        response = APIResponse(
            status_code=404,
            error="Not found",
            request_id="req-123"
        )
        
        d = response.to_dict()
        
        assert d["status_code"] == 404
        assert d["error"] == "Not found"
        assert "data" not in d
    
    @pytest.mark.asyncio
    async def test_request_id_generation(self):
        """Test that request IDs are auto-generated."""
        request1 = APIRequest(method=HTTPMethod.GET, path="/test")
        request2 = APIRequest(method=HTTPMethod.GET, path="/test")
        
        assert request1.request_id != request2.request_id
        assert len(request1.request_id) > 0


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestGraphQLSchema:
    """Tests for GraphQL schema and queries."""

    @pytest.mark.asyncio
    async def test_type_definition(self, graphql_schema):
        """Test GraphQL type definition."""
        graphql_schema.define_type("User", {
            "id": "ID!",
            "name": "String!",
            "email": "String"
        })
        
        assert "User" in graphql_schema.types
        assert graphql_schema.types["User"]["id"] == "ID!"
        assert graphql_schema.types["User"]["name"] == "String!"
    
    @pytest.mark.asyncio
    async def test_query_definition(self, graphql_schema):
        """Test GraphQL query definition."""
        graphql_schema.define_query(
            "user",
            "User",
            {"id": "ID!"}
        )
        
        assert "user" in graphql_schema.queries
        assert graphql_schema.queries["user"]["return_type"] == "User"
        assert graphql_schema.queries["user"]["args"]["id"] == "ID!"
    
    @pytest.mark.asyncio
    async def test_mutation_definition(self, graphql_schema):
        """Test GraphQL mutation definition."""
        graphql_schema.define_mutation(
            "createUser",
            "User!",
            {"name": "String!", "email": "String"}
        )
        
        assert "createUser" in graphql_schema.mutations
        assert graphql_schema.mutations["createUser"]["return_type"] == "User!"
    
    @pytest.mark.asyncio
    async def test_resolver_registration(self, graphql_schema):
        """Test GraphQL resolver registration."""
        def user_resolver(args):
            return {"id": "1", "name": "Test User"}
        
        graphql_schema.register_resolver("user", user_resolver)
        
        assert "user" in graphql_schema.resolvers
        assert graphql_schema.resolvers["user"] == user_resolver
    
    @pytest.mark.asyncio
    async def test_schema_string_generation(self, graphql_schema):
        """Test GraphQL schema string generation."""
        graphql_schema.define_type("User", {
            "id": "ID!",
            "name": "String!"
        })
        
        graphql_schema.define_query("user", "User", {"id": "ID!"})
        graphql_schema.define_mutation("createUser", "User!", {"name": "String!"})
        
        schema_str = graphql_schema.to_schema_string()
        
        assert "type User" in schema_str
        assert "type Query" in schema_str
        assert "type Mutation" in schema_str
        assert "user(id: ID!): User" in schema_str
        assert "createUser(name: String!): User!" in schema_str
    
    @pytest.mark.asyncio
    async def test_schema_with_knowledge_types(self, graphql_schema):
        """Test GraphQL schema with knowledge-specific types."""
        # Define KnowledgeItem type
        graphql_schema.define_type("KnowledgeItem", {
            "id": "ID!",
            "content": "String!",
            "type": "String!",
            "createdAt": "String!",
            "tags": "[String!]!"
        })
        
        # Define queries
        graphql_schema.define_query("knowledgeItem", "KnowledgeItem", {"id": "ID!"})
        graphql_schema.define_query("search", "[KnowledgeItem!]!", {"query": "String!"})
        
        # Define mutations
        graphql_schema.define_mutation("createKnowledge", "KnowledgeItem!", {"content": "String!"})
        graphql_schema.define_mutation("updateKnowledge", "KnowledgeItem!", {"id": "ID!", "content": "String"})
        
        schema_str = graphql_schema.to_schema_string()
        
        # Verify schema contains expected elements
        assert "type KnowledgeItem" in schema_str
        assert "id: ID!" in schema_str
        assert "content: String!" in schema_str
        assert "knowledgeItem(id: ID!): KnowledgeItem" in schema_str
        assert "search(query: String!): [KnowledgeItem!]!" in schema_str
        assert "createKnowledge(content: String!): KnowledgeItem!" in schema_str


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestKnowledgeAPIFactoryREST:
    """Tests for KnowledgeAPIFactory REST endpoints."""

    @pytest.mark.asyncio
    async def test_create_rest_api_registers_all_routes(self, mock_platform):
        """Test that REST API factory registers all expected routes."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        # Check that expected routes are registered
        paths = [(route.path, route.method.value) for route in gateway.routes]
        
        assert ("/knowledge", "GET") in paths
        assert ("/knowledge", "POST") in paths
        assert ("/knowledge/{id}", "GET") in paths
        assert ("/knowledge/{id}", "PUT") in paths
        assert ("/knowledge/{id}", "DELETE") in paths
        assert ("/search", "GET") in paths
        assert ("/recommendations", "GET") in paths
        assert ("/health", "GET") in paths
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, mock_platform):
        """Test health check endpoint."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(method=HTTPMethod.GET, path="/health")
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_unhealthy_platform(self, mock_platform):
        """Test health check when platform is unhealthy."""
        mock_platform.health_check = MagicMock(return_value={"status": "degraded"})
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(method=HTTPMethod.GET, path="/health")
        response = await gateway.handle_request(request)
        
        assert response.status_code == 503
    
    @pytest.mark.asyncio
    async def test_health_check_requires_no_auth(self, mock_platform):
        """Test that health check doesn't require authentication."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        # Find health route
        health_route = next(r for r in gateway.routes if r.path == "/health")
        
        assert health_route.requires_auth is False
    
    @pytest.mark.asyncio
    async def test_search_route_has_rate_limit(self, mock_platform):
        """Test that search route has rate limiting."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        # Find search route
        search_route = next(r for r in gateway.routes if r.path == "/search")
        
        assert search_route.rate_limit == 100
    
    @pytest.mark.asyncio
    async def test_create_knowledge_requires_body(self, mock_platform):
        """Test that create knowledge requires request body."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/knowledge",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 400
        assert "Request body required" in response.error
    
    @pytest.mark.asyncio
    async def test_create_knowledge_success(self, mock_platform):
        """Test successful knowledge creation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/knowledge",
            user_id="user123",
            body={"content": "Test knowledge", "type": "concept"}
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 201
        assert response.data["created"] is True
        assert "id" in response.data
    
    @pytest.mark.asyncio
    async def test_get_knowledge_returns_item(self, mock_platform):
        """Test getting a knowledge item."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/knowledge/test-id-123",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["id"] == "test-id-123"
    
    @pytest.mark.asyncio
    async def test_update_knowledge_success(self, mock_platform):
        """Test updating knowledge."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.PUT,
            path="/knowledge/test-id-123",
            user_id="user123",
            body={"content": "Updated content"}
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["updated"] is True
        assert response.data["id"] == "test-id-123"
    
    @pytest.mark.asyncio
    async def test_delete_knowledge_success(self, mock_platform):
        """Test deleting knowledge."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.DELETE,
            path="/knowledge/test-id-123",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 204
        assert response.data["deleted"] is True
    
    @pytest.mark.asyncio
    async def test_list_knowledge(self, mock_platform):
        """Test listing knowledge items."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/knowledge",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert "items" in response.data
        assert "total" in response.data
    
    @pytest.mark.asyncio
    async def test_search_with_query_params(self, mock_platform):
        """Test search with query parameters."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/search",
            user_id="user123",
            query_params={"q": "artificial intelligence", "limit": "10"}
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["query"] == "artificial intelligence"
    
    @pytest.mark.asyncio
    async def test_recommendations_endpoint(self, mock_platform):
        """Test recommendations endpoint."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/recommendations",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert "recommendations" in response.data
    
    @pytest.mark.asyncio
    async def test_knowledge_endpoints_require_auth(self, mock_platform):
        """Test that knowledge endpoints require authentication."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        # Test GET /knowledge without auth
        request = APIRequest(method=HTTPMethod.GET, path="/knowledge")
        response = await gateway.handle_request(request)
        assert response.status_code == 401
        
        # Test POST /knowledge without auth
        request = APIRequest(method=HTTPMethod.POST, path="/knowledge", body={"test": "data"})
        response = await gateway.handle_request(request)
        assert response.status_code == 401


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestKnowledgeAPIFactoryGraphQL:
    """Tests for KnowledgeAPIFactory GraphQL schema."""

    @pytest.mark.asyncio
    async def test_create_graphql_schema(self, mock_platform):
        """Test GraphQL schema creation."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        # Check types
        assert "KnowledgeItem" in schema.types
        assert "SearchResult" in schema.types
        
        # Check queries
        assert "knowledgeItem" in schema.queries
        assert "search" in schema.queries
        
        # Check mutations
        assert "createKnowledge" in schema.mutations
        assert "updateKnowledge" in schema.mutations
    
    @pytest.mark.asyncio
    async def test_graphql_knowledge_item_type(self, mock_platform):
        """Test KnowledgeItem type definition in GraphQL schema."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        knowledge_type = schema.types["KnowledgeItem"]
        assert knowledge_type["id"] == "ID!"
        assert knowledge_type["content"] == "String!"
        assert knowledge_type["type"] == "String!"
        assert knowledge_type["createdAt"] == "String!"
        assert knowledge_type["updatedAt"] == "String"
        assert knowledge_type["tags"] == "[String!]!"
        assert knowledge_type["metadata"] == "JSON"
    
    @pytest.mark.asyncio
    async def test_graphql_search_result_type(self, mock_platform):
        """Test SearchResult type definition in GraphQL schema."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        search_result_type = schema.types["SearchResult"]
        assert search_result_type["item"] == "KnowledgeItem!"
        assert search_result_type["score"] == "Float!"
        assert search_result_type["highlights"] == "[String!]"
    
    @pytest.mark.asyncio
    async def test_graphql_knowledge_item_query(self, mock_platform):
        """Test knowledgeItem query definition."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        query = schema.queries["knowledgeItem"]
        assert query["return_type"] == "KnowledgeItem"
        assert query["args"]["id"] == "ID!"
    
    @pytest.mark.asyncio
    async def test_graphql_search_query(self, mock_platform):
        """Test search query definition."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        query = schema.queries["search"]
        assert query["return_type"] == "[SearchResult!]!"
        assert query["args"]["query"] == "String!"
        assert query["args"]["limit"] == "Int"
    
    @pytest.mark.asyncio
    async def test_graphql_create_knowledge_mutation(self, mock_platform):
        """Test createKnowledge mutation definition."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        mutation = schema.mutations["createKnowledge"]
        assert mutation["return_type"] == "KnowledgeItem!"
        # Mutation uses input type pattern
        assert "input" in mutation["args"]
    
    @pytest.mark.asyncio
    async def test_graphql_update_knowledge_mutation(self, mock_platform):
        """Test updateKnowledge mutation definition."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        
        mutation = schema.mutations["updateKnowledge"]
        assert mutation["return_type"] == "KnowledgeItem!"
        assert mutation["args"]["id"] == "ID!"
        # Mutation uses input type pattern
        assert "input" in mutation["args"]
    
    @pytest.mark.asyncio
    async def test_graphql_schema_string_complete(self, mock_platform):
        """Test that generated schema string is complete and valid."""
        schema = KnowledgeAPIFactory.create_graphql_schema(mock_platform)
        schema_str = schema.to_schema_string()
        
        # Should contain all types
        assert "type KnowledgeItem" in schema_str
        assert "type SearchResult" in schema_str
        
        # Should contain Query type
        assert "type Query" in schema_str
        assert "knowledgeItem(id: ID!): KnowledgeItem" in schema_str
        # Search query has filters argument as well
        assert "search(query: String!" in schema_str
        assert "[SearchResult!]!" in schema_str
        
        # Should contain Mutation type
        assert "type Mutation" in schema_str
        # Mutations use input type pattern
        assert "createKnowledge(" in schema_str
        assert "updateKnowledge(" in schema_str


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestAPIDocumentation:
    """Tests for API documentation generation."""

    @pytest.mark.asyncio
    async def test_generate_openapi_spec(self, api_gateway):
        """Test OpenAPI spec generation."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/users",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True
        ))
        api_gateway.register_route(Route(
            path="/users",
            method=HTTPMethod.POST,
            handler=handler,
            requires_auth=True
        ))
        api_gateway.register_route(Route(
            path="/public",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        spec = APIDocumentation.generate_openapi_spec(api_gateway)
        
        assert spec["openapi"] == "3.0.0"
        assert spec["info"]["title"] == "Knowledge API"
        assert "/users" in spec["paths"]
        assert "get" in spec["paths"]["/users"]
        assert "post" in spec["paths"]["/users"]
        assert "/public" in spec["paths"]
    
    @pytest.mark.asyncio
    async def test_openapi_security_schemes(self, api_gateway):
        """Test OpenAPI security schemes."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/protected",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True
        ))
        
        spec = APIDocumentation.generate_openapi_spec(api_gateway)
        
        assert "components" in spec
        assert "securitySchemes" in spec["components"]
        assert "bearerAuth" in spec["components"]["securitySchemes"]
        assert spec["components"]["securitySchemes"]["bearerAuth"]["type"] == "http"
        assert spec["components"]["securitySchemes"]["bearerAuth"]["scheme"] == "bearer"
    
    @pytest.mark.asyncio
    async def test_openapi_route_security(self, api_gateway):
        """Test OpenAPI route security definitions."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        api_gateway.register_route(Route(
            path="/protected",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=True
        ))
        api_gateway.register_route(Route(
            path="/public",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        spec = APIDocumentation.generate_openapi_spec(api_gateway)
        
        # Protected route should have security
        assert spec["paths"]["/protected"]["get"]["security"] == [{"bearerAuth": []}]
        
        # Public route should have empty security
        assert spec["paths"]["/public"]["get"]["security"] == []
    
    @pytest.mark.asyncio
    async def test_openapi_spec_with_knowledge_routes(self, mock_platform):
        """Test OpenAPI spec generation with knowledge API routes."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        spec = APIDocumentation.generate_openapi_spec(gateway)
        
        # Should contain all knowledge routes
        assert "/knowledge" in spec["paths"]
        assert "/knowledge/{id}" in spec["paths"]
        assert "/search" in spec["paths"]
        assert "/health" in spec["paths"]
        
        # Health should have empty security (public)
        assert spec["paths"]["/health"]["get"]["security"] == []


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestMiddleware:
    """Tests for middleware functionality."""

    @pytest.mark.asyncio
    async def test_middleware_execution(self, api_gateway):
        """Test that middleware is executed."""
        def handler(req): return APIResponse(status_code=200, data={"from": "handler"})
        
        middleware_called = []
        
        def test_middleware(req):
            middleware_called.append("called")
            return None  # Continue to handler
        
        api_gateway.add_middleware(test_middleware)
        api_gateway.register_route(Route(
            path="/test",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/test")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 200
        assert "called" in middleware_called
    
    @pytest.mark.asyncio
    async def test_middleware_short_circuit(self, api_gateway):
        """Test that middleware can short-circuit request."""
        def handler(req): return APIResponse(status_code=200, data={"from": "handler"})
        
        def blocking_middleware(req):
            return APIResponse(status_code=403, error="Blocked by middleware")
        
        api_gateway.add_middleware(blocking_middleware)
        api_gateway.register_route(Route(
            path="/test",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/test")
        response = await api_gateway.handle_request(request)
        
        assert response.status_code == 403
        assert "Blocked by middleware" in response.error
    
    @pytest.mark.asyncio
    async def test_multiple_middleware_execution_order(self, api_gateway):
        """Test middleware execution order."""
        def handler(req): return APIResponse(status_code=200, data={})
        
        execution_order = []
        
        def middleware1(req):
            execution_order.append(1)
            return None
        
        def middleware2(req):
            execution_order.append(2)
            return None
        
        api_gateway.add_middleware(middleware1)
        api_gateway.add_middleware(middleware2)
        api_gateway.register_route(Route(
            path="/test",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/test")
        await api_gateway.handle_request(request)
        
        assert execution_order == [1, 2]


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestCRUDOperations:
    """Comprehensive CRUD operation tests."""

    @pytest.mark.asyncio
    async def test_list_knowledge(self, mock_platform):
        """Test LIST operation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/knowledge",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert "items" in response.data
        assert "total" in response.data
    
    @pytest.mark.asyncio
    async def test_create_knowledge(self, mock_platform):
        """Test CREATE operation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/knowledge",
            user_id="user123",
            body={"content": "Test knowledge item", "type": "concept"}
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 201
        assert response.data["created"] is True
    
    @pytest.mark.asyncio
    async def test_read_knowledge(self, mock_platform):
        """Test READ operation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/knowledge/item-123",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["id"] == "item-123"
    
    @pytest.mark.asyncio
    async def test_update_knowledge(self, mock_platform):
        """Test UPDATE operation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.PUT,
            path="/knowledge/item-123",
            user_id="user123",
            body={"content": "Updated content"}
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 200
        assert response.data["updated"] is True
    
    @pytest.mark.asyncio
    async def test_delete_knowledge(self, mock_platform):
        """Test DELETE operation."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.DELETE,
            path="/knowledge/item-123",
            user_id="user123"
        )
        response = await gateway.handle_request(request)
        
        assert response.status_code == 204
        assert response.data["deleted"] is True
    
    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, mock_platform):
        """Test full CRUD workflow."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        user_id = "user123"
        
        # 1. Create
        create_request = APIRequest(
            method=HTTPMethod.POST,
            path="/knowledge",
            user_id=user_id,
            body={"content": "New knowledge item", "type": "note"}
        )
        create_response = await gateway.handle_request(create_request)
        assert create_response.status_code == 201
        item_id = create_response.data["id"]
        
        # 2. Read
        read_request = APIRequest(
            method=HTTPMethod.GET,
            path=f"/knowledge/{item_id}",
            user_id=user_id
        )
        read_response = await gateway.handle_request(read_request)
        assert read_response.status_code == 200
        assert read_response.data["id"] == item_id
        
        # 3. Update
        update_request = APIRequest(
            method=HTTPMethod.PUT,
            path=f"/knowledge/{item_id}",
            user_id=user_id,
            body={"content": "Updated knowledge item"}
        )
        update_response = await gateway.handle_request(update_request)
        assert update_response.status_code == 200
        assert update_response.data["updated"] is True
        
        # 4. Delete
        delete_request = APIRequest(
            method=HTTPMethod.DELETE,
            path=f"/knowledge/{item_id}",
            user_id=user_id
        )
        delete_response = await gateway.handle_request(delete_request)
        assert delete_response.status_code == 204
        assert delete_response.data["deleted"] is True


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestValidation:
    """Tests for request/response validation."""

    @pytest.mark.asyncio
    async def test_valid_http_methods(self):
        """Test all valid HTTP methods."""
        methods = [HTTPMethod.GET, HTTPMethod.POST, HTTPMethod.PUT, 
                   HTTPMethod.PATCH, HTTPMethod.DELETE]
        
        for method in methods:
            request = APIRequest(method=method, path="/test")
            assert request.method == method
            assert request.to_dict()["method"] == method.value
    
    @pytest.mark.asyncio
    async def test_request_with_body_validation(self):
        """Test request body validation."""
        # Valid JSON body
        request = APIRequest(
            method=HTTPMethod.POST,
            path="/test",
            body={"key": "value", "number": 123, "nested": {"a": 1}}
        )
        
        assert request.body is not None
        assert request.body["key"] == "value"
        assert request.body["number"] == 123
        assert request.body["nested"]["a"] == 1
    
    @pytest.mark.asyncio
    async def test_response_headers_validation(self):
        """Test response headers are properly set."""
        response = APIResponse(
            status_code=200,
            data={},
            headers={"Content-Type": "application/json", "X-Custom-Header": "value", "X-Request-ID": "abc123"}
        )
        
        assert response.headers["Content-Type"] == "application/json"
        assert response.headers["X-Custom-Header"] == "value"
        assert response.headers["X-Request-ID"] == "abc123"
    
    @pytest.mark.asyncio
    async def test_request_timestamp_generation(self):
        """Test that request timestamps are auto-generated."""
        before = datetime.utcnow()
        request = APIRequest(method=HTTPMethod.GET, path="/test")
        after = datetime.utcnow()
        
        assert before <= request.timestamp <= after
    
    @pytest.mark.asyncio
    async def test_response_error_vs_data_exclusivity(self):
        """Test that response contains either data or error, not both in to_dict."""
        # Response with data
        data_response = APIResponse(status_code=200, data={"key": "value"})
        data_dict = data_response.to_dict()
        assert "data" in data_dict
        assert "error" not in data_dict
        
        # Response with error
        error_response = APIResponse(status_code=400, error="Bad request")
        error_dict = error_response.to_dict()
        assert "error" in error_dict
        assert "data" not in error_dict


@pytest.mark.skipif(not GATEWAY_AVAILABLE, reason="API Gateway module not available")
class TestAdvancedScenarios:
    """Advanced test scenarios."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, mock_platform):
        """Test handling of concurrent requests."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        async def make_request(user_id: str):
            request = APIRequest(
                method=HTTPMethod.GET,
                path="/knowledge",
                user_id=user_id
            )
            return await gateway.handle_request(request)
        
        # Make 10 concurrent requests
        tasks = [make_request(f"user{i}") for i in range(10)]
        responses = await asyncio.gather(*tasks)
        
        # All should succeed
        for response in responses:
            assert response.status_code == 200
            assert "items" in response.data
    
    @pytest.mark.asyncio
    async def test_request_id_propagation(self, api_gateway):
        """Test that request_id is propagated to response."""
        def handler(req): 
            return APIResponse(status_code=200, data={"received_id": req.request_id})
        
        api_gateway.register_route(Route(
            path="/test",
            method=HTTPMethod.GET,
            handler=handler,
            requires_auth=False
        ))
        
        request = APIRequest(method=HTTPMethod.GET, path="/test")
        response = await api_gateway.handle_request(request)
        
        assert response.request_id == request.request_id
    
    @pytest.mark.asyncio
    async def test_query_params_parsing(self, mock_platform):
        """Test query parameters parsing."""
        gateway = KnowledgeAPIFactory.create_rest_api(mock_platform)
        
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/search",
            user_id="user123",
            query_params={
                "q": "machine learning",
                "limit": "10",
                "offset": "0",
                "sort": "relevance"
            }
        )
        
        response = await gateway.handle_request(request)
        assert response.status_code == 200
        # Verify query params are accessible
        assert request.query_params["q"] == "machine learning"
        assert request.query_params["limit"] == "10"
    
    @pytest.mark.asyncio
    async def test_tenant_isolation_in_request(self):
        """Test that tenant_id is properly handled in requests."""
        request = APIRequest(
            method=HTTPMethod.GET,
            path="/knowledge",
            user_id="user123",
            tenant_id="tenant456",
            headers={"X-Tenant-ID": "tenant456"}
        )
        
        assert request.tenant_id == "tenant456"
        dict_repr = request.to_dict()
        assert dict_repr["tenant_id"] == "tenant456"


# ============================================================================
# Run tests if executed directly
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
