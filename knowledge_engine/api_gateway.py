"""
API Gateway

Provides REST and GraphQL APIs for the knowledge platform:
- RESTful endpoints
- GraphQL schema and resolvers
- Authentication and rate limiting
- Request/response transformation
- API versioning
- Documentation generation
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid

logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


@dataclass
class APIRequest:
    """API request representation."""
    method: HTTPMethod
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    query_params: Dict[str, str] = field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    user_id: Optional[str] = None
    tenant_id: Optional[str] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "method": self.method.value,
            "path": self.path,
            "headers": self.headers,
            "query_params": self.query_params,
            "body": self.body,
            "user_id": self.user_id,
            "tenant_id": self.tenant_id,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class APIResponse:
    """API response representation."""
    status_code: int
    data: Any = None
    headers: Dict[str, str] = field(default_factory=dict)
    error: Optional[str] = None
    request_id: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "status_code": self.status_code,
            "request_id": self.request_id
        }
        
        if self.error:
            result["error"] = self.error
        else:
            result["data"] = self.data
        
        return result


@dataclass
class Route:
    """API route definition."""
    path: str
    method: HTTPMethod
    handler: Callable[[APIRequest], APIResponse]
    requires_auth: bool = True
    required_permissions: Set[str] = field(default_factory=set)
    rate_limit: Optional[int] = None  # requests per minute


class RateLimiter:
    """Simple rate limiter."""
    
    def __init__(self):
        self._requests: Dict[str, List[datetime]] = {}
    
    def check_rate_limit(self, key: str, limit: int, window_seconds: int = 60) -> bool:
        """Check if request is within rate limit."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        # Get recent requests
        requests = self._requests.get(key, [])
        requests = [r for r in requests if r > window_start]
        
        # Check limit
        if len(requests) >= limit:
            return False
        
        # Record request
        requests.append(now)
        self._requests[key] = requests
        
        return True
    
    def get_remaining(self, key: str, limit: int, window_seconds: int = 60) -> int:
        """Get remaining requests in window."""
        now = datetime.utcnow()
        window_start = now - timedelta(seconds=window_seconds)
        
        requests = self._requests.get(key, [])
        requests = [r for r in requests if r > window_start]
        
        return max(0, limit - len(requests))


class RESTAPIGateway:
    """
    REST API Gateway.
    """
    
    def __init__(self):
        self.routes: List[Route] = []
        self.rate_limiter = RateLimiter()
        self.middleware: List[Callable[[APIRequest], Optional[APIResponse]]] = []
        
    def register_route(self, route: Route):
        """Register a route."""
        self.routes.append(route)
        logger.info(f"Registered route: {route.method.value} {route.path}")
    
    def add_middleware(self, middleware: Callable[[APIRequest], Optional[APIResponse]]):
        """Add middleware."""
        self.middleware.append(middleware)
    
    async def handle_request(self, request: APIRequest) -> APIResponse:
        """Handle an incoming request."""
        # Apply middleware
        for mw in self.middleware:
            response = mw(request)
            if response:
                return response
        
        # Find matching route
        route = self._match_route(request.path, request.method)
        
        if not route:
            return APIResponse(
                status_code=404,
                error="Not Found",
                request_id=request.request_id
            )
        
        # Check authentication
        if route.requires_auth and not request.user_id:
            return APIResponse(
                status_code=401,
                error="Unauthorized",
                request_id=request.request_id
            )
        
        # Check rate limit
        if route.rate_limit:
            rate_key = f"{request.user_id or 'anonymous'}:{route.path}"
            if not self.rate_limiter.check_rate_limit(rate_key, route.rate_limit):
                return APIResponse(
                    status_code=429,
                    error="Rate limit exceeded",
                    request_id=request.request_id,
                    headers={
                        "X-RateLimit-Remaining": "0",
                        "Retry-After": "60"
                    }
                )
        
        # Execute handler
        try:
            response = route.handler(request)
            if hasattr(response, '__await__'):
                response = await response
            response.request_id = request.request_id
            return response
            
        except Exception as e:
            logger.error(f"Error handling request: {e}")
            return APIResponse(
                status_code=500,
                error="Internal Server Error",
                request_id=request.request_id
            )
    
    def _match_route(self, path: str, method: HTTPMethod) -> Optional[Route]:
        """Match request to route."""
        for route in self.routes:
            if route.method == method and self._path_matches(route.path, path):
                return route
        return None
    
    def _path_matches(self, route_path: str, request_path: str) -> bool:
        """Check if path matches route pattern."""
        # Simple pattern matching - convert route to regex
        pattern = route_path.replace("{", "(?P<").replace("}", ">[^/]+)")
        return bool(re.match(f"^{pattern}$", request_path))


class GraphQLSchema:
    """
    GraphQL schema definition (simplified).
    """
    
    def __init__(self):
        self.types: Dict[str, Dict[str, str]] = {}
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.mutations: Dict[str, Dict[str, Any]] = {}
        self.resolvers: Dict[str, Callable] = {}
    
    def define_type(self, name: str, fields: Dict[str, str]):
        """Define a GraphQL type."""
        self.types[name] = fields
    
    def define_query(self, name: str, return_type: str, args: Dict[str, str] = None):
        """Define a query."""
        self.queries[name] = {
            "return_type": return_type,
            "args": args or {}
        }
    
    def define_mutation(self, name: str, return_type: str, args: Dict[str, str] = None):
        """Define a mutation."""
        self.mutations[name] = {
            "return_type": return_type,
            "args": args or {}
        }
    
    def register_resolver(self, field: str, resolver: Callable):
        """Register a resolver."""
        self.resolvers[field] = resolver
    
    def to_schema_string(self) -> str:
        """Generate GraphQL schema string."""
        lines = []
        
        # Types
        for type_name, fields in self.types.items():
            lines.append(f"type {type_name} {{")
            for field_name, field_type in fields.items():
                lines.append(f"  {field_name}: {field_type}")
            lines.append("}")
            lines.append("")
        
        # Queries
        if self.queries:
            lines.append("type Query {")
            for query_name, query_def in self.queries.items():
                args_str = ""
                if query_def["args"]:
                    args_list = [f"{k}: {v}" for k, v in query_def["args"].items()]
                    args_str = f"({', '.join(args_list)})"
                lines.append(f"  {query_name}{args_str}: {query_def['return_type']}")
            lines.append("}")
            lines.append("")
        
        # Mutations
        if self.mutations:
            lines.append("type Mutation {")
            for mutation_name, mutation_def in self.mutations.items():
                args_str = ""
                if mutation_def["args"]:
                    args_list = [f"{k}: {v}" for k, v in mutation_def["args"].items()]
                    args_str = f"({', '.join(args_list)})"
                lines.append(f"  {mutation_name}{args_str}: {mutation_def['return_type']}")
            lines.append("}")
        
        return "\n".join(lines)


class APIDocumentation:
    """Generate API documentation."""
    
    @staticmethod
    def generate_openapi_spec(
        gateway: RESTAPIGateway,
        title: str = "Knowledge API",
        version: str = "1.0.0"
    ) -> Dict[str, Any]:
        """Generate OpenAPI specification."""
        spec = {
            "openapi": "3.0.0",
            "info": {
                "title": title,
                "version": version,
                "description": "Knowledge Platform API"
            },
            "paths": {}
        }
        
        for route in gateway.routes:
            path = route.path
            method = route.method.value.lower()
            
            if path not in spec["paths"]:
                spec["paths"][path] = {}
            
            spec["paths"][path][method] = {
                "summary": f"{route.method.value} {path}",
                "security": [{"bearerAuth": []}] if route.requires_auth else [],
                "responses": {
                    "200": {"description": "Success"},
                    "401": {"description": "Unauthorized"},
                    "429": {"description": "Rate limit exceeded"}
                }
            }
        
        spec["components"] = {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "bearerFormat": "JWT"
                }
            }
        }
        
        return spec


class KnowledgeAPIFactory:
    """
    Factory for creating Knowledge Platform APIs.
    """
    
    @staticmethod
    def create_rest_api(platform) -> RESTAPIGateway:
        """Create REST API for knowledge platform."""
        gateway = RESTAPIGateway()
        
        # Knowledge routes
        gateway.register_route(Route(
            path="/knowledge",
            method=HTTPMethod.GET,
            handler=lambda req: KnowledgeAPIFactory._list_knowledge(req, platform),
            requires_auth=True
        ))
        
        gateway.register_route(Route(
            path="/knowledge",
            method=HTTPMethod.POST,
            handler=lambda req: KnowledgeAPIFactory._create_knowledge(req, platform),
            requires_auth=True
        ))
        
        gateway.register_route(Route(
            path="/knowledge/{id}",
            method=HTTPMethod.GET,
            handler=lambda req: KnowledgeAPIFactory._get_knowledge(req, platform),
            requires_auth=True
        ))
        
        gateway.register_route(Route(
            path="/knowledge/{id}",
            method=HTTPMethod.PUT,
            handler=lambda req: KnowledgeAPIFactory._update_knowledge(req, platform),
            requires_auth=True
        ))
        
        gateway.register_route(Route(
            path="/knowledge/{id}",
            method=HTTPMethod.DELETE,
            handler=lambda req: KnowledgeAPIFactory._delete_knowledge(req, platform),
            requires_auth=True
        ))
        
        # Search route
        gateway.register_route(Route(
            path="/search",
            method=HTTPMethod.GET,
            handler=lambda req: KnowledgeAPIFactory._search(req, platform),
            requires_auth=True,
            rate_limit=100
        ))
        
        # Recommendations route
        gateway.register_route(Route(
            path="/recommendations",
            method=HTTPMethod.GET,
            handler=lambda req: KnowledgeAPIFactory._get_recommendations(req, platform),
            requires_auth=True
        ))
        
        # Health check (public)
        gateway.register_route(Route(
            path="/health",
            method=HTTPMethod.GET,
            handler=lambda req: KnowledgeAPIFactory._health_check(req, platform),
            requires_auth=False
        ))
        
        return gateway
    
    @staticmethod
    def _list_knowledge(req: APIRequest, platform) -> APIResponse:
        """List knowledge items."""
        # Implementation would query platform
        return APIResponse(
            status_code=200,
            data={"items": [], "total": 0}
        )
    
    @staticmethod
    def _create_knowledge(req: APIRequest, platform) -> APIResponse:
        """Create knowledge item."""
        if not req.body:
            return APIResponse(status_code=400, error="Request body required")
        
        # Implementation would call platform.add_knowledge()
        return APIResponse(
            status_code=201,
            data={"id": str(uuid.uuid4()), "created": True}
        )
    
    @staticmethod
    def _get_knowledge(req: APIRequest, platform) -> APIResponse:
        """Get knowledge item."""
        # Extract ID from path
        item_id = req.path.split("/")[-1]
        
        return APIResponse(
            status_code=200,
            data={"id": item_id, "content": "..."}
        )
    
    @staticmethod
    def _update_knowledge(req: APIRequest, platform) -> APIResponse:
        """Update knowledge item."""
        item_id = req.path.split("/")[-1]
        
        return APIResponse(
            status_code=200,
            data={"id": item_id, "updated": True}
        )
    
    @staticmethod
    def _delete_knowledge(req: APIRequest, platform) -> APIResponse:
        """Delete knowledge item."""
        item_id = req.path.split("/")[-1]
        
        return APIResponse(
            status_code=204,
            data={"id": item_id, "deleted": True}
        )
    
    @staticmethod
    def _search(req: APIRequest, platform) -> APIResponse:
        """Search knowledge."""
        query = req.query_params.get("q", "")
        
        return APIResponse(
            status_code=200,
            data={"query": query, "results": []}
        )
    
    @staticmethod
    def _get_recommendations(req: APIRequest, platform) -> APIResponse:
        """Get recommendations."""
        return APIResponse(
            status_code=200,
            data={"recommendations": []}
        )
    
    @staticmethod
    def _health_check(req: APIRequest, platform) -> APIResponse:
        """Health check endpoint."""
        health = platform.health_check() if hasattr(platform, 'health_check') else {"status": "ok"}
        
        return APIResponse(
            status_code=200 if health.get("status") == "healthy" else 503,
            data=health
        )
    
    @staticmethod
    def create_graphql_schema(platform) -> GraphQLSchema:
        """Create GraphQL schema for knowledge platform."""
        schema = GraphQLSchema()
        
        # Define types
        schema.define_type("KnowledgeItem", {
            "id": "ID!",
            "content": "String!",
            "type": "String!",
            "createdAt": "String!",
            "updatedAt": "String",
            "tags": "[String!]!",
            "metadata": "JSON"
        })
        
        schema.define_type("SearchResult", {
            "item": "KnowledgeItem!",
            "score": "Float!",
            "highlights": "[String!]"
        })
        
        # Define queries
        schema.define_query(
            "knowledgeItem",
            "KnowledgeItem",
            {"id": "ID!"}
        )
        
        schema.define_query(
            "search",
            "[SearchResult!]!",
            {"query": "String!", "limit": "Int"}
        )
        
        # Define mutations
        schema.define_mutation(
            "createKnowledge",
            "KnowledgeItem!",
            {"content": "String!", "type": "String"}
        )
        
        schema.define_mutation(
            "updateKnowledge",
            "KnowledgeItem!",
            {"id": "ID!", "content": "String"}
        )
        
        return schema


__all__ = [
    "RESTAPIGateway",
    "GraphQLSchema",
    "APIRequest",
    "APIResponse",
    "Route",
    "RateLimiter",
    "KnowledgeAPIFactory",
    "APIDocumentation"
]
