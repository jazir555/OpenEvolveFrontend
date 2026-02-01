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
        
        schema.define_type("DeleteStatus", {
            "success": "Boolean!",
            "id": "ID!",
            "message": "String"
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
            {"query": "String!", "filters": "JSON", "limit": "Int"}
        )
        
        # Define mutations
        schema.define_mutation(
            "createKnowledge",
            "KnowledgeItem!",
            {"input": "CreateKnowledgeInput!"}
        )
        
        schema.define_mutation(
            "updateKnowledge",
            "KnowledgeItem!",
            {"id": "ID!", "input": "UpdateKnowledgeInput!"}
        )
        
        schema.define_mutation(
            "deleteKnowledge",
            "DeleteStatus!",
            {"id": "ID!"}
        )
        
        # Define input types (for schema documentation)
        schema.define_type("CreateKnowledgeInput", {
            "content": "String!",
            "type": "String",
            "tags": "[String!]",
            "metadata": "JSON",
            "source": "String",
            "confidence": "Float"
        })
        
        schema.define_type("UpdateKnowledgeInput", {
            "content": "String",
            "tags": "[String!]",
            "metadata": "JSON",
            "confidence": "Float"
        })
        
        # Register resolvers
        KnowledgeAPIFactory._register_graphql_resolvers(schema, platform)
        
        return schema
    
    @staticmethod
    def _register_graphql_resolvers(schema: GraphQLSchema, platform):
        """Register GraphQL resolvers for the schema."""
        
        # ==================== Query Resolvers ====================
        
        async def resolve_knowledge_item(parent, info, id: str) -> Optional[Dict[str, Any]]:
            """
            Resolver for knowledgeItem(id) query.
            
            Args:
                parent: Parent object (None for root query)
                info: GraphQL execution info
                id: Knowledge item ID
                
            Returns:
                Knowledge item dictionary or None if not found
            """
            try:
                # Access platform through knowledge_engine
                if hasattr(platform, 'knowledge_engine'):
                    item = await platform.knowledge_engine.get_knowledge(id)
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'knowledge_engine'):
                    item = await platform.platform.knowledge_engine.get_knowledge(id)
                else:
                    logger.error("Platform does not have knowledge_engine attribute")
                    return None
                
                if item is None:
                    return None
                
                return KnowledgeAPIFactory._knowledge_item_to_graphql(item)
                
            except Exception as e:
                logger.error(f"Error resolving knowledgeItem({id}): {e}")
                raise Exception(f"Failed to retrieve knowledge item: {str(e)}")
        
        async def resolve_search(
            parent,
            info,
            query: str,
            filters: Optional[Dict[str, Any]] = None,
            limit: int = 10
        ) -> List[Dict[str, Any]]:
            """
            Resolver for search(query, filters, limit) query.
            
            Args:
                parent: Parent object (None for root query)
                info: GraphQL execution info
                query: Search query string
                filters: Optional search filters
                limit: Maximum number of results
                
            Returns:
                List of search result dictionaries
            """
            try:
                # Get user_id from context if available
                user_id = None
                if info and hasattr(info, 'context') and info.context:
                    user_id = info.context.get('user_id')
                
                # Access platform search method
                if hasattr(platform, 'search'):
                    results = await platform.search(
                        query=query,
                        user_id=user_id,
                        filters=filters,
                        max_results=limit
                    )
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'search'):
                    results = await platform.platform.search(
                        query=query,
                        user_id=user_id,
                        filters=filters,
                        max_results=limit
                    )
                else:
                    logger.error("Platform does not have search method")
                    return []
                
                return KnowledgeAPIFactory._search_results_to_graphql(results)
                
            except Exception as e:
                logger.error(f"Error resolving search({query}): {e}")
                raise Exception(f"Failed to search knowledge: {str(e)}")
        
        # ==================== Mutation Resolvers ====================
        
        async def resolve_create_knowledge(
            parent,
            info,
            input: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Resolver for createKnowledge(input) mutation.
            
            Args:
                parent: Parent object (None for root mutation)
                info: GraphQL execution info
                input: Create knowledge input containing content, type, tags, etc.
                
            Returns:
                Created knowledge item dictionary
            """
            try:
                # Get user_id from context if available
                user_id = None
                if info and hasattr(info, 'context') and info.context:
                    user_id = info.context.get('user_id')
                
                content = input.get('content')
                if not content:
                    raise ValueError("Content is required")
                
                # Extract optional fields
                knowledge_type_str = input.get('type', 'TEXT')
                tags = set(input.get('tags', []))
                metadata = input.get('metadata', {})
                source = input.get('source', 'graphql_api')
                confidence = input.get('confidence', 1.0)
                
                # Access platform add_knowledge method
                if hasattr(platform, 'add_knowledge'):
                    item, _ = await platform.add_knowledge(
                        content=content,
                        knowledge_type=knowledge_type_str,
                        tags=tags,
                        metadata=metadata,
                        source=source,
                        confidence=confidence,
                        user_id=user_id
                    )
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'add_knowledge'):
                    item, _ = await platform.platform.add_knowledge(
                        content=content,
                        knowledge_type=knowledge_type_str,
                        tags=tags,
                        metadata=metadata,
                        source=source,
                        confidence=confidence,
                        user_id=user_id
                    )
                else:
                    raise Exception("Platform does not have add_knowledge method")
                
                return KnowledgeAPIFactory._knowledge_item_to_graphql(item)
                
            except ValueError as e:
                logger.warning(f"Validation error in createKnowledge: {e}")
                raise Exception(f"Validation error: {str(e)}")
            except Exception as e:
                logger.error(f"Error resolving createKnowledge: {e}")
                raise Exception(f"Failed to create knowledge item: {str(e)}")
        
        async def resolve_update_knowledge(
            parent,
            info,
            id: str,
            input: Dict[str, Any]
        ) -> Dict[str, Any]:
            """
            Resolver for updateKnowledge(id, input) mutation.
            
            Args:
                parent: Parent object (None for root mutation)
                info: GraphQL execution info
                id: Knowledge item ID to update
                input: Update knowledge input containing fields to update
                
            Returns:
                Updated knowledge item dictionary
            """
            try:
                # Get user_id from context if available
                user_id = None
                if info and hasattr(info, 'context') and info.context:
                    user_id = info.context.get('user_id')
                
                # Check if item exists first
                if hasattr(platform, 'knowledge_engine'):
                    existing = await platform.knowledge_engine.get_knowledge(id)
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'knowledge_engine'):
                    existing = await platform.platform.knowledge_engine.get_knowledge(id)
                else:
                    raise Exception("Platform does not have knowledge_engine attribute")
                
                if existing is None:
                    raise Exception(f"Knowledge item with id '{id}' not found")
                
                # Build update parameters
                new_content = input.get('content')
                confidence = input.get('confidence')
                
                # Handle metadata merge if provided
                if 'metadata' in input and existing.metadata:
                    metadata = {**existing.metadata, **input['metadata']}
                elif 'metadata' in input:
                    metadata = input['metadata']
                else:
                    metadata = None
                
                # Handle tags merge if provided
                if 'tags' in input:
                    tags = set(input['tags'])
                else:
                    tags = None
                
                # Access platform update_knowledge method
                if hasattr(platform, 'update_knowledge'):
                    updated_item = await platform.update_knowledge(
                        item_id=id,
                        new_content=new_content if new_content else existing.content,
                        user_id=user_id,
                        confidence=confidence
                    )
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'update_knowledge'):
                    updated_item = await platform.platform.update_knowledge(
                        item_id=id,
                        new_content=new_content if new_content else existing.content,
                        user_id=user_id,
                        confidence=confidence
                    )
                else:
                    raise Exception("Platform does not have update_knowledge method")
                
                if updated_item is None:
                    raise Exception(f"Failed to update knowledge item with id '{id}'")
                
                # Update additional fields if the platform supports it
                if tags and hasattr(updated_item, 'tags'):
                    updated_item.tags = tags
                if metadata and hasattr(updated_item, 'metadata'):
                    updated_item.metadata = metadata
                
                return KnowledgeAPIFactory._knowledge_item_to_graphql(updated_item)
                
            except Exception as e:
                logger.error(f"Error resolving updateKnowledge({id}): {e}")
                raise Exception(f"Failed to update knowledge item: {str(e)}")
        
        async def resolve_delete_knowledge(
            parent,
            info,
            id: str
        ) -> Dict[str, Any]:
            """
            Resolver for deleteKnowledge(id) mutation.
            
            Args:
                parent: Parent object (None for root mutation)
                info: GraphQL execution info
                id: Knowledge item ID to delete
                
            Returns:
                Delete status dictionary
            """
            try:
                # Get user_id from context if available
                user_id = None
                if info and hasattr(info, 'context') and info.context:
                    user_id = info.context.get('user_id')
                
                # Access platform delete_knowledge method
                if hasattr(platform, 'delete_knowledge'):
                    success = await platform.delete_knowledge(item_id=id, user_id=user_id)
                elif hasattr(platform, 'platform') and hasattr(platform.platform, 'delete_knowledge'):
                    success = await platform.platform.delete_knowledge(item_id=id, user_id=user_id)
                else:
                    raise Exception("Platform does not have delete_knowledge method")
                
                if success:
                    return {
                        "success": True,
                        "id": id,
                        "message": "Knowledge item deleted successfully"
                    }
                else:
                    return {
                        "success": False,
                        "id": id,
                        "message": "Knowledge item not found or could not be deleted"
                    }
                
            except Exception as e:
                logger.error(f"Error resolving deleteKnowledge({id}): {e}")
                return {
                    "success": False,
                    "id": id,
                    "message": f"Failed to delete knowledge item: {str(e)}"
                }
        
        # ==================== Type Resolvers ====================
        
        async def resolve_knowledge_item_type(item: Dict[str, Any], info) -> str:
            """Resolver for KnowledgeItem.type field."""
            return item.get('type', 'TEXT')
        
        async def resolve_knowledge_item_tags(item: Dict[str, Any], info) -> List[str]:
            """Resolver for KnowledgeItem.tags field."""
            tags = item.get('tags', [])
            return list(tags) if tags else []
        
        async def resolve_search_result_item(result: Dict[str, Any], info) -> Dict[str, Any]:
            """Resolver for SearchResult.item field."""
            return result.get('item', result) if isinstance(result, dict) else result
        
        async def resolve_search_result_score(result: Dict[str, Any], info) -> float:
            """Resolver for SearchResult.score field."""
            return result.get('score', result.get('relevance_score', 0.0))
        
        # Register all resolvers
        schema.register_resolver("knowledgeItem", resolve_knowledge_item)
        schema.register_resolver("search", resolve_search)
        schema.register_resolver("createKnowledge", resolve_create_knowledge)
        schema.register_resolver("updateKnowledge", resolve_update_knowledge)
        schema.register_resolver("deleteKnowledge", resolve_delete_knowledge)
    
    @staticmethod
    def _knowledge_item_to_graphql(item) -> Dict[str, Any]:
        """
        Convert a KnowledgeItem to GraphQL response format.
        
        Args:
            item: KnowledgeItem object or dictionary
            
        Returns:
            Dictionary formatted for GraphQL response
        """
        # Handle dataclass objects with to_dict method
        if hasattr(item, 'to_dict'):
            data = item.to_dict()
        elif isinstance(item, dict):
            data = item
        else:
            # Extract attributes directly
            data = {
                "id": getattr(item, 'id', ''),
                "content": getattr(item, 'content', ''),
                "knowledge_type": getattr(item, 'knowledge_type', 'TEXT'),
                "tags": getattr(item, 'tags', []),
                "metadata": getattr(item, 'metadata', {}),
                "created_at": getattr(item, 'created_at', None),
                "updated_at": getattr(item, 'updated_at', None),
                "source": getattr(item, 'source', 'unknown'),
                "confidence": getattr(item, 'confidence', 1.0)
            }
        
        # Map to GraphQL schema field names
        created_at = data.get('created_at')
        updated_at = data.get('updated_at')
        
        # Handle datetime serialization
        if hasattr(created_at, 'isoformat'):
            created_at = created_at.isoformat()
        if hasattr(updated_at, 'isoformat'):
            updated_at = updated_at.isoformat()
        
        # Handle knowledge_type (could be enum or string)
        knowledge_type = data.get('knowledge_type', 'TEXT')
        if hasattr(knowledge_type, 'value'):
            knowledge_type = knowledge_type.value
        
        # Handle tags (could be set or list)
        tags = data.get('tags', [])
        if isinstance(tags, set):
            tags = list(tags)
        
        return {
            "id": data.get('id', ''),
            "content": str(data.get('content', '')),
            "type": str(knowledge_type),
            "createdAt": created_at or '',
            "updatedAt": updated_at,
            "tags": tags or [],
            "metadata": data.get('metadata', {}) or {}
        }
    
    @staticmethod
    def _search_results_to_graphql(results) -> List[Dict[str, Any]]:
        """
        Convert search results to GraphQL response format.
        
        Args:
            results: List of SearchResult objects or dictionaries
            
        Returns:
            List of search result dictionaries formatted for GraphQL
        """
        if not results:
            return []
        
        graphql_results = []
        for result in results:
            # Handle SearchResult objects
            if hasattr(result, 'to_dict'):
                result_dict = result.to_dict()
                item = result_dict.get('item')
                score = result_dict.get('relevance_score', 0.0)
                highlights = result_dict.get('match_details', {}).get('highlights', [])
            elif isinstance(result, dict):
                item = result.get('item', result)
                score = result.get('score', result.get('relevance_score', 0.0))
                highlights = result.get('highlights', [])
            else:
                # Handle SearchResult dataclass directly
                item = getattr(result, 'item', result)
                score = getattr(result, 'relevance_score', getattr(result, 'score', 0.0))
                highlights = []
            
            graphql_results.append({
                "item": KnowledgeAPIFactory._knowledge_item_to_graphql(item) if item else None,
                "score": float(score),
                "highlights": highlights if highlights else []
            })
        
        return graphql_results


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
