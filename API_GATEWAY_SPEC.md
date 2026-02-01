# OpenEvolve-Knowledge Engine API Gateway Specification

## Document Information
- **Version**: 1.0
- **Date**: February 1, 2026
- **Status**: Draft
- **Authors**: OpenEvolve Team

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [API Gateway Components](#api-gateway-components)
4. [Routing and Load Balancing](#routing-and-load-balancing)
5. [Authentication and Authorization](#authentication-and-authorization)
6. [Rate Limiting and Throttling](#rate-limiting-and-throttling)
7. [Security](#security)
8. [Monitoring and Observability](#monitoring-and-observability)
9. [Performance](#performance)
10. [Configuration](#configuration)

## Overview

### Purpose
This document specifies the API Gateway architecture that serves as the central entry point for all OpenEvolve-Knowledge Engine interactions. The API Gateway provides routing, authentication, rate limiting, and security controls for all system communications.

### Goals
- Provide centralized API management and routing
- Implement security controls and authentication
- Enable rate limiting and traffic shaping
- Support microservices architecture patterns
- Ensure high availability and scalability
- Provide comprehensive monitoring and logging

### Non-Goals
- Specifying internal implementation of individual services
- Defining specific business logic of individual APIs
- Detailing UI components or user interfaces

## Architecture

### High-Level Architecture
```
┌─────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│   Clients       │    │  API Gateway         │    │  Backend        │
│                 │    │                     │    │  Services       │
│  • OpenEvolve   │◄──►│  • Authentication   │◄──►│  • Evolution    │
│  • Knowledge    │    │  • Authorization    │    │    Services     │
│    Engine       │    │  • Rate Limiting    │    │  • Knowledge    │
│  • External     │    │  • Routing          │    │    Services     │
│    Services     │    │  • Load Balancing   │    │  • Analytics    │
└─────────────────┘    │  • Monitoring       │    │    Services     │
                       │  • Caching          │    │  • Data         │
                       │  • Request/Response │    │    Services     │
                       │    Transformation   │    └─────────────────┘
                       └──────────────────────┘
                                    ▲
                       ┌──────────────────────┐
                       │  Gateway Management  │
                       │                     │
                       │  • Configuration    │
                       │  • Policy Management│
                       │  • Certificate      │
                       │    Management       │
                       │  • Health Monitoring│
                       └──────────────────────┘
```

### Component Roles
- **Authentication Service**: Validates API requests and tokens
- **Rate Limiter**: Controls request rates and prevents abuse
- **Router**: Directs requests to appropriate backend services
- **Transformer**: Transforms requests/responses as needed
- **Monitor**: Tracks API usage and performance metrics
- **Cache**: Caches responses to improve performance

## API Gateway Components

### 1. Authentication Service
```python
class AuthenticationService:
    def __init__(self, config):
        self.token_validator = TokenValidator(config.token_config)
        self.api_key_manager = APIKeyManager(config.api_key_config)
        self.oauth_provider = OAuthProvider(config.oauth_config)
        self.rate_limiter = RateLimiter(config.rate_limit_config)
    
    async def authenticate_request(self, request):
        # Check for API key
        api_key = self.extract_api_key(request)
        if api_key:
            return await self.validate_api_key(api_key)
        
        # Check for JWT token
        token = self.extract_jwt_token(request)
        if token:
            return await self.validate_jwt_token(token)
        
        # Check for OAuth token
        oauth_token = self.extract_oauth_token(request)
        if oauth_token:
            return await self.validate_oauth_token(oauth_token)
        
        # No authentication provided
        return {
            "authenticated": False,
            "error": "Authentication required",
            "status_code": 401
        }
    
    def extract_api_key(self, request):
        # Extract API key from header or query parameter
        api_key = request.headers.get("X-API-Key") or request.args.get("api_key")
        return api_key
    
    def extract_jwt_token(self, request):
        # Extract JWT token from Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header[7:]  # Remove "Bearer " prefix
        return None
    
    async def validate_api_key(self, api_key):
        validation_result = await self.api_key_manager.validate_api_key(api_key)
        if not validation_result.valid:
            return {
                "authenticated": False,
                "error": "Invalid API key",
                "status_code": 401
            }
        
        # Check rate limits
        if not await self.rate_limiter.allow_request(
            validation_result.owner_id, validation_result.permissions
        ):
            return {
                "authenticated": False,
                "error": "Rate limit exceeded",
                "status_code": 429
            }
        
        return {
            "authenticated": True,
            "user_id": validation_result.owner_id,
            "permissions": validation_result.permissions,
            "tenant_id": validation_result.tenant_id
        }
```

### 2. Rate Limiting Service
```python
class RateLimiter:
    def __init__(self, config):
        self.redis_client = RedisClient(config.redis_config)
        self.limit_configs = config.limit_configs
        self.default_limits = config.default_limits
    
    async def allow_request(self, user_id, permissions):
        # Get rate limits for user
        limits = await self.get_rate_limits(user_id, permissions)
        
        # Check each limit
        for limit_type, limit_config in limits.items():
            if not await self.check_limit(user_id, limit_type, limit_config):
                return False
        
        # Record request
        await self.record_request(user_id, limits)
        
        return True
    
    async def check_limit(self, user_id, limit_type, limit_config):
        # Get current count
        current_count = await self.get_current_count(user_id, limit_type)
        
        # Check if within limit
        if current_count >= limit_config["limit"]:
            return False
        
        return True
    
    async def get_rate_limits(self, user_id, permissions):
        # Determine limits based on user role and permissions
        limits = self.default_limits.copy()
        
        # Apply role-based limits
        user_role = await self.get_user_role(user_id)
        if user_role in self.limit_configs:
            limits.update(self.limit_configs[user_role])
        
        # Apply permission-based limits
        for perm in permissions:
            if perm in self.limit_configs:
                limits.update(self.limit_configs[perm])
        
        return limits
    
    async def get_current_count(self, user_id, limit_type):
        # Get current request count from Redis
        key = f"rate_limit:{user_id}:{limit_type}"
        return await self.redis_client.get(key) or 0
    
    async def record_request(self, user_id, limits):
        # Record request for each limit type
        for limit_type, limit_config in limits.items():
            key = f"rate_limit:{user_id}:{limit_type}"
            await self.redis_client.incr(key)
            
            # Set expiration
            await self.redis_client.expire(key, limit_config["window"])
```

### 3. Request Router
```python
class RequestRouter:
    def __init__(self, config):
        self.service_registry = ServiceRegistry(config.registry_config)
        self.load_balancer = LoadBalancer(config.load_balancer_config)
        self.route_table = self.build_route_table(config.routes)
    
    async def route_request(self, request):
        # Determine target service based on path
        target_service = await self.get_target_service(request.path)
        
        if not target_service:
            return {
                "error": "Service not found",
                "status_code": 404
            }
        
        # Get available instances
        instances = await self.service_registry.get_instances(target_service)
        
        if not instances:
            return {
                "error": "No available instances",
                "status_code": 503
            }
        
        # Select instance using load balancing algorithm
        selected_instance = await self.load_balancer.select_instance(
            instances, request
        )
        
        # Build target URL
        target_url = f"{selected_instance.protocol}://{selected_instance.host}:{selected_instance.port}{request.path}"
        
        return {
            "target_url": target_url,
            "selected_instance": selected_instance,
            "headers": self.prepare_headers(request.headers, selected_instance)
        }
    
    async def get_target_service(self, path):
        # Match path to service using route table
        for route_pattern, service_name in self.route_table.items():
            if self.match_path(path, route_pattern):
                return service_name
        
        return None
    
    def match_path(self, path, pattern):
        # Simple path matching (could be enhanced with regex)
        if pattern.endswith("/*"):
            # Prefix match
            prefix = pattern[:-2]  # Remove /*
            return path.startswith(prefix)
        else:
            # Exact match
            return path == pattern
    
    def prepare_headers(self, original_headers, selected_instance):
        # Prepare headers for forwarding
        headers = original_headers.copy()
        
        # Add gateway-specific headers
        headers["X-Forwarded-For"] = original_headers.get("X-Real-IP", "")
        headers["X-Forwarded-Proto"] = original_headers.get("X-Forwarded-Proto", "https")
        headers["X-Forwarded-Host"] = selected_instance.host
        headers["X-Original-Path"] = original_headers.get("X-Original-Path", "")
        
        # Remove hop-by-hop headers
        hop_by_hop_headers = [
            "connection", "keep-alive", "proxy-authenticate", 
            "proxy-authorization", "te", "trailers", "transfer-encoding"
        ]
        
        for header in hop_by_hop_headers:
            headers.pop(header, None)
        
        return headers
```

### 4. Response Transformer
```python
class ResponseTransformer:
    def __init__(self, config):
        self.transform_rules = config.transform_rules
        self.compression_service = CompressionService(config.compression_config)
        self.cors_handler = CORSHandler(config.cors_config)
    
    async def transform_response(self, response, request):
        # Apply transformation rules based on request
        transformed_response = await self.apply_transformations(
            response, request
        )
        
        # Apply compression if needed
        compressed_response = await self.apply_compression(
            transformed_response, request
        )
        
        # Add CORS headers
        cors_response = await self.add_cors_headers(
            compressed_response, request
        )
        
        # Add security headers
        secured_response = await self.add_security_headers(
            cors_response, request
        )
        
        return secured_response
    
    async def apply_transformations(self, response, request):
        # Get transformation rules for this endpoint
        transform_rules = await self.get_transform_rules(request.path)
        
        transformed_data = response.data
        
        for rule in transform_rules:
            if rule.condition and not self.evaluate_condition(rule.condition, request):
                continue
            
            if rule.operation == "add_field":
                transformed_data = self.add_field(transformed_data, rule.field, rule.value)
            elif rule.operation == "remove_field":
                transformed_data = self.remove_field(transformed_data, rule.field)
            elif rule.operation == "rename_field":
                transformed_data = self.rename_field(transformed_data, rule.old_name, rule.new_name)
            elif rule.operation == "filter":
                transformed_data = self.filter_data(transformed_data, rule.filter_expr)
            elif rule.operation == "format":
                transformed_data = self.format_data(transformed_data, rule.format_type)
        
        return {
            "data": transformed_data,
            "status_code": response.status_code,
            "headers": response.headers
        }
    
    async def get_transform_rules(self, path):
        # Get transformation rules for the given path
        rules = []
        
        for pattern, rule_list in self.transform_rules.items():
            if self.match_path(path, pattern):
                rules.extend(rule_list)
        
        return rules
    
    def evaluate_condition(self, condition, request):
        # Evaluate condition expression
        # This could be enhanced with a proper expression evaluator
        if condition.get("field") == "method":
            return request.method == condition.get("value")
        elif condition.get("field") == "user_role":
            return request.user_role in condition.get("values", [])
        
        return True
```

## Routing and Load Balancing

### 1. Service Discovery
```python
class ServiceRegistry:
    def __init__(self, config):
        self.consul_client = ConsulClient(config.consul_config)
        self.etcd_client = EtcdClient(config.etcd_config)
        self.service_cache = TTLCache(config.cache_config)
    
    async def register_service(self, service_info):
        # Register service in discovery system
        registration = {
            "ID": service_info.id,
            "Name": service_info.name,
            "Address": service_info.host,
            "Port": service_info.port,
            "Tags": service_info.tags,
            "Meta": service_info.metadata,
            "Check": {
                "HTTP": f"http://{service_info.host}:{service_info.port}/health",
                "Interval": "10s",
                "Timeout": "1s"
            }
        }
        
        await self.consul_client.register_service(registration)
    
    async def get_instances(self, service_name):
        # Get healthy instances from service discovery
        instances = await self.consul_client.get_healthy_instances(service_name)
        
        # Filter and validate instances
        valid_instances = []
        for instance in instances:
            if await self.validate_instance(instance):
                valid_instances.append(instance)
        
        return valid_instances
    
    async def validate_instance(self, instance):
        # Validate instance health
        try:
            health_check_url = f"http://{instance.Address}:{instance.Port}/health"
            response = await httpx.get(health_check_url, timeout=2.0)
            return response.status_code == 200
        except:
            return False
```

### 2. Load Balancing Algorithms
```python
class LoadBalancer:
    def __init__(self, config):
        self.algorithm = config.algorithm
        self.health_checker = HealthChecker(config.health_config)
        self.metrics_collector = MetricsCollector(config.metrics_config)
    
    async def select_instance(self, instances, request):
        if not instances:
            return None
        
        if self.algorithm == "round_robin":
            return await self.round_robin_select(instances)
        elif self.algorithm == "least_connections":
            return await self.least_connections_select(instances)
        elif self.algorithm == "weighted_round_robin":
            return await self.weighted_round_robin_select(instances)
        elif self.algorithm == "ip_hash":
            return await self.ip_hash_select(instances, request)
        elif self.algorithm == "random":
            return await self.random_select(instances)
        else:
            # Default to round robin
            return await self.round_robin_select(instances)
    
    async def round_robin_select(self, instances):
        # Simple round-robin selection
        if not hasattr(self, '_rr_counter'):
            self._rr_counter = 0
        
        selected = instances[self._rr_counter % len(instances)]
        self._rr_counter += 1
        
        return selected
    
    async def least_connections_select(self, instances):
        # Select instance with least active connections
        least_loaded = min(instances, key=lambda inst: inst.active_connections)
        return least_loaded
    
    async def weighted_round_robin_select(self, instances):
        # Select based on weights
        total_weight = sum(inst.weight for inst in instances)
        random_weight = random.randint(0, total_weight)
        
        current_weight = 0
        for instance in instances:
            current_weight += instance.weight
            if random_weight <= current_weight:
                return instance
        
        # Fallback to last instance
        return instances[-1]
    
    async def ip_hash_select(self, instances, request):
        # Select based on client IP hash
        client_ip = request.headers.get("X-Forwarded-For", request.client.host)
        ip_hash = hash(client_ip) % len(instances)
        return instances[ip_hash]
```

### 3. Circuit Breaker Pattern
```python
class CircuitBreaker:
    def __init__(self, config):
        self.failure_threshold = config.failure_threshold
        self.reset_timeout = config.reset_timeout
        self.success_threshold = config.success_threshold
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
    
    async def call(self, func, *args, **kwargs):
        if self.state == "OPEN":
            if self.is_reset_timeout_expired():
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self.on_success()
            elif self.state == "CLOSED":
                self.reset_counters()
            
            return result
            
        except Exception as e:
            if self.state == "HALF_OPEN":
                self.on_failure()
                raise CircuitBreakerOpenError("Circuit breaker opened after failure in half-open state")
            else:
                self.on_failure()
                raise e
    
    def on_failure(self):
        self.failure_count += 1
        self.last_failure_time = datetime.utcnow()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
    
    def on_success(self):
        self.success_count += 1
        
        if self.success_count >= self.success_threshold:
            self.state = "CLOSED"
            self.reset_counters()
    
    def is_reset_timeout_expired(self):
        if not self.last_failure_time:
            return True
        
        elapsed = datetime.utcnow() - self.last_failure_time
        return elapsed.total_seconds() > self.reset_timeout
    
    def reset_counters(self):
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
```

## Authentication and Authorization

### 1. JWT Token Validation
```python
class JWTValidator:
    def __init__(self, config):
        self.jwks_client = PyJWKClient(config.jwks_url)
        self.issuer = config.issuer
        self.audience = config.audience
        self.algorithms = config.algorithms
    
    async def validate_token(self, token):
        try:
            # Get signing key
            signing_key = await self.jwks_client.get_signing_key_from_jwt(token)
            
            # Decode token
            decoded_token = jwt.decode(
                token,
                signing_key.key,
                algorithms=self.algorithms,
                audience=self.audience,
                issuer=self.issuer,
                options={"verify_exp": True, "verify_iat": True}
            )
            
            # Check if token is revoked
            if await self.is_token_revoked(token):
                raise TokenRevokedError("Token has been revoked")
            
            return {
                "valid": True,
                "claims": decoded_token,
                "user_id": decoded_token.get("sub"),
                "tenant_id": decoded_token.get("tenant_id"),
                "scopes": decoded_token.get("scope", "").split()
            }
            
        except jwt.ExpiredSignatureError:
            return {
                "valid": False,
                "error": "Token has expired",
                "error_code": "TOKEN_EXPIRED"
            }
        except jwt.InvalidTokenError as e:
            return {
                "valid": False,
                "error": f"Invalid token: {str(e)}",
                "error_code": "INVALID_TOKEN"
            }
    
    async def is_token_revoked(self, token):
        # Check if token exists in revoked tokens set
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        return await self.redis_client.sismember("revoked_tokens", token_hash)
```

### 2. OAuth 2.0 Integration
```python
class OAuthProvider:
    def __init__(self, config):
        self.auth_server = config.auth_server
        self.client_id = config.client_id
        self.client_secret = config.client_secret
        self.scopes = config.scopes
    
    async def validate_oauth_token(self, token):
        # Validate token with OAuth provider
        validation_url = f"{self.auth_server}/introspect"
        
        data = {
            "token": token,
            "client_id": self.client_id,
            "client_secret": self.client_secret
        }
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(validation_url, data=data)
                
                if response.status_code == 200:
                    token_info = response.json()
                    
                    if token_info.get("active", False):
                        return {
                            "valid": True,
                            "user_id": token_info.get("sub"),
                            "client_id": token_info.get("client_id"),
                            "scopes": token_info.get("scope", "").split(),
                            "expires_at": token_info.get("exp"),
                            "token_type": token_info.get("token_type")
                        }
                    else:
                        return {
                            "valid": False,
                            "error": "Token is not active",
                            "error_code": "TOKEN_INACTIVE"
                        }
                else:
                    return {
                        "valid": False,
                        "error": "Token validation failed",
                        "error_code": "VALIDATION_FAILED"
                    }
                    
        except Exception as e:
            return {
                "valid": False,
                "error": f"Token validation error: {str(e)}",
                "error_code": "VALIDATION_ERROR"
            }
```

## Rate Limiting and Throttling

### 1. Advanced Rate Limiting
```python
class AdvancedRateLimiter:
    def __init__(self, config):
        self.redis_client = RedisClient(config.redis_config)
        self.sliding_window = SlidingWindowCounter(config.window_config)
        self.token_bucket = TokenBucketLimiter(config.bucket_config)
        self.leaky_bucket = LeakyBucketLimiter(config.leaky_config)
    
    async def check_rate_limit(self, user_id, resource, action):
        # Apply multiple rate limiting strategies
        sliding_window_result = await self.sliding_window.check_limit(
            user_id, resource, action
        )
        
        token_bucket_result = await self.token_bucket.check_limit(
            user_id, resource, action
        )
        
        leaky_bucket_result = await self.leaky_bucket.check_limit(
            user_id, resource, action
        )
        
        # All limits must pass
        if (sliding_window_result.allowed and 
            token_bucket_result.allowed and 
            leaky_bucket_result.allowed):
            # Record successful request
            await self.record_request(user_id, resource, action)
            return {
                "allowed": True,
                "retry_after": None,
                "limits": {
                    "sliding_window": sliding_window_result,
                    "token_bucket": token_bucket_result,
                    "leaky_bucket": leaky_bucket_result
                }
            }
        else:
            # Find the most restrictive limit
            blocked_by = []
            if not sliding_window_result.allowed:
                blocked_by.append("sliding_window")
            if not token_bucket_result.allowed:
                blocked_by.append("token_bucket")
            if not leaky_bucket_result.allowed:
                blocked_by.append("leaky_bucket")
            
            # Calculate retry after (minimum of all limits)
            retry_after = min(
                sliding_window_result.retry_after or float('inf'),
                token_bucket_result.retry_after or float('inf'),
                leaky_bucket_result.retry_after or float('inf')
            )
            
            return {
                "allowed": False,
                "blocked_by": blocked_by,
                "retry_after": retry_after if retry_after != float('inf') else None,
                "limits": {
                    "sliding_window": sliding_window_result,
                    "token_bucket": token_bucket_result,
                    "leaky_bucket": leaky_bucket_result
                }
            }
    
    async def record_request(self, user_id, resource, action):
        # Record request across all rate limiting systems
        await self.sliding_window.record_request(user_id, resource, action)
        await self.token_bucket.record_request(user_id, resource, action)
        await self.leaky_bucket.record_request(user_id, resource, action)
```

### 2. Sliding Window Counter
```python
class SlidingWindowCounter:
    def __init__(self, config):
        self.redis_client = RedisClient(config.redis_config)
        self.window_size = config.window_size  # in seconds
        self.max_requests = config.max_requests
    
    async def check_limit(self, user_id, resource, action):
        current_time = int(time.time())
        window_start = current_time - self.window_size
        
        # Use Redis sorted set to track requests in window
        key = f"rate_limit:{user_id}:{resource}:{action}"
        
        # Remove old entries outside the window
        await self.redis_client.zremrangebyscore(key, "-inf", window_start)
        
        # Count requests in current window
        current_count = await self.redis_client.zcard(key)
        
        # Check if within limit
        allowed = current_count < self.max_requests
        
        # Calculate retry after
        retry_after = None
        if not allowed:
            oldest_request = await self.redis_client.zrange(key, 0, 0, withscores=True)
            if oldest_request:
                oldest_time = oldest_request[0][1]
                retry_after = int(oldest_time + self.window_size - current_time)
        
        return {
            "allowed": allowed,
            "current_count": current_count,
            "max_requests": self.max_requests,
            "retry_after": retry_after
        }
    
    async def record_request(self, user_id, resource, action):
        current_time = int(time.time())
        key = f"rate_limit:{user_id}:{resource}:{action}"
        
        # Add current request to sorted set
        await self.redis_client.zadd(key, {current_time: current_time})
        
        # Set expiration to clean up automatically
        await self.redis_client.expire(key, self.window_size + 60)  # Extra 60 seconds
```

## Security

### 1. Security Headers
```python
class SecurityHeaders:
    def __init__(self, config):
        self.hsts_enabled = config.hsts_enabled
        self.hsts_max_age = config.hsts_max_age
        self.csp_enabled = config.csp_enabled
        self.csp_policy = config.csp_policy
        self.xss_protection = config.xss_protection
        self.frame_options = config.frame_options
    
    def add_security_headers(self, response):
        # HTTP Strict Transport Security
        if self.hsts_enabled:
            response.headers["Strict-Transport-Security"] = f"max-age={self.hsts_max_age}; includeSubDomains; preload"
        
        # Content Security Policy
        if self.csp_enabled:
            response.headers["Content-Security-Policy"] = self.csp_policy
        
        # XSS Protection
        if self.xss_protection:
            response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Frame Options
        if self.frame_options:
            response.headers["X-Frame-Options"] = self.frame_options
        
        # Other security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response
```

### 2. Request Validation
```python
class RequestValidator:
    def __init__(self, config):
        self.schema_validator = SchemaValidator(config.schema_config)
        self.injection_detector = InjectionDetector(config.injection_config)
        self.size_limiter = SizeLimiter(config.size_config)
        self.content_filter = ContentFilter(config.filter_config)
    
    async def validate_request(self, request):
        # Validate request size
        size_validation = await self.size_limiter.validate(request)
        if not size_validation.valid:
            return {
                "valid": False,
                "error": size_validation.error,
                "error_code": "REQUEST_TOO_LARGE"
            }
        
        # Validate request schema
        schema_validation = await self.schema_validator.validate(request)
        if not schema_validation.valid:
            return {
                "valid": False,
                "error": "Invalid request schema",
                "error_code": "INVALID_SCHEMA",
                "details": schema_validation.errors
            }
        
        # Check for injection attacks
        injection_check = await self.injection_detector.scan(request)
        if injection_check.malicious:
            return {
                "valid": False,
                "error": "Potential injection attack detected",
                "error_code": "INJECTION_DETECTED",
                "details": injection_check.threats
            }
        
        # Filter content
        content_filter_result = await self.content_filter.filter(request)
        if not content_filter_result.allowed:
            return {
                "valid": False,
                "error": "Content filtered",
                "error_code": "CONTENT_FILTERED",
                "details": content_filter_result.reasons
            }
        
        return {
            "valid": True,
            "sanitized_request": content_filter_result.sanitized_request
        }
```

### 3. WAF Integration
```python
class WebApplicationFirewall:
    def __init__(self, config):
        self.rule_engine = RuleEngine(config.rule_config)
        self.signature_db = SignatureDatabase(config.signature_config)
        self.machine_learning = MLThreatDetector(config.ml_config)
        self.rate_limiter = AdvancedRateLimiter(config.rate_limit_config)
    
    async def scan_request(self, request):
        # Apply signature-based detection
        signature_match = await self.signature_db.scan_request(request)
        if signature_match.matched:
            return {
                "blocked": True,
                "reason": "Signature match",
                "threat_type": signature_match.threat_type,
                "confidence": signature_match.confidence
            }
        
        # Apply rule-based detection
        rule_match = await self.rule_engine.evaluate_request(request)
        if rule_match.matched:
            return {
                "blocked": True,
                "reason": "Rule match",
                "rule_id": rule_match.rule_id,
                "confidence": rule_match.confidence
            }
        
        # Apply ML-based detection
        ml_result = await self.machine_learning.analyze_request(request)
        if ml_result.threat_detected:
            return {
                "blocked": ml_result.block_request,
                "reason": "ML threat detection",
                "threat_type": ml_result.threat_type,
                "confidence": ml_result.confidence
            }
        
        # Apply behavioral analysis
        behavior_result = await self.analyze_behavior(request)
        if behavior_result.anomalous:
            return {
                "blocked": behavior_result.block_request,
                "reason": "Behavioral anomaly",
                "anomaly_type": behavior_result.anomaly_type,
                "confidence": behavior_result.confidence
            }
        
        return {
            "blocked": False,
            "reason": "No threats detected"
        }
    
    async def analyze_behavior(self, request):
        # Analyze request behavior patterns
        user_id = request.headers.get("X-User-ID") or self.extract_user_id(request)
        
        # Get user behavior profile
        profile = await self.get_user_behavior_profile(user_id)
        
        # Compare current request to profile
        anomaly_score = self.calculate_anomaly_score(request, profile)
        
        # Determine if anomalous
        is_anomalous = anomaly_score > self.anomaly_threshold
        
        return {
            "anomalous": is_anomalous,
            "anomaly_score": anomaly_score,
            "anomaly_type": self.classify_anomaly_type(request, profile),
            "confidence": self.calculate_confidence(anomaly_score),
            "block_request": is_anomalous and anomaly_score > self.block_threshold
        }
```

## Monitoring and Observability

### 1. Metrics Collection
```python
class MetricsCollector:
    def __init__(self, config):
        self.prometheus_client = PrometheusClient(config.prometheus_config)
        self.statsd_client = StatsDClient(config.statsd_config)
        self.event_bus = EventBus(config.event_config)
    
    async def collect_request_metrics(self, request, response, duration_ms):
        # Request count
        await self.prometheus_client.increment_counter(
            "api_requests_total",
            labels={
                "method": request.method,
                "endpoint": self.get_endpoint(request.path),
                "status_code": str(response.status_code),
                "user_id": request.user_id if hasattr(request, 'user_id') else "anonymous"
            }
        )
        
        # Request duration
        await self.prometheus_client.observe_histogram(
            "api_request_duration_seconds",
            duration_ms / 1000.0,  # Convert to seconds
            labels={
                "method": request.method,
                "endpoint": self.get_endpoint(request.path),
                "status_code": str(response.status_code)
            }
        )
        
        # Error rate
        if 400 <= response.status_code < 600:
            await self.prometheus_client.increment_counter(
                "api_errors_total",
                labels={
                    "method": request.method,
                    "endpoint": self.get_endpoint(request.path),
                    "status_code": str(response.status_code),
                    "error_type": self.get_error_type(response.status_code)
                }
            )
        
        # Rate limiting metrics
        if hasattr(request, 'rate_limit_info'):
            await self.prometheus_client.set_gauge(
                "api_rate_limit_remaining",
                request.rate_limit_info.remaining,
                labels={
                    "user_id": request.user_id,
                    "limit_type": request.rate_limit_info.limit_type
                }
            )
        
        # Send event for real-time monitoring
        await self.event_bus.publish("api_request_completed", {
            "timestamp": datetime.utcnow().isoformat(),
            "method": request.method,
            "path": request.path,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "user_id": getattr(request, 'user_id', 'anonymous'),
            "ip_address": request.client.host,
            "user_agent": request.headers.get('User-Agent', '')
        })
    
    def get_endpoint(self, path):
        # Simplified endpoint extraction (could be enhanced with regex)
        # Remove query parameters and normalize
        endpoint = path.split('?')[0]
        
        # Replace dynamic segments with placeholders
        # e.g., /users/123 -> /users/{id}
        import re
        endpoint = re.sub(r'/\d+', '/{id}', endpoint)
        endpoint = re.sub(r'/[a-fA-F0-9-]+/', '/{uuid}/', endpoint)
        
        return endpoint
```

### 2. Distributed Tracing
```python
class DistributedTracer:
    def __init__(self, config):
        self.tracer = Tracer(config.tracer_config)
        self.span_processor = SpanProcessor(config.span_config)
        self.propagator = TraceContextPropagator()
    
    async def start_trace(self, request, operation_name):
        # Extract trace context from request headers
        trace_context = self.propagator.extract(request.headers)
        
        # Start new span
        span = self.tracer.start_span(
            operation_name,
            context=trace_context,
            kind=SpanKind.SERVER
        )
        
        # Add request attributes to span
        span.set_attribute("http.method", request.method)
        span.set_attribute("http.url", request.url)
        span.set_attribute("http.user_agent", request.headers.get("User-Agent"))
        span.set_attribute("http.client_ip", request.client.host)
        
        return span
    
    async def finish_trace(self, span, response, error=None):
        if error:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error))
            span.set_status(Status(StatusCode.ERROR, str(error)))
        else:
            span.set_attribute("http.status_code", response.status_code)
            if 400 <= response.status_code < 600:
                span.set_status(Status(StatusCode.ERROR))
            else:
                span.set_status(Status(StatusCode.OK))
        
        # End span
        span.end()
```

### 3. Health Monitoring
```python
class HealthMonitor:
    def __init__(self, config):
        self.service_registry = ServiceRegistry(config.registry_config)
        self.health_checker = HealthChecker(config.health_config)
        self.alert_manager = AlertManager(config.alert_config)
    
    async def perform_health_check(self):
        # Check gateway health
        gateway_health = await self.check_gateway_health()
        
        # Check upstream services
        service_healths = await self.check_upstream_services()
        
        # Aggregate results
        overall_health = self.aggregate_health(gateway_health, service_healths)
        
        # Send alerts if unhealthy
        if not overall_health.healthy:
            await self.alert_manager.send_alert(
                "gateway_unhealthy",
                "API Gateway is experiencing issues",
                severity="high",
                details=overall_health.details
            )
        
        return overall_health
    
    async def check_gateway_health(self):
        # Check internal components
        checks = {
            "authentication_service": await self.health_checker.check_component("auth"),
            "rate_limiter": await self.health_checker.check_component("rate_limiter"),
            "router": await self.health_checker.check_component("router"),
            "cache": await self.health_checker.check_component("cache"),
            "database": await self.health_checker.check_component("database")
        }
        
        # Overall health
        healthy = all(check.healthy for check in checks.values())
        
        return {
            "healthy": healthy,
            "component_health": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def check_upstream_services(self):
        # Get all registered services
        services = await self.service_registry.get_all_services()
        
        service_healths = {}
        for service in services:
            instances = await self.service_registry.get_instances(service.name)
            instance_healths = []
            
            for instance in instances:
                health = await self.health_checker.check_service_instance(instance)
                instance_healths.append(health)
            
            service_healths[service.name] = {
                "instances": instance_healths,
                "healthy_instances": sum(1 for h in instance_healths if h.healthy),
                "total_instances": len(instance_healths),
                "overall_healthy": all(h.healthy for h in instance_healths)
            }
        
        return service_healths
```

## Performance

### 1. Performance Metrics
- **Request Latency**: Time from request arrival to response
- **Throughput**: Requests processed per second
- **Error Rate**: Percentage of failed requests
- **Resource Utilization**: CPU, memory, and network usage
- **Cache Hit Rate**: Percentage of requests served from cache

### 2. Performance Targets
- **Request Latency**: <50ms for 95% of requests, <100ms for 99%
- **Throughput**: 10,000+ requests/second
- **Error Rate**: <0.1% for successful requests
- **Cache Hit Rate**: >90% for common requests
- **Availability**: 99.9% uptime

### 3. Caching Strategy
```python
class GatewayCache:
    def __init__(self, config):
        self.primary_cache = RedisCache(config.redis_config)
        self.secondary_cache = InMemoryCache(config.memory_config)
        self.cache_manager = CacheManager(config.manager_config)
    
    async def get_cached_response(self, request):
        # Generate cache key
        cache_key = await self.generate_cache_key(request)
        
        # Try primary cache first
        cached_response = await self.primary_cache.get(cache_key)
        if cached_response:
            return cached_response
        
        # Try secondary cache
        cached_response = await self.secondary_cache.get(cache_key)
        if cached_response:
            # Warm primary cache
            await self.primary_cache.set(cache_key, cached_response)
            return cached_response
        
        return None
    
    async def cache_response(self, request, response):
        # Generate cache key
        cache_key = await self.generate_cache_key(request)
        
        # Determine cacheability
        if await self.is_cacheable(request, response):
            # Get cache TTL
            ttl = await self.get_cache_ttl(request, response)
            
            # Store in both caches
            await self.primary_cache.set(cache_key, response, ttl)
            await self.secondary_cache.set(cache_key, response, ttl)
    
    async def generate_cache_key(self, request):
        # Generate unique cache key based on request
        key_parts = [
            request.method,
            request.path,
            str(sorted(request.query_params.items())),
            request.headers.get("Accept", ""),
            request.headers.get("Accept-Encoding", "")
        ]
        
        key_string = "|".join(key_parts)
        return hashlib.sha256(key_string.encode()).hexdigest()
    
    async def is_cacheable(self, request, response):
        # Check if request/response is cacheable
        if request.method != "GET":
            return False
        
        if response.status_code not in [200, 203, 300, 301, 410]:
            return False
        
        # Check cache-control headers
        cache_control = response.headers.get("Cache-Control", "")
        if "no-cache" in cache_control or "no-store" in cache_control:
            return False
        
        # Check if authenticated
        if hasattr(request, 'user_id') and request.user_id:
            # Don't cache authenticated responses by default
            return False
        
        return True
    
    async def get_cache_ttl(self, request, response):
        # Determine cache TTL based on response
        cache_control = response.headers.get("Cache-Control", "")
        
        # Look for max-age directive
        import re
        max_age_match = re.search(r'max-age=(\d+)', cache_control)
        if max_age_match:
            return int(max_age_match.group(1))
        
        # Default TTL based on status code
        if response.status_code == 200:
            return 300  # 5 minutes for successful responses
        elif response.status_code in [301, 302]:
            return 3600  # 1 hour for redirects
        else:
            return 60  # 1 minute for other responses
```

## Configuration

### 1. Gateway Configuration Schema
```yaml
api_gateway:
  server:
    host: "0.0.0.0"
    port: 8080
    workers: 4
    timeout: 30
    keep_alive: 60

  ssl:
    enabled: true
    cert_file: "/path/to/cert.pem"
    key_file: "/path/to/key.pem"
    redirect_http_to_https: true

  authentication:
    jwt:
      issuer: "https://auth.openevolve.org"
      audience: ["https://api.openevolve.org"]
      jwks_url: "https://auth.openevolve.org/.well-known/jwks.json"
      algorithms: ["RS256"]
    
    api_keys:
      enabled: true
      storage:
        type: "redis"
        redis_url: "redis://localhost:6379"
    
    oauth:
      enabled: true
      provider: "custom"
      auth_server: "https://auth.openevolve.org"
      client_id: "your-client-id"
      client_secret: "your-client-secret"

  rate_limiting:
    enabled: true
    strategy: "sliding_window"
    sliding_window:
      window_size: 60
      max_requests: 1000
    token_bucket:
      capacity: 100
      refill_rate: 10
    leaky_bucket:
      capacity: 100
      leak_rate: 5

  routing:
    load_balancing:
      algorithm: "weighted_round_robin"
      health_check_interval: 30
      failure_threshold: 3
      reset_timeout: 60

  security:
    hsts:
      enabled: true
      max_age: 31536000
    csp:
      enabled: true
      policy: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
    xss_protection: true
    frame_options: "DENY"

  caching:
    enabled: true
    primary:
      type: "redis"
      redis_url: "redis://localhost:6379"
      ttl: 300
    secondary:
      type: "memory"
      max_size: 1000
      ttl: 60

  monitoring:
    prometheus:
      enabled: true
      endpoint: "/metrics"
    tracing:
      enabled: true
      provider: "jaeger"
      endpoint: "http://jaeger:14268/api/traces"
    logging:
      level: "INFO"
      format: "json"
      output: "stdout"

  cors:
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
    allow_headers: ["*"]
    allow_credentials: false
    max_age: 3600

  compression:
    enabled: true
    algorithm: "gzip"
    min_size: 1024
    level: 6

  circuit_breaker:
    enabled: true
    failure_threshold: 5
    reset_timeout: 60
    success_threshold: 3
```

### 2. Route Configuration
```yaml
routes:
  - path: "/api/v1/evolution/**"
    service: "evolution-service"
    methods: ["GET", "POST", "PUT", "DELETE"]
    strip_prefix: true
    rewrite_path: "/{path}"
    authentication: true
    rate_limiting: true
    circuit_breaker: true
    timeout: 30
    retries: 3

  - path: "/api/v1/knowledge/**"
    service: "knowledge-service"
    methods: ["GET", "POST", "PUT", "DELETE"]
    strip_prefix: true
    rewrite_path: "/{path}"
    authentication: true
    rate_limiting: true
    circuit_breaker: true
    timeout: 45
    retries: 2

  - path: "/api/v1/analytics/**"
    service: "analytics-service"
    methods: ["GET", "POST"]
    strip_prefix: true
    rewrite_path: "/{path}"
    authentication: true
    rate_limiting: true
    circuit_breaker: false
    timeout: 60
    retries: 1

  - path: "/health"
    service: "gateway"
    methods: ["GET"]
    authentication: false
    rate_limiting: false
    circuit_breaker: false

  - path: "/metrics"
    service: "prometheus"
    methods: ["GET"]
    authentication: false
    rate_limiting: false
    circuit_breaker: false
```

## Appendix

### Glossary
- **API Gateway**: Centralized entry point for API requests
- **Service Discovery**: Mechanism for locating service instances
- **Load Balancing**: Distributing requests across multiple instances
- **Rate Limiting**: Controlling request frequency
- **Circuit Breaker**: Preventing cascading failures
- **Distributed Tracing**: Tracking requests across services

### References
- API Gateway Patterns
- OAuth 2.0 Security Best Practices
- Rate Limiting Strategies
- Circuit Breaker Pattern Implementation
- Distributed Tracing with OpenTelemetry

### Change Log
- **v1.0** - Initial specification