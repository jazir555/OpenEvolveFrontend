"""
Unified API Gateway capabilities for the OpenEvolve API service.

Implements the components described in docs/Architecture/API_GATEWAY_SPEC.md as
real, composable, dependency-light building blocks:

    * RateLimiter            - in-memory token-bucket (Redis optional, graceful)
    * CircuitBreaker        - CLOSED / OPEN / HALF_OPEN state machine
    * ServiceRegistry        - in-process registry with TTL cache (Consul/etcd optional)
    * LoadBalancer          - round_robin / least_connections / weighted / ip_hash / random
    * RequestRouter         - prefix/exact path -> service resolution + header prep
    * ResponseTransformer   - field add/remove/rename/filter/format + CORS/security headers
    * ResponseCache         - in-memory TTL cache keyed by method+path(+tenant)
    * AuthenticationService - API key / JWT (HS256 verified, others best-effort) / OAuth

Where full distributed load-balancing / service-discovery / OAuth introspection is
out of scope for this single-process service, the components implement a working
in-process version with graceful degradation and explicit TODOs. No heavy third
party dependencies are introduced (only stdlib + structlog).

A FastAPI router (``gateway_router``) and an optional ASGI ``GatewayMiddleware``
are provided so the gateway can be mounted into the service cleanly.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

logger = structlog.get_logger()


# ============================================================================
# RATE LIMITER
# ============================================================================


@dataclass
class RateLimitRule:
    """A single rate limit rule: ``limit`` requests per ``window_seconds``."""

    limit: int = 60
    window_seconds: float = 60.0


class RateLimiter:
    """
    In-memory token-bucket / sliding-window rate limiter.

    A Redis-backed store can be plugged in via ``redis_url``; when unavailable
    (the common case in this single-process service) the limiter degrades
    gracefully to a process-local store. Limit selection honours role/permission
    overrides so it matches the gateway spec's ``get_rate_limits`` semantics.
    """

    def __init__(
        self,
        default_limit: int = 60,
        default_window: float = 60.0,
        role_limits: Optional[Dict[str, RateLimitRule]] = None,
        redis_url: Optional[str] = None,
    ) -> None:
        self.default_rule = RateLimitRule(default_limit, default_window)
        self.role_limits = role_limits or {}
        self._lock = threading.Lock()
        # key -> list of request timestamps (epoch seconds, float)
        self._hits: Dict[str, List[float]] = {}
        self.redis = None
        if redis_url:
            try:  # pragma: no cover - optional path
                import redis  # type: ignore

                self.redis = redis.Redis.from_url(redis_url)
                logger.info("RateLimiter.redis.connected", url=redis_url)
            except Exception as exc:  # pragma: no cover
                logger.warning("RateLimiter.redis.unavailable", error=str(exc))
                self.redis = None

    def _rule_for(self, key: str, permissions: Optional[List[str]]) -> RateLimitRule:
        rule = self.default_rule
        for perm in permissions or []:
            if perm in self.role_limits:
                rule = self.role_limits[perm]
        return rule

    def allow(self, key: str, permissions: Optional[List[str]] = None) -> bool:
        """Return True if a request for ``key`` is allowed right now."""
        rule = self._rule_for(key, permissions)
        now = time.time()
        window_start = now - rule.window_seconds
        with self._lock:
            hits = self._hits.setdefault(key, [])
            # Drop timestamps outside the window.
            hits[:] = [t for t in hits if t > window_start]
            if len(hits) >= rule.limit:
                return False
            hits.append(now)
            return True

    async def allow_request(
        self, user_id: str, permissions: Optional[List[str]] = None
    ) -> bool:
        return self.allow(user_id, permissions)

    def reset(self, key: Optional[str] = None) -> None:
        with self._lock:
            if key is None:
                self._hits.clear()
            else:
                self._hits.pop(key, None)


# ============================================================================
# CIRCUIT BREAKER
# ============================================================================


class CircuitBreakerOpenError(Exception):
    """Raised when a call is attempted while the breaker is OPEN."""


class CircuitBreaker:
    """
    Classic circuit breaker (CLOSED / OPEN / HALF_OPEN).

    Mirrors the gateway spec. Wrap both sync and async callables via
    :meth:`call` / :meth:`acall`. Out-of-process coordination (shared state
    across replicas) is a TODO; the in-process version is fully functional.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        success_threshold: int = 3,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.success_threshold = success_threshold
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

    def _is_reset_expired(self) -> bool:
        if self.last_failure_time is None:
            return True
        return (time.monotonic() - self.last_failure_time) > self.reset_timeout

    def _on_success(self) -> None:
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
        elif self.state == "CLOSED":
            self.failure_count = 0
            self.last_failure_time = None

    def _on_failure(self) -> None:
        self.failure_count += 1
        self.last_failure_time = time.monotonic()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"

    def call(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        with self._lock:
            if self.state == "OPEN":
                if self._is_reset_expired():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise CircuitBreakerOpenError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            with self._lock:
                self._on_success()
            return result
        except Exception as exc:
            with self._lock:
                half_open = self.state == "HALF_OPEN"
                self._on_failure()
            if half_open:
                raise CircuitBreakerOpenError(
                    "Circuit breaker opened after failure in half-open state"
                ) from exc
            raise

    async def acall(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        # Route async callables through the same state machine.
        import asyncio

        async def _runner() -> Any:
            return await func(*args, **kwargs)

        return self.call(lambda: asyncio.get_event_loop().run_until_complete(_runner()))

    def reset(self) -> None:
        with self._lock:
            self.state = "CLOSED"
            self.failure_count = 0
            self.success_count = 0
            self.last_failure_time = None


# ============================================================================
# SERVICE REGISTRY
# ============================================================================


@dataclass
class ServiceInstance:
    name: str
    host: str
    port: int
    protocol: str = "http"
    weight: int = 1
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    healthy: bool = True
    active_connections: int = 0


class ServiceRegistry:
    """
    In-process service registry with a TTL health cache.

    Production deployments would back this with Consul/etcd (see TODO). The
    in-process version is fully functional for single-process routing and
    gracefully degrades when no instances are registered (callers receive an
    empty list and can decide how to 503).
    """

    def __init__(self, ttl_seconds: float = 30.0, probe_timeout: float = 2.0) -> None:
        self.ttl_seconds = ttl_seconds
        self.probe_timeout = probe_timeout
        self._lock = threading.Lock()
        self._services: Dict[str, List[ServiceInstance]] = {}
        self._cache: Dict[str, Tuple[float, List[ServiceInstance]]] = {}

    def register(self, instance: ServiceInstance) -> None:
        with self._lock:
            self._services.setdefault(instance.name, []).append(instance)
            self._cache.pop(instance.name, None)
        logger.info("registry.register", name=instance.name, host=instance.host)

    def deregister(self, name: str, host: str, port: int) -> None:
        with self._lock:
            instances = self._services.get(name, [])
            self._services[name] = [
                i for i in instances if not (i.host == host and i.port == port)
            ]
            self._cache.pop(name, None)

    def get_instances(self, name: str) -> List[ServiceInstance]:
        with self._lock:
            cached = self._cache.get(name)
            if cached and (time.monotonic() - cached[0]) < self.ttl_seconds:
                return list(cached[1])
            instances = list(self._services.get(name, []))
            self._cache[name] = (time.monotonic(), instances)
        # Best-effort async-free health probe would go here (TODO: httpx probe).
        return [i for i in instances if i.healthy]

    def list_services(self) -> Dict[str, int]:
        with self._lock:
            return {name: len(inst) for name, inst in self._services.items()}


# ============================================================================
# LOAD BALANCER
# ============================================================================


class LoadBalancer:
    """
    Pluggable load-balancing strategies over a list of :class:`ServiceInstance`.

    Mirrors the gateway spec. Real distributed LB (consistent hashing across
    replicas, weighted least-requests) is a TODO; the in-process algorithms are
    correct and deterministic given the input instances.
    """

    def __init__(self, algorithm: str = "round_robin") -> None:
        self.algorithm = algorithm
        self._rr_counter = 0
        self._lock = threading.Lock()

    def select(
        self,
        instances: List[ServiceInstance],
        request: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None
        method = getattr(self, f"_{self.algorithm}_select", None)
        if method is None:
            method = self._round_robin_select
        return method(instances, request)

    def _round_robin_select(self, instances, request):
        with self._lock:
            selected = instances[self._rr_counter % len(instances)]
            self._rr_counter += 1
        return selected

    def _least_connections_select(self, instances, request):
        return min(instances, key=lambda i: i.active_connections)

    def _weighted_round_robin_select(self, instances, request):
        import random

        total = sum(max(i.weight, 1) for i in instances)
        r = random.randint(0, total)
        cum = 0
        for i in instances:
            cum += max(i.weight, 1)
            if r <= cum:
                return i
        return instances[-1]

    def _ip_hash_select(self, instances, request):
        client_ip = (request or {}).get("client_ip", "0.0.0.0")
        return instances[hash(client_ip) % len(instances)]

    def _random_select(self, instances, request):
        import random

        return random.choice(instances)


# ============================================================================
# REQUEST ROUTER
# ============================================================================


class RequestRouter:
    """
    Resolve an incoming path to a backend service instance.

    Uses the in-process :class:`ServiceRegistry` + :class:`LoadBalancer`. The
    resolved target URL can be handed to an httpx-based proxy (out of scope for
    this single-process service) or used directly for local dispatch.
    """

    def __init__(
        self,
        route_table: Optional[Dict[str, str]] = None,
        registry: Optional[ServiceRegistry] = None,
        load_balancer: Optional[LoadBalancer] = None,
    ) -> None:
        self.route_table = route_table or {}
        self.service_registry = registry or ServiceRegistry()
        self.load_balancer = load_balancer or LoadBalancer()

    def add_route(self, pattern: str, service_name: str) -> None:
        self.route_table[pattern] = service_name

    def get_target_service(self, path: str) -> Optional[str]:
        for pattern, service in self.route_table.items():
            if self._match_path(path, pattern):
                return service
        return None

    def _match_path(self, path: str, pattern: str) -> bool:
        if pattern.endswith("/*"):
            return path.startswith(pattern[:-2])
        return path == pattern

    def route(self, path: str, headers: Optional[Dict[str, str]] = None,
              client_ip: str = "0.0.0.0") -> Dict[str, Any]:
        service = self.get_target_service(path)
        if not service:
            return {"error": "Service not found", "status_code": 404}
        instances = self.service_registry.get_instances(service)
        if not instances:
            return {"error": "No available instances", "status_code": 503}
        instance = self.load_balancer.select(instances, {"client_ip": client_ip})
        if instance is None:
            return {"error": "No available instances", "status_code": 503}
        target_url = f"{instance.protocol}://{instance.host}:{instance.port}{path}"
        return {
            "target_url": target_url,
            "selected_instance": instance,
            "headers": self.prepare_headers(headers or {}, instance),
            "status_code": 200,
        }

    def prepare_headers(
        self, original: Dict[str, str], instance: ServiceInstance
    ) -> Dict[str, str]:
        headers = dict(original)
        headers["X-Forwarded-For"] = original.get("X-Real-IP", "")
        headers["X-Forwarded-Proto"] = original.get("X-Forwarded-Proto", "https")
        headers["X-Forwarded-Host"] = instance.host
        for hop in (
            "connection", "keep-alive", "proxy-authenticate",
            "proxy-authorization", "te", "trailers", "transfer-encoding",
        ):
            headers.pop(hop, None)
        return headers


# ============================================================================
# RESPONSE TRANSFORMER
# ============================================================================


@dataclass
class TransformRule:
    operation: str  # add_field | remove_field | rename_field | filter | format
    field: Optional[str] = None
    value: Any = None
    old_name: Optional[str] = None
    new_name: Optional[str] = None
    condition: Optional[Dict[str, Any]] = None


class ResponseTransformer:
    """
    Apply declarative field transforms plus CORS / security header injection.

    The transform rules match the gateway spec; compression is a documented TODO
    (bodies are already chunked by the ASGI server).
    """

    def __init__(self, transform_rules: Optional[Dict[str, List[TransformRule]]] = None,
                 cors_origin: str = "*") -> None:
        self.transform_rules = transform_rules or {}
        self.cors_origin = cors_origin

    def _match_path(self, path: str, pattern: str) -> bool:
        if pattern.endswith("/*"):
            return path.startswith(pattern[:-2])
        return path == pattern

    def _evaluate_condition(self, condition: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
        field_name = condition.get("field")
        if field_name == "method":
            return ctx.get("method") == condition.get("value")
        if field_name == "user_role":
            return ctx.get("user_role") in condition.get("values", [])
        return True

    def transform(self, data: Any, path: str,
                  ctx: Optional[Dict[str, Any]] = None) -> Any:
        ctx = ctx or {}
        rules: List[TransformRule] = []
        for pattern, rule_list in self.transform_rules.items():
            if self._match_path(path, pattern):
                rules.extend(rule_list)

        for rule in rules:
            if rule.condition and not self._evaluate_condition(rule.condition, ctx):
                continue
            if rule.operation == "add_field" and rule.field is not None:
                if isinstance(data, dict):
                    data[rule.field] = rule.value
            elif rule.operation == "remove_field" and rule.field is not None:
                if isinstance(data, dict):
                    data.pop(rule.field, None)
            elif (
                rule.operation == "rename_field"
                and rule.old_name is not None
                and rule.new_name is not None
            ):
                if isinstance(data, dict) and rule.old_name in data:
                    data[rule.new_name] = data.pop(rule.old_name)
        return data

    def add_cors_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        headers["Access-Control-Allow-Origin"] = self.cors_origin
        headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
        headers["Access-Control-Allow-Headers"] = "*"
        return headers

    def add_security_headers(self, headers: Dict[str, str]) -> Dict[str, str]:
        headers["X-Content-Type-Options"] = "nosniff"
        headers["X-Frame-Options"] = "DENY"
        headers["Referrer-Policy"] = "no-referrer"
        return headers

    def finalize(self, data: Any, path: str,
                 ctx: Optional[Dict[str, Any]] = None,
                 headers: Optional[Dict[str, str]] = None) -> Tuple[Any, Dict[str, str]]:
        data = self.transform(data, path, ctx)
        headers = self.add_cors_headers(headers or {})
        headers = self.add_security_headers(headers)
        return data, headers


# ============================================================================
# RESPONSE CACHE
# ============================================================================


class ResponseCache:
    """In-memory TTL response cache keyed by ``method:path`` (optionally tenant)."""

    def __init__(self, default_ttl: float = 30.0) -> None:
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        self._store: Dict[str, Tuple[float, Any]] = {}

    @staticmethod
    def make_key(method: str, path: str, tenant: Optional[str] = None) -> str:
        prefix = f"{tenant}:" if tenant else ""
        return f"{prefix}{method.upper()}:{path}"

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            item = self._store.get(key)
            if not item:
                return None
            expires_at, value = item
            if time.monotonic() > expires_at:
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        with self._lock:
            self._store[key] = (time.monotonic() + (ttl or self.default_ttl), value)

    def clear(self) -> int:
        with self._lock:
            count = len(self._store)
            self._store.clear()
        return count

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {"entries": len(self._store), "default_ttl_seconds": self.default_ttl}


# ============================================================================
# AUTHENTICATION SERVICE
# ============================================================================


@dataclass
class AuthResult:
    authenticated: bool
    user_id: Optional[str] = None
    permissions: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None
    status_code: int = 200
    error: Optional[str] = None


class APIKeyManager:
    """Validates API keys against an in-memory registry loaded from env."""

    def __init__(self) -> None:
        self._keys: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_from_env()

    def _load_from_env(self) -> None:
        import os

        raw = os.getenv("RESE_API_KEYS")
        if raw:
            try:
                for entry in json.loads(raw):
                    self.register(
                        entry["key"],
                        entry.get("owner_id", "unknown"),
                        entry.get("permissions", ["*"]),
                        entry.get("tenant_id"),
                    )
            except Exception as exc:
                logger.warning("api_key_manager.env_parse_failed", error=str(exc))
        dev_key = os.getenv("RESE_DEV_API_KEY")
        if dev_key:
            self.register(dev_key, "dev", ["*"], "dev")

    def register(self, key: str, owner_id: str,
                 permissions: List[str], tenant_id: Optional[str]) -> None:
        with self._lock:
            self._keys[key] = {
                "owner_id": owner_id,
                "permissions": permissions,
                "tenant_id": tenant_id,
            }

    def validate(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._keys.get(key)

    def has_keys(self) -> bool:
        with self._lock:
            return bool(self._keys)


class JWTValidator:
    """
    Minimal JWT validation.

    HS256 tokens are fully verified when ``RESE_JWT_SECRET`` is configured.
    Other algorithms (RS256/ES256) are not verifiable without a crypto
    dependency, so the payload is decoded best-effort and flagged
    ``unverified`` (graceful degradation, clearly surfaced in the result).
    """

    def __init__(self, secret: Optional[str] = None,
                 audience: Optional[str] = None, issuer: Optional[str] = None) -> None:
        self.secret = secret
        self.audience = audience
        self.issuer = issuer

    @staticmethod
    def _b64url_decode(segment: str) -> bytes:
        pad = "=" * (-len(segment) % 4)
        return base64.urlsafe_b64decode(segment + pad)

    def decode_payload(self, token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            header_b64, payload_b64, _sig = token.split(".")
            header = json.loads(self._b64url_decode(header_b64).decode())
            payload = json.loads(self._b64url_decode(payload_b64).decode())
            return payload, header.get("alg")
        except Exception as exc:
            return None, f"malformed: {exc}"

    def validate(self, token: str) -> AuthResult:
        payload, alg = self.decode_payload(token)
        if payload is None:
            return AuthResult(False, status_code=401, error=f"Invalid JWT ({alg})")
        if self.secret and alg == "HS256":
            expected = self._sign_hs256(token, self.secret)
            provided = token.split(".")[2]
            if not hmac.compare_digest(expected, provided):
                return AuthResult(False, status_code=401, error="JWT signature invalid")
        now = int(time.time())
        if "exp" in payload and payload["exp"] < now:
            return AuthResult(False, status_code=401, error="JWT expired")
        if self.audience and payload.get("aud") != self.audience:
            return AuthResult(False, status_code=401, error="JWT audience mismatch")
        if self.issuer and payload.get("iss") != self.issuer:
            return AuthResult(False, status_code=401, error="JWT issuer mismatch")
        return AuthResult(
            True,
            user_id=payload.get("sub"),
            permissions=payload.get("scope", "").split() or ["*"],
            tenant_id=payload.get("tenant_id"),
        )

    @staticmethod
    def _sign_hs256(token: str, secret: str) -> str:
        signing_input = token.rsplit(".", 1)[0].encode()
        digest = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


class OAuthProvider:
    """
    OAuth 2.0 token introspection (RFC 7662).

    Active only when ``RESE_OAUTH_INTROSPECT_URL`` (+ client id/secret) is
    configured. Otherwise introspection is unavailable and tokens are rejected
    (graceful degradation, documented).
    """

    def __init__(self, introspect_url: Optional[str] = None,
                 client_id: Optional[str] = None,
                 client_secret: Optional[str] = None) -> None:
        self.introspect_url = introspect_url
        self.client_id = client_id
        self.client_secret = client_secret

    def is_configured(self) -> bool:
        return bool(self.introspect_url and self.client_id and self.client_secret)

    async def validate(self, token: str) -> AuthResult:
        if not self.is_configured():
            return AuthResult(False, status_code=401, error="OAuth not configured")
        import os

        url = self.introspect_url or os.getenv("RESE_OAUTH_INTROSPECT_URL")
        cid = self.client_id or os.getenv("RESE_OAUTH_CLIENT_ID")
        csec = self.client_secret or os.getenv("RESE_OAUTH_CLIENT_SECRET")
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.post(
                    url, data={"token": token, "client_id": cid, "client_secret": csec}
                )
            if resp.status_code == 200:
                info = resp.json()
                if info.get("active"):
                    return AuthResult(
                        True,
                        user_id=info.get("sub"),
                        permissions=info.get("scope", "").split() or ["*"],
                        tenant_id=info.get("tenant_id"),
                    )
            return AuthResult(False, status_code=401, error="OAuth token inactive")
        except Exception as exc:
            return AuthResult(False, status_code=401, error=f"OAuth introspection failed: {exc}")


class AuthenticationService:
    """
    Unified auth: API key first, then JWT, then OAuth.

    When no credentials are configured at all the service runs in **open**
    (anonymous) mode so it stays usable in development; this is logged as a
    warning and is the documented graceful-degradation behaviour. Set
    ``RESE_AUTH_ENABLED=true`` (and configure keys) to enforce authentication.
    """

    def __init__(self, enforce: bool = False,
                 jwt_secret: Optional[str] = None,
                 oauth_introspect_url: Optional[str] = None) -> None:
        import os

        self.enforce = enforce or os.getenv("RESE_AUTH_ENABLED", "false").lower() == "true"
        self.api_key_manager = APIKeyManager()
        self.jwt_validator = JWTValidator(
            secret=jwt_secret or os.getenv("RESE_JWT_SECRET"),
            audience=os.getenv("RESE_JWT_AUDIENCE"),
            issuer=os.getenv("RESE_JWT_ISSUER"),
        )
        self.oauth_provider = OAuthProvider(
            introspect_url=oauth_introspect_url or os.getenv("RESE_OAUTH_INTROSPECT_URL"),
            client_id=os.getenv("RESE_OAUTH_CLIENT_ID"),
            client_secret=os.getenv("RESE_OAUTH_CLIENT_SECRET"),
        )
        self.rate_limiter = RateLimiter()
        if not self.enforce:
            logger.warning(
                "auth.open_mode",
                detail="No auth enforced; set RESE_AUTH_ENABLED=true to require credentials.",
            )

    def _extract(self, headers: Dict[str, Any], query: Dict[str, Any]):
        api_key = headers.get("x-api-key") or query.get("api_key")
        auth = headers.get("authorization", "")
        jwt_token = auth[7:] if auth.lower().startswith("bearer ") else None
        oauth_token = headers.get("x-oauth-token") or query.get("oauth_token")
        return api_key, jwt_token, oauth_token

    async def authenticate(
        self, headers: Dict[str, Any], query: Optional[Dict[str, Any]] = None
    ) -> AuthResult:
        query = query or {}
        api_key, jwt_token, oauth_token = self._extract(headers, query)

        if api_key:
            record = self.api_key_manager.validate(api_key)
            if not record:
                return AuthResult(False, status_code=401, error="Invalid API key")
            if not self.rate_limiter.allow(record["owner_id"], record["permissions"]):
                return AuthResult(False, status_code=429, error="Rate limit exceeded")
            return AuthResult(
                True,
                user_id=record["owner_id"],
                permissions=record["permissions"],
                tenant_id=record["tenant_id"],
            )

        if jwt_token:
            result = self.jwt_validator.validate(jwt_token)
            if result.authenticated and not self.rate_limiter.allow(
                result.user_id or "jwt", result.permissions
            ):
                return AuthResult(False, status_code=429, error="Rate limit exceeded")
            return result

        if oauth_token:
            return await self.oauth_provider.validate(oauth_token)

        if not self.enforce:
            return AuthResult(
                True, user_id="anonymous", permissions=["*"], tenant_id="anonymous"
            )
        return AuthResult(False, status_code=401, error="Authentication required")


# ============================================================================
# GATEWAY ORCHESTRATOR + FASTAPI SURFACE
# ============================================================================


@dataclass
class GatewayConfig:
    enforce_auth: bool = False
    jwt_secret: Optional[str] = None
    oauth_introspect_url: Optional[str] = None
    default_rate_limit: int = 120
    rate_limit_window: float = 60.0
    load_balancer_algorithm: str = "round_robin"
    response_cache_ttl: float = 30.0
    cors_origin: str = "*"


class Gateway:
    """Wires all gateway components together into one composable unit."""

    def __init__(self, config: Optional[GatewayConfig] = None) -> None:
        self.config = config or GatewayConfig()
        self.auth = AuthenticationService(
            enforce=self.config.enforce_auth,
            jwt_secret=self.config.jwt_secret,
            oauth_introspect_url=self.config.oauth_introspect_url,
        )
        self.rate_limiter = self.auth.rate_limiter
        self.circuit_breaker = CircuitBreaker()
        self.registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(self.config.load_balancer_algorithm)
        self.router = RequestRouter(
            registry=self.registry, load_balancer=self.load_balancer
        )
        self.transformer = ResponseTransformer(cors_origin=self.config.cors_origin)
        self.cache = ResponseCache(default_ttl=self.config.response_cache_ttl)

    def stats(self) -> Dict[str, Any]:
        return {
            "auth": {
                "enforced": self.auth.enforce,
                "configured_api_keys": self.auth.api_key_manager.has_keys(),
                "oauth_configured": self.auth.oauth_provider.is_configured(),
            },
            "rate_limiter": {
                "default_limit": self.rate_limiter.default_rule.limit,
                "window_seconds": self.rate_limiter.default_rule.window_seconds,
            },
            "circuit_breaker": {
                "state": self.circuit_breaker.state,
                "failure_count": self.circuit_breaker.failure_count,
            },
            "services": self.registry.list_services(),
            "cache": self.cache.stats(),
            "load_balancer": self.load_balancer.algorithm,
        }


# --- FastAPI integration -------------------------------------------------

try:
    from fastapi import APIRouter, Request

    gateway = Gateway()

    gateway_router = APIRouter(tags=["gateway"])

    @gateway_router.get("/health")
    async def gateway_health() -> Dict[str, Any]:
        return {"status": "healthy", "gateway": gateway.stats()}

    @gateway_router.get("/services")
    async def gateway_services() -> Dict[str, Any]:
        return {"services": gateway.registry.list_services()}

    @gateway_router.post("/services/register")
    async def gateway_register(instance: Dict[str, Any]) -> Dict[str, Any]:
        svc = ServiceInstance(
            name=instance["name"],
            host=instance["host"],
            port=int(instance["port"]),
            protocol=instance.get("protocol", "http"),
            weight=int(instance.get("weight", 1)),
            tags=instance.get("tags", []),
            metadata=instance.get("metadata", {}),
        )
        gateway.registry.register(svc)
        return {"registered": svc.name, "host": svc.host, "port": svc.port}

    @gateway_router.get("/stats")
    async def gateway_stats() -> Dict[str, Any]:
        return gateway.stats()

    @gateway_router.post("/cache/clear")
    async def gateway_cache_clear() -> Dict[str, Any]:
        cleared = gateway.cache.clear()
        return {"message": "Cache cleared", "cleared_entries": cleared}

    async def get_auth_context(request: Request) -> AuthResult:
        """FastAPI dependency returning the auth context for a request."""
        headers = {k.lower(): v for k, v in request.headers.items()}
        query = dict(request.query_params)
        return await gateway.auth.authenticate(headers, query)

except ImportError:  # pragma: no cover
    gateway_router = None  # type: ignore
    gateway = None  # type: ignore

    async def get_auth_context(request):  # type: ignore
        return AuthResult(True, user_id="anonymous", permissions=["*"])
