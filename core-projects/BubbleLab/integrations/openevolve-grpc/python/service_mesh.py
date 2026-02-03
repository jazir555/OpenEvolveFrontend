"""
OpenEvolve Service Mesh

Provides service discovery, load balancing, health checking, and circuit breaking
for the OpenEvolve gRPC services.
"""

import asyncio
import json
import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
from collections import deque

import grpc

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class Endpoint:
    """Represents a service endpoint"""
    host: str
    port: int
    weight: int = 1
    metadata: Dict = field(default_factory=dict)
    
    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"
    
    def __hash__(self):
        return hash(self.address)
    
    def __eq__(self, other):
        if isinstance(other, Endpoint):
            return self.address == other.address
        return False


@dataclass
class EndpointHealth:
    """Health information for an endpoint"""
    endpoint: Endpoint
    is_healthy: bool = True
    last_check: float = field(default_factory=time.time)
    response_time_ms: float = 0.0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    
    @property
    def failure_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5
    success_threshold: int = 3
    timeout_seconds: float = 30.0
    half_open_max_calls: int = 3


class CircuitBreaker:
    """
    Circuit breaker implementation for fault tolerance.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests are rejected immediately
    - HALF_OPEN: Testing if service recovered, limited requests allowed
    """
    
    def __init__(
        self,
        endpoint: Endpoint,
        config: CircuitBreakerConfig = None
    ):
        self.endpoint = endpoint
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.last_failure_time: Optional[float] = None
        self.half_open_calls = 0
        self._lock = asyncio.Lock()
    
    async def can_execute(self) -> bool:
        """Check if request can be executed"""
        async with self._lock:
            if self.state == CircuitState.CLOSED:
                return True
            
            if self.state == CircuitState.OPEN:
                # Check if timeout has passed
                if self.last_failure_time:
                    elapsed = time.time() - self.last_failure_time
                    if elapsed > self.config.timeout_seconds:
                        self.state = CircuitState.HALF_OPEN
                        self.half_open_calls = 0
                        logger.info(f"Circuit breaker for {self.endpoint.address} entering HALF_OPEN")
                        return True
                return False
            
            if self.state == CircuitState.HALF_OPEN:
                if self.half_open_calls < self.config.half_open_max_calls:
                    self.half_open_calls += 1
                    return True
                return False
            
            return True
    
    async def record_success(self):
        """Record successful request"""
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                # If enough successes in half-open, close the circuit
                if self.half_open_calls >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.last_failure_time = None
                    self.half_open_calls = 0
                    logger.info(f"Circuit breaker for {self.endpoint.address} CLOSED")
    
    async def record_failure(self):
        """Record failed request"""
        async with self._lock:
            self.last_failure_time = time.time()
            
            if self.state == CircuitState.HALF_OPEN:
                # Failure in half-open goes back to open
                self.state = CircuitState.OPEN
                self.half_open_calls = 0
                logger.warning(f"Circuit breaker for {self.endpoint.address} OPEN (half-open failure)")
            elif self.state == CircuitState.CLOSED:
                # Count failures to potentially open circuit
                # This would need to be connected to actual failure tracking
                pass
    
    async def should_open(self, consecutive_failures: int) -> bool:
        """Check if circuit should open based on failures"""
        async with self._lock:
            if consecutive_failures >= self.config.failure_threshold:
                if self.state == CircuitState.CLOSED:
                    self.state = CircuitState.OPEN
                    self.last_failure_time = time.time()
                    logger.warning(f"Circuit breaker for {self.endpoint.address} OPEN")
                    return True
            return False


class LoadBalancer:
    """
    Load balancer for distributing requests across endpoints.
    
    Supports multiple strategies:
    - ROUND_ROBIN: Distribute evenly
    - RANDOM: Random selection
    - WEIGHTED: Weighted by endpoint weight
    - LEAST_CONNECTIONS: Fewest active connections
    - HEALTH_BASED: Based on health scores
    """
    
    STRATEGIES = ['round_robin', 'random', 'weighted', 'least_connections', 'health_based']
    
    def __init__(
        self,
        strategy: str = 'round_robin',
        health_tracker: 'HealthTracker' = None
    ):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown strategy: {strategy}. Use: {self.STRATEGIES}")
        
        self.strategy = strategy
        self.health_tracker = health_tracker
        self._round_robin_index = 0
        self._connection_counts: Dict[Endpoint, int] = {}
        self._lock = asyncio.Lock()
    
    async def select_endpoint(
        self,
        endpoints: List[Endpoint],
        exclude: Set[Endpoint] = None
    ) -> Optional[Endpoint]:
        """Select an endpoint based on the configured strategy"""
        
        exclude = exclude or set()
        available = [e for e in endpoints if e not in exclude]
        
        if not available:
            return None
        
        if self.strategy == 'round_robin':
            return await self._round_robin(available)
        elif self.strategy == 'random':
            return await self._random(available)
        elif self.strategy == 'weighted':
            return await self._weighted(available)
        elif self.strategy == 'least_connections':
            return await self._least_connections(available)
        elif self.strategy == 'health_based':
            return await self._health_based(available)
        
        return available[0]
    
    async def _round_robin(self, endpoints: List[Endpoint]) -> Endpoint:
        """Round-robin selection"""
        async with self._lock:
            endpoint = endpoints[self._round_robin_index % len(endpoints)]
            self._round_robin_index += 1
            return endpoint
    
    async def _random(self, endpoints: List[Endpoint]) -> Endpoint:
        """Random selection"""
        return random.choice(endpoints)
    
    async def _weighted(self, endpoints: List[Endpoint]) -> Endpoint:
        """Weighted random selection"""
        total_weight = sum(e.weight for e in endpoints)
        r = random.uniform(0, total_weight)
        
        current = 0
        for endpoint in endpoints:
            current += endpoint.weight
            if r <= current:
                return endpoint
        
        return endpoints[-1]
    
    async def _least_connections(self, endpoints: List[Endpoint]) -> Endpoint:
        """Select endpoint with fewest active connections"""
        counts = [(self._connection_counts.get(e, 0), e) for e in endpoints]
        counts.sort(key=lambda x: x[0])
        return counts[0][1]
    
    async def _health_based(self, endpoints: List[Endpoint]) -> Endpoint:
        """Select based on health scores"""
        if not self.health_tracker:
            return await self._round_robin(endpoints)
        
        # Get health scores
        health_scores = []
        for endpoint in endpoints:
            health = self.health_tracker.get_health(endpoint)
            if health and health.is_healthy:
                # Higher score = better health + faster response
                score = (1 - health.failure_rate) * 100 + (1000 / (health.response_time_ms + 1))
                health_scores.append((score, endpoint))
        
        if not health_scores:
            # All unhealthy, fall back to round robin
            return await self._round_robin(endpoints)
        
        # Select best
        health_scores.sort(key=lambda x: x[0], reverse=True)
        return health_scores[0][1]
    
    async def record_connection_start(self, endpoint: Endpoint):
        """Record connection start"""
        async with self._lock:
            self._connection_counts[endpoint] = self._connection_counts.get(endpoint, 0) + 1
    
    async def record_connection_end(self, endpoint: Endpoint):
        """Record connection end"""
        async with self._lock:
            self._connection_counts[endpoint] = max(0, self._connection_counts.get(endpoint, 0) - 1)


class HealthTracker:
    """
    Tracks health of service endpoints.
    
    Performs periodic health checks and maintains health statistics.
    """
    
    def __init__(
        self,
        check_interval_seconds: float = 30.0,
        timeout_seconds: float = 5.0,
        unhealthy_threshold: int = 3,
        healthy_threshold: int = 2
    ):
        self.check_interval = check_interval_seconds
        self.timeout = timeout_seconds
        self.unhealthy_threshold = unhealthy_threshold
        self.healthy_threshold = healthy_threshold
        
        self._health_map: Dict[Endpoint, EndpointHealth] = {}
        self._check_callbacks: List[Callable[[Endpoint, EndpointHealth], None]] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
    
    def add_endpoint(self, endpoint: Endpoint):
        """Add an endpoint to track"""
        self._health_map[endpoint] = EndpointHealth(endpoint=endpoint)
    
    def remove_endpoint(self, endpoint: Endpoint):
        """Remove an endpoint from tracking"""
        if endpoint in self._health_map:
            del self._health_map[endpoint]
    
    def get_health(self, endpoint: Endpoint) -> Optional[EndpointHealth]:
        """Get health info for an endpoint"""
        return self._health_map.get(endpoint)
    
    def get_healthy_endpoints(self) -> List[Endpoint]:
        """Get list of healthy endpoints"""
        return [
            e for e, h in self._health_map.items()
            if h.is_healthy
        ]
    
    def on_health_change(self, callback: Callable[[Endpoint, EndpointHealth], None]):
        """Register callback for health changes"""
        self._check_callbacks.append(callback)
    
    async def start(self):
        """Start health check loop"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._health_check_loop())
        logger.info("Health tracker started")
    
    async def stop(self):
        """Stop health check loop"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Health tracker stopped")
    
    async def _health_check_loop(self):
        """Main health check loop"""
        while self._running:
            try:
                await self._check_all_endpoints()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _check_all_endpoints(self):
        """Check all endpoints"""
        tasks = [
            self._check_endpoint(endpoint)
            for endpoint in list(self._health_map.keys())
        ]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_endpoint(self, endpoint: Endpoint):
        """Check a single endpoint"""
        start_time = time.time()
        
        try:
            # Perform health check
            # This would connect to the gRPC health service
            is_healthy = await self._perform_health_check(endpoint)
            
            response_time = (time.time() - start_time) * 1000
            
            async with self._lock:
                health = self._health_map.get(endpoint)
                if health:
                    previous_health = health.is_healthy
                    
                    health.last_check = time.time()
                    health.response_time_ms = response_time
                    health.total_requests += 1
                    
                    if is_healthy:
                        health.consecutive_successes += 1
                        health.consecutive_failures = 0
                        
                        if health.consecutive_successes >= self.healthy_threshold:
                            health.is_healthy = True
                    else:
                        health.consecutive_failures += 1
                        health.consecutive_successes = 0
                        health.failed_requests += 1
                        
                        if health.consecutive_failures >= self.unhealthy_threshold:
                            health.is_healthy = False
                    
                    # Notify if health changed
                    if previous_health != health.is_healthy:
                        for callback in self._check_callbacks:
                            try:
                                callback(endpoint, health)
                            except Exception as e:
                                logger.error(f"Health callback error: {e}")
        
        except Exception as e:
            logger.error(f"Health check failed for {endpoint.address}: {e}")
            
            async with self._lock:
                health = self._health_map.get(endpoint)
                if health:
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                    
                    if health.consecutive_failures >= self.unhealthy_threshold:
                        health.is_healthy = False
    
    async def _perform_health_check(self, endpoint: Endpoint) -> bool:
        """Perform actual health check against endpoint"""
        # This would implement the actual gRPC health check
        # For now, assume healthy
        return True


class ServiceMesh:
    """
    Service mesh for OpenEvolve gRPC services.
    
    Combines service discovery, load balancing, health checking, and circuit breaking
    into a unified interface.
    """
    
    def __init__(
        self,
        load_balancing_strategy: str = 'round_robin',
        health_check_interval: float = 30.0,
        circuit_breaker_config: CircuitBreakerConfig = None
    ):
        self.endpoints: List[Endpoint] = []
        self.health_tracker = HealthTracker(check_interval_seconds=health_check_interval)
        self.load_balancer = LoadBalancer(
            strategy=load_balancing_strategy,
            health_tracker=self.health_tracker
        )
        self.circuit_breakers: Dict[Endpoint, CircuitBreaker] = {}
        self.circuit_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Request retry tracking
        self._retry_attempts: Dict[str, int] = {}
        self._lock = asyncio.Lock()
    
    def add_endpoint(self, host: str, port: int, weight: int = 1, **metadata):
        """Add a service endpoint"""
        endpoint = Endpoint(host=host, port=port, weight=weight, metadata=metadata)
        self.endpoints.append(endpoint)
        self.health_tracker.add_endpoint(endpoint)
        self.circuit_breakers[endpoint] = CircuitBreaker(endpoint, self.circuit_config)
        logger.info(f"Added endpoint: {endpoint.address}")
    
    def remove_endpoint(self, host: str, port: int):
        """Remove a service endpoint"""
        endpoint = Endpoint(host=host, port=port)
        if endpoint in self.endpoints:
            self.endpoints.remove(endpoint)
            self.health_tracker.remove_endpoint(endpoint)
            if endpoint in self.circuit_breakers:
                del self.circuit_breakers[endpoint]
            logger.info(f"Removed endpoint: {endpoint.address}")
    
    async def start(self):
        """Start the service mesh"""
        await self.health_tracker.start()
        
        # Register for health changes
        self.health_tracker.on_health_change(self._on_health_change)
        
        logger.info("Service mesh started")
    
    async def stop(self):
        """Stop the service mesh"""
        await self.health_tracker.stop()
        logger.info("Service mesh stopped")
    
    def _on_health_change(self, endpoint: Endpoint, health: EndpointHealth):
        """Handle health status changes"""
        status = "healthy" if health.is_healthy else "unhealthy"
        logger.info(f"Endpoint {endpoint.address} is now {status}")
        
        # Update circuit breaker
        circuit = self.circuit_breakers.get(endpoint)
        if circuit:
            if not health.is_healthy:
                asyncio.create_task(circuit.should_open(health.consecutive_failures))
    
    async def execute_with_resilience(
        self,
        operation: Callable[[Endpoint], Any],
        max_retries: int = 3,
        retry_delay_ms: float = 1000
    ) -> any:
        """
        Execute an operation with full resilience patterns.
        
        This combines load balancing, circuit breaking, and retry logic.
        """
        tried_endpoints: Set[Endpoint] = set()
        last_error: Optional[Exception] = None
        
        for attempt in range(max_retries):
            # Select endpoint
            endpoint = await self._select_healthy_endpoint(tried_endpoints)
            
            if not endpoint:
                # No healthy endpoints available
                if last_error:
                    raise last_error
                raise RuntimeError("No healthy endpoints available")
            
            tried_endpoints.add(endpoint)
            circuit = self.circuit_breakers.get(endpoint)
            
            # Check circuit breaker
            if circuit and not await circuit.can_execute():
                logger.debug(f"Circuit open for {endpoint.address}, skipping")
                continue
            
            # Execute
            await self.load_balancer.record_connection_start(endpoint)
            
            try:
                start_time = time.time()
                result = await operation(endpoint)
                response_time = (time.time() - start_time) * 1000
                
                # Record success
                if circuit:
                    await circuit.record_success()
                
                # Update health tracker
                health = self.health_tracker.get_health(endpoint)
                if health:
                    health.consecutive_successes += 1
                    health.consecutive_failures = 0
                    health.response_time_ms = response_time
                
                return result
                
            except Exception as e:
                last_error = e
                
                # Record failure
                if circuit:
                    await circuit.record_failure()
                
                # Update health tracker
                health = self.health_tracker.get_health(endpoint)
                if health:
                    health.consecutive_failures += 1
                    health.consecutive_successes = 0
                    health.failed_requests += 1
                    
                    # Check if we should open the circuit
                    await circuit.should_open(health.consecutive_failures)
                
                logger.warning(f"Request to {endpoint.address} failed (attempt {attempt + 1}): {e}")
                
                # Wait before retry
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay_ms / 1000 * (2 ** attempt))
                    
            finally:
                await self.load_balancer.record_connection_end(endpoint)
        
        # All retries exhausted
        if last_error:
            raise last_error
        raise RuntimeError("All retry attempts failed")
    
    async def _select_healthy_endpoint(self, exclude: Set[Endpoint] = None) -> Optional[Endpoint]:
        """Select a healthy endpoint using load balancing"""
        exclude = exclude or set()
        
        # Get healthy endpoints that aren't excluded
        healthy = [
            e for e in self.health_tracker.get_healthy_endpoints()
            if e not in exclude
        ]
        
        if not healthy:
            # Fall back to any endpoint if all are excluded
            healthy = [e for e in self.endpoints if e not in exclude]
        
        if not healthy:
            return None
        
        return await self.load_balancer.select_endpoint(healthy)
    
    def get_stats(self) -> Dict:
        """Get service mesh statistics"""
        stats = {
            'total_endpoints': len(self.endpoints),
            'healthy_endpoints': len(self.health_tracker.get_healthy_endpoints()),
            'circuit_breakers': {},
            'endpoint_health': {}
        }
        
        for endpoint, circuit in self.circuit_breakers.items():
            stats['circuit_breakers'][endpoint.address] = circuit.state.value
        
        for endpoint in self.endpoints:
            health = self.health_tracker.get_health(endpoint)
            if health:
                stats['endpoint_health'][endpoint.address] = {
                    'is_healthy': health.is_healthy,
                    'failure_rate': health.failure_rate,
                    'response_time_ms': health.response_time_ms,
                    'total_requests': health.total_requests,
                    'failed_requests': health.failed_requests
                }
        
        return stats


# Convenience function to create a service mesh
def create_service_mesh(
    endpoints: List[tuple],
    strategy: str = 'round_robin',
    health_check_interval: float = 30.0
) -> ServiceMesh:
    """
    Create a service mesh with the given endpoints.
    
    Args:
        endpoints: List of (host, port) tuples or (host, port, weight) tuples
        strategy: Load balancing strategy
        health_check_interval: Seconds between health checks
    
    Returns:
        Configured ServiceMesh instance
    """
    mesh = ServiceMesh(
        load_balancing_strategy=strategy,
        health_check_interval=health_check_interval
    )
    
    for endpoint in endpoints:
        if len(endpoint) == 2:
            host, port = endpoint
            mesh.add_endpoint(host, port)
        else:
            host, port, weight = endpoint
            mesh.add_endpoint(host, port, weight)
    
    return mesh
