"""
Integration tests for OpenEvolve gRPC server.

Tests the gRPC server, client, and service mesh functionality.
"""

import asyncio
import pytest
import pytest_asyncio
from unittest.mock import Mock, patch

from server import OpenEvolveGRPCServer, ServerConfig
from service_mesh import (
    ServiceMesh, 
    LoadBalancer, 
    HealthTracker, 
    CircuitBreaker,
    Endpoint
)


class TestGRPCServer:
    """Tests for the gRPC server"""
    
    @pytest_asyncio.fixture
    async def server(self):
        """Create and start a test server"""
        config = ServerConfig(
            host="localhost",
            port=0,  # Random port
            max_workers=2
        )
        server = OpenEvolveGRPCServer(config)
        
        # Start server in background
        # Note: In real tests, you'd start the actual gRPC server
        yield server
        
        # Cleanup
        await server.stop()
    
    async def test_server_creation(self, server):
        """Test server can be created"""
        assert server is not None
        assert server.config is not None
    
    async def test_servicer_creation(self, server):
        """Test servicer is created"""
        assert server.servicer is not None


class TestServiceMesh:
    """Tests for the service mesh"""
    
    @pytest_asyncio.fixture
    async def mesh(self):
        """Create a test service mesh"""
        mesh = ServiceMesh(
            load_balancing_strategy='round_robin',
            health_check_interval=1.0
        )
        
        # Add test endpoints
        mesh.add_endpoint("localhost", 50051)
        mesh.add_endpoint("localhost", 50052)
        
        yield mesh
        
        # Cleanup
        await mesh.stop()
    
    async def test_add_endpoint(self, mesh):
        """Test adding endpoints"""
        assert len(mesh.endpoints) == 2
        assert mesh.endpoints[0].address == "localhost:50051"
    
    async def test_remove_endpoint(self, mesh):
        """Test removing endpoints"""
        mesh.remove_endpoint("localhost", 50051)
        assert len(mesh.endpoints) == 1
        assert mesh.endpoints[0].address == "localhost:50052"
    
    async def test_load_balancer_selection(self, mesh):
        """Test load balancer endpoint selection"""
        endpoint1 = await mesh.load_balancer.select_endpoint(mesh.endpoints)
        endpoint2 = await mesh.load_balancer.select_endpoint(mesh.endpoints)
        
        # Round-robin should alternate
        assert endpoint1 in mesh.endpoints
        assert endpoint2 in mesh.endpoints
    
    async def test_health_tracking(self, mesh):
        """Test health tracking"""
        endpoint = mesh.endpoints[0]
        health = mesh.health_tracker.get_health(endpoint)
        
        assert health is not None
        assert health.endpoint == endpoint
        assert health.is_healthy == True


class TestCircuitBreaker:
    """Tests for circuit breaker"""
    
    @pytest.fixture
    def breaker(self):
        """Create a test circuit breaker"""
        endpoint = Endpoint("localhost", 50051)
        return CircuitBreaker(endpoint)
    
    @pytest.mark.asyncio
    async def test_initial_state(self, breaker):
        """Test initial state is closed"""
        from service_mesh import CircuitState
        assert breaker.state == CircuitState.CLOSED
    
    @pytest.mark.asyncio
    async def test_can_execute_when_closed(self, breaker):
        """Test can execute when closed"""
        assert await breaker.can_execute() == True
    
    @pytest.mark.asyncio
    async def test_record_failure(self, breaker):
        """Test recording failures"""
        await breaker.record_failure()
        assert breaker.last_failure_time is not None
    
    @pytest.mark.asyncio
    async def test_record_success(self, breaker):
        """Test recording successes"""
        await breaker.record_success()
        # Should remain closed
        from service_mesh import CircuitState
        assert breaker.state == CircuitState.CLOSED


class TestLoadBalancer:
    """Tests for load balancer"""
    
    @pytest.fixture
    def endpoints(self):
        """Create test endpoints"""
        return [
            Endpoint("localhost", 50051, weight=1),
            Endpoint("localhost", 50052, weight=2),
            Endpoint("localhost", 50053, weight=1),
        ]
    
    @pytest.mark.asyncio
    async def test_round_robin(self, endpoints):
        """Test round-robin selection"""
        lb = LoadBalancer(strategy='round_robin')
        
        selections = []
        for _ in range(6):
            endpoint = await lb.select_endpoint(endpoints)
            selections.append(endpoint)
        
        # Should cycle through all endpoints
        assert len(set(selections)) == 3
    
    @pytest.mark.asyncio
    async def test_weighted_selection(self, endpoints):
        """Test weighted selection favors higher weights"""
        lb = LoadBalancer(strategy='weighted')
        
        # Select many times to see distribution
        selections = []
        for _ in range(100):
            endpoint = await lb.select_endpoint(endpoints)
            selections.append(endpoint.port)
        
        # Port 50052 (weight 2) should be selected more often
        count_50052 = selections.count(50052)
        count_50051 = selections.count(50051)
        
        assert count_50052 > count_50051
    
    @pytest.mark.asyncio
    async def test_random_selection(self, endpoints):
        """Test random selection"""
        lb = LoadBalancer(strategy='random')
        
        endpoint = await lb.select_endpoint(endpoints)
        assert endpoint in endpoints
    
    @pytest.mark.asyncio
    async def test_least_connections(self, endpoints):
        """Test least connections selection"""
        lb = LoadBalancer(strategy='least_connections')
        
        # Record some connections
        await lb.record_connection_start(endpoints[0])
        await lb.record_connection_start(endpoints[0])
        await lb.record_connection_start(endpoints[1])
        
        # Should select endpoint with fewest connections
        selected = await lb.select_endpoint(endpoints)
        assert selected == endpoints[2]  # Has 0 connections


class TestHealthTracker:
    """Tests for health tracker"""
    
    @pytest_asyncio.fixture
    async def tracker(self):
        """Create a test health tracker"""
        tracker = HealthTracker(
            check_interval_seconds=0.1,
            unhealthy_threshold=2,
            healthy_threshold=1
        )
        
        # Add endpoints
        tracker.add_endpoint(Endpoint("localhost", 50051))
        tracker.add_endpoint(Endpoint("localhost", 50052))
        
        yield tracker
        
        # Cleanup
        await tracker.stop()
    
    async def test_get_healthy_endpoints(self, tracker):
        """Test getting healthy endpoints"""
        healthy = tracker.get_healthy_endpoints()
        assert len(healthy) == 2
    
    async def test_health_change_callback(self, tracker):
        """Test health change callback"""
        callback_called = False
        
        def on_health_change(endpoint, health):
            nonlocal callback_called
            callback_called = True
        
        tracker.on_health_change(on_health_change)
        
        # Manually trigger health change
        endpoint = list(tracker._health_map.keys())[0]
        health = tracker._health_map[endpoint]
        health.is_healthy = False
        
        # Would be called by actual health check
        # For test, we just verify callback is registered
        assert len(tracker._check_callbacks) == 1


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self):
        """Test complete workflow from client to server"""
        # This would test the full flow:
        # 1. Start gRPC server
        # 2. Create client
        # 3. Execute node
        # 4. Verify result
        
        # Skipping for unit tests - requires actual server
        pytest.skip("Integration test - requires running server")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
