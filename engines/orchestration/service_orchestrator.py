"""
Service Orchestrator - License: Apache 2.0

Central orchestrator for managing all OpenEvolve services:
- Unified MCP Server
- GraphQL API
- Event Bus
- OpenTelemetry
- REST API
- Workflow Engine

Dependencies (all permissive licenses):
- fastapi: MIT
- uvicorn: BSD
- pydantic: MIT
- valkey-py: MIT
- opentelemetry: Apache 2.0

Author: OpenEvolve
Date: 2026-02-02
"""

import asyncio
import logging
import signal
import sys
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading
import os

# FastAPI - MIT
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Uvicorn - BSD
import uvicorn

# **ACTUAL INTEGRATION**: Alerting, knowledge, and adaptive for Service Orchestrator
try:
    from alerting_system import get_alert_manager, AlertSeverity
    ALERTING_AVAILABLE = True
except ImportError:
    ALERTING_AVAILABLE = False

try:
    from knowledge_engine.enterprise_knowledge_engine import get_knowledge_engine, KnowledgeArtifact
    KNOWLEDGE_AVAILABLE = True
except ImportError:
    KNOWLEDGE_AVAILABLE = False

try:
    from adaptive_strategy_selector import StrategyPerformanceTracker, StrategyPerformanceData
    ADAPTIVE_AVAILABLE = True
except ImportError:
    ADAPTIVE_AVAILABLE = False


logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service lifecycle status."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    ERROR = "error"
    STOPPING = "stopping"


@dataclass
class ServiceInfo:
    """Information about a managed service."""
    name: str
    status: ServiceStatus = ServiceStatus.STOPPED
    start_time: Optional[datetime] = None
    stop_time: Optional[datetime] = None
    error_message: Optional[str] = None
    port: Optional[int] = None
    pid: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def uptime_seconds(self) -> Optional[float]:
        """Calculate uptime in seconds."""
        if self.start_time and self.status == ServiceStatus.RUNNING:
            return (datetime.utcnow() - self.start_time).total_seconds()
        return None


class ManagedService:
    """Base class for managed services."""
    
    def __init__(self, name: str, port: Optional[int] = None):
        self.name = name
        self.port = port
        self.info = ServiceInfo(name=name, port=port)
        self._stop_event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        
    async def start(self) -> bool:
        """Start the service. Returns True if successful."""
        # Default implementation - should be overridden by subclasses
        self.info.status = ServiceStatus.RUNNING
        self.info.last_start_time = datetime.now()
        return True

    async def stop(self) -> bool:
        """Stop the service. Returns True if successful."""
        # Default implementation - should be overridden by subclasses
        self.info.status = ServiceStatus.STOPPED
        self.info.last_stop_time = datetime.now()
        return True
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check. Returns health info dict."""
        return {"status": "unknown"}


class MCPService(ManagedService):
    """Managed MCP Server service."""
    
    def __init__(self):
        super().__init__("mcp_server", port=None)
        self.server = None
        
    async def start(self) -> bool:
        """Start MCP server."""
        try:
            from unified_mcp_server import get_unified_mcp_server
            
            self.info.status = ServiceStatus.STARTING
            self.server = get_unified_mcp_server()
            self.server.register_all_tools()
            
            # MCP runs on stdio, not network
            self._task = asyncio.create_task(self.server.run())
            
            self.info.status = ServiceStatus.RUNNING
            self.info.start_time = datetime.utcnow()
            self.info.pid = os.getpid()
            logger.info("MCP Server started")
            return True
            
        except Exception as e:
            self.info.status = ServiceStatus.ERROR
            self.info.error_message = str(e)
            logger.error(f"MCP Server failed to start: {e}")
            return False
            
    async def stop(self) -> bool:
        """Stop MCP server."""
        self.info.status = ServiceStatus.STOPPING
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.info.status = ServiceStatus.STOPPED
        self.info.stop_time = datetime.utcnow()
        logger.info("MCP Server stopped")
        return True
        
    async def health_check(self) -> Dict[str, Any]:
        """Check MCP server health."""
        return {
            "status": self.info.status.value,
            "tools_registered": len(self.server.registry._tools) if self.server else 0,
            "uptime_seconds": self.info.uptime_seconds
        }


class GraphQLService(ManagedService):
    """Managed GraphQL API service."""
    
    def __init__(self, port: int = 8001):
        super().__init__("graphql_api", port=port)
        self.app = None
        self.server = None
        
    async def start(self) -> bool:
        """Start GraphQL server."""
        try:
            from graphql_server import create_graphql_app
            
            self.info.status = ServiceStatus.STARTING
            self.app = create_graphql_app()
            
            # Run in background
            config = uvicorn.Config(
                self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )
            self.server = uvicorn.Server(config)
            self._task = asyncio.create_task(self.server.serve())
            
            # Wait for server to start
            for _ in range(50):  # 5 seconds max
                if self.server.started:
                    break
                await asyncio.sleep(0.1)
            
            if self.server.started:
                self.info.status = ServiceStatus.RUNNING
                self.info.start_time = datetime.utcnow()
                self.info.pid = os.getpid()
                logger.info(f"GraphQL API started on port {self.port}")
                return True
            else:
                raise TimeoutError("Server failed to start")
                
        except Exception as e:
            self.info.status = ServiceStatus.ERROR
            self.info.error_message = str(e)
            logger.error(f"GraphQL API failed to start: {e}")
            return False
            
    async def stop(self) -> bool:
        """Stop GraphQL server."""
        self.info.status = ServiceStatus.STOPPING
        if self.server:
            self.server.should_exit = True
            if self._task:
                await self._task
        self.info.status = ServiceStatus.STOPPED
        self.info.stop_time = datetime.utcnow()
        logger.info("GraphQL API stopped")
        return True
        
    async def health_check(self) -> Dict[str, Any]:
        """Check GraphQL API health."""
        healthy = self.server is not None and self.server.started
        return {
            "status": "healthy" if healthy else "unhealthy",
            "port": self.port,
            "endpoint": f"http://localhost:{self.port}/graphql",
            "uptime_seconds": self.info.uptime_seconds
        }


class RESTAPIService(ManagedService):
    """Managed REST API service."""
    
    def __init__(self, port: int = 8000):
        super().__init__("rest_api", port=port)
        self.server = None
        
    async def start(self) -> bool:
        """Start REST API server."""
        try:
            from api_server import app
            from telemetry import telemetry
            
            self.info.status = ServiceStatus.STARTING
            
            # Instrument with telemetry
            telemetry.instrument_fastapi(app)
            
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=self.port,
                log_level="info"
            )
            self.server = uvicorn.Server(config)
            self._task = asyncio.create_task(self.server.serve())
            
            # Wait for startup
            for _ in range(50):
                if self.server.started:
                    break
                await asyncio.sleep(0.1)
            
            if self.server.started:
                self.info.status = ServiceStatus.RUNNING
                self.info.start_time = datetime.utcnow()
                self.info.pid = os.getpid()
                logger.info(f"REST API started on port {self.port}")
                return True
            else:
                raise TimeoutError("Server failed to start")
                
        except Exception as e:
            self.info.status = ServiceStatus.ERROR
            self.info.error_message = str(e)
            logger.error(f"REST API failed to start: {e}")
            return False
            
    async def stop(self) -> bool:
        """Stop REST API."""
        self.info.status = ServiceStatus.STOPPING
        if self.server:
            self.server.should_exit = True
            if self._task:
                await self._task
        self.info.status = ServiceStatus.STOPPED
        self.info.stop_time = datetime.utcnow()
        logger.info("REST API stopped")
        return True
        
    async def health_check(self) -> Dict[str, Any]:
        """Check REST API health."""
        healthy = self.server is not None and self.server.started
        return {
            "status": "healthy" if healthy else "unhealthy",
            "port": self.port,
            "endpoint": f"http://localhost:{self.port}",
            "uptime_seconds": self.info.uptime_seconds
        }


class EventBusService(ManagedService):
    """Managed Event Bus service."""
    
    def __init__(self):
        super().__init__("event_bus", port=None)
        self.bus = None
        
    async def start(self) -> bool:
        """Start Event Bus."""
        try:
            from event_bus import get_event_bus
            
            self.info.status = ServiceStatus.STARTING
            self.bus = await get_event_bus()
            
            self.info.status = ServiceStatus.RUNNING
            self.info.start_time = datetime.utcnow()
            self.info.metadata["connected"] = self.bus._connected
            logger.info("Event Bus started")
            return True
            
        except Exception as e:
            self.info.status = ServiceStatus.ERROR
            self.info.error_message = str(e)
            logger.error(f"Event Bus failed to start: {e}")
            return False
            
    async def stop(self) -> bool:
        """Stop Event Bus."""
        self.info.status = ServiceStatus.STOPPING
        if self.bus:
            await self.bus.disconnect()
        self.info.status = ServiceStatus.STOPPED
        self.info.stop_time = datetime.utcnow()
        logger.info("Event Bus stopped")
        return True
        
    async def health_check(self) -> Dict[str, Any]:
        """Check Event Bus health."""
        connected = self.bus is not None and self.bus._connected
        return {
            "status": "healthy" if connected else "degraded",
            "valkey_connected": connected,
            "subscribers": sum(len(s) for s in self.bus._subscribers.values()) if self.bus else 0,
            "uptime_seconds": self.info.uptime_seconds
        }


class ServiceOrchestrator:
    """
    Central orchestrator for managing all OpenEvolve services.
    
    Provides:
    - Service lifecycle management (start/stop)
    - Health monitoring
n    - Graceful shutdown
    - Service discovery
    - Configuration management
    
    License: Apache 2.0
    """
    
    def __init__(self):
        self.services: Dict[str, ManagedService] = {}
        self._shutdown_event = asyncio.Event()
        self._health_check_task: Optional[asyncio.Task] = None
        self._orchestrator_app: Optional[FastAPI] = None
        
    def register_service(self, service: ManagedService) -> None:
        """Register a service with the orchestrator."""
        self.services[service.name] = service
        logger.info(f"Registered service: {service.name}")
        
    async def start_all(self, service_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """
        Start all or specified services.

        Returns:
            Dict mapping service name to success status
        """
        import time
        start_time = time.time()

        results = {}
        services_to_start = (
            [self.services[name] for name in service_names if name in self.services]
            if service_names
            else list(self.services.values())
        )

        # Start services in order
        for service in services_to_start:
            logger.info(f"Starting service: {service.name}")
            success = await service.start()
            results[service.name] = success

            if not success:
                logger.error(f"Failed to start {service.name}")
                # **ACTUAL INTEGRATION**: Trigger alert for failed service start
                self._trigger_service_alerts("start_service", False, service.name, "Service failed to start")

        # Start health check monitoring
        self._health_check_task = asyncio.create_task(self._health_monitor())

        # Start orchestrator API
        await self._start_orchestrator_api()

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance
        duration = time.time() - start_time
        all_success = all(results.values())
        self._extract_service_knowledge("start_all", "all_services", results)
        self._track_service_performance("start_all", all_success, duration, len(results))

        # If any service failed, trigger an alert
        if not all_success:
            failed_services = [name for name, success in results.items() if not success]
            self._trigger_service_alerts("start_all", False, None, f"Some services failed: {failed_services}")

        return results
        
    async def stop_all(self, timeout: float = 30.0) -> Dict[str, bool]:
        """
        Stop all services gracefully.

        Returns:
            Dict mapping service name to success status
        """
        import time
        start_time = time.time()

        results = {}

        # Cancel health check
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass

        # Stop services in reverse order
        for service in reversed(list(self.services.values())):
            logger.info(f"Stopping service: {service.name}")
            try:
                success = await asyncio.wait_for(
                    service.stop(),
                    timeout=timeout / len(self.services)
                )
                results[service.name] = success
            except asyncio.TimeoutError:
                logger.error(f"Timeout stopping {service.name}")
                results[service.name] = False

        self._shutdown_event.set()

        # **ACTUAL INTEGRATION**: Extract knowledge and track performance
        duration = time.time() - start_time
        all_success = all(results.values())
        self._extract_service_knowledge("stop_all", "all_services", results)
        self._track_service_performance("stop_all", all_success, duration, len(results))

        return results
        
    async def _health_monitor(self) -> None:
        """Background task to monitor service health."""
        while not self._shutdown_event.is_set():
            try:
                for service in self.services.values():
                    if service.info.status == ServiceStatus.RUNNING:
                        health = await service.health_check()
                        if health.get("status") != "healthy":
                            logger.warning(
                                f"Service {service.name} health check: {health['status']}"
                            )
                            service.info.status = ServiceStatus.DEGRADED
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(5)
                
    async def _start_orchestrator_api(self, port: int = 8080) -> None:
        """Start the orchestrator management API."""
        self._orchestrator_app = FastAPI(
            title="OpenEvolve Service Orchestrator",
            description="Management API for OpenEvolve services"
        )
        
        @self._orchestrator_app.get("/health")
        async def health():
            """Get overall system health."""
            services_health = {}
            for name, service in self.services.items():
                services_health[name] = {
                    "status": service.info.status.value,
                    "health": await service.health_check()
                }
            
            all_healthy = all(
                s.info.status == ServiceStatus.RUNNING
                for s in self.services.values()
            )
            
            return {
                "status": "healthy" if all_healthy else "degraded",
                "services": services_health,
                "timestamp": datetime.utcnow().isoformat()
            }
        
        @self._orchestrator_app.get("/services")
        async def list_services():
            """List all services and their status."""
            return {
                name: {
                    "status": s.info.status.value,
                    "port": s.info.port,
                    "uptime_seconds": s.info.uptime_seconds,
                    "start_time": s.info.start_time.isoformat() if s.info.start_time else None
                }
                for name, s in self.services.items()
            }
        
        @self._orchestrator_app.post("/services/{name}/restart")
        async def restart_service(name: str):
            """Restart a specific service."""
            if name not in self.services:
                raise HTTPException(status_code=404, detail="Service not found")
            
            service = self.services[name]
            await service.stop()
            success = await service.start()
            
            return {"success": success, "status": service.info.status.value}
        
        # Start in background
        config = uvicorn.Config(
            self._orchestrator_app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        asyncio.create_task(server.serve())
        logger.info(f"Orchestrator API started on port {port}")

    # =========================================================================
    # ACTUAL INTEGRATION METHODS - Alerting, knowledge, and adaptive for Service Orchestrator
    # =========================================================================

    def _trigger_service_alerts(
        self,
        operation: str,
        success: bool,
        service_name: Optional[str] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """**ACTUAL INTEGRATION**: Trigger alerts for service operation failures."""
        if not ALERTING_AVAILABLE:
            return

        try:
            alert_manager = get_alert_manager()

            # Alert on failures
            if not success:
                alert_manager.create_alert(
                    title=f"Service Orchestrator Alert: {operation}",
                    description=f"Service operation '{operation}' failed" +
                                 (f" for service '{service_name}'" if service_name else "") +
                                 ". " + (f"Error: {error}" if error else ""),
                    severity=AlertSeverity.HIGH.value,
                    source="service_orchestrator",
                    component="service_management",
                    metadata=metadata or {}
                )

        except Exception as e:
            logger.error(f"Failed to trigger service orchestrator alert: {e}")

    def _extract_service_knowledge(
        self,
        operation: str,
        service_name: str,
        result: Dict[str, bool]
    ) -> bool:
        """**ACTUAL INTEGRATION**: Extract service operation knowledge to knowledge engine."""
        if not KNOWLEDGE_AVAILABLE:
            return False

        try:
            knowledge_engine = get_knowledge_engine()

            artifact = KnowledgeArtifact(
                artifact_id=f"svc_orch_{operation}_{service_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                artifact_type="service_operation",
                source_component="service_orchestrator",
                title=f"Service Operation: {operation} - {service_name}",
                content={
                    "operation": operation,
                    "service_name": service_name,
                    "results": result,
                    "timestamp": datetime.now().isoformat()
                },
                metadata={
                    "total_services": len(result),
                    "successful_services": sum(1 for v in result.values() if v)
                },
                tags=["service_orchestrator", operation, service_name]
            )

            knowledge_engine.store_artifact(artifact)
            logger.debug(f"Extracted service orchestrator knowledge for {operation}")
            return True

        except Exception as e:
            logger.error(f"Failed to extract service orchestrator knowledge: {e}")
            return False

    def _track_service_performance(
        self,
        operation: str,
        success: bool,
        duration_seconds: float,
        services_count: int = 0
    ):
        """**ACTUAL INTEGRATION**: Track service operation performance in adaptive selector."""
        if not ADAPTIVE_AVAILABLE:
            return

        try:
            tracker = StrategyPerformanceTracker()

            quality = 1.0 if success else 0.0

            performance_data = StrategyPerformanceData(
                strategy_name=f"service_orchestrator_{operation}",
                success_count=1 if success else 0,
                failure_count=0 if success else 1,
                average_quality=quality,
                last_used=datetime.now(),
                total_attempts=1,
                metadata={
                    "operation": operation,
                    "duration_seconds": duration_seconds,
                    "services_count": services_count
                }
            )

            if hasattr(tracker, 'performance_history'):
                tracker.performance_history.append(performance_data)
                logger.debug(f"Tracked service orchestrator performance for {operation}")

        except Exception as e:
            logger.error(f"Failed to track service orchestrator performance: {e}")

    def get_service(self, name: str) -> Optional[ManagedService]:
        """Get a service by name."""
        return self.services.get(name)
        
    async def wait_for_shutdown(self) -> None:
        """Wait for shutdown signal."""
        await self._shutdown_event.wait()


# Global orchestrator instance
_orchestrator: Optional[ServiceOrchestrator] = None


def get_orchestrator() -> ServiceOrchestrator:
    """Get or create the global orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ServiceOrchestrator()
    return _orchestrator


async def run_all_services(
    rest_port: int = 8000,
    graphql_port: int = 8001,
    orchestrator_port: int = 8080,
    enable_mcp: bool = True,
    enable_event_bus: bool = True
) -> None:
    """
    Run all OpenEvolve services.
    
    This is the main entry point for running the complete system.
    
    Args:
        rest_port: Port for REST API
        graphql_port: Port for GraphQL API
        orchestrator_port: Port for orchestrator management API
        enable_mcp: Whether to start MCP server
        enable_event_bus: Whether to start Event Bus
    """
    orchestrator = get_orchestrator()
    
    # Register services
    orchestrator.register_service(RESTAPIService(port=rest_port))
    orchestrator.register_service(GraphQLService(port=graphql_port))
    
    if enable_event_bus:
        orchestrator.register_service(EventBusService())
    
    if enable_mcp:
        orchestrator.register_service(MCPService())
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler(sig):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(orchestrator.stop_all())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: signal_handler(s))
    
    # Start all services
    logger.info("Starting OpenEvolve services...")
    results = await orchestrator.start_all()
    
    success_count = sum(1 for r in results.values() if r)
    logger.info(f"Started {success_count}/{len(results)} services successfully")
    
    if success_count == 0:
        logger.error("No services started, exiting")
        return
    
    # Print access information
    print("\n" + "=" * 60)
    print("OPENEVOLVE SERVICES RUNNING")
    print("=" * 60)
    print(f"REST API:       http://localhost:{rest_port}")
    print(f"GraphQL API:    http://localhost:{graphql_port}/graphql")
    print(f"GraphQL IDE:    http://localhost:{graphql_port}/graphql")
    print(f"Orchestrator:   http://localhost:{orchestrator_port}")
    print(f"MCP Server:     stdio/sse (Claude/Cursor integration)")
    print("=" * 60)
    print("Press Ctrl+C to stop all services")
    print("=" * 60 + "\n")
    
    # Wait for shutdown
    await orchestrator.wait_for_shutdown()
    
    logger.info("OpenEvolve services stopped")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(run_all_services())
