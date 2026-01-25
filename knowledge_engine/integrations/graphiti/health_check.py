"""
Graphiti Health Checker for OpenEvolve Knowledge Engine

This module provides health checking capabilities for the Graphiti temporal knowledge graph system.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from .graphiti_temporal_bridge import GraphitiTemporalBridge


logger = logging.getLogger(__name__)


class GraphitiHealthChecker:
    """
    Health checker for Graphiti temporal knowledge bridge.
    
    Provides methods to check the health status of the Graphiti connection
    and related services.
    """
    
    def __init__(self, bridge: GraphitiTemporalBridge):
        """
        Initialize the health checker.
        
        Args:
            bridge: GraphitiTemporalBridge instance to check
        """
        self.bridge = bridge
        logger.info({
            "msg": "GraphitiHealthChecker initialized",
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    async def check_connection(self) -> Dict[str, Any]:
        """
        Check if the Graphiti connection is healthy.
        
        Returns:
            Dictionary with health status
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Checking Graphiti connection health",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Check if bridge is initialized
            if not self.bridge._initialized:
                return {
                    "status": "unhealthy",
                    "details": "Bridge not initialized",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Check if client exists
            if not self.bridge.client:
                return {
                    "status": "unhealthy", 
                    "details": "Graphiti client not available",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            
            # Try a simple operation to verify connection
            try:
                # Attempt to get a list of entities as a basic connectivity test
                entities = await self.bridge.client.get_entity_list()
                
                # If we get here, connection is working
                return {
                    "status": "healthy",
                    "details": f"Connected, {len(entities)} entities in graph",
                    "entity_count": len(entities),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            except Exception as e:
                logger.error({
                    "msg": "Graphiti connection test failed",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                return {
                    "status": "unhealthy",
                    "details": f"Connection test failed: {str(e)}",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error({
                "msg": "Unexpected error during health check",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {
                "status": "error",
                "details": f"Unexpected error: {str(e)}",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def check_full_health(self) -> Dict[str, Any]:
        """
        Perform a comprehensive health check of the Graphiti system.
        
        Returns:
            Dictionary with detailed health information
        """
        start_time = datetime.now(timezone.utc)
        
        logger.info({
            "msg": "Performing full Graphiti health check",
            "timestamp": start_time.isoformat()
        })
        
        try:
            # Check basic connection
            connection_check = await self.check_connection()
            
            # Additional checks could go here:
            # - Check specific Graphiti services
            # - Verify specific functionality
            # - Test performance metrics
            
            # For now, return connection check with additional info
            full_health = {
                "overall_status": connection_check["status"],
                "connection": connection_check,
                "checks_performed": ["connection"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "execution_time_ms": (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            }
            
            logger.info({
                "msg": "Full health check completed",
                "overall_status": full_health["overall_status"],
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return full_health
            
        except Exception as e:
            logger.error({
                "msg": "Full health check failed",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def periodic_health_check(self, interval_seconds: int = 60) -> None:
        """
        Perform periodic health checks.
        
        Args:
            interval_seconds: Interval between checks in seconds
        """
        logger.info({
            "msg": "Starting periodic health checks",
            "interval_seconds": interval_seconds,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        while True:
            try:
                health = await self.check_connection()
                logger.info({
                    "msg": "Periodic health check result",
                    "status": health["status"],
                    "details": health["details"],
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                
                await asyncio.sleep(interval_seconds)
            except Exception as e:
                logger.error({
                    "msg": "Error in periodic health check",
                    "error": str(e),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
                await asyncio.sleep(interval_seconds)