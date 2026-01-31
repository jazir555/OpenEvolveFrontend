"""
Final Integration Layer

Complete integration of all enhanced knowledge engine components:
- Unified platform
- NLP layer
- Multi-tenancy
- Backup/Recovery
- API Gateway
- Performance monitoring
- Knowledge versioning
- Import/Export
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import uuid

# Import all components
from unified_knowledge_platform import UnifiedKnowledgePlatform
from nlp_layer import NLPEngine, DocumentAnalysis
from multi_tenant import TenantManager, TenantContext
from backup_recovery import BackupEngine, LocalBackupStorage, DisasterRecovery
from api_gateway import (
    RESTAPIGateway, GraphQLSchema, KnowledgeAPIFactory,
    APIRequest, APIResponse, HTTPMethod
)
from enhanced_knowledge_core import KnowledgeType

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    timestamp: datetime
    cpu_percent: float
    memory_mb: float
    request_latency_ms: float
    requests_per_second: float
    active_connections: int
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "cpu_percent": self.cpu_percent,
            "memory_mb": self.memory_mb,
            "request_latency_ms": self.request_latency_ms,
            "requests_per_second": self.requests_per_second,
            "active_connections": self.active_connections,
            "cache_hit_rate": self.cache_hit_rate
        }


class PerformanceMonitor:
    """Monitor platform performance and auto-scale if needed."""
    
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self._running = False
        self._monitoring_task: Optional[asyncio.Task] = None
        self._callbacks: List[Callable[[PerformanceMetrics], None]] = []
        
    async def start(self):
        """Start performance monitoring."""
        self._running = True
        self._monitoring_task = asyncio.create_task(self._monitor_loop())
        logger.info("Performance monitoring started")
    
    async def stop(self):
        """Stop performance monitoring."""
        self._running = False
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        logger.info("Performance monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                metrics = await self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only last 1000 metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-1000:]
                
                # Notify callbacks
                for callback in self._callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(metrics)
                        else:
                            callback(metrics)
                    except Exception as e:
                        logger.error(f"Metrics callback error: {e}")
                
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics."""
        # In a real implementation, this would use psutil or similar
        # For now, return simulated metrics
        
        import random
        return PerformanceMetrics(
            timestamp=datetime.utcnow(),
            cpu_percent=random.uniform(10, 60),
            memory_mb=random.uniform(100, 500),
            request_latency_ms=random.uniform(10, 100),
            requests_per_second=random.uniform(10, 1000),
            active_connections=random.randint(1, 100),
            cache_hit_rate=random.uniform(0.7, 0.95)
        )
    
    def on_metrics(self, callback: Callable[[PerformanceMetrics], None]):
        """Register metrics callback."""
        self._callbacks.append(callback)
    
    def get_average_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """Get average metrics over a time period."""
        cutoff = datetime.utcnow() - timedelta(minutes=minutes)
        recent = [m for m in self.metrics_history if m.timestamp > cutoff]
        
        if not recent:
            return {}
        
        return {
            "avg_cpu": sum(m.cpu_percent for m in recent) / len(recent),
            "avg_memory_mb": sum(m.memory_mb for m in recent) / len(recent),
            "avg_latency_ms": sum(m.request_latency_ms for m in recent) / len(recent),
            "avg_rps": sum(m.requests_per_second for m in recent) / len(recent),
            "avg_cache_hit_rate": sum(m.cache_hit_rate for m in recent) / len(recent)
        }
    
    def should_scale_up(self) -> bool:
        """Determine if system should scale up."""
        if len(self.metrics_history) < 10:
            return False
        
        recent = self.metrics_history[-10:]
        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_latency = sum(m.request_latency_ms for m in recent) / len(recent)
        
        return avg_cpu > 80 or avg_latency > 200


class KnowledgeVersioning:
    """Version control for knowledge items."""
    
    def __init__(self):
        self._versions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    
    def create_version(
        self,
        item_id: str,
        content: Any,
        author_id: str,
        change_message: str = ""
    ) -> Dict[str, Any]:
        """Create a new version of a knowledge item."""
        version = {
            "version_id": str(uuid.uuid4()),
            "version_number": len(self._versions[item_id]) + 1,
            "content": content,
            "author_id": author_id,
            "change_message": change_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self._versions[item_id].append(version)
        
        return version
    
    def get_version(self, item_id: str, version_number: int) -> Optional[Dict[str, Any]]:
        """Get a specific version."""
        versions = self._versions.get(item_id, [])
        if 1 <= version_number <= len(versions):
            return versions[version_number - 1]
        return None
    
    def get_version_history(self, item_id: str) -> List[Dict[str, Any]]:
        """Get version history for an item."""
        return self._versions.get(item_id, [])
    
    def diff_versions(
        self,
        item_id: str,
        version1: int,
        version2: int
    ) -> Dict[str, Any]:
        """Get diff between two versions."""
        v1 = self.get_version(item_id, version1)
        v2 = self.get_version(item_id, version2)
        
        if not v1 or not v2:
            return {"error": "Version not found"}
        
        # Simple diff
        content1 = str(v1["content"])
        content2 = str(v2["content"])
        
        return {
            "version1": version1,
            "version2": version2,
            "added": len(content2) - len(content1) if len(content2) > len(content1) else 0,
            "removed": len(content1) - len(content2) if len(content1) > len(content2) else 0,
            "content1_preview": content1[:100] + "..." if len(content1) > 100 else content1,
            "content2_preview": content2[:100] + "..." if len(content2) > 100 else content2
        }
    
    def revert_to_version(
        self,
        item_id: str,
        version_number: int,
        author_id: str
    ) -> Optional[Dict[str, Any]]:
        """Revert to a previous version."""
        target_version = self.get_version(item_id, version_number)
        if not target_version:
            return None
        
        # Create new version with reverted content
        return self.create_version(
            item_id,
            target_version["content"],
            author_id,
            f"Reverted to version {version_number}"
        )


class ImportExportManager:
    """Import and export knowledge data."""
    
    def __init__(self):
        self._exporters: Dict[str, Callable] = {}
        self._importers: Dict[str, Callable] = {}
    
    def register_exporter(self, format: str, handler: Callable):
        """Register an export handler."""
        self._exporters[format] = handler
    
    def register_importer(self, format: str, handler: Callable):
        """Register an import handler."""
        self._importers[format] = handler
    
    async def export_data(
        self,
        items: List[Any],
        format: str = "json",
        include_metadata: bool = True
    ) -> bytes:
        """Export knowledge items to a format."""
        if format == "json":
            data = {
                "export_date": datetime.utcnow().isoformat(),
                "item_count": len(items),
                "items": [
                    {
                        "id": item.id,
                        "content": item.content,
                        "type": item.knowledge_type.value,
                        "created_at": item.created_at.isoformat(),
                        "metadata": item.metadata if include_metadata else {}
                    }
                    for item in items
                ]
            }
            return json.dumps(data, indent=2).encode('utf-8')
        
        elif format == "csv":
            # Simple CSV export
            lines = ["id,type,content,created_at"]
            for item in items:
                content = str(item.content).replace('"', '""').replace("\n", " ")
                lines.append(f'"{item.id}","{item.knowledge_type.value}","{content}","{item.created_at.isoformat()}"')
            return "\n".join(lines).encode('utf-8')
        
        else:
            handler = self._exporters.get(format)
            if handler:
                return await handler(items)
            raise ValueError(f"Unsupported export format: {format}")
    
    async def import_data(
        self,
        data: bytes,
        format: str = "json",
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import knowledge data from a format."""
        if format == "json":
            imported = json.loads(data.decode('utf-8'))
            return {
                "imported_count": len(imported.get("items", [])),
                "source": imported.get("export_date", "unknown"),
                "items": imported.get("items", [])
            }
        
        else:
            handler = self._importers.get(format)
            if handler:
                return await handler(data)
            raise ValueError(f"Unsupported import format: {format}")


class CompleteKnowledgePlatform:
    """
    Complete knowledge platform with all enhancements.
    
    This is the final, fully-featured knowledge platform that combines:
    - Core knowledge engine
    - Distributed coordination
    - Real-time collaboration
    - ML intelligence
    - Workflow automation
    - Security layer
    - NLP processing
    - Multi-tenancy
    - Backup/recovery
    - API gateway
    - Performance monitoring
    - Version control
    - Import/export
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        storage_path: str = "./knowledge_data",
        enable_all: bool = True
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.storage_path = Path(storage_path)
        self.initialized_at = datetime.utcnow()
        
        # Core platform
        self.platform = UnifiedKnowledgePlatform(
            node_id=self.node_id,
            storage_path=str(self.storage_path / "core"),
            enable_distributed=enable_all,
            enable_collaboration=enable_all,
            enable_ml=enable_all,
            enable_workflows=enable_all,
            enable_security=enable_all
        )
        
        # Additional components
        self.nlp = NLPEngine() if enable_all else None
        self.tenant_manager = TenantManager() if enable_all else None
        
        # Backup/Recovery
        backup_storage = LocalBackupStorage(str(self.storage_path / "backups"))
        self.backup_engine = BackupEngine(backup_storage) if enable_all else None
        self.disaster_recovery = DisasterRecovery(self.backup_engine) if enable_all else None
        
        # API
        self.rest_api = KnowledgeAPIFactory.create_rest_api(self.platform) if enable_all else None
        self.graphql_schema = KnowledgeAPIFactory.create_graphql_schema(self.platform) if enable_all else None
        
        # Additional features
        self.performance_monitor = PerformanceMonitor() if enable_all else None
        self.versioning = KnowledgeVersioning() if enable_all else None
        self.import_export = ImportExportManager() if enable_all else None
        
        self._running = False
    
    async def initialize(self):
        """Initialize all components."""
        logger.info(f"Initializing CompleteKnowledgePlatform node {self.node_id}")
        
        # Initialize core platform
        await self.platform.initialize()
        
        # Initialize workflow engine if present
        if self.platform.workflow_engine:
            await self.platform.workflow_engine.start()
        
        # Initialize performance monitoring
        if self.performance_monitor:
            await self.performance_monitor.start()
        
        self._running = True
        logger.info("CompleteKnowledgePlatform fully initialized")
    
    async def shutdown(self):
        """Shutdown all components."""
        logger.info("Shutting down CompleteKnowledgePlatform")
        
        self._running = False
        
        if self.performance_monitor:
            await self.performance_monitor.stop()
        
        if self.platform.workflow_engine:
            await self.platform.workflow_engine.stop()
        
        await self.platform.shutdown()
        
        logger.info("CompleteKnowledgePlatform shutdown complete")
    
    # ==================== Enhanced Operations ====================
    
    async def add_knowledge_with_nlp(
        self,
        content: str,
        user_id: str,
        tenant_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add knowledge with full NLP analysis."""
        # NLP analysis
        nlp_results = None
        if self.nlp:
            nlp_results = self.nlp.analyze(content)
        
        # Add to platform
        item, ml_analysis = await self.platform.add_knowledge(
            content=content,
            knowledge_type=KnowledgeType.TEXT,
            tags=set(nlp_results.keywords[:5]) if nlp_results else set(),
            metadata={
                "author_id": user_id,
                "tenant_id": tenant_id,
                "nlp_analysis": nlp_results.to_dict() if nlp_results else None
            },
            user_id=user_id
        )
        
        # Create version
        if self.versioning:
            self.versioning.create_version(item.id, content, user_id, "Initial version")
        
        return {
            "item": item.to_dict() if hasattr(item, 'to_dict') else str(item),
            "nlp_analysis": nlp_results.to_dict() if nlp_results else None,
            "ml_analysis": ml_analysis
        }
    
    async def search_with_nlp(
        self,
        query: str,
        user_id: str,
        use_semantic: bool = True
    ) -> Dict[str, Any]:
        """Search with NLP query understanding."""
        # Analyze query
        query_analysis = None
        if self.nlp:
            query_analysis = self.nlp.analyze(query)
        
        # Perform search
        results = await self.platform.search(
            query=query,
            user_id=user_id,
            search_mode="semantic" if use_semantic else "keyword"
        )
        
        return {
            "query": query,
            "query_analysis": query_analysis.to_dict() if query_analysis else None,
            "results_count": len(results),
            "results": [r.to_dict() if hasattr(r, 'to_dict') else str(r) for r in results]
        }
    
    def create_tenant(
        self,
        name: str,
        slug: str,
        owner_id: str,
        plan: str = "basic"
    ) -> Optional[Any]:
        """Create a new tenant."""
        if not self.tenant_manager:
            return None
        
        return self.tenant_manager.create_tenant(
            name=name,
            slug=slug,
            owner_id=owner_id,
            plan=plan
        )
    
    async def backup(
        self,
        backup_type: str = "full",
        include_tenants: Optional[List[str]] = None
    ) -> Optional[Any]:
        """Create a backup."""
        if not self.backup_engine:
            return None
        
        from backup_recovery import BackupType
        
        bt = BackupType(backup_type)
        
        return await self.backup_engine.create_backup(
            source_path=str(self.storage_path),
            backup_type=bt
        )
    
    async def export(
        self,
        tenant_id: Optional[str] = None,
        format: str = "json"
    ) -> bytes:
        """Export knowledge data."""
        if not self.import_export:
            return b"{}"
        
        # Get items from platform
        items = list(self.platform.knowledge_engine._items.values())
        
        # Filter by tenant if specified
        if tenant_id:
            items = [
                item for item in items
                if item.metadata.get("tenant_id") == tenant_id
            ]
        
        return await self.import_export.export_data(items, format)
    
    # ==================== API Handlers ====================
    
    async def handle_api_request(self, request: APIRequest) -> APIResponse:
        """Handle API request."""
        if not self.rest_api:
            return APIResponse(
                status_code=503,
                error="API not available"
            )
        
        return await self.rest_api.handle_request(request)
    
    # ==================== Monitoring ====================
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics."""
        stats = {
            "node_id": self.node_id,
            "uptime_seconds": (datetime.utcnow() - self.initialized_at).total_seconds(),
            "core_platform": self.platform.get_platform_stats(),
            "components": {
                "nlp": self.nlp is not None,
                "multi_tenant": self.tenant_manager is not None,
                "backup": self.backup_engine is not None,
                "api": self.rest_api is not None,
                "monitoring": self.performance_monitor is not None,
                "versioning": self.versioning is not None
            }
        }
        
        if self.performance_monitor:
            stats["performance"] = self.performance_monitor.get_average_metrics()
        
        if self.tenant_manager:
            stats["tenants"] = {
                "total": len(self.tenant_manager.tenants),
                "active": sum(1 for t in self.tenant_manager.tenants.values() if t.status == "active")
            }
        
        if self.backup_engine:
            stats["backups"] = self.backup_engine.get_backup_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check."""
        health = self.platform.health_check()
        
        # Check additional components
        if self.nlp:
            health["components"]["nlp"] = {"status": "healthy"}
        
        if self.tenant_manager:
            health["components"]["multi_tenant"] = {"status": "healthy"}
        
        if self.backup_engine:
            backup_stats = self.backup_engine.get_backup_stats()
            health["components"]["backup"] = {
                "status": "healthy",
                "backups_count": backup_stats["total_backups"]
            }
        
        # Overall health
        component_statuses = [
            c.get("status", "unknown") 
            for c in health["components"].values()
        ]
        
        if any(s == "unhealthy" for s in component_statuses):
            health["status"] = "degraded"
        if all(s == "unhealthy" for s in component_statuses):
            health["status"] = "critical"
        
        return health


# Factory function
async def create_complete_platform(
    node_id: Optional[str] = None,
    storage_path: str = "./knowledge_data",
    **kwargs
) -> CompleteKnowledgePlatform:
    """Factory to create and initialize complete platform."""
    platform = CompleteKnowledgePlatform(
        node_id=node_id,
        storage_path=storage_path,
        **kwargs
    )
    await platform.initialize()
    return platform


__all__ = [
    "CompleteKnowledgePlatform",
    "create_complete_platform",
    "PerformanceMonitor",
    "KnowledgeVersioning",
    "ImportExportManager"
]
