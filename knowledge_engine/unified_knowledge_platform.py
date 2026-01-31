"""
Unified Knowledge Platform

Main integration layer that combines all enhanced knowledge engine components:
- Enhanced knowledge engine core
- Distributed coordination
- Real-time collaboration
- ML intelligence
- Workflow automation
- Security layer

Provides a single, unified interface for the complete knowledge platform.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import uuid

# Import all components
from enhanced_knowledge_engine import EnhancedKnowledgeEngine
from distributed_coordination import DistributedKnowledgeCoordinator
from realtime_collaboration import RealtimeCollaborationServer, CollaborationEvent
from ml_intelligence import MLIntelligenceEngine
from workflow_automation import WorkflowEngine, TriggerType, ActionType, Trigger, Action
from security_layer import (
    SecurityManager, Permission, EncryptionLevel,
    User, AccessPolicy
)
from enhanced_knowledge_core import KnowledgeType, KnowledgeItem, SearchResult

logger = logging.getLogger(__name__)


class UnifiedKnowledgePlatform:
    """
    Unified interface for the complete knowledge platform.
    
    Integrates all enhanced features:
    - Knowledge management (CRUD, search)
    - Distributed consensus
    - Real-time collaboration
    - ML-powered insights
    - Workflow automation
    - Security & access control
    """
    
    def __init__(
        self,
        node_id: Optional[str] = None,
        address: str = "localhost",
        port: int = 8080,
        peers: Optional[List[Tuple[str, str, int]]] = None,
        storage_path: Optional[str] = None,
        enable_distributed: bool = True,
        enable_collaboration: bool = True,
        enable_ml: bool = True,
        enable_workflows: bool = True,
        enable_security: bool = True,
        master_key: Optional[str] = None
    ):
        self.node_id = node_id or str(uuid.uuid4())
        self.initialized_at = datetime.utcnow()
        
        # Core engine
        self.knowledge_engine = EnhancedKnowledgeEngine(
            storage_path=storage_path,
            enable_graph=True,
            enable_learning=True
        )
        
        # Distributed coordination
        self.distributed_coordinator: Optional[DistributedKnowledgeCoordinator] = None
        if enable_distributed:
            self.distributed_coordinator = DistributedKnowledgeCoordinator(
                node_id=self.node_id,
                address=address,
                port=port,
                peers=peers or [],
                data_dir=f"{storage_path}/raft" if storage_path else None
            )
        
        # Real-time collaboration
        self.collaboration_server: Optional[RealtimeCollaborationServer] = None
        if enable_collaboration:
            self.collaboration_server = RealtimeCollaborationServer()
        
        # ML intelligence
        self.ml_engine: Optional[MLIntelligenceEngine] = None
        if enable_ml:
            self.ml_engine = MLIntelligenceEngine()
        
        # Workflow automation
        self.workflow_engine: Optional[WorkflowEngine] = None
        if enable_workflows:
            self.workflow_engine = WorkflowEngine()
        
        # Security
        self.security_manager: Optional[SecurityManager] = None
        if enable_security:
            self.security_manager = SecurityManager(master_key=master_key)
        
        self._running = False
        
    async def initialize(self):
        """Initialize all platform components."""
        logger.info(f"Initializing UnifiedKnowledgePlatform node {self.node_id}")
        
        # Initialize core engine
        await self.knowledge_engine.initialize()
        
        # Initialize distributed coordination
        if self.distributed_coordinator:
            await self.distributed_coordinator.start()
            logger.info("Distributed coordination initialized")
        
        # Initialize collaboration server
        if self.collaboration_server:
            await self.collaboration_server.start()
            # Wire collaboration to knowledge engine events
            self.knowledge_engine.add_event_handler(self._on_knowledge_event)
            logger.info("Real-time collaboration initialized")
        
        # Initialize workflow engine
        if self.workflow_engine:
            await self.workflow_engine.start()
            # Wire workflow engine to knowledge events
            self.knowledge_engine.add_event_handler(self._on_knowledge_event_for_workflows)
            logger.info("Workflow automation initialized")
        
        self._running = True
        logger.info("UnifiedKnowledgePlatform fully initialized")
    
    async def shutdown(self):
        """Shutdown all platform components."""
        logger.info("Shutting down UnifiedKnowledgePlatform")
        
        self._running = False
        
        if self.workflow_engine:
            await self.workflow_engine.stop()
        
        if self.collaboration_server:
            await self.collaboration_server.stop()
        
        if self.distributed_coordinator:
            await self.distributed_coordinator.stop()
        
        await self.knowledge_engine.shutdown()
        
        logger.info("UnifiedKnowledgePlatform shutdown complete")
    
    # ==================== Knowledge Operations ====================
    
    async def add_knowledge(
        self,
        content: Any,
        knowledge_type: KnowledgeType = KnowledgeType.TEXT,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None,
        source: str = "unknown",
        confidence: float = 1.0,
        user_id: Optional[str] = None
    ) -> Tuple[KnowledgeItem, Dict[str, Any]]:
        """
        Add knowledge with full platform integration.
        
        Returns:
            Tuple of (knowledge_item, ml_analysis)
        """
        # Check permissions if security enabled
        if self.security_manager and user_id:
            has_permission, reason = self.security_manager.access_control.check_permission(
                user_id, "global", Permission.WRITE
            )
            if not has_permission:
                raise PermissionError(f"Access denied: {reason}")
        
        # Submit to distributed coordinator if enabled
        if self.distributed_coordinator:
            try:
                await self.distributed_coordinator.submit_knowledge_add(
                    str(content), metadata or {}
                )
            except Exception as e:
                logger.warning(f"Distributed submission failed: {e}")
        
        # Add to knowledge engine
        item = await self.knowledge_engine.add_knowledge(
            content=content,
            knowledge_type=knowledge_type,
            metadata=metadata,
            tags=tags,
            source=source,
            confidence=confidence
        )
        
        # ML analysis
        ml_analysis = {}
        if self.ml_engine and isinstance(content, str):
            ml_analysis = await self.ml_engine.analyze_content(content)
            
            # Auto-tag if no tags provided
            if not tags and ml_analysis.get("tags"):
                for tag in ml_analysis["tags"]:
                    item.add_tag(tag)
            
            # Add to duplicate detection
            self.ml_engine.add_item_for_dedup(item.id, content)
            
            # Add embedding for recommendations
            if item.embedding:
                self.ml_engine.add_item_embedding(item.id, item.embedding.vector.tolist())
        
        # Set access policy if security enabled
        if self.security_manager and user_id:
            self.security_manager.access_control.create_access_policy(
                item_id=item.id,
                owner_id=user_id,
                encryption_level=EncryptionLevel.STANDARD
            )
        
        return item, ml_analysis
    
    async def search(
        self,
        query: str,
        user_id: Optional[str] = None,
        search_mode: str = "hybrid",
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10
    ) -> List[SearchResult]:
        """
        Search with permission checking and ML-enhanced ranking.
        """
        # Perform search
        results = await self.knowledge_engine.search(
            query=query,
            search_mode=search_mode,
            filters=filters,
            max_results=max_results * 2  # Get more for filtering
        )
        
        # Filter by permissions if security enabled
        if self.security_manager and user_id:
            filtered_results = []
            for result in results:
                has_permission, _ = self.security_manager.access_control.check_permission(
                    user_id, result.item.id, Permission.READ
                )
                if has_permission:
                    filtered_results.append(result)
            results = filtered_results
        
        # ML-enhanced ranking if enabled
        if self.ml_engine and user_id:
            # Get personalized recommendations
            recommendations = self.ml_engine.get_recommendations(
                user_id=user_id,
                num_recommendations=max_results
            )
            
            # Boost scores for recommended items
            rec_ids = {r.item_id for r in recommendations}
            for result in results:
                if result.item.id in rec_ids:
                    result.relevance_score *= 1.2  # 20% boost
            
            # Re-sort
            results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Record search interaction for recommendations
        if self.ml_engine and user_id:
            self.ml_engine.record_interaction(user_id, "search_query", "search")
        
        return results[:max_results]
    
    async def update_knowledge(
        self,
        item_id: str,
        new_content: Any,
        user_id: Optional[str] = None,
        confidence: Optional[float] = None
    ) -> Optional[KnowledgeItem]:
        """Update knowledge with security and workflow integration."""
        # Check permissions
        if self.security_manager and user_id:
            has_permission, reason = self.security_manager.access_control.check_permission(
                user_id, item_id, Permission.WRITE
            )
            if not has_permission:
                raise PermissionError(f"Access denied: {reason}")
        
        # Log audit event
        if self.security_manager:
            self.security_manager.audit_logger.log_event(
                user_id=user_id,
                action="update",
                resource_type="knowledge_item",
                resource_id=item_id,
                status="success"
            )
        
        return await self.knowledge_engine.update_knowledge(
            item_id, new_content, confidence
        )
    
    async def delete_knowledge(
        self,
        item_id: str,
        user_id: Optional[str] = None
    ) -> bool:
        """Delete knowledge with security checks."""
        # Check permissions
        if self.security_manager and user_id:
            has_permission, reason = self.security_manager.access_control.check_permission(
                user_id, item_id, Permission.DELETE
            )
            if not has_permission:
                raise PermissionError(f"Access denied: {reason}")
        
        # Log audit event
        if self.security_manager:
            self.security_manager.audit_logger.log_event(
                user_id=user_id,
                action="delete",
                resource_type="knowledge_item",
                resource_id=item_id,
                status="success"
            )
        
        return await self.knowledge_engine.delete_knowledge(item_id)
    
    # ==================== ML Intelligence ====================
    
    async def analyze_content(
        self,
        content: str,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze content using ML."""
        if not self.ml_engine:
            return {"error": "ML engine not enabled"}
        
        return await self.ml_engine.analyze_content(content, title)
    
    def get_recommendations(
        self,
        user_id: Optional[str] = None,
        item_id: Optional[str] = None,
        num_recommendations: int = 5
    ) -> List[Any]:
        """Get personalized recommendations."""
        if not self.ml_engine:
            return []
        
        return self.ml_engine.get_recommendations(user_id, item_id, num_recommendations)
    
    # ==================== Workflow Automation ====================
    
    def create_workflow(
        self,
        name: str,
        description: str,
        triggers: List[Trigger],
        actions: List[Action]
    ) -> Any:
        """Create a workflow."""
        if not self.workflow_engine:
            raise RuntimeError("Workflow engine not enabled")
        
        return self.workflow_engine.create_workflow(name, description, triggers, actions)
    
    async def trigger_workflow(
        self,
        event_type: str,
        event_data: Dict[str, Any]
    ):
        """Manually trigger workflows."""
        if self.workflow_engine:
            await self.workflow_engine.process_event({
                "type": event_type,
                "data": event_data
            })
    
    # ==================== Security ====================
    
    def create_user(
        self,
        username: str,
        email: str,
        roles: Optional[List[str]] = None,
        is_admin: bool = False
    ) -> User:
        """Create a user."""
        if not self.security_manager:
            raise RuntimeError("Security not enabled")
        
        return self.security_manager.access_control.create_user(
            username, email, roles, is_admin
        )
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate a user."""
        if not self.security_manager:
            return None
        
        # Hash password and authenticate
        password_hash = self.security_manager.encryption.hash_sensitive(password)
        return self.security_manager.access_control.authenticate_user(
            username, password_hash
        )
    
    def get_security_audit(self, days: int = 30) -> Dict[str, Any]:
        """Get security audit report."""
        if not self.security_manager:
            return {"error": "Security not enabled"}
        
        return self.security_manager.get_security_audit(days)
    
    # ==================== Real-time Collaboration ====================
    
    async def client_connected(
        self,
        session_id: str,
        user_id: str,
        user_name: str,
        connection: Any
    ):
        """Handle client connection for collaboration."""
        if self.collaboration_server:
            await self.collaboration_server.client_connected(
                session_id, user_id, user_name, connection
            )
    
    async def client_disconnected(self, session_id: str):
        """Handle client disconnection."""
        if self.collaboration_server:
            await self.collaboration_server.client_disconnected(session_id)
    
    # ==================== Event Handlers ====================
    
    def _on_knowledge_event(self, event):
        """Handle knowledge events for collaboration."""
        if not self.collaboration_server:
            return
        
        # Map knowledge event to collaboration event
        event_type_map = {
            "created": CollaborationEvent,
            "updated": CollaborationEvent,
            "deleted": CollaborationEvent
        }
        
        # Broadcast to collaborators
        # This is simplified - real implementation would need proper event mapping
        pass
    
    def _on_knowledge_event_for_workflows(self, event):
        """Handle knowledge events for workflows."""
        if not self.workflow_engine:
            return
        
        # Map to workflow event and process
        event_type_map = {
            "created": TriggerType.KNOWLEDGE_CREATED,
            "updated": TriggerType.KNOWLEDGE_UPDATED,
            "deleted": TriggerType.KNOWLEDGE_DELETED
        }
        
        workflow_event = {
            "type": event_type_map.get(event.event_type, event.event_type).value,
            "data": {
                "item_id": event.item_id,
                **event.data
            }
        }
        
        # Process asynchronously
        asyncio.create_task(self.workflow_engine.process_event(workflow_event))
    
    # ==================== Platform Stats ====================
    
    def get_platform_stats(self) -> Dict[str, Any]:
        """Get comprehensive platform statistics."""
        stats = {
            "node_id": self.node_id,
            "uptime_seconds": (datetime.utcnow() - self.initialized_at).total_seconds(),
            "knowledge_engine": self.knowledge_engine.get_stats(),
            "components": {
                "distributed": self.distributed_coordinator is not None,
                "collaboration": self.collaboration_server is not None,
                "ml": self.ml_engine is not None,
                "workflows": self.workflow_engine is not None,
                "security": self.security_manager is not None
            }
        }
        
        if self.workflow_engine:
            stats["workflows"] = self.workflow_engine.get_all_stats()
        
        if self.distributed_coordinator:
            stats["distributed"] = self.distributed_coordinator.get_stats()
        
        if self.collaboration_server:
            stats["collaboration"] = self.collaboration_server.get_stats()
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components."""
        health = {
            "status": "healthy",
            "node_id": self.node_id,
            "timestamp": datetime.utcnow().isoformat(),
            "components": {}
        }
        
        # Check knowledge engine
        ke_health = self.knowledge_engine.get_health_check()
        health["components"]["knowledge_engine"] = ke_health
        
        # Check other components
        if self.distributed_coordinator:
            raft_stats = self.distributed_coordinator.raft.get_stats()
            health["components"]["distributed"] = {
                "status": "healthy" if raft_stats["state"] != "error" else "unhealthy",
                "state": raft_stats["state"]
            }
        
        if self.collaboration_server:
            collab_stats = self.collaboration_server.get_stats()
            health["components"]["collaboration"] = {
                "status": "healthy",
                "connected_clients": collab_stats["connected_clients"]
            }
        
        if self.workflow_engine:
            health["components"]["workflows"] = {
                "status": "healthy",
                "active_workflows": len(self.workflow_engine.workflows)
            }
        
        # Overall status
        component_statuses = [c["status"] for c in health["components"].values()]
        if any(s == "unhealthy" for s in component_statuses):
            health["status"] = "degraded"
        
        return health


# Factory function
async def create_unified_platform(
    node_id: Optional[str] = None,
    address: str = "localhost",
    port: int = 8080,
    storage_path: Optional[str] = None,
    **kwargs
) -> UnifiedKnowledgePlatform:
    """Factory function to create and initialize a unified platform."""
    platform = UnifiedKnowledgePlatform(
        node_id=node_id,
        address=address,
        port=port,
        storage_path=storage_path,
        **kwargs
    )
    await platform.initialize()
    return platform


__all__ = [
    "UnifiedKnowledgePlatform",
    "create_unified_platform"
]
