"""
OpenEvolve Integration Module

Integrates Knowledge Engine with OpenEvolve for project context injection,
real-time updates, and multi-project support.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)


class ProjectLifecycleStage(Enum):
    """Stages in a project lifecycle"""
    INITIALIZED = "initialized"
    PLANNING = "planning"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    COMPLETED = "completed"
    ARCHIVED = "archived"


@dataclass
class ProjectContext:
    """Context for an OpenEvolve project"""
    project_id: str
    name: str
    description: str = ""
    stage: ProjectLifecycleStage = ProjectLifecycleStage.INITIALIZED
    metadata: Dict[str, Any] = field(default_factory=dict)
    team_members: List[str] = field(default_factory=list)
    workflows: List[str] = field(default_factory=list)
    knowledge_graph_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "description": self.description,
            "stage": self.stage.value,
            "metadata": self.metadata,
            "team_members": self.team_members,
            "workflows": self.workflows,
            "knowledge_graph_id": self.knowledge_graph_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass
class ContextUpdate:
    """An update to project context"""
    project_id: str
    update_type: str
    data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    source: str = ""


class OpenEvolveIntegration:
    """
    Integration between Knowledge Engine and OpenEvolve
    
    Provides:
    - Project context injection into queries
    - Real-time updates from OpenEvolve
    - Multi-project support
    - Project lifecycle hooks
    """
    
    def __init__(self, knowledge_engine=None):
        self.ke = knowledge_engine
        self.projects: Dict[str, ProjectContext] = {}
        self.active_project: Optional[str] = None
        self.update_handlers: List[Callable] = []
        self.lifecycle_hooks: Dict[str, List[Callable]] = {
            stage.value: [] for stage in ProjectLifecycleStage
        }
        self._update_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
        # Event subscriptions
        self._subscribers: Dict[str, List[Callable]] = {}
    
    def register_project(self, context: ProjectContext) -> str:
        """Register a new project with the Knowledge Engine"""
        self.projects[context.project_id] = context
        
        # Create knowledge graph for project
        if self.ke:
            kg_name = f"project_{context.project_id}"
            # Would create KG here
            context.knowledge_graph_id = kg_name
        
        logger.info(f"Registered project: {context.name} ({context.project_id})")
        
        # Trigger lifecycle hook
        self._trigger_lifecycle_hook(ProjectLifecycleStage.INITIALIZED, context)
        
        return context.project_id
    
    def get_project(self, project_id: str) -> Optional[ProjectContext]:
        """Get project by ID"""
        return self.projects.get(project_id)
    
    def set_active_project(self, project_id: str):
        """Set the active project for queries"""
        if project_id not in self.projects:
            raise ValueError(f"Project {project_id} not found")
        
        self.active_project = project_id
        logger.info(f"Active project set to: {project_id}")
    
    def get_active_project(self) -> Optional[ProjectContext]:
        """Get the currently active project"""
        if self.active_project:
            return self.projects.get(self.active_project)
        return None
    
    def inject_context(self, query: str, project_id: Optional[str] = None) -> str:
        """
        Inject project context into a query
        
        This enriches queries with project-specific information
        to improve relevance of results.
        """
        project = self.get_project(project_id or self.active_project)
        
        if not project:
            return query
        
        # Build context string
        context_parts = [
            f"Project: {project.name}",
            f"Stage: {project.stage.value}",
        ]
        
        if project.description:
            context_parts.append(f"Description: {project.description}")
        
        if project.metadata:
            # Add relevant metadata
            for key in ['domain', 'technology', 'focus']:
                if key in project.metadata:
                    context_parts.append(f"{key.capitalize()}: {project.metadata[key]}")
        
        context_str = " | ".join(context_parts)
        
        # Combine with query
        return f"[Context: {context_str}] {query}"
    
    async def start_realtime_updates(self):
        """Start processing real-time updates"""
        self._running = True
        
        while self._running:
            try:
                update = await asyncio.wait_for(
                    self._update_queue.get(),
                    timeout=1.0
                )
                await self._process_update(update)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing update: {e}")
    
    def stop_realtime_updates(self):
        """Stop processing real-time updates"""
        self._running = False
    
    async def _process_update(self, update: ContextUpdate):
        """Process a context update"""
        logger.debug(f"Processing update: {update.update_type} for {update.project_id}")
        
        # Get project
        project = self.get_project(update.project_id)
        if not project:
            logger.warning(f"Update for unknown project: {update.project_id}")
            return
        
        # Apply update based on type
        if update.update_type == "stage_change":
            new_stage = ProjectLifecycleStage(update.data.get('stage'))
            old_stage = project.stage
            project.stage = new_stage
            project.updated_at = datetime.utcnow()
            
            # Trigger lifecycle hook
            self._trigger_lifecycle_hook(new_stage, project)
            
            logger.info(f"Project {project.name} stage: {old_stage.value} -> {new_stage.value}")
        
        elif update.update_type == "metadata_update":
            project.metadata.update(update.data.get('metadata', {}))
            project.updated_at = datetime.utcnow()
        
        elif update.update_type == "member_added":
            member = update.data.get('member')
            if member and member not in project.team_members:
                project.team_members.append(member)
                project.updated_at = datetime.utcnow()
        
        elif update.update_type == "workflow_added":
            workflow = update.data.get('workflow')
            if workflow and workflow not in project.workflows:
                project.workflows.append(workflow)
                project.updated_at = datetime.utcnow()
        
        # Notify handlers
        for handler in self.update_handlers:
            try:
                await handler(update)
            except Exception as e:
                logger.error(f"Update handler failed: {e}")
        
        # Notify subscribers
        await self._notify_subscribers(update.project_id, update)
    
    def queue_update(self, update: ContextUpdate):
        """Queue an update for processing"""
        self._update_queue.put_nowait(update)
    
    def add_update_handler(self, handler: Callable):
        """Add a handler for context updates"""
        self.update_handlers.append(handler)
    
    def remove_update_handler(self, handler: Callable):
        """Remove an update handler"""
        if handler in self.update_handlers:
            self.update_handlers.remove(handler)
    
    def register_lifecycle_hook(
        self,
        stage: ProjectLifecycleStage,
        callback: Callable
    ):
        """Register a callback for a lifecycle stage"""
        self.lifecycle_hooks[stage.value].append(callback)
    
    def _trigger_lifecycle_hook(self, stage: ProjectLifecycleStage, project: ProjectContext):
        """Trigger callbacks for a lifecycle stage"""
        for callback in self.lifecycle_hooks.get(stage.value, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(project))
                else:
                    callback(project)
            except Exception as e:
                logger.error(f"Lifecycle hook failed: {e}")
    
    async def subscribe(self, project_id: str, callback: Callable):
        """Subscribe to updates for a specific project"""
        if project_id not in self._subscribers:
            self._subscribers[project_id] = []
        self._subscribers[project_id].append(callback)
    
    async def unsubscribe(self, project_id: str, callback: Callable):
        """Unsubscribe from updates"""
        if project_id in self._subscribers and callback in self._subscribers[project_id]:
            self._subscribers[project_id].remove(callback)
    
    async def _notify_subscribers(self, project_id: str, update: ContextUpdate):
        """Notify subscribers of an update"""
        for callback in self._subscribers.get(project_id, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(update)
                else:
                    callback(update)
            except Exception as e:
                logger.error(f"Subscriber notification failed: {e}")
    
    def export_project_context(self, project_id: str) -> Dict[str, Any]:
        """Export project context for external use"""
        project = self.get_project(project_id)
        if not project:
            return {}
        
        return {
            "project": project.to_dict(),
            "knowledge_summary": self._get_knowledge_summary(project_id),
        }
    
    def _get_knowledge_summary(self, project_id: str) -> Dict[str, Any]:
        """Get knowledge graph summary for a project"""
        # Would query actual knowledge graph here
        return {
            "entity_count": 0,
            "relation_count": 0,
            "last_updated": datetime.utcnow().isoformat()
        }
    
    def list_projects(self) -> List[ProjectContext]:
        """List all registered projects"""
        return list(self.projects.values())
    
    def archive_project(self, project_id: str):
        """Archive a project"""
        project = self.get_project(project_id)
        if project:
            project.stage = ProjectLifecycleStage.ARCHIVED
            self._trigger_lifecycle_hook(ProjectLifecycleStage.ARCHIVED, project)
            logger.info(f"Archived project: {project.name}")
    
    def delete_project(self, project_id: str):
        """Delete a project"""
        if project_id in self.projects:
            del self.projects[project_id]
            if self.active_project == project_id:
                self.active_project = None
            logger.info(f"Deleted project: {project_id}")
    
    def get_integration_status(self) -> Dict[str, Any]:
        """Get integration status"""
        return {
            "projects_count": len(self.projects),
            "active_project": self.active_project,
            "update_handlers": len(self.update_handlers),
            "lifecycle_hooks": {
                stage: len(hooks)
                for stage, hooks in self.lifecycle_hooks.items()
            },
            "realtime_running": self._running,
        }


class ProjectContextInjector:
    """
    Injects project context into various operations
    """
    
    def __init__(self, integration: OpenEvolveIntegration):
        self.integration = integration
    
    def inject_into_query(self, query: str) -> str:
        """Inject context into a search/query"""
        return self.integration.inject_context(query)
    
    def inject_into_prompt(self, prompt: str) -> str:
        """Inject context into an LLM prompt"""
        project = self.integration.get_active_project()
        
        if not project:
            return prompt
        
        context = f"""You are working on the project "{project.name}".
Current stage: {project.stage.value}
Description: {project.description or 'No description'}

"""
        
        return context + prompt
    
    def get_relevant_knowledge(self, topic: str) -> List[Dict[str, Any]]:
        """Get knowledge relevant to current project and topic"""
        project = self.integration.get_active_project()
        
        if not project or not self.integration.ke:
            return []
        
        # Query knowledge graph with project context
        enriched_query = f"Project: {project.name} {topic}"
        
        # Would query actual knowledge engine here
        return []
