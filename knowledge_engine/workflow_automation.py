"""
Workflow Automation Engine

Provides automation capabilities for the knowledge engine:
- Trigger-based actions
- Workflow definitions and execution
- Scheduled tasks
- Conditional logic
- Integration with external systems
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of workflow triggers."""
    # Knowledge events
    KNOWLEDGE_CREATED = "knowledge_created"
    KNOWLEDGE_UPDATED = "knowledge_updated"
    KNOWLEDGE_DELETED = "knowledge_deleted"
    KNOWLEDGE_VIEWED = "knowledge_viewed"
    
    # Time-based
    SCHEDULED = "scheduled"
    CRON = "cron"
    
    # External
    WEBHOOK = "webhook"
    API_CALL = "api_call"
    
    # Conditions
    CONDITION_MET = "condition_met"
    QUALITY_THRESHOLD = "quality_threshold"


class ActionType(Enum):
    """Types of workflow actions."""
    SEND_NOTIFICATION = "send_notification"
    UPDATE_KNOWLEDGE = "update_knowledge"
    CREATE_KNOWLEDGE = "create_knowledge"
    DELETE_KNOWLEDGE = "delete_knowledge"
    ADD_TAGS = "add_tags"
    REMOVE_TAGS = "remove_tags"
    CALL_WEBHOOK = "call_webhook"
    EXECUTE_SCRIPT = "execute_script"
    SEND_EMAIL = "send_email"
    CREATE_TASK = "create_task"
    UPDATE_METADATA = "update_metadata"
    ARCHIVE = "archive"


@dataclass
class Trigger:
    """Workflow trigger definition."""
    trigger_id: str
    trigger_type: TriggerType
    conditions: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def matches(self, event: Dict[str, Any]) -> bool:
        """Check if trigger matches an event."""
        if not self.enabled:
            return False
        
        # Check trigger type
        event_type = event.get("type")
        if self.trigger_type.value != event_type:
            return False
        
        # Check conditions
        for key, expected_value in self.conditions.items():
            actual_value = event.get("data", {}).get(key)
            if actual_value != expected_value:
                return False
        
        return True


@dataclass
class Action:
    """Workflow action definition."""
    action_id: str
    action_type: ActionType
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def execute(self, context: WorkflowContext) -> ActionResult:
        """Execute the action."""
        # This would dispatch to specific action handlers
        return ActionResult(
            success=True,
            action_id=self.action_id,
            output={}
        )


@dataclass
class ActionResult:
    """Result of action execution."""
    success: bool
    action_id: str
    output: Dict[str, Any]
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0


@dataclass
class WorkflowContext:
    """Context for workflow execution."""
    workflow_id: str
    execution_id: str
    trigger_event: Dict[str, Any]
    variables: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.utcnow)
    
    def get_variable(self, name: str, default: Any = None) -> Any:
        """Get a variable value."""
        return self.variables.get(name, default)
    
    def set_variable(self, name: str, value: Any):
        """Set a variable value."""
        self.variables[name] = value


@dataclass
class Workflow:
    """Workflow definition."""
    workflow_id: str
    name: str
    description: str
    triggers: List[Trigger]
    actions: List[Action]
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "created_at": self.created_at.isoformat(),
            "triggers": [
                {
                    "trigger_id": t.trigger_id,
                    "trigger_type": t.trigger_type.value,
                    "conditions": t.conditions,
                    "enabled": t.enabled
                }
                for t in self.triggers
            ],
            "actions": [
                {
                    "action_id": a.action_id,
                    "action_type": a.action_type.value,
                    "parameters": a.parameters
                }
                for a in self.actions
            ]
        }


@dataclass
class WorkflowExecution:
    """Record of a workflow execution."""
    execution_id: str
    workflow_id: str
    trigger_event: Dict[str, Any]
    status: str  # "running", "completed", "failed"
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: List[ActionResult] = field(default_factory=list)
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "workflow_id": self.workflow_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_message": self.error_message,
            "results": [
                {
                    "success": r.success,
                    "action_id": r.action_id,
                    "error_message": r.error_message
                }
                for r in self.results
            ]
        }


class ActionRegistry:
    """Registry of available actions."""
    
    def __init__(self):
        self._actions: Dict[ActionType, Callable[[Action, WorkflowContext], ActionResult]] = {}
        
    def register(
        self, 
        action_type: ActionType, 
        handler: Callable[[Action, WorkflowContext], ActionResult]
    ):
        """Register an action handler."""
        self._actions[action_type] = handler
        
    def execute(
        self, 
        action: Action, 
        context: WorkflowContext
    ) -> ActionResult:
        """Execute an action."""
        handler = self._actions.get(action.action_type)
        if handler:
            start_time = datetime.utcnow()
            try:
                result = handler(action, context)
                execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                result.execution_time_ms = execution_time
                return result
            except Exception as e:
                return ActionResult(
                    success=False,
                    action_id=action.action_id,
                    output={},
                    error_message=str(e),
                    execution_time_ms=(datetime.utcnow() - start_time).total_seconds() * 1000
                )
        else:
            return ActionResult(
                success=False,
                action_id=action.action_id,
                output={},
                error_message=f"No handler for action type: {action.action_type}"
            )


class Scheduler:
    """Task scheduler for scheduled workflows."""
    
    def __init__(self):
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._running = False
        
    async def start(self):
        """Start the scheduler."""
        self._running = True
        
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        for task in self._scheduled_tasks.values():
            task.cancel()
        
    def schedule(
        self, 
        task_id: str, 
        callback: Callable, 
        interval_seconds: float
    ):
        """Schedule a recurring task."""
        async def task_wrapper():
            while self._running:
                try:
                    await asyncio.sleep(interval_seconds)
                    if self._running:
                        await callback()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Scheduled task error: {e}")
        
        self._scheduled_tasks[task_id] = asyncio.create_task(task_wrapper())
        
    def cancel(self, task_id: str):
        """Cancel a scheduled task."""
        task = self._scheduled_tasks.pop(task_id, None)
        if task:
            task.cancel()


class WorkflowEngine:
    """
    Main workflow automation engine.
    """
    
    def __init__(self):
        self.workflows: Dict[str, Workflow] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.action_registry = ActionRegistry()
        self.scheduler = Scheduler()
        
        # Event handlers
        self._event_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        
        # Execution history
        self._execution_history: List[WorkflowExecution] = []
        
        # Register default actions
        self._register_default_actions()
        
    def _register_default_actions(self):
        """Register default action handlers."""
        # These would be implemented with actual functionality
        
        def handle_notification(action: Action, context: WorkflowContext) -> ActionResult:
            message = action.parameters.get("message", "")
            recipient = action.parameters.get("recipient", "")
            logger.info(f"Notification: {message} to {recipient}")
            return ActionResult(success=True, action_id=action.action_id, output={"sent": True})
        
        def handle_webhook(action: Action, context: WorkflowContext) -> ActionResult:
            url = action.parameters.get("url", "")
            method = action.parameters.get("method", "POST")
            logger.info(f"Webhook: {method} to {url}")
            return ActionResult(success=True, action_id=action.action_id, output={"status": 200})
        
        def handle_add_tags(action: Action, context: WorkflowContext) -> ActionResult:
            tags = action.parameters.get("tags", [])
            item_id = context.get_variable("item_id")
            logger.info(f"Adding tags {tags} to item {item_id}")
            return ActionResult(success=True, action_id=action.action_id, output={"tags_added": tags})
        
        def handle_update_metadata(action: Action, context: WorkflowContext) -> ActionResult:
            metadata = action.parameters.get("metadata", {})
            item_id = context.get_variable("item_id")
            logger.info(f"Updating metadata for item {item_id}: {metadata}")
            return ActionResult(success=True, action_id=action.action_id, output={"updated": True})
        
        self.action_registry.register(ActionType.SEND_NOTIFICATION, handle_notification)
        self.action_registry.register(ActionType.CALL_WEBHOOK, handle_webhook)
        self.action_registry.register(ActionType.ADD_TAGS, handle_add_tags)
        self.action_registry.register(ActionType.UPDATE_METADATA, handle_update_metadata)
    
    async def start(self):
        """Start the workflow engine."""
        await self.scheduler.start()
        logger.info("WorkflowEngine started")
        
    async def stop(self):
        """Stop the workflow engine."""
        await self.scheduler.stop()
        logger.info("WorkflowEngine stopped")
    
    def create_workflow(
        self,
        name: str,
        description: str,
        triggers: List[Trigger],
        actions: List[Action]
    ) -> Workflow:
        """
        Create a new workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            triggers: List of triggers
            actions: List of actions
            
        Returns:
            Created workflow
        """
        workflow = Workflow(
            workflow_id=str(uuid.uuid4()),
            name=name,
            description=description,
            triggers=triggers,
            actions=actions
        )
        
        self.workflows[workflow.workflow_id] = workflow
        
        # If any trigger is scheduled, set up scheduling
        for trigger in triggers:
            if trigger.trigger_type in (TriggerType.SCHEDULED, TriggerType.CRON):
                self._schedule_workflow(workflow, trigger)
        
        logger.info(f"Created workflow: {name} ({workflow.workflow_id})")
        
        return workflow
    
    def _schedule_workflow(self, workflow: Workflow, trigger: Trigger):
        """Set up scheduled execution for a workflow."""
        interval = trigger.conditions.get("interval_seconds", 3600)
        
        async def scheduled_execution():
            event = {
                "type": TriggerType.SCHEDULED.value,
                "data": {"scheduled_at": datetime.utcnow().isoformat()}
            }
            await self._execute_workflow(workflow, event)
        
        task_id = f"workflow_{workflow.workflow_id}"
        self.scheduler.schedule(task_id, scheduled_execution, interval)
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow."""
        if workflow_id in self.workflows:
            # Cancel any scheduled tasks
            task_id = f"workflow_{workflow_id}"
            self.scheduler.cancel(task_id)
            
            del self.workflows[workflow_id]
            logger.info(f"Deleted workflow: {workflow_id}")
            return True
        return False
    
    def enable_workflow(self, workflow_id: str) -> bool:
        """Enable a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = True
            return True
        return False
    
    def disable_workflow(self, workflow_id: str) -> bool:
        """Disable a workflow."""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            workflow.enabled = False
            return True
        return False
    
    async def process_event(self, event: Dict[str, Any]):
        """
        Process an event and trigger matching workflows.
        
        Args:
            event: Event dictionary with 'type' and 'data' keys
        """
        triggered_workflows = []
        
        for workflow in self.workflows.values():
            if not workflow.enabled:
                continue
            
            for trigger in workflow.triggers:
                if trigger.matches(event):
                    triggered_workflows.append(workflow)
                    break
        
        # Execute triggered workflows
        execution_tasks = [
            self._execute_workflow(workflow, event)
            for workflow in triggered_workflows
        ]
        
        if execution_tasks:
            await asyncio.gather(*execution_tasks, return_exceptions=True)
    
    async def _execute_workflow(
        self, 
        workflow: Workflow, 
        event: Dict[str, Any]
    ) -> WorkflowExecution:
        """
        Execute a workflow.
        
        Args:
            workflow: Workflow to execute
            event: Trigger event
            
        Returns:
            Execution record
        """
        execution_id = str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            workflow_id=workflow.workflow_id,
            trigger_event=event,
            status="running",
            started_at=datetime.utcnow()
        )
        
        self.executions[execution_id] = execution
        
        # Create context
        context = WorkflowContext(
            workflow_id=workflow.workflow_id,
            execution_id=execution_id,
            trigger_event=event,
            variables={"event_data": event.get("data", {})}
        )
        
        # Execute actions
        try:
            for action in workflow.actions:
                result = self.action_registry.execute(action, context)
                execution.results.append(result)
                
                if not result.success:
                    execution.status = "failed"
                    execution.error_message = result.error_message
                    break
            else:
                execution.status = "completed"
            
        except Exception as e:
            execution.status = "failed"
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {e}")
        
        execution.completed_at = datetime.utcnow()
        self._execution_history.append(execution)
        
        # Notify callbacks
        for callback in self._event_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(execution.to_dict())
                else:
                    callback(execution.to_dict())
            except Exception as e:
                logger.error(f"Workflow callback error: {e}")
        
        return execution
    
    def on_execution_complete(self, callback: Callable):
        """Register a callback for workflow execution completion."""
        self._event_callbacks.append(callback)
    
    def get_workflow_stats(self, workflow_id: str) -> Dict[str, Any]:
        """Get statistics for a workflow."""
        executions = [
            e for e in self._execution_history 
            if e.workflow_id == workflow_id
        ]
        
        if not executions:
            return {"executions": 0}
        
        total = len(executions)
        successful = sum(1 for e in executions if e.status == "completed")
        failed = sum(1 for e in executions if e.status == "failed")
        
        avg_duration = 0.0
        if executions:
            durations = [
                (e.completed_at - e.started_at).total_seconds()
                for e in executions
                if e.completed_at
            ]
            if durations:
                avg_duration = sum(durations) / len(durations)
        
        return {
            "executions": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0.0,
            "average_duration_seconds": avg_duration
        }
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get overall workflow statistics."""
        return {
            "total_workflows": len(self.workflows),
            "enabled_workflows": sum(1 for w in self.workflows.values() if w.enabled),
            "total_executions": len(self._execution_history),
            "workflows": {
                workflow_id: self.get_workflow_stats(workflow_id)
                for workflow_id in self.workflows
            }
        }


class WorkflowTemplates:
    """
    Pre-built workflow templates for common use cases.
    """
    
    @staticmethod
    def auto_tag_new_knowledge() -> Workflow:
        """Template: Automatically tag new knowledge items."""
        return Workflow(
            workflow_id="",
            name="Auto-tag New Knowledge",
            description="Automatically adds tags to new knowledge items based on content",
            triggers=[
                Trigger(
                    trigger_id="trigger-1",
                    trigger_type=TriggerType.KNOWLEDGE_CREATED
                )
            ],
            actions=[
                Action(
                    action_id="action-1",
                    action_type=ActionType.ADD_TAGS,
                    parameters={"tags": ["auto-tagged"]}
                )
            ]
        )
    
    @staticmethod
    def notify_on_high_quality() -> Workflow:
        """Template: Notify when high-quality knowledge is created."""
        return Workflow(
            workflow_id="",
            name="High Quality Notification",
            description="Sends notification when high-quality knowledge is created",
            triggers=[
                Trigger(
                    trigger_id="trigger-1",
                    trigger_type=TriggerType.KNOWLEDGE_CREATED,
                    conditions={"quality_score": ">=0.9"}
                )
            ],
            actions=[
                Action(
                    action_id="action-1",
                    action_type=ActionType.SEND_NOTIFICATION,
                    parameters={
                        "message": "High-quality knowledge item created!",
                        "recipient": "admin"
                    }
                )
            ]
        )
    
    @staticmethod
    def archive_old_knowledge() -> Workflow:
        """Template: Archive old knowledge items."""
        return Workflow(
            workflow_id="",
            name="Archive Old Knowledge",
            description="Automatically archives knowledge items older than 1 year",
            triggers=[
                Trigger(
                    trigger_id="trigger-1",
                    trigger_type=TriggerType.SCHEDULED,
                    conditions={"interval_seconds": 86400}  # Daily
                )
            ],
            actions=[
                Action(
                    action_id="action-1",
                    action_type=ActionType.ARCHIVE,
                    parameters={"age_days": 365}
                )
            ]
        )
    
    @staticmethod
    def webhook_on_update() -> Workflow:
        """Template: Call webhook on knowledge update."""
        return Workflow(
            workflow_id="",
            name="Webhook on Update",
            description="Calls external webhook when knowledge is updated",
            triggers=[
                Trigger(
                    trigger_id="trigger-1",
                    trigger_type=TriggerType.KNOWLEDGE_UPDATED
                )
            ],
            actions=[
                Action(
                    action_id="action-1",
                    action_type=ActionType.CALL_WEBHOOK,
                    parameters={
                        "url": "https://example.com/webhook",
                        "method": "POST"
                    }
                )
            ]
        )


__all__ = [
    "WorkflowEngine",
    "Workflow",
    "Trigger",
    "Action",
    "TriggerType",
    "ActionType",
    "WorkflowContext",
    "WorkflowExecution",
    "ActionRegistry",
    "Scheduler",
    "WorkflowTemplates"
]
