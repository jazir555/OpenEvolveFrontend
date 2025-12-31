"""
A2A (Agent-to-Agent) Protocol

Implementation of the Agent-to-Agent communication protocol
for inter-agent messaging and coordination.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import asyncio
import uuid

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of A2A messages"""
    # Solution workflow
    SOLUTION_SUBMITTED = "solution_submitted"
    SOLUTION_REQUEST = "solution_request"

    # Critique workflow
    CRITIQUE_SUBMITTED = "critique_submitted"
    CRITIQUE_REQUEST = "critique_request"

    # Verification workflow
    VERIFICATION_SUBMITTED = "verification_submitted"
    VERIFICATION_REQUEST = "verification_request"

    # Collaboration
    REFINEMENT_REQUEST = "refinement_request"
    CLARIFICATION_REQUEST = "clarification_request"
    COLLABORATION_PROPOSAL = "collaboration_proposal"

    # Status
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    ACKNOWLEDGMENT = "acknowledgment"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class A2AMessage:
    """
    Agent-to-Agent message.

    Represents a message sent between agents in the workflow.
    """

    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType = MessageType.STATUS_UPDATE
    sender: str = "unknown"
    recipient: str = "unknown"
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=lambda: datetime.utcnow().timestamp())
    reply_to: Optional[str] = None
    requires_response: bool = False
    response_deadline: Optional[float] = None

    # Workflow-specific fields
    sub_problem_id: Optional[str] = None
    artifact_id: Optional[str] = None
    workflow_stage: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            "message_id": self.message_id,
            "message_type": self.message_type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "content": self.content,
            "metadata": self.metadata,
            "priority": self.priority.value,
            "timestamp": self.timestamp,
            "reply_to": self.reply_to,
            "requires_response": self.requires_response,
            "response_deadline": self.response_deadline,
            "sub_problem_id": self.sub_problem_id,
            "artifact_id": self.artifact_id,
            "workflow_stage": self.workflow_stage
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "A2AMessage":
        """Create message from dictionary"""
        return cls(
            message_id=data.get("message_id", str(uuid.uuid4())),
            message_type=MessageType(data.get("message_type", "status_update")),
            sender=data.get("sender", "unknown"),
            recipient=data.get("recipient", "unknown"),
            content=data.get("content", ""),
            metadata=data.get("metadata", {}),
            priority=MessagePriority(data.get("priority", 2)),
            timestamp=data.get("timestamp", datetime.utcnow().timestamp()),
            reply_to=data.get("reply_to"),
            requires_response=data.get("requires_response", False),
            response_deadline=data.get("response_deadline"),
            sub_problem_id=data.get("sub_problem_id"),
            artifact_id=data.get("artifact_id"),
            workflow_stage=data.get("workflow_stage")
        )


class A2AProtocol:
    """
    Agent-to-Agent communication protocol handler.

    Manages message passing between agents, handles message routing,
    and provides reliable delivery guarantees.

    Usage:
        protocol = A2AProtocol()

        # Register message handler
        protocol.register_handler("blue_team", handler_function)

        # Send message
        await protocol.send_message(
            sender="red_team",
            recipient="blue_team",
            message_type=MessageType.REFINEMENT_REQUEST,
            content="Please address these issues...",
            metadata={"issues": [...]}
        )

        # Receive messages
        messages = await protocol.get_messages("blue_team")
    """

    def __init__(self, enable_persistence: bool = False):
        """
        Initialize the A2A protocol.

        Args:
            enable_persistence: Whether to persist messages to storage
        """
        self.enable_persistence = enable_persistence

        # Message queues for each agent
        self.message_queues: Dict[str, List[A2AMessage]] = {}
        self.message_handlers: Dict[str, List[Callable]] = {}

        # Message tracking
        self.sent_messages: Dict[str, A2AMessage] = {}
        self.pending_responses: Dict[str, A2AMessage] = {}

        # Statistics
        self.stats = {
            "messages_sent": 0,
            "messages_delivered": 0,
            "messages_failed": 0,
            "responses_received": 0
        }

        logger.info("A2A Protocol initialized")

    def register_handler(
        self,
        agent_id: str,
        handler: Callable[[A2AMessage], Any]
    ):
        """
        Register a message handler for an agent.

        Args:
            agent_id: Agent identifier
            handler: Async function to handle messages
        """
        if agent_id not in self.message_handlers:
            self.message_handlers[agent_id] = []

        self.message_handlers[agent_id].append(handler)
        logger.info(f"Registered handler for agent {agent_id}")

    async def send_message(
        self,
        sender: str,
        recipient: str,
        message_type: MessageType,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        priority: MessagePriority = MessagePriority.NORMAL,
        reply_to: Optional[str] = None,
        requires_response: bool = False,
        response_timeout: float = 300,
        sub_problem_id: Optional[str] = None,
        artifact_id: Optional[str] = None,
        workflow_stage: Optional[str] = None
    ) -> A2AMessage:
        """
        Send a message from one agent to another.

        Args:
            sender: Sender agent ID
            recipient: Recipient agent ID
            message_type: Type of message
            content: Message content
            metadata: Additional metadata
            priority: Message priority
            reply_to: Message ID this is in reply to
            requires_response: Whether response is required
            response_timeout: Timeout for response in seconds
            sub_problem_id: Associated sub-problem ID
            artifact_id: Associated artifact ID
            workflow_stage: Workflow stage

        Returns:
            Sent message

        Example:
            >>> await protocol.send_message(
            ...     sender="red_team",
            ...     recipient="blue_team",
            ...     message_type=MessageType.REFINEMENT_REQUEST,
            ...     content="Please address security concerns...",
            ...     sub_problem_id="sub_1"
            ... )
        """
        message = A2AMessage(
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            content=content,
            metadata=metadata or {},
            priority=priority,
            reply_to=reply_to,
            requires_response=requires_response,
            sub_problem_id=sub_problem_id,
            artifact_id=artifact_id,
            workflow_stage=workflow_stage
        )

        # Set response deadline if response required
        if requires_response:
            import time
            message.response_deadline = time.time() + response_timeout

        # Add to recipient's queue
        if recipient not in self.message_queues:
            self.message_queues[recipient] = []

        self.message_queues[recipient].append(message)

        # Track sent message
        self.sent_messages[message.message_id] = message
        self.stats["messages_sent"] += 1

        # Track pending response if needed
        if requires_response:
            self.pending_responses[message.message_id] = message

        # Persist if enabled
        if self.enable_persistence:
            await self._persist_message(message)

        logger.info(
            f"Message sent: {sender} → {recipient} "
            f"({message_type.value}, priority={priority.name})"
        )

        return message

    async def get_messages(
        self,
        agent_id: str,
        wait: bool = False,
        timeout: float = 5
    ) -> List[A2AMessage]:
        """
        Get messages for an agent.

        Args:
            agent_id: Agent to get messages for
            wait: Whether to wait for messages if queue is empty
            timeout: How long to wait (if wait=True)

        Returns:
            List of messages for the agent

        Example:
            >>> messages = await protocol.get_messages("blue_team")
            >>> for msg in messages:
            ...     await handle_message(msg)
        """
        if agent_id not in self.message_queues:
            self.message_queues[agent_id] = []

        # Wait for messages if requested and queue is empty
        if wait and not self.message_queues[agent_id]:
            try:
                await asyncio.wait_for(
                    self._wait_for_message(agent_id),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                pass

        messages = self.message_queues[agent_id].copy()
        self.message_queues[agent_id].clear()

        self.stats["messages_delivered"] += len(messages)

        return messages

    async def _wait_for_message(self, agent_id: str):
        """Wait for a message to arrive for the agent"""
        while agent_id not in self.message_queues or not self.message_queues[agent_id]:
            await asyncio.sleep(0.1)

    async def send_reply(
        self,
        original_message: A2AMessage,
        reply_content: str,
        reply_metadata: Optional[Dict[str, Any]] = None
    ) -> A2AMessage:
        """
        Send a reply to a message.

        Args:
            original_message: Message being replied to
            reply_content: Reply content
            reply_metadata: Additional metadata for reply

        Returns:
            Sent reply message

        Example:
            >>> await protocol.send_reply(
            ...     original_message=critique_message,
            ...     reply_content="I've addressed the security concerns..."
            ... )
        """
        reply = await self.send_message(
            sender=original_message.recipient,
            recipient=original_message.sender,
            message_type=MessageType.ACKNOWLEDGMENT,
            content=reply_content,
            metadata=reply_metadata or {},
            reply_to=original_message.message_id,
            sub_problem_id=original_message.sub_problem_id,
            artifact_id=original_message.artifact_id,
            workflow_stage=original_message.workflow_stage
        )

        # Mark original message as responded
        if original_message.message_id in self.pending_responses:
            del self.pending_responses[original_message.message_id]
            self.stats["responses_received"] += 1

        return reply

    async def broadcast(
        self,
        sender: str,
        recipients: List[str],
        message_type: MessageType,
        content: str,
        **kwargs
    ) -> List[A2AMessage]:
        """
        Broadcast a message to multiple recipients.

        Args:
            sender: Sender agent ID
            recipients: List of recipient agent IDs
            message_type: Type of message
            content: Message content
            **kwargs: Additional arguments for send_message

        Returns:
            List of sent messages

        Example:
            >>> await protocol.broadcast(
            ...     sender="orchestrator",
            ...     recipients=["blue_team", "red_team", "gold_team"],
            ...     message_type=MessageType.STATUS_UPDATE,
            ...     content="All teams please report status"
            ... )
        """
        messages = []

        for recipient in recipients:
            message = await self.send_message(
                sender=sender,
                recipient=recipient,
                message_type=message_type,
                content=content,
                **kwargs
            )
            messages.append(message)

        return messages

    async def check_pending_responses(self) -> List[A2AMessage]:
        """
        Check for pending responses that have timed out.

        Returns:
            List of timed out messages
        """
        import time
        current_time = time.time()
        timed_out = []

        for message_id, message in self.pending_responses.items():
            if message.response_deadline and current_time > message.response_deadline:
                timed_out.append(message)
                del self.pending_responses[message_id]

        return timed_out

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get protocol statistics.

        Returns:
            Statistics dict
        """
        return {
            **self.stats,
            "pending_responses": len(self.pending_responses),
            "queues": {
                agent_id: len(messages)
                for agent_id, messages in self.message_queues.items()
            }
        }

    async def _persist_message(self, message: A2AMessage):
        """Persist message to storage (if enabled)"""
        # This would integrate with the storage manager
        # For now, it's a placeholder
        pass

    async def process_message(self, message: A2AMessage) -> Any:
        """
        Process a message through registered handlers.

        Args:
            message: Message to process

        Returns:
            Handler results (if any)
        """
        handlers = self.message_handlers.get(message.recipient, [])

        if not handlers:
            logger.warning(f"No handlers registered for {message.recipient}")
            return None

        results = []
        for handler in handlers:
            try:
                result = await handler(message)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler error for {message.recipient}: {e}")
                results.append({"error": str(e)})

        return results

    def clear_queues(self, agent_id: Optional[str] = None):
        """
        Clear message queues.

        Args:
            agent_id: Specific agent to clear, or None to clear all
        """
        if agent_id:
            if agent_id in self.message_queues:
                self.message_queues[agent_id].clear()
        else:
            for queue in self.message_queues.values():
                queue.clear()

        logger.info(f"Cleared message queues for {agent_id or 'all agents'}")


class MessageBuilder:
    """
    Builder for creating common A2A messages.

    Provides convenience methods for creating standard workflow messages.
    """

    @staticmethod
    def solution_submitted(
        sender: str,
        recipient: str,
        solution: str,
        sub_problem_id: str,
        artifact_id: str
    ) -> A2AMessage:
        """Create a solution submitted message"""
        return A2AMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.SOLUTION_SUBMITTED,
            content=f"Solution submitted for sub-problem {sub_problem_id}",
            metadata={"solution_preview": solution[:200]},
            sub_problem_id=sub_problem_id,
            artifact_id=artifact_id
        )

    @staticmethod
    def critique_submitted(
        sender: str,
        recipient: str,
        critique: str,
        issues: List[str],
        sub_problem_id: str,
        artifact_id: str
    ) -> A2AMessage:
        """Create a critique submitted message"""
        return A2AMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.CRITIQUE_SUBMITTED,
            content=f"Critique submitted for sub-problem {sub_problem_id}",
            metadata={
                "critique_preview": critique[:200],
                "issues_count": len(issues),
                "issues": issues
            },
            sub_problem_id=sub_problem_id,
            artifact_id=artifact_id
        )

    @staticmethod
    def refinement_request(
        sender: str,
        recipient: str,
        issues: List[str],
        sub_problem_id: str,
        artifact_id: str
    ) -> A2AMessage:
        """Create a refinement request message"""
        return A2AMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.REFINEMENT_REQUEST,
            content=f"Refinement requested for sub-problem {sub_problem_id}",
            metadata={
                "issues": issues,
                "issues_count": len(issues)
            },
            priority=MessagePriority.HIGH,
            requires_response=True,
            sub_problem_id=sub_problem_id,
            artifact_id=artifact_id
        )

    @staticmethod
    def verification_result(
        sender: str,
        recipient: str,
        passes: bool,
        score: float,
        sub_problem_id: str,
        artifact_id: str
    ) -> A2AMessage:
        """Create a verification result message"""
        return A2AMessage(
            sender=sender,
            recipient=recipient,
            message_type=MessageType.VERIFICATION_SUBMITTED,
            content=f"Verification {'PASSED' if passes else 'FAILED'} for sub-problem {sub_problem_id}",
            metadata={
                "passes": passes,
                "score": score
            },
            priority=MessagePriority.HIGH,
            sub_problem_id=sub_problem_id,
            artifact_id=artifact_id
        )
