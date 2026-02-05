"""
OpenEvolve Slack Integration Adapter

This adapter integrates the decomposition engine with Slack for
team communication and notifications.

FEATURES:
- Send notifications to Slack channels
- Post decomposition results
- Alert on quality threshold failures
- Share progress updates
- Interactive buttons for workflow actions
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

try:
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False

from plugin_system import PluginBase, PluginMetadata
from decomposition_engine import DecompositionPlan, SubProblem

logger = logging.getLogger(__name__)


class SlackNotificationType(Enum):
    """Types of Slack notifications."""
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class SlackConfig:
    """Slack integration configuration."""
    bot_token: str
    signing_secret: str
    default_channel: str
    notification_types: List[SlackNotificationType] = field(default_factory=list)
    username: str = "OpenEvolve"
    icon_emoji: str = ":robot_face:"


@dataclass
class SlackMessage:
    """Slack message structure."""
    channel: str
    text: str
    attachments: List[Dict[str, Any]] = field(default_factory=list)
    blocks: List[Dict[str, Any]] = field(default_factory=list)


class SlackAdapter(PluginBase):
    """
    Slack integration plugin for OpenEvolve.

    This plugin sends notifications and updates to Slack channels
    based on decomposition engine events.

    Example:
        ```python
        from plugin_integrations.slack_adapter import SlackAdapter, SlackConfig

        config = SlackConfig(
            bot_token="xoxb-your-token",
            signing_secret="your-signing-secret",
            default_channel="#decomposition"
        )

        adapter = SlackAdapter(config)
        adapter.activate()

        # Send notification
        adapter.send_notification("Decomposition complete!", SlackNotificationType.SUCCESS)
        ```
    """

    def __init__(self, config: Optional[SlackConfig] = None):
        metadata = PluginMetadata(
            name="slack_adapter",
            version="1.0.0",
            description="Slack integration for team notifications",
            author="OpenEvolve",
            license="MIT",
            tags=["slack", "notifications", "communication"],
            category="integration"
        )

        super().__init__(metadata)

        self.config = config
        self._slack_client: Optional['WebClient'] = None

    def activate(self) -> bool:
        """Activate the Slack adapter."""
        if not SLACK_AVAILABLE:
            logger.error("Slack SDK not available. Install with: pip install slack-sdk")
            return False

        if not self.config:
            logger.error("Slack configuration not provided")
            return False

        try:
            # Initialize Slack client
            self._slack_client = WebClient(token=self.config.bot_token)

            # Test connection
            auth_response = self._slack_client.auth_test()
            logger.info(f"Connected to Slack workspace: {auth_response['team']}")

            # Register hooks
            self.register_hooks()

            return super().activate()

        except SlackApiError as e:
            logger.error(f"Failed to connect to Slack: {e.response['error']}")
            return False

    def deactivate(self) -> bool:
        """Deactivate the Slack adapter."""
        self._slack_client = None
        return super().deactivate()

    def register_hooks(self) -> None:
        """Register plugin hooks."""
        self.register_hook(
            "on_after_decompose",
            self.on_after_decompose,
            priority=100
        )

        self.register_hook(
            "on_workflow_complete",
            self.on_workflow_complete,
            priority=100
        )

        self.register_hook(
            "on_quality_threshold_failed",
            self.on_quality_threshold_failed,
            priority=100
        )

        self.register_hook(
            "on_workflow_error",
            self.on_workflow_error,
            priority=100
        )

    def on_after_decompose(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called after decomposition.

        Sends notification with decomposition summary.
        """
        plan = context.get('plan')
        if not plan:
            return context

        try:
            self.send_decomposition_summary(plan)
        except Exception as e:
            logger.error(f"Failed to send decomposition summary: {e}")

        return context

    def on_workflow_complete(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when workflow completes.

        Sends success notification.
        """
        plan = context.get('plan')
        if not plan:
            return context

        try:
            self.send_notification(
                f"[OK] Workflow complete for: {plan.original_problem[:100]}",
                SlackNotificationType.SUCCESS
            )
        except Exception as e:
            logger.error(f"Failed to send completion notification: {e}")

        return context

    def on_quality_threshold_failed(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when quality threshold fails.

        Sends warning notification.
        """
        quality_score = context.get('quality_score')
        threshold = context.get('threshold')

        try:
            message = f"[WARN] Quality threshold failed: {quality_score:.2f} < {threshold:.2f}"
            self.send_notification(message, SlackNotificationType.WARNING)
        except Exception as e:
            logger.error(f"Failed to send threshold warning: {e}")

        return context

    def on_workflow_error(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called when workflow encounters an error.

        Sends error notification.
        """
        error = context.get('error')

        try:
            message = f"[FAIL] Workflow error: {str(error)[:200]}"
            self.send_notification(message, SlackNotificationType.ERROR)
        except Exception as e:
            logger.error(f"Failed to send error notification: {e}")

        return context

    def send_notification(
        self,
        message: str,
        notification_type: SlackNotificationType = SlackNotificationType.INFO,
        channel: Optional[str] = None
    ) -> bool:
        """
        Send a notification to Slack.

        Args:
            message: Message text
            notification_type: Type of notification
            channel: Optional channel (default: configured default)

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        # Check if this notification type is enabled
        if notification_type not in self.config.notification_types:
            return False

        target_channel = channel or self.config.default_channel

        try:
            # Send message with color-coded attachment
            color = self._get_notification_color(notification_type)

            self._slack_client.chat_postMessage(
                channel=target_channel,
                text=message,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                attachments=[
                    {
                        "color": color,
                        "text": message
                    }
                ]
            )

            logger.info(f"Sent {notification_type.value} notification to {target_channel}")
            return True

        except SlackApiError as e:
            logger.error(f"Failed to send notification: {e.response['error']}")
            return False

    def send_decomposition_summary(self, plan: DecompositionPlan) -> bool:
        """
        Send a formatted decomposition summary to Slack.

        Args:
            plan: Decomposition plan

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        try:
            # Create rich message with blocks
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": "🎯 Decomposition Complete"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Problem:*\n{plan.original_problem[:100]}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Strategy:*\n{plan.strategy.strategy_name}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Sub-problems:*\n{len(plan.sub_problems)}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Quality Score:*\n{plan.quality_scores.overall_score:.2f}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Quality Breakdown:*\n"
                                f"* Cohesion: {plan.quality_scores.cohesion:.2f}\n"
                                f"* Completeness: {plan.quality_scores.completeness:.2f}\n"
                                f"* Clarity: {plan.quality_scores.clarity:.2f}"
                    }
                }
            ]

            # Add sub-problems section
            if plan.sub_problems:
                subproblem_text = "\n".join(
                    f"* {sp.title} (complexity: {sp.complexity_score.value:.2f})"
                    for sp in plan.sub_problems[:5]  # Limit to 5
                )

                if len(plan.sub_problems) > 5:
                    subproblem_text += f"\n* ... and {len(plan.sub_problems) - 5} more"

                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Sub-problems:*\n{subproblem_text}"
                    }
                })

            self._slack_client.chat_postMessage(
                channel=self.config.default_channel,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                blocks=blocks
            )

            logger.info(f"Sent decomposition summary for plan {plan.plan_id}")
            return True

        except SlackApiError as e:
            logger.error(f"Failed to send decomposition summary: {e.response['error']}")
            return False

    def send_subproblem_update(self, subproblem: SubProblem, status: str) -> bool:
        """
        Send an update for a sub-problem.

        Args:
            subproblem: Sub-problem
            status: Status update

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        try:
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"📝 Sub-problem Update: {status}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {
                            "type": "mrkdwn",
                            "text": f"*Title:*\n{subproblem.title}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Type:*\n{subproblem.problem_type.value}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Complexity:*\n{subproblem.complexity_score.value:.2f}"
                        },
                        {
                            "type": "mrkdwn",
                            "text": f"*Status:*\n{status}"
                        }
                    ]
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Description:*\n{subproblem.description[:200]}"
                    }
                }
            ]

            self._slack_client.chat_postMessage(
                channel=self.config.default_channel,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                blocks=blocks
            )

            logger.info(f"Sent sub-problem update for {subproblem.id}")
            return True

        except SlackApiError as e:
            logger.error(f"Failed to send sub-problem update: {e.response['error']}")
            return False

    def send_interactive_message(
        self,
        message: str,
        actions: List[Dict[str, str]],
        channel: Optional[str] = None
    ) -> bool:
        """
        Send an interactive message with buttons.

        Args:
            message: Message text
            actions: List of action buttons
            channel: Optional channel

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        target_channel = channel or self.config.default_channel

        try:
            # Create button elements
            elements = [
                {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": action.get('text', 'Button')
                    },
                    "value": action.get('value', ''),
                    "action_id": action.get('action_id', 'button_click')
                }
                for action in actions
            ]

            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": message
                    }
                },
                {
                    "type": "actions",
                    "elements": elements
                }
            ]

            self._slack_client.chat_postMessage(
                channel=target_channel,
                username=self.config.username,
                icon_emoji=self.config.icon_emoji,
                blocks=blocks
            )

            logger.info(f"Sent interactive message to {target_channel}")
            return True

        except SlackApiError as e:
            logger.error(f"Failed to send interactive message: {e.response['error']}")
            return False

    def _get_notification_color(self, notification_type: SlackNotificationType) -> str:
        """Get color for notification type."""
        colors = {
            SlackNotificationType.INFO: "#36a64f",      # Green
            SlackNotificationType.SUCCESS: "#36a64f",   # Green
            SlackNotificationType.WARNING: "#ff9900",   # Orange
            SlackNotificationType.ERROR: "#ff0000",     # Red
        }
        return colors.get(notification_type, "#36a64f")

    def create_channel(self, name: str) -> bool:
        """
        Create a new Slack channel.

        Args:
            name: Channel name (without #)

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        try:
            self._slack_client.conversations_create(name=name)
            logger.info(f"Created Slack channel: {name}")
            return True
        except SlackApiError as e:
            logger.error(f"Failed to create channel: {e.response['error']}")
            return False

    def invite_to_channel(self, channel: str, user_ids: List[str]) -> bool:
        """
        Invite users to a channel.

        Args:
            channel: Channel name or ID
            user_ids: List of user IDs

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        try:
            for user_id in user_ids:
                self._slack_client.conversations_invite(
                    channel=channel,
                    users=user_id
                )
            logger.info(f"Invited {len(user_ids)} users to {channel}")
            return True
        except SlackApiError as e:
            logger.error(f"Failed to invite users: {e.response['error']}")
            return False

    def upload_file(
        self,
        file_content: str,
        filename: str,
        channel: Optional[str] = None,
        title: Optional[str] = None
    ) -> bool:
        """
        Upload a file to Slack.

        Args:
            file_content: File content
            filename: Filename
            channel: Optional channel
            title: Optional file title

        Returns:
            True if successful
        """
        if not self._slack_client:
            return False

        target_channel = channel or self.config.default_channel

        try:
            self._slack_client.files_upload_v2(
                content=file_content,
                filename=filename,
                channels=target_channel,
                title=title or filename
            )
            logger.info(f"Uploaded file {filename} to {target_channel}")
            return True
        except SlackApiError as e:
            logger.error(f"Failed to upload file: {e.response['error']}")
            return False


# Factory function
def create_slack_adapter(config: SlackConfig) -> SlackAdapter:
    """Create a Slack adapter instance."""
    return SlackAdapter(config)
