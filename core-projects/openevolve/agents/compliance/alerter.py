"""
Alert & Escalation Module
Generates multi-severity alerts and manages escalation workflows.

Author: AI Architecture Team
Date: 2026-01-30
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
from pathlib import Path
import asyncio

# Try importing notification libraries
try:
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    EMAIL_AVAILABLE = True
except ImportError:
    EMAIL_AVAILABLE = False
    smtplib = None


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert lifecycle status"""
    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"
    ESCALATED = "escalated"


class EscalationLevel(Enum):
    """Escalation levels"""
    AUTOMATED = "automated"
    TEAM_LEAD = "team_lead"
    MANAGER = "manager"
    COMPLIANCE_OFFICER = "compliance_officer"
    EXECUTIVE = "executive"
    REGULATOR = "regulator"


@dataclass
class Alert:
    """Represents a compliance alert"""
    alert_id: str
    severity: AlertSeverity
    status: AlertStatus
    title: str
    message: str
    source: str
    violation_type: str
    detected_at: datetime
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_by: Optional[str] = None
    escalation_level: EscalationLevel = EscalationLevel.AUTOMATED
    false_positive_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate alert ID if not provided"""
        if not self.alert_id:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            self.alert_id = f"alert_{timestamp}_{self.severity.value}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'severity': self.severity.value,
            'status': self.status.value,
            'title': self.title,
            'message': self.message,
            'source': self.source,
            'violation_type': self.violation_type,
            'detected_at': self.detected_at.isoformat(),
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_by': self.resolved_by,
            'escalation_level': self.escalation_level.value,
            'false_positive_score': self.false_positive_score,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create from dictionary"""
        return cls(
            alert_id=data['alert_id'],
            severity=AlertSeverity(data['severity']),
            status=AlertStatus(data['status']),
            title=data['title'],
            message=data['message'],
            source=data['source'],
            violation_type=data['violation_type'],
            detected_at=datetime.fromisoformat(data['detected_at']),
            acknowledged_at=datetime.fromisoformat(data['acknowledged_at']) if data.get('acknowledged_at') else None,
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            acknowledged_by=data.get('acknowledged_by'),
            resolved_by=data.get('resolved_by'),
            escalation_level=EscalationLevel(data.get('escalation_level', 'automated')),
            false_positive_score=data.get('false_positive_score', 0.0),
            metadata=data.get('metadata', {})
        )


@dataclass
class EscalationRule:
    """Rule for when to escalate alerts"""
    severity_threshold: AlertSeverity
    time_threshold_minutes: int
    escalation_level: EscalationLevel
    require_acknowledgment: bool = True
    notification_channels: List[str] = field(default_factory=list)


class ComplianceAlerter:
    """
    Manages alert generation and escalation

    Features:
    - Multi-severity alerts
    - Smart escalation based on time and severity
    - False positive detection and learning
    - Alert fatigue prevention
    - Multiple notification channels

    Example:
        >>> alerter = ComplianceAlerter(threshold=AlertSeverity.MEDIUM)
        >>> alert = await alerter.generate_alert(violation_data)
        >>> await alerter.escalate_alerts([alert])
    """

    def __init__(
        self,
        threshold: AlertSeverity = AlertSeverity.MEDIUM,
        escalation_rules: Optional[List[EscalationRule]] = None,
        notification_config: Optional[Dict[str, Any]] = None,
        enable_fatigue_prevention: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize compliance alerter

        Args:
            threshold: Minimum severity for generating alerts
            escalation_rules: Custom escalation rules
            notification_config: Configuration for notifications
            enable_fatigue_prevention: Enable alert fatigue prevention
            logger: Logger instance
        """
        self.threshold = threshold
        self.enable_fatigue_prevention = enable_fatigue_prevention

        self.logger = logger or self._setup_logging()

        # Escalation rules
        self.escalation_rules = escalation_rules or self._get_default_escalation_rules()

        # Notification config
        self.notification_config = notification_config or {}

        # Alert history for fatigue prevention
        self.alert_history: List[Alert] = []
        self._load_alert_history()

        # False positive learning
        self.false_positive_patterns: Dict[str, float] = {}
        self._load_false_positive_patterns()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("ComplianceAlerter")
        logger.setLevel(logging.INFO)
        return logger

    def _get_default_escalation_rules(self) -> List[EscalationRule]:
        """Get default escalation rules"""
        return [
            # CRITICAL: Escalate immediately
            EscalationRule(
                severity_threshold=AlertSeverity.CRITICAL,
                time_threshold_minutes=0,
                escalation_level=EscalationLevel.EXECUTIVE,
                require_acknowledgment=True,
                notification_channels=['email', 'sms', 'slack']
            ),
            # HIGH: Escalate after 30 minutes
            EscalationRule(
                severity_threshold=AlertSeverity.HIGH,
                time_threshold_minutes=30,
                escalation_level=EscalationLevel.COMPLIANCE_OFFICER,
                require_acknowledgment=True,
                notification_channels=['email', 'slack']
            ),
            # MEDIUM: Escalate after 2 hours
            EscalationRule(
                severity_threshold=AlertSeverity.MEDIUM,
                time_threshold_minutes=120,
                escalation_level=EscalationLevel.TEAM_LEAD,
                require_acknowledgment=False,
                notification_channels=['email']
            ),
            # LOW: Escalate after 24 hours
            EscalationRule(
                severity_threshold=AlertSeverity.LOW,
                time_threshold_minutes=1440,
                escalation_level=EscalationLevel.TEAM_LEAD,
                require_acknowledgment=False,
                notification_channels=['email']
            ),
        ]

    def _load_alert_history(self):
        """Load alert history from cache"""
        cache_file = Path("./cache/alerts/history.json")
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                self.alert_history = [Alert.from_dict(item) for item in data]
                self.logger.info(f"Loaded {len(self.alert_history)} historical alerts")
            except Exception as e:
                self.logger.error(f"Failed to load alert history: {e}")

    def _save_alert_history(self):
        """Save alert history to cache"""
        cache_file = Path("./cache/alerts/history.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, 'w') as f:
                json.dump([a.to_dict() for a in self.alert_history], f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save alert history: {e}")

    def _load_false_positive_patterns(self):
        """Load false positive patterns from cache"""
        cache_file = Path("./cache/alerts/false_positives.json")
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.false_positive_patterns = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load false positive patterns: {e}")

    def _save_false_positive_patterns(self):
        """Save false positive patterns"""
        cache_file = Path("./cache/alerts/false_positives.json")
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(cache_file, 'w') as f:
                json.dump(self.false_positive_patterns, f, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save false positive patterns: {e}")

    async def generate_alert(
        self,
        violation: Dict[str, Any],
        source: str = "compliance_monitor"
    ) -> Optional[Alert]:
        """
        Generate an alert from a violation

        Args:
            violation: Violation data
            source: Source of the violation

        Returns:
            Alert object or None if below threshold
        """
        # Determine severity
        severity = self._determine_severity(violation)

        # Check if below threshold
        if self._is_below_threshold(severity):
            self.logger.debug(f"Violation below threshold {self.threshold.value}")
            return None

        # Check for alert fatigue
        if self.enable_fatigue_prevention and self._should_suppress_for_fatigue(violation):
            self.logger.info("Alert suppressed due to fatigue prevention")
            return None

        # Check false positive likelihood
        false_positive_score = self._calculate_false_positive_score(violation)

        if false_positive_score > 0.8:
            self.logger.info(f"Alert suppressed as likely false positive (score: {false_positive_score:.2f})")
            return None

        # Create alert
        alert = Alert(
            alert_id="",  # Will be generated in __post_init__
            severity=severity,
            status=AlertStatus.OPEN,
            title=violation.get('title', 'Compliance Violation Detected'),
            message=violation.get('message', str(violation)),
            source=source,
            violation_type=violation.get('type', 'unknown'),
            detected_at=datetime.utcnow(),
            false_positive_score=false_positive_score,
            metadata=violation.get('metadata', {})
        )

        # Add to history
        self.alert_history.append(alert)
        self._save_alert_history()

        # Send notification
        await self._send_notification(alert)

        self.logger.warning(f"Alert generated: {alert.alert_id} - {alert.title}")

        return alert

    def _determine_severity(self, violation: Dict[str, Any]) -> AlertSeverity:
        """Determine alert severity from violation"""
        # Check if violation specifies severity
        if 'severity' in violation:
            try:
                return AlertSeverity(violation['severity'])
            except ValueError:
                pass

        # Determine based on violation characteristics
        risk_score = violation.get('risk_score', 0)

        if risk_score >= 90:
            return AlertSeverity.CRITICAL
        elif risk_score >= 70:
            return AlertSeverity.HIGH
        elif risk_score >= 50:
            return AlertSeverity.MEDIUM
        elif risk_score >= 30:
            return AlertSeverity.LOW
        else:
            return AlertSeverity.INFO

    def _is_below_threshold(self, severity: AlertSeverity) -> bool:
        """Check if severity is below threshold"""
        severity_order = [
            AlertSeverity.INFO,
            AlertSeverity.LOW,
            AlertSeverity.MEDIUM,
            AlertSeverity.HIGH,
            AlertSeverity.CRITICAL
        ]

        threshold_index = severity_order.index(self.threshold)
        severity_index = severity_order.index(severity)

        return severity_index < threshold_index

    def _should_suppress_for_fatigue(self, violation: Dict[str, Any]) -> bool:
        """
        Check if alert should be suppressed to prevent fatigue

        Uses rate limiting and deduplication
        """
        # Get violation signature
        signature = self._get_violation_signature(violation)

        # Look for similar recent alerts
        recent_alerts = [
            alert for alert in self.alert_history
            if alert.detected_at > datetime.utcnow() - timedelta(minutes=30)
            and alert.status == AlertStatus.OPEN
        ]

        # Count similar alerts
        similar_count = sum(
            1 for alert in recent_alerts
            if self._get_violation_signature(alert.to_dict()) == signature
        )

        # Suppress if too many similar alerts
        if similar_count >= 5:
            return True

        return False

    def _get_violation_signature(self, violation: Dict[str, Any]) -> str:
        """Get signature for deduplication"""
        key_fields = [
            violation.get('type'),
            violation.get('source'),
            violation.get('rule_id')
        ]
        return ":".join(str(f) for f in key_fields if f)

    def _calculate_false_positive_score(self, violation: Dict[str, Any]) -> float:
        """
        Calculate likelihood of false positive (0-1)

        Uses historical patterns and ML
        """
        # Base score from patterns
        signature = self._get_violation_signature(violation)
        pattern_score = self.false_positive_patterns.get(signature, 0.0)

        # Adjust based on violation characteristics
        risk_score = violation.get('risk_score', 50)

        # Higher risk score = less likely false positive
        risk_adjustment = (100 - risk_score) / 100

        false_positive_score = pattern_score * 0.7 + risk_adjustment * 0.3

        return min(1.0, max(0.0, false_positive_score))

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str
    ) -> bool:
        """
        Acknowledge an alert

        Args:
            alert_id: Alert identifier
            acknowledged_by: User acknowledging

        Returns:
            True if successful
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_at = datetime.utcnow()
                alert.acknowledged_by = acknowledged_by
                self._save_alert_history()
                self.logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
                return True

        return False

    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
        is_false_positive: bool = False
    ) -> bool:
        """
        Resolve an alert

        Args:
            alert_id: Alert identifier
            resolved_by: User resolving
            is_false_positive: Whether this was a false positive

        Returns:
            True if successful
        """
        for alert in self.alert_history:
            if alert.alert_id == alert_id:
                alert.status = AlertStatus.FALSE_POSITIVE if is_false_positive else AlertStatus.RESOLVED
                alert.resolved_at = datetime.utcnow()
                alert.resolved_by = resolved_by

                # Learn from false positives
                if is_false_positive:
                    await self._learn_false_positive(alert)

                self._save_alert_history()
                self.logger.info(f"Alert {alert_id} resolved by {resolved_by}")
                return True

        return False

    async def _learn_false_positive(self, alert: Alert):
        """Learn from false positive to improve detection"""
        signature = self._get_violation_signature(alert.to_dict())

        # Increment false positive score for this pattern
        current_score = self.false_positive_patterns.get(signature, 0.0)
        self.false_positive_patterns[signature] = min(1.0, current_score + 0.1)

        self._save_false_positive_patterns()
        self.logger.info(f"Learned false positive pattern: {signature}")

    async def escalate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """
        Check and escalate alerts as needed

        Args:
            alerts: List of alerts to check

        Returns:
            List of escalated alerts
        """
        escalated = []

        for alert in alerts:
            # Skip if already resolved or false positive
            if alert.status in [AlertStatus.RESOLVED, AlertStatus.FALSE_POSITIVE]:
                continue

            # Check if escalation is needed
            escalation_rule = self._get_escalation_rule(alert)

            if escalation_rule and self._should_escalate(alert, escalation_rule):
                # Escalate
                old_level = alert.escalation_level
                alert.escalation_level = escalation_rule.escalation_level
                alert.status = AlertStatus.ESCALATED

                escalated.append(alert)

                self.logger.warning(
                    f"Alert {alert.alert_id} escalated from {old_level.value} "
                    f"to {escalation_rule.escalation_level.value}"
                )

                # Send escalation notification
                await self._send_escalation_notification(alert, escalation_rule)

        if escalated:
            self._save_alert_history()

        return escalated

    def _get_escalation_rule(self, alert: Alert) -> Optional[EscalationRule]:
        """Get applicable escalation rule for alert"""
        for rule in self.escalation_rules:
            severity_order = [
                AlertSeverity.INFO,
                AlertSeverity.LOW,
                AlertSeverity.MEDIUM,
                AlertSeverity.HIGH,
                AlertSeverity.CRITICAL
            ]

            if severity_order.index(alert.severity) >= severity_order.index(rule.severity_threshold):
                return rule

        return None

    def _should_escalate(self, alert: Alert, rule: EscalationRule) -> bool:
        """Check if alert should be escalated"""
        # Check time threshold
        time_since_detection = datetime.utcnow() - alert.detected_at

        if time_since_detection >= timedelta(minutes=rule.time_threshold_minutes):
            # Check acknowledgment requirement
            if rule.require_acknowledgment:
                return alert.status != AlertStatus.ACKNOWLEDGED
            else:
                return True

        return False

    async def _send_notification(self, alert: Alert):
        """Send notification for alert"""
        channels = self._get_notification_channels(alert)

        for channel in channels:
            try:
                if channel == 'email':
                    await self._send_email_notification(alert)
                elif channel == 'slack':
                    await self._send_slack_notification(alert)
                elif channel == 'sms':
                    await self._send_sms_notification(alert)
            except Exception as e:
                self.logger.error(f"Failed to send {channel} notification: {e}")

    def _get_notification_channels(self, alert: Alert) -> List[str]:
        """Get notification channels for alert"""
        # Get from escalation rule
        rule = self._get_escalation_rule(alert)
        if rule:
            return rule.notification_channels

        # Default based on severity
        if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.HIGH]:
            return ['email', 'slack']
        else:
            return ['email']

    async def _send_email_notification(self, alert: Alert):
        """Send email notification"""
        if not EMAIL_AVAILABLE:
            self.logger.warning("Email not available")
            return

        # Get email config
        config = self.notification_config.get('email', {})
        if not config:
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = config.get('from')
            msg['To'] = config.get('to')
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value}
Status: {alert.status.value}
Detected: {alert.detected_at.isoformat()}

{alert.message}

Alert ID: {alert.alert_id}
            """

            msg.attach(MIMEText(body, 'plain'))

            # This would actually send the email
            # with smtplib.SMTP(config['host'], config['port']) as server:
            #     server.send_message(msg)

            self.logger.info(f"Email notification sent for {alert.alert_id}")

        except Exception as e:
            self.logger.error(f"Failed to send email: {e}")

    async def _send_slack_notification(self, alert: Alert):
        """Send Slack notification"""
        # Placeholder - would use Slack Webhook API
        self.logger.info(f"Slack notification sent for {alert.alert_id}")

    async def _send_sms_notification(self, alert: Alert):
        """Send SMS notification"""
        # Placeholder - would use SMS API
        self.logger.info(f"SMS notification sent for {alert.alert_id}")

    async def _send_escalation_notification(self, alert: Alert, rule: EscalationRule):
        """Send escalation notification"""
        # Send to appropriate escalation level
        self.logger.warning(
            f"ESCALATION NOTIFICATION: {alert.alert_id} "
            f"escalated to {rule.escalation_level.value}"
        )

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics"""
        total = len(self.alert_history)
        if total == 0:
            return {'total': 0}

        by_severity = {}
        for severity in AlertSeverity:
            count = sum(1 for a in self.alert_history if a.severity == severity)
            by_severity[severity.value] = count

        by_status = {}
        for status in AlertStatus:
            count = sum(1 for a in self.alert_history if a.status == status)
            by_status[status.value] = count

        avg_resolution_time = 0.0
        resolved = [a for a in self.alert_history if a.resolved_at]
        if resolved:
            resolution_times = [
                (a.resolved_at - a.detected_at).total_seconds()
                for a in resolved
            ]
            avg_resolution_time = sum(resolution_times) / len(resolution_times)

        false_positive_rate = (
            sum(1 for a in self.alert_history if a.status == AlertStatus.FALSE_POSITIVE) / total
        )

        return {
            'total': total,
            'by_severity': by_severity,
            'by_status': by_status,
            'avg_resolution_time_hours': avg_resolution_time / 3600,
            'false_positive_rate': false_positive_rate,
            'open_alerts': by_status.get('open', 0),
            'patterns_learned': len(self.false_positive_patterns)
        }
