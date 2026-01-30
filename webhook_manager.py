"""
OpenEvolve Webhook Manager - External Notification System

This module provides a comprehensive webhook system for external notifications
and integrations with third-party services.

FEATURES:
- Webhook registration and management
- Event-based webhook triggering
- Retry logic with exponential backoff
- Signature verification
- Payload transformation
- Rate limiting
- Dead letter queue for failed webhooks
- Webhook logging and monitoring
"""

import os
import json
import hmac
import hashlib
import time
import threading
import requests
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

logger = logging.getLogger(__name__)


class WebhookEvent(Enum):
    """Webhook event types for decomposition engine."""
    ON_DECOMPOSE = "on_decompose"
    ON_COMPLETE = "on_complete"
    ON_ERROR = "on_error"
    ON_SUBPROBLEM_CREATED = "on_subproblem_created"
    ON_QUALITY_ASSESSMENT = "on_quality_assessment"
    ON_SOLUTION_ASSEMBLED = "on_solution_assembled"
    ON_GAUNTLET_COMPLETE = "on_gauntlet_complete"
    ON_CONFLICT_DETECTED = "on_conflict_detected"
    ON_STATE_CHANGE = "on_state_change"
    ON_CHECKPOINT = "on_checkpoint"


class WebhookStatus(Enum):
    """Webhook delivery status."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"
    DISABLED = "disabled"


@dataclass
class WebhookConfig:
    """Configuration for a webhook."""
    id: str
    name: str
    url: str
    events: List[str]
    secret: Optional[str] = None
    headers: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    retry_config: 'RetryConfig' = None
    transform: Optional[str] = None  # JSON path or template
    rate_limit: int = 100  # Max calls per minute

    def __post_init__(self):
        if self.retry_config is None:
            self.retry_config = RetryConfig()


@dataclass
class RetryConfig:
    """Retry configuration for webhooks."""
    max_retries: int = 3
    backoff_multiplier: float = 2.0
    initial_delay: float = 1.0  # seconds
    max_delay: float = 60.0  # seconds


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""
    webhook_id: str
    event_type: str
    payload: Dict[str, Any]
    status: WebhookStatus
    status_code: Optional[int] = None
    response_body: Optional[str] = None
    attempt_number: int = 1
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    duration_ms: float = 0.0


class WebhookError(Exception):
    """Base exception for webhook errors."""
    pass


class WebhookDeliveryError(WebhookError):
    """Raised when webhook delivery fails."""
    pass


class WebhookValidationError(WebhookError):
    """Raised when webhook validation fails."""
    pass


class WebhookSignatureError(WebhookError):
    """Raised when webhook signature verification fails."""
    pass


class PayloadTransformer:
    """Transforms webhook payloads based on configuration."""

    @staticmethod
    def transform(payload: Dict[str, Any], transform_config: Optional[str]) -> Dict[str, Any]:
        """
        Transform payload based on configuration.

        Args:
            payload: Original payload
            transform_config: Transformation template or JSON path

        Returns:
            Transformed payload
        """
        if not transform_config:
            return payload

        # Simple template transformation
        if transform_config.startswith("template:"):
            return PayloadTransformer._apply_template(payload, transform_config[9:])

        # JSON path transformation (simplified)
        if transform_config.startswith("path:"):
            return PayloadTransformer._apply_path(payload, transform_config[5:])

        return payload

    @staticmethod
    def _apply_template(payload: Dict[str, Any], template: str) -> Dict[str, Any]:
        """Apply template transformation."""
        try:
            # Simple variable substitution
            result = template
            for key, value in payload.items():
                result = result.replace(f"{{{key}}}", str(value))
            return json.loads(result)
        except (ValueError, TypeError, json.JSONDecodeError) as e:
            logger.error(f"Template transformation failed: {e}")
            return payload

    @staticmethod
    def _apply_path(payload: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Apply JSON path transformation."""
        try:
            keys = path.split(".")
            result = payload
            for key in keys:
                result = result.get(key, {})
            return result if isinstance(result, dict) else {"value": result}
        except (ValueError, TypeError, AttributeError) as e:
            logger.error(f"Path transformation failed: {e}")
            return payload


class WebhookSignature:
    """Handles webhook signature generation and verification."""

    @staticmethod
    def generate_signature(payload: Dict[str, Any], secret: str) -> str:
        """
        Generate HMAC signature for payload.

        Args:
            payload: Payload data
            secret: Webhook secret

        Returns:
            Hex signature
        """
        payload_str = json.dumps(payload, sort_keys=True)
        signature = hmac.new(
            secret.encode(),
            payload_str.encode(),
            hashlib.sha256
        ).hexdigest()
        return f"sha256={signature}"

    @staticmethod
    def verify_signature(payload: Dict[str, Any], signature: str, secret: str) -> bool:
        """
        Verify webhook signature.

        Args:
            payload: Payload data
            signature: Signature from header
            secret: Webhook secret

        Returns:
            True if signature valid
        """
        expected = WebhookSignature.generate_signature(payload, secret)
        return hmac.compare_digest(expected, signature)


class RateLimiter:
    """Rate limiter for webhook deliveries."""

    def __init__(self, max_calls: int = 100, period: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_calls: Maximum calls allowed
            period: Time period in seconds
        """
        self.max_calls = max_calls
        self.period = period
        self._calls: Dict[str, List[datetime]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, webhook_id: str) -> bool:
        """
        Check if webhook is allowed to be called.

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if allowed
        """
        with self._lock:
            now = datetime.now()
            cutoff = now - timedelta(seconds=self.period)

            # Clean old calls
            if webhook_id in self._calls:
                self._calls[webhook_id] = [
                    call_time for call_time in self._calls[webhook_id]
                    if call_time > cutoff
                ]

            # Check limit
            call_count = len(self._calls.get(webhook_id, []))
            if call_count >= self.max_calls:
                return False

            # Record call
            if webhook_id not in self._calls:
                self._calls[webhook_id] = []
            self._calls[webhook_id].append(now)

            return True

    def get_remaining_calls(self, webhook_id: str) -> int:
        """Get remaining calls for webhook."""
        with self._lock:
            if webhook_id not in self._calls:
                return self.max_calls

            return max(0, self.max_calls - len(self._calls[webhook_id]))


class WebhookManager:
    """
    Manages webhook registration, delivery, and monitoring.

    This is the main entry point for webhook functionality. It handles:
    - Webhook registration and configuration
    - Event delivery to webhooks
    - Retry logic with backoff
    - Rate limiting
    - Signature verification
    - Delivery logging
    """

    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize webhook manager.

        args:
            storage_path: Path to store webhook delivery logs
        """
        self._webhooks: Dict[str, WebhookConfig] = {}
        self._deliveries: List[WebhookDelivery] = []
        self._rate_limiters: Dict[str, RateLimiter] = {}
        self._storage_path = storage_path or ".openevolve/webhooks"
        self._executor = ThreadPoolExecutor(max_workers=10)
        self._lock = threading.Lock()
        self._load_webhooks()

    def _load_webhooks(self) -> None:
        """Load webhooks from storage."""
        storage_file = os.path.join(self._storage_path, "webhooks.json")
        if os.path.exists(storage_file):
            try:
                with open(storage_file, 'r') as f:
                    data = json.load(f)
                    for webhook_data in data.get('webhooks', []):
                        webhook = WebhookConfig(**webhook_data)
                        self._webhooks[webhook.id] = webhook
                        self._rate_limiters[webhook.id] = RateLimiter(webhook.rate_limit)

                logger.info(f"Loaded {len(self._webhooks)} webhooks from storage")
            except (OSError, IOError, json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to load webhooks: {e}")

    def _save_webhooks(self) -> None:
        """Save webhooks to storage."""
        os.makedirs(self._storage_path, exist_ok=True)
        storage_file = os.path.join(self._storage_path, "webhooks.json")

        try:
            data = {
                'webhooks': [
                    {
                        **asdict(webhook),
                        'retry_config': asdict(webhook.retry_config)
                    }
                    for webhook in self._webhooks.values()
                ]
            }

            with open(storage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to save webhooks: {e}")

    def register_webhook(self, webhook: WebhookConfig) -> bool:
        """
        Register a new webhook.

        Args:
            webhook: Webhook configuration

        Returns:
            True if successful

        Raises:
            WebhookValidationError: If webhook configuration is invalid
        """
        # Validate webhook
        self._validate_webhook(webhook)

        with self._lock:
            # Check for duplicate ID
            if webhook.id in self._webhooks:
                raise WebhookValidationError(f"Webhook with ID {webhook.id} already exists")

            # Store webhook
            self._webhooks[webhook.id] = webhook
            self._rate_limiters[webhook.id] = RateLimiter(webhook.rate_limit)

            # Save to storage
            self._save_webhooks()

            logger.info(f"Registered webhook: {webhook.name} ({webhook.id})")
            return True

    def _validate_webhook(self, webhook: WebhookConfig) -> None:
        """Validate webhook configuration."""
        if not webhook.url:
            raise WebhookValidationError("Webhook URL is required")

        if not webhook.url.startswith(("http://", "https://")):
            raise WebhookValidationError("Webhook URL must start with http:// or https://")

        if not webhook.events:
            raise WebhookValidationError("Webhook must have at least one event")

        # Validate events
        valid_events = [e.value for e in WebhookEvent]
        for event in webhook.events:
            if event not in valid_events:
                raise WebhookValidationError(f"Invalid event: {event}")

    def unregister_webhook(self, webhook_id: str) -> bool:
        """
        Unregister a webhook.

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if successful
        """
        with self._lock:
            if webhook_id not in self._webhooks:
                logger.warning(f"Webhook {webhook_id} not found")
                return False

            del self._webhooks[webhook_id]
            if webhook_id in self._rate_limiters:
                del self._rate_limiters[webhook_id]

            self._save_webhooks()
            logger.info(f"Unregistered webhook: {webhook_id}")
            return True

    def update_webhook(self, webhook_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update webhook configuration.

        Args:
            webhook_id: Webhook identifier
            updates: Fields to update

        Returns:
            True if successful
        """
        with self._lock:
            if webhook_id not in self._webhooks:
                logger.warning(f"Webhook {webhook_id} not found")
                return False

            webhook = self._webhooks[webhook_id]

            # Update fields
            for key, value in updates.items():
                if hasattr(webhook, key):
                    setattr(webhook, key, value)

            # Re-validate
            self._validate_webhook(webhook)

            # Update rate limiter if rate_limit changed
            if 'rate_limit' in updates:
                self._rate_limiters[webhook_id] = RateLimiter(webhook.rate_limit)

            self._save_webhooks()
            logger.info(f"Updated webhook: {webhook_id}")
            return True

    def trigger_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        """
        Trigger an event to all registered webhooks.

        Args:
            event_type: Type of event
            payload: Event payload
        """
        # Find webhooks that subscribe to this event
        webhooks_to_trigger = [
            webhook for webhook in self._webhooks.values()
            if webhook.enabled and event_type in webhook.events
        ]

        if not webhooks_to_trigger:
            logger.debug(f"No webhooks registered for event: {event_type}")
            return

        logger.info(f"Triggering event {event_type} to {len(webhooks_to_trigger)} webhooks")

        # Submit delivery tasks
        futures = []
        for webhook in webhooks_to_trigger:
            future = self._executor.submit(
                self._deliver_webhook,
                webhook,
                event_type,
                payload
            )
            futures.append(future)

        # Wait for all deliveries (optional, can be made async)
        for future in as_completed(futures):
            try:
                future.result()
            except (RuntimeError, requests.RequestException) as e:
                logger.error(f"Webhook delivery failed: {e}")

    def _deliver_webhook(
        self,
        webhook: WebhookConfig,
        event_type: str,
        payload: Dict[str, Any],
        attempt: int = 1
    ) -> WebhookDelivery:
        """
        Deliver webhook to its endpoint.

        Args:
            webhook: Webhook configuration
            event_type: Event type
            payload: Event payload
            attempt: Attempt number

        Returns:
            Delivery record
        """
        start_time = time.time()

        # Check rate limit
        rate_limiter = self._rate_limiters.get(webhook.id)
        if rate_limiter and not rate_limiter.is_allowed(webhook.id):
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event_type,
                payload=payload,
                status=WebhookStatus.FAILED,
                attempt_number=attempt,
                error="Rate limit exceeded"
            )
            self._record_delivery(delivery)
            return delivery

        # Transform payload
        transformed_payload = PayloadTransformer.transform(payload, webhook.transform)

        # Prepare headers
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "OpenEvolve-Webhook/1.0",
            "X-Webhook-Event": event_type,
            "X-Webhook-ID": webhook.id,
            **webhook.headers
        }

        # Add signature if secret is configured
        if webhook.secret:
            signature = WebhookSignature.generate_signature(transformed_payload, webhook.secret)
            headers["X-Webhook-Signature"] = signature

        # Deliver webhook
        try:
            response = requests.post(
                webhook.url,
                json=transformed_payload,
                headers=headers,
                timeout=10
            )

            duration = (time.time() - start_time) * 1000

            if response.status_code >= 200 and response.status_code < 300:
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event_type=event_type,
                    payload=transformed_payload,
                    status=WebhookStatus.SUCCESS,
                    status_code=response.status_code,
                    response_body=response.text[:1000],  # Limit response size
                    attempt_number=attempt,
                    duration_ms=duration
                )
                logger.info(f"Webhook {webhook.id} delivered successfully in {duration:.2f}ms")
            else:
                delivery = WebhookDelivery(
                    webhook_id=webhook.id,
                    event_type=event_type,
                    payload=transformed_payload,
                    status=WebhookStatus.FAILED,
                    status_code=response.status_code,
                    response_body=response.text[:1000],
                    attempt_number=attempt,
                    duration_ms=duration,
                    error=f"HTTP {response.status_code}"
                )
                logger.warning(f"Webhook {webhook.id} failed with status {response.status_code}")

        except requests.RequestException as e:
            duration = (time.time() - start_time) * 1000
            delivery = WebhookDelivery(
                webhook_id=webhook.id,
                event_type=event_type,
                payload=transformed_payload,
                status=WebhookStatus.FAILED,
                attempt_number=attempt,
                duration_ms=duration,
                error=str(e)
            )
            logger.error(f"Webhook {webhook.id} delivery failed: {e}")

        self._record_delivery(delivery)

        # Retry if failed and retries remaining
        if delivery.status == WebhookStatus.FAILED and attempt < webhook.retry_config.max_retries:
            delay = self._calculate_retry_delay(webhook, attempt)
            logger.info(f"Retrying webhook {webhook.id} in {delay}s (attempt {attempt + 1}/{webhook.retry_config.max_retries})")

            time.sleep(delay)
            return self._deliver_webhook(webhook, event_type, payload, attempt + 1)

        return delivery

    def _calculate_retry_delay(self, webhook: WebhookConfig, attempt: int) -> float:
        """Calculate retry delay with exponential backoff."""
        delay = webhook.retry_config.initial_delay * (
            webhook.retry_config.backoff_multiplier ** (attempt - 1)
        )
        return min(delay, webhook.retry_config.max_delay)

    def _record_delivery(self, delivery: WebhookDelivery) -> None:
        """Record webhook delivery."""
        with self._lock:
            self._deliveries.append(delivery)

            # Keep only last 1000 deliveries in memory
            if len(self._deliveries) > 1000:
                self._deliveries = self._deliveries[-1000:]

        # Save to file periodically
        if len(self._deliveries) % 100 == 0:
            self._save_deliveries()

    def _save_deliveries(self) -> None:
        """Save delivery logs to storage."""
        os.makedirs(self._storage_path, exist_ok=True)
        delivery_file = os.path.join(self._storage_path, "deliveries.json")

        try:
            data = [
                {
                    **asdict(delivery),
                    'status': delivery.status.value,
                    'timestamp': delivery.timestamp.isoformat()
                }
                for delivery in self._deliveries[-100:]  # Save last 100
            ]

            with open(delivery_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

        except (OSError, IOError, TypeError) as e:
            logger.error(f"Failed to save deliveries: {e}")

    def get_webhook(self, webhook_id: str) -> Optional[WebhookConfig]:
        """Get webhook by ID."""
        return self._webhooks.get(webhook_id)

    def list_webhooks(self) -> List[WebhookConfig]:
        """List all registered webhooks."""
        return list(self._webhooks.values())

    def get_deliveries(
        self,
        webhook_id: Optional[str] = None,
        limit: int = 100
    ) -> List[WebhookDelivery]:
        """
        Get delivery logs.

        Args:
            webhook_id: Filter by webhook ID
            limit: Maximum number of deliveries to return

        Returns:
            List of delivery records
        """
        deliveries = self._deliveries

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        return deliveries[-limit:]

    def get_delivery_stats(self, webhook_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get delivery statistics.

        Args:
            webhook_id: Filter by webhook ID

        Returns:
            Statistics dictionary
        """
        deliveries = self._deliveries

        if webhook_id:
            deliveries = [d for d in deliveries if d.webhook_id == webhook_id]

        if not deliveries:
            return {}

        total = len(deliveries)
        successful = len([d for d in deliveries if d.status == WebhookStatus.SUCCESS])
        failed = len([d for d in deliveries if d.status == WebhookStatus.FAILED])

        avg_duration = sum(d.duration_ms for d in deliveries) / total if total > 0 else 0

        return {
            "total_deliveries": total,
            "successful": successful,
            "failed": failed,
            "success_rate": successful / total if total > 0 else 0,
            "average_duration_ms": avg_duration,
            "last_delivery": deliveries[-1].timestamp.isoformat() if deliveries else None
        }

    def test_webhook(self, webhook_id: str) -> bool:
        """
        Test a webhook delivery.

        Args:
            webhook_id: Webhook identifier

        Returns:
            True if test successful
        """
        webhook = self.get_webhook(webhook_id)
        if not webhook:
            logger.error(f"Webhook {webhook_id} not found")
            return False

        test_payload = {
            "test": True,
            "timestamp": datetime.now().isoformat(),
            "message": "Test webhook delivery"
        }

        delivery = self._deliver_webhook(webhook, "test", test_payload)
        return delivery.status == WebhookStatus.SUCCESS

    def shutdown(self) -> None:
        """Shutdown webhook manager."""
        self._executor.shutdown(wait=True)
        self._save_deliveries()
        logger.info("Webhook manager shutdown complete")


# Singleton instance
_webhook_manager_instance: Optional[WebhookManager] = None


def get_webhook_manager() -> WebhookManager:
    """Get global webhook manager instance."""
    global _webhook_manager_instance
    if _webhook_manager_instance is None:
        _webhook_manager_instance = WebhookManager()
    return _webhook_manager_instance


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    wm = get_webhook_manager()

    # Register a test webhook
    webhook_secret = os.environ.get('WEBHOOK_SECRET')
    if not webhook_secret:
        raise ValueError("WEBHOOK_SECRET environment variable must be set for webhook signature verification")
    
    webhook = WebhookConfig(
        id="test_webhook",
        name="Test Webhook",
        url="https://httpbin.org/post",
        events=[WebhookEvent.ON_DECOMPOSE.value, WebhookEvent.ON_COMPLETE.value],
        secret=webhook_secret
    )

    wm.register_webhook(webhook)

    # Trigger an event
    wm.trigger_event(WebhookEvent.ON_DECOMPOSE.value, {
        "problem": "Test problem",
        "timestamp": datetime.now().isoformat()
    })

    # Get delivery stats
    stats = wm.get_delivery_stats(webhook_id="test_webhook")
    print(f"Delivery stats: {stats}")

    wm.shutdown()
