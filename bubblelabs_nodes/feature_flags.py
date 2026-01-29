"""
Feature Flag System for Gauntlet Components

Provides feature flags with gradual rollout capabilities,
monitoring, and automatic rollback support.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import logging
import os
from datetime import datetime
from .gauntlet_metrics import MetricsCollector, get_metrics_collector

logger = logging.getLogger(__name__)


class FeatureState(Enum):
    """Feature states"""
    DISABLED = "disabled"
    ENABLED = "enabled"
    ROLLBACK = "rollback"


@dataclass
class FeatureFlag:
    """Represents a feature flag"""
    name: str
    state: FeatureState
    rollout_percentage: float  # 0-100
    enabled_users: List[str] = None
    enabled_teams: List[str] = None
    description: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.enabled_users is None:
            self.enabled_users = []
        if self.enabled_teams is None:
            self.enabled_teams = []
        if self.metadata is None:
            self.metadata = {}

    def is_enabled_for(
        self,
        user_id: str = None,
        team: str = None,
        context: Dict = None
    ) -> bool:
        """
        Check if feature is enabled for given user/team/context.

        Args:
            user_id: Optional user identifier
            team: Optional team identifier
            context: Additional context

        Returns:
            True if feature enabled
        """
        # If disabled, not enabled for anyone
        if self.state == FeatureState.DISABLED:
            return False

        # If rollback, disabled
        if self.state == FeatureState.ROLLBACK:
            return False

        # If fully enabled (100% rollout)
        if self.rollout_percentage >= 100:
            return True

        # Check user whitelist
        if user_id and user_id in self.enabled_users:
            return True

        # Check team whitelist
        if team and team in self.enabled_teams:
            return True

        # Check rollout percentage (hash-based)
        if context and 'user_id' in context:
            hash_val = hash(context['user_id']) % 100
            return hash_val < self.rollout_percentage

        return False


class FeatureFlagManager:
    """
    Manages feature flags with rollout and monitoring.
    """

    def __init__(self, metrics_collector: MetricsCollector = None):
        self.metrics = metrics_collector or get_metrics_collector()
        self.flags: Dict[str, FeatureFlag] = {}
        self._load_flags_from_config()

    def _load_flags_from_config(self):
        """Load feature flags from environment/config"""
        # PARALLEL_EXECUTION flag
        parallel_enabled = os.getenv('PARALLEL_EXECUTION_ENABLED', 'false').lower() == 'true'
        self.flags['parallel_execution'] = FeatureFlag(
            name='parallel_execution',
            state=FeatureState.ENABLED if parallel_enabled else FeatureState.DISABLED,
            rollout_percentage=100 if parallel_enabled else 0,
            description='Enable parallel execution of independent problems',
        )

        # FUZZING flag
        fuzzing_enabled = os.getenv('FUZZING_ENABLED', 'false').lower() == 'true'
        self.flags['fuzzing'] = FeatureFlag(
            name='fuzzing',
            state=FeatureState.ENABLED if fuzzing_enabled else FeatureState.DISABLED,
            rollout_percentage=100 if fuzzing_enabled else 0,
            description='Enable automated fuzzing of solutions',
        )

        # CACHING flag
        caching_enabled = os.getenv('CACHE_ENABLED', 'true').lower() == 'true'
        self.flags['caching'] = FeatureFlag(
            name='caching',
            state=FeatureState.ENABLED if caching_enabled else FeatureState.DISABLED,
            rollout_percentage=100 if caching_enabled else 0,
            description='Enable solution caching',
        )

        # CHECKPOINTING flag
        checkpointing_enabled = os.getenv('CHECKPOINTING_ENABLED', 'true').lower() == 'true'
        self.flags['checkpointing'] = FeatureFlag(
            name='checkpointing',
            state=FeatureState.ENABLED if checkpointing_enabled else FeatureState.DISABLED,
            rollout_percentage=100 if checkpointing_enabled else 0,
            description='Enable checkpointing and resume',
        )

    def register_flag(self, flag: FeatureFlag):
        """Register a feature flag"""
        self.flags[flag.name] = flag
        logger.info(f"Registered feature flag: {flag.name}")

    def get_flag(self, feature_name: str) -> Optional[FeatureFlag]:
        """Get a feature flag by name"""
        return self.flags.get(feature_name)

    def is_enabled(
        self,
        feature_name: str,
        user_id: str = None,
        team: str = None,
        context: Dict = None
    ) -> bool:
        """
        Check if feature is enabled.

        Args:
            feature_name: Name of the feature
            user_id: Optional user ID
            team: Optional team
            context: Additional context

        Returns:
            True if feature enabled
        """
        flag = self.flags.get(feature_name)
        if not flag:
            logger.warning(f"Unknown feature flag: {feature_name}")
            return False

        enabled = flag.is_enabled_for(user_id, team, context)

        # Track metrics
        if enabled:
            awaitable = self.metrics.increment_counter(f'feature_{feature_name}_enabled')
        else:
            awaitable = self.metrics.increment_counter(f'feature_{feature_name}_disabled')

        return enabled

    async def set_rollout_percentage(
        self,
        feature_name: str,
        percentage: float
    ) -> bool:
        """
        Set rollout percentage for a feature.

        Args:
            feature_name: Name of the feature
            percentage: Rollout percentage (0-100)

        Returns:
            True if successful
        """
        if percentage < 0 or percentage > 100:
            logger.error(f"Invalid rollout percentage: {percentage}")
            return False

        flag = self.flags.get(feature_name)
        if not flag:
            logger.error(f"Unknown feature flag: {feature_name}")
            return False

        old_percentage = flag.rollout_percentage
        flag.rollout_percentage = percentage

        # Update state based on rollout
        if percentage == 0:
            flag.state = FeatureState.DISABLED
        elif percentage == 100:
            flag.state = FeatureState.ENABLED

        await self.metrics.set_gauge(
            f'feature_{feature_name}_rollout_percentage',
            percentage
        )

        logger.info(
            f"Updated {feature_name} rollout: "
            f"{old_percentage}% -> {percentage}%"
        )

        return True

    def enable_feature(self, feature_name: str) -> bool:
        """Enable a feature (100% rollout)"""
        flag = self.flags.get(feature_name)
        if not flag:
            logger.error(f"Unknown feature flag: {feature_name}")
            return False

        flag.state = FeatureState.ENABLED
        flag.rollout_percentage = 100

        logger.info(f"Enabled feature: {feature_name}")
        return True

    def disable_feature(self, feature_name: str) -> bool:
        """Disable a feature (0% rollout)"""
        flag = self.flags.get(feature_name)
        if not flag:
            logger.error(f"Unknown feature flag: {feature_name}")
            return False

        flag.state = FeatureState.DISABLED
        flag.rollout_percentage = 0

        logger.info(f"Disabled feature: {feature_name}")
        return True

    def rollback_feature(self, feature_name: str) -> bool:
        """Rollback a feature (emergency disable)"""
        flag = self.flags.get(feature_name)
        if not flag:
            logger.error(f"Unknown feature flag: {feature_name}")
            return False

        flag.state = FeatureState.ROLLBACK

        logger.error(f"Rolled back feature: {feature_name}")
        return True

    def get_all_flags(self) -> Dict[str, FeatureFlag]:
        """Get all feature flags"""
        return self.flags.copy()

    def get_feature_states(self) -> Dict[str, str]:
        """Get states of all features"""
        return {
            name: flag.state.value
            for name, flag in self.flags.items()
        }


class DeploymentMonitor:
    """
    Monitors deployment metrics and validates feature rollouts.
    """

    def __init__(self, feature_manager: FeatureFlagManager):
        self.feature_manager = feature_manager
        self.metrics = get_metrics_collector()
        self.error_rates = {}  # feature -> [recent_errors]

    async def record_success(self, feature_name: str):
        """Record successful operation"""
        if feature_name not in self.error_rates:
            self.error_rates[feature_name] = []

        # Track last 100 operations
        self.error_rates[feature_name].append(False)

        if len(self.error_rates[feature_name]) > 100:
            self.error_rates[feature_name] = self.error_rates[feature_name][-100:]

        # Calculate error rate
        errors = sum(1 for e in self.error_rates[feature_name] if e)
        total = len(self.error_rates[feature_name])
        error_rate = errors / total if total > 0 else 0

        await self.metrics.set_gauge(
            f'feature_{feature_name}_error_rate',
            error_rate
        )

        # Auto-rollback if error rate too high
        if error_rate > 0.5:  # 50% error rate
            logger.error(
                f"High error rate for {feature_name}: "
                f"{error_rate:.1%} - triggering rollback"
            )
            self.feature_manager.rollback_feature(feature_name)

    async def record_failure(self, feature_name: str, error: Exception):
        """Record failed operation"""
        if feature_name not in self.error_rates:
            self.error_rates[feature_name] = []

        self.error_rates[feature_name].append(True)

        logger.warning(
            f"Feature {feature_name} failure: {error}"
        )

        # Check success
        await self.record_success(feature_name)

    def get_error_rate(self, feature_name: str) -> float:
        """Get current error rate for a feature"""
        if feature_name not in self.error_rates:
            return 0.0

        errors = sum(1 for e in self.error_rates[feature_name] if e)
        total = len(self.error_rates[feature_name])
        return errors / total if total > 0 else 0.0


def create_feature_flag_manager() -> FeatureFlagManager:
    """Create a feature flag manager"""
    return FeatureFlagManager()


def create_deployment_monitor(feature_manager: FeatureFlagManager) -> DeploymentMonitor:
    """Create a deployment monitor"""
    return DeploymentMonitor(feature_manager)
