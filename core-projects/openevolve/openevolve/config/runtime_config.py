"""
Runtime Configuration Updates

This module provides functionality to update configuration parameters during evolution
without requiring a restart. Supports single parameter updates, batch updates, and
rollback capabilities.
"""

import copy
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel

from ..unified.config import UnifiedEvolutionConfig
from ..unified.config_validator import ConfigValidator


class ValidationResult:
    """Validation result wrapper"""
    def __init__(self, is_valid: bool, errors: List[str] = None):
        self.is_valid = is_valid
        self.errors = errors or []


class SimpleConfigValidator:
    """Simplified config validator for runtime updates"""

    def __init__(self):
        pass

    def validate_config(self, config: UnifiedEvolutionConfig) -> ValidationResult:
        """Validate full configuration"""
        validator = ConfigValidator(config)
        errors, warnings = validator.validate()
        return ValidationResult(is_valid=len(errors) == 0, errors=errors)

    def validate_parameter(self, scope: str, param_name: str, value: Any) -> ValidationResult:
        """Validate a single parameter"""
        # Basic validation - can be extended
        try:
            # Check if value type is appropriate
            if scope == "common":
                if param_name == "max_iterations":
                    if not isinstance(value, int) or value < 1:
                        return ValidationResult(False, [f"max_iterations must be positive int"])
                elif param_name == "concurrency":
                    if not isinstance(value, int) or value < 1:
                        return ValidationResult(False, [f"concurrency must be positive int"])

            return ValidationResult(True)
        except Exception as e:
            return ValidationResult(False, [str(e)])


logger = logging.getLogger(__name__)


class ConfigUpdate(BaseModel):
    """Record of a configuration update"""
    timestamp: datetime
    parameter: str
    old_value: Any
    new_value: Any
    update_type: str = "single"  # single, batch, rollback


class RuntimeConfigUpdater:
    """
    Update configuration parameters during evolution

    Enables dynamic configuration changes without restart:
    - Single parameter updates
    - Batch atomic updates
    - Rollback capabilities
    - Update history tracking
    """

    def __init__(self, current_config: UnifiedEvolutionConfig):
        """
        Initialize runtime config updater

        Args:
            current_config: Current configuration to manage
        """
        self.current_config = current_config
        self.update_history: List[ConfigUpdate] = []
        self.validators = SimpleConfigValidator()
        self._config_snapshots: Dict[datetime, UnifiedEvolutionConfig] = {}

    async def update_parameter(
        self,
        param_name: str,
        new_value: Any,
        validate: bool = True,
        scope: str = "common"
    ) -> bool:
        """
        Update a single parameter at runtime

        Args:
            param_name: Parameter to update (supports dot notation, e.g., "llm.temperature")
            new_value: New value
            validate: Whether to validate before applying
            scope: Configuration scope (common, llm, database, evaluator, pes, qd, mo, adversarial, openevolve)

        Returns:
            True if update successful, False otherwise
        """
        # Validate new value
        if validate:
            validation_result = self.validators.validate_parameter(
                scope, param_name, new_value
            )
            if not validation_result.is_valid:
                logger.error(f"Validation failed for {scope}.{param_name}: {validation_result.errors}")
                return False

        # Parse dot notation
        if "." in param_name:
            parts = param_name.split(".")
            target_scope = parts[0]
            target_param = ".".join(parts[1:])
        else:
            target_scope = scope
            target_param = param_name

        # Get old value
        try:
            old_value = self._get_parameter_value(target_scope, target_param)
        except (AttributeError, KeyError) as e:
            logger.error(f"Failed to get current value for {target_scope}.{target_param}: {e}")
            return False

        # Apply update
        try:
            self._set_parameter_value(target_scope, target_param, new_value)

            # Record update
            self.update_history.append(ConfigUpdate(
                timestamp=datetime.utcnow(),
                parameter=f"{target_scope}.{target_param}",
                old_value=old_value,
                new_value=new_value,
                update_type="single"
            ))

            logger.info(f"Updated {target_scope}.{target_param}: {old_value} -> {new_value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update {target_scope}.{target_param}: {e}")
            return False

    async def update_parameters(
        self,
        updates: Dict[str, Any],
        validate: bool = True,
        rollback_on_error: bool = True
    ) -> bool:
        """
        Update multiple parameters atomically

        Args:
            updates: Dict of {param_name: new_value} (supports dot notation)
            validate: Validate all updates before applying
            rollback_on_error: Rollback all if any update fails

        Returns:
            True if all updates successful
        """
        # Create snapshot for rollback
        backup = self._create_snapshot()

        # Validate all updates first
        if validate:
            for param_path, new_value in updates.items():
                # Parse scope and parameter
                if "." in param_path:
                    parts = param_path.split(".", 1)
                    scope = parts[0]
                    param = parts[1] if len(parts) > 1 else ""
                else:
                    scope = "common"
                    param = param_path

                validation_result = self.validators.validate_parameter(
                    scope, param, new_value
                )
                if not validation_result.is_valid:
                    logger.error(f"Validation failed for {param_path}: {validation_result.errors}")
                    if rollback_on_error:
                        self._restore_snapshot(backup)
                    return False

        # Apply updates
        applied_updates = []
        try:
            for param_path, new_value in updates.items():
                # Parse scope and parameter
                if "." in param_path:
                    parts = param_path.split(".", 1)
                    scope = parts[0]
                    param = parts[1] if len(parts) > 1 else ""
                else:
                    scope = "common"
                    param = param_path

                # Get old value
                old_value = self._get_parameter_value(scope, param)

                # Apply update
                self._set_parameter_value(scope, param, new_value)

                applied_updates.append(ConfigUpdate(
                    timestamp=datetime.utcnow(),
                    parameter=param_path,
                    old_value=old_value,
                    new_value=new_value,
                    update_type="batch"
                ))

            # Add all to history
            self.update_history.extend(applied_updates)
            logger.info(f"Successfully applied {len(applied_updates)} batch updates")
            return True

        except Exception as e:
            logger.error(f"Failed to apply batch updates: {e}")
            if rollback_on_error:
                self._restore_snapshot(backup)
                logger.info("Rolled back to pre-update state")
            return False

    def get_update_history(
        self,
        limit: Optional[int] = None,
        parameter_filter: Optional[str] = None
    ) -> List[ConfigUpdate]:
        """
        Get history of all runtime updates

        Args:
            limit: Maximum number of updates to return (None = all)
            parameter_filter: Filter by parameter name (supports partial match)

        Returns:
            List of configuration updates
        """
        history = self.update_history

        # Filter by parameter
        if parameter_filter:
            history = [
                u for u in history
                if parameter_filter.lower() in u.parameter.lower()
            ]

        # Limit
        if limit:
            history = history[-limit:]

        return history

    def rollback_to(self, timestamp: datetime) -> bool:
        """
        Rollback configuration to state at timestamp

        Args:
            timestamp: Rollback to this timestamp

        Returns:
            True if rollback successful
        """
        # Find snapshot closest to but before timestamp
        closest_timestamp = None
        for snap_ts in self._config_snapshots.keys():
            if snap_ts <= timestamp:
                if closest_timestamp is None or snap_ts > closest_timestamp:
                    closest_timestamp = snap_ts

        if closest_timestamp is None:
            logger.error(f"No snapshot found before {timestamp}")
            return False

        # Restore snapshot
        snapshot = self._config_snapshots[closest_timestamp]
        self._restore_snapshot(snapshot)

        # Truncate update history
        self.update_history = [
            u for u in self.update_history
            if u.timestamp < timestamp
        ]

        logger.info(f"Rolled back configuration to {closest_timestamp}")
        return True

    def rollback_updates(self, num_updates: int) -> bool:
        """
        Rollback the last N updates

        Args:
            num_updates: Number of recent updates to rollback

        Returns:
            True if rollback successful
        """
        if num_updates > len(self.update_history):
            logger.error(f"Cannot rollback {num_updates} updates, only {len(self.update_history)} in history")
            return False

        # Create snapshot of current state
        current_snapshot = self._create_snapshot()

        # Find timestamp before the updates to rollback
        updates_to_rollback = self.update_history[-num_updates:]
        target_timestamp = updates_to_rollback[0].timestamp

        # Rollback
        success = self.rollback_to(target_timestamp)

        if success:
            # Remove rolled-back updates from history
            self.update_history = self.update_history[:-num_updates]
            logger.info(f"Rolled back {num_updates} updates")

        return success

    def create_snapshot(self, label: Optional[str] = None) -> datetime:
        """
        Create a snapshot of current configuration

        Args:
            label: Optional label for the snapshot

        Returns:
            Timestamp of the snapshot
        """
        timestamp = datetime.utcnow()
        self._config_snapshots[timestamp] = self._create_snapshot()

        if label:
            logger.info(f"Created configuration snapshot '{label}' at {timestamp}")
        else:
            logger.info(f"Created configuration snapshot at {timestamp}")

        return timestamp

    def list_snapshots(self) -> List[Dict[str, Any]]:
        """
        List all configuration snapshots

        Returns:
            List of snapshot metadata
        """
        return [
            {
                "timestamp": ts,
                "label": snapshot.metadata.get("snapshot_label", "unnamed")
            }
            for ts, snapshot in sorted(self._config_snapshots.items())
        ]

    def _get_parameter_value(self, scope: str, param_name: str) -> Any:
        """Get current value of a parameter"""
        # Get the config object for the scope
        config_obj = getattr(self.current_config, scope, None)
        if config_obj is None:
            raise AttributeError(f"Invalid scope: {scope}")

        # Handle nested parameters
        if "." in param_name:
            parts = param_name.split(".")
            value = config_obj
            for part in parts:
                value = getattr(value, part)
            return value
        else:
            return getattr(config_obj, param_name)

    def _set_parameter_value(self, scope: str, param_name: str, value: Any) -> None:
        """Set value of a parameter"""
        # Get the config object for the scope
        config_obj = getattr(self.current_config, scope, None)
        if config_obj is None:
            raise AttributeError(f"Invalid scope: {scope}")

        # Handle nested parameters
        if "." in param_name:
            parts = param_name.split(".")
            obj = config_obj
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(config_obj, param_name, value)

    def _create_snapshot(self) -> UnifiedEvolutionConfig:
        """Create a deep copy of current configuration"""
        snapshot = copy.deepcopy(self.current_config)
        return snapshot

    def _restore_snapshot(self, snapshot: UnifiedEvolutionConfig) -> None:
        """Restore configuration from snapshot"""
        # Deep copy to avoid reference issues, then copy field values into the
        # existing current_config object in place so external references to it
        # (e.g. fixtures or held config objects) remain valid after rollback.
        new_config = copy.deepcopy(snapshot)
        for name in type(self.current_config).model_fields:
            setattr(self.current_config, name, getattr(new_config, name))


class ConfigWatcherCallback:
    """Base class for configuration change callbacks"""

    async def on_config_change(
        self,
        old_config: UnifiedEvolutionConfig,
        new_config: UnifiedEvolutionConfig,
        changes: List[ConfigUpdate]
    ) -> None:
        """
        Called when configuration changes

        Args:
            old_config: Previous configuration
            new_config: New configuration
            changes: List of applied changes
        """
        raise NotImplementedError("Subclasses must implement on_config_change")
