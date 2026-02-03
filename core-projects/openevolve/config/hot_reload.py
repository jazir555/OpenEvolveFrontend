"""
Configuration Hot-Reload System

Watches configuration files for changes and automatically reloads them.
"""

import os
import time
import logging
import threading
from typing import Any, Callable, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ConfigChangeEvent:
    """
    Event representing a configuration change.

    Attributes:
        filepath: Path to changed file
        old_config: Previous configuration
        new_config: New configuration
        changes: Dict of changed parameters
        timestamp: When the change was detected
    """
    filepath: str
    old_config: Dict[str, Any]
    new_config: Dict[str, Any]
    changes: Dict[str, Any]
    timestamp: datetime


class ConfigHotReload:
    """
    Watch config files for changes and auto-reload.

    Features:
    - File watching using polling (no external dependencies)
    - Debouncing to avoid excessive reloads
    - Validation before applying changes
    - Callback notification system
    - Thread-safe operation

    Note: Uses polling instead of watchdog for minimal dependencies.
          For high-performance needs, watchdog can be used.
    """

    def __init__(
        self,
        config_file: str,
        callback: Callable[[ConfigChangeEvent], None],
        poll_interval: float = 1.0,
        debounce_delay: float = 2.0
    ):
        """
        Initialize ConfigHotReload.

        Args:
            config_file: Path to config file to watch
            callback: Function to call when config changes
            poll_interval: How often to check for changes (seconds)
            debounce_delay: How long to wait before reloading after change (seconds)
        """
        self.config_file = os.path.abspath(config_file)
        self.callback = callback
        self.poll_interval = poll_interval
        self.debounce_delay = debounce_delay

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

        # Track file state
        self._last_mtime: Optional[float] = None
        self._last_size: Optional[int] = None
        self._current_config: Optional[Dict[str, Any]] = None
        self._last_change_time: Optional[float] = None

        # Statistics
        self._reload_count = 0
        self._error_count = 0

        # Lazy import
        from .config_loader import ConfigLoader
        self.loader = ConfigLoader()
        from .validator import ConfigValidator
        self.validator = ConfigValidator()

    def start(self) -> None:
        """
        Start watching config file for changes.

        Runs in background thread.
        """
        if self._running:
            logger.warning("Hot-reload already running")
            return

        # Check if file exists
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # Load initial config
        self._current_config = self.loader.load_auto(self.config_file)
        self._update_file_state()

        # Start watching thread
        self._running = True
        self._thread = threading.Thread(target=self._watch_loop, daemon=True)
        self._thread.start()

        logger.info(f"Started hot-reload for {self.config_file}")

    def stop(self) -> None:
        """Stop watching config file"""
        if not self._running:
            return

        self._running = False

        # Wait for thread to finish
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

        logger.info(f"Stopped hot-reload for {self.config_file}")

    def on_file_changed(self, event: Optional[dict] = None) -> None:
        """
        Handle file change event.

        Called internally when file change is detected.

        Args:
            event: Optional event dict (for compatibility with watchdog)
        """
        try:
            # Load new config
            new_config = self.loader.load_auto(self.config_file)

            # Check if actually changed
            if self._current_config == new_config:
                logger.debug("Config unchanged, skipping reload")
                return

            old_config = self._current_config or {}

            # Validate new config
            validation = self.validator.validate(new_config)
            if not validation.is_valid:
                logger.error(
                    f"Invalid config, not applying: {validation.get_error_messages()}"
                )
                self._error_count += 1
                return

            # Calculate changes
            changes = self._calculate_changes(old_config, new_config)

            # Create change event
            change_event = ConfigChangeEvent(
                filepath=self.config_file,
                old_config=old_config,
                new_config=new_config,
                changes=changes,
                timestamp=datetime.now()
            )

            # Update current config
            with self._lock:
                self._current_config = new_config

            # Notify callback
            if self.callback:
                try:
                    self.callback(change_event)
                    self._reload_count += 1
                    logger.info(
                        f"Config reloaded successfully: {len(changes)} parameters changed"
                    )
                except Exception as e:
                    logger.error(f"Error in config change callback: {e}")

        except Exception as e:
            logger.error(f"Error handling config file change: {e}")
            self._error_count += 1

    def validate_and_apply(self, new_config: Dict[str, Any]) -> bool:
        """
        Validate and apply new configuration.

        Args:
            new_config: New configuration to apply

        Returns:
            True if applied successfully, False otherwise
        """
        # Validate
        validation = self.validator.validate(new_config)
        if not validation.is_valid:
            logger.error(f"Invalid config: {validation.get_error_messages()}")
            return False

        # Apply
        old_config = self._current_config or {}
        changes = self._calculate_changes(old_config, new_config)

        with self._lock:
            self._current_config = new_config

        # Notify
        if self.callback:
            change_event = ConfigChangeEvent(
                filepath=self.config_file,
                old_config=old_config,
                new_config=new_config,
                changes=changes,
                timestamp=datetime.now()
            )
            self.callback(change_event)

        return True

    def notify_listeners(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> None:
        """
        Notify all listeners of config changes.

        Args:
            old_config: Previous configuration
            new_config: New configuration
        """
        if self.callback:
            changes = self._calculate_changes(old_config, new_config)
            change_event = ConfigChangeEvent(
                filepath=self.config_file,
                old_config=old_config,
                new_config=new_config,
                changes=changes,
                timestamp=datetime.now()
            )
            self.callback(change_event)

    def get_current_config(self) -> Optional[Dict[str, Any]]:
        """
        Get current configuration.

        Returns:
            Current config dict or None if not loaded
        """
        with self._lock:
            return self._current_config.copy() if self._current_config else None

    def get_stats(self) -> Dict[str, Any]:
        """
        Get hot-reload statistics.

        Returns:
            Dict with reload_count, error_count, etc.
        """
        return {
            'running': self._running,
            'config_file': self.config_file,
            'reload_count': self._reload_count,
            'error_count': self._error_count,
            'last_mtime': self._last_mtime,
        }

    def _watch_loop(self) -> None:
        """Main watch loop (runs in background thread)"""
        while self._running:
            try:
                # Check for changes
                if self._has_file_changed():
                    now = time.time()

                    # Debounce: wait for file to stabilize
                    if self._last_change_time is None:
                        self._last_change_time = now
                    elif now - self._last_change_time >= self.debounce_delay:
                        # File has stabilized, reload
                        self.on_file_changed()
                        self._last_change_time = None
                        self._update_file_state()

            except Exception as e:
                logger.error(f"Error in watch loop: {e}")

            # Wait before next check
            time.sleep(self.poll_interval)

    def _has_file_changed(self) -> bool:
        """Check if file has changed since last check"""
        if not os.path.exists(self.config_file):
            return False

        stat = os.stat(self.config_file)
        mtime = stat.st_mtime
        size = stat.st_size

        # Check if modified or size changed
        if self._last_mtime is None or self._last_size is None:
            return True

        return mtime != self._last_mtime or size != self._last_size

    def _update_file_state(self) -> None:
        """Update tracked file state"""
        if os.path.exists(self.config_file):
            stat = os.stat(self.config_file)
            self._last_mtime = stat.st_mtime
            self._last_size = stat.st_size

    def _calculate_changes(
        self,
        old_config: Dict[str, Any],
        new_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate differences between configs.

        Args:
            old_config: Old configuration
            new_config: New configuration

        Returns:
            Dict of changed parameters
        """
        changes = {}

        # Find changed and new keys
        for key in new_config:
            if key not in old_config or old_config[key] != new_config[key]:
                changes[key] = {
                    'old': old_config.get(key),
                    'new': new_config[key]
                }

        # Find deleted keys
        for key in old_config:
            if key not in new_config:
                changes[key] = {
                    'old': old_config[key],
                    'new': None
                }

        return changes


class MultiFileHotReload:
    """
    Watch multiple config files for changes.

    Useful when config is split across multiple files.
    """

    def __init__(
        self,
        config_files: list,
        callback: Callable[[str, ConfigChangeEvent], None],
        poll_interval: float = 1.0
    ):
        """
        Initialize multi-file hot-reload.

        Args:
            config_files: List of config file paths to watch
            callback: Function called for each file change (receives filepath, event)
            poll_interval: How often to check for changes
        """
        self.config_files = config_files
        self.callback = callback
        self.poll_interval = poll_interval

        self._watchers: Dict[str, ConfigHotReload] = {}
        self._running = False

    def start(self) -> None:
        """Start watching all config files"""
        self._running = True

        for filepath in self.config_files:
            watcher = ConfigHotReload(
                config_file=filepath,
                callback=lambda event, fp=filepath: self._on_change(fp, event),
                poll_interval=self.poll_interval
            )
            watcher.start()
            self._watchers[filepath] = watcher

        logger.info(f"Started watching {len(self.config_files)} config files")

    def stop(self) -> None:
        """Stop watching all config files"""
        self._running = False

        for watcher in self._watchers.values():
            watcher.stop()

        self._watchers.clear()
        logger.info("Stopped watching all config files")

    def _on_change(self, filepath: str, event: ConfigChangeEvent) -> None:
        """Handle file change and notify callback"""
        if self.callback:
            self.callback(filepath, event)

    def get_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get stats for all watchers"""
        return {
            filepath: watcher.get_stats()
            for filepath, watcher in self._watchers.items()
        }


# Convenience function for simple hot-reload
def watch_config_file(
    config_file: str,
    callback: Callable[[ConfigChangeEvent], None]
) -> ConfigHotReload:
    """
    Start watching a config file.

    Args:
        config_file: Path to config file
        callback: Function to call when config changes

    Returns:
        ConfigHotReload instance

    Example:
        def on_change(event):
            print(f"Config changed: {event.changes}")

        watcher = watch_config_file('config.yaml', on_change)
        # watcher runs in background
        # ... do work ...
        watcher.stop()
    """
    watcher = ConfigHotReload(config_file, callback)
    watcher.start()
    return watcher
