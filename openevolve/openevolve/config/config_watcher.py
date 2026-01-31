"""
Configuration File Watcher

This module provides hot-reload functionality for configuration files. Monitors
configuration files for changes and automatically reloads them when detected.
"""

import os
import threading
import time
import logging
from typing import Callable, List, Optional
from pathlib import Path

from ..unified.config import UnifiedEvolutionConfig
from .runtime_config import ConfigUpdate, SimpleConfigValidator, ValidationResult


logger = logging.getLogger(__name__)


class ConfigFileWatcher:
    """
    Watch configuration files for changes and auto-reload

    Features:
    - Background thread monitoring
    - Multiple file format support (YAML, JSON)
    - Validation before reload
    - Callback notifications
    - Debouncing to prevent duplicate reloads
    """

    def __init__(
        self,
        config_file: str,
        poll_interval: float = 1.0,
        debounce_delay: float = 2.0
    ):
        """
        Initialize config file watcher

        Args:
            config_file: Path to configuration file to watch
            poll_interval: Seconds between checks (default: 1.0)
            debounce_delay: Seconds to wait after last change before reloading (default: 2.0)
        """
        self.config_file = Path(config_file)
        if not self.config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        self.last_modified = self.config_file.stat().st_mtime
        self.last_size = self.config_file.stat().st_size

        self.poll_interval = poll_interval
        self.debounce_delay = debounce_delay

        self.callbacks: List[Callable[[UnifiedEvolutionConfig], None]] = []
        self.running = False
        self.watcher_thread: Optional[threading.Thread] = None
        self.pending_reload = False
        self.reload_timer: Optional[threading.Timer] = None

        self.validators = SimpleConfigValidator()
        self.current_config: Optional[UnifiedEvolutionConfig] = None
        self.reload_count = 0
        self.reload_errors = 0

    def start(self) -> None:
        """Start watching config file in background thread"""
        if self.running:
            logger.warning("Watcher already running")
            return

        self.running = True
        self.watcher_thread = threading.Thread(
            target=self._watch_loop,
            daemon=True,
            name="ConfigFileWatcher"
        )
        self.watcher_thread.start()
        logger.info(f"Started watching {self.config_file}")

    def stop(self) -> None:
        """Stop watching"""
        if not self.running:
            return

        self.running = False

        # Cancel pending reload
        if self.reload_timer:
            self.reload_timer.cancel()
            self.reload_timer = None

        # Wait for thread to finish
        if self.watcher_thread:
            self.watcher_thread.join(timeout=5)
            if self.watcher_thread.is_alive():
                logger.warning("Watcher thread did not stop gracefully")

        logger.info(f"Stopped watching {self.config_file}")

    def register_callback(
        self,
        callback: Callable[[UnifiedEvolutionConfig], None]
    ) -> None:
        """
        Register callback to be called on config change

        Args:
            callback: Function to call with new config
        """
        self.callbacks.append(callback)
        logger.debug(f"Registered callback: {callback.__name__}")

    def unregister_callback(
        self,
        callback: Callable[[UnifiedEvolutionConfig], None]
    ) -> None:
        """
        Unregister a callback

        Args:
            callback: Callback function to remove
        """
        if callback in self.callbacks:
            self.callbacks.remove(callback)
            logger.debug(f"Unregistered callback: {callback.__name__}")

    def _watch_loop(self) -> None:
        """Background loop to watch for file changes"""
        while self.running:
            try:
                # Check for changes
                if self._has_file_changed():
                    logger.info(f"Config file {self.config_file} changed")

                    # Schedule reload after debounce delay
                    if self.reload_timer:
                        self.reload_timer.cancel()

                    self.reload_timer = threading.Timer(
                        self.debounce_delay,
                        self._schedule_reload
                    )
                    self.reload_timer.start()

                # Sleep before next check
                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error in watcher loop: {e}", exc_info=True)
                time.sleep(5)  # Wait before retrying

    def _has_file_changed(self) -> bool:
        """Check if file has changed since last check"""
        try:
            current_stat = self.config_file.stat()
            return (
                current_stat.st_mtime != self.last_modified or
                current_stat.st_size != self.last_size
            )
        except FileNotFoundError:
            logger.error(f"Config file disappeared: {self.config_file}")
            return False
        except Exception as e:
            logger.error(f"Error checking file changes: {e}")
            return False

    def _schedule_reload(self) -> None:
        """Schedule config reload in background thread"""
        if not self.running:
            return

        # Run reload in separate thread to avoid blocking watcher
        reload_thread = threading.Thread(
            target=self._reload_config,
            daemon=True,
            name="ConfigReloader"
        )
        reload_thread.start()

    def _reload_config(self) -> None:
        """Reload configuration from file"""
        try:
            logger.info(f"Reloading config from {self.config_file}")

            # Load new config
            new_config = self._load_config()

            # Validate
            validation_result = self.validators.validate_config(new_config)
            if not validation_result.is_valid:
                self.reload_errors += 1
                logger.error(f"Invalid config, skipping reload: {validation_result.errors}")
                return

            # Update file state
            self.last_modified = self.config_file.stat().st_mtime
            self.last_size = self.config_file.stat().st_size

            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Callback error in {callback.__name__}: {e}", exc_info=True)

            self.current_config = new_config
            self.reload_count += 1
            logger.info(f"Config reloaded successfully (reload #{self.reload_count})")

        except Exception as e:
            self.reload_errors += 1
            logger.error(f"Failed to reload config: {e}", exc_info=True)

    def _load_config(self) -> UnifiedEvolutionConfig:
        """Load config from file"""
        # Determine file type
        suffix = self.config_file.suffix.lower()

        if suffix in ['.yaml', '.yml']:
            return UnifiedEvolutionConfig.from_yaml_file(self.config_file)
        elif suffix == '.json':
            return UnifiedEvolutionConfig.from_json_file(self.config_file)
        else:
            raise ValueError(f"Unsupported config file format: {suffix}")

    def get_stats(self) -> dict:
        """
        Get watcher statistics

        Returns:
            Dictionary with watcher stats
        """
        return {
            "file": str(self.config_file),
            "running": self.running,
            "last_modified": self.last_modified,
            "reload_count": self.reload_count,
            "reload_errors": self.reload_errors,
            "callbacks_registered": len(self.callbacks),
            "poll_interval": self.poll_interval,
            "debounce_delay": self.debounce_delay
        }


class MultiConfigWatcher:
    """
    Watch multiple configuration files for changes

    Manages multiple ConfigFileWatcher instances and coordinates their callbacks.
    """

    def __init__(
        self,
        poll_interval: float = 1.0,
        debounce_delay: float = 2.0
    ):
        """
        Initialize multi-config watcher

        Args:
            poll_interval: Seconds between checks
            debounce_delay: Seconds to wait after last change before reloading
        """
        self.poll_interval = poll_interval
        self.debounce_delay = debounce_delay

        self.watchers: dict = {}  # {file_path: ConfigFileWatcher}
        self.global_callbacks: List[Callable[[str, UnifiedEvolutionConfig], None]] = []

    def add_file(
        self,
        file_path: str,
        callbacks: Optional[List[Callable[[UnifiedEvolutionConfig], None]]] = None
    ) -> ConfigFileWatcher:
        """
        Add a config file to watch

        Args:
            file_path: Path to config file
            callbacks: Optional list of callbacks specific to this file

        Returns:
            ConfigFileWatcher instance
        """
        if file_path in self.watchers:
            logger.warning(f"Already watching {file_path}")
            return self.watchers[file_path]

        watcher = ConfigFileWatcher(
            file_path,
            poll_interval=self.poll_interval,
            debounce_delay=self.debounce_delay
        )

        # Add file-specific callbacks
        if callbacks:
            for callback in callbacks:
                watcher.register_callback(callback)

        # Add global callback
        watcher.register_callback(
            lambda config: self._on_config_change(file_path, config)
        )

        self.watchers[file_path] = watcher
        logger.info(f"Added watcher for {file_path}")

        return watcher

    def remove_file(self, file_path: str) -> None:
        """
        Stop watching a config file

        Args:
            file_path: Path to config file
        """
        if file_path not in self.watchers:
            logger.warning(f"Not watching {file_path}")
            return

        watcher = self.watchers[file_path]
        watcher.stop()
        del self.watchers[file_path]

        logger.info(f"Removed watcher for {file_path}")

    def start_all(self) -> None:
        """Start all watchers"""
        for watcher in self.watchers.values():
            watcher.start()
        logger.info(f"Started {len(self.watchers)} watchers")

    def stop_all(self) -> None:
        """Stop all watchers"""
        for watcher in self.watchers.values():
            watcher.stop()
        logger.info(f"Stopped {len(self.watchers)} watchers")

    def register_global_callback(
        self,
        callback: Callable[[str, UnifiedEvolutionConfig], None]
    ) -> None:
        """
        Register callback for all config changes

        Args:
            callback: Function called with (file_path, new_config)
        """
        self.global_callbacks.append(callback)

    def _on_config_change(
        self,
        file_path: str,
        new_config: UnifiedEvolutionConfig
    ) -> None:
        """Handle config change (called by individual watchers)"""
        for callback in self.global_callbacks:
            try:
                callback(file_path, new_config)
            except Exception as e:
                logger.error(f"Global callback error: {e}", exc_info=True)

    def get_stats(self) -> dict:
        """
        Get statistics for all watchers

        Returns:
            Dictionary with stats for each watcher
        """
        return {
            file_path: watcher.get_stats()
            for file_path, watcher in self.watchers.items()
        }
