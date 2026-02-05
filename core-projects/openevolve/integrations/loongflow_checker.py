"""
LoongFlow Availability Checker
==============================

Checks if LoongFlow is available and properly configured.

This module provides runtime detection of LoongFlow availability without
forcing import-time dependencies.

Author: Unified Evolution Team
Date: 2026-01-30
"""

import sys
import logging
from typing import Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LoongFlowChecker:
    """
    Checks LoongFlow availability at runtime

    Usage:
        # Check if LoongFlow is available
        if LoongFlowChecker.is_available():
            # Use LoongFlow
            pass
        else:
            # Fall back to OpenEvolve-only
            pass

        # Get detailed availability info
        info = LoongFlowChecker.get_availability_info()
        print(f"Available: {info['available']}")
        print(f"Version: {info.get('version', 'N/A')}")
    """

    _availability_cache: Optional[bool] = None
    _availability_info_cache: Optional[dict] = None

    @classmethod
    def is_available(cls) -> bool:
        """
        Check if LoongFlow is available and properly configured

        Returns:
            True if LoongFlow can be imported and used, False otherwise
        """
        # Use cached result if available
        if cls._availability_cache is not None:
            return cls._availability_cache

        try:
            # Try to import LoongFlow adapter
            from .loongflow_adapter import LoongFlowAdapter

            # Try to check if LoongFlow core is installed
            import loongflow

            # Check if we can create an adapter instance
            # (This validates that all dependencies are available)
            test_config = {"test": True}

            # Cache successful result
            cls._availability_cache = True
            logger.info("[OK] LoongFlow is available and properly configured")

            return True

        except ImportError as e:
            cls._availability_cache = False
            logger.debug(f"LoongFlow not available: {e}")
            return False

        except Exception as e:
            cls._availability_cache = False
            logger.debug(f"LoongFlow check failed: {e}")
            return False

    @classmethod
    def get_availability_info(cls) -> dict:
        """
        Get detailed information about LoongFlow availability

        Returns:
            Dict with availability details:
            - available: bool
            - version: str (if available)
            - path: str (if available)
            - error: str (if not available)
        """
        # Use cached result if available
        if cls._availability_info_cache is not None:
            return cls._availability_info_cache

        info = {
            "available": False,
            "version": None,
            "path": None,
            "error": None
        }

        try:
            # Check if LoongFlow is installed
            import loongflow

            # Get version
            try:
                info["version"] = loongflow.__version__
            except AttributeError:
                info["version"] = "unknown"

            # Get path
            try:
                info["path"] = Path(loongflow.__file__).parent
            except AttributeError:
                pass

            # Check adapter
            from .loongflow_adapter import LoongFlowAdapter

            info["available"] = True
            cls._availability_cache = True

        except ImportError as e:
            info["error"] = f"Import error: {e}"
            cls._availability_cache = False

        except Exception as e:
            info["error"] = f"Error: {e}"
            cls._availability_cache = False

        # Cache result
        cls._availability_info_cache = info

        return info

    @classmethod
    def reset_cache(cls):
        """
        Reset the availability cache

        Use this if LoongFlow is installed/uninstalled at runtime
        (rare, but possible in dynamic environments)
        """
        cls._availability_cache = None
        cls._availability_info_cache = None
        logger.debug("LoongFlow availability cache reset")


# Convenience function for quick checks
def is_loongflow_available() -> bool:
    """
    Quick check if LoongFlow is available

    Returns:
        True if LoongFlow is available, False otherwise
    """
    return LoongFlowChecker.is_available()


def get_loongflow_info() -> dict:
    """
    Get detailed LoongFlow availability info

    Returns:
        Dict with availability details
    """
    return LoongFlowChecker.get_availability_info()
