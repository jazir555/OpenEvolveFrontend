"""
LoongFlow Availability Checker

Checks if LoongFlow is installed and available for use with OpenEvolve.
Provides detailed diagnostics about LoongFlow installation and capabilities.
"""

import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


class LoongFlowChecker:
    """
    Check LoongFlow availability and capabilities.

    This class provides static methods to detect whether LoongFlow is installed,
    check its version, verify that required components are available, and
    determine if it can be used for evolution.

    Example:
        >>> if LoongFlowChecker.is_available():
        ...     version = LoongFlowChecker.get_version()
        ...     print(f"LoongFlow {version} is available")
        ... else:
        ...     issues = LoongFlowChecker.check_requirements()
        ...     print(f"LoongFlow not available: {issues}")
    """

    @staticmethod
    def is_installed() -> bool:
        """
        Check if LoongFlow package is installed.

        Attempts to import the loongflow package to determine if it's
        installed in the current Python environment.

        Returns:
            True if loongflow can be imported, False otherwise
        """
        try:
            import loongflow  # noqa: F401
            return True
        except ImportError:
            return False

    @staticmethod
    def get_version() -> Optional[str]:
        """
        Get LoongFlow version if available.

        Attempts to retrieve the version string from the loongflow package.
        Returns None if LoongFlow is not installed or version cannot be determined.

        Returns:
            Version string (e.g., "0.1.0") or None if not available
        """
        if not LoongFlowChecker.is_installed():
            return None

        try:
            import loongflow

            # Try to get version from __version__ attribute
            if hasattr(loongflow, '__version__'):
                return loongflow.__version__

            # Try to get version from VERSION attribute
            if hasattr(loongflow, 'VERSION'):
                return loongflow.VERSION

            # Version exists but can't determine it
            return "unknown"

        except Exception as e:
            logger.warning(f"Failed to get LoongFlow version: {e}")
            return None

    @staticmethod
    def check_requirements() -> List[str]:
        """
        Check if LoongFlow requirements are met.

        Performs detailed checks on LoongFlow's core components to determine
        if all required functionality is available. Returns a list of any
        issues found.

        Returns:
            List of issue descriptions. Empty list means all checks passed.
        """
        issues = []

        # Check if LoongFlow is installed
        if not LoongFlowChecker.is_installed():
            issues.append("LoongFlow package not installed")
            return issues

        # Check for GeneralEvolveAgent
        try:
            from loongflow.agents.general_agent import GeneralEvolveAgent  # noqa: F401
        except ImportError as e:
            issues.append(f"Cannot import GeneralEvolveAgent: {e}")

        # Check for PES context (if applicable)
        try:
            from loongflow.framework.pes.context import PESContext  # noqa: F401
        except ImportError:
            # This is optional - may not exist in all LoongFlow versions
            pass

        # Check for core memory components
        try:
            from loongflow.agentsdk.memory.evolution.base_memory import BaseMemory  # noqa: F401
        except ImportError:
            # This is optional
            pass

        return issues

    @staticmethod
    def is_available(requirement_check: bool = False) -> bool:
        """
        Check if LoongFlow is available for use.

        Args:
            requirement_check: If True, perform deep check (slower but more thorough).
                If False, only checks if package is installed.

        Returns:
            True if LoongFlow can be used, False otherwise
        """
        if not LoongFlowChecker.is_installed():
            return False

        if requirement_check:
            issues = LoongFlowChecker.check_requirements()
            return len(issues) == 0

        return True

    @staticmethod
    def get_diagnostics() -> dict:
        """
        Get comprehensive diagnostics about LoongFlow availability.

        Returns a dictionary with detailed information about LoongFlow's
        installation status, version, and any issues found.

        Returns:
            Dictionary with keys:
                - installed: bool - Whether LoongFlow is installed
                - version: str or None - Version if available
                - available: bool - Whether LoongFlow can be used
                - issues: List[str] - List of any issues found
                - components: Dict[str, bool] - Status of individual components
        """
        installed = LoongFlowChecker.is_installed()
        version = LoongFlowChecker.get_version()
        issues = LoongFlowChecker.check_requirements()
        available = len(issues) == 0

        # Check individual components
        components = {
            "general_agent": False,
            "pes_context": False,
            "memory_system": False,
        }

        if installed:
            try:
                from loongflow.agents.general_agent import GeneralEvolveAgent  # noqa: F401
                components["general_agent"] = True
            except ImportError:
                pass

            try:
                from loongflow.framework.pes.context import PESContext  # noqa: F401
                components["pes_context"] = True
            except ImportError:
                pass

            try:
                from loongflow.agentsdk.memory.evolution.base_memory import BaseMemory  # noqa: F401
                components["memory_system"] = True
            except ImportError:
                pass

        return {
            "installed": installed,
            "version": version,
            "available": available,
            "issues": issues,
            "components": components,
        }

    @staticmethod
    def print_diagnostics():
        """
        Print human-readable diagnostics to console.

        Useful for debugging and user-facing status messages.
        """
        diagnostics = LoongFlowChecker.get_diagnostics()

        print("\n" + "=" * 60)
        print("LoongFlow Availability Diagnostics")
        print("=" * 60)

        print(f"\nInstalled: {diagnostics['installed']}")
        print(f"Version: {diagnostics['version'] or 'N/A'}")
        print(f"Available: {diagnostics['available']}")

        print("\nComponents:")
        for component, status in diagnostics['components'].items():
            status_str = "✓" if status else "✗"
            print(f"  {status_str} {component}")

        if diagnostics['issues']:
            print("\nIssues:")
            for issue in diagnostics['issues']:
                print(f"  • {issue}")
        else:
            print("\n✓ All requirements met")

        print("=" * 60 + "\n")
