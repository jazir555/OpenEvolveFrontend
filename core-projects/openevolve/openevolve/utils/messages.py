"""
User-Facing Messages for LoongFlow Integration

Provides clear, informative messages for users about LoongFlow status,
availability, and fallback behavior.
"""

from typing import Optional


class LoongFlowMessages:
    """
    User-facing messages for LoongFlow status and behavior.

    This class provides formatted, user-friendly messages for various
    scenarios related to LoongFlow integration:
        - LoongFlow is disabled by configuration
        - LoongFlow is not installed
        - LoongFlow is required but not available
        - System is falling back to OpenEvolve-only mode

    Example:
        >>> if not LoongFlowChecker.is_available():
        ...     print(LoongFlowMessages.not_available_message(fallback_enabled=True))
    """

    @staticmethod
    def disabled_message() -> str:
        """
        Message when LoongFlow is disabled in configuration.

        Returns:
            Formatted message explaining that LoongFlow is disabled and
            OpenEvolve-only mode will be used.
        """
        return """
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Disabled                                          ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow PES has been disabled in the configuration.        ║
║  Evolution will proceed using OpenEvolve-only mode.           ║
║                                                              ║
║  OpenEvolve modes available:                                 ║
║    • Standard - Basic evolutionary optimization              ║
║    • QD (Quality-Diversity) - Behavioral space exploration   ║
║    • MO (Multi-Objective) - Pareto optimization             ║
║    • Adversarial - Robustness testing                        ║
║                                                              ║
║  To enable LoongFlow, set enable_loongflow=True              ║
╚═══════════════════════════════════════════════════════════════╝
        """.strip()

    @staticmethod
    def not_available_message(fallback_enabled: bool = True) -> str:
        """
        Message when LoongFlow is not installed.

        Args:
            fallback_enabled: Whether automatic fallback to OpenEvolve is enabled

        Returns:
            Formatted message explaining LoongFlow is not available and
            what actions the user can take.
        """
        if fallback_enabled:
            return """
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Not Available                                    ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow package is not installed.                        ║
║  Automatically falling back to OpenEvolve-only mode.         ║
║                                                              ║
║  To install LoongFlow:                                      ║
║    pip install git+https://github.com/baidu-baige/LoongFlow.git║
║                                                              ║
║  To disable fallback and require LoongFlow:                  ║
║    Set require_loongflow=True in configuration               ║
╚═══════════════════════════════════════════════════════════════╝
            """.strip()
        else:
            return """
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Required But Not Available                        ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow package is not installed but is required.          ║
║  Please install LoongFlow or disable require_loongflow.       ║
║                                                              ║
║  To install LoongFlow:                                      ║
║    pip install git+https://github.com/baidu-baige/LoongFlow.git║
║                                                              ║
║  To disable the requirement:                                ║
║    Set require_loongflow=False in configuration              ║
╚═══════════════════════════════════════════════════════════════╝
            """.strip()

    @staticmethod
    def using_openevolve_message(mode: str = "standard") -> str:
        """
        Message when using OpenEvolve-only mode.

        Args:
            mode: The evolution mode being used

        Returns:
            Formatted message explaining OpenEvolve mode is active
        """
        return f"""
╔═══════════════════════════════════════════════════════════════╗
║  Using OpenEvolve-Only Mode                                 ║
║  ────────────────────────────────────────────────────────────  ║
║  Evolution will proceed using OpenEvolve.                    ║
║  Mode: {mode:<55} ║
║                                                              ║
║  OpenEvolve provides:                                        ║
║    • Quality-Diversity optimization (MAP-Elites)             ║
║    • Multi-objective optimization (NSGA-II)                  ║
║    • Adversarial co-evolution                                ║
║    • Island model parallelism                               ║
║    • Steady-state evolution                                   ║
║                                                              ║
║  All evolutionary features are available.                    ║
╚═══════════════════════════════════════════════════════════════╝
        """.strip()

    @staticmethod
    def using_loongflow_message() -> str:
        """
        Message when LoongFlow is successfully initialized.

        Returns:
            Formatted message confirming LoongFlow is in use
        """
        return """
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow PES Initialized                                  ║
║  ────────────────────────────────────────────────────────────  ║
║  LoongFlow Plan-Execute-Summarize system is active.          ║
║                                                              ║
║  Features enabled:                                           ║
║    • Automated planning strategies                           ║
║    • Memory-guided evolution                                ║
║    • Execution pattern learning                             ║
║    • Multi-iteration summarization                          ║
║                                                              ║
║  Evolution will benefit from advanced PES capabilities.      ║
╚═══════════════════════════════════════════════════════════════╝
        """.strip()

    @staticmethod
    def initialization_failed_message(error: str, fallback_enabled: bool) -> str:
        """
        Message when LoongFlow initialization fails.

        Args:
            error: The error message
            fallback_enabled: Whether fallback is enabled

        Returns:
            Formatted error message with next steps
        """
        base_message = f"""
╔═══════════════════════════════════════════════════════════════╗
║  LoongFlow Initialization Failed                            ║
║  ────────────────────────────────────────────────────────────  ║
║  Failed to initialize LoongFlow:                             ║
║  {error[:54]}... ║
"""

        if fallback_enabled:
            base_message += """
║                                                              ║
║  Falling back to OpenEvolve-only mode.                      ║
║  All features will remain available.                         ║
"""
        else:
            base_message += """
║                                                              ║
║  Fallback is disabled. Please fix the issue or enable       ║
║  fallback by setting require_loongflow=False.               ║
"""

        base_message += """
╚═══════════════════════════════════════════════════════════════╝
        """.strip()

        return base_message

    @staticmethod
    def capability_summary(
        loongflow_available: bool,
        openevolve_mode: str,
        capabilities: dict
    ) -> str:
        """
        Summary of system capabilities.

        Args:
            loongflow_available: Whether LoongFlow is available
            openevolve_mode: OpenEvolve mode being used
            capabilities: Dictionary of capabilities

        Returns:
            Formatted capability summary
        """
        system = "LoongFlow PES" if loongflow_available else f"OpenEvolve ({openevolve_mode})"

        summary = f"""
╔═══════════════════════════════════════════════════════════════╗
║  Evolution System Capabilities                              ║
║  ────────────────────────────────────────────────────────────  ║
║  Active System: {system:<45} ║
║                                                              ║
║  Features:                                                   ║
"""

        if loongflow_available:
            summary += """
║    ✓ Plan-Execute-Summarize workflow                        ║
║    ✓ Memory-guided evolution                                ║
║    ✓ Automated strategy selection                           ║
║    ✓ Multi-step reasoning                                   ║
"""
        else:
            summary += """
║    ✓ Quality-Diversity optimization                         ║
║    ✓ Multi-objective Pareto fronts                          ║
║    ✓ Adversarial robustness testing                         ║
║    ✓ Island model parallelism                              ║
"""

        summary += """
║                                                              ║
║  System is ready for evolution tasks.                       ║
╚═══════════════════════════════════════════════════════════════╝
        """.strip()

        return summary

    @staticmethod
    def log_diagnostics(diagnostics: dict) -> str:
        """
        Log-friendly diagnostics message.

        Args:
            diagnostics: Diagnostics dictionary from LoongFlowChecker

        Returns:
            Formatted diagnostics string for logging
        """
        lines = [
            "LoongFlow Diagnostics:",
            f"  Installed: {diagnostics['installed']}",
            f"  Version: {diagnostics['version'] or 'N/A'}",
            f"  Available: {diagnostics['available']}",
            "",
            "Components:"
        ]

        for component, status in diagnostics['components'].items():
            status_str = "✓" if status else "✗"
            lines.append(f"    {status_str} {component}")

        if diagnostics['issues']:
            lines.append("")
            lines.append("Issues:")
            for issue in diagnostics['issues']:
                lines.append(f"    • {issue}")

        return "\n".join(lines)
