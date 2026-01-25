"""
Adversarial Circular Import Resolver Adapter

Fixes circular import issue between:
- adversarial_maker_integration.py → adversarial.py → red_team.py
- adversarial_maker_integration.py → openevolve_imports.py → adversarial.py

Bug Fixed:
- Line 244 in adversarial_maker_integration.py:
  attack_method: RedTeamStrategy = RedTeamStrategy.ADVERSARIAL
- This uses RedTeamStrategy as a default argument, but RedTeamStrategy
  is set to None when imports fail, causing AttributeError

Solution:
- Lazy import wrapper that defers RedTeamStrategy import until runtime
- Provides fallback value when RedTeamStrategy is unavailable
- No modifications to core adversarial files

Usage:
    from integrations.bug_fixes import AdversarialImportResolver

    # Resolve imports safely
    resolver = AdversarialImportResolver()
    RedTeamStrategy = resolver.get_red_team_strategy()

    # Or use as a context manager
    with AdversarialImportResolver():
        # Code that uses RedTeamStrategy
        pass
"""

import logging
from typing import Any, Optional, Type, TYPE_CHECKING

logger = logging.getLogger(__name__)


class AdversarialImportResolver:
    """
    Resolves circular imports in adversarial system.

    Provides lazy imports and fallback values for:
    - RedTeamStrategy
    - RedTeamMember
    - RedTeamAssessment
    - IssueFinding
    - IssueCategory
    """

    def __init__(self):
        """Initialize the import resolver."""
        self._red_team_module = None
        self._red_team_strategy = None
        self._imports_loaded = False

    def _load_imports(self) -> None:
        """
        Load adversarial imports with error handling.

        Attempts to import from:
        1. openevolve_imports._red_team_module (with fallback)
        2. red_team module directly
        3. Provides stubs if unavailable
        """
        if self._imports_loaded:
            return

        try:
            # Try importing via openevolve_imports first
            from openevolve_imports import _red_team_module
            self._red_team_module = _red_team_module
            self._red_team_strategy = _red_team_module.RedTeamStrategy
            logger.debug("Loaded RedTeamStrategy from openevolve_imports")
        except (ImportError, AttributeError) as e:
            logger.debug(f"Could not load from openevolve_imports: {e}")
            try:
                # Try direct import from red_team
                from red_team import RedTeamStrategy
                self._red_team_strategy = RedTeamStrategy
                logger.debug("Loaded RedTeamStrategy from red_team module")
            except ImportError as e2:
                logger.warning(f"Could not import RedTeamStrategy: {e2}")
                logger.info("RedTeamStrategy will use fallback values")
                self._red_team_strategy = None

        self._imports_loaded = True

    def get_red_team_strategy(self) -> Optional[Type]:
        """
        Get RedTeamStrategy class with safe fallback.

        Returns:
            RedTeamStrategy class if available, None otherwise
        """
        self._load_imports()
        return self._red_team_strategy

    def get_default_strategy(self) -> Any:
        """
        Get default strategy value.

        Returns:
            RedTeamStrategy.ADVERSARIAL if available, else "ADVERSARIAL" string
        """
        strategy_class = self.get_red_team_strategy()

        if strategy_class is not None:
            return strategy_class.ADVERSARIAL
        else:
            # Fallback value
            return "ADVERSARIAL"

    def is_available(self) -> bool:
        """
        Check if RedTeamStrategy is available.

        Returns:
            True if RedTeamStrategy imported successfully
        """
        self._load_imports()
        return self._red_team_strategy is not None

    def __enter__(self):
        """Context manager entry."""
        self._load_imports()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        return False


class RedTeamStrategyProxy:
    """
    Proxy for RedTeamStrategy that handles import failures gracefully.

    This class can be used as a drop-in replacement for RedTeamStrategy
    in default argument values, avoiding the circular import issue.

    Usage:
        # Instead of:
        def __init__(self, attack_method: RedTeamStrategy = RedTeamStrategy.ADVERSARIAL):
            pass

        # Use:
        from integrations.bug_fixes import RedTeamStrategyProxy

        def __init__(self, attack_method = RedTeamStrategyProxy.DEFAULT):
            attack_method = RedTeamStrategyProxy.resolve(attack_method)
            pass
    """

    # Fallback values
    ADVERSARIAL = "ADVERSARIAL"
    MANUAL = "MANUAL"
    AUTOMATED = "AUTOMATED"
    HYBRID = "HYBRID"
    DEFAULT = "ADVERSARIAL"

    @classmethod
    def resolve(cls, value: Any) -> Any:
        """
        Resolve a strategy value to the actual RedTeamStrategy enum.

        Args:
            value: Strategy value (enum or string fallback)

        Returns:
            RedTeamStrategy enum if available, else string value
        """
        resolver = AdversarialImportResolver()
        strategy_class = resolver.get_red_team_strategy()

        if strategy_class is None:
            # Return string fallback
            return value

        if isinstance(value, str):
            # Convert string to enum
            try:
                return getattr(strategy_class, value)
            except AttributeError:
                logger.warning(f"Invalid strategy name: {value}")
                return strategy_class.ADVERSARIAL

        # Already an enum
        return value

    @classmethod
    def get_default(cls) -> Any:
        """
        Get the default strategy value.

        Safe to use in default arguments without triggering imports.
        """
        return cls.DEFAULT


# Convenience functions
def get_red_team_strategy() -> Optional[Type]:
    """
    Quick access to RedTeamStrategy class.

    Usage:
        from integrations.bug_fixes.adversarial_import_resolver import get_red_team_strategy

        RedTeamStrategy = get_red_team_strategy()
        if RedTeamStrategy:
            strategy = RedTeamStrategy.ADVERSARIAL
    """
    resolver = AdversarialImportResolver()
    return resolver.get_red_team_strategy()


def get_default_strategy() -> Any:
    """
    Quick access to default strategy value.

    Usage:
        from integrations.bug_fixes.adversarial_import_resolver import get_default_strategy

        def __init__(self, attack_method = get_default_strategy()):
            attack_method = RedTeamStrategyProxy.resolve(attack_method)
    """
    return RedTeamStrategyProxy.get_default()


# Example of how to patch the core issue (without editing core file)
def patch_adversarial_maker_init():
    """
    Monkey-patch adversarial_maker_integration.MAKERRedTeamMember.__init__
    to use safe default values.

    This is a temporary workaround until the core file is fixed.
    Apply this at application startup:

        from integrations.bug_fixes.adversarial_import_resolver import patch_adversarial_maker_init
        patch_adversarial_maker_init()
    """
    try:
        from adversarial_maker_integration import MAKERRedTeamMember
        original_init = MAKERRedTeamMember.__init__

        def patched_init(self, name, specializations, expertise_level=7,
                        attack_method=None, maker_config=None):
            # Use safe default
            if attack_method is None:
                attack_method = get_default_strategy()

            # Resolve to actual enum if available
            attack_method = RedTeamStrategyProxy.resolve(attack_method)

            # Call original with resolved value
            original_init(self, name, specializations, expertise_level,
                         attack_method, maker_config)

        MAKERRedTeamMember.__init__ = patched_init
        logger.info("Patched MAKERRedTeamMember.__init__ with safe defaults")
        return True

    except ImportError as e:
        logger.error(f"Could not patch MAKERRedTeamMember: {e}")
        return False
