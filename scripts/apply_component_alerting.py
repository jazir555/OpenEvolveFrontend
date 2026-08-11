"""
Alerting Integration for Existing Components

This module applies universal alerting to existing core components by
importing and wrapping their key functions with alerting decorators.
"""

import logging
from typing import Any, Dict

from universal_alerting_integration import (
    get_universal_alerting,
    alert_roma_operation,
    alert_decomposition_operation,
    alert_z3_operation,
    alert_crewai_operation,
    alert_knowledge_graph_operation,
    alert_cache_operation,
)

logger = logging.getLogger(__name__)


class ComponentAlertingWrapper:
    """
    Wrapper class that adds alerting to existing component methods.
    """

    def __init__(self):
        """Initialize the alerting wrapper."""
        self.alerting = get_universal_alerting()
        self._wrapped_components = {}

    def wrap_roma_mdap_maker(self):
        """Wrap ROMA-MDAP-MAKER component with alerting."""
        try:
            from roma_mdap_maker_engine import ROMAMDAPMakerEngine

            original_methods = {}

            # Wrap key methods
            for method_name in ['solve', 'decompose', 'validate', 'optimize']:
                if hasattr(ROMAMDAPMakerEngine, method_name):
                    original_method = getattr(ROMAMDAPMakerEngine, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'roma_mdap_maker',
                                f'ROMA.{name}',
                                severity_on_error='error'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(ROMAMDAPMakerEngine, method_name, wrapped_method)

            self._wrapped_components['roma_mdap_maker'] = {
                'class': ROMAMDAPMakerEngine,
                'original_methods': original_methods
            }
            logger.info("ROMA-MDAP-MAKER alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap ROMA-MDAP-MAKER: {e}")

    def wrap_decomposition_engine(self):
        """Wrap Decomposition Engine component with alerting."""
        try:
            from decomposition_engine import DecompositionEngine

            original_methods = {}

            # Wrap key methods
            for method_name in ['decompose', 'analyze', 'validate', 'optimize']:
                if hasattr(DecompositionEngine, method_name):
                    original_method = getattr(DecompositionEngine, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'decomposition_engine',
                                f'Decomposition.{name}',
                                severity_on_error='error'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(DecompositionEngine, method_name, wrapped_method)

            self._wrapped_components['decomposition_engine'] = {
                'class': DecompositionEngine,
                'original_methods': original_methods
            }
            logger.info("Decomposition Engine alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap Decomposition Engine: {e}")

    def wrap_z3_verification(self):
        """Wrap Z3 Verification component with alerting."""
        try:
            from verification_engine import VerificationEngine

            original_methods = {}

            # Wrap key methods
            for method_name in ['verify_with_z3', 'verify_formal', 'verify_constraints']:
                if hasattr(VerificationEngine, method_name):
                    original_method = getattr(VerificationEngine, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'z3_verification',
                                f'Z3.{name}',
                                severity_on_error='warning'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(VerificationEngine, method_name, wrapped_method)

            self._wrapped_components['z3_verification'] = {
                'class': VerificationEngine,
                'original_methods': original_methods
            }
            logger.info("Z3 Verification alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap Z3 Verification: {e}")

    def wrap_crewai_workflows(self):
        """Wrap CrewAI Workflows component with alerting."""
        try:
            from bubblelabs_crewai_bridge import BubbleLabCrewAIBridge

            original_methods = {}

            # Wrap key methods
            for method_name in ['execute_workflow', 'create_crew', 'run_task']:
                if hasattr(BubbleLabCrewAIBridge, method_name):
                    original_method = getattr(BubbleLabCrewAIBridge, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'crewai_workflows',
                                f'CrewAI.{name}',
                                severity_on_error='error'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(BubbleLabCrewAIBridge, method_name, wrapped_method)

            self._wrapped_components['crewai_workflows'] = {
                'class': BubbleLabCrewAIBridge,
                'original_methods': original_methods
            }
            logger.info("CrewAI Workflows alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap CrewAI Workflows: {e}")

    def wrap_knowledge_graph(self):
        """Wrap Knowledge Graph component with alerting."""
        try:
            from bubblelabs_knowledge_integration import KnowledgeGraphVisualizer

            original_methods = {}

            # Wrap key methods
            for method_name in ['build_graph_from_data', 'create_interactive_plot']:
                if hasattr(KnowledgeGraphVisualizer, method_name):
                    original_method = getattr(KnowledgeGraphVisualizer, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'knowledge_graph',
                                f'KnowledgeGraph.{name}',
                                severity_on_error='warning'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(KnowledgeGraphVisualizer, method_name, wrapped_method)

            self._wrapped_components['knowledge_graph'] = {
                'class': KnowledgeGraphVisualizer,
                'original_methods': original_methods
            }
            logger.info("Knowledge Graph alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap Knowledge Graph: {e}")

    def wrap_caching_systems(self):
        """Wrap Caching Systems component with alerting."""
        try:
            from c2c_cache_manager import C2CCacheManager

            original_methods = {}

            # Wrap key methods
            for method_name in ['cache_ensemble_result', 'get_cached_result', 'invalidate']:
                if hasattr(C2CCacheManager, method_name):
                    original_method = getattr(C2CCacheManager, method_name)
                    original_methods[method_name] = original_method

                    # Create wrapped version
                    def create_wrapper(original_func, name):
                        def wrapped(self, *args, **kwargs):
                            with self.alerting.alert_context(
                                'caching_systems',
                                f'Cache.{name}',
                                severity_on_error='warning'
                            ):
                                return original_func(self, *args, **kwargs)
                        return wrapped

                    wrapped_method = create_wrapper(original_method, method_name)
                    setattr(C2CCacheManager, method_name, wrapped_method)

            self._wrapped_components['caching_systems'] = {
                'class': C2CCacheManager,
                'original_methods': original_methods
            }
            logger.info("Caching Systems alerting integration applied")

        except ImportError as e:
            logger.warning(f"Could not wrap Caching Systems: {e}")

    def unwrap_component(self, component_name: str):
        """
        Restore original methods for a component.

        Args:
            component_name: Name of component to unwrap
        """
        if component_name not in self._wrapped_components:
            logger.warning(f"Component {component_name} not wrapped")
            return

        component_info = self._wrapped_components[component_name]
        cls = component_info['class']
        original_methods = component_info['original_methods']

        for method_name, original_method in original_methods.items():
            setattr(cls, method_name, original_method)

        del self._wrapped_components[component_name]
        logger.info(f"Restored original methods for {component_name}")

    def unwrap_all(self):
        """Restore original methods for all wrapped components."""
        for component_name in list(self._wrapped_components.keys()):
            self.unwrap_component(component_name)

    def wrap_all(self):
        """Apply alerting to all supported components."""
        logger.info("Applying universal alerting integration to all components...")
        self.wrap_roma_mdap_maker()
        self.wrap_decomposition_engine()
        self.wrap_z3_verification()
        self.wrap_crewai_workflows()
        self.wrap_knowledge_graph()
        self.wrap_caching_systems()
        logger.info("Universal alerting integration complete")


# Global wrapper instance
_wrapper: ComponentAlertingWrapper = None


def get_alerting_wrapper() -> ComponentAlertingWrapper:
    """Get or create the alerting wrapper singleton."""
    global _wrapper
    if _wrapper is None:
        _wrapper = ComponentAlertingWrapper()
    return _wrapper


def apply_universal_alerting():
    """
    Apply universal alerting to all supported components.

    Call this function during system initialization to enable
    alerting across all major components.
    """
    wrapper = get_alerting_wrapper()
    wrapper.wrap_all()
    logger.info("Universal alerting integration applied successfully")


__all__ = [
    'ComponentAlertingWrapper',
    'get_alerting_wrapper',
    'apply_universal_alerting',
]
