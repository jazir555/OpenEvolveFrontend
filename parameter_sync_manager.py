"""
Parameter Synchronization Manager for OpenEvolve-BubbleLabs Integration

This module provides bi-directional parameter synchronization between 
the UI and BubbleLabs UI, ensuring all OpenEvolve parameters 
are consistently maintained across both interfaces.
"""


from ui_shim import ui as st
import time
from typing import Dict, Any, Callable, List
from dataclasses import dataclass, asdict
from enum import Enum
import json
import threading
from datetime import datetime


class ParameterSyncStatus(Enum):
    """Status of parameter synchronization"""
    SYNCED = "synced"
    OUT_OF_SYNC = "out_of_sync"
    SYNCING = "syncing"
    ERROR = "error"


@dataclass
class ParameterChange:
    """Represents a parameter change event"""
    name: str
    old_value: Any
    new_value: Any
    source_ui: str  # 'ui' or 'bubblelabs'
    timestamp: float
    synced: bool = False


class ParameterSyncManager:
    """
    Manages bi-directional parameter synchronization between UI and BubbleLabs UIs.
    All OpenEvolve parameters from the UI sidebar are synchronized to BubbleLabs UI.
    """
    
    def __init__(self):
        self.parameter_mapping = self._initialize_parameter_mapping()
        self.change_history: List[ParameterChange] = []
        self.last_sync_times: Dict[str, float] = {}
        self.sync_callbacks: List[Callable] = []
        self._lock = threading.Lock()
        
        # Initialize session state for synchronization
        self._ensure_session_state_initialized()
    
    def _ensure_session_state_initialized(self):
        """Ensure all parameter sync session state is initialized"""
        if 'param_sync_initialized' not in st.session_state:
            st.session_state['param_sync_initialized'] = True
            
            # Initialize sync state flags
            st.session_state['params_synced_to_bubblelabs'] = False
            st.session_state['params_synced_from_bubblelabs'] = False
            st.session_state['last_sync_time'] = time.time()
            st.session_state['sync_conflicts'] = []
    
    def _initialize_parameter_mapping(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize the mapping between UI and BubbleLabs parameters.
        This includes all parameters from the UI sidebar that need to be synchronized.
        """
        return {
            # Provider Configuration Parameters
            "provider": {
                "ui_key": "provider",
                "bubblelabs_key": "provider",
                "type": "str",
                "validation": {"options": ["openai", "anthropic", "google", "openrouter", "ollama"]}
            },
            "api_key": {
                "ui_key": "api_key",
                "bubblelabs_key": "api_key",
                "type": "str",
                "validation": {"min_length": 10}
            },
            "base_url": {
                "ui_key": "base_url",
                "bubblelabs_key": "base_url",
                "type": "str",
                "validation": {"is_url": True}
            },
            "model": {
                "ui_key": "model",
                "bubblelabs_key": "model",
                "type": "str",
                "validation": {"options": [
                    "gpt-4", "gpt-4o", "gpt-3.5-turbo", "claude-3-opus", 
                    "claude-3-sonnet", "llama-2-70b", "llama-3-70b", 
                    "gemini-pro", "gemini-1.5-pro", "mistral-large"
                ]}
            },
            
            # Generation Parameters
            "temperature": {
                "ui_key": "temperature",
                "bubblelabs_key": "temperature",
                "type": "float",
                "validation": {"min": 0.0, "max": 2.0}
            },
            "top_p": {
                "ui_key": "top_p",
                "bubblelabs_key": "top_p",
                "type": "float",
                "validation": {"min": 0.0, "max": 1.0}
            },
            "frequency_penalty": {
                "ui_key": "frequency_penalty",
                "bubblelabs_key": "frequency_penalty",
                "type": "float",
                "validation": {"min": -2.0, "max": 2.0}
            },
            "presence_penalty": {
                "ui_key": "presence_penalty",
                "bubblelabs_key": "presence_penalty",
                "type": "float",
                "validation": {"min": -2.0, "max": 2.0}
            },
            "max_tokens": {
                "ui_key": "max_tokens",
                "bubblelabs_key": "max_tokens",
                "type": "int",
                "validation": {"min": 1, "max": 100000}
            },
            "seed": {
                "ui_key": "seed",
                "bubblelabs_key": "seed",
                "type": "int",
                "validation": {"min": -1, "max": 999999}
            },
            
            # Evolution Parameters
            "max_iterations": {
                "ui_key": "max_iterations",
                "bubblelabs_key": "max_iterations",
                "type": "int",
                "validation": {"min": 1, "max": 200}
            },
            "population_size": {
                "ui_key": "population_size",
                "bubblelabs_key": "population_size",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "num_islands": {
                "ui_key": "num_islands",
                "bubblelabs_key": "num_islands",
                "type": "int",
                "validation": {"min": 1, "max": 10}
            },
            "migration_interval": {
                "ui_key": "migration_interval",
                "bubblelabs_key": "migration_interval",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "migration_rate": {
                "ui_key": "migration_rate",
                "bubblelabs_key": "migration_rate",
                "type": "float",
                "validation": {"min": 0.0, "max": 1.0}
            },
            "archive_size": {
                "ui_key": "archive_size",
                "bubblelabs_key": "archive_size",
                "type": "int",
                "validation": {"min": 0, "max": 100}
            },
            "elite_ratio": {
                "ui_key": "elite_ratio",
                "bubblelabs_key": "elite_ratio",
                "type": "float",
                "validation": {"min": 0.0, "max": 1.0}
            },
            "exploration_ratio": {
                "ui_key": "exploration_ratio",
                "bubblelabs_key": "exploration_ratio",
                "type": "float",
                "validation": {"min": 0.0, "max": 1.0}
            },
            "exploitation_ratio": {
                "ui_key": "exploitation_ratio",
                "bubblelabs_key": "exploitation_ratio",
                "type": "float",
                "validation": {"min": 0.0, "max": 1.0}
            },
            "checkpoint_interval": {
                "ui_key": "checkpoint_interval",
                "bubblelabs_key": "checkpoint_interval",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            
            # Advanced Evolution Features
            "enable_qd_evolution": {
                "ui_key": "enable_qd_evolution",
                "bubblelabs_key": "enable_qd_evolution",
                "type": "bool",
                "validation": {}
            },
            "enable_multi_objective": {
                "ui_key": "enable_multi_objective",
                "bubblelabs_key": "enable_multi_objective",
                "type": "bool",
                "validation": {}
            },
            "enable_adversarial": {
                "ui_key": "enable_adversarial",
                "bubblelabs_key": "enable_adversarial",
                "type": "bool",
                "validation": {}
            },
            "enable_symbolic_regression": {
                "ui_key": "enable_symbolic_regression",
                "bubblelabs_key": "enable_symbolic_regression",
                "type": "bool",
                "validation": {}
            },
            "enable_neuroevolution": {
                "ui_key": "enable_neuroevolution",
                "bubblelabs_key": "enable_neuroevolution",
                "type": "bool",
                "validation": {}
            },
            "enable_evolution_tracing": {
                "ui_key": "enable_evolution_tracing",
                "bubblelabs_key": "enable_evolution_tracing",
                "type": "bool",
                "validation": {}
            },
            "enable_artifact_feedback": {
                "ui_key": "enable_artifact_feedback",
                "bubblelabs_key": "enable_artifact_feedback",
                "type": "bool",
                "validation": {}
            },
            "enable_llm_feedback": {
                "ui_key": "enable_llm_feedback",
                "bubblelabs_key": "enable_llm_feedback",
                "type": "bool",
                "validation": {}
            },
            "enable_early_stopping": {
                "ui_key": "enable_early_stopping",
                "bubblelabs_key": "enable_early_stopping",
                "type": "bool",
                "validation": {}
            },
            
            # Performance Optimization
            "memory_limit_mb": {
                "ui_key": "memory_limit_mb",
                "bubblelabs_key": "memory_limit_mb",
                "type": "int",
                "validation": {"min": 100, "max": 32768}
            },
            "cpu_limit": {
                "ui_key": "cpu_limit",
                "bubblelabs_key": "cpu_limit",
                "type": "float",
                "validation": {"min": 0.1, "max": 32.0}
            },
            "parallel_evaluations": {
                "ui_key": "parallel_evaluations",
                "bubblelabs_key": "parallel_evaluations",
                "type": "int",
                "validation": {"min": 1, "max": 32}
            },
            "max_code_length": {
                "ui_key": "max_code_length",
                "bubblelabs_key": "max_code_length",
                "type": "int",
                "validation": {"min": 100, "max": 100000}
            },
            "evaluator_timeout": {
                "ui_key": "evaluator_timeout",
                "bubblelabs_key": "evaluator_timeout",
                "type": "int",
                "validation": {"min": 10, "max": 3600}
            },
            "max_evaluation_retries": {
                "ui_key": "max_evaluation_retries",
                "bubblelabs_key": "max_evaluation_retries",
                "type": "int",
                "validation": {"min": 1, "max": 10}
            },
            
            # Adversarial Testing Parameters
            "red_team_samples": {
                "ui_key": "red_team_samples",
                "bubblelabs_key": "red_team_samples",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "blue_team_samples": {
                "ui_key": "blue_team_samples",
                "bubblelabs_key": "blue_team_samples",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "evaluator_samples": {
                "ui_key": "evaluator_samples",
                "bubblelabs_key": "evaluator_samples",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "confidence_threshold": {
                "ui_key": "confidence_threshold",
                "bubblelabs_key": "confidence_threshold",
                "type": "float",
                "validation": {"min": 0.5, "max": 1.0}
            },
            "max_adversarial_iterations": {
                "ui_key": "max_adversarial_iterations",
                "bubblelabs_key": "max_adversarial_iterations",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            
            # Feature and Quality Parameters
            "feature_dimensions": {
                "ui_key": "feature_dimensions",
                "bubblelabs_key": "feature_dimensions",
                "type": "list",
                "validation": {"options": ["complexity", "diversity", "length", "readability", "performance", "security"]}
            },
            "feature_bins": {
                "ui_key": "feature_bins",
                "bubblelabs_key": "feature_bins",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "diversity_metric": {
                "ui_key": "diversity_metric",
                "bubblelabs_key": "diversity_metric",
                "type": "str",
                "validation": {"options": ["edit_distance", "ast_similarity", "ngram_overlap", "semantic_distance"]}
            },
            
            # Early Stopping Parameters
            "early_stopping_patience": {
                "ui_key": "early_stopping_patience",
                "bubblelabs_key": "early_stopping_patience",
                "type": "int",
                "validation": {"min": 1, "max": 100}
            },
            "convergence_threshold": {
                "ui_key": "convergence_threshold",
                "bubblelabs_key": "convergence_threshold",
                "type": "float",
                "validation": {"min": 0.0, "max": 0.1}
            }
        }
    
    def register_sync_callback(self, callback: Callable):
        """Register a callback function to be called when parameters are synchronized"""
        self.sync_callbacks.append(callback)
    
    def _validate_parameter(self, param_name: str, value: Any) -> bool:
        """Validate a parameter value based on its type and constraints"""
        if param_name not in self.parameter_mapping:
            return False  # Unknown parameters should fail validation
        
        mapping = self.parameter_mapping[param_name]
        validation = mapping.get("validation", {})
        
        # Type validation
        expected_type = mapping["type"]
        if expected_type == "int":
            if not isinstance(value, int):
                return False
        elif expected_type == "float":
            if not isinstance(value, (int, float)):
                return False
        elif expected_type == "str":
            if not isinstance(value, str):
                return False
        elif expected_type == "bool":
            if not isinstance(value, bool):
                return False
        elif expected_type == "list":
            if not isinstance(value, list):
                return False
        
        # Specific validation rules
        if "min" in validation:
            if value < validation["min"]:
                return False
        if "max" in validation:
            if value > validation["max"]:
                return False
        if "min_length" in validation:
            if isinstance(value, str) and len(value) < validation["min_length"]:
                return False
        if "options" in validation:
            if value not in validation["options"]:
                return False
        if "is_url" in validation and validation["is_url"]:
            if isinstance(value, str) and not value.startswith(('http://', 'https://')):
                return False
        
        return True
    
    def _record_parameter_change(self, param_name: str, old_value: Any, new_value: Any, source_ui: str):
        """Record a parameter change for history and conflict detection"""
        change = ParameterChange(
            name=param_name,
            old_value=old_value,
            new_value=new_value,
            source_ui=source_ui,
            timestamp=time.time()
        )
        self.change_history.append(change)
        
        # Keep only recent changes (last 1000)
        if len(self.change_history) > 1000:
            self.change_history = self.change_history[-1000:]
    
    def sync_from_ui_to_bubblelabs(self) -> Dict[str, Any]:
        """
        Synchronize all parameters from UI session state to BubbleLabs.
        This is called when parameters are changed in UI.
        """
        with self._lock:
            changes_made = {}
            errors = []
            
            for param_name, mapping in self.parameter_mapping.items():
                ui_key = mapping["ui_key"]
                
                # Check if parameter exists in UI session state
                if ui_key in st.session_state:
                    ui_value = st.session_state[ui_key]
                    
                    # Validate the parameter
                    if self._validate_parameter(param_name, ui_value):
                        # In a real system, this would update the BubbleLabs system
                        # For now, we'll just record the sync operation
                        old_last_sync = self.last_sync_times.get(param_name, 0)
                        self.last_sync_times[param_name] = time.time()
                        
                        self._record_parameter_change(
                            param_name, 
                            "unknown",  # We don't track old values in this simple implementation
                            ui_value, 
                            "ui"
                        )
                        
                        changes_made[param_name] = {
                            "value": ui_value,
                            "sync_time": time.time(),
                            "from_ui": "ui"
                        }
                    else:
                        errors.append(f"Invalid value for parameter '{param_name}': {ui_value}")
            
            # Update session state to indicate sync status
            st.session_state['params_synced_to_bubblelabs'] = True
            st.session_state['last_sync_time'] = time.time()
            
            # Call registered callbacks
            for callback in self.sync_callbacks:
                try:
                    callback(changes_made, "ui_to_bubblelabs")
                except (TypeError, ValueError, RuntimeError) as e:
                    errors.append(f"Error in sync callback: {str(e)}")
            
            return {
                "status": "success" if not errors else "partial",
                "changes_made": changes_made,
                "errors": errors,
                "timestamp": time.time()
            }
    
    def sync_from_bubblelabs_to_ui(self, bubblelabs_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize parameters from BubbleLabs to UI.
        This is called when parameters are changed in BubbleLabs UI.
        """
        with self._lock:
            changes_made = {}
            errors = []
            
            for param_name, param_value in bubblelabs_params.items():
                if param_name in self.parameter_mapping:
                    mapping = self.parameter_mapping[param_name]
                    ui_key = mapping["ui_key"]
                    
                    # Validate the parameter
                    if self._validate_parameter(param_name, param_value):
                        # Update UI session state
                        st.session_state[ui_key] = param_value
                        
                        old_last_sync = self.last_sync_times.get(param_name, 0)
                        self.last_sync_times[param_name] = time.time()
                        
                        self._record_parameter_change(
                            param_name, 
                            "unknown",  # We don't track old values in this simple implementation
                            param_value, 
                            "bubblelabs"
                        )
                        
                        changes_made[param_name] = {
                            "value": param_value,
                            "sync_time": time.time(),
                            "from_ui": "bubblelabs"
                        }
                    else:
                        errors.append(f"Invalid value for parameter '{param_name}': {param_value}")
            
            # Update session state to indicate sync status
            st.session_state['params_synced_from_bubblelabs'] = True
            st.session_state['last_sync_time'] = time.time()
            
            # Call registered callbacks
            for callback in self.sync_callbacks:
                try:
                    callback(changes_made, "bubblelabs_to_ui")
                except (TypeError, ValueError, RuntimeError) as e:
                    errors.append(f"Error in sync callback: {str(e)}")
            
            return {
                "status": "success" if not errors else "partial",
                "changes_made": changes_made,
                "errors": errors,
                "timestamp": time.time()
            }
    
    def get_parameter_sync_status(self) -> Dict[str, Any]:
        """Get the current synchronization status of all parameters"""
        status = {
            "last_full_sync": st.session_state.get('last_sync_time', 0),
            "params_synced_to_bubblelabs": st.session_state.get('params_synced_to_bubblelabs', False),
            "params_synced_from_bubblelabs": st.session_state.get('params_synced_from_bubblelabs', False),
            "parameter_statuses": {},
            "conflicts": st.session_state.get('sync_conflicts', [])
        }
        
        for param_name, mapping in self.parameter_mapping.items():
            ui_key = mapping["ui_key"]
            
            # Check if parameter exists in UI
            ui_exists = ui_key in st.session_state
            ui_value = st.session_state.get(ui_key) if ui_exists else None
            
            # Determine sync status for this parameter
            last_sync = self.last_sync_times.get(param_name, 0)
            is_synced = last_sync > 0
            
            param_status = {
                "ui_exists": ui_exists,
                "ui_value": ui_value,
                "last_sync_time": last_sync,
                "is_synced": is_synced,
                "validation_status": self._validate_parameter(param_name, ui_value) if ui_exists else False
            }
            
            status["parameter_statuses"][param_name] = param_status
        
        return status
    
    def get_recent_changes(self, limit: int = 50) -> List[ParameterChange]:
        """Get the most recent parameter changes"""
        return self.change_history[-limit:]
    
    def force_resync_all(self) -> Dict[str, Any]:
        """Force resynchronization of all parameters between UIs"""
        # First, sync from UI to BubbleLabs
        ui_sync_result = self.sync_from_ui_to_bubblelabs()
        
        # Then, if we had bubblelabs parameters to sync back, we would do that here
        # In a real system, this would involve getting the current state from BubbleLabs
        bubblelabs_params = {}  # This would come from BubbleLabs in a real implementation
        bubblelabs_sync_result = self.sync_from_bubblelabs_to_ui(bubblelabs_params)
        
        return {
            "ui_to_bubblelabs": ui_sync_result,
            "bubblelabs_to_ui": bubblelabs_sync_result,
            "timestamp": time.time()
        }
    
    def get_sync_metrics(self) -> Dict[str, Any]:
        """Get metrics about parameter synchronization"""
        total_params = len(self.parameter_mapping)
        synced_params = sum(1 for param_name in self.parameter_mapping 
                           if self.last_sync_times.get(param_name, 0) > 0)
        
        return {
            "total_parameters": total_params,
            "synced_parameters": synced_params,
            "sync_percentage": (synced_params / total_params * 100) if total_params > 0 else 0,
            "change_history_size": len(self.change_history),
            "last_sync_time": st.session_state.get('last_sync_time', 0),
            "is_fully_synced": synced_params == total_params
        }


# Global instance of the parameter sync manager
parameter_sync_manager = ParameterSyncManager()


def initialize_parameter_sync():
    """
    Initialize parameter synchronization for the OpenEvolve-BubbleLabs integration.
    This should be called when the application starts.
    """
    # Initialize all the parameters that need to be synchronized
    default_parameters = {
        "provider": "openai",
        "temperature": 0.7,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "max_tokens": 4096,
        "seed": 42,
        "max_iterations": 100,
        "population_size": 50,
        "num_islands": 5,
        "migration_rate": 0.1,
        "archive_size": 100,
        "enable_qd_evolution": False,
        "enable_multi_objective": False,
        "enable_adversarial": False,
        "memory_limit_mb": 2048,
        "cpu_limit": 1.0,
        "confidence_threshold": 0.8,
        "feature_dimensions": ["complexity", "diversity"]
    }
    
    # Set default values in session state if they don't exist
    for param_name, default_value in default_parameters.items():
        if param_name in parameter_sync_manager.parameter_mapping:
            ui_key = parameter_sync_manager.parameter_mapping[param_name]["ui_key"]
            if ui_key not in st.session_state:
                st.session_state[ui_key] = default_value
    
    # Perform initial sync
    result = parameter_sync_manager.force_resync_all()
    return result
