"""Pressure Valve - Meta-cognitive monitoring system.

Monitors reasoning pressure and triggers system switching between:
- Soar (System 2): Default symbolic reasoning
- ACT-R (System 1): When pressure builds
- Evolutionary: When pressure >= threshold

Pressure Calculation:
    Pressure = w1×depth + w2×time_factor + w3×impasse_count + w4×ambiguity
    where weights sum to 1.0

Pressure Metrics:
    - Subgoal depth (recursion level)
    - Time in current state (ms)
    - Number of impasses encountered
    - Ambiguity score (number of competing operators)
    - Memory pressure (working memory load)

Thresholds:
    - soar_to_actr_depth: 3 (subgoal depth before ACT-R)
    - actr_to_evo_pressure: 0.9 (pressure for evolutionary fallback)
    - time_threshold_ms: 500 (time before switching)
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum, auto

from .config import PressureValveConfig
from .soar_engine import Impasse

logger = logging.getLogger(__name__)


class SystemType(Enum):
    """Available reasoning systems."""
    SOAR = "soar"  # System 2 - Symbolic
    ACT_R = "act_r"  # System 1 - Heuristic
    EVOLUTIONARY = "evolutionary"  # GA fallback


@dataclass
class PressureMetrics:
    """Metrics used for pressure calculation."""
    subgoal_depth: int = 0
    time_in_state_ms: float = 0.0
    impasse_count: int = 0
    ambiguity_score: int = 0
    memory_load: int = 0
    
    # Additional metrics
    cycle_count: int = 0
    consecutive_failures: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "subgoal_depth": self.subgoal_depth,
            "time_in_state_ms": self.time_in_state_ms,
            "impasse_count": self.impasse_count,
            "ambiguity_score": self.ambiguity_score,
            "memory_load": self.memory_load,
            "cycle_count": self.cycle_count,
            "consecutive_failures": self.consecutive_failures,
        }


@dataclass
class ThresholdConfig:
    """Configurable thresholds for system switching."""
    
    # Depth threshold: switch from Soar to ACT-R
    soar_to_actr_depth: int = 3
    
    # Pressure threshold: switch from ACT-R to Evolutionary
    actr_to_evo_pressure: float = 0.9
    
    # Time threshold (ms) before considering switch
    time_threshold_ms: int = 500
    
    # Impasse count before considering switch
    impasse_threshold: int = 5
    
    # Ambiguity threshold (number of competing operators)
    ambiguity_threshold: int = 3
    
    # Memory load threshold
    memory_threshold: int = 7


class PressureMonitor:
    """Tracks cognitive pressure metrics."""
    
    def __init__(self, config: PressureValveConfig):
        self.config = config
        self.metrics_history: list = []
        self.current_metrics = PressureMetrics()
        self.state_start_time: Optional[float] = None
        self.impasse_count = 0
    
    def start_state_monitoring(self):
        """Start monitoring a new state."""
        self.state_start_time = time.time() * 1000  # ms
        self.current_metrics = PressureMetrics()
    
    def update_metrics(
        self,
        subgoal_depth: int = 0,
        impasse: Optional[Impasse] = None,
        ambiguity_score: int = 0,
        memory_load: int = 0,
        cycle_count: int = 0,
        failure: bool = False
    ):
        """Update pressure metrics."""
        # Calculate time in state
        if self.state_start_time:
            self.current_metrics.time_in_state_ms = (
                time.time() * 1000 - self.state_start_time
            )
        
        self.current_metrics.subgoal_depth = subgoal_depth
        self.current_metrics.ambiguity_score = ambiguity_score
        self.current_metrics.memory_load = memory_load
        self.current_metrics.cycle_count = cycle_count
        
        if impasse:
            self.impasse_count += 1
            self.current_metrics.impasse_count = self.impasse_count
        
        if failure:
            self.current_metrics.consecutive_failures += 1
        else:
            self.current_metrics.consecutive_failures = 0
        
        # Record history
        self.metrics_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": self.current_metrics.to_dict()
        })
    
    def compute_pressure(self) -> float:
        """
        Calculate pressure using weighted formula:
        Pressure = w1×depth + w2×time_factor + w3×impasse_count + w4×ambiguity
        """
        metrics = self.current_metrics
        
        # Normalize metrics to 0-1 range
        depth_factor = min(1.0, metrics.subgoal_depth / self.config.soar_to_actr_depth)
        
        time_factor = min(1.0, metrics.time_in_state_ms / self.config.time_threshold_ms)
        
        impasse_factor = min(1.0, metrics.impasse_count / 10)  # Normalize by 10
        
        ambiguity_factor = min(1.0, metrics.ambiguity_score / 5)  # Normalize by 5
        
        # Calculate weighted pressure
        pressure = (
            self.config.weight_depth * depth_factor +
            self.config.weight_time * time_factor +
            self.config.weight_impasses * impasse_factor +
            self.config.weight_ambiguity * ambiguity_factor
        )
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, pressure))
    
    def get_current_metrics(self) -> PressureMetrics:
        """Get current metrics."""
        return self.current_metrics
    
    def reset(self):
        """Reset all metrics."""
        self.metrics_history = []
        self.current_metrics = PressureMetrics()
        self.state_start_time = None
        self.impasse_count = 0


class SystemSwitcher:
    """Handles switching between reasoning systems."""
    
    def __init__(self, config: PressureValveConfig):
        self.config = config
        self.current_system: SystemType = SystemType.SOAR
        self.switch_history: list = []
        self.thresholds = ThresholdConfig(
            soar_to_actr_depth=config.soar_to_actr_depth,
            actr_to_evo_pressure=config.actr_to_evo_pressure,
            time_threshold_ms=config.time_threshold_ms
        )
    
    def get_current_system(self) -> SystemType:
        """Get currently active system."""
        return self.current_system
    
    def switch_system(self, target: SystemType, reason: str) -> bool:
        """
        Switch to a different reasoning system.
        
        Returns:
            True if switch was successful
        """
        if target == self.current_system:
            return True
        
        old_system = self.current_system
        self.current_system = target
        
        self.switch_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from": old_system.value,
            "to": target.value,
            "reason": reason
        })
        
        logger.info(f"Switched from {old_system.value} to {target.value}: {reason}")
        return True
    
    def should_switch_to_actr(
        self,
        metrics: PressureMetrics,
        soar_state: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Check if we should switch from Soar to ACT-R."""
        # Check depth
        if metrics.subgoal_depth >= self.thresholds.soar_to_actr_depth:
            return True, f"Subgoal depth {metrics.subgoal_depth} >= threshold"
        
        # Check time
        if metrics.time_in_state_ms >= self.thresholds.time_threshold_ms:
            return True, f"Time {metrics.time_in_state_ms:.0f}ms >= threshold"
        
        # Check impasses
        if metrics.impasse_count >= self.thresholds.impasse_threshold:
            return True, f"Impasse count {metrics.impasse_count} >= threshold"
        
        # Check ambiguity
        if metrics.ambiguity_score >= self.thresholds.ambiguity_threshold:
            return True, f"Ambiguity {metrics.ambiguity_score} >= threshold"
        
        return False, ""
    
    def should_switch_to_evolutionary(
        self,
        pressure: float,
        actr_failure: bool = False
    ) -> Tuple[bool, str]:
        """Check if we should switch from ACT-R to Evolutionary."""
        if actr_failure:
            return True, "ACT-R failure"
        
        if pressure >= self.thresholds.actr_to_evo_pressure:
            return True, f"Pressure {pressure:.2f} >= threshold"
        
        return False, ""


class PressureValve:
    """
    Main Pressure Valve - Meta-cognitive monitoring system.
    
    Coordinates system switching based on pressure metrics.
    """
    
    def __init__(self, config: Optional[PressureValveConfig] = None):
        self.config = config or PressureValveConfig()
        self.monitor = PressureMonitor(self.config)
        self.switcher = SystemSwitcher(self.config)
        
        # Callbacks for system switching
        self.on_switch_to_actr: Optional[callable] = None
        self.on_switch_to_evolutionary: Optional[callable] = None
        self.on_switch_to_soar: Optional[callable] = None
    
    def register_callbacks(
        self,
        on_switch_to_actr: Optional[callable] = None,
        on_switch_to_evolutionary: Optional[callable] = None,
        on_switch_to_soar: Optional[callable] = None
    ):
        """Register callbacks for system switching."""
        self.on_switch_to_actr = on_switch_to_actr
        self.on_switch_to_evolutionary = on_switch_to_evolutionary
        self.on_switch_to_soar = on_switch_to_soar
    
    def compute_pressure(self, state: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Calculate pressure 0-1."""
        # Update metrics from context
        self.monitor.update_metrics(
            subgoal_depth=context.get("subgoal_depth", 0),
            ambiguity_score=context.get("ambiguity_score", 0),
            memory_load=context.get("memory_load", 0),
            cycle_count=context.get("cycle_count", 0),
            failure=context.get("failure", False)
        )
        
        return self.monitor.compute_pressure()
    
    def check_and_switch(
        self,
        soar_state: Dict[str, Any],
        actr_state: Dict[str, Any],
        actr_failure: bool = False
    ) -> SystemType:
        """
        Check thresholds and switch systems if needed.
        
        Returns:
            Current system type after any switches
        """
        current = self.switcher.get_current_system()
        metrics = self.monitor.get_current_metrics()
        pressure = self.monitor.compute_pressure()
        
        # Check for evolutionary fallback (highest priority)
        if current in [SystemType.SOAR, SystemType.ACT_R]:
            should_switch, reason = self.switcher.should_switch_to_evolutionary(
                pressure, actr_failure
            )
            if should_switch:
                self.switcher.switch_system(SystemType.EVOLUTIONARY, reason)
                if self.on_switch_to_evolutionary:
                    self.on_switch_to_evolutionary()
                return SystemType.EVOLUTIONARY
        
        # Check for ACT-R switch
        if current == SystemType.SOAR:
            should_switch, reason = self.switcher.should_switch_to_actr(metrics, soar_state)
            if should_switch:
                self.switcher.switch_system(SystemType.ACT_R, reason)
                if self.on_switch_to_actr:
                    self.on_switch_to_actr()
                return SystemType.ACT_R
        
        # Check if we can switch back to Soar (pressure reduced)
        if current == SystemType.ACT_R:
            if (metrics.subgoal_depth < self.config.soar_to_actr_depth and
                pressure < 0.5):
                self.switcher.switch_system(SystemType.SOAR, "Pressure reduced")
                if self.on_switch_to_soar:
                    self.on_switch_to_soar()
                return SystemType.SOAR
        
        return current
    
    def get_pressure_metrics(self) -> Dict[str, Any]:
        """Get current pressure metrics."""
        return {
            "pressure": self.monitor.compute_pressure(),
            "metrics": self.monitor.get_current_metrics().to_dict(),
            "current_system": self.switcher.get_current_system().value,
            "switch_history": self.switcher.switch_history
        }
    
    def start_monitoring(self):
        """Start monitoring a new reasoning session."""
        self.monitor.start_state_monitoring()
        self.switcher.current_system = SystemType.SOAR
    
    def record_impasse(self, impasse: Impasse):
        """Record an impasse occurrence."""
        self.monitor.update_metrics(impasse=impasse)
    
    def record_failure(self):
        """Record a failure."""
        self.monitor.update_metrics(failure=True)
    
    def reset(self):
        """Reset the pressure valve."""
        self.monitor.reset()
        self.switcher.switch_history = []
        self.switcher.current_system = SystemType.SOAR
    
    def get_stats(self) -> Dict[str, Any]:
        """Get valve statistics."""
        return {
            "current_system": self.switcher.get_current_system().value,
            "pressure": self.monitor.compute_pressure(),
            "metrics": self.monitor.get_current_metrics().to_dict(),
            "switch_count": len(self.switcher.switch_history),
            "switch_history": self.switcher.switch_history
        }
