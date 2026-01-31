"""
AdaptationEngine - Adaptive Learning System for OpenEvolve Knowledge Engine

Learns from operational experience and adapts system behavior to improve performance.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class Experience:
    """Single experience/observation for learning."""
    query: str
    success: bool
    processing_time_ms: float
    components_used: List[str]
    error_type: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: f"exp_{datetime.now(timezone.utc).timestamp()}")


@dataclass
class ComponentProfile:
    """Performance profile for a system component."""
    name: str
    total_invocations: int = 0
    successful_invocations: int = 0
    total_processing_time_ms: float = 0.0
    error_types: Dict[str, int] = field(default_factory=dict)
    average_success_rate: float = 1.0
    average_processing_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def record_invocation(self, success: bool, processing_time_ms: float, error_type: Optional[str] = None):
        """Record a component invocation."""
        self.total_invocations += 1
        if success:
            self.successful_invocations += 1
        else:
            self.error_types[error_type or "unknown"] = self.error_types.get(error_type or "unknown", 0) + 1
        
        self.total_processing_time_ms += processing_time_ms
        self.average_processing_time_ms = self.total_processing_time_ms / self.total_invocations
        self.average_success_rate = self.successful_invocations / self.total_invocations
        self.last_updated = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "average_success_rate": self.average_success_rate,
            "average_processing_time_ms": self.average_processing_time_ms,
            "error_types": self.error_types.copy(),
            "last_updated": self.last_updated.isoformat()
        }


@dataclass
class AdaptationStrategy:
    """Strategy for adapting system behavior."""
    target_component: str
    action: str  # e.g., "increase_timeout", "reduce_complexity", "disable_component"
    reason: str
    confidence: float
    expected_improvement: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AdaptationEngine:
    """
    Engine for adaptive learning from operational experience.
    
    Learns patterns from successes and failures, adapts component configurations,
    and suggests improvements based on accumulated experience.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        memory_retention_hours: int = 168,  # 7 days
        experience_buffer_size: int = 1000,
        enable_auto_adaptation: bool = True
    ):
        self.learning_rate = learning_rate
        self.memory_retention_hours = memory_retention_hours
        self.experience_buffer_size = experience_buffer_size
        self.enable_auto_adaptation = enable_auto_adaptation
        
        # Experience storage
        self.experiences: deque = deque(maxlen=experience_buffer_size)
        self.component_profiles: Dict[str, ComponentProfile] = {}
        
        # Adaptation state
        self.adaptation_strategies: List[AdaptationStrategy] = []
        self.active_adaptations: Dict[str, Any] = {}
        self.adaptation_callbacks: List[Callable] = []
        
        # Performance tracking
        self.global_stats = {
            "total_experiences": 0,
            "successful_experiences": 0,
            "total_processing_time_ms": 0,
            "average_processing_time_ms": 0,
            "global_success_rate": 1.0
        }
        
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info({
            "msg": "AdaptationEngine initialized",
            "learning_rate": learning_rate,
            "memory_retention_hours": memory_retention_hours,
            "experience_buffer_size": experience_buffer_size
        })
    
    async def start(self):
        """Start the adaptation engine and background tasks."""
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info({"msg": "AdaptationEngine started"})
    
    async def stop(self):
        """Stop the adaptation engine and cleanup."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        logger.info({"msg": "AdaptationEngine stopped"})
    
    async def record_experience(
        self,
        query: str,
        success: bool,
        processing_time_ms: float,
        components_used: List[str],
        error_type: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Experience:
        """
        Record a new experience for learning.
        
        Args:
            query: The query or input that was processed
            success: Whether the operation succeeded
            processing_time_ms: Time taken to process
            components_used: List of component names used
            error_type: Type of error if failed
            metadata: Additional metadata about the experience
            
        Returns:
            The recorded Experience object
        """
        experience = Experience(
            query=query,
            success=success,
            processing_time_ms=processing_time_ms,
            components_used=components_used,
            error_type=error_type,
            metadata=metadata or {}
        )
        
        async with self._lock:
            self.experiences.append(experience)
            self.global_stats["total_experiences"] += 1
            if success:
                self.global_stats["successful_experiences"] += 1
            self.global_stats["total_processing_time_ms"] += processing_time_ms
            
            # Update global averages
            total = self.global_stats["total_experiences"]
            self.global_stats["global_success_rate"] = (
                self.global_stats["successful_experiences"] / total
            )
            self.global_stats["average_processing_time_ms"] = (
                self.global_stats["total_processing_time_ms"] / total
            )
            
            # Update component profiles
            for component in components_used:
                if component not in self.component_profiles:
                    self.component_profiles[component] = ComponentProfile(name=component)
                
                comp_time = processing_time_ms / len(components_used)  # Approximate per-component time
                self.component_profiles[component].record_invocation(
                    success=success,
                    processing_time_ms=comp_time,
                    error_type=error_type
                )
        
        # Trigger adaptation analysis if auto-adaptation is enabled
        if self.enable_auto_adaptation:
            await self._analyze_and_adapt()
        
        logger.debug({
            "msg": "Experience recorded",
            "experience_id": experience.id,
            "success": success,
            "components": components_used
        })
        
        return experience
    
    async def get_component_performance(self, component_name: str) -> Optional[ComponentProfile]:
        """Get performance profile for a component."""
        async with self._lock:
            return self.component_profiles.get(component_name)
    
    async def get_all_component_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance profiles for all components."""
        async with self._lock:
            return {
                name: profile.to_dict() 
                for name, profile in self.component_profiles.items()
            }
    
    async def get_recent_experiences(
        self, 
        count: int = 20,
        successful_only: bool = False
    ) -> List[Experience]:
        """Get recent experiences."""
        async with self._lock:
            experiences = list(self.experiences)[-count:]
            if successful_only:
                experiences = [e for e in experiences if e.success]
            return experiences
    
    async def suggest_adaptations(self) -> List[AdaptationStrategy]:
        """
        Analyze current performance and suggest adaptations.
        
        Returns:
            List of suggested adaptation strategies
        """
        suggestions = []
        
        async with self._lock:
            # Analyze component performance
            for name, profile in self.component_profiles.items():
                if profile.total_invocations < 10:  # Need enough data
                    continue
                
                # Check for consistently underperforming components
                if profile.average_success_rate < 0.7:
                    suggestions.append(AdaptationStrategy(
                        target_component=name,
                        action="increase_timeout",
                        reason=f"Low success rate: {profile.average_success_rate:.2%}",
                        confidence=1.0 - profile.average_success_rate,
                        expected_improvement=0.1
                    ))
                
                # Check for slow components
                avg_time = self.global_stats["average_processing_time_ms"]
                if avg_time > 0 and profile.average_processing_time_ms > avg_time * 2:
                    suggestions.append(AdaptationStrategy(
                        target_component=name,
                        action="reduce_complexity",
                        reason=f"Slow processing: {profile.average_processing_time_ms:.0f}ms vs avg {avg_time:.0f}ms",
                        confidence=0.7,
                        expected_improvement=0.15
                    ))
                
                # Check for error patterns
                if profile.error_types:
                    most_common_error = max(profile.error_types.items(), key=lambda x: x[1])
                    if most_common_error[1] > profile.total_invocations * 0.3:
                        suggestions.append(AdaptationStrategy(
                            target_component=name,
                            action="add_retry_logic",
                            reason=f"Frequent {most_common_error[0]} errors ({most_common_error[1]} times)",
                            confidence=0.8,
                            expected_improvement=0.2
                        ))
        
        return suggestions
    
    async def apply_adaptation(self, strategy: AdaptationStrategy) -> bool:
        """
        Apply an adaptation strategy.
        
        Args:
            strategy: The adaptation strategy to apply
            
        Returns:
            True if adaptation was applied successfully
        """
        async with self._lock:
            self.active_adaptations[strategy.target_component] = {
                "strategy": strategy,
                "applied_at": datetime.now(timezone.utc).isoformat(),
                "status": "active"
            }
            self.adaptation_strategies.append(strategy)
        
        # Notify callbacks
        for callback in self.adaptation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(strategy)
                else:
                    callback(strategy)
            except Exception as e:
                logger.warning({
                    "msg": "Adaptation callback failed",
                    "error": str(e)
                })
        
        logger.info({
            "msg": "Adaptation applied",
            "component": strategy.target_component,
            "action": strategy.action,
            "reason": strategy.reason
        })
        
        return True
    
    async def get_learning_summary(self) -> Dict[str, Any]:
        """Get a summary of the learning state."""
        async with self._lock:
            return {
                "global_stats": self.global_stats.copy(),
                "component_count": len(self.component_profiles),
                "experience_count": len(self.experiences),
                "active_adaptations": len(self.active_adaptations),
                "total_adaptations": len(self.adaptation_strategies),
                "component_performance": {
                    name: profile.to_dict()
                    for name, profile in self.component_profiles.items()
                }
            }
    
    async def _analyze_and_adapt(self):
        """Internal method to analyze and potentially auto-adapt."""
        if not self.enable_auto_adaptation:
            return
        
        suggestions = await self.suggest_adaptations()
        
        # Auto-apply high-confidence adaptations
        for suggestion in suggestions:
            if suggestion.confidence > 0.8:
                await self.apply_adaptation(suggestion)
    
    async def _periodic_cleanup(self):
        """Periodically clean up old experiences."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_experiences()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error({
                    "msg": "Error in adaptation cleanup",
                    "error": str(e)
                })
    
    async def _cleanup_old_experiences(self):
        """Remove experiences older than memory_retention_hours."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.memory_retention_hours)
        
        async with self._lock:
            old_count = len(self.experiences)
            self.experiences = deque(
                [e for e in self.experiences if e.timestamp > cutoff_time],
                maxlen=self.experience_buffer_size
            )
            removed = old_count - len(self.experiences)
        
        if removed > 0:
            logger.info({
                "msg": "Cleaned up old experiences",
                "removed": removed,
                "remaining": len(self.experiences)
            })
    
    async def close(self):
        """Close the adaptation engine and cleanup resources."""
        await self.stop()
        logger.info({"msg": "AdaptationEngine closed"})
