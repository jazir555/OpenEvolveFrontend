"""
ReflectionEngine - System Reflection and Improvement for OpenEvolve Knowledge Engine

Periodically reflects on system performance, identifies patterns, and generates
actionable insights for continuous improvement.
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ReflectionInsight:
    """A single insight generated during reflection."""
    category: str  # e.g., "performance", "reliability", "optimization"
    severity: str  # e.g., "low", "medium", "high", "critical"
    description: str
    recommendation: str
    confidence: float
    affected_components: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    id: str = field(default_factory=lambda: f"insight_{datetime.now(timezone.utc).timestamp()}")


@dataclass
class ReflectionSession:
    """A complete reflection session with all insights."""
    start_time: datetime
    end_time: Optional[datetime] = None
    insights: List[ReflectionInsight] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: f"session_{datetime.now(timezone.utc).timestamp()}")
    
    def complete(self):
        """Mark the reflection session as complete."""
        self.end_time = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (
                (self.end_time - self.start_time).total_seconds() 
                if self.end_time else None
            ),
            "insights_count": len(self.insights),
            "insights": [
                {
                    "id": i.id,
                    "category": i.category,
                    "severity": i.severity,
                    "description": i.description,
                    "recommendation": i.recommendation,
                    "confidence": i.confidence,
                    "affected_components": i.affected_components
                }
                for i in self.insights
            ],
            "summary": self.summary
        }


@dataclass
class Pattern:
    """Identified pattern in system behavior."""
    pattern_type: str  # e.g., "success_pattern", "failure_pattern", "performance_pattern"
    description: str
    frequency: int
    first_seen: datetime
    last_seen: datetime
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat()
        }


class ReflectionEngine:
    """
    Engine for system reflection and continuous improvement.
    
    Analyzes historical performance, identifies patterns, generates insights,
    and recommends improvements based on accumulated experience.
    """
    
    def __init__(
        self,
        reflection_frequency: int = 10,  # Reflect every N operations
        min_operations_before_reflection: int = 5,
        enable_auto_reflection: bool = True,
        max_insights_history: int = 100,
        pattern_detection_threshold: int = 3
    ):
        self.reflection_frequency = reflection_frequency
        self.min_operations_before_reflection = min_operations_before_reflection
        self.enable_auto_reflection = enable_auto_reflection
        self.max_insights_history = max_insights_history
        self.pattern_detection_threshold = pattern_detection_threshold
        
        # Reflection state
        self.reflection_history: List[ReflectionSession] = []
        self.insights_history: List[ReflectionInsight] = []
        self.patterns: Dict[str, Pattern] = {}
        
        # Operation tracking
        self.operation_count = 0
        self.last_reflection_operation = 0
        
        # Callbacks for insights
        self.insight_callbacks: List[Callable] = []
        self.reflection_callbacks: List[Callable] = []
        
        # Statistics
        self.reflection_stats = {
            "total_reflections": 0,
            "total_insights_generated": 0,
            "insights_by_category": defaultdict(int),
            "insights_by_severity": defaultdict(int)
        }
        
        self._lock = asyncio.Lock()
        self._reflection_task: Optional[asyncio.Task] = None
        
        logger.info({
            "msg": "ReflectionEngine initialized",
            "reflection_frequency": reflection_frequency,
            "enable_auto_reflection": enable_auto_reflection
        })
    
    async def start(self):
        """Start the reflection engine."""
        if self.enable_auto_reflection:
            self._reflection_task = asyncio.create_task(self._auto_reflection_loop())
        logger.info({"msg": "ReflectionEngine started"})
    
    async def stop(self):
        """Stop the reflection engine."""
        if self._reflection_task:
            self._reflection_task.cancel()
            try:
                await self._reflection_task
            except asyncio.CancelledError:
                pass
        logger.info({"msg": "ReflectionEngine stopped"})
    
    async def record_operation(
        self,
        operation_type: str,
        success: bool,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record an operation for reflection analysis.
        
        Args:
            operation_type: Type of operation performed
            success: Whether the operation succeeded
            metadata: Additional operation metadata
        """
        async with self._lock:
            self.operation_count += 1
            
            # Check if auto-reflection should trigger
            if self.enable_auto_reflection:
                ops_since_last = self.operation_count - self.last_reflection_operation
                if (ops_since_last >= self.reflection_frequency and 
                    self.operation_count >= self.min_operations_before_reflection):
                    # Trigger reflection in background
                    asyncio.create_task(self.reflect())
        
        logger.debug({
            "msg": "Operation recorded for reflection",
            "operation_type": operation_type,
            "success": success,
            "total_operations": self.operation_count
        })
    
    async def reflect(
        self,
        operation_history: Optional[List[Dict[str, Any]]] = None,
        component_performance: Optional[Dict[str, Any]] = None,
        focus_areas: Optional[List[str]] = None
    ) -> ReflectionSession:
        """
        Perform a reflection session on system performance.
        
        Args:
            operation_history: Recent operation history
            component_performance: Component performance data
            focus_areas: Specific areas to focus reflection on
            
        Returns:
            The completed ReflectionSession with insights
        """
        session = ReflectionSession(start_time=datetime.now(timezone.utc))
        focus_areas = focus_areas or ["performance", "reliability", "optimization"]
        
        logger.info({
            "msg": "Starting reflection session",
            "session_id": session.id,
            "focus_areas": focus_areas
        })
        
        try:
            # Analyze performance trends
            if "performance" in focus_areas and component_performance:
                performance_insights = await self._analyze_performance(component_performance)
                session.insights.extend(performance_insights)
            
            # Analyze reliability patterns
            if "reliability" in focus_areas and operation_history:
                reliability_insights = await self._analyze_reliability(operation_history)
                session.insights.extend(reliability_insights)
            
            # Analyze for optimization opportunities
            if "optimization" in focus_areas:
                optimization_insights = await self._analyze_optimization_opportunities(
                    operation_history, component_performance
                )
                session.insights.extend(optimization_insights)
            
            # Detect patterns
            if operation_history:
                await self._update_patterns(operation_history)
            
            # Generate summary
            session.summary = self._generate_summary(session.insights)
            session.complete()
            
            # Store session and insights
            async with self._lock:
                self.reflection_history.append(session)
                self.insights_history.extend(session.insights)
                self.last_reflection_operation = self.operation_count
                
                # Limit history size
                if len(self.reflection_history) > 50:
                    self.reflection_history = self.reflection_history[-50:]
                if len(self.insights_history) > self.max_insights_history:
                    self.insights_history = self.insights_history[-self.max_insights_history:]
                
                # Update stats
                self.reflection_stats["total_reflections"] += 1
                self.reflection_stats["total_insights_generated"] += len(session.insights)
                for insight in session.insights:
                    self.reflection_stats["insights_by_category"][insight.category] += 1
                    self.reflection_stats["insights_by_severity"][insight.severity] += 1
            
            # Notify callbacks
            await self._notify_reflection_complete(session)
            
            logger.info({
                "msg": "Reflection session completed",
                "session_id": session.id,
                "insights_count": len(session.insights),
                "duration_seconds": (session.end_time - session.start_time).total_seconds()
            })
            
        except Exception as e:
            logger.error({
                "msg": "Reflection session failed",
                "session_id": session.id,
                "error": str(e)
            })
            session.complete()
        
        return session
    
    async def get_recent_insights(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        count: int = 10
    ) -> List[ReflectionInsight]:
        """Get recent insights with optional filtering."""
        async with self._lock:
            insights = self.insights_history[-count:]
            
            if category:
                insights = [i for i in insights if i.category == category]
            if severity:
                insights = [i for i in insights if i.severity == severity]
            
            return insights[-count:]  # Re-apply limit after filtering
    
    async def get_patterns(self, pattern_type: Optional[str] = None) -> List[Pattern]:
        """Get identified patterns with optional filtering."""
        async with self._lock:
            patterns = list(self.patterns.values())
            if pattern_type:
                patterns = [p for p in patterns if p.pattern_type == pattern_type]
            return patterns
    
    async def get_reflection_summary(self) -> Dict[str, Any]:
        """Get a summary of reflection history and insights."""
        async with self._lock:
            recent_insights = self.insights_history[-20:] if self.insights_history else []
            
            return {
                "total_reflections": self.reflection_stats["total_reflections"],
                "total_insights": self.reflection_stats["total_insights_generated"],
                "operation_count": self.operation_count,
                "reflection_frequency": self.reflection_frequency,
                "insights_by_category": dict(self.reflection_stats["insights_by_category"]),
                "insights_by_severity": dict(self.reflection_stats["insights_by_severity"]),
                "recent_insights": [
                    {
                        "id": i.id,
                        "category": i.category,
                        "severity": i.severity,
                        "description": i.description[:100] + "..." if len(i.description) > 100 else i.description,
                        "timestamp": i.timestamp.isoformat()
                    }
                    for i in recent_insights
                ],
                "pattern_count": len(self.patterns)
            }
    
    async def add_insight_callback(self, callback: Callable):
        """Add a callback to be notified when new insights are generated."""
        self.insight_callbacks.append(callback)
    
    async def add_reflection_callback(self, callback: Callable):
        """Add a callback to be notified when reflection completes."""
        self.reflection_callbacks.append(callback)
    
    async def _analyze_performance(
        self, 
        component_performance: Dict[str, Any]
    ) -> List[ReflectionInsight]:
        """Analyze performance data and generate insights."""
        insights = []
        
        # Check for slow components
        for comp_name, perf in component_performance.items():
            if isinstance(perf, dict) and "avg_processing_time" in perf:
                avg_time = perf.get("avg_processing_time", 0)
                if avg_time > 1000:  # More than 1 second
                    insights.append(ReflectionInsight(
                        category="performance",
                        severity="medium",
                        description=f"Component '{comp_name}' has high average processing time: {avg_time:.0f}ms",
                        recommendation=f"Consider optimizing '{comp_name}' or implementing caching",
                        confidence=0.8,
                        affected_components=[comp_name],
                        metrics={"avg_processing_time_ms": avg_time}
                    ))
        
        return insights
    
    async def _analyze_reliability(
        self, 
        operation_history: List[Dict[str, Any]]
    ) -> List[ReflectionInsight]:
        """Analyze reliability patterns and generate insights."""
        insights = []
        
        if not operation_history:
            return insights
        
        # Calculate success rate
        success_count = sum(1 for op in operation_history if op.get("result_success", False))
        total = len(operation_history)
        success_rate = success_count / total if total > 0 else 1.0
        
        # Check overall success rate
        if success_rate < 0.8:
            insights.append(ReflectionInsight(
                category="reliability",
                severity="high" if success_rate < 0.5 else "medium",
                description=f"Low success rate detected: {success_rate:.1%} over last {total} operations",
                recommendation="Investigate common failure patterns and implement error handling improvements",
                confidence=0.9,
                metrics={"success_rate": success_rate, "operations_analyzed": total}
            ))
        
        # Check for error clustering
        error_counts = defaultdict(int)
        for op in operation_history:
            if not op.get("result_success", False):
                error_type = op.get("error_type", "unknown")
                error_counts[error_type] += 1
        
        for error_type, count in error_counts.items():
            if count >= self.pattern_detection_threshold:
                insights.append(ReflectionInsight(
                    category="reliability",
                    severity="medium",
                    description=f"Recurring '{error_type}' errors detected ({count} occurrences)",
                    recommendation=f"Review error handling for '{error_type}' and consider targeted fixes",
                    confidence=min(0.95, 0.6 + count * 0.1),
                    metrics={"error_count": count, "error_type": error_type}
                ))
        
        return insights
    
    async def _analyze_optimization_opportunities(
        self,
        operation_history: Optional[List[Dict[str, Any]]],
        component_performance: Optional[Dict[str, Any]]
    ) -> List[ReflectionInsight]:
        """Analyze for optimization opportunities."""
        insights = []
        
        # Check for component usage patterns
        if component_performance:
            unused_components = [
                name for name, perf in component_performance.items()
                if isinstance(perf, dict) and perf.get("total_ops", 0) == 0
            ]
            if len(unused_components) > len(component_performance) * 0.5:
                insights.append(ReflectionInsight(
                    category="optimization",
                    severity="low",
                    description=f"Many components are unused: {', '.join(unused_components[:3])}...",
                    recommendation="Consider removing unused components or reviewing integration",
                    confidence=0.7,
                    affected_components=unused_components
                ))
        
        return insights
    
    async def _update_patterns(self, operation_history: List[Dict[str, Any]]):
        """Update pattern detection based on operation history."""
        # Simple pattern: repeated query patterns
        query_patterns = defaultdict(int)
        for op in operation_history:
            query = op.get("query", "")
            # Normalize query for pattern matching
            normalized = query.lower().strip()[:50]  # First 50 chars
            if normalized:
                query_patterns[normalized] += 1
        
        for query, count in query_patterns.items():
            if count >= self.pattern_detection_threshold:
                pattern_key = f"query_{hash(query) % 10000}"
                if pattern_key not in self.patterns:
                    self.patterns[pattern_key] = Pattern(
                        pattern_type="success_pattern" if count > 0 else "failure_pattern",
                        description=f"Frequent query pattern: '{query}...'",
                        frequency=count,
                        first_seen=datetime.now(timezone.utc),
                        last_seen=datetime.now(timezone.utc)
                    )
                else:
                    self.patterns[pattern_key].frequency += count
                    self.patterns[pattern_key].last_seen = datetime.now(timezone.utc)
    
    def _generate_summary(self, insights: List[ReflectionInsight]) -> Dict[str, Any]:
        """Generate a summary of insights."""
        if not insights:
            return {"status": "no_insights", "message": "No insights generated in this session"}
        
        by_category = defaultdict(list)
        by_severity = defaultdict(list)
        
        for insight in insights:
            by_category[insight.category].append(insight)
            by_severity[insight.severity].append(insight)
        
        return {
            "status": "completed",
            "total_insights": len(insights),
            "insights_by_category": {k: len(v) for k, v in by_category.items()},
            "insights_by_severity": {k: len(v) for k, v in by_severity.items()},
            "critical_issues": len(by_severity.get("critical", [])),
            "high_priority_recommendations": [
                i.recommendation for i in insights 
                if i.severity in ["high", "critical"]
            ]
        }
    
    async def _notify_reflection_complete(self, session: ReflectionSession):
        """Notify all reflection callbacks."""
        for callback in self.reflection_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(session)
                else:
                    callback(session)
            except Exception as e:
                logger.warning({
                    "msg": "Reflection callback failed",
                    "error": str(e)
                })
        
        # Also notify for individual insights
        for insight in session.insights:
            for callback in self.insight_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(insight)
                    else:
                        callback(insight)
                except Exception as e:
                    logger.warning({
                        "msg": "Insight callback failed",
                        "error": str(e)
                    })
    
    async def _auto_reflection_loop(self):
        """Background loop for automatic reflection."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                async with self._lock:
                    ops_since_last = self.operation_count - self.last_reflection_operation
                    should_reflect = (
                        ops_since_last >= self.reflection_frequency and
                        self.operation_count >= self.min_operations_before_reflection
                    )
                
                if should_reflect:
                    await self.reflect()
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error({
                    "msg": "Error in auto-reflection loop",
                    "error": str(e)
                })
    
    async def close(self):
        """Close the reflection engine and cleanup resources."""
        await self.stop()
        logger.info({"msg": "ReflectionEngine closed"})
