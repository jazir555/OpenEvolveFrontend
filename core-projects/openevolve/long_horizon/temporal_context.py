"""
Temporal Context Manager for Long-Horizon Agents

Implements time-aware reasoning and temporal knowledge graphs.
Follows CLAUDE.md principles:
- Law of Runtime Truth: Verify all temporal data
- Law of Idempotency: All operations replay-safe
- Law of UTC: All timestamps in UTC

Author: Claude (Sonnet 4.5)
Date: January 30, 2026
"""

import structlog
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import numpy as np

from .schemas.temporal_schemas import (
    TemporalEvent,
    CausalLink,
    TemporalPattern,
    TimeWindow,
    TrendAnalysis
)


logger = structlog.get_logger()


class TemporalContextError(Exception):
    """Base exception for temporal context errors"""
    pass


class TemporalContextManager:
    """
    Manages temporal context for long-horizon agents.

    Features:
    - Time-aware reasoning (deadlines, milestones, recurring events)
    - Temporal knowledge graphs with causal links
    - Historical context retrieval
    - Trend and anomaly detection
    - Recurring pattern recognition

    All times in UTC. All operations idempotent.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize Temporal Context Manager.

        Args:
            config: Optional configuration
        """
        self.config = config or {}

        # Event storage
        self._events: Dict[str, TemporalEvent] = {}
        self._events_by_type: Dict[str, List[str]] = defaultdict(list)
        self._events_by_workflow: Dict[str, List[str]] = defaultdict(list)

        # Causal links
        self._causal_links: Dict[str, CausalLink] = {}
        self._causal_graph: Dict[str, List[str]] = defaultdict(list)  # event_id -> effects

        # Patterns
        self._patterns: Dict[str, TemporalPattern] = {}

        logger.info(
            "temporal_context_manager_initialized",
            config_keys=list(self.config.keys())
        )

    async def add_event(self, event: TemporalEvent) -> None:
        """
        Add a temporal event (idempotent).

        Args:
            event: Event to add
        """
        # Validate UTC timestamp
        if event.timestamp.tzinfo is None:
            raise ValueError("Event timestamp must be timezone-aware (UTC)")

        # Store event
        self._events[event.event_id] = event
        self._events_by_type[event.event_type].append(event.event_id)
        if event.workflow_id:
            self._events_by_workflow[event.workflow_id].append(event.event_id)

        logger.info(
            "event_added",
            event_id=event.event_id,
            event_type=event.event_type,
            timestamp=event.timestamp.isoformat()
        )

    async def get_events(
        self,
        time_window: TimeWindow,
        event_types: Optional[List[str]] = None,
        importance_threshold: float = 0.0,
        workflow_id: Optional[str] = None
    ) -> List[TemporalEvent]:
        """
        Get events within a time window.

        Args:
            time_window: Time range to query
            event_types: Optional filter by event types
            importance_threshold: Minimum importance score
            workflow_id: Optional filter by workflow

        Returns:
            List of events matching criteria
        """
        events = []

        # Build candidate set
        candidate_ids = set(self._events.keys())

        if event_types:
            type_ids = set()
            for et in event_types:
                type_ids.update(self._events_by_type.get(et, []))
            candidate_ids &= type_ids

        if workflow_id:
            workflow_ids = set(self._events_by_workflow.get(workflow_id, []))
            candidate_ids &= workflow_ids

        # Filter by time window and importance
        for event_id in candidate_ids:
            event = self._events[event_id]

            if not (time_window.start_time <= event.timestamp <= time_window.end_time):
                continue

            if event.importance < importance_threshold:
                continue

            events.append(event)

        # Sort by timestamp
        events.sort(key=lambda e: e.timestamp)

        return events

    async def add_causal_link(self, link: CausalLink) -> None:
        """
        Add a causal relationship between events (idempotent).

        Args:
            link: Causal link to add
        """
        self._causal_links[link.link_id] = link
        self._causal_graph[link.cause_event_id].append(link.effect_event_id)

        logger.info(
            "causal_link_added",
            link_id=link.link_id,
            cause=link.cause_event_id,
            effect=link.effect_event_id,
            strength=link.strength
        )

    async def get_causal_chain(
        self,
        event_id: str,
        max_depth: int = 10,
        direction: str = "forward"
    ) -> List[TemporalEvent]:
        """
        Get causal chain starting from or ending at an event.

        Args:
            event_id: Starting/ending event ID
            max_depth: Maximum chain length
            direction: "forward" (effects) or "backward" (causes)

        Returns:
            List of events in causal order
        """
        chain = []
        visited = set()
        current = event_id

        for _ in range(max_depth):
            if current in visited or current not in self._events:
                break

            visited.add(current)
            chain.append(self._events[current])

            # Get next event in chain
            if direction == "forward":
                links = self._causal_graph.get(current, [])
                if not links:
                    break
                current = links[0]  # Take first link
            else:  # backward
                # Find links where this is the effect
                causes = [
                    link.cause_event_id
                    for link in self._causal_links.values()
                    if link.effect_event_id == current
                ]
                if not causes:
                    break
                current = causes[0]

        return chain

    async def detect_patterns(
        self,
        event_type: str,
        time_window: TimeWindow
    ) -> List[TemporalPattern]:
        """
        Detect recurring patterns in events.

        Args:
            event_type: Type of events to analyze
            time_window: Time window to analyze

        Returns:
            List of detected patterns
        """
        # Get events of this type
        events = await self.get_events(
            time_window=time_window,
            event_types=[event_type]
        )

        if len(events) < 3:  # Need minimum events for pattern
            return []

        patterns = []

        # Detect periodic patterns
        if len(events) >= 3:
            # Calculate inter-arrival times
            timestamps = [e.timestamp for e in events]
            intervals = [
                (timestamps[i+1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)
            ]

            # Check for periodicity
            if intervals:
                avg_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                cv = std_interval / avg_interval if avg_interval > 0 else float('inf')

                # Low coefficient of variation indicates periodicity
                if cv < 0.2:  # 20% or less variation
                    pattern = TemporalPattern(
                        pattern_id=self._generate_id('pattern'),
                        pattern_type='periodic',
                        description=f'Periodic occurrence of {event_type}',
                        pattern_expression=f'every {avg_interval:.0f}s',
                        period_seconds=avg_interval,
                        event_types=[event_type],
                        occurrence_count=len(events),
                        confidence=1.0 - cv,
                        discovered_at=datetime.now(timezone.utc),
                        last_observed=timestamps[-1]
                    )

                    # Predict next occurrence
                    pattern.next_occurrence = timestamps[-1] + timedelta(seconds=avg_interval)

                    patterns.append(pattern)

        # Store patterns
        for pattern in patterns:
            self._patterns[pattern.pattern_id] = pattern

        logger.info(
            "patterns_detected",
            event_type=event_type,
            patterns_found=len(patterns)
        )

        return patterns

    async def analyze_trend(
        self,
        metric_name: str,
        time_window: TimeWindow,
        workflow_id: Optional[str] = None
    ) -> TrendAnalysis:
        """
        Analyze trend in a metric over time.

        Args:
            metric_name: Metric to analyze
            time_window: Time window to analyze
            workflow_id: Optional workflow filter

        Returns:
            Trend analysis result
        """
        # Get events with this metric
        events = await self.get_events(
            time_window=time_window,
            workflow_id=workflow_id
        )

        # Extract metric values from events
        metric_values = []
        timestamps = []

        for event in events:
            if metric_name in event.event_data:
                metric_values.append(event.event_data[metric_name])
                timestamps.append(event.timestamp)

        if len(metric_values) < 2:
            return TrendAnalysis(
                analysis_id=self._generate_id('trend'),
                trend_type='insufficient_data',
                metric_name=metric_name,
                analysis_window=time_window,
                is_anomaly=False,
                confidence=0.0,
                analyzed_by='temporal_context_manager'
            )

        # Perform linear regression
        x = np.arange(len(metric_values))
        y = np.array(metric_values)

        # Simple linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Calculate correlation
        correlation = np.corrcoef(x, y)[0, 1]

        # Determine trend type
        if abs(slope) < 0.01:
            trend_type = 'stable'
        elif slope > 0:
            trend_type = 'increasing'
        else:
            trend_type = 'decreasing'

        # Check for anomalies (values far from trend line)
        predicted = slope * x + intercept
        residuals = y - predicted
        std_residual = np.std(residuals)

        anomalies = np.abs(residuals) > 2 * std_residual
        is_anomaly = np.any(anomalies)

        anomaly_score = float(np.max(np.abs(residuals / std_residual))) if is_anomaly else None

        # Determine impact
        if is_anomaly and anomaly_score > 3:
            impact_level = 'high'
        elif is_anomaly:
            impact_level = 'medium'
        else:
            impact_level = 'none'

        analysis = TrendAnalysis(
            analysis_id=self._generate_id('trend'),
            trend_type=trend_type,
            metric_name=metric_name,
            slope=float(slope),
            correlation=float(correlation),
            analysis_window=time_window,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            threshold=2.0,
            confidence=min(abs(correlation), 1.0),
            impact_level=impact_level,
            analyzed_by='temporal_context_manager'
        )

        logger.info(
            "trend_analyzed",
            metric_name=metric_name,
            trend_type=trend_type,
            is_anomaly=is_anomaly
        )

        return analysis

    async def get_context_at_time(
        self,
        timestamp: datetime,
        window_seconds: int = 3600
    ) -> Dict[str, Any]:
        """
        Get temporal context around a specific time.

        Args:
            timestamp: Point in time (UTC)
            window_seconds: Seconds before/after to include

        Returns:
            Context dictionary with events, patterns, trends
        """
        time_window = TimeWindow(
            window_id=self._generate_id('window'),
            start_time=timestamp - timedelta(seconds=window_seconds),
            end_time=timestamp + timedelta(seconds=window_seconds)
        )

        # Get events
        events = await self.get_events(time_window=time_window)

        # Get relevant patterns
        recent_patterns = [
            p for p in self._patterns.values()
            if p.last_observed and (timestamp - p.last_observed).total_seconds() < window_seconds
        ]

        context = {
            'timestamp': timestamp,
            'time_window': time_window,
            'events': [e.dict() for e in events],
            'event_count': len(events),
            'patterns': [p.dict() for p in recent_patterns],
            'recent_patterns': len(recent_patterns)
        }

        return context

    async def predict_next_occurrence(
        self,
        event_type: str
    ) -> Optional[datetime]:
        """
        Predict next occurrence of an event type based on patterns.

        Args:
            event_type: Event type to predict

        Returns:
            Predicted next occurrence time (UTC) or None
        """
        # Find patterns for this event type
        patterns = [
            p for p in self._patterns.values()
            if event_type in p.event_types and p.next_occurrence
        ]

        if not patterns:
            return None

        # Return most confident prediction
        best_pattern = max(patterns, key=lambda p: p.confidence)
        return best_pattern.next_occurrence

    async def get_deadline_status(
        self,
        deadline: datetime,
        current_time: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Get status relative to a deadline.

        Args:
            deadline: Deadline time (UTC)
            current_time: Current time (UTC, defaults to now)

        Returns:
            Status dictionary with time remaining, urgency, etc.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        time_remaining = (deadline - current_time).total_seconds()

        if time_remaining < 0:
            status = 'overdue'
            urgency = 'critical'
        elif time_remaining < 3600:  # < 1 hour
            status = 'urgent'
            urgency = 'high'
        elif time_remaining < 86400:  # < 1 day
            status = 'approaching'
            urgency = 'medium'
        else:
            status = 'on_track'
            urgency = 'low'

        return {
            'deadline': deadline,
            'current_time': current_time,
            'time_remaining_seconds': time_remaining,
            'time_remaining_hours': time_remaining / 3600,
            'status': status,
            'urgency': urgency
        }

    def _generate_id(self, prefix: str) -> str:
        """Generate unique ID with prefix"""
        import uuid
        return f"{prefix}_{uuid.uuid4().hex[:16]}"
