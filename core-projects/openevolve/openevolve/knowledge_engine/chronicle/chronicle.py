"""
Chronicle - Temporal Episodic Memory

Stores agent experiences as a timeline of episodes.
Enables the system to remember:
- "I already tried Strategy X 20 minutes ago and it failed"
- "First we tried A, then B failed, so we did C"
- Narrative of the problem-solving process

This complements Knowledge Graphs (which store facts) with narrative memory.

Copyright 2026 OpenEvolve

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
import json
import hashlib
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class EpisodeType(Enum):
    """Types of episodes"""
    ACTION = "action"           # An action was taken
    DECISION = "decision"       # A decision was made
    OBSERVATION = "observation" # Something was observed
    FAILURE = "failure"         # Something failed
    SUCCESS = "success"         # Something succeeded
    LEARNING = "learning"       # Something was learned
    HEALING = "healing"         # Self-healing occurred
    STRATEGY = "strategy"       # Strategy was applied


@dataclass
class Episode:
    """
    A single episode in the agent's experience timeline.
    
    Example:
        Episode(
            episode_id="ep_001",
            episode_type=EpisodeType.ACTION,
            timestamp="2026-01-30T10:00:00Z",
            agent="BlueTeam",
            action="Attempted to fix Z3 timeout",
            context={"error": "Z3 solver timeout"},
            outcome="Failed - timeout persisted",
            lesson_learned="Need to increase timeout or simplify constraints",
            related_episodes=["ep_000"],
            tags=["z3", "timeout", "constraint-solving"]
        )
    """
    episode_id: str
    episode_type: EpisodeType
    timestamp: str
    agent: str
    action: str
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    lesson_learned: Optional[str] = None
    related_episodes: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'episode_id': self.episode_id,
            'episode_type': self.episode_type.value,
            'timestamp': self.timestamp,
            'agent': self.agent,
            'action': self.action,
            'context': self.context,
            'outcome': self.outcome,
            'lesson_learned': self.lesson_learned,
            'related_episodes': self.related_episodes,
            'tags': self.tags,
            'session_id': self.session_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        return cls(
            episode_id=data['episode_id'],
            episode_type=EpisodeType(data['episode_type']),
            timestamp=data['timestamp'],
            agent=data['agent'],
            action=data['action'],
            context=data.get('context', {}),
            outcome=data.get('outcome'),
            lesson_learned=data.get('lesson_learned'),
            related_episodes=data.get('related_episodes', []),
            tags=data.get('tags', []),
            session_id=data.get('session_id')
        )
    
    @property
    def timestamp_dt(self) -> datetime:
        """Get timestamp as datetime object"""
        return datetime.fromisoformat(self.timestamp.replace('Z', '+00:00'))


@dataclass
class ChronicleQuery:
    """Query for searching episodes"""
    agent: Optional[str] = None
    episode_types: Optional[List[EpisodeType]] = None
    tags: Optional[List[str]] = None
    time_range: Optional[Tuple[datetime, datetime]] = None
    context_filter: Optional[Dict[str, Any]] = None
    limit: int = 10
    
    def matches(self, episode: Episode) -> bool:
        """Check if an episode matches this query"""
        if self.agent and episode.agent != self.agent:
            return False
        
        if self.episode_types and episode.episode_type not in self.episode_types:
            return False
        
        if self.tags and not any(tag in episode.tags for tag in self.tags):
            return False
        
        if self.time_range:
            start, end = self.time_range
            ep_time = episode.timestamp_dt
            if not (start <= ep_time <= end):
                return False
        
        if self.context_filter:
            for key, value in self.context_filter.items():
                if episode.context.get(key) != value:
                    return False
        
        return True


class Chronicle:
    """
    Temporal Episodic Memory - stores experiences as a timeline.
    
    Unlike Knowledge Graphs (which store facts), the Chronicle stores:
    - The narrative of what happened
    - When things happened
    - What was tried and failed
    - Lessons learned from experience
    
    Key Features:
    - Event sourcing pattern
    - Time-series storage
    - Pattern detection across episodes
    - "Have we tried this before?" queries
    - Strategy effectiveness tracking
    
    Example Usage:
        chronicle = Chronicle()
        
        # Record an episode
        chronicle.record_episode(
            agent="BlueTeam",
            action="Attempted fix for Z3 timeout",
            episode_type=EpisodeType.FAILURE,
            outcome="Still timing out",
            lesson_learned="Need to increase solver timeout"
        )
        
        # Query: Have we tried this before?
        similar = chronicle.find_similar_episodes(
            action_pattern="Z3 timeout",
            time_window=timedelta(hours=1)
        )
        # Returns: "Yes, 20 minutes ago with same result"
    """
    
    def __init__(
        self,
        storage_path: Optional[str] = None,
        max_episodes: int = 10000,
        auto_persist: bool = True
    ):
        """
        Initialize Chronicle.
        
        Args:
            storage_path: Path to persist episodes
            max_episodes: Maximum episodes to keep in memory
            auto_persist: Auto-save to disk
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.max_episodes = max_episodes
        self.auto_persist = auto_persist
        
        # In-memory storage
        self.episodes: Dict[str, Episode] = {}
        self.episode_sequence: List[str] = []  # Ordered by time
        
        # Indexes for fast querying
        self.agent_index: Dict[str, Set[str]] = {}
        self.tag_index: Dict[str, Set[str]] = {}
        self.type_index: Dict[str, Set[str]] = {}
        self.session_index: Dict[str, Set[str]] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Load existing data
        if self.storage_path:
            self._load()
        
        logger.info({
            'msg': 'Chronicle initialized',
            'episodes_loaded': len(self.episodes),
            'storage': str(self.storage_path) if self.storage_path else None
        })
    
    def record_episode(
        self,
        agent: str,
        action: str,
        episode_type: EpisodeType,
        context: Optional[Dict[str, Any]] = None,
        outcome: Optional[str] = None,
        lesson_learned: Optional[str] = None,
        related_episodes: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        session_id: Optional[str] = None
    ) -> Episode:
        """
        Record a new episode in the chronicle.
        
        Args:
            agent: Agent that performed the action
            action: Description of what was done
            episode_type: Type of episode
            context: Additional context
            outcome: Result of the action
            lesson_learned: What was learned
            related_episodes: IDs of related episodes
            tags: Tags for categorization
            session_id: Session this episode belongs to
            
        Returns:
            The created Episode
        """
        with self._lock:
            # Generate ID
            timestamp = datetime.now(timezone.utc)
            episode_id = f"ep_{timestamp.strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(action.encode()).hexdigest()[:6]}"
            
            # Create episode
            episode = Episode(
                episode_id=episode_id,
                episode_type=episode_type,
                timestamp=timestamp.isoformat(),
                agent=agent,
                action=action,
                context=context or {},
                outcome=outcome,
                lesson_learned=lesson_learned,
                related_episodes=related_episodes or [],
                tags=tags or [],
                session_id=session_id
            )
            
            # Store episode
            self.episodes[episode_id] = episode
            self.episode_sequence.append(episode_id)
            
            # Update indexes
            self._update_indexes(episode)
            
            # Enforce max episodes limit
            if len(self.episodes) > self.max_episodes:
                self._evict_oldest()
            
            # Persist
            if self.auto_persist and self.storage_path:
                self._persist_episode(episode)
            
            logger.debug({
                'msg': 'Episode recorded',
                'episode_id': episode_id,
                'agent': agent,
                'type': episode_type.value,
                'action': action[:50]
            })
            
            return episode_id
    
    def _update_indexes(self, episode: Episode):
        """Update all indexes for an episode"""
        # Agent index
        if episode.agent not in self.agent_index:
            self.agent_index[episode.agent] = set()
        self.agent_index[episode.agent].add(episode.episode_id)
        
        # Tag index
        for tag in episode.tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(episode.episode_id)
        
        # Type index
        type_key = episode.episode_type.value
        if type_key not in self.type_index:
            self.type_index[type_key] = set()
        self.type_index[type_key].add(episode.episode_id)
        
        # Session index
        if episode.session_id:
            if episode.session_id not in self.session_index:
                self.session_index[episode.session_id] = set()
            self.session_index[episode.session_id].add(episode.episode_id)
    
    def _evict_oldest(self):
        """Remove oldest episodes when limit reached"""
        while len(self.episodes) > self.max_episodes:
            oldest_id = self.episode_sequence.pop(0)
            if oldest_id in self.episodes:
                episode = self.episodes[oldest_id]
                del self.episodes[oldest_id]
                self._remove_from_indexes(episode)
    
    def _remove_from_indexes(self, episode: Episode):
        """Remove episode from all indexes"""
        self.agent_index.get(episode.agent, set()).discard(episode.episode_id)
        for tag in episode.tags:
            self.tag_index.get(tag, set()).discard(episode.episode_id)
        self.type_index.get(episode.episode_type.value, set()).discard(episode.episode_id)
        if episode.session_id:
            self.session_index.get(episode.session_id, set()).discard(episode.episode_id)
    
    def query(self, query: ChronicleQuery) -> List[Episode]:
        """
        Query episodes matching criteria.
        
        Args:
            query: ChronicleQuery with filters
            
        Returns:
            List of matching episodes (newest first)
        """
        with self._lock:
            # Start with candidate set from indexes
            candidates = None
            
            if query.agent:
                candidates = self.agent_index.get(query.agent, set())
            
            if query.episode_types:
                type_sets = [
                    self.type_index.get(t.value, set())
                    for t in query.episode_types
                ]
                type_union = set().union(*type_sets) if type_sets else set()
                candidates = candidates & type_union if candidates is not None else type_union
            
            if query.tags:
                tag_sets = [
                    self.tag_index.get(tag, set())
                    for tag in query.tags
                ]
                tag_union = set().union(*tag_sets) if tag_sets else set()
                candidates = candidates & tag_union if candidates is not None else tag_union
            
            # If no indexes used, check all episodes
            if candidates is None:
                candidates = set(self.episodes.keys())
            
            # Filter and sort
            results = [
                self.episodes[eid]
                for eid in candidates
                if query.matches(self.episodes[eid])
            ]
            
            # Sort by timestamp (newest first)
            results.sort(key=lambda e: e.timestamp_dt, reverse=True)
            
            return results[:query.limit]
    
    def find_similar_episodes(
        self,
        action_pattern: str,
        time_window: Optional[timedelta] = None,
        agent: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> List[Episode]:
        """
        Find episodes with similar actions.
        
        Args:
            action_pattern: Pattern to match in action descriptions
            time_window: Only look within this time window
            agent: Filter by agent
            tags: Filter by tags
            
        Returns:
            List of similar episodes
        """
        query = ChronicleQuery(
            agent=agent,
            tags=tags,
            limit=100
        )
        
        if time_window:
            end = datetime.now(timezone.utc)
            start = end - time_window
            query.time_range = (start, end)
        
        candidates = self.query(query)
        
        # Filter by action pattern
        pattern_lower = action_pattern.lower()
        similar = [
            ep for ep in candidates
            if pattern_lower in ep.action.lower() or
            any(pattern_lower in str(v).lower() for v in ep.context.values())
        ]
        
        return similar
    
    def have_we_tried_this(
        self,
        action_description: str,
        time_window: timedelta = timedelta(hours=1)
    ) -> Tuple[bool, Optional[str], List[Episode]]:
        """
        Check if we've tried this action before.
        
        Returns:
            Tuple of (tried_before, lesson_learned, related_episodes)
        """
        similar = self.find_similar_episodes(action_description, time_window)
        
        if not similar:
            return False, None, []
        
        # Get lessons learned from similar episodes
        lessons = [
            ep.lesson_learned for ep in similar
            if ep.lesson_learned
        ]
        
        lesson = lessons[0] if lessons else None
        
        return True, lesson, similar
    
    def get_strategy_effectiveness(
        self,
        strategy_name: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of a strategy over time.
        
        Args:
            strategy_name: Name of the strategy
            time_window: Time window to analyze
            
        Returns:
            Effectiveness report
        """
        episodes = self.find_similar_episodes(strategy_name, time_window)
        
        if not episodes:
            return {'strategy': strategy_name, 'uses': 0}
        
        successes = sum(1 for ep in episodes if ep.episode_type == EpisodeType.SUCCESS)
        failures = sum(1 for ep in episodes if ep.episode_type == EpisodeType.FAILURE)
        total = len(episodes)
        
        return {
            'strategy': strategy_name,
            'uses': total,
            'successes': successes,
            'failures': failures,
            'success_rate': successes / total if total > 0 else 0,
            'recent_lessons': [
                ep.lesson_learned for ep in episodes[:5]
                if ep.lesson_learned
            ]
        }
    
    def get_timeline(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        agent: Optional[str] = None
    ) -> List[Episode]:
        """
        Get episodes in chronological order.
        
        Args:
            start: Start time
            end: End time
            agent: Filter by agent
            
        Returns:
            Chronologically ordered episodes
        """
        query = ChronicleQuery(
            agent=agent,
            time_range=(start, end) if start and end else None,
            limit=self.max_episodes
        )
        
        results = self.query(query)
        # Reverse to get chronological order (oldest first)
        results.reverse()
        return results
    
    def get_session_narrative(self, session_id: str) -> List[Episode]:
        """
        Get all episodes for a session as a narrative.
        
        Args:
            session_id: Session ID
            
        Returns:
            Chronologically ordered episodes for the session
        """
        episode_ids = self.session_index.get(session_id, set())
        episodes = [self.episodes[eid] for eid in episode_ids]
        episodes.sort(key=lambda e: e.timestamp_dt)
        return episodes
    
    def _persist_episode(self, episode: Episode):
        """Persist a single episode to disk"""
        if not self.storage_path:
            return
        
        try:
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
            # Store in date-based directory
            date_str = episode.timestamp_dt.strftime('%Y-%m-%d')
            date_dir = self.storage_path / date_str
            date_dir.mkdir(exist_ok=True)
            
            file_path = date_dir / f"{episode.episode_id}.json"
            with open(file_path, 'w') as f:
                json.dump(episode.to_dict(), f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to persist episode: {e}")
    
    def _load(self):
        """Load episodes from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            # Find all episode files
            episode_files = list(self.storage_path.rglob("ep_*.json"))
            
            for file_path in sorted(episode_files):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        episode = Episode.from_dict(data)
                        
                        self.episodes[episode.episode_id] = episode
                        self.episode_sequence.append(episode.episode_id)
                        self._update_indexes(episode)
                        
                except Exception as e:
                    logger.warning(f"Failed to load episode from {file_path}: {e}")
            
            # Sort sequence by timestamp
            self.episode_sequence.sort(
                key=lambda eid: self.episodes[eid].timestamp_dt
            )
            
            logger.info(f"Loaded {len(self.episodes)} episodes from disk")
            
        except Exception as e:
            logger.error(f"Failed to load chronicle: {e}")
    
    def save(self):
        """Force save all episodes to disk"""
        if not self.storage_path:
            return
        
        for episode in self.episodes.values():
            self._persist_episode(episode)
        
        logger.info(f"Saved {len(self.episodes)} episodes to disk")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get chronicle statistics"""
        with self._lock:
            type_counts = {
                ep_type: len(eids)
                for ep_type, eids in self.type_index.items()
            }
            
            agent_counts = {
                agent: len(eids)
                for agent, eids in self.agent_index.items()
            }
            
            return {
                'total_episodes': len(self.episodes),
                'by_type': type_counts,
                'by_agent': agent_counts,
                'unique_tags': len(self.tag_index),
                'date_range': {
                    'oldest': self.episodes[self.episode_sequence[0]].timestamp if self.episode_sequence else None,
                    'newest': self.episodes[self.episode_sequence[-1]].timestamp if self.episode_sequence else None
                }
            }


# Integration helper for MasterKnowledgeEngine
class ChronicleIntegration:
    """Helper to integrate Chronicle with MasterKnowledgeEngine"""
    
    def __init__(self, chronicle: Chronicle):
        self.chronicle = chronicle
    
    def record_healing_attempt(
        self,
        failed_component: str,
        healing_strategy: str,
        outcome: str,
        lesson: Optional[str] = None
    ) -> Episode:
        """Record a self-healing attempt"""
        return self.chronicle.record_episode(
            agent="SelfHealingSystem",
            action=f"Attempted healing: {healing_strategy} for {failed_component}",
            episode_type=EpisodeType.HEALING,
            context={
                'failed_component': failed_component,
                'strategy': healing_strategy
            },
            outcome=outcome,
            lesson_learned=lesson,
            tags=['healing', failed_component, healing_strategy]
        )
    
    def record_component_execution(
        self,
        component: str,
        success: bool,
        outcome: str,
        duration_ms: float
    ) -> Episode:
        """Record a component execution"""
        return self.chronicle.record_episode(
            agent=component,
            action=f"Component execution: {component}",
            episode_type=EpisodeType.SUCCESS if success else EpisodeType.FAILURE,
            context={'duration_ms': duration_ms},
            outcome=outcome,
            tags=['execution', component]
        )
    
    def check_for_loops(
        self,
        action_pattern: str,
        threshold: int = 3,
        time_window: timedelta = timedelta(minutes=10)
    ) -> Tuple[bool, List[Episode]]:
        """
        Check if we're in a loop (repeating the same failed action).
        
        Returns:
            Tuple of (is_loop, recent_similar_episodes)
        """
        similar = self.chronicle.find_similar_episodes(action_pattern, time_window)
        
        # Count failures
        failures = [ep for ep in similar if ep.episode_type == EpisodeType.FAILURE]
        
        is_loop = len(failures) >= threshold
        
        if is_loop:
            logger.warning({
                'msg': 'Loop detected',
                'pattern': action_pattern,
                'failure_count': len(failures),
                'recommendation': 'Try a different strategy or escalate'
            })
        
        return is_loop, failures
