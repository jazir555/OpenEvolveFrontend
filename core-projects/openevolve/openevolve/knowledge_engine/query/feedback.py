"""
Feedback Loop for Query Improvement

Collects and processes user feedback to improve query results over time.

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

import logging
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of feedback"""
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RELEVANT = "relevant"
    IRRELEVANT = "irrelevant"
    CLICKED = "clicked"
    SKIPPED = "skipped"
    COMMENT = "comment"


@dataclass
class QueryFeedback:
    """Feedback for a query-result pair"""
    query: str
    result_id: str
    feedback_type: FeedbackType
    user_id: Optional[str] = None
    comment: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "result_id": self.result_id,
            "feedback_type": self.feedback_type.value,
            "user_id": self.user_id,
            "comment": self.comment,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class QueryImprovement:
    """Suggested improvement for a query pattern"""
    pattern: str  # Regex pattern or query string
    suggestion: str
    reason: str
    confidence: float
    based_on_feedback_count: int


class FeedbackLoop:
    """
    Feedback loop for continuous query improvement
    
    Collects user feedback and uses it to:
    - Improve result ranking
    - Identify query patterns that need improvement
    - Suggest query expansions
    - Learn user preferences
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.feedback_history: List[QueryFeedback] = []
        self._lock = threading.RLock()
        
        # Query pattern statistics
        self.pattern_stats: Dict[str, Dict[str, Any]] = {}
        
        # User preferences (user_id -> preferences)
        self.user_preferences: Dict[str, Dict[str, Any]] = {}
        
        # Load existing feedback
        if self.storage_path:
            self._load_feedback()
    
    def record_feedback(self, feedback: QueryFeedback):
        """Record user feedback"""
        with self._lock:
            self.feedback_history.append(feedback)
            
            # Update pattern stats
            self._update_pattern_stats(feedback)
            
            # Update user preferences
            if feedback.user_id:
                self._update_user_preferences(feedback)
        
        # Persist feedback
        if self.storage_path:
            self._save_feedback()
        
        logger.debug(f"Recorded feedback: {feedback.feedback_type.value} for '{feedback.query}'")
    
    def _update_pattern_stats(self, feedback: QueryFeedback):
        """Update statistics for query patterns"""
        # Extract pattern from query
        pattern = self._extract_pattern(feedback.query)
        
        if pattern not in self.pattern_stats:
            self.pattern_stats[pattern] = {
                "total_feedback": 0,
                "positive": 0,
                "negative": 0,
                "queries": []
            }
        
        stats = self.pattern_stats[pattern]
        stats["total_feedback"] += 1
        stats["queries"].append(feedback.query)
        
        if feedback.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.RELEVANT, FeedbackType.CLICKED):
            stats["positive"] += 1
        elif feedback.feedback_type in (FeedbackType.THUMBS_DOWN, FeedbackType.IRRELEVANT, FeedbackType.SKIPPED):
            stats["negative"] += 1
        
        # Keep only last 100 queries
        stats["queries"] = stats["queries"][-100:]
    
    def _update_user_preferences(self, feedback: QueryFeedback):
        """Update user preference model"""
        user_id = feedback.user_id
        
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "liked_topics": [],
                "disliked_topics": [],
                "preferred_sources": [],
                "query_history": []
            }
        
        prefs = self.user_preferences[user_id]
        
        # Extract topics from query
        topics = self._extract_topics(feedback.query)
        
        if feedback.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.RELEVANT):
            prefs["liked_topics"].extend(topics)
        elif feedback.feedback_type in (FeedbackType.THUMBS_DOWN, FeedbackType.IRRELEVANT):
            prefs["disliked_topics"].extend(topics)
        
        # Track source preference
        source = feedback.metadata.get("source")
        if source and feedback.feedback_type == FeedbackType.THUMBS_UP:
            if source not in prefs["preferred_sources"]:
                prefs["preferred_sources"].append(source)
        
        # Keep lists manageable
        prefs["liked_topics"] = prefs["liked_topics"][-50:]
        prefs["disliked_topics"] = prefs["disliked_topics"][-50:]
    
    def _extract_pattern(self, query: str) -> str:
        """Extract a pattern from a query"""
        # Simple pattern extraction: first 2-3 words
        words = query.lower().split()
        if len(words) >= 3:
            return " ".join(words[:3])
        return query.lower()
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topics from a query"""
        # Simple topic extraction: nouns and proper nouns
        import re
        words = re.findall(r'\b[A-Z][a-z]+\b|\b[a-z]{5,}\b', query)
        return [w.lower() for w in words][:5]
    
    def get_suggestions(self, query: str) -> List[QueryImprovement]:
        """Get improvement suggestions for a query"""
        suggestions = []
        
        # Check if similar queries had issues
        pattern = self._extract_pattern(query)
        
        if pattern in self.pattern_stats:
            stats = self.pattern_stats[pattern]
            
            if stats["total_feedback"] > 5:
                positive_rate = stats["positive"] / stats["total_feedback"]
                
                if positive_rate < 0.5:
                    suggestions.append(QueryImprovement(
                        pattern=pattern,
                        suggestion="Consider reformulating this query type",
                        reason=f"Low satisfaction rate: {positive_rate:.1%}",
                        confidence=1 - positive_rate,
                        based_on_feedback_count=stats["total_feedback"]
                    ))
        
        return suggestions
    
    def adjust_scores(
        self,
        query: str,
        results: List[Dict[str, Any]],
        user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Adjust result scores based on feedback history
        
        This implements a simple learning-to-rank adjustment
        based on past feedback.
        """
        if not self.feedback_history:
            return results
        
        adjusted = []
        
        for result in results:
            score_adjustment = 0.0
            result_id = result.get("id", "")
            
            # Find similar past feedback
            for feedback in self.feedback_history:
                if feedback.result_id == result_id:
                    if feedback.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.RELEVANT):
                        score_adjustment += 0.1
                    elif feedback.feedback_type in (FeedbackType.THUMBS_DOWN, FeedbackType.IRRELEVANT):
                        score_adjustment -= 0.2
            
            # Apply user preferences
            if user_id and user_id in self.user_preferences:
                prefs = self.user_preferences[user_id]
                
                # Boost preferred sources
                source = result.get("source", "")
                if source in prefs.get("preferred_sources", []):
                    score_adjustment += 0.05
                
                # Boost liked topics
                result_text = result.get("content", "") + " " + str(result.get("metadata", {}))
                for topic in prefs.get("liked_topics", []):
                    if topic in result_text.lower():
                        score_adjustment += 0.02
            
            # Create adjusted result
            adjusted_result = result.copy()
            adjusted_result["score"] = result.get("score", 0) + score_adjustment
            adjusted_result["score_adjustment"] = score_adjustment
            adjusted.append(adjusted_result)
        
        # Re-sort by adjusted scores
        adjusted.sort(key=lambda r: r["score"], reverse=True)
        
        return adjusted
    
    def get_insights(self) -> Dict[str, Any]:
        """Get insights from collected feedback"""
        with self._lock:
            total_feedback = len(self.feedback_history)
            
            if total_feedback == 0:
                return {"message": "No feedback collected yet"}
            
            # Calculate overall metrics
            positive = sum(
                1 for f in self.feedback_history
                if f.feedback_type in (FeedbackType.THUMBS_UP, FeedbackType.RELEVANT)
            )
            
            negative = sum(
                1 for f in self.feedback_history
                if f.feedback_type in (FeedbackType.THUMBS_DOWN, FeedbackType.IRRELEVANT)
            )
            
            satisfaction_rate = positive / (positive + negative) if (positive + negative) > 0 else 0
            
            # Find problematic patterns
            problematic = [
                {
                    "pattern": pattern,
                    "satisfaction_rate": stats["positive"] / stats["total_feedback"],
                    "feedback_count": stats["total_feedback"]
                }
                for pattern, stats in self.pattern_stats.items()
                if stats["total_feedback"] >= 5
                and stats["positive"] / stats["total_feedback"] < 0.5
            ]
            
            return {
                "total_feedback": total_feedback,
                "satisfaction_rate": satisfaction_rate,
                "positive_count": positive,
                "negative_count": negative,
                "unique_patterns": len(self.pattern_stats),
                "problematic_patterns": sorted(
                    problematic,
                    key=lambda x: x["satisfaction_rate"]
                )[:10]
            }
    
    def _save_feedback(self):
        """Save feedback to disk"""
        if not self.storage_path:
            return
        
        try:
            data = {
                "feedback": [f.to_dict() for f in self.feedback_history],
                "pattern_stats": self.pattern_stats,
                "user_preferences": self.user_preferences
            }
            
            self.storage_path.write_text(json.dumps(data, indent=2))
            
        except Exception as e:
            logger.error(f"Failed to save feedback: {e}")
    
    def _load_feedback(self):
        """Load feedback from disk"""
        if not self.storage_path or not self.storage_path.exists():
            return
        
        try:
            data = json.loads(self.storage_path.read_text())
            
            self.feedback_history = [
                QueryFeedback(
                    query=f["query"],
                    result_id=f["result_id"],
                    feedback_type=FeedbackType(f["feedback_type"]),
                    user_id=f.get("user_id"),
                    comment=f.get("comment"),
                    metadata=f.get("metadata", {}),
                    timestamp=datetime.fromisoformat(f["timestamp"])
                )
                for f in data.get("feedback", [])
            ]
            
            self.pattern_stats = data.get("pattern_stats", {})
            self.user_preferences = data.get("user_preferences", {})
            
            logger.info(f"Loaded {len(self.feedback_history)} feedback entries")
            
        except Exception as e:
            logger.error(f"Failed to load feedback: {e}")
    
    def export_feedback(self, output_path: str):
        """Export feedback to a file for analysis"""
        with self._lock:
            data = {
                "export_time": datetime.utcnow().isoformat(),
                "feedback_count": len(self.feedback_history),
                "feedback": [f.to_dict() for f in self.feedback_history],
                "insights": self.get_insights()
            }
            
            Path(output_path).write_text(json.dumps(data, indent=2))
        
        logger.info(f"Exported feedback to {output_path}")
