"""Score conversation trajectories with multiple judges.

Multi-judge consensus for robust evaluation of conversation paths.
"""

import statistics
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class CriterionType(Enum):
    """Types of scoring criteria."""
    COHERENCE = "coherence"
    GOAL_PROGRESS = "goal_progress"
    TONE = "tone"
    RELEVANCE = "relevance"
    CLARITY = "clarity"
    ENGAGEMENT = "engagement"
    EFFECTIVENESS = "effectiveness"
    USER_SATISFACTION = "user_satisfaction"


@dataclass
class ScoreResult:
    """Scoring output from trajectory evaluation.
    
    Attributes:
        overall_score: Aggregate score (0-10)
        criteria_scores: Scores per criterion
        judge_scores: Individual judge scores
        explanation: Text explanation of scoring
        confidence: Confidence in the score (0-1)
        metadata: Additional scoring data
    """
    overall_score: float
    criteria_scores: Dict[str, float] = field(default_factory=dict)
    judge_scores: List[float] = field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "overall_score": self.overall_score,
            "criteria_scores": self.criteria_scores,
            "judge_scores": self.judge_scores,
            "explanation": self.explanation,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }
    
    def get_variance(self) -> float:
        """Get variance in judge scores (measure of disagreement)."""
        if len(self.judge_scores) < 2:
            return 0.0
        return statistics.variance(self.judge_scores)
    
    def get_consensus_level(self) -> str:
        """Get consensus level based on score variance."""
        variance = self.get_variance()
        if variance < 0.5:
            return "high"
        elif variance < 2.0:
            return "medium"
        else:
            return "low"


@dataclass
class Judge:
    """Individual judge for trajectory scoring.
    
    Attributes:
        name: Judge identifier
        criteria: List of criteria this judge evaluates
        weights: Importance weights for each criterion
        bias: Systematic bias (can be positive or negative)
        scoring_function: Custom scoring function
    """
    name: str
    criteria: List[str] = field(default_factory=list)
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    scoring_function: Optional[Callable] = None
    llm_client: Optional[Any] = None
    
    def __post_init__(self):
        """Initialize default criteria and weights."""
        if not self.criteria:
            self.criteria = [
                "coherence",
                "goal_progress", 
                "tone",
                "relevance",
            ]
        
        if not self.weights:
            # Equal weights
            weight = 1.0 / len(self.criteria)
            self.weights = {c: weight for c in self.criteria}
    
    def evaluate(self, path: List[Any]) -> Dict[str, Any]:
        """Evaluate a conversation path.
        
        Args:
            path: List of ConversationNode objects
            
        Returns:
            Dict with scores and evaluation details
        """
        if not path:
            return {
                "overall": 0.0,
                "criteria": {c: 0.0 for c in self.criteria},
                "explanation": "Empty path",
            }
        
        # Use custom scoring function if provided
        if self.scoring_function:
            return self.scoring_function(path, self.criteria, self.weights)
        
        # Use LLM-based scoring if available
        if self.llm_client is not None:
            try:
                return self._evaluate_with_llm(path)
            except Exception as e:
                logger.warning(f"LLM scoring failed for {self.name}: {e}")
        
        # Fallback to heuristic scoring
        return self._evaluate_heuristic(path)
    
    def _evaluate_heuristic(self, path: List[Any]) -> Dict[str, Any]:
        """Evaluate path using heuristic rules."""
        criteria_scores = {}
        
        # Get conversation history
        messages = []
        for node in path:
            if hasattr(node, 'message'):
                messages.append(node.message)
            elif isinstance(node, dict):
                messages.append(node.get('message', ''))
        
        conversation_text = ' '.join(messages)
        
        # Coherence: Check for logical flow
        coherence_score = self._score_coherence(path, messages)
        criteria_scores["coherence"] = coherence_score
        
        # Goal progress: Based on depth and positive indicators
        goal_progress = self._score_goal_progress(path, conversation_text)
        criteria_scores["goal_progress"] = goal_progress
        
        # Tone: Check for positive language
        tone_score = self._score_tone(conversation_text)
        criteria_scores["tone"] = tone_score
        
        # Relevance: Check message relevance (simplified)
        relevance_score = self._score_relevance(path)
        criteria_scores["relevance"] = relevance_score
        
        # Clarity: Check for clarity markers
        clarity_score = self._score_clarity(conversation_text)
        criteria_scores["clarity"] = clarity_score
        
        # Engagement: Conversation length and depth
        engagement_score = self._score_engagement(path)
        criteria_scores["engagement"] = engagement_score
        
        # Calculate weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for criterion, score in criteria_scores.items():
            weight = self.weights.get(criterion, 0.25)
            weighted_sum += score * weight
            total_weight += weight
        
        overall = (weighted_sum / total_weight) + self.bias if total_weight > 0 else 5.0
        overall = max(0.0, min(10.0, overall))  # Clamp to 0-10
        
        return {
            "overall": overall,
            "criteria": criteria_scores,
            "explanation": f"Heuristic evaluation by {self.name}",
        }
    
    def _evaluate_with_llm(self, path: List[Any]) -> Dict[str, Any]:
        """Evaluate path using LLM."""
        # Placeholder for LLM-based evaluation
        # In actual implementation, would call LLM with structured prompt
        return self._evaluate_heuristic(path)
    
    def _score_coherence(self, path: List[Any], messages: List[str]) -> float:
        """Score conversation coherence."""
        if len(path) < 2:
            return 7.0  # Single message is coherent by default
        
        score = 7.0
        
        # Check for response-to-message flow
        for i in range(1, len(messages)):
            prev_msg = messages[i-1].lower()
            curr_msg = messages[i].lower()
            
            # Check for continuity markers
            continuity_words = ["yes", "no", "but", "and", "however", "agree", "disagree"]
            has_continuity = any(w in curr_msg for w in continuity_words)
            
            if has_continuity:
                score += 0.5
            
            # Penalize completely unrelated responses
            word_overlap = len(set(prev_msg.split()) & set(curr_msg.split()))
            if word_overlap == 0 and len(prev_msg) > 10:
                score -= 1.0
        
        return max(0.0, min(10.0, score))
    
    def _score_goal_progress(self, path: List[Any], text: str) -> float:
        """Score progress toward conversation goal."""
        # Higher score for longer, successful conversations
        base_score = min(5.0 + len(path) * 0.5, 8.0)
        
        # Check for goal achievement markers
        success_markers = ["thank", "great", "perfect", "solved", "understand", "clear"]
        success_count = sum(1 for m in success_markers if m in text.lower())
        
        score = base_score + success_count * 0.5
        return min(10.0, score)
    
    def _score_tone(self, text: str) -> float:
        """Score conversation tone."""
        positive_words = [
            "thank", "good", "great", "excellent", "helpful", "clear",
            "appreciate", "nice", "wonderful", "perfect", "thanks"
        ]
        negative_words = [
            "bad", "terrible", "awful", "hate", "useless", "stupid",
            "wrong", "annoying", "frustrated", "angry"
        ]
        
        positive_count = sum(1 for w in positive_words if w in text.lower())
        negative_count = sum(1 for w in negative_words if w in text.lower())
        
        score = 7.0 + (positive_count * 0.5) - (negative_count * 1.0)
        return max(0.0, min(10.0, score))
    
    def _score_relevance(self, path: List[Any]) -> float:
        """Score message relevance."""
        # Higher score for paths with more context retention
        if not path:
            return 0.0
        
        # Check node metadata for relevance indicators
        relevance_scores = []
        for node in path:
            if hasattr(node, 'metadata') and node.metadata:
                rel = node.metadata.get('relevance', 0.5)
                relevance_scores.append(rel)
        
        if relevance_scores:
            return sum(relevance_scores) / len(relevance_scores) * 10
        
        return 7.0  # Default
    
    def _score_clarity(self, text: str) -> float:
        """Score message clarity."""
        # Factors: sentence length, question marks, structure
        sentences = text.split('.')
        avg_sentence_len = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        
        # Optimal sentence length around 15-20 words
        if avg_sentence_len < 25:
            score = 8.0
        elif avg_sentence_len < 40:
            score = 6.0
        else:
            score = 4.0
        
        # Penalize excessive questions
        question_count = text.count('?')
        if question_count > 5:
            score -= 1.0
        
        # Bonus for structure markers
        structure_markers = ["first", "second", "third", "finally", "however", "therefore"]
        structure_count = sum(1 for m in structure_markers if m in text.lower())
        score += structure_count * 0.3
        
        return max(0.0, min(10.0, score))
    
    def _score_engagement(self, path: List[Any]) -> float:
        """Score user engagement."""
        if not path:
            return 0.0
        
        # Length bonus
        length_score = min(len(path) * 0.8, 6.0)
        
        # Depth bonus
        max_depth = 0
        for node in path:
            if hasattr(node, 'depth'):
                max_depth = max(max_depth, node.depth)
        
        depth_score = min(max_depth * 0.5, 2.0)
        
        return min(10.0, 4.0 + length_score + depth_score)


@dataclass
class TrajectoryScorer:
    """Main scoring engine for conversation trajectories.
    
    Uses multiple independent judges for robust evaluation.
    """
    judges: List[Judge] = field(default_factory=list)
    aggregation_method: str = "median"  # median, mean, weighted_mean
    require_consensus: bool = True
    min_judges: int = 3
    
    def __post_init__(self):
        """Initialize default judges if none provided."""
        if not self.judges:
            self.judges = self._create_default_judges()
    
    def _create_default_judges(self) -> List[Judge]:
        """Create default set of diverse judges."""
        return [
            Judge(
                name="CoherenceJudge",
                criteria=["coherence", "clarity", "relevance"],
                weights={"coherence": 0.4, "clarity": 0.3, "relevance": 0.3},
            ),
            Judge(
                name="GoalJudge",
                criteria=["goal_progress", "effectiveness", "relevance"],
                weights={"goal_progress": 0.5, "effectiveness": 0.3, "relevance": 0.2},
            ),
            Judge(
                name="ToneJudge",
                criteria=["tone", "user_satisfaction", "engagement"],
                weights={"tone": 0.3, "user_satisfaction": 0.4, "engagement": 0.3},
            ),
        ]
    
    def add_judge(self, judge: Judge) -> None:
        """Add a judge to the panel."""
        self.judges.append(judge)
    
    def score_trajectory(self, path: List[Any]) -> ScoreResult:
        """Score a conversation trajectory with multiple judges.
        
        Args:
            path: List of conversation nodes
            
        Returns:
            ScoreResult with aggregated scores
        """
        if not path:
            return ScoreResult(
                overall_score=0.0,
                explanation="Empty trajectory",
                confidence=0.0,
            )
        
        # Get scores from all judges
        judge_results = []
        all_criteria_scores: Dict[str, List[float]] = {}
        
        for judge in self.judges:
            try:
                result = judge.evaluate(path)
                judge_results.append(result)
                
                # Collect criteria scores
                for criterion, score in result.get("criteria", {}).items():
                    if criterion not in all_criteria_scores:
                        all_criteria_scores[criterion] = []
                    all_criteria_scores[criterion].append(score)
            except Exception as e:
                logger.warning(f"Judge {judge.name} failed: {e}")
        
        if not judge_results:
            return ScoreResult(
                overall_score=5.0,
                explanation="All judges failed",
                confidence=0.0,
            )
        
        # Extract overall scores
        overall_scores = [r.get("overall", 5.0) for r in judge_results]
        
        # Aggregate overall score
        overall = self.aggregate_scores(overall_scores)
        
        # Aggregate criteria scores
        criteria_scores = {}
        for criterion, scores in all_criteria_scores.items():
            criteria_scores[criterion] = self.aggregate_scores(scores)
        
        # Calculate confidence based on agreement
        if len(overall_scores) >= 2:
            variance = statistics.variance(overall_scores)
            confidence = max(0.0, 1.0 - (variance / 10.0))
        else:
            confidence = 0.5
        
        # Build explanation
        explanation = self._build_explanation(judge_results, overall)
        
        return ScoreResult(
            overall_score=overall,
            criteria_scores=criteria_scores,
            judge_scores=overall_scores,
            explanation=explanation,
            confidence=confidence,
            metadata={
                "num_judges": len(judge_results),
                "judge_names": [j.name for j in self.judges],
                "variance": statistics.variance(overall_scores) if len(overall_scores) >= 2 else 0.0,
            }
        )
    
    def aggregate_scores(self, scores: List[float]) -> float:
        """Aggregate multiple scores using configured method.
        
        Args:
            scores: List of scores
            
        Returns:
            Aggregated score
        """
        if not scores:
            return 0.0
        
        if self.aggregation_method == "median":
            return statistics.median(scores)
        elif self.aggregation_method == "mean":
            return statistics.mean(scores)
        elif self.aggregation_method == "weighted_mean":
            # Weight by inverse variance (more agreement = higher weight)
            if len(scores) < 2:
                return scores[0]
            try:
                var = statistics.variance(scores)
                if var < 0.01:  # Avoid division by zero
                    return statistics.mean(scores)
                weights = [1.0 / max(var, 0.1) for _ in scores]
                total_weight = sum(weights)
                return sum(s * w for s, w in zip(scores, weights)) / total_weight
            except statistics.StatisticsError:
                return statistics.mean(scores)
        else:
            return statistics.median(scores)
    
    def _build_explanation(self, judge_results: List[Dict], overall: float) -> str:
        """Build human-readable explanation of scoring."""
        parts = [f"Overall score: {overall:.1f}/10"]
        
        parts.append(f"Judges: {len(judge_results)}")
        
        for result in judge_results:
            judge_name = result.get("judge_name", "Unknown")
            score = result.get("overall", 0.0)
            parts.append(f"  {judge_name}: {score:.1f}")
        
        # Add strongest criterion
        all_criteria = {}
        for result in judge_results:
            for criterion, score in result.get("criteria", {}).items():
                if criterion not in all_criteria:
                    all_criteria[criterion] = []
                all_criteria[criterion].append(score)
        
        if all_criteria:
            avg_criteria = {c: statistics.mean(s) for c, s in all_criteria.items()}
            best = max(avg_criteria, key=avg_criteria.get)
            worst = min(avg_criteria, key=avg_criteria.get)
            parts.append(f"Best criterion: {best} ({avg_criteria[best]:.1f})")
            parts.append(f"Weakest criterion: {worst} ({avg_criteria[worst]:.1f})")
        
        return "\n".join(parts)
    
    def compare_trajectories(self, paths: List[List[Any]]) -> List[ScoreResult]:
        """Score and compare multiple trajectories.
        
        Args:
            paths: List of conversation paths
            
        Returns:
            List of ScoreResults in same order
        """
        results = []
        for path in paths:
            result = self.score_trajectory(path)
            results.append(result)
        
        return results
    
    def rank_trajectories(self, paths: List[List[Any]]) -> List[Tuple[int, ScoreResult]]:
        """Rank trajectories by score.
        
        Args:
            paths: List of conversation paths
            
        Returns:
            List of (original_index, score_result) sorted by score descending
        """
        results = self.compare_trajectories(paths)
        indexed = list(enumerate(results))
        indexed.sort(key=lambda x: x[1].overall_score, reverse=True)
        return indexed
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get scorer statistics."""
        return {
            "num_judges": len(self.judges),
            "judge_names": [j.name for j in self.judges],
            "aggregation_method": self.aggregation_method,
            "require_consensus": self.require_consensus,
        }


# Type hint for Any
from typing import Any
