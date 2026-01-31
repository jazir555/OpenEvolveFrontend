"""
Knowledge Analytics Engine

Provides advanced analytics and insights for the knowledge base including:
- Trend analysis
- Knowledge quality metrics
- Usage analytics
- Predictive insights
- Anomaly detection
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
import statistics
from collections import Counter

from enhanced_knowledge_core import KnowledgeItem, KnowledgeType

logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesPoint:
    """A single point in a time series."""
    timestamp: datetime
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    metric_name: str
    direction: str  # "increasing", "decreasing", "stable"
    slope: float
    confidence: float
    recent_average: float
    historical_average: float
    change_percent: float
    forecast: List[TimeSeriesPoint] = field(default_factory=list)


@dataclass
class KnowledgeCluster:
    """A cluster of related knowledge items."""
    cluster_id: str
    centroid: List[float]
    items: List[str]  # item IDs
    dominant_type: KnowledgeType
    dominant_tags: Set[str]
    coherence_score: float
    created_at: datetime = field(default_factory=datetime.utcnow)


class TrendAnalyzer:
    """Analyze trends in knowledge metrics over time."""
    
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self._time_series: Dict[str, List[TimeSeriesPoint]] = defaultdict(list)
    
    def add_data_point(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Add a data point to a time series."""
        point = TimeSeriesPoint(
            timestamp=datetime.utcnow(),
            value=value,
            metadata=metadata or {}
        )
        self._time_series[metric_name].append(point)
        
        # Keep only recent data
        cutoff = datetime.utcnow() - timedelta(days=self.window_size)
        self._time_series[metric_name] = [
            p for p in self._time_series[metric_name] 
            if p.timestamp > cutoff
        ]
    
    def analyze_trend(self, metric_name: str) -> Optional[TrendAnalysis]:
        """Analyze trend for a specific metric."""
        series = self._time_series.get(metric_name, [])
        
        if len(series) < 3:
            return None
        
        # Split into recent and historical
        mid = len(series) // 2
        recent = series[mid:]
        historical = series[:mid]
        
        recent_avg = statistics.mean(p.value for p in recent)
        historical_avg = statistics.mean(p.value for p in historical) if historical else recent_avg
        
        # Calculate trend using simple linear regression
        x = list(range(len(series)))
        y = [p.value for p in series]
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2) if (n * sum_x2 - sum_x ** 2) != 0 else 0
        
        # Determine direction
        if slope > 0.01:
            direction = "increasing"
        elif slope < -0.01:
            direction = "decreasing"
        else:
            direction = "stable"
        
        # Calculate confidence (R-squared approximation)
        mean_y = sum_y / n
        ss_tot = sum((yi - mean_y) ** 2 for yi in y)
        ss_res = sum((yi - (slope * xi + (sum_y - slope * sum_x) / n)) ** 2 for xi, yi in zip(x, y))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Calculate percent change
        change_percent = ((recent_avg - historical_avg) / historical_avg * 100) if historical_avg != 0 else 0
        
        # Simple forecast (next 7 points)
        forecast = []
        for i in range(7):
            forecast_value = slope * (n + i) + (sum_y - slope * sum_x) / n
            forecast.append(TimeSeriesPoint(
                timestamp=datetime.utcnow() + timedelta(days=i),
                value=max(0, forecast_value)
            ))
        
        return TrendAnalysis(
            metric_name=metric_name,
            direction=direction,
            slope=slope,
            confidence=r_squared,
            recent_average=recent_avg,
            historical_average=historical_avg,
            change_percent=change_percent,
            forecast=forecast
        )
    
    def get_all_trends(self) -> Dict[str, TrendAnalysis]:
        """Get trend analysis for all metrics."""
        return {
            name: analysis 
            for name, analysis in 
            [(name, self.analyze_trend(name)) for name in self._time_series.keys()]
            if analysis is not None
        }


class KnowledgeQualityAnalyzer:
    """Analyze knowledge quality metrics."""
    
    def __init__(self):
        self.quality_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "fair": 0.5,
            "poor": 0.3
        }
    
    def analyze_completeness(self, item: KnowledgeItem) -> float:
        """Analyze completeness of a knowledge item."""
        score = 0.0
        checks = 0
        
        # Check content presence
        if item.content:
            score += 1.0
        checks += 1
        
        # Check metadata
        if item.metadata:
            score += min(1.0, len(item.metadata) / 5.0)
        checks += 1
        
        # Check tags
        if item.tags:
            score += min(1.0, len(item.tags) / 3.0)
        checks += 1
        
        # Check embedding
        if item.embedding:
            score += 1.0
        checks += 1
        
        return score / checks if checks > 0 else 0.0
    
    def analyze_consistency(self, items: List[KnowledgeItem]) -> float:
        """Analyze consistency across knowledge items."""
        if not items:
            return 0.0
        
        # Check for duplicate content
        content_hashes = [hash(str(item.content)) for item in items]
        unique_hashes = len(set(content_hashes))
        duplication_ratio = unique_hashes / len(items) if items else 1.0
        
        # Check metadata consistency
        all_metadata_keys = set()
        for item in items:
            all_metadata_keys.update(item.metadata.keys())
        
        if not all_metadata_keys:
            metadata_consistency = 1.0
        else:
            metadata_scores = []
            for item in items:
                item_keys = set(item.metadata.keys())
                coverage = len(item_keys) / len(all_metadata_keys)
                metadata_scores.append(coverage)
            metadata_consistency = statistics.mean(metadata_scores)
        
        return (duplication_ratio + metadata_consistency) / 2.0
    
    def calculate_quality_score(self, item: KnowledgeItem) -> Dict[str, float]:
        """Calculate comprehensive quality score for an item."""
        completeness = self.analyze_completeness(item)
        
        # Confidence factor
        confidence_score = item.confidence
        
        # Freshness factor
        age_days = (datetime.utcnow() - item.created_at).days
        freshness = max(0.0, 1.0 - (age_days / 365.0))  # Decay over 1 year
        
        # Version factor (more versions = more refined)
        version_factor = min(1.0, item.version / 5.0)
        
        overall = statistics.mean([completeness, confidence_score, freshness, version_factor])
        
        return {
            "overall": overall,
            "completeness": completeness,
            "confidence": confidence_score,
            "freshness": freshness,
            "version_factor": version_factor,
            "quality_category": self._categorize_quality(overall)
        }
    
    def _categorize_quality(self, score: float) -> str:
        """Categorize quality score."""
        for category, threshold in sorted(
            self.quality_thresholds.items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            if score >= threshold:
                return category
        return "invalid"
    
    def generate_quality_report(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Generate comprehensive quality report."""
        if not items:
            return {"error": "No items to analyze"}
        
        scores = [self.calculate_quality_score(item) for item in items]
        
        categories = Counter(s["quality_category"] for s in scores)
        
        return {
            "total_items": len(items),
            "average_overall_score": statistics.mean(s["overall"] for s in scores),
            "average_completeness": statistics.mean(s["completeness"] for s in scores),
            "average_confidence": statistics.mean(s["confidence"] for s in scores),
            "quality_distribution": dict(categories),
            "consistency_score": self.analyze_consistency(items),
            "recommendations": self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores: List[Dict[str, float]]) -> List[str]:
        """Generate improvement recommendations."""
        recommendations = []
        
        avg_completeness = statistics.mean(s["completeness"] for s in scores)
        avg_confidence = statistics.mean(s["confidence"] for s in scores)
        
        if avg_completeness < 0.7:
            recommendations.append("Improve item completeness by adding more metadata and tags")
        
        if avg_confidence < 0.7:
            recommendations.append("Review low-confidence items and verify content accuracy")
        
        poor_count = sum(1 for s in scores if s["quality_category"] == "poor")
        if poor_count > len(scores) * 0.2:
            recommendations.append(f"Address {poor_count} low-quality items through review or removal")
        
        return recommendations


class UsageAnalytics:
    """Track and analyze knowledge usage patterns."""
    
    def __init__(self, retention_days: int = 90):
        self.retention_days = retention_days
        self._access_log: List[Dict[str, Any]] = []
        self._search_log: List[Dict[str, Any]] = []
        self._feedback_log: List[Dict[str, Any]] = []
    
    def log_access(self, item_id: str, user_id: Optional[str] = None, context: Optional[Dict] = None):
        """Log an item access."""
        self._access_log.append({
            "item_id": item_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "context": context or {}
        })
        self._cleanup_old_logs()
    
    def log_search(self, query: str, results_count: int, user_id: Optional[str] = None):
        """Log a search query."""
        self._search_log.append({
            "query": query,
            "results_count": results_count,
            "user_id": user_id,
            "timestamp": datetime.utcnow()
        })
        self._cleanup_old_logs()
    
    def log_feedback(self, item_id: str, feedback_type: str, score: float):
        """Log user feedback."""
        self._feedback_log.append({
            "item_id": item_id,
            "feedback_type": feedback_type,
            "score": score,
            "timestamp": datetime.utcnow()
        })
        self._cleanup_old_logs()
    
    def _cleanup_old_logs(self):
        """Remove logs older than retention period."""
        cutoff = datetime.utcnow() - timedelta(days=self.retention_days)
        
        self._access_log = [log for log in self._access_log if log["timestamp"] > cutoff]
        self._search_log = [log for log in self._search_log if log["timestamp"] > cutoff]
        self._feedback_log = [log for log in self._feedback_log if log["timestamp"] > cutoff]
    
    def get_popular_items(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Get most frequently accessed items."""
        item_counts = Counter(log["item_id"] for log in self._access_log)
        return item_counts.most_common(limit)
    
    def get_search_trends(self) -> Dict[str, Any]:
        """Analyze search trends."""
        if not self._search_log:
            return {"error": "No search data available"}
        
        # Popular queries
        query_counts = Counter(log["query"] for log in self._search_log)
        
        # Average results per search
        avg_results = statistics.mean(log["results_count"] for log in self._search_log)
        
        # Searches over time (last 7 days)
        daily_counts = defaultdict(int)
        for log in self._search_log:
            day_key = log["timestamp"].strftime("%Y-%m-%d")
            daily_counts[day_key] += 1
        
        return {
            "total_searches": len(self._search_log),
            "unique_queries": len(query_counts),
            "top_queries": query_counts.most_common(10),
            "average_results_count": avg_results,
            "daily_search_volume": dict(daily_counts)
        }
    
    def get_feedback_summary(self) -> Dict[str, Any]:
        """Summarize user feedback."""
        if not self._feedback_log:
            return {"error": "No feedback data available"}
        
        type_counts = Counter(log["feedback_type"] for log in self._feedback_log)
        avg_score = statistics.mean(log["score"] for log in self._feedback_log)
        
        # Feedback over time
        daily_feedback = defaultdict(lambda: {"count": 0, "avg_score": []})
        for log in self._feedback_log:
            day_key = log["timestamp"].strftime("%Y-%m-%d")
            daily_feedback[day_key]["count"] += 1
            daily_feedback[day_key]["avg_score"].append(log["score"])
        
        for day in daily_feedback:
            scores = daily_feedback[day]["avg_score"]
            daily_feedback[day]["avg_score"] = statistics.mean(scores) if scores else 0
        
        return {
            "total_feedback": len(self._feedback_log),
            "feedback_types": dict(type_counts),
            "average_score": avg_score,
            "daily_feedback": dict(daily_feedback)
        }


class AnomalyDetector:
    """Detect anomalies in knowledge metrics."""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity
    
    def detect_statistical_anomalies(
        self, 
        data: List[float], 
        labels: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods."""
        if len(data) < 3:
            return []
        
        mean = statistics.mean(data)
        std_dev = statistics.stdev(data) if len(data) > 1 else 0
        
        anomalies = []
        for i, value in enumerate(data):
            z_score = (value - mean) / std_dev if std_dev > 0 else 0
            
            if abs(z_score) > self.sensitivity:
                anomalies.append({
                    "index": i,
                    "label": labels[i] if labels else str(i),
                    "value": value,
                    "z_score": z_score,
                    "type": "high" if z_score > 0 else "low"
                })
        
        return anomalies
    
    def detect_knowledge_gaps(
        self, 
        items: List[KnowledgeItem],
        expected_coverage: Optional[Set[str]] = None
    ) -> List[Dict[str, Any]]:
        """Detect gaps in knowledge coverage."""
        gaps = []
        
        # Analyze type coverage
        type_counts = Counter(item.knowledge_type for item in items)
        total = len(items)
        
        for ktype in KnowledgeType:
            count = type_counts.get(ktype, 0)
            percentage = count / total * 100 if total > 0 else 0
            
            if percentage < 5:  # Less than 5% coverage
                gaps.append({
                    "type": "low_coverage",
                    "category": ktype.value,
                    "current_count": count,
                    "percentage": percentage,
                    "severity": "high" if percentage == 0 else "medium"
                })
        
        # Check for expected topics
        if expected_coverage:
            actual_tags = set()
            for item in items:
                actual_tags.update(item.tags)
            
            missing = expected_coverage - actual_tags
            for tag in missing:
                gaps.append({
                    "type": "missing_topic",
                    "topic": tag,
                    "severity": "medium"
                })
        
        return gaps


class KnowledgeAnalyticsEngine:
    """Main analytics engine integrating all analyzers."""
    
    def __init__(self, knowledge_engine=None):
        self.knowledge_engine = knowledge_engine
        self.trend_analyzer = TrendAnalyzer()
        self.quality_analyzer = KnowledgeQualityAnalyzer()
        self.usage_analytics = UsageAnalytics()
        self.anomaly_detector = AnomalyDetector()
        
        self._last_report: Optional[Dict[str, Any]] = None
        self._last_report_time: Optional[datetime] = None
    
    def record_metric(self, metric_name: str, value: float, metadata: Optional[Dict] = None):
        """Record a metric for trend analysis."""
        self.trend_analyzer.add_data_point(metric_name, value, metadata)
    
    def generate_comprehensive_report(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Generate comprehensive analytics report."""
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "quality": self.quality_analyzer.generate_quality_report(items),
            "trends": self.trend_analyzer.get_all_trends(),
            "usage": {
                "search": self.usage_analytics.get_search_trends(),
                "feedback": self.usage_analytics.get_feedback_summary(),
                "popular_items": self.usage_analytics.get_popular_items()
            },
            "anomalies": self._detect_anomalies(items),
            "gaps": self.anomaly_detector.detect_knowledge_gaps(items)
        }
        
        # Add insights
        report["insights"] = self._generate_insights(report)
        
        self._last_report = report
        self._last_report_time = datetime.utcnow()
        
        return report
    
    def _detect_anomalies(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Detect anomalies in knowledge base."""
        anomalies = {}
        
        # Check for confidence anomalies
        confidences = [item.confidence for item in items]
        confidence_labels = [item.id for item in items]
        anomalies["confidence"] = self.anomaly_detector.detect_statistical_anomalies(
            confidences, confidence_labels
        )
        
        # Check for age anomalies (very old items)
        ages = [(datetime.utcnow() - item.created_at).days for item in items]
        anomalies["age"] = self.anomaly_detector.detect_statistical_anomalies(
            ages, confidence_labels
        )
        
        return anomalies
    
    def _generate_insights(self, report: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable insights from the report."""
        insights = []
        
        # Quality insights
        quality = report.get("quality", {})
        avg_score = quality.get("average_overall_score", 0)
        
        if avg_score < 0.6:
            insights.append({
                "category": "quality",
                "severity": "high",
                "message": f"Overall knowledge quality is low ({avg_score:.2f}). Consider review process."
            })
        
        # Trend insights
        trends = report.get("trends", {})
        for metric, trend in trends.items():
            if trend.direction == "decreasing" and trend.confidence > 0.7:
                insights.append({
                    "category": "trend",
                    "severity": "medium",
                    "message": f"{metric} is declining ({trend.change_percent:.1f}% change)"
                })
        
        # Usage insights
        usage = report.get("usage", {})
        search_trends = usage.get("search", {})
        if search_trends.get("average_results_count", 10) < 3:
            insights.append({
                "category": "search",
                "severity": "medium",
                "message": "Low search result counts suggest potential knowledge gaps"
            })
        
        # Gap insights
        gaps = report.get("gaps", [])
        high_priority_gaps = [g for g in gaps if g.get("severity") == "high"]
        if high_priority_gaps:
            insights.append({
                "category": "coverage",
                "severity": "high",
                "message": f"{len(high_priority_gaps)} high-priority knowledge gaps detected"
            })
        
        return insights
    
    def get_dashboard_data(self, items: List[KnowledgeItem]) -> Dict[str, Any]:
        """Get data formatted for dashboard display."""
        report = self.generate_comprehensive_report(items)
        
        return {
            "summary": {
                "total_items": len(items),
                "avg_quality": report["quality"].get("average_overall_score", 0),
                "active_trends": len([t for t in report["trends"].values() if t.direction != "stable"]),
                "open_issues": len(report["insights"])
            },
            "charts": {
                "quality_distribution": report["quality"].get("quality_distribution", {}),
                "knowledge_types": self._get_type_distribution(items),
                "daily_activity": report["usage"].get("search", {}).get("daily_search_volume", {})
            },
            "alerts": report["insights"][:5]  # Top 5 insights
        }
    
    def _get_type_distribution(self, items: List[KnowledgeItem]) -> Dict[str, int]:
        """Get distribution of knowledge types."""
        type_counts = Counter(item.knowledge_type.value for item in items)
        return dict(type_counts)


__all__ = [
    "KnowledgeAnalyticsEngine",
    "TrendAnalyzer",
    "KnowledgeQualityAnalyzer",
    "UsageAnalytics",
    "AnomalyDetector",
    "TrendAnalysis",
    "KnowledgeCluster"
]
