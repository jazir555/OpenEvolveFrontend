"""
Analytics Z3 Connector

Feeds Z3 solving metrics into the analytics system for performance tracking
and pattern analysis over time.

Integrates with:
- analytics_dashboard.py
- analytics_manager.py
- z3_performance_monitor.py
- CAV-NLP for enhanced mathematical content analysis

Author: OpenEvolve
Created: 2026-02-02
"""

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)

try:
    from z3prover_integration import Z3SolverEngine, Z3Config
    Z3_AVAILABLE = True
except ImportError:
    Z3_AVAILABLE = False

try:
    from z3_performance_monitor import get_z3_performance_monitor
    MONITOR_AVAILABLE = True
except ImportError:
    MONITOR_AVAILABLE = False

# CAV-NLP Integration
try:
    from openevolve.cav_nlp_integration.adapter import Z3LeanAideBridge, create_z3_lean_bridge
    from openevolve.cav_nlp_integration.data_structures import (
        ConstraintType,
        Z3Constraint,
        Lean4Constraint,
        VerificationBridgeResult,
        CanonicalizationResult,
    )
    CAV_NLP_AVAILABLE = True
except ImportError:
    CAV_NLP_AVAILABLE = False
    logger.warning("CAV-NLP integration not available for analytics")


@dataclass
class Z3AnalyticsEvent:
    """A Z3 solving event for analytics."""
    event_id: str
    timestamp: datetime
    operation_type: str  # "solve", "optimize", "prove", "verify"
    problem_category: str
    execution_time_ms: float
    result_status: str
    constraint_count: int
    variable_count: int
    memory_usage_mb: float
    solver_version: str = "unknown"


@dataclass
class Z3MetricsAggregation:
    """Aggregated Z3 metrics."""
    period_start: datetime
    period_end: datetime
    total_operations: int
    avg_execution_time_ms: float
    success_rate: float
    sat_rate: float
    unsat_rate: float
    timeout_rate: float
    error_rate: float
    top_problem_categories: List[Dict[str, Any]]
    performance_trends: Dict[str, float]


@dataclass
class AnalyticsResult:
    """Result of analyzing natural language query using CAV-NLP.
    
    Attributes:
        success: Whether analysis succeeded
        natural_language: Original natural language query
        formalized_query: Formal representation of the query
        mathematical_structure: Extracted mathematical structure
        constraint_type: Type of constraint detected
        variables: List of variables found
        complexity_score: Estimated complexity (0-1)
        canonical_form: Canonical representation
        confidence: Confidence in analysis (0-1)
        metadata: Additional analysis metadata
    """
    success: bool
    natural_language: str
    formalized_query: Optional[str] = None
    mathematical_structure: Optional[Dict[str, Any]] = None
    constraint_type: Optional[str] = None
    variables: List[str] = field(default_factory=list)
    complexity_score: float = 0.0
    canonical_form: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnalyticsZ3Connector:
    """
    Connects Z3 solving metrics to the analytics system.
    
    Tracks:
    - Solver performance over time
    - Problem type distribution
    - Success/failure patterns
    - Resource utilization
    - Constraint solving trends
    
    CAV-NLP Integration:
    - Analyzes natural language queries
    - Canonicalizes constraints for consistent comparison
    - Extracts mathematical structure from queries
    """
    
    def __init__(self):
        self.event_buffer: List[Z3AnalyticsEvent] = []
        self.aggregations: Dict[str, Z3MetricsAggregation] = {}
        self.daily_stats = defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0.0,
            "successes": 0,
            "failures": 0,
            "sat_count": 0,
            "unsat_count": 0
        })
        
        # Initialize CAV-NLP bridge for enhanced analytics
        self.cav_nlp_bridge = None
        self._cav_nlp_available = False
        if CAV_NLP_AVAILABLE:
            try:
                self.cav_nlp_bridge = create_z3_lean_bridge()
                self._cav_nlp_available = True
                logger.info("CAV-NLP bridge initialized for analytics")
            except Exception as e:
                logger.warning(f"Failed to initialize CAV-NLP bridge: {e}")
    
    def record_solving_event(
        self,
        operation_type: str,
        result_status: str,
        execution_time_ms: float,
        constraint_count: int = 0,
        variable_count: int = 0,
        problem_category: str = "unknown",
        memory_usage_mb: float = 0.0
    ) -> None:
        """Record a Z3 solving event."""
        event = Z3AnalyticsEvent(
            event_id=f"z3_{int(time.time() * 1000)}",
            timestamp=datetime.utcnow(),
            operation_type=operation_type,
            problem_category=problem_category,
            execution_time_ms=execution_time_ms,
            result_status=result_status,
            constraint_count=constraint_count,
            variable_count=variable_count,
            memory_usage_mb=memory_usage_mb
        )
        
        self.event_buffer.append(event)
        self._update_daily_stats(event)
    
    def _update_daily_stats(self, event: Z3AnalyticsEvent) -> None:
        """Update daily statistics."""
        date_key = event.timestamp.strftime("%Y-%m-%d")
        stats = self.daily_stats[date_key]
        
        stats["count"] += 1
        stats["total_time_ms"] += event.execution_time_ms
        
        if event.result_status in ["sat", "proven", "verified"]:
            stats["successes"] += 1
        else:
            stats["failures"] += 1
        
        if event.result_status == "sat":
            stats["sat_count"] += 1
        elif event.result_status == "unsat":
            stats["unsat_count"] += 1
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict[str, Any]:
        """Get daily Z3 solving report."""
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")
        
        stats = self.daily_stats.get(date, {
            "count": 0,
            "total_time_ms": 0.0,
            "successes": 0,
            "failures": 0,
            "sat_count": 0,
            "unsat_count": 0
        })
        
        count = stats["count"]
        if count == 0:
            return {
                "date": date,
                "operations": 0,
                "avg_execution_time_ms": 0.0,
                "success_rate": 0.0
            }
        
        return {
            "date": date,
            "operations": count,
            "avg_execution_time_ms": stats["total_time_ms"] / count,
            "success_rate": stats["successes"] / count,
            "sat_rate": stats["sat_count"] / count,
            "unsat_rate": stats["unsat_count"] / count
        }
    
    def get_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get Z3 performance trends over time."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        daily_reports = []
        current = start_date
        
        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            report = self.get_daily_report(date_str)
            daily_reports.append(report)
            current += timedelta(days=1)
        
        # Calculate trends
        if len(daily_reports) >= 2:
            first_avg = daily_reports[0].get("avg_execution_time_ms", 0)
            last_avg = daily_reports[-1].get("avg_execution_time_ms", 0)
            
            first_success = daily_reports[0].get("success_rate", 0)
            last_success = daily_reports[-1].get("success_rate", 0)
            
            return {
                "period_days": days,
                "daily_reports": daily_reports,
                "execution_time_trend": last_avg - first_avg,
                "success_rate_trend": last_success - first_success,
                "trend_direction": "improving" if last_avg < first_avg else "degrading"
            }
        
        return {
            "period_days": days,
            "daily_reports": daily_reports,
            "trend_direction": "insufficient_data"
        }
    
    def export_to_analytics_dashboard(self) -> Dict[str, Any]:
        """Export metrics for analytics dashboard."""
        today_report = self.get_daily_report()
        trends = self.get_performance_trends(days=7)
        
        return {
            "z3_metrics": {
                "today": today_report,
                "trends": trends,
                "summary": {
                    "total_operations_today": today_report["operations"],
                    "avg_response_time_ms": today_report["avg_execution_time_ms"],
                    "success_rate": today_report["success_rate"],
                    "trend": trends.get("trend_direction", "unknown")
                }
            }
        }
    
    # ========================================================================
    # CAV-NLP Integration Methods
    # ========================================================================
    
    def analyze_nl_query(self, natural_language: str) -> AnalyticsResult:
        """Analyze natural language query using CAV-NLP.
        
        This method:
        1. Formalizes the natural language query
        2. Extracts mathematical structure
        3. Determines constraint type
        4. Generates canonical form
        
        Args:
            natural_language: Natural language query to analyze
            
        Returns:
            AnalyticsResult with analysis results
        """
        if not self._cav_nlp_available or self.cav_nlp_bridge is None:
            logger.warning("CAV-NLP not available for NL query analysis")
            return AnalyticsResult(
                success=False,
                natural_language=natural_language,
                metadata={"error": "CAV-NLP not available"}
            )
        
        try:
            # Use CAV-NLP parser to formalize query
            parser = getattr(self.cav_nlp_bridge, 'parser', None)
            canonicalizer = getattr(self.cav_nlp_bridge, 'canonicalizer', None)
            
            formalized_query = None
            canonical_form = None
            mathematical_structure = {}
            variables = []
            constraint_type = "unknown"
            confidence = 0.5
            
            # Step 1: Parse and formalize using CAV-NLP
            if parser is not None:
                try:
                    if hasattr(parser, 'canonicalize'):
                        parse_result = parser.canonicalize(natural_language)
                        formalized_query = str(parse_result)
                        confidence = 0.7
                    elif hasattr(parser, 'normalize'):
                        parse_result = parser.normalize(natural_language)
                        formalized_query = str(parse_result)
                        confidence = 0.6
                    
                    # Extract structure from parse result
                    mathematical_structure = self._extract_structure_from_parse(
                        natural_language, parse_result
                    )
                except Exception as e:
                    logger.debug(f"CAV-NLP parsing failed: {e}")
            
            # Step 2: Determine constraint type
            constraint_type = self._determine_constraint_type(natural_language)
            
            # Step 3: Canonicalize for comparison
            if canonicalizer is not None and formalized_query:
                try:
                    canonical_form = self.canonicalize_for_comparison(formalized_query)
                    confidence = min(confidence + 0.2, 1.0)
                except Exception as e:
                    logger.debug(f"CAV-NLP canonicalization failed: {e}")
            
            # Step 4: Extract variables
            variables = self._extract_variables_from_nl(natural_language)
            
            # Step 5: Calculate complexity score
            complexity_score = self._calculate_complexity(
                natural_language, variables, formalized_query
            )
            
            return AnalyticsResult(
                success=True,
                natural_language=natural_language,
                formalized_query=formalized_query or natural_language,
                mathematical_structure=mathematical_structure,
                constraint_type=constraint_type,
                variables=variables,
                complexity_score=complexity_score,
                canonical_form=canonical_form,
                confidence=confidence,
                metadata={
                    "cav_nlp_available": self._cav_nlp_available,
                    "parser_used": parser is not None,
                    "canonicalizer_used": canonicalizer is not None
                }
            )
            
        except Exception as e:
            logger.error(f"Error analyzing NL query: {e}")
            return AnalyticsResult(
                success=False,
                natural_language=natural_language,
                metadata={"error": str(e)}
            )
    
    def canonicalize_for_comparison(self, constraint) -> str:
        """Canonicalize constraint for consistent comparison.
        
        Uses CAV-NLP canonicalization to create a standard form
        that enables semantic matching between similar constraints.
        
        Args:
            constraint: Constraint to canonicalize (string or Z3 expression)
            
        Returns:
            Canonical form string
        """
        if not self._cav_nlp_available or self.cav_nlp_bridge is None:
            # Fallback: return string representation
            return str(constraint)
        
        try:
            canonicalizer = getattr(self.cav_nlp_bridge, 'canonicalizer', None)
            parser = getattr(self.cav_nlp_bridge, 'parser', None)
            
            # Try CAV-NLP canonicalization first
            if canonicalizer is not None:
                try:
                    if hasattr(canonicalizer, 'canonicalize_text'):
                        result = canonicalizer.canonicalize_text(str(constraint))
                        if hasattr(result, 'canonical'):
                            return result.canonical
                        return str(result)
                    elif hasattr(canonicalizer, 'canonicalize'):
                        result = canonicalizer.canonicalize(str(constraint))
                        if hasattr(result, 'canonical'):
                            return result.canonical
                        return str(result)
                except Exception as e:
                    logger.debug(f"Canonicalizer failed: {e}")
            
            # Try parser canonicalization as fallback
            if parser is not None:
                try:
                    if hasattr(parser, 'canonicalize'):
                        result = parser.canonicalize(str(constraint))
                        return str(result)
                except Exception as e:
                    logger.debug(f"Parser canonicalization failed: {e}")
            
            # Final fallback: basic normalization
            return self._basic_canonicalize(str(constraint))
            
        except Exception as e:
            logger.warning(f"CAV-NLP canonicalization error: {e}")
            return str(constraint)
    
    def _extract_structure_from_parse(
        self,
        natural_language: str,
        parse_result: Any
    ) -> Dict[str, Any]:
        """Extract mathematical structure from parse result."""
        structure = {
            "original_text": natural_language,
            "operators": [],
            "predicates": [],
            "quantifiers": []
        }
        
        text_lower = natural_language.lower()
        
        # Detect quantifiers
        quantifier_keywords = {
            "for all": "forall", "forall": "forall", "∀": "forall",
            "there exists": "exists", "exists": "exists", "∃": "exists",
            "for every": "forall", "for each": "forall"
        }
        for kw, qtype in quantifier_keywords.items():
            if kw in text_lower:
                structure["quantifiers"].append(qtype)
        
        # Detect arithmetic operators
        operator_keywords = ["plus", "minus", "times", "divided by", "+", "-", "*", "/"]
        for op in operator_keywords:
            if op in natural_language:
                structure["operators"].append(op)
        
        # Detect comparison predicates
        predicate_keywords = ["equals", "equal to", "greater than", "less than",
                             ">=", "<=", ">", "<", "=", "≠"]
        for pred in predicate_keywords:
            if pred in natural_language:
                structure["predicates"].append(pred)
        
        # Add parse result metadata if available
        if parse_result is not None:
            if hasattr(parse_result, 'dag'):
                structure["has_dependency_graph"] = True
            if hasattr(parse_result, 'variables'):
                structure["parsed_variables"] = parse_result.variables
        
        return structure
    
    def _determine_constraint_type(self, text: str) -> str:
        """Determine constraint type from text."""
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['forall', 'exists', '∀', '∃', 'for all', 'there exists']):
            return "quantified"
        elif any(kw in text_lower for kw in ['array', 'list', 'sequence']):
            return "array"
        elif any(kw in text_lower for kw in ['bit', 'binary', 'bitwise']):
            return "bitvector"
        elif any(kw in text_lower for kw in ['square', 'power', 'exponential', 'log', 'multiply', 'times']) and \
             any(kw in text_lower for kw in ['x', 'y', 'variable']):
            return "nonlinear"
        elif any(kw in text_lower for kw in ['plus', 'minus', 'sum', 'difference', '+', '-']):
            return "arithmetic"
        else:
            return "boolean"
    
    def _extract_variables_from_nl(self, text: str) -> List[str]:
        """Extract variable names from natural language text."""
        import re
        
        # Common mathematical variable patterns
        # Single letters often used as variables
        var_pattern = r'\b([a-zA-Z])\b'
        matches = re.findall(var_pattern, text)
        
        # Filter out common words
        common_words = {'a', 'i', 'x', 'y', 'z', 'n', 'm', 'k', 'j', 't', 's'}
        variables = [v for v in matches if v.lower() in common_words]
        
        # Also look for explicit variable declarations
        explicit_pattern = r'(?:variable|let|where)\s+(\w+)'
        explicit_matches = re.findall(explicit_pattern, text.lower())
        
        return list(set(variables + explicit_matches))
    
    def _calculate_complexity(
        self,
        natural_language: str,
        variables: List[str],
        formalized_query: Optional[str]
    ) -> float:
        """Calculate complexity score (0-1)."""
        score = 0.0
        
        # Length factor
        score += min(len(natural_language) / 200, 0.2)
        
        # Variable count factor
        score += min(len(variables) * 0.1, 0.3)
        
        # Keyword complexity
        complex_keywords = ['forall', 'exists', 'implies', 'iff', 'recursion', 'induction']
        for kw in complex_keywords:
            if kw in natural_language.lower():
                score += 0.1
        
        # Formalization success
        if formalized_query:
            score += 0.2
        
        return min(score, 1.0)
    
    def _basic_canonicalize(self, text: str) -> str:
        """Basic canonicalization without CAV-NLP."""
        # Normalize whitespace
        text = ' '.join(text.split())
        
        # Normalize common operators
        replacements = {
            '&&': 'and', '||': 'or', '!': 'not',
            '==': '=', '!=': '≠',
            '>=': '≥', '<=': '≤'
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Convert to lowercase for consistency
        return text.lower().strip()


def get_analytics_z3_connector():
    """Get global analytics Z3 connector."""
    return AnalyticsZ3Connector()


if __name__ == "__main__":
    print("Analytics Z3 Connector initialized")
    
    # Demo CAV-NLP integration if available
    connector = get_analytics_z3_connector()
    
    if connector._cav_nlp_available:
        print("\nCAV-NLP integration available!")
        
        # Test NL query analysis
        test_query = "For all x, if x is greater than 0 then x plus 1 is greater than 1"
        result = connector.analyze_nl_query(test_query)
        
        print(f"\nTest Query: {test_query}")
        print(f"Success: {result.success}")
        print(f"Constraint Type: {result.constraint_type}")
        print(f"Variables: {result.variables}")
        print(f"Complexity: {result.complexity_score:.2f}")
        print(f"Confidence: {result.confidence:.2f}")
    else:
        print("\nCAV-NLP integration not available (graceful degradation active)")
