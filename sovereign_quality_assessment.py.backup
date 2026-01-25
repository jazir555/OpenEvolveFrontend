"""
Sovereign-Grade Problem Decomposition System - Quality Assessment
Implements decomposition-specific quality metrics and reporting.
"""

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass

from sovereign_data_models import (
    DecompositionPlan, SubProblem, QualityScores, generate_id
)

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Detailed quality metrics for decomposition."""
    coherence_score: float
    completeness_score: float
    feasibility_score: float
    integration_score: float
    balance_score: float
    clarity_score: float
    overall_score: float
    details: Dict[str, Any]


@dataclass
class QualityReport:
    """Comprehensive quality report."""
    plan_id: str
    metrics: QualityMetrics
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    meets_thresholds: bool
    generated_at: datetime


class QualityAssessor:
    """Assesses decomposition and solution quality with LLM-powered evaluation."""
    
    def __init__(self, openevolve_client=None):
        self.logger = logging.getLogger(__name__)
        self.openevolve_client = openevolve_client
        
        if not self.openevolve_client:
            try:
                from openevolve_client import OpenEvolveClient
                self.openevolve_client = OpenEvolveClient()
            except:
                self.logger.warning("OpenEvolve client not available for quality assessment")
        
        # Quality thresholds
        self.thresholds = {
            'coherence': 0.75,
            'completeness': 0.80,
            'feasibility': 0.70,
            'integration': 0.75,
            'balance': 0.70,
            'clarity': 0.75,
            'overall': 0.75
        }
    
    def assess_with_llm(self, plan: DecompositionPlan) -> QualityMetrics:
        """
        Comprehensive LLM-based quality assessment.
        
        Uses LLM to evaluate all quality dimensions with detailed reasoning.
        """
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for quality assessment.")
        
        # Build plan summary
        sp_summary = "\n".join([
            f"{i+1}. {sp.title} ({sp.type.value})\n   {sp.description[:100]}..."
            for i, sp in enumerate(plan.sub_problems[:8])
        ])
        
        prompt = f"""You are an expert quality assessor for problem decomposition plans. Evaluate this decomposition across multiple quality dimensions.

DECOMPOSITION PLAN:
Strategy: {plan.strategy.value}
Sub-problems: {len(plan.sub_problems)}

{sp_summary}

QUALITY ASSESSMENT:
Rate each dimension 0-100:

1. COHERENCE: Logical consistency, alignment, no contradictions
2. COMPLETENESS: Full coverage, no gaps, all aspects addressed
3. FEASIBILITY: Realistic, achievable, properly scoped
4. INTEGRATION: Sub-problems integrate well, clear interfaces
5. BALANCE: Even distribution of complexity and effort
6. CLARITY: Clear, understandable, well-defined

Provide assessment in EXACT format:
Coherence: <score>
Completeness: <score>
Feasibility: <score>
Integration: <score>
Balance: <score>
Clarity: <score>
Strengths: <strength1> | <strength2> | <strength3>
Weaknesses: <weakness1> | <weakness2>
Recommendations: <rec1> | <rec2> | <rec3>

Be thorough and specific."""
        
        result = self.openevolve_client.evolve(
            content=prompt,
            evolution_mode="standard",
            content_type="analysis",
            max_iterations=1,
            temperature=0.3,
            max_tokens=800
        )
        
        if not result.success or not result.best_code:
            raise RuntimeError("LLM evolution failed to produce a result for quality assessment.")
            
        return self._parse_quality_response(result.best_code)
    
    def _parse_quality_response(self, response: str) -> QualityMetrics:
        """Parse LLM quality assessment response."""
        lines = response.strip().split('\n')
        
        scores = {
            'coherence': 75.0,
            'completeness': 75.0,
            'feasibility': 75.0,
            'integration': 75.0,
            'balance': 75.0,
            'clarity': 75.0
        }
        
        strengths = []
        weaknesses = []
        recommendations = []
        
        for line in lines:
            line = line.strip()
            if ':' not in line:
                continue
            
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            
            if key in scores:
                try:
                    scores[key] = float(value)
                except ValueError as e:
                    self.logger.debug("Failed to parse quality score '%s': %s", value, e)
            elif key == 'strengths':
                strengths = [s.strip() for s in value.split('|') if s.strip()]
            elif key == 'weaknesses':
                weaknesses = [w.strip() for w in value.split('|') if w.strip()]
            elif key == 'recommendations':
                recommendations = [r.strip() for r in value.split('|') if r.strip()]
        
        overall = sum(scores.values()) / len(scores)
        
        return QualityMetrics(
            coherence_score=scores['coherence'] / 100.0,
            completeness_score=scores['completeness'] / 100.0,
            feasibility_score=scores['feasibility'] / 100.0,
            integration_score=scores['integration'] / 100.0,
            balance_score=scores['balance'] / 100.0,
            clarity_score=scores['clarity'] / 100.0,
            overall_score=overall / 100.0,
            details={
                'strengths': strengths,
                'weaknesses': weaknesses,
                'recommendations': recommendations,
                'method': 'llm'
            }
        )
    

    
    def generate_quality_report(self, plan: DecompositionPlan) -> QualityReport:
        """
        Generates comprehensive quality report using LLM-based analysis.
        
        Args:
            plan: The decomposition plan to assess
            
        Returns:
            QualityReport with detailed metrics and recommendations
            
        Raises:
            RuntimeError: If LLM analysis fails or is unavailable.
        """
        self.logger.info(f"Generating quality report for plan {plan.id}")
        
        if not self.openevolve_client:
            raise RuntimeError("OpenEvolve client not available for quality assessment.")

        try:
            llm_metrics = self.assess_with_llm(plan)
            if not llm_metrics:
                raise ValueError("LLM quality assessment returned no result.")

            self.logger.info("Using LLM-based quality assessment")
            
            meets_thresholds = self.check_quality_thresholds(llm_metrics)
            
            return QualityReport(
                plan_id=plan.id,
                metrics=llm_metrics,
                strengths=llm_metrics.details.get('strengths', []),
                weaknesses=llm_metrics.details.get('weaknesses', []),
                recommendations=llm_metrics.details.get('recommendations', []),
                meets_thresholds=meets_thresholds,
                generated_at=datetime.now()
            )
        except Exception as e:
            self.logger.error(f"LLM quality assessment failed: {e}")
            raise RuntimeError(f"Failed to generate quality report using LLM: {e}") from e
    
    def check_quality_thresholds(self, metrics: QualityMetrics) -> bool:
        """
        Validates scores meet minimum thresholds.
        
        Args:
            metrics: QualityMetrics to check
            
        Returns:
            True if all thresholds are met
        """
        return all([
            metrics.coherence_score >= self.thresholds['coherence'],
            metrics.completeness_score >= self.thresholds['completeness'],
            metrics.feasibility_score >= self.thresholds['feasibility'],
            metrics.integration_score >= self.thresholds['integration'],
            metrics.balance_score >= self.thresholds['balance'],
            metrics.clarity_score >= self.thresholds['clarity'],
            metrics.overall_score >= self.thresholds['overall']
        ])
    
    def update_plan_quality_scores(self, plan: DecompositionPlan) -> QualityScores:
        """
        Updates plan with quality scores.
        
        Args:
            plan: The decomposition plan to update
            
        Returns:
            QualityScores object
        """
        report = self.generate_quality_report(plan)
        
        quality_scores = QualityScores(
            coherence_score=report.metrics.coherence_score,
            completeness_score=report.metrics.completeness_score,
            feasibility_score=report.metrics.feasibility_score,
            integration_score=report.metrics.integration_score,
            overall_score=report.metrics.overall_score,
            meets_thresholds=report.meets_thresholds,
            details={
                'balance_score': report.metrics.balance_score,
                'clarity_score': report.metrics.clarity_score,
                'strengths': report.strengths,
                'weaknesses': report.weaknesses,
                'recommendations': report.recommendations
            },
            timestamp=report.generated_at
        )
        
        # Update plan
        plan.quality_scores = quality_scores
        
        return quality_scores
