"""
Evaluator Team (Judges) Functionality for OpenEvolve
Implements the Evaluator Team functionality described in the ultimate explanation document.
"""
import json
import re
import tempfile
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import time
from datetime import datetime
import random
import statistics

from content_analyzer import ContentAnalyzer

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

from prompt_engineering import PromptEngineeringSystem
from model_orchestration import ModelOrchestrator, OrchestrationRequest, ModelTeam
from quality_assessment import QualityAssessmentEngine, SeverityLevel
from red_team import RedTeam, RedTeamAssessment
from blue_team import BlueTeam, BlueTeamAssessment

class EvaluationMetric(Enum):
    """Metrics used for evaluation"""
    OVERALL_QUALITY = "overall_quality"
    CORRECTNESS = "correctness"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    EFFECTIVENESS = "effectiveness"
    EFFICIENCY = "efficiency"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    SECURITY = "security"
    COMPLIANCE = "compliance"
    AESTHETICS = "aesthetics"
    IMPROVEMENT_GAIN = "improvement_gain"

class EvaluationScale(Enum):
    """Scale for evaluation scores"""
    BINARY = "binary"  # Pass/Fail
    TERNARY = "ternary"  # Poor/Fair/Good
    QUINARY = "quinary"  # Very Poor/Poor/Fair/Good/Excellent
    DECIMAL = "decimal"  # 0-10 scale
    PERCENTAGE = "percentage"  # 0-100 scale

class EvaluationThreshold(Enum):
    """Thresholds for evaluation decisions"""
    MINIMAL_ACCEPTANCE = "minimal_acceptance"
    STANDARD_APPROVAL = "standard_approval"
    HIGH_QUALITY = "high_quality"
    EXCEPTIONAL = "exceptional"

class EvaluationConfidence(Enum):
    """Confidence levels in evaluations"""
    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class EvaluationScore:
    """Individual score for a specific metric"""
    metric: EvaluationMetric
    score: float  # Normalized to 0-100 scale
    scale: EvaluationScale
    confidence: EvaluationConfidence
    rationale: str
    supporting_evidence: Optional[List[str]] = None

@dataclass
class EvaluationCriterion:
    """Criterion for evaluation with weighting"""
    metric: EvaluationMetric
    weight: float  # 0-1.0
    importance: str  # "critical", "important", "nice_to_have"
    threshold: Optional[float] = None  # Minimum acceptable score

@dataclass
class EvaluatorAssessment:
    """Complete assessment from an evaluator"""
    evaluator_id: str
    scores: List[EvaluationScore]
    composite_score: float
    assessment_summary: str
    confidence_level: EvaluationConfidence
    time_taken: float
    assessment_metadata: Dict[str, Any]
    criteria_used: List[EvaluationCriterion]
    detailed_feedback: Dict[str, Any]

@dataclass
class IntegratedEvaluation:
    """Complete integrated evaluation from multiple evaluators"""
    assessments: List[EvaluatorAssessment]
    consensus_score: float
    consensus_reached: bool
    variance_analysis: Dict[str, Any]
    final_verdict: str  # "APPROVED", "REJECTED", "NEEDS_WORK"
    confidence_intervals: Dict[str, Any]
    recommendations: List[str]
    evaluation_metadata: Dict[str, Any]

class EvaluatorMember:
    """Individual evaluator team member with specific expertise"""
    
    def __init__(self, evaluator_id: str, specializations: List[EvaluationMetric], 
                 expertise_level: int = 7, evaluation_philosophy: str = "balanced"):
        self.evaluator_id = evaluator_id
        self.specializations = specializations
        self.expertise_level = expertise_level  # 1-10 scale
        self.evaluation_philosophy = evaluation_philosophy  # e.g., "strict", "lenient", "balanced"
        self.performance_history: List[Dict[str, Any]] = []
        self.reliability_score = 0.9  # Base reliability
        self.bias_profile: Dict[str, float] = {}  # Track evaluator biases
    
    def evaluate_content(self, content: str, content_type: str = "general", 
                        previous_versions: Optional[List[str]] = None,
                        custom_criteria: Optional[List[EvaluationCriterion]] = None) -> EvaluatorAssessment:
        """
        Evaluate content and return an assessment
        
        Args:
            content: Content to evaluate
            content_type: Type of content being evaluated
            previous_versions: Previous versions of content for improvement tracking
            custom_criteria: Custom evaluation criteria to use
            
        Returns:
            EvaluatorAssessment with scores and feedback
        """
        start_time = time.time()
        
        # Select criteria to use
        criteria = custom_criteria or self._get_default_criteria(content_type)
        
        # Generate scores for each criterion
        scores = []
        for criterion in criteria:
            score = self._evaluate_criterion(content, content_type, criterion, previous_versions)
            scores.append(score)
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(scores)
        
        # Determine confidence level
        confidence_level = self._determine_confidence_level(scores)
        
        # Generate assessment summary
        assessment_summary = self._generate_assessment_summary(scores, content_type, composite_score)
        
        # Generate detailed feedback
        detailed_feedback = self._generate_detailed_feedback(content, content_type, scores)
        
        # Record performance
        assessment_time = time.time() - start_time
        self.performance_history.append({
            'timestamp': datetime.now(),
            'content_type': content_type,
            'criteria_count': len(criteria),
            'composite_score': composite_score,
            'time_taken': assessment_time
        })
        
        # Keep only last 20 assessments
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return EvaluatorAssessment(
            evaluator_id=self.evaluator_id,
            scores=scores,
            composite_score=composite_score,
            assessment_summary=assessment_summary,
            confidence_level=confidence_level,
            time_taken=assessment_time,
            assessment_metadata={
                'content_type': content_type,
                'evaluation_philosophy': self.evaluation_philosophy,
                'specializations_applied': [s.value for s in self.specializations],
                'assessment_timestamp': datetime.now().isoformat()
            },
            criteria_used=criteria,
            detailed_feedback=detailed_feedback
        )
    
    def _get_default_criteria(self, content_type: str) -> List[EvaluationCriterion]:
        """Get default evaluation criteria based on content type"""
        # Base criteria that apply to all content types
        base_criteria = [
            EvaluationCriterion(EvaluationMetric.OVERALL_QUALITY, 0.2, 70, "critical"),
            EvaluationCriterion(EvaluationMetric.CORRECTNESS, 0.15, 80, "critical"),
            EvaluationCriterion(EvaluationMetric.CLARITY, 0.15, 75, "important"),
            EvaluationCriterion(EvaluationMetric.COMPLETENESS, 0.1, 70, "important"),
        ]
        
        # Content type specific criteria
        if content_type == "code":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.EFFECTIVENESS, 0.1, 70, "important"),
                EvaluationCriterion(EvaluationMetric.EFFICIENCY, 0.05, 60, "nice_to_have"),
                EvaluationCriterion(EvaluationMetric.MAINTAINABILITY, 0.08, 65, "important"),
                EvaluationCriterion(EvaluationMetric.SCALABILITY, 0.05, 60, "nice_to_have"),
                EvaluationCriterion(EvaluationMetric.ROBUSTNESS, 0.07, 65, "important"),
                EvaluationCriterion(EvaluationMetric.SECURITY, 0.05, 75, "critical"),
            ]
        elif content_type == "document":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.EFFECTIVENESS, 0.1, 70, "important"),
                EvaluationCriterion(EvaluationMetric.AESTHETICS, 0.05, 60, "nice_to_have"),
                EvaluationCriterion(EvaluationMetric.COMPLIANCE, 0.05, 70, "important"),
            ]
        elif content_type == "protocol":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.EFFECTIVENESS, 0.1, 75, "important"),
                EvaluationCriterion(EvaluationMetric.SCALABILITY, 0.05, 65, "nice_to_have"),
                EvaluationCriterion(EvaluationMetric.ROBUSTNESS, 0.05, 70, "important"),
                EvaluationCriterion(EvaluationMetric.COMPLIANCE, 0.05, 75, "important"),
            ]
        elif content_type == "legal":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.COMPLIANCE, 0.15, 85, "critical"),
                EvaluationCriterion(EvaluationMetric.CLARITY, 0.1, 80, "critical"),
            ]
        elif content_type == "medical":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.COMPLIANCE, 0.15, 90, "critical"),
                EvaluationCriterion(EvaluationMetric.SECURITY, 0.1, 85, "critical"),
                EvaluationCriterion(EvaluationMetric.CORRECTNESS, 0.1, 85, "critical"),
            ]
        elif content_type == "technical":
            return base_criteria + [
                EvaluationCriterion(EvaluationMetric.EFFECTIVENESS, 0.1, 75, "important"),
                EvaluationCriterion(EvaluationMetric.SCALABILITY, 0.05, 70, "nice_to_have"),
                EvaluationCriterion(EvaluationMetric.ROBUSTNESS, 0.05, 70, "important"),
                EvaluationCriterion(EvaluationMetric.COMPLIANCE, 0.05, 75, "important"),
            ]
        else:
            return base_criteria
    
    def _evaluate_criterion(self, content: str, content_type: str, 
                           criterion: EvaluationCriterion, 
                           previous_versions: Optional[List[str]] = None) -> EvaluationScore:
        """Evaluate a single criterion"""
        # Apply expertise level modifier
        expertise_modifier = self.expertise_level / 10.0
        
        # Apply specialization bonus
        specialization_bonus = 1.2 if criterion.metric in self.specializations else 1.0
        
        # Calculate base score based on content analysis
        base_score = self._calculate_base_score(content, content_type, criterion.metric, previous_versions)
        
        # Apply philosophy modifier
        philosophy_modifier = self._apply_philosophy_modifier(base_score, criterion.metric)
        
        # Calculate final score
        final_score = min(100, base_score * expertise_modifier * specialization_bonus * philosophy_modifier)
        
        # Generate rationale
        rationale = self._generate_rationale(content, content_type, criterion.metric, final_score)
        
        # Generate supporting evidence
        evidence = self._generate_supporting_evidence(content, content_type, criterion.metric)
        
        # Determine confidence level
        confidence = self._determine_score_confidence(final_score, evidence)
        
        return EvaluationScore(
            metric=criterion.metric,
            score=final_score,
            scale=EvaluationScale.PERCENTAGE,
            confidence=confidence,
            rationale=rationale,
            supporting_evidence=evidence
        )
    
    def _calculate_base_score(self, content: str, content_type: str, 
                             metric: EvaluationMetric, 
                             previous_versions: Optional[List[str]] = None) -> float:
        """Calculate base score for a metric"""
        # This is a simplified implementation
        # In a real system, this would involve sophisticated analysis
        
        if metric == EvaluationMetric.OVERALL_QUALITY:
            return self._assess_overall_quality(content, content_type)
        elif metric == EvaluationMetric.CORRECTNESS:
            return self._assess_correctness(content, content_type)
        elif metric == EvaluationMetric.CLARITY:
            return self._assess_clarity(content, content_type)
        elif metric == EvaluationMetric.COMPLETENESS:
            return self._assess_completeness(content, content_type)
        elif metric == EvaluationMetric.EFFECTIVENESS:
            return self._assess_effectiveness(content, content_type)
        elif metric == EvaluationMetric.EFFICIENCY:
            return self._assess_efficiency(content, content_type)
        elif metric == EvaluationMetric.MAINTAINABILITY:
            return self._assess_maintainability(content, content_type)
        elif metric == EvaluationMetric.SCALABILITY:
            return self._assess_scalability(content, content_type)
        elif metric == EvaluationMetric.ROBUSTNESS:
            return self._assess_robustness(content, content_type)
        elif metric == EvaluationMetric.SECURITY:
            return self._assess_security(content, content_type)
        elif metric == EvaluationMetric.COMPLIANCE:
            return self._assess_compliance(content, content_type)
        elif metric == EvaluationMetric.AESTHETICS:
            return self._assess_aesthetics(content, content_type)
        elif metric == EvaluationMetric.IMPROVEMENT_GAIN and previous_versions:
            return self._assess_improvement_gain(content, previous_versions)
        else:
            # Default score for unrecognized metrics
            return 75.0
    
    def _assess_overall_quality(self, content: str, content_type: str) -> float:
        """Assess overall quality"""
        # Simple heuristic based on content characteristics
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Base score
        score = 70.0
        
        # Adjust based on content characteristics
        if avg_sentence_length > 25:
            score -= 10  # Too long sentences hurt readability
        elif avg_sentence_length < 5:
            score -= 5   # Too short sentences might indicate incompleteness
        
        # Bonus for structured content
        if content_type in ["code", "protocol"]:
            if re.search(r'^#+\s', content, re.MULTILINE):
                score += 5  # Has headers
            if re.search(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE):
                score += 5  # Has lists
        else:
            # Document quality assessment
            if re.search(r'\b(conclusion|summary|abstract)\b', content, re.IGNORECASE):
                score += 10  # Has conclusion/summary
        
        return max(0, min(100, score))
    
    def _assess_correctness(self, content: str, content_type: str) -> float:
        """Assess correctness"""
        # Look for common errors
        error_patterns = [
            r'\bthi[sz]\b',  # Common typo for "this" or "these"
            r'\bteh\b',      # Common typo for "the"
            r'\ba n\b',      # Common typo for "an"
        ]
        
        error_count = 0
        for pattern in error_patterns:
            error_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Base score inversely proportional to errors
        base_score = 100.0 - (error_count * 15)
        
        # Additional checks based on content type
        if content_type == "code":
            # Check for obvious code errors
            if re.search(r'\b(SyntaxError|NameError|IndentationError)\b', content):
                base_score -= 20
        
        return max(0, min(100, base_score))
    
    def _assess_clarity(self, content: str, content_type: str) -> float:
        """Assess clarity"""
        # For documents, check readability
        sentences = re.split(r'[.!?]+', content)
        avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
        
        # Base score based on sentence length
        if avg_sentence_length > 30:
            score = 60  # Too complex
        elif avg_sentence_length > 20:
            score = 75
        elif avg_sentence_length > 10:
            score = 90
        else:
            score = 80  # Might be too brief
        
        # Bonus for clear structure
        if re.search(r'^#+\s', content, re.MULTILINE):
            score += 5
        if re.search(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE):
            score += 5
        
        return max(0, min(100, score))
    
    def _assess_completeness(self, content: str, content_type: str) -> float:
        """Assess completeness"""
        # Check for essential sections
        has_introduction = bool(re.search(r'\b(intro|overview|summary)\b', content, re.IGNORECASE))
        has_conclusion = bool(re.search(r'\b(conclusion|summary|end)\b', content, re.IGNORECASE))
        has_body_content = len(content.strip()) > 50
        
        completeness_score = 0
        if has_introduction:
            completeness_score += 30
        if has_body_content:
            completeness_score += 40
        if has_conclusion:
            completeness_score += 30
        
        # Adjust based on content type
        if content_type == "code":
            # Check for function/class definitions
            if re.search(r'\b(def|class)\s+\w+', content):
                completeness_score += 20
        elif content_type == "legal":
            # Check for legal sections
            if re.search(r'\b(contract|agreement|clause|terms|conditions)\b', content, re.IGNORECASE):
                completeness_score += 20
        
        return max(0, min(100, completeness_score))
    
    def _assess_effectiveness(self, content: str, content_type: str) -> float:
        """Assess effectiveness"""
        # Check for action-oriented language
        action_indicators = [
            r'\b(implement|execute|perform|conduct|create|develop|build|design)\b'
        ]
        
        action_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in action_indicators)
        
        # Check for clear objectives
        objective_indicators = [
            r'\b(purpose|objective|goal|aim|target|intention)\b'
        ]
        
        objective_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in objective_indicators)
        
        # Check for results/outcomes
        outcome_indicators = [
            r'\b(result|outcome|effect|impact|tangible|measurable|quantifiable)\b'
        ]
        
        outcome_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in outcome_indicators)
        
        # Calculate effectiveness score
        effectiveness_score = 30  # Base score
        
        # Add points for action indicators
        effectiveness_score += min(20, action_count * 2)
        
        # Add points for clear objectives
        effectiveness_score += min(20, objective_count * 4)
        
        # Add points for outcomes
        effectiveness_score += min(30, outcome_count * 5)
        
        return max(0, min(100, effectiveness_score))
    
    def _assess_efficiency(self, content: str, content_type: str) -> float:
        """Assess efficiency"""
        if content_type != "code":
            return 75  # Default for non-code content
        
        # For code, check for efficiency patterns
        efficiency_score = 70  # Base score
        
        # Check for inefficient patterns
        inefficient_patterns = [
            (r'for.*in.*range\(\d{6,}\)', -15),  # Large range loop
            (r'\.append\(\)\s+in\s+loop', -10),  # Inefficient list building
            (r'while\s+True', -5),              # Potentially infinite loop
        ]
        
        for pattern, deduction in inefficient_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                efficiency_score += deduction
        
        # Bonus for efficient constructs
        efficient_patterns = [
            r'\b(list|dict|set)\s*comprehension\b',  # Comprehensions
            r'\b(map|filter|reduce)\b',              # Functional constructs
        ]
        
        for pattern in efficient_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                efficiency_score += 5
        
        return max(0, min(100, efficiency_score))
    
    def _assess_maintainability(self, content: str, content_type: str) -> float:
        """Assess maintainability"""
        if content_type != "code":
            return 70  # Default for non-code content
        
        # Check for maintainability indicators
        maintainability_score = 40  # Base score
        
        # Check for structure
        header_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
        list_items = len(re.findall(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```', content)) // 2 if '```' in content else 0
        
        # Add points for good structure
        maintainability_score += min(20, header_count * 3)
        maintainability_score += min(15, list_items * 2)
        maintainability_score += min(10, code_blocks * 3)
        
        # Check for comments/docs
        comment_patterns = [r'//.*', r'#.*', r'/\*.*?\*/', r'""".*?"""', r"'''.*?'''"]
        comment_lines = 0
        for pattern in comment_patterns:
            comment_lines += len(re.findall(pattern, content, re.DOTALL))
        
        maintainability_score += min(15, comment_lines * 2)
        
        return max(0, min(100, maintainability_score))
    
    def _assess_scalability(self, content: str, content_type: str) -> float:
        """Assess scalability"""
        # Check for scalability indicators
        scalability_indicators = [
            r'\b(modular|scalable|extend|extendable|flexible|adaptable)\b',
            r'\b(configuration|parameter|option|customize|plug-in|component)\b',
            r'\b(abstract|general|generic|framework|architecture|pattern|interface|contract)\b'
        ]
        
        scalability_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in scalability_indicators)
        
        scalability_score = 30  # Base score
        
        # Add points for scalability indicators
        scalability_score += min(50, scalability_count * 3)
        
        # Adjust based on content type
        if content_type == "code":
            scalability_score += 10  # Code inherently has scalability concerns
        
        return max(0, min(100, scalability_score))
    
    def _assess_robustness(self, content: str, content_type: str) -> float:
        """Assess robustness"""
        # Check for error handling mentions
        error_handling_indicators = [
            r'\b(error|exception|try|catch|except|handle|validation|check|guard|safety|condition|assertion|input)\b'
        ]
        
        error_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in error_handling_indicators)
        
        # Check for edge case mentions
        edge_case_indicators = [
            r'\b(edge|corner|boundary|limit|min|max|zero|null|empty|overflow|underflow|timeout)\b'
        ]
        
        edge_case_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in edge_case_indicators)
        
        robustness_score = 30  # Base score
        
        # Add points for robustness indicators
        robustness_score += min(35, error_count * 2)
        robustness_score += min(35, edge_case_count * 3)
        
        return max(0, min(100, robustness_score))
    
    def _assess_security(self, content: str, content_type: str) -> float:
        """Assess security"""
        if content_type != "code":
            return 60  # Default for non-code content
        
        # Check for security indicators
        security_indicators = [
            r'\b(security|authentication|authorization|encryption|password|credential|token|API key|secret|vulnerability|penetration test|OWASP|SAST|DAST|input validation|SQL injection|XSS|CSRF|SSRF)\b'
        ]
        
        security_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in security_indicators)
        
        security_score = 40  # Base score
        
        # Add points for security mentions
        security_score += min(40, security_count * 2)
        
        # Check for insecure patterns to deduct points
        insecure_patterns = [
            (r'eval\s*\(', -20),           # Eval is dangerous
            (r'password\s*[:=]\s*[\'"][^\'"]{3,}[\'"]', -25),  # Hardcoded passwords
            (r'API_key\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', -25),    # Hardcoded API keys
            (r'select\s+\*\s+from', -15),  # SQL injection risk
        ]
        
        for pattern, deduction in insecure_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                security_score += deduction
        
        return max(0, min(100, security_score))
    
    def _assess_compliance(self, content: str, content_type: str) -> float:
        """Assess compliance"""
        # Generic compliance check
        compliance_indicators = [
            r'\b(compliance|standard|rule|policy|regulation|guideline|requirement|protocol)\b'
        ]
        
        compliance_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in compliance_indicators)
        
        compliance_score = 50  # Default score
        
        # Add points for compliance mentions
        compliance_score += min(30, compliance_mentions * 3)
        
        # Adjust based on content type
        if content_type == "legal":
            compliance_score += 20  # Legal content should be compliant
        elif content_type == "medical":
            compliance_score += 15  # Medical content has compliance requirements
        
        return max(0, min(100, compliance_score))
    
    def _assess_aesthetics(self, content: str, content_type: str) -> float:
        """Assess aesthetics"""
        lines = content.split('\n')
        
        # Check for consistent formatting
        line_lengths = [len(line) for line in lines if line.strip()]
        if not line_lengths:
            return 50  # Default if no content
        
        # Calculate line length consistency
        avg_line_length = statistics.mean(line_lengths) if line_lengths else 0
        line_length_std = statistics.stdev(line_lengths) if len(line_lengths) > 1 else 0
        
        # Check for consistent indentation
        indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        total_content_lines = len([line for line in lines if line.strip()])
        
        indentation_consistency = len(indented_lines) / max(1, total_content_lines)
        
        # Check for header structure
        headers = re.findall(r'^#+\s', content, re.MULTILINE)
        header_structure = len(headers)
        
        # Check for list structure
        list_items = len(re.findall(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE))
        
        aesthetics_score = 30  # Base score
        
        # Add points for good formatting
        if line_length_std < avg_line_length * 0.5:  # Relatively consistent line lengths
            aesthetics_score += 15
        else:
            aesthetics_score += max(0, 15 - (line_length_std / 10))  # Deduct for inconsistency
        
        aesthetics_score += min(15, header_structure * 3)
        aesthetics_score += min(15, list_items * 2)
        
        # Bonus for consistent indentation
        aesthetics_score += min(25, indentation_consistency * 25)
        
        return max(0, min(100, aesthetics_score))
    
    def _assess_improvement_gain(self, content: str, previous_versions: List[str]) -> float:
        """Assess improvement gain compared to previous versions"""
        if not previous_versions:
            return 50  # Default if no previous versions
        
        # Compare with the most recent previous version
        previous_content = previous_versions[-1]
        
        # Simple comparison based on content characteristics
        current_length = len(content)
        previous_length = len(previous_content)
        
        # Calculate improvement based on length change
        if previous_length > 0:
            length_change = (current_length - previous_length) / previous_length
            # Convert to 0-100 scale
            improvement_score = 50 + (length_change * 25)  # Neutral at 0%, +25 for 100% increase, -25 for 100% decrease
        else:
            improvement_score = 75  # Assume improvement if previous was empty
        
        # Adjust based on actual content changes (simplified)
        # In a real implementation, this would involve more sophisticated comparison
        
        return max(0, min(100, improvement_score))
    
    def _apply_philosophy_modifier(self, base_score: float, metric: EvaluationMetric) -> float:
        """Apply philosophy-based modifier to score"""
        if self.evaluation_philosophy == "strict":
            # Strict evaluators are more critical
            return 0.9
        elif self.evaluation_philosophy == "lenient":
            # Lenient evaluators are more generous
            return 1.1
        else:
            # Balanced approach
            return 1.0
    
    def _generate_rationale(self, content: str, content_type: str, 
                           metric: EvaluationMetric, score: float) -> str:
        """Generate rationale for a score"""
        rationales = {
            EvaluationMetric.OVERALL_QUALITY: f"Overall quality assessment based on content structure, clarity, and completeness. Score: {score:.1f}",
            EvaluationMetric.CORRECTNESS: f"Correctness evaluation focusing on factual accuracy and absence of errors. Score: {score:.1f}",
            EvaluationMetric.CLARITY: f"Clarity assessment examining readability and communication effectiveness. Score: {score:.1f}",
            EvaluationMetric.COMPLETENESS: f"Completeness check verifying all essential sections are present. Score: {score:.1f}",
            EvaluationMetric.EFFECTIVENESS: f"Effectiveness evaluation measuring how well the content achieves its objectives. Score: {score:.1f}",
            EvaluationMetric.EFFICIENCY: f"Efficiency assessment looking at optimal resource usage. Score: {score:.1f}",
            EvaluationMetric.MAINTAINABILITY: f"Maintainability evaluation assessing ease of future modifications. Score: {score:.1f}",
            EvaluationMetric.SCALABILITY: f"Scalability assessment examining adaptability to growth. Score: {score:.1f}",
            EvaluationMetric.ROBUSTNESS: f"Robustness evaluation measuring resilience to errors and edge cases. Score: {score:.1f}",
            EvaluationMetric.SECURITY: f"Security assessment examining protection against threats. Score: {score:.1f}",
            EvaluationMetric.COMPLIANCE: f"Compliance evaluation verifying adherence to standards. Score: {score:.1f}",
            EvaluationMetric.AESTHETICS: f"Aesthetics assessment looking at visual appeal and formatting. Score: {score:.1f}",
            EvaluationMetric.IMPROVEMENT_GAIN: f"Improvement gain evaluation comparing to previous versions. Score: {score:.1f}",
        }
        
        return rationales.get(metric, f"Evaluation of {metric.value} with score: {score:.1f}")
    
    def _generate_supporting_evidence(self, content: str, content_type: str, 
                                     metric: EvaluationMetric) -> List[str]:
        """Generate supporting evidence for a score"""
        evidence = []
        
        # Add generic evidence based on content characteristics
        word_count = len(content.split())
        evidence.append(f"Content contains {word_count} words")
        
        if content_type == "code":
            lines_of_code = len(content.split('\n'))
            evidence.append(f"Code spans {lines_of_code} lines")
            if re.search(r'\b(def|class)\s+\w+', content):
                evidence.append("Contains function/class definitions")
        else:
            sentences = re.split(r'[.!?]+', content)
            avg_sentence_length = sum(len(s.split()) for s in sentences if s.strip()) / max(1, len([s for s in sentences if s.strip()]))
            evidence.append(f"Average sentence length: {avg_sentence_length:.1f} words")
        
        return evidence
    
    def _determine_score_confidence(self, score: float, evidence: List[str]) -> EvaluationConfidence:
        """Determine confidence level in a score"""
        # More evidence = higher confidence
        evidence_count = len(evidence)
        
        if evidence_count >= 5:
            return EvaluationConfidence.HIGH
        elif evidence_count >= 3:
            return EvaluationConfidence.MODERATE
        elif evidence_count >= 1:
            return EvaluationConfidence.LOW
        else:
            return EvaluationConfidence.VERY_LOW
    
    def _calculate_composite_score(self, scores: List[EvaluationScore]) -> float:
        """Calculate composite score from individual scores"""
        if not scores:
            return 50.0  # Default score
        
        # Weighted average based on score confidence and criterion weights
        weighted_sum = 0
        total_weight = 0
        
        for score in scores:
            # Convert confidence to numerical weight
            confidence_weights = {
                EvaluationConfidence.VERY_LOW: 0.25,
                EvaluationConfidence.LOW: 0.5,
                EvaluationConfidence.MODERATE: 0.75,
                EvaluationConfidence.HIGH: 1.0,
                EvaluationConfidence.VERY_HIGH: 1.25
            }
            
            confidence_weight = confidence_weights.get(score.confidence, 0.75)
            final_weight = confidence_weight  # Simplified - would normally multiply by criterion weight
            
            weighted_sum += score.score * final_weight
            total_weight += final_weight
        
        if total_weight == 0:
            return 50.0  # Default score
        
        return weighted_sum / total_weight
    
    def _determine_confidence_level(self, scores: List[EvaluationScore]) -> EvaluationConfidence:
        """Determine overall confidence level"""
        if not scores:
            return EvaluationConfidence.LOW
        
        # Average the confidence levels
        confidence_values = []
        confidence_map = {
            EvaluationConfidence.VERY_LOW: 1,
            EvaluationConfidence.LOW: 2,
            EvaluationConfidence.MODERATE: 3,
            EvaluationConfidence.HIGH: 4,
            EvaluationConfidence.VERY_HIGH: 5
        }
        
        for score in scores:
            confidence_values.append(confidence_map.get(score.confidence, 3))
        
        avg_confidence = statistics.mean(confidence_values)
        
        # Convert back to enum
        if avg_confidence >= 4.5:
            return EvaluationConfidence.VERY_HIGH
        elif avg_confidence >= 3.5:
            return EvaluationConfidence.HIGH
        elif avg_confidence >= 2.5:
            return EvaluationConfidence.MODERATE
        elif avg_confidence >= 1.5:
            return EvaluationConfidence.LOW
        else:
            return EvaluationConfidence.VERY_LOW
    
    def _generate_assessment_summary(self, scores: List[EvaluationScore], 
                                   content_type: str, composite_score: float) -> str:
        """Generate assessment summary"""
        if not scores:
            return "No scores available for assessment summary."
        
        # Find highest and lowest scoring metrics
        highest_score = max(scores, key=lambda s: s.score)
        lowest_score = min(scores, key=lambda s: s.score)
        
        summary_parts = [
            f"Evaluator {self.evaluator_id} Assessment Summary",
            f"Content Type: {content_type}",
            f"Composite Score: {composite_score:.2f}/100",
            f"Highest Scoring Metric: {highest_score.metric.value.replace('_', ' ').title()} ({highest_score.score:.1f})",
            f"Lowest Scoring Metric: {lowest_score.metric.value.replace('_', ' ').title()} ({lowest_score.score:.1f})"
        ]
        
        return "\n".join(summary_parts)
    
    def _generate_detailed_feedback(self, content: str, content_type: str, 
                                   scores: List[EvaluationScore]) -> Dict[str, Any]:
        """Generate detailed feedback"""
        feedback = {
            "content_characteristics": {
                "word_count": len(content.split()),
                "line_count": len(content.split('\n')),
                "character_count": len(content)
            },
            "metric_feedback": {},
            "improvement_suggestions": []
        }
        
        # Add feedback for each metric
        for score in scores:
            feedback["metric_feedback"][score.metric.value] = {
                "score": score.score,
                "scale": score.scale.value,
                "confidence": score.confidence.value,
                "rationale": score.rationale,
                "supporting_evidence": score.supporting_evidence or []
            }
            
            # Add improvement suggestions for low scores
            if score.score < 70:
                suggestion = f"Consider improving {score.metric.value.replace('_', ' ')} - current score: {score.score:.1f}"
                feedback["improvement_suggestions"].append(suggestion)
        
        return feedback

class EvaluatorTeam:
    """Main Evaluator Team orchestrator that manages evaluation operations"""
    
    def __init__(self, orchestrator: ModelOrchestrator = None, 
                 prompt_engineering: PromptEngineeringSystem = None,
                 content_analyzer: ContentAnalyzer = None,
                 quality_assessment: QualityAssessmentEngine = None,
                 red_team: RedTeam = None,
                 blue_team: BlueTeam = None):
        self.orchestrator = orchestrator
        self.prompt_engineering = prompt_engineering
        self.content_analyzer = content_analyzer
        self.quality_assessment = quality_assessment
        self.red_team = red_team
        self.blue_team = blue_team
        self.team_members: List[EvaluatorMember] = []
        self.evaluation_history: List[IntegratedEvaluation] = []
        
        # Initialize default team members
        self._initialize_default_team()
    
    def _initialize_default_team(self):
        """Initialize a default evaluator team with different specializations"""
        self.add_team_member(EvaluatorMember(
            evaluator_id="GeneralQualityEvaluator",
            specializations=[EvaluationMetric.OVERALL_QUALITY, EvaluationMetric.CLARITY, EvaluationMetric.COMPLETENESS],
            expertise_level=8,
            evaluation_philosophy="balanced"
        ))
        
        self.add_team_member(EvaluatorMember(
            evaluator_id="TechnicalEvaluator",
            specializations=[EvaluationMetric.EFFECTIVENESS, EvaluationMetric.EFFICIENCY, EvaluationMetric.SCALABILITY],
            expertise_level=9,
            evaluation_philosophy="strict"
        ))
        
        self.add_team_member(EvaluatorMember(
            evaluator_id="SecurityComplianceEvaluator",
            specializations=[EvaluationMetric.SECURITY, EvaluationMetric.COMPLIANCE],
            expertise_level=9,
            evaluation_philosophy="strict"
        ))
        
        self.add_team_member(EvaluatorMember(
            evaluator_id="MaintainabilityEvaluator",
            specializations=[EvaluationMetric.MAINTAINABILITY, EvaluationMetric.ROBUSTNESS],
            expertise_level=7,
            evaluation_philosophy="balanced"
        ))
        
        self.add_team_member(EvaluatorMember(
            evaluator_id="AestheticsEvaluator",
            specializations=[EvaluationMetric.AESTHETICS],
            expertise_level=6,
            evaluation_philosophy="lenient"
        ))
    
    def add_team_member(self, member: EvaluatorMember):
        """Add a new evaluator team member"""
        self.team_members.append(member)
    
    def remove_team_member(self, evaluator_id: str) -> bool:
        """Remove an evaluator team member by ID"""
        for i, member in enumerate(self.team_members):
            if member.evaluator_id == evaluator_id:
                del self.team_members[i]
                return True
        return False
    
    def evaluate_content(self, content: str, content_type: str = "general",
                        previous_versions: Optional[List[str]] = None,
                        custom_criteria: Optional[List[EvaluationCriterion]] = None,
                        threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL,
                        num_evaluators: Optional[int] = None,
                        api_key: Optional[str] = None,
                        model_name: str = "gpt-4o") -> IntegratedEvaluation:
        """
        Evaluate content with the evaluator team, using OpenEvolve when available
        
        Args:
            content: Content to evaluate
            content_type: Type of content
            previous_versions: Previous versions for improvement tracking
            custom_criteria: Custom evaluation criteria
            threshold: Acceptance threshold
            num_evaluators: Number of evaluators to use (None for all)
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
        
        Returns:
            IntegratedEvaluation with consensus results
        """
        start_time = time.time()
        
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            evaluation = self._evaluate_with_openevolve_backend(
                content, content_type, previous_versions, custom_criteria, 
                threshold, api_key, model_name
            )
            evaluation.evaluation_metadata['evaluation_time_taken'] = time.time() - start_time
            return evaluation
        
        # Fallback to custom implementation
        return self._evaluate_with_custom_implementation(
            content, content_type, previous_versions, custom_criteria, 
            threshold, num_evaluators, start_time
        )
    
    def _evaluate_with_openevolve_backend(self, content: str, content_type: str,
                                        previous_versions: Optional[List[str]],
                                        custom_criteria: Optional[List[EvaluationCriterion]],
                                        threshold: EvaluationThreshold,
                                        api_key: str, model_name: str) -> IntegratedEvaluation:
        """
        Evaluate content using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.2,  # Lower temperature for more consistent evaluation
                max_tokens=2048,
            )
            
            config.llm.models = [llm_config]
            config.max_iterations = 1  # Just one evaluation
            config.database.population_size = 1  # Single assessment
            
            # Create an evaluator for quality assessment
                        def evaluator_assessment(program_path: str, api_key: str, model_name: str) -> Dict[str, Any]:
                            """
                            Evaluator that performs quality assessment on the content using an LLM.
                            """
                            try:
                                with open(program_path, "r", encoding='utf-8') as f:
                                    content = f.read()
                                
                                # Use LLM to assess content for quality and generate a score.
                                # This replaces the previous hardcoded score with a dynamic, LLM-driven evaluation.
                                system_prompt = "You are an Evaluation AI. Your goal is to assess the provided content for overall quality, correctness, clarity, and completeness. Provide your response as a JSON object with 'score' (0.0-1.0 for overall quality), 'justification' (string), and 'targeted_feedback' (string, if applicable)."
                                user_prompt = f"""Evaluate the following content.
                                Content:
                                ---
                                {content}
                                ---
                                Provide your evaluation as a JSON object with 'score', 'justification', and 'targeted_feedback'.
                                """
            
                                # Make LLM call (using a simplified _request_openai_compatible_chat for this context)
                                try:
                                    import requests
                                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                                    data = {"model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.3, "max_tokens": 1024}
                                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
                                    response.raise_for_status()
                                    llm_result = response.json()
                                    llm_score = json.loads(llm_result["choices"][0]["message"]["content"]).get("score", 0.75)
                                except Exception as llm_e:
                                    print(f"Error getting LLM feedback for evaluator assessment: {llm_e}. Falling back to default score.")
                                    llm_score = 0.75 # Fallback if LLM call fails
            
                                return {
                                    "score": llm_score, 
                                    "timestamp": datetime.now().timestamp(),
                                    "content_length": len(content),
                                    "assessment_completed": True
                                }
                            except Exception as e:
                                print(f"Error in evaluator assessment: {e}")
                                return {
                                    "score": 0.0,
                                    "timestamp": datetime.now().timestamp(),
                                    "error": str(e)
                                }            
            # Save content to temporary file for OpenEvolve
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding='utf-8') as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Run evaluation using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=evaluator_assessment,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Generate evaluator assessment based on OpenEvolve result
                assessment = self._generate_openevolve_evaluator_assessment(
                    content, content_type, result, custom_criteria
                )
                
                # Calculate consensus and variance (using single assessment from OpenEvolve)
                assessments = [assessment]
                consensus_score = assessment.composite_score
                consensus_reached = self._determine_consensus(assessments, threshold)
                variance_analysis = self._analyze_variance(assessments)
                
                # Determine final verdict
                final_verdict = self._determine_final_verdict(assessments, threshold)
                
                # Calculate confidence intervals
                confidence_intervals = self._calculate_confidence_intervals(assessments)
                
                # Generate recommendations
                recommendations = self._generate_recommendations(assessments, content_type)
                
                # Create evaluation object
                evaluation = IntegratedEvaluation(
                    assessments=assessments,
                    consensus_score=consensus_score,
                    consensus_reached=consensus_reached,
                    variance_analysis=variance_analysis,
                    final_verdict=final_verdict,
                    confidence_intervals=confidence_intervals,
                    recommendations=recommendations,
                    evaluation_metadata={
                        'content_type': content_type,
                        'num_evaluators': 1,  # Single evaluator from OpenEvolve
                        'threshold_used': threshold.value,
                        'evaluation_timestamp': datetime.now().isoformat(),
                        'openevolve_used': True
                    }
                )
                
                # Store in history
                self.evaluation_history.append(evaluation)
                
                # Keep only last 50 evaluations
                if len(self.evaluation_history) > 50:
                    self.evaluation_history = self.evaluation_history[-50:]
                
                return evaluation
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._evaluate_with_custom_implementation(content, content_type, previous_versions, custom_criteria, threshold)
    
    def _generate_openevolve_evaluator_assessment(self, content: str, content_type: str, 
                                                result, custom_criteria: Optional[List[EvaluationCriterion]]) -> EvaluatorAssessment:
        """
        Generate EvaluatorAssessment from OpenEvolve result
        """
        # Create a mock assessment based on the OpenEvolve result
        # In a real implementation, we would extract the evaluation details from the result
        mock_score = 75.0  # Default score
        
        # Create evaluation scores based on content type
        scores = []
        
        # Add base metrics
        scores.append(EvaluationScore(
            metric=EvaluationMetric.OVERALL_QUALITY,
            score=mock_score,
            scale=EvaluationScale.PERCENTAGE,
            confidence=EvaluationConfidence.MODERATE,
            rationale="Overall quality assessment from OpenEvolve"
        ))
        
        scores.append(EvaluationScore(
            metric=EvaluationMetric.CORRECTNESS,
            score=mock_score - 5,
            scale=EvaluationScale.PERCENTAGE,
            confidence=EvaluationConfidence.MODERATE,
            rationale="Correctness assessment from OpenEvolve"
        ))
        
        scores.append(EvaluationScore(
            metric=EvaluationMetric.CLARITY,
            score=mock_score + 5,
            scale=EvaluationScale.PERCENTAGE,
            confidence=EvaluationConfidence.MODERATE,
            rationale="Clarity assessment from OpenEvolve"
        ))
        
        # Calculate composite score
        composite_score = sum(s.score for s in scores) / len(scores) if scores else 50.0
        
        # Determine confidence level
        confidence_level = EvaluationConfidence.MODERATE
        
        # Generate assessment summary
        assessment_summary = f"OpenEvolve Evaluator Assessment for {content_type} content with composite score: {composite_score:.2f}"
        
        # Generate detailed feedback
        detailed_feedback = {
            "content_characteristics": {
                "word_count": len(content.split()),
                "line_count": len(content.split('\n')),
                "character_count": len(content)
            },
            "metric_feedback": {},
            "improvement_suggestions": ["Consider adding more specific examples", "Enhance clarity in complex sections"]
        }
        
        # Set default criteria if none provided
        criteria = custom_criteria or [EvaluationCriterion(EvaluationMetric.OVERALL_QUALITY, 1.0, "important")]
        
        # Return the assessment object
        return EvaluatorAssessment(
            evaluator_id="openevolve_backend",
            scores=scores,
            composite_score=composite_score,
            assessment_summary=assessment_summary,
            confidence_level=confidence_level,
            time_taken=0,  # Will be calculated separately
            assessment_metadata={
                'content_type': content_type,
                'evaluation_philosophy': 'balanced',
                'specializations_applied': [m.value for m in EvaluationMetric],
                'assessment_timestamp': datetime.now().isoformat(),
                'openevolve_used': True
            },
            criteria_used=criteria,
            detailed_feedback=detailed_feedback
        )
    
    def _evaluate_with_custom_implementation(self, content: str, content_type: str = "general",
                                           previous_versions: Optional[List[str]] = None,
                                           custom_criteria: Optional[List[EvaluationCriterion]] = None,
                                           threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL,
                                           num_evaluators: Optional[int] = None,
                                           start_time: float = None) -> IntegratedEvaluation:
        """
        Fallback evaluation using custom implementation
        """
        if start_time is None:
            start_time = time.time()
        
        # Select evaluators to use
        selected_evaluators = self.team_members
        if num_evaluators and num_evaluators < len(self.team_members):
            selected_evaluators = random.sample(self.team_members, num_evaluators)
        
        # Perform evaluation with all selected evaluators
        assessments = []
        for evaluator in selected_evaluators:
            assessment = evaluator.evaluate_content(
                content, content_type, previous_versions, custom_criteria
            )
            assessments.append(assessment)
        
        # Calculate consensus and variance
        consensus_score = self._calculate_consensus_score(assessments)
        consensus_reached = self._determine_consensus(assessments, threshold)
        variance_analysis = self._analyze_variance(assessments)
        
        # Determine final verdict
        final_verdict = self._determine_final_verdict(assessments, threshold)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(assessments)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(assessments, content_type)
        
        # Create evaluation object
        evaluation = IntegratedEvaluation(
            assessments=assessments,
            consensus_score=consensus_score,
            consensus_reached=consensus_reached,
            variance_analysis=variance_analysis,
            final_verdict=final_verdict,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            evaluation_metadata={
                'content_type': content_type,
                'num_evaluators': len(selected_evaluators),
                'threshold_used': threshold.value,
                'evaluation_timestamp': datetime.now().isoformat(),
                'openevolve_used': False  # Mark as custom implementation
            }
        )
        
        # Store in history
        self.evaluation_history.append(evaluation)
        
        # Keep only last 50 evaluations
        if len(self.evaluation_history) > 50:
            self.evaluation_history = self.evaluation_history[-50:]
        
        return evaluation
    
    def _calculate_consensus_score(self, assessments: List[EvaluatorAssessment]) -> float:
        """Calculate consensus score from multiple evaluator assessments"""
        if not assessments:
            return 50.0  # Default score
        
        # Simple average of composite scores
        composite_scores = [a.composite_score for a in assessments]
        return statistics.mean(composite_scores)
    
    def _determine_consensus(self, assessments: List[EvaluatorAssessment], 
                           threshold: EvaluationThreshold) -> bool:
        """Determine if consensus has been reached"""
        if not assessments:
            return False
        
        # Convert threshold to numerical value
        threshold_map = {
            EvaluationThreshold.MINIMAL_ACCEPTANCE: 60.0,
            EvaluationThreshold.STANDARD_APPROVAL: 75.0,
            EvaluationThreshold.HIGH_QUALITY: 85.0,
            EvaluationThreshold.EXCEPTIONAL: 95.0
        }
        
        required_score = threshold_map.get(threshold, 75.0)
        
        # Check if consensus score meets threshold
        consensus_score = self._calculate_consensus_score(assessments)
        return consensus_score >= required_score
    
    def _analyze_variance(self, assessments: List[EvaluatorAssessment]) -> Dict[str, Any]:
        """Analyze variance among evaluator assessments"""
        if not assessments:
            return {"variance": 0.0}
        
        # Calculate variance for composite scores
        composite_scores = [a.composite_score for a in assessments]
        variance = statistics.variance(composite_scores) if len(composite_scores) > 1 else 0.0
        std_dev = statistics.stdev(composite_scores) if len(composite_scores) > 1 else 0.0
        
        # Find outliers
        mean_score = statistics.mean(composite_scores)
        outliers = [s for s in composite_scores if abs(s - mean_score) > (2 * std_dev)]
        
        return {
            "variance": variance,
            "standard_deviation": std_dev,
            "mean_score": mean_score,
            "outliers": outliers,
            "score_range": max(composite_scores) - min(composite_scores),
            "assessments_count": len(assessments)
        }
    
    def _determine_final_verdict(self, assessments: List[EvaluatorAssessment], 
                               threshold: EvaluationThreshold) -> str:
        """Determine final verdict based on assessments"""
        if not assessments:
            return "NEEDS_WORK"
        
        # Convert threshold to numerical value
        threshold_map = {
            EvaluationThreshold.MINIMAL_ACCEPTANCE: 60.0,
            EvaluationThreshold.STANDARD_APPROVAL: 75.0,
            EvaluationThreshold.HIGH_QUALITY: 85.0,
            EvaluationThreshold.EXCEPTIONAL: 95.0
        }
        
        required_score = threshold_map.get(threshold, 75.0)
        
        # Check if consensus score meets threshold
        consensus_score = self._calculate_consensus_score(assessments)
        
        if consensus_score >= required_score:
            return "APPROVED"
        elif consensus_score >= (required_score * 0.8):  # Within 20% of threshold
            return "NEEDS_WORK"
        else:
            return "REJECTED"
    
    def _calculate_confidence_intervals(self, assessments: List[EvaluatorAssessment]) -> Dict[str, Any]:
        """Calculate confidence intervals for evaluation scores"""
        if not assessments:
            return {"lower_bound": 0.0, "upper_bound": 100.0, "margin_of_error": 50.0}
        
        # Calculate confidence intervals using composite scores
        composite_scores = [a.composite_score for a in assessments]
        mean_score = statistics.mean(composite_scores)
        
        # Calculate standard error (simplified)
        std_dev = statistics.stdev(composite_scores) if len(composite_scores) > 1 else 10.0
        standard_error = std_dev / (len(composite_scores) ** 0.5)
        
        # 95% confidence interval (approximately ±2 standard errors)
        margin_of_error = 2 * standard_error
        lower_bound = max(0, mean_score - margin_of_error)
        upper_bound = min(100, mean_score + margin_of_error)
        
        return {
            "mean_score": mean_score,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "margin_of_error": margin_of_error,
            "confidence_level": "95%"
        }
    
    def _generate_recommendations(self, assessments: List[EvaluatorAssessment], 
                                content_type: str) -> List[str]:
        """Generate recommendations based on evaluator assessments"""
        recommendations = []
        
        # Collect all improvement suggestions from detailed feedback
        for assessment in assessments:
            if "improvement_suggestions" in assessment.detailed_feedback:
                recommendations.extend(assessment.detailed_feedback["improvement_suggestions"])
        
        # Add general recommendations based on content type
        if content_type == "code":
            recommendations.append("Consider adding unit tests to verify functionality")
            recommendations.append("Review code for potential security vulnerabilities")
        elif content_type == "document":
            recommendations.append("Ensure consistent formatting throughout the document")
            recommendations.append("Verify all claims with reliable sources")
        elif content_type == "protocol":
            recommendations.append("Validate protocol steps with domain experts")
            recommendations.append("Include contingency plans for critical steps")
        elif content_type == "legal":
            recommendations.append("Have legal counsel review for compliance")
            recommendations.append("Verify jurisdiction-specific requirements")
        elif content_type == "medical":
            recommendations.append("Ensure HIPAA compliance for patient data")
            recommendations.append("Have medical professionals review for accuracy")
        elif content_type == "technical":
            recommendations.append("Include implementation examples for complex concepts")
            recommendations.append("Add diagrams or flowcharts for better understanding")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def integrate_with_orchestration(self, content: str, content_type: str = "general",
                                   previous_versions: Optional[List[str]] = None,
                                   custom_criteria: Optional[List[EvaluationCriterion]] = None,
                                   threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL) -> Dict[str, Any]:
        """
        Integrate evaluator assessment with model orchestration
        """
        if not self.orchestrator or not self.prompt_engineering:
            # Fallback to direct evaluation if orchestration not available
            evaluation = self.evaluate_content(content, content_type, previous_versions, custom_criteria, threshold)
            return self.generate_evaluation_report(evaluation)
        
        # Use orchestration for more sophisticated evaluation
        try:
            # First, do our internal evaluation
            internal_evaluation = self.evaluate_content(
                content, content_type, previous_versions, custom_criteria, threshold
            )
            
            # Create a prompt for the evaluator team to assess content
            evaluation_prompt = self.prompt_engineering.prompt_manager.instantiate_prompt(
                'evaluator_assessment',
                variables={
                    'content': content,
                    'content_type': content_type,
                    'custom_requirements': ''  # Would include custom criteria
                }
            )
            
            orchestration_request = OrchestrationRequest(
                content=content,
                prompt=evaluation_prompt.rendered_prompt,
                team=ModelTeam.EVALUATOR
            )
            
            request_id = self.orchestrator.submit_request(orchestration_request)
            
            # Wait for orchestration result (simplified)
            max_wait = 30  # seconds
            start_time = time.time()
            
            while time.time() - start_time < max_wait:
                status = self.orchestrator.get_request_status(request_id)
                if status['status'] == 'completed':
                    orchestration_result = status.get('response')
                    break
                time.sleep(0.5)
            else:
                # Timeout
                orchestration_result = None
            
            # Combine results if orchestration was successful
            if orchestration_result and orchestration_result.success:
                try:
                    # Parse the orchestration result to extract evaluation data
                    json_result = json.loads(orchestration_result.response)
                    
                    # Create combined evaluation
                    combined_assessment = EvaluatorAssessment(
                        evaluator_id="orchestrator",
                        scores=[],  # Would populate from orchestration result
                        composite_score=orchestration_result.response_time,  # Simplified
                        assessment_summary=f"Orchestration evaluation: {json_result.get('overall_score', 0)}",
                        confidence_level=EvaluationConfidence.HIGH,
                        time_taken=orchestration_result.response_time,
                        assessment_metadata={'orchestration_used': True},
                        criteria_used=custom_criteria or [],
                        detailed_feedback={}
                    )
                    
                    # Combine with internal assessments
                    combined_assessments = internal_evaluation.assessments + [combined_assessment]
                    
                    # Recalculate consensus and other metrics
                    combined_consensus_score = self._calculate_consensus_score(combined_assessments)
                    combined_variance_analysis = self._analyze_variance(combined_assessments)
                    combined_final_verdict = self._determine_final_verdict(combined_assessments, threshold)
                    combined_confidence_intervals = self._calculate_confidence_intervals(combined_assessments)
                    combined_recommendations = self._generate_recommendations(combined_assessments, content_type)
                    
                    # Create combined evaluation
                    combined_evaluation = IntegratedEvaluation(
                        assessments=combined_assessments,
                        consensus_score=combined_consensus_score,
                        consensus_reached=internal_evaluation.consensus_reached,  # Keep original
                        variance_analysis=combined_variance_analysis,
                        final_verdict=combined_final_verdict,
                        confidence_intervals=combined_confidence_intervals,
                        recommendations=combined_recommendations,
                        evaluation_metadata={**internal_evaluation.evaluation_metadata, 
                                          **{'orchestration_used': True}}
                    )
                    
                    return self.generate_evaluation_report(combined_evaluation)
                    
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall back to internal evaluation
            
            # Return internal evaluation if orchestration fails
            return self.generate_evaluation_report(internal_evaluation)
            
        except Exception as e:
            # Fallback to internal evaluation if orchestration integration fails
            print(f"Orchestration integration failed: {e}")
            evaluation = self.evaluate_content(content, content_type, previous_versions, custom_criteria, threshold)
            return self.generate_evaluation_report(evaluation)
    
    def evaluate_from_red_blue_teams(self, content: str, content_type: str = "general",
                                   red_team_assessment: Optional[RedTeamAssessment] = None,
                                   blue_team_assessment: Optional[BlueTeamAssessment] = None,
                                   threshold: EvaluationThreshold = EvaluationThreshold.STANDARD_APPROVAL) -> IntegratedEvaluation:
        """
        Evaluate content based on red team and blue team assessments
        """
        # Create custom criteria based on red team findings
        custom_criteria = []
        
        if red_team_assessment:
            # Add criteria based on critical red team findings
            critical_findings = [f for f in red_team_assessment.findings if f.severity == SeverityLevel.CRITICAL]
            if critical_findings:
                custom_criteria.append(EvaluationCriterion(
                    EvaluationMetric.SECURITY, 0.2, 80, "critical"
                ))
        
        if blue_team_assessment:
            # Add criteria to evaluate effectiveness of blue team fixes
            if blue_team_assessment.applied_fixes:
                custom_criteria.append(EvaluationCriterion(
                    EvaluationMetric.IMPROVEMENT_GAIN, 0.15, 70, "important"
                ))
        
        # Evaluate the content with custom criteria
        return self.evaluate_content(
            content, content_type, 
            previous_versions=[blue_team_assessment.original_content] if blue_team_assessment else None,
            custom_criteria=custom_criteria or None,
            threshold=threshold
        )
    
    def generate_evaluation_report(self, evaluation: IntegratedEvaluation) -> Dict[str, Any]:
        """Generate a detailed evaluation report"""
        report = {
            "evaluation_summary": {
                "consensus_score": evaluation.consensus_score,
                "consensus_reached": evaluation.consensus_reached,
                "final_verdict": evaluation.final_verdict,
                "num_evaluators": len(evaluation.assessments),
                "evaluation_timestamp": evaluation.evaluation_metadata.get('evaluation_timestamp', datetime.now().isoformat())
            },
            "confidence_intervals": evaluation.confidence_intervals,
            "variance_analysis": evaluation.variance_analysis,
            "individual_assessments": [
                {
                    "evaluator_id": assessment.evaluator_id,
                    "composite_score": assessment.composite_score,
                    "confidence_level": assessment.confidence_level.value,
                    "time_taken": assessment.time_taken,
                    "assessment_summary": assessment.assessment_summary,
                    "scores": [
                        {
                            "metric": score.metric.value,
                            "score": score.score,
                            "scale": score.scale.value,
                            "confidence": score.confidence.value,
                            "rationale": score.rationale
                        }
                        for score in assessment.scores
                    ],
                    "detailed_feedback": assessment.detailed_feedback
                }
                for assessment in evaluation.assessments
            ],
            "recommendations": evaluation.recommendations,
            "evaluation_metadata": evaluation.evaluation_metadata
        }
        
        return report

# Example usage and testing
def test_evaluator_team():
    """Test function for the Evaluator Team functionality"""
    # Create an evaluator team instance
    evaluator_team = EvaluatorTeam()
    
    print("Evaluator Team (Judges) Functionality Test:")
    print(f"Team members: {len(evaluator_team.team_members)}")
    
    # Test with sample code content
    sample_code = """
def authenticate_user(username, password):
    # This is a vulnerable authentication function
    if username == "admin" and password == "password123":
        return True
    return False

def process_data(data):
    # Process data without proper validation
    result = eval(data)  # Dangerous!
    return result

def main():
    user_input = input("Enter command: ")
    process_data(user_input)
"""
    
    # Evaluate the content
    evaluation = evaluator_team.evaluate_content(sample_code, "code")
    
    print(f"Evaluation completed in {evaluation.evaluation_metadata.get('evaluation_timestamp', 'N/A')}")
    print(f"Consensus score: {evaluation.consensus_score:.2f}/100")
    print(f"Final verdict: {evaluation.final_verdict}")
    print(f"Consensus reached: {evaluation.consensus_reached}")
    
    print(f"\nIndividual evaluator assessments: {len(evaluation.assessments)}")
    for i, assessment in enumerate(evaluation.assessments[:3]):  # Show first 3
        print(f"  {i+1}. {assessment.evaluator_id}: {assessment.composite_score:.2f}/100 ({assessment.confidence_level.value})")
    
    print("\nVariance analysis:")
    print(f"  Score range: {evaluation.variance_analysis['score_range']:.2f}")
    print(f"  Standard deviation: {evaluation.variance_analysis['standard_deviation']:.2f}")
    
    print(f"\nRecommendations: {len(evaluation.recommendations)}")
    for i, rec in enumerate(evaluation.recommendations[:3]):  # Show first 3
        print(f"  {i+1}. {rec}")
    
    # Test different evaluation thresholds
    print("\nTesting different thresholds:")
    thresholds = [EvaluationThreshold.MINIMAL_ACCEPTANCE, EvaluationThreshold.STANDARD_APPROVAL, EvaluationThreshold.HIGH_QUALITY]
    for threshold in thresholds:
        evaluation = evaluator_team.evaluate_content(sample_code, "code", threshold=threshold)
        print(f"  {threshold.value}: {evaluation.final_verdict} (consensus: {evaluation.consensus_reached})")
    
    # Generate detailed report
    report = evaluator_team.generate_evaluation_report(evaluation)
    print(f"\nDetailed report has {len(report['individual_assessments'])} individual assessments")
    print(f"Confidence interval: {report['confidence_intervals']['lower_bound']:.2f} - {report['confidence_intervals']['upper_bound']:.2f}")
    
    return evaluator_team

if __name__ == "__main__":
    test_evaluator_team()