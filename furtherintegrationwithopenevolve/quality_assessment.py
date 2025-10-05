"""
Quality Assessment Engine for OpenEvolve
Implements the Quality Assessment functionality described in the ultimate explanation document.
"""
import json
import re
import tempfile
import os
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import statistics
import numpy as np
from scipy import stats
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import hashlib
import time
from datetime import datetime

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    from openevolve.evaluation_result import EvaluationResult
    from openevolve.evaluator import Evaluator
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

class QualityDimension(Enum):
    """Enumeration of quality dimensions"""
    CORRECTNESS = "correctness"
    COMPLETENESS = "completeness"
    CLARITY = "clarity"
    EFFECTIVENESS = "effectiveness"
    EFFICIENCY = "efficiency"
    MAINTAINABILITY = "maintainability"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"
    AESTHETICS = "aesthetics"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"

class SeverityLevel(Enum):
    """Severity levels for quality issues"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class QualityAssessmentResult:
    """Result of a quality assessment"""
    scores: Dict[QualityDimension, float]
    composite_score: float
    dimension_breakdown: Dict[QualityDimension, Dict[str, Any]]
    issues: List[Dict[str, Any]]
    recommendations: List[str]
    assessment_metadata: Dict[str, Any]
    timestamp: str = ""

@dataclass
class QualityThreshold:
    """Configuration for quality thresholds"""
    dimension: QualityDimension
    min_score: float
    max_score: float
    weight: float
    critical: bool = False

@dataclass
class QualityIssue:
    """Represents a quality issue found during assessment"""
    dimension: QualityDimension
    severity: SeverityLevel
    description: str
    location: Optional[Union[int, str]] = None  # Line number, section, etc.
    suggested_fix: Optional[str] = None
    confidence: float = 1.0  # 0-1 confidence in the issue

class QualityAssessmentEngine:
    """Main engine for quality assessment"""
    
    def __init__(self):
        self.thresholds: List[QualityThreshold] = self._get_default_thresholds()
        self.assessment_methods: Dict[QualityDimension, Callable] = self._initialize_assessment_methods()
        self.issue_detectors: Dict[QualityDimension, Callable] = self._initialize_issue_detectors()
        self.aggregation_methods: Dict[str, Callable] = self._initialize_aggregation_methods()
        self.content_analyzer = None  # Will be set from content_analyzer.py if available
        self.prompt_analyzer = None   # Will be set from prompt_engineering.py if available
        self.model_orchestration = None  # Will be set from model_orchestration.py if available
    
    def _get_default_thresholds(self) -> List[QualityThreshold]:
        """Get default quality thresholds"""
        return [
            QualityThreshold(QualityDimension.CORRECTNESS, 70.0, 100.0, 0.15, False),
            QualityThreshold(QualityDimension.COMPLETENESS, 75.0, 100.0, 0.15, False),
            QualityThreshold(QualityDimension.CLARITY, 75.0, 100.0, 0.15, False),
            QualityThreshold(QualityDimension.EFFECTIVENESS, 75.0, 100.0, 0.15, False),
            QualityThreshold(QualityDimension.EFFICIENCY, 60.0, 100.0, 0.1, False),
            QualityThreshold(QualityDimension.MAINTAINABILITY, 70.0, 100.0, 0.1, False),
            QualityThreshold(QualityDimension.SCALABILITY, 65.0, 100.0, 0.05, False),
            QualityThreshold(QualityDimension.ROBUSTNESS, 70.0, 100.0, 0.1, False),
            QualityThreshold(QualityDimension.AESTHETICS, 50.0, 100.0, 0.05, False),
        ]
    
    def _initialize_assessment_methods(self) -> Dict[QualityDimension, Callable]:
        """Initialize assessment methods for each dimension"""
        return {
            QualityDimension.CORRECTNESS: self._assess_correctness,
            QualityDimension.COMPLETENESS: self._assess_completeness,
            QualityDimension.CLARITY: self._assess_clarity,
            QualityDimension.EFFECTIVENESS: self._assess_effectiveness,
            QualityDimension.EFFICIENCY: self._assess_efficiency,
            QualityDimension.MAINTAINABILITY: self._assess_maintainability,
            QualityDimension.SCALABILITY: self._assess_scalability,
            QualityDimension.ROBUSTNESS: self._assess_robustness,
            QualityDimension.AESTHETICS: self._assess_aesthetics,
            QualityDimension.COMPLIANCE: self._assess_compliance,
            QualityDimension.SECURITY: self._assess_security,
            QualityDimension.PERFORMANCE: self._assess_performance,
        }
    
    def _initialize_issue_detectors(self) -> Dict[QualityDimension, Callable]:
        """Initialize issue detection methods for each dimension"""
        return {
            QualityDimension.CORRECTNESS: self._detect_correctness_issues,
            QualityDimension.COMPLETENESS: self._detect_completeness_issues,
            QualityDimension.CLARITY: self._detect_clarity_issues,
            QualityDimension.EFFECTIVENESS: self._detect_effectiveness_issues,
            QualityDimension.EFFICIENCY: self._detect_efficiency_issues,
            QualityDimension.MAINTAINABILITY: self._detect_maintainability_issues,
            QualityDimension.SCALABILITY: self._detect_scalability_issues,
            QualityDimension.ROBUSTNESS: self._detect_robustness_issues,
            QualityDimension.AESTHETICS: self._detect_aesthetics_issues,
            QualityDimension.COMPLIANCE: self._detect_compliance_issues,
            QualityDimension.SECURITY: self._detect_security_issues,
            QualityDimension.PERFORMANCE: self._detect_performance_issues,
        }
    
    def _initialize_aggregation_methods(self) -> Dict[str, Callable]:
        """Initialize methods for aggregating multiple quality scores"""
        return {
            'weighted_average': self._weighted_average_aggregation,
            'geometric_mean': self._geometric_mean_aggregation,
            'harmonic_mean': self._harmonic_mean_aggregation,
        }
    
    def assess_quality(self, content: str, content_type: str = "general", 
                      custom_requirements: Optional[Dict[str, Any]] = None,
                      api_key: Optional[str] = None,
                      model_name: str = "gpt-4o") -> QualityAssessmentResult:
        """
        Assess the quality of content across multiple dimensions,
        using OpenEvolve backend when available
        
        Args:
            content: The content to assess
            content_type: Type of content (code, document, etc.)
            custom_requirements: Custom requirements to check
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
            
        Returns:
            QualityAssessmentResult with detailed quality metrics
        """
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            return self._assess_quality_with_openevolve_backend(
                content, content_type, custom_requirements, api_key, model_name
            )
        
        # Fallback to custom implementation
        return self._assess_quality_with_custom_implementation(
            content, content_type, custom_requirements
        )
    
    def _assess_quality_with_openevolve_backend(self, content: str, content_type: str, 
                                              custom_requirements: Optional[Dict[str, Any]], 
                                              api_key: str, model_name: str) -> QualityAssessmentResult:
        """
        Assess quality using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.2,  # Lower temperature for more consistent quality assessment
                max_tokens=2048,
            )
            
            config.llm.models = [llm_config]
            config.evolution.max_iterations = 1  # Just one quality assessment
            config.evolution.population_size = 1  # Single assessment
            
            # Create an evaluator for quality assessment
            def quality_evaluator(program_path: str) -> Dict[str, Any]:
                """
                Evaluator that performs quality assessment on the content
                """
                try:
                    with open(program_path, "r", encoding='utf-8') as f:
                        content = f.read()
                    
                    # Perform basic quality checks
                    word_count = len(content.split())
                    char_count = len(content)
                    
                    # Return quality metrics
                    return {
                        "score": 0.7,  # Placeholder quality score
                        "timestamp": datetime.now().timestamp(),
                        "content_length": len(content),
                        "word_count": word_count,
                        "character_count": char_count,
                        "assessment_completed": True
                    }
                except Exception as e:
                    print(f"Error in quality evaluator: {e}")
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
                # Run quality assessment using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=quality_evaluator,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Generate quality assessment results based on OpenEvolve output
                quality_result = self._generate_openevolve_quality_result(
                    content, content_type, result, custom_requirements
                )
                
                return quality_result
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._assess_quality_with_custom_implementation(content, content_type, custom_requirements)
    
    def _generate_openevolve_quality_result(self, content: str, content_type: str,
                                          result, custom_requirements: Optional[Dict[str, Any]]) -> QualityAssessmentResult:
        """
        Generate QualityAssessmentResult from OpenEvolve result
        """
        # Create mock scores based on the OpenEvolve result
        # In a real implementation, we would extract specific quality metrics from the result
        scores = {}
        
        # Add base quality dimensions with mock scores
        for dimension in QualityDimension:
            if dimension == QualityDimension.CORRECTNESS:
                scores[dimension] = 75.0
            elif dimension == QualityDimension.COMPLETENESS:
                scores[dimension] = 80.0
            elif dimension == QualityDimension.CLARITY:
                scores[dimension] = 85.0
            elif dimension == QualityDimension.EFFECTIVENESS:
                scores[dimension] = 70.0
            elif dimension == QualityDimension.EFFICIENCY:
                scores[dimension] = 75.0
            elif dimension == QualityDimension.MAINTAINABILITY:
                scores[dimension] = 80.0
            elif dimension == QualityDimension.SCALABILITY:
                scores[dimension] = 65.0
            elif dimension == QualityDimension.ROBUSTNESS:
                scores[dimension] = 70.0
            elif dimension == QualityDimension.AESTHETICS:
                scores[dimension] = 85.0
            elif dimension == QualityDimension.COMPLIANCE:
                scores[dimension] = 75.0
            elif dimension == QualityDimension.SECURITY:
                scores[dimension] = 60.0
            elif dimension == QualityDimension.PERFORMANCE:
                scores[dimension] = 70.0
            else:
                scores[dimension] = 75.0
        
        # Calculate composite score
        composite_score = sum(scores.values()) / len(scores) if scores else 50.0
        
        # Generate mock issues and recommendations
        all_issues = [
            QualityIssue(
                dimension=QualityDimension.CORRECTNESS,
                severity=SeverityLevel.MEDIUM,
                description="Quality assessment performed via OpenEvolve backend",
                confidence=0.8
            )
        ]
        
        recommendations = [
            "Consider using more specific examples",
            "Improve clarity in complex sections",
            "Add more comprehensive testing"
        ]
        
        # Create dimension breakdown
        dimension_breakdown = {}
        for dimension, score in scores.items():
            dimension_breakdown[dimension] = {
                'score': score,
                'issues': [issue for issue in all_issues if issue.dimension == dimension],
                'recommendations': recommendations if dimension == QualityDimension.CORRECTNESS else []
            }
        
        return QualityAssessmentResult(
            scores=scores,
            composite_score=composite_score,
            dimension_breakdown=dimension_breakdown,
            issues=all_issues,
            recommendations=recommendations,
            assessment_metadata={
                'content_length': len(content),
                'content_type': content_type,
                'assessment_timestamp': datetime.now().isoformat(),
                'method_used': 'openevolve_backend_multi_dimensional',
                'openevolve_used': True
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _assess_quality_with_custom_implementation(self, content: str, content_type: str = "general", 
                                                 custom_requirements: Optional[Dict[str, Any]] = None) -> QualityAssessmentResult:
        """
        Fallback quality assessment using custom implementation
        """
        scores = {}
        dimension_breakdown = {}
        all_issues = []
        recommendations = []
        
        # Assess each quality dimension
        for dimension, method in self.assessment_methods.items():
            if content_type == "code" and dimension == QualityDimension.COMPLIANCE:
                score = self._assess_code_compliance(content)
            elif content_type == "legal" and dimension == QualityDimension.COMPLIANCE:
                score = self._assess_legal_compliance(content)
            elif content_type == "medical" and dimension == QualityDimension.COMPLIANCE:
                score = self._assess_medical_compliance(content)
            elif content_type == "technical" and dimension == QualityDimension.COMPLIANCE:
                score = self._assess_technical_compliance(content)
            else:
                score = method(content, custom_requirements)
            
            scores[dimension] = score
            
            # Generate breakdown for this dimension
            issues = self.issue_detectors[dimension](content, custom_requirements)
            all_issues.extend(issues)
            
            # Add dimension-specific recommendations
            dim_recommendations = self._generate_recommendations_for_dimension(dimension, issues)
            recommendations.extend(dim_recommendations)
            
            # Store breakdown
            dimension_breakdown[dimension] = {
                'score': score,
                'issues': issues,
                'recommendations': dim_recommendations
            }
        
        # Calculate composite score
        composite_score = self._calculate_composite_score(scores, self.thresholds)
        
        # Apply custom requirements if any
        if custom_requirements:
            scores, composite_score = self._apply_custom_requirements(
                scores, composite_score, content, custom_requirements
            )
        
        return QualityAssessmentResult(
            scores=scores,
            composite_score=composite_score,
            dimension_breakdown=dimension_breakdown,
            issues=all_issues,
            recommendations=recommendations,
            assessment_metadata={
                'content_length': len(content),
                'content_type': content_type,
                'assessment_timestamp': datetime.now().isoformat(),
                'method_used': 'comprehensive_multi_dimensional',
                'openevolve_used': False  # Mark as custom implementation
            },
            timestamp=datetime.now().isoformat()
        )
    
    def _assess_correctness(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess correctness of the content"""
        # For text content, check basic structural correctness
        lines = content.split('\n')
        sentences = re.split(r'[.!?]+', content)
        words = content.split()
        
        # Calculate correctness indicators
        completeness_ratio = len([s for s in sentences if s.strip()]) / max(1, len(sentences))
        word_count = len(words)
        avg_sentence_length = word_count / max(1, len([s for s in sentences if s.strip()]))
        
        # Check for common error patterns
        error_patterns = [
            r'\bthi[sz]\b',  # Common typo for "this"
            r'\bteh\b',      # Common typo for "the"
            r'\ba n\b',      # Common typo for "an"
        ]
        
        error_count = 0
        for pattern in error_patterns:
            error_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        # Calculate base correctness score
        base_score = 100.0
        base_score -= error_count * 10  # Deduct points for errors
        
        # Adjust based on content characteristics
        if avg_sentence_length < 3 or avg_sentence_length > 50:
            base_score -= 10  # Sentences too short or too long
        if completeness_ratio < 0.8:
            base_score -= 15  # Incomplete sentences
        
        return max(0, min(100, base_score))
    
    def _assess_completeness(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess completeness of the content"""
        # Check if content has essential sections
        has_introduction = bool(re.search(r'^(#|##|###)?\s*(intro|overview|summary|introduction)', content, re.MULTILINE | re.IGNORECASE))
        has_conclusion = bool(re.search(r'^(#|##|###)?\s*(conclusion|summary|end)', content, re.MULTILINE | re.IGNORECASE))
        has_body_content = len(content.strip()) > 50  # At least 50 characters of content
        
        completeness_score = 0
        if has_introduction:
            completeness_score += 25
        if has_body_content:
            completeness_score += 50
        if has_conclusion:
            completeness_score += 25
        
        # Additional checks based on content type expectations
        word_count = len(content.split())
        if word_count < 100:
            completeness_score -= 20  # Too short for completeness
        elif word_count > 5000:
            completeness_score += 10  # Good length for detailed content
        
        return max(0, min(100, completeness_score))
    
    def _assess_clarity(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess clarity of the content"""
        # Use readability metrics
        readability = flesch_reading_ease(content)
        grade_level = flesch_kincaid_grade(content)
        
        # Convert readability score to 0-100 scale (Flesch score is -14.14 to 121.22)
        clarity_score = max(0, min(100, (readability + 15) * 4))  # Adjust scale
        
        # Adjust for grade level appropriateness
        if 6 <= grade_level <= 10:  # Good readability range
            clarity_score += 10
        elif grade_level > 12:  # Too complex
            clarity_score -= 20
        elif grade_level < 6:  # Possibly too simple for technical content
            clarity_score -= 5
        
        # Check for clarity indicators
        sentence_count = len([s for s in re.split(r'[.!?]+', content) if s.strip()])
        question_count = len(re.findall(r'\?', content))
        list_count = len(re.findall(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE))
        
        # Bonus for good structure
        if sentence_count > 5 and question_count > 0:
            clarity_score += 5  # Questions can improve clarity
        if list_count > 3:
            clarity_score += 10  # Good use of lists improves clarity
        
        return max(0, min(100, clarity_score))
    
    def _assess_effectiveness(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess effectiveness of the content in achieving its purpose"""
        # Check for action-oriented language
        action_indicators = [
            r'\bimplement\b', r'\bexecute\b', r'\bperform\b', r'\bconduct\b',
            r'\bcreate\b', r'\bdevelop\b', r'\bbuild\b', r'\bdesign\b'
        ]
        
        action_count = sum(len(re.findall(pattern, content)) for pattern in action_indicators)
        
        # Check for clear objectives
        objective_indicators = [
            r'\bpurpose\b', r'\bobjective\b', r'\bgoal\b', r'\baim\b',
            r'\btarget\b', r'\bintention\b', r'\bintent\b'
        ]
        
        objective_count = sum(len(re.findall(pattern, content)) for pattern in objective_indicators)
        
        # Check for results/outcomes
        outcome_indicators = [
            r'\bresult\b', r'\boutcome\b', r'\beffect\b', r'\bimpact\b',
            r'\btangible\b', r'\bmeasurable\b', r'\bquantifiable\b'
        ]
        
        outcome_count = sum(len(re.findall(pattern, content)) for pattern in outcome_indicators)
        
        # Calculate effectiveness score
        effectiveness_score = 30  # Base score for a complete document
        
        # Add points for action indicators
        effectiveness_score += min(20, action_count * 2)
        
        # Add points for clear objectives
        effectiveness_score += min(20, objective_count * 4)
        
        # Add points for outcomes
        effectiveness_score += min(30, outcome_count * 5)
        
        return max(0, min(100, effectiveness_score))
    
    def _assess_efficiency(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess efficiency (conciseness vs verbosity)"""
        word_count = len(content.split())
        char_count = len(content)
        sentence_count = len([s for s in re.split(r'[.!?]+', content) if s.strip()])
        
        if sentence_count == 0:
            return 50  # Default score if no sentences
        
        avg_sentence_length = word_count / sentence_count
        avg_word_length = char_count / word_count if word_count > 0 else 0
        
        # Check for redundancy
        words = content.lower().split()
        unique_words = set(words)
        redundancy_ratio = 1 - (len(unique_words) / len(words)) if words else 0
        
        efficiency_score = 100
        
        # Deduct for excessive length (assuming optimal range)
        if avg_sentence_length > 30:
            efficiency_score -= 15  # Sentences too long
        elif avg_sentence_length < 5:
            efficiency_score -= 10  # Sentences too short
        
        if redundancy_ratio > 0.3:
            efficiency_score -= 20  # Too much redundancy
        
        if avg_word_length > 6:  # Very long words might indicate complexity
            efficiency_score -= 10
        
        # Bonus for efficient structure
        if 10 < avg_sentence_length < 20 and redundancy_ratio < 0.2:
            efficiency_score += 10
        
        return max(0, min(100, efficiency_score))
    
    def _assess_maintainability(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess maintainability (organization, clarity for future edits)"""
        lines = content.split('\n')
        
        # Check for structure
        header_count = len(re.findall(r'^#+\s', content, re.MULTILINE))
        list_items = len(re.findall(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE))
        code_blocks = len(re.findall(r'```', content)) // 2 if '```' in content else 0
        
        # Check for comments/docs (for code)
        comment_patterns = [r'//.*', r'#.*', r'/\*.*?\*/', r'""".*?"""', r"'''.*?'''"]
        comment_lines = 0
        for pattern in comment_patterns:
            comment_lines += len(re.findall(pattern, content, re.DOTALL))
        
        maintainability_score = 20  # Base score
        
        # Add points for good structure
        maintainability_score += min(30, header_count * 5)
        maintainability_score += min(20, list_items * 2)
        maintainability_score += min(10, code_blocks * 3)
        maintainability_score += min(20, comment_lines * 2)
        
        # Check for consistent indentation (for code)
        indented_lines = len([line for line in lines if line.startswith('    ') or line.startswith('\t')])
        if indented_lines > 0:
            maintainability_score += 10
        
        return max(0, min(100, maintainability_score))
    
    def _assess_scalability(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess scalability (potential to grow/adapt)"""
        # For documentation/procedures, check for modular design mentions
        scalability_indicators = [
            r'\bmodular\b', r'\bscalable\b', r'\bextend\b', r'\bextendable\b',
            r'\bflexible\b', r'\badaptable\b', r'\bconfiguration\b', r'\bparameter\b',
            r'\boption\b', r'\bcustomize\b', r'\bplug-in\b', r'\bcomponent\b'
        ]
        
        scalability_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in scalability_indicators)
        
        # Check for abstraction language
        abstraction_indicators = [
            r'\babstract\b', r'\bgeneral\b', r'\bgeneric\b', r'\bframework\b',
            r'\barchitecture\b', r'\bpattern\b', r'\binterface\b', r'\bcontract\b'
        ]
        
        abstraction_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in abstraction_indicators)
        
        scalability_score = 20  # Base score
        
        # Add points for scalability indicators
        scalability_score += min(40, scalability_count * 5)
        scalability_score += min(40, abstraction_count * 5)
        
        return max(0, min(100, scalability_score))
    
    def _assess_robustness(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess robustness (handling of edge cases, errors)"""
        # Check for error handling mentions
        error_handling_indicators = [
            r'\berror\b', r'\bexception\b', r'\btry\b', r'\bcatch\b', r'\bexcept\b',
            r'\bhandle\b', r'\bvalidation\b', r'\bcheck\b', r'\bguard\b', r'\bsafety\b',
            r'\bcondition\b', r'\bassertion\b', r'\binput\b', r'\bvalidation\b'
        ]
        
        error_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in error_handling_indicators)
        
        # Check for edge case mentions
        edge_case_indicators = [
            r'\bedge\b', r'\bcorner\b', r'\bboundary\b', r'\blimit\b',
            r'\bminimum\b', r'\bmaximum\b', r'\bzero\b', r'\bnull\b', r'\bempty\b',
            r'\boverflow\b', r'\bunderflow\b', r'\btimeout\b'
        ]
        
        edge_case_count = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in edge_case_indicators)
        
        robustness_score = 30  # Base score
        
        # Add points for robustness indicators
        robustness_score += min(35, error_count * 3)
        robustness_score += min(35, edge_case_count * 4)
        
        return max(0, min(100, robustness_score))
    
    def _assess_aesthetics(self, content, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess aesthetics (visual appeal, formatting)"""
        lines = content.split('\n')
        
        # Check for consistent formatting
        line_lengths = [len(line) for line in lines if line.strip()]
        if not line_lengths:
            return 50  # Default if no content
        
        # Calculate line length consistency
        avg_line_length = statistics.mean(line_lengths)
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
    
    def _assess_compliance(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess compliance with standards and regulations"""
        # Generic compliance check
        compliance_indicators = [
            r'\bcompliance\b', r'\bstandard\b', r'\brule\b', r'\bpolicy\b',
            r'\bregulation\b', r'\bguideline\b', r'\brequirement\b', r'\bprotocol\b'
        ]
        
        compliance_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in compliance_indicators)
        
        compliance_score = 50  # Default score
        
        # Add points for compliance mentions
        compliance_score += min(30, compliance_mentions * 5)
        
        # Check for custom requirements if provided
        if custom_requirements and 'compliance_rules' in custom_requirements:
            rules_satisfied = 0
            total_rules = len(custom_requirements['compliance_rules'])
            
            for rule in custom_requirements['compliance_rules']:
                if rule.lower() in content.lower():
                    rules_satisfied += 1
            
            if total_rules > 0:
                rule_compliance = (rules_satisfied / total_rules) * 100
                compliance_score = (compliance_score + rule_compliance) / 2
        
        return max(0, min(100, compliance_score))
    
    def _assess_code_compliance(self, content: str) -> float:
        """Assess compliance for code content"""
        # Check for common code style indicators
        style_indicators = [
            r'# noqa',      # Linting exceptions
            r'# type:',     # Type hints
            r'def \w+\([^)]*\):',  # Function definitions
            r'class \w+',   # Class definitions
            r'if __name__ == ', # Main guard
        ]
        
        style_count = sum(len(re.findall(pattern, content)) for pattern in style_indicators)
        
        # Check for documentation strings
        docstring_patterns = [r'""".*?"""', r"'''.*?'''"]
        docstring_count = sum(len(re.findall(pattern, content, re.DOTALL)) for pattern in docstring_patterns)
        
        # Check for comments
        comment_count = len(re.findall(r'#.*', content))
        
        code_compliance_score = 20  # Base score
        
        code_compliance_score += min(30, style_count * 3)
        code_compliance_score += min(30, docstring_count * 5)
        code_compliance_score += min(20, comment_count * 0.5)
        
        return max(0, min(100, code_compliance_score))
    
    def _assess_legal_compliance(self, content: str) -> float:
        """Assess compliance for legal content"""
        # Check for legal compliance indicators
        legal_indicators = [
            r'\bdisclaimer\b', r'\bwarranty\b', r'\bliability\b',
            r'\bindemnity\b', r'\bconfidentiality\b', r'\bseverability\b',
            r'\bgoverning law\b', r'\bjurisdiction\b', r'\bforce majeure\b',
            r'\bnotarize\b', r'\bwitness\b', r'\bexecution\b'
        ]
        
        legal_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in legal_indicators)
        
        legal_compliance_score = 20  # Base score
        legal_compliance_score += min(80, legal_mentions * 5)
        
        return max(0, min(100, legal_compliance_score))
    
    def _assess_medical_compliance(self, content: str) -> float:
        """Assess compliance for medical content"""
        # Check for medical compliance indicators
        medical_indicators = [
            r'\bHIPAA\b', r'\bFDA\b', r'\bclinical\b', r'\bmedical device\b',
            r'\bpatient rights\b', r'\binformed consent\b', r'\bprivacy\b',
            r'\bconfidentiality\b', r'\bmedical advice\b', r'\bdiagnosis\b',
            r'\btreatment\b', r'\bprescription\b'
        ]
        
        medical_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in medical_indicators)
        
        medical_compliance_score = 20  # Base score
        medical_compliance_score += min(80, medical_mentions * 5)
        
        return max(0, min(100, medical_compliance_score))
    
    def _assess_technical_compliance(self, content: str) -> float:
        """Assess compliance for technical content"""
        # Check for technical compliance indicators
        tech_indicators = [
            r'\bISO\b', r'\bIEEE\b', r'\bRFC\b', r'\bstandard\b',
            r'\bprotocol\b', r'\bspecification\b', r'\bcompliance\b',
            r'\bsecurity\b', r'\bvalidation\b', r'\bverification\b',
            r'\bquality assurance\b', r'\btesting\b', r'\bcertification\b'
        ]
        
        tech_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in tech_indicators)
        
        tech_compliance_score = 20  # Base score
        tech_compliance_score += min(80, tech_mentions * 5)
        
        return max(0, min(100, tech_compliance_score))
    
    def _assess_security(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess security considerations in content"""
        security_indicators = [
            r'\bsecurity\b', r'\bauthentication\b', r'\bauthorization\b', r'\bencryption\b',
            r'\bpassword\b', r'\bcredential\b', r'\btoken\b', r'\bAPI key\b', r'\bsecret\b',
            r'\bvulnerability\b', r'\bpenetration test\b', r'\bOWASP\b', r'\bSAST\b', r'\bDAST\b',
            r'\binput validation\b', r'\bSQL injection\b', r'\bXSS\b', r'\bCSRF\b', r'\bSSRF\b'
        ]
        
        security_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in security_indicators)
        
        security_score = 30  # Base score
        
        # Add points for security mentions
        security_score += min(70, security_mentions * 3)
        
        # Check for security-sensitive content that should be protected
        sensitive_patterns = [
            r'password\s*[:=]\s*\S+', r'API_key\s*[:=]\s*\S+', r'token\s*[:=]\s*\S+',
            r'secret\s*[:=]\s*\S+', r'key\s*[:=]\s*\S+'
        ]
        
        sensitive_exposures = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in sensitive_patterns)
        
        # Deduct points for sensitive exposures
        security_score -= sensitive_exposures * 20
        
        return max(0, min(100, security_score))
    
    def _assess_performance(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> float:
        """Assess performance considerations in content"""
        performance_indicators = [
            r'\bperformance\b', r'\boptimize\b', r'\befficiency\b', r'\bspeed\b', r'\bfast\b',
            r'\bcache\b', r'\bindex\b', r'\balgorithm\b', r'\bcomplexity\b', r'\bO\(n\)\b',
            r'\bmemory\b', r'\bCPU\b', r'\bGPU\b', r'\bparallel\b', r'\bconcurrent\b',
            r'\bscalability\b', r'\bload\b', r'\bthroughput\b', r'\blatency\b', r'\bbandwidth\b'
        ]
        
        performance_mentions = sum(len(re.findall(pattern, content, re.IGNORECASE)) for pattern in performance_indicators)
        
        performance_score = 20  # Base score
        
        # Add points for performance mentions
        performance_score += min(80, performance_mentions * 3)
        
        return max(0, min(100, performance_score))
    
    def _detect_correctness_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect correctness issues in content"""
        issues = []
        
        # Check for common language errors
        error_patterns = [
            (r'\bthi[sz]\b', 'Possible typo for "this" or "these"'),
            (r'\bteh\b', 'Possible typo for "the"'),
            (r'\ba n\b', 'Possible typo for "an"'),
            (r'\bi\'m\snot\b', 'Consider "I am not" or "I\'m not" for clarity'),
        ]
        
        for pattern, description in error_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.CORRECTNESS,
                    severity=SeverityLevel.LOW,
                    description=description,
                    location=match.start(),
                    confidence=0.8
                ))
        
        # Check for factual inconsistencies
        sentences = re.split(r'[.!?]+', content)
        for i, sentence in enumerate(sentences):
            # Check for contradicting statements
            if re.search(r'does not.*but.*does', sentence, re.IGNORECASE):
                issues.append(QualityIssue(
                    dimension=QualityDimension.CORRECTNESS,
                    severity=SeverityLevel.MEDIUM,
                    description="Possible logical contradiction in sentence",
                    location=i,
                    confidence=0.7
                ))
        
        return issues
    
    def _detect_completeness_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect completeness issues in content"""
        issues = []
        
        # Check if content has essential sections
        sections = re.findall(r'^#+\s+(.*)', content, re.MULTILINE)
        if len(sections) < 2:
            issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=SeverityLevel.HIGH,
                description="Content might be missing essential sections. Consider adding more structure with headers.",
                confidence=0.9
            ))
        
        # Check content length relative to expectations
        word_count = len(content.split())
        if word_count < 100:
            issues.append(QualityIssue(
                dimension=QualityDimension.COMPLETENESS,
                severity=SeverityLevel.MEDIUM,
                description="Content appears to be very brief. Consider adding more detail.",
                confidence=0.8
            ))
        
        return issues
    
    def _detect_clarity_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect clarity issues in content"""
        issues = []
        
        sentences = re.split(r'[.!?]+', content)
        
        for i, sentence in enumerate(sentences):
            stripped = sentence.strip()
            if len(stripped) == 0:
                continue
                
            words = stripped.split()
            if len(words) > 30:  # Very long sentences can hurt clarity
                issues.append(QualityIssue(
                    dimension=QualityDimension.CLARITY,
                    severity=SeverityLevel.MEDIUM,
                    description=f"Sentence {i+1} is very long ({len(words)} words) and may hurt readability.",
                    location=i,
                    confidence=0.9
                ))
        
        # Check for ambiguous language
        ambiguous_patterns = [
            (r'\bit\b', 'Ambiguous pronoun "it" - consider being more specific'),
            (r'\bthis\b', 'Ambiguous pronoun "this" - consider being more specific'),
            (r'\bthat\b', 'Ambiguous pronoun "that" - consider being more specific'),
        ]
        
        for pattern, description in ambiguous_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.CLARITY,
                    severity=SeverityLevel.LOW,
                    description=description,
                    location=match.start(),
                    confidence=0.6
                ))
        
        return issues
    
    def _detect_effectiveness_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect effectiveness issues in content"""
        issues = []
        
        # Check for vague language that reduces effectiveness
        vague_patterns = [
            (r'\bsomehow\b', 'Vague term reduces effectiveness'),
            (r'\bsomewhat\b', 'Vague term reduces effectiveness'),
            (r'\bmaybe\b', 'Vague term reduces effectiveness'),
            (r'\bperhaps\b', 'Vague term reduces effectiveness'),
            (r'\bquite\b', 'Vague term reduces effectiveness'),
        ]
        
        for pattern, description in vague_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.EFFECTIVENESS,
                    severity=SeverityLevel.MEDIUM,
                    description=description,
                    location=match.start(),
                    confidence=0.7
                ))
        
        # Check for missing concrete examples
        if not re.search(r'example|sample|demonstrate|show', content, re.IGNORECASE):
            issues.append(QualityIssue(
                dimension=QualityDimension.EFFECTIVENESS,
                severity=SeverityLevel.LOW,
                description="Content lacks concrete examples that would enhance effectiveness",
                confidence=0.6
            ))
        
        return issues
    
    def _detect_efficiency_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect efficiency issues in content"""
        issues = []
        
        # Check for redundant phrases
        redundant_patterns = [
            (r'\b(very|really|quite|extremely)\s+(important|critical|essential|vital)\b', 'Redundant intensifier'),
            (r'\b(free|gratis)\s+of\s+charge\b', 'Redundant phrase'),
            (r'\b(advance|forward)\s+planning\b', 'Redundant phrase'),
        ]
        
        for pattern, description in redundant_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.EFFICIENCY,
                    severity=SeverityLevel.LOW,
                    description=description,
                    location=match.start(),
                    confidence=0.8
                ))
        
        return issues
    
    def _detect_maintainability_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect maintainability issues in content"""
        issues = []
        
        # Check for magic numbers/values without explanation
        magic_numbers = re.findall(r'\b\d{3,}\b', content)
        if len(magic_numbers) > 3:
            issues.append(QualityIssue(
                dimension=QualityDimension.MAINTAINABILITY,
                severity=SeverityLevel.MEDIUM,
                description=f"Found {len(magic_numbers)} potential magic numbers that may hurt maintainability",
                confidence=0.7
            ))
        
        # Check for poor structure (for code-like content)
        lines = content.split('\n')
        consecutive_long_lines = 0
        for line in lines:
            if len(line) > 100:  # Very long lines hurt maintainability
                consecutive_long_lines += 1
                if consecutive_long_lines >= 3:
                    issues.append(QualityIssue(
                        dimension=QualityDimension.MAINTAINABILITY,
                        severity=SeverityLevel.MEDIUM,
                        description="Multiple consecutive long lines hurt code maintainability",
                        confidence=0.8
                    ))
                    break
            else:
                consecutive_long_lines = 0
        
        return issues
    
    def _detect_scalability_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect scalability issues in content"""
        issues = []
        
        # Check for hardcoded limitations
        hardcoded_limits = re.findall(r'(max|limit|size)\s*[:=]\s*\d+', content, re.IGNORECASE)
        if hardcoded_limits:
            issues.append(QualityIssue(
                dimension=QualityDimension.SCALABILITY,
                severity=SeverityLevel.HIGH,
                description=f"Found hardcoded limits that may hurt scalability: {', '.join(hardcoded_limits[:3])}",
                confidence=0.8
            ))
        
        return issues
    
    def _detect_robustness_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect robustness issues in content"""
        issues = []
        
        # Check for missing error handling language
        if not re.search(r'(error|exception|catch|handle|validation)', content, re.IGNORECASE):
            issues.append(QualityIssue(
                dimension=QualityDimension.ROBUSTNESS,
                severity=SeverityLevel.MEDIUM,
                description="Content appears to lack error handling considerations, which hurts robustness",
                confidence=0.7
            ))
        
        # Check for unsafe operations
        unsafe_patterns = [
            (r'eval\s*\(', 'Use of eval() is unsafe'),
            (r'exec\s*\(', 'Use of exec() is unsafe'),
            (r'os\.system\s*\(', 'Direct system calls may be unsafe'),
        ]
        
        for pattern, description in unsafe_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.ROBUSTNESS,
                    severity=SeverityLevel.CRITICAL,
                    description=description,
                    location=match.start(),
                    confidence=0.9
                ))
        
        return issues
    
    def _detect_aesthetics_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect aesthetics issues in content"""
        issues = []
        
        lines = content.split('\n')
        
        # Check for inconsistent indentation
        indentation_levels = []
        for line in lines:
            if line.strip():  # Skip empty lines
                leading_spaces = len(line) - len(line.lstrip())
                indentation_levels.append(leading_spaces)
        
        if len(indentation_levels) > 2:
            # Check if indentation is consistent
            unique_levels = set(indentation_levels)
            if len(unique_levels) > 3:  # More than a few consistent levels
                issues.append(QualityIssue(
                    dimension=QualityDimension.AESTHETICS,
                    severity=SeverityLevel.MEDIUM,
                    description="Inconsistent indentation levels hurt visual aesthetics",
                    confidence=0.7
                ))
        
        # Check for line length consistency
        line_lengths = [len(line) for line in lines if line.strip()]
        if line_lengths and statistics.stdev(line_lengths) > 40:  # High variation in line lengths
            issues.append(QualityIssue(
                dimension=QualityDimension.AESTHETICS,
                severity=SeverityLevel.LOW,
                description="High variation in line lengths hurts visual consistency",
                confidence=0.6
            ))
        
        return issues
    
    def _detect_compliance_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect compliance issues in content"""
        issues = []
        
        # Check for missing compliance considerations
        if custom_requirements and 'required_compliance_terms' in custom_requirements:
            required_terms = custom_requirements['required_compliance_terms']
            for term in required_terms:
                if term.lower() not in content.lower():
                    issues.append(QualityIssue(
                        dimension=QualityDimension.COMPLIANCE,
                        severity=SeverityLevel.HIGH,
                        description=f"Missing required compliance term: {term}",
                        confidence=0.9
                    ))
        
        return issues
    
    def _detect_security_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect security issues in content"""
        issues = []
        
        # Check for exposed credentials
        credential_patterns = [
            (r'password\s*[:=]\s*[\'"]?(\w{8,})[\'"]?', 'Exposed password'),
            (r'API_key\s*[:=]\s*[\'"]?(\w{10,})[\'"]?', 'Exposed API key'),
            (r'token\s*[:=]\s*[\'"]?(\w{10,})[\'"]?', 'Exposed token'),
            (r'secret\s*[:=]\s*[\'"]?(\w{8,})[\'"]?', 'Exposed secret'),
        ]
        
        for pattern, description in credential_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.SECURITY,
                    severity=SeverityLevel.CRITICAL,
                    description=description,
                    location=match.start(),
                    confidence=0.95
                ))
        
        # Check for unsafe operations
        unsafe_ops = re.findall(r'(eval|exec|os\.system|subprocess\.call)\s*\(', content)
        if unsafe_ops:
            issues.append(QualityIssue(
                dimension=QualityDimension.SECURITY,
                severity=SeverityLevel.HIGH,
                description=f"Found potentially unsafe operations: {', '.join(set(unsafe_ops))}",
                confidence=0.8
            ))
        
        return issues
    
    def _detect_performance_issues(self, content: str, custom_requirements: Optional[Dict[str, Any]] = None) -> List[QualityIssue]:
        """Detect performance issues in content"""
        issues = []
        
        # Check for performance-inefficient patterns
        inefficient_patterns = [
            (r'for.*in.*range\(\d{6,}\)', 'Looping over very large range'),
            (r'while\s+True', 'Infinite loop possibility'),
            (r'\.append\(\)\s+in\s+loop', 'Repeated list append in loop (consider list comprehension)'),
        ]
        
        for pattern, description in inefficient_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                issues.append(QualityIssue(
                    dimension=QualityDimension.PERFORMANCE,
                    severity=SeverityLevel.MEDIUM,
                    description=description,
                    location=match.start(),
                    confidence=0.7
                ))
        
        return issues
    
    def _calculate_composite_score(self, scores: Dict[QualityDimension, float], 
                                 thresholds: List[QualityThreshold]) -> float:
        """Calculate composite quality score from individual dimension scores"""
        weighted_sum = 0
        total_weight = 0
        
        for threshold in thresholds:
            score = scores.get(threshold.dimension, 50)  # Default to 50 if not assessed
            weighted_sum += score * threshold.weight
            total_weight += threshold.weight
        
        if total_weight == 0:
            return 50  # Default score
        
        composite_score = weighted_sum / total_weight
        return max(0, min(100, composite_score))
    
    def _apply_custom_requirements(self, scores: Dict[QualityDimension, float], 
                                 composite_score: float, content: str, 
                                 custom_requirements: Dict[str, Any]) -> Tuple[Dict[QualityDimension, float], float]:
        """Apply custom requirements to adjust scores"""
        modified_scores = scores.copy()
        modified_composite = composite_score
        
        if 'dimension_multipliers' in custom_requirements:
            multipliers = custom_requirements['dimension_multipliers']
            for dimension, multiplier in multipliers.items():
                try:
                    dim_enum = QualityDimension(dimension)
                    old_score = modified_scores.get(dim_enum, 50)
                    new_score = max(0, min(100, old_score * multiplier))
                    modified_scores[dim_enum] = new_score
                except ValueError:
                    continue  # Skip invalid dimension names
        
        # Recalculate composite score with modified scores
        total_weight = sum(t.weight for t in self.thresholds)
        if total_weight > 0:
            weighted_sum = sum(modified_scores.get(t.dimension, 50) * t.weight for t in self.thresholds)
            modified_composite = weighted_sum / total_weight
        else:
            modified_composite = composite_score
        
        return modified_scores, modified_composite
    
    def _generate_recommendations_for_dimension(self, dimension: QualityDimension, 
                                              issues: List[QualityIssue]) -> List[str]:
        """Generate recommendations for a specific quality dimension"""
        recommendations = []
        
        if dimension == QualityDimension.CORRECTNESS:
            recommendations.append("Review content for factual accuracy and consistency.")
            if any(issue.severity == SeverityLevel.CRITICAL for issue in issues):
                recommendations.append("Address critical correctness issues immediately.")
        
        elif dimension == QualityDimension.CLARITY:
            recommendations.extend([
                "Break down long sentences for better readability.",
                "Use more concrete examples to illustrate concepts.",
                "Consider your audience's knowledge level when explaining concepts."
            ])
        
        elif dimension == QualityDimension.EFFECTIVENESS:
            recommendations.extend([
                "Use more action-oriented language to enhance effectiveness.",
                "Provide clear outcomes and measurable results where possible.",
                "Consider adding specific examples to demonstrate effectiveness."
            ])
        
        elif dimension == QualityDimension.SECURITY:
            recommendations.extend([
                "Implement proper input validation and output encoding.",
                "Use secure coding practices and avoid unsafe operations.",
                "Store sensitive information securely using encryption."
            ])
        
        # Add specific issues as recommendations
        for issue in issues:
            if issue.suggested_fix:
                recommendations.append(issue.suggested_fix)
            elif issue.description:
                recommendations.append(f"Address: {issue.description}")
        
        return recommendations
    
    def _weighted_average_aggregation(self, scores: Dict[QualityDimension, float], 
                                    weights: Dict[QualityDimension, float]) -> float:
        """Calculate weighted average of scores"""
        total_weighted_score = 0
        total_weight = 0
        
        for dimension, score in scores.items():
            weight = weights.get(dimension, 1.0)
            total_weighted_score += score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0
        
        return total_weighted_score / total_weight
    
    def _geometric_mean_aggregation(self, scores: Dict[QualityDimension, float], 
                                  weights: Dict[QualityDimension, float] = None) -> float:
        """Calculate geometric mean of scores"""
        if not scores:
            return 0
            
        # Convert scores to values between 0.01 and 1 (add 0.01 to avoid log(0))
        values = [max(0.01, min(1.0, score / 100.0)) for score in scores.values()]
        
        # Calculate geometric mean
        log_sum = sum(np.log(val) for val in values)
        geo_mean = np.exp(log_sum / len(values))
        
        # Convert back to 0-100 scale
        return geo_mean * 100
    
    def _harmonic_mean_aggregation(self, scores: Dict[QualityDimension, float], 
                                 weights: Dict[QualityDimension, float] = None) -> float:
        """Calculate harmonic mean of scores"""
        if not scores:
            return 0
            
        # Convert to values between 0.01 and 1
        values = [max(0.01, min(1.0, score / 100.0)) for score in scores.values()]
        
        # Calculate harmonic mean
        harmonic_mean = len(values) / sum(1/val for val in values)
        
        # Convert back to 0-100 scale
        return harmonic_mean * 100
    
    def get_quality_report(self, assessment_result: QualityAssessmentResult) -> str:
        """Generate a quality report from assessment results"""
        report = []
        report.append("Quality Assessment Report")
        report.append("=" * 50)
        report.append(f"Assessment Time: {assessment_result.timestamp}")
        report.append(f"Composite Score: {assessment_result.composite_score:.2f}/100")
        report.append("")
        
        report.append("Dimension Scores:")
        for dimension, score in assessment_result.scores.items():
            report.append(f"  {dimension.value.title()}: {score:.2f}/100")
        report.append("")
        
        report.append(f"Issues Found: {len(assessment_result.issues)}")
        if assessment_result.issues:
            for issue in assessment_result.issues[:10]:  # Show first 10 issues
                severity_str = issue.severity.value.upper()
                report.append(f"  [{severity_str}] {issue.dimension.value}: {issue.description}")
            if len(assessment_result.issues) > 10:
                report.append(f"  ... and {len(assessment_result.issues) - 10} more issues")
        report.append("")
        
        report.append(f"Recommendations: {len(assessment_result.recommendations)}")
        for rec in assessment_result.recommendations[:5]:  # Show first 5 recommendations
            report.append(f"  • {rec}")
        if len(assessment_result.recommendations) > 5:
            report.append(f"  ... and {len(assessment_result.recommendations) - 5} more recommendations")
        
        return "\n".join(report)

# Example usage and testing
def test_quality_assessment_engine():
    """Test function for the Quality Assessment Engine"""
    engine = QualityAssessmentEngine()
    
    # Test with sample content
    sample_content = """
# Sample Technical Documentation

## Overview
This document describes a technical process for data validation. The system should always validate inputs before processing them.

## Requirements
- All user inputs must be validated
- Error handling must be implemented
- Security tokens should be stored securely

## Implementation
The function validate_input() performs validation checks. It checks for valid formats and returns appropriate error messages.

### Error Handling
The system implements proper error handling with try-catch blocks. Any validation failures result in appropriate error responses.
"""
    
    print("Quality Assessment Engine Test:")
    
    # Perform assessment
    result = engine.assess_quality(sample_content, "technical")
    
    print(f"Composite Score: {result.composite_score:.2f}/100")
    print(f"Issues Found: {len(result.issues)}")
    print(f"Recommendations: {len(result.recommendations)}")
    
    print("\nTop 3 Dimension Scores:")
    sorted_scores = sorted(result.scores.items(), key=lambda x: x[1], reverse=True)
    for i, (dimension, score) in enumerate(sorted_scores[:3]):
        print(f"  {i+1}. {dimension.value}: {score:.2f}")
    
    print("\nSample Issues Found:")
    for issue in result.issues[:3]:
        print(f"  - {issue.severity.value.upper()}: {issue.description} [{issue.dimension.value}]")
    
    print("\nSample Recommendations:")
    for rec in result.recommendations[:3]:
        print(f"  - {rec}")
    
    # Generate quality report
    report = engine.get_quality_report(result)
    print(f"\nQuality Report:\n{report}")
    
    return engine

if __name__ == "__main__":
    test_quality_assessment_engine()