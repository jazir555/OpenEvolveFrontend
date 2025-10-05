"""
Quality Assurance Mechanisms for OpenEvolve
Implements the Quality Assurance Mechanisms functionality described in the ultimate explanation document.
"""
import json
import re
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import time
from datetime import datetime
import random
import copy
import statistics
import hashlib
import logging
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QualityAssuranceType(Enum):
    """Types of quality assurance mechanisms"""
    VALIDATION = "validation"
    VERIFICATION = "verification"
    TESTING = "testing"
    MONITORING = "monitoring"
    AUDITING = "auditing"
    COMPLIANCE = "compliance"
    SECURITY = "security"
    PERFORMANCE = "performance"
    CORRECTNESS = "correctness"
    CONSISTENCY = "consistency"
    RELIABILITY = "reliability"
    MAINTAINABILITY = "maintainability"
    USABILITY = "usability"

class QualityGateType(Enum):
    """Types of quality gates"""
    PRE_PROCESSING = "pre_processing"
    POST_PROCESSING = "post_processing"
    MODEL_INPUT = "model_input"
    MODEL_OUTPUT = "model_output"
    EVOLUTION_STEP = "evolution_step"
    ADVERSARIAL_ROUND = "adversarial_round"
    EVALUATION_PHASE = "evaluation_phase"
    FINAL_APPROVAL = "final_approval"
    EXPORT_RELEASE = "export_release"

class QualityIssueSeverity(Enum):
    """Severity levels for quality issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKER = "blocker"

class QualityAssuranceStatus(Enum):
    """Status of quality assurance checks"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"

@dataclass
class QualityAssuranceRule:
    """Individual quality assurance rule"""
    rule_id: str
    name: str
    description: str
    qa_type: QualityAssuranceType
    gate_type: QualityGateType
    severity: QualityIssueSeverity
    enabled: bool = True
    conditions: List[Dict[str, Any]] = None
    actions: List[Dict[str, Any]] = None
    failure_message: Optional[str] = None
    success_message: Optional[str] = None
    remediation_steps: Optional[List[str]] = None
    validation_function: Optional[Callable] = None
    tags: List[str] = None
    version_added: str = "1.0.0"
    last_updated: str = ""
    
    def __post_init__(self):
        if self.conditions is None:
            self.conditions = []
        if self.actions is None:
            self.actions = []
        if self.tags is None:
            self.tags = []
        if not self.last_updated:
            self.last_updated = datetime.now().isoformat()

@dataclass
class QualityAssuranceResult:
    """Result of a quality assurance check"""
    rule_id: str
    rule_name: str
    status: QualityAssuranceStatus
    issues: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    execution_time: float
    timestamp: str
    context: Dict[str, Any]
    remediation_available: bool = False
    remediation_suggested: Optional[List[str]] = None

@dataclass
class QualityGateResult:
    """Result of a quality gate check"""
    gate_type: QualityGateType
    overall_status: QualityAssuranceStatus
    results: List[QualityAssuranceResult]
    passed_count: int
    failed_count: int
    total_checks: int
    execution_time: float
    timestamp: str
    context: Dict[str, Any]
    blocking_issues: List[Dict[str, Any]]
    recommendations: List[str]

class QualityAssuranceMechanism(ABC):
    """Abstract base class for quality assurance mechanisms"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True
        self.rules: Dict[str, QualityAssuranceRule] = {}
        self.results_history: List[QualityAssuranceResult] = []
        self.execution_stats: Dict[str, Any] = {
            "total_executions": 0,
            "passed_count": 0,
            "failed_count": 0,
            "average_execution_time": 0.0,
            "error_count": 0
        }
    
    @abstractmethod
    def validate(self, content: str, context: Dict[str, Any]) -> QualityAssuranceResult:
        """Validate content according to this mechanism"""
        pass
    
    def add_rule(self, rule: QualityAssuranceRule):
        """Add a validation rule"""
        self.rules[rule.rule_id] = rule
        logger.info(f"Added QA rule {rule.rule_id} to {self.name}")
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a validation rule"""
        if rule_id in self.rules:
            del self.rules[rule_id]
            logger.info(f"Removed QA rule {rule_id} from {self.name}")
            return True
        return False
    
    def get_rule(self, rule_id: str) -> Optional[QualityAssuranceRule]:
        """Get a validation rule by ID"""
        return self.rules.get(rule_id)
    
    def list_rules(self) -> List[QualityAssuranceRule]:
        """List all validation rules"""
        return list(self.rules.values())
    
    def _record_result(self, result: QualityAssuranceResult):
        """Record a validation result"""
        self.results_history.append(result)
        
        # Update execution statistics
        self.execution_stats["total_executions"] += 1
        if result.status == QualityAssuranceStatus.PASSED:
            self.execution_stats["passed_count"] += 1
        elif result.status in [QualityAssuranceStatus.FAILED, QualityAssuranceStatus.BLOCKER]:
            self.execution_stats["failed_count"] += 1
        elif result.status == QualityAssuranceStatus.ERROR:
            self.execution_stats["error_count"] += 1
        
        # Update average execution time
        total_time = self.execution_stats["average_execution_time"] * (self.execution_stats["total_executions"] - 1)
        total_time += result.execution_time
        self.execution_stats["average_execution_time"] = total_time / self.execution_stats["total_executions"]
        
        # Keep history within reasonable limits
        if len(self.results_history) > 1000:
            self.results_history = self.results_history[-500:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return copy.deepcopy(self.execution_stats)
    
    def get_recent_results(self, count: int = 10) -> List[QualityAssuranceResult]:
        """Get recent validation results"""
        return self.results_history[-count:] if self.results_history else []

class InputValidationMechanism(QualityAssuranceMechanism):
    """Input validation quality assurance mechanism"""
    
    def __init__(self):
        super().__init__("Input Validation", "Validates input content for correctness and safety")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        # Length validation rule
        self.add_rule(QualityAssuranceRule(
            rule_id="input_length_check",
            name="Input Length Validation",
            description="Ensures input content is within acceptable length limits",
            qa_type=QualityAssuranceType.VALIDATION,
            gate_type=QualityGateType.MODEL_INPUT,
            severity=QualityIssueSeverity.ERROR,
            conditions=[
                {"min_length": 1},
                {"max_length": 100000}
            ],
            failure_message="Input content length is outside acceptable limits",
            success_message="Input content length is within acceptable limits",
            validation_function=self._validate_length
        ))
        
        # Syntax validation rule
        self.add_rule(QualityAssuranceRule(
            rule_id="input_syntax_check",
            name="Input Syntax Validation",
            description="Checks for basic syntax correctness",
            qa_type=QualityAssuranceType.VALIDATION,
            gate_type=QualityGateType.MODEL_INPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"forbidden_patterns": [r"<script>", r"javascript:", r"on\w+\s*="]}
            ],
            failure_message="Input contains potentially unsafe syntax patterns",
            success_message="Input syntax appears safe",
            validation_function=self._validate_syntax
        ))
        
        # Encoding validation rule
        self.add_rule(QualityAssuranceRule(
            rule_id="input_encoding_check",
            name="Input Encoding Validation",
            description="Validates character encoding",
            qa_type=QualityAssuranceType.VALIDATION,
            gate_type=QualityGateType.MODEL_INPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"valid_encodings": ["utf-8", "ascii"]}
            ],
            failure_message="Input contains invalid character encoding",
            success_message="Input encoding is valid",
            validation_function=self._validate_encoding
        ))
    
    def validate(self, content: str, context: Dict[str, Any]) -> QualityAssuranceResult:
        """Validate input content"""
        start_time = time.time()
        
        issues = []
        metrics = {
            "content_length": len(content),
            "content_words": len(content.split()),
            "content_lines": len(content.split('\n'))
        }
        
        # Apply all rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                rule_issues = self._apply_rule(rule, content, context)
                issues.extend(rule_issues)
            except Exception as e:
                issues.append({
                    "rule_id": rule.rule_id,
                    "severity": QualityIssueSeverity.ERROR.value,
                    "message": f"Error applying rule {rule.rule_id}: {str(e)}",
                    "details": {"error": str(e)}
                })
        
        # Determine overall status
        status = QualityAssuranceStatus.PASSED
        blocker_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.BLOCKER.value]
        error_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.ERROR.value]
        
        if blocker_issues:
            status = QualityAssuranceStatus.BLOCKER
        elif error_issues:
            status = QualityAssuranceStatus.FAILED
        
        execution_time = time.time() - start_time
        
        result = QualityAssuranceResult(
            rule_id="input_validation",
            rule_name="Input Validation",
            status=status,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            context=context,
            remediation_available=bool(issues),
            remediation_suggested=self._suggest_remediations(issues)
        )
        
        self._record_result(result)
        return result
    
    def _apply_rule(self, rule: QualityAssuranceRule, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a validation rule"""
        if rule.validation_function:
            return rule.validation_function(content, rule.conditions, context)
        else:
            # Default validation - check if content is not empty
            if not content.strip():
                return [{
                    "rule_id": rule.rule_id,
                    "severity": rule.severity.value,
                    "message": rule.failure_message or "Content is empty",
                    "details": {"content_length": len(content)}
                }]
            else:
                return []  # No issues
    
    def _validate_length(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate content length"""
        issues = []
        
        for condition in conditions:
            min_length = condition.get("min_length", 0)
            max_length = condition.get("max_length", float('inf'))
            
            content_length = len(content)
            
            if content_length < min_length:
                issues.append({
                    "rule_id": "input_length_check",
                    "severity": QualityIssueSeverity.ERROR.value,
                    "message": f"Content length {content_length} is below minimum {min_length}",
                    "details": {
                        "actual_length": content_length,
                        "min_length": min_length,
                        "max_length": max_length
                    }
                })
            elif content_length > max_length:
                issues.append({
                    "rule_id": "input_length_check",
                    "severity": QualityIssueSeverity.ERROR.value,
                    "message": f"Content length {content_length} exceeds maximum {max_length}",
                    "details": {
                        "actual_length": content_length,
                        "min_length": min_length,
                        "max_length": max_length
                    }
                })
        
        return issues
    
    def _validate_syntax(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate content syntax"""
        issues = []
        
        for condition in conditions:
            forbidden_patterns = condition.get("forbidden_patterns", [])
            
            for pattern in forbidden_patterns:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            "rule_id": "input_syntax_check",
                            "severity": QualityIssueSeverity.WARNING.value,
                            "message": f"Found potentially unsafe pattern: {pattern}",
                            "details": {
                                "pattern": pattern,
                                "matches": matches[:5],  # Limit to first 5 matches
                                "match_count": len(matches)
                            }
                        })
                except re.error as e:
                    issues.append({
                        "rule_id": "input_syntax_check",
                        "severity": QualityIssueSeverity.ERROR.value,
                        "message": f"Invalid regex pattern: {pattern}",
                        "details": {"error": str(e)}
                    })
        
        return issues
    
    def _validate_encoding(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate content encoding"""
        issues = []
        
        # Check for valid UTF-8 encoding
        try:
            content.encode('utf-8').decode('utf-8')
        except UnicodeError as e:
            issues.append({
                "rule_id": "input_encoding_check",
                "severity": QualityIssueSeverity.WARNING.value,
                "message": "Content contains invalid UTF-8 encoding",
                "details": {"error": str(e)}
            })
        
        return issues
    
    def _suggest_remediations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Suggest remediations for validation issues"""
        remediations = []
        
        for issue in issues:
            severity = issue.get("severity")
            message = issue.get("message", "")
            
            if severity == QualityIssueSeverity.BLOCKER.value:
                remediations.append("Content must be corrected to remove blocking issues before proceeding")
            elif severity == QualityIssueSeverity.ERROR.value:
                remediations.append("Review and correct the identified errors in the content")
            elif severity == QualityIssueSeverity.WARNING.value:
                remediations.append("Consider reviewing the warnings to improve content quality")
            elif "length" in message.lower():
                remediations.append("Adjust content length to meet specified limits")
            elif "syntax" in message.lower() or "pattern" in message.lower():
                remediations.append("Review content for unsafe syntax patterns and remove them")
            elif "encoding" in message.lower():
                remediations.append("Ensure content uses valid UTF-8 encoding")
        
        return list(set(remediations))  # Remove duplicates

class OutputValidationMechanism(QualityAssuranceMechanism):
    """Output validation quality assurance mechanism"""
    
    def __init__(self):
        super().__init__("Output Validation", "Validates model output for quality and safety")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default validation rules"""
        # Quality scoring rule
        self.add_rule(QualityAssuranceRule(
            rule_id="output_quality_score",
            name="Output Quality Scoring",
            description="Evaluates overall quality of model output",
            qa_type=QualityAssuranceType.VERIFICATION,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"min_quality_score": 75.0}
            ],
            failure_message="Output quality score is below threshold",
            success_message="Output quality score meets requirements",
            validation_function=self._validate_quality_score
        ))
        
        # Completeness validation rule
        self.add_rule(QualityAssuranceRule(
            rule_id="output_completeness_check",
            name="Output Completeness Validation",
            description="Ensures output is complete and comprehensive",
            qa_type=QualityAssuranceType.VERIFICATION,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"required_sections": ["introduction", "body", "conclusion"]}
            ],
            failure_message="Output is missing required sections",
            success_message="Output contains all required sections",
            validation_function=self._validate_completeness
        ))
        
        # Consistency validation rule
        self.add_rule(QualityAssuranceRule(
            rule_id="output_consistency_check",
            name="Output Consistency Validation",
            description="Checks for consistency in output formatting and terminology",
            qa_type=QualityAssuranceType.VERIFICATION,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"consistency_threshold": 0.8}
            ],
            failure_message="Output consistency is below threshold",
            success_message="Output consistency meets requirements",
            validation_function=self._validate_consistency
        ))
    
    def validate(self, content: str, context: Dict[str, Any]) -> QualityAssuranceResult:
        """Validate output content"""
        start_time = time.time()
        
        issues = []
        metrics = {
            "content_length": len(content),
            "content_words": len(content.split()),
            "content_lines": len(content.split('\n')),
            "quality_score": 0.0,
            "completeness_score": 0.0,
            "consistency_score": 0.0
        }
        
        # Apply all rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                rule_issues = self._apply_rule(rule, content, context)
                issues.extend(rule_issues)
                
                # Update metrics based on rule results
                if rule.rule_id == "output_quality_score":
                    metrics["quality_score"] = self._calculate_quality_score(content)
                elif rule.rule_id == "output_completeness_check":
                    metrics["completeness_score"] = self._calculate_completeness_score(content)
                elif rule.rule_id == "output_consistency_check":
                    metrics["consistency_score"] = self._calculate_consistency_score(content)
                    
            except Exception as e:
                issues.append({
                    "rule_id": rule.rule_id,
                    "severity": QualityIssueSeverity.ERROR.value,
                    "message": f"Error applying rule {rule.rule_id}: {str(e)}",
                    "details": {"error": str(e)}
                })
        
        # Determine overall status
        status = QualityAssuranceStatus.PASSED
        warning_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.WARNING.value]
        error_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.ERROR.value]
        
        if error_issues:
            status = QualityAssuranceStatus.FAILED
        elif warning_issues:
            status = QualityAssuranceStatus.PASSED  # Warnings don't fail the check
        
        execution_time = time.time() - start_time
        
        result = QualityAssuranceResult(
            rule_id="output_validation",
            rule_name="Output Validation",
            status=status,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            context=context,
            remediation_available=bool(issues),
            remediation_suggested=self._suggest_remediations(issues)
        )
        
        self._record_result(result)
        return result
    
    def _apply_rule(self, rule: QualityAssuranceRule, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a validation rule"""
        if rule.validation_function:
            return rule.validation_function(content, rule.conditions, context)
        else:
            # Default validation - check if content is not empty
            if not content.strip():
                return [{
                    "rule_id": rule.rule_id,
                    "severity": rule.severity.value,
                    "message": rule.failure_message or "Content is empty",
                    "details": {"content_length": len(content)}
                }]
            else:
                return []  # No issues
    
    def _validate_quality_score(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate output quality score"""
        issues = []
        
        quality_score = self._calculate_quality_score(content)
        
        for condition in conditions:
            min_quality_score = condition.get("min_quality_score", 0.0)
            
            if quality_score < min_quality_score:
                issues.append({
                    "rule_id": "output_quality_score",
                    "severity": QualityIssueSeverity.WARNING.value,
                    "message": f"Output quality score {quality_score:.2f} is below threshold {min_quality_score}",
                    "details": {
                        "actual_score": quality_score,
                        "threshold": min_quality_score,
                        "score_components": self._analyze_quality_components(content)
                    }
                })
        
        return issues
    
    def _calculate_quality_score(self, content: str) -> float:
        """Calculate a quality score for the content"""
        # Simple heuristic-based scoring
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Base score
        score = 50.0
        
        # Adjust based on sentence length (ideal is 15-25 words per sentence)
        if 15 <= avg_sentence_length <= 25:
            score += 20
        elif 10 <= avg_sentence_length <= 30:
            score += 10
        else:
            score -= 10  # Too short or too long sentences
        
        # Bonus for structure
        if re.search(r'^#+\s', content, re.MULTILINE):
            score += 10  # Has headers
        if re.search(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE):
            score += 10  # Has lists
        
        # Penalty for very short content
        if word_count < 50:
            score -= 20
        
        return max(0, min(100, score))
    
    def _analyze_quality_components(self, content: str) -> Dict[str, float]:
        """Analyze components contributing to quality score"""
        word_count = len(content.split())
        sentence_count = len(re.split(r'[.!?]+', content))
        avg_sentence_length = word_count / max(1, sentence_count)
        
        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "avg_sentence_length": avg_sentence_length,
            "has_headers": bool(re.search(r'^#+\s', content, re.MULTILINE)),
            "has_lists": bool(re.search(r'^\s*[-*]\s|^[\d]+\.\s', content, re.MULTILINE))
        }
    
    def _validate_completeness(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate output completeness"""
        issues = []
        
        completeness_score = self._calculate_completeness_score(content)
        
        for condition in conditions:
            required_sections = condition.get("required_sections", [])
            missing_sections = []
            
            for section in required_sections:
                if not re.search(rf'\b{section}\b', content, re.IGNORECASE):
                    missing_sections.append(section)
            
            if missing_sections:
                issues.append({
                    "rule_id": "output_completeness_check",
                    "severity": QualityIssueSeverity.WARNING.value,
                    "message": f"Output is missing required sections: {', '.join(missing_sections)}",
                    "details": {
                        "missing_sections": missing_sections,
                        "found_sections": [sec for sec in required_sections if re.search(rf'\b{sec}\b', content, re.IGNORECASE)],
                        "completeness_score": completeness_score
                    }
                })
        
        return issues
    
    def _calculate_completeness_score(self, content: str) -> float:
        """Calculate completeness score"""
        # Check for essential sections
        has_introduction = bool(re.search(r'\b(intro|overview|summary)\b', content, re.IGNORECASE))
        has_body = bool(re.search(r'\b(body|main|content)\b', content, re.IGNORECASE))
        has_conclusion = bool(re.search(r'\b(conclusion|summary|end)\b', content, re.IGNORECASE))
        
        # Simple scoring
        score = 0
        if has_introduction:
            score += 30
        if has_body:
            score += 40
        if has_conclusion:
            score += 30
        
        return score
    
    def _validate_consistency(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate output consistency"""
        issues = []
        
        consistency_score = self._calculate_consistency_score(content)
        
        for condition in conditions:
            consistency_threshold = condition.get("consistency_threshold", 0.0)
            
            if consistency_score < consistency_threshold:
                issues.append({
                    "rule_id": "output_consistency_check",
                    "severity": QualityIssueSeverity.WARNING.value,
                    "message": f"Output consistency score {consistency_score:.2f} is below threshold {consistency_threshold}",
                    "details": {
                        "actual_score": consistency_score,
                        "threshold": consistency_threshold,
                        "consistency_metrics": self._analyze_consistency_metrics(content)
                    }
                })
        
        return issues
    
    def _calculate_consistency_score(self, content: str) -> float:
        """Calculate consistency score"""
        # Simple consistency check based on formatting
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return 0.0
        
        # Check indentation consistency
        indented_lines = [line for line in non_empty_lines if line.startswith(('  ', '\t'))]
        indentation_consistency = len(indented_lines) / len(non_empty_lines)
        
        # Check header consistency
        headers = re.findall(r'^#+\s', content, re.MULTILINE)
        header_consistency = 1.0 if headers else 0.0  # Simplified
        
        # Combined score
        score = (indentation_consistency * 0.7 + header_consistency * 0.3) * 100
        
        return max(0, min(100, score))
    
    def _analyze_consistency_metrics(self, content: str) -> Dict[str, Any]:
        """Analyze metrics contributing to consistency score"""
        lines = content.split('\n')
        non_empty_lines = [line for line in lines if line.strip()]
        
        if not non_empty_lines:
            return {"indentation_consistency": 0.0, "header_consistency": 0.0}
        
        # Indentation analysis
        indented_lines = [line for line in non_empty_lines if line.startswith(('  ', '\t'))]
        indentation_consistency = len(indented_lines) / len(non_empty_lines)
        
        # Header analysis
        headers = re.findall(r'^#+\s', content, re.MULTILINE)
        header_consistency = 1.0 if headers else 0.0
        
        return {
            "indentation_consistency": indentation_consistency,
            "header_consistency": header_consistency,
            "total_lines": len(lines),
            "non_empty_lines": len(non_empty_lines),
            "indented_lines": len(indented_lines),
            "headers": len(headers)
        }
    
    def _suggest_remediations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Suggest remediations for validation issues"""
        remediations = []
        
        for issue in issues:
            rule_id = issue.get("rule_id", "")
            message = issue.get("message", "")
            
            if "quality" in rule_id.lower():
                remediations.append("Improve content structure with clear headers and lists")
                remediations.append("Ensure sentences are of appropriate length (15-25 words ideal)")
                remediations.append("Add more detailed content to meet minimum length requirements")
            elif "completeness" in rule_id.lower():
                missing_sections = issue.get("details", {}).get("missing_sections", [])
                if missing_sections:
                    remediations.append(f"Add missing sections: {', '.join(missing_sections)}")
                remediations.append("Ensure all required content sections are present")
            elif "consistency" in rule_id.lower():
                remediations.append("Standardize formatting and indentation throughout the content")
                remediations.append("Use consistent heading styles and numbering")
                remediations.append("Maintain uniform spacing and structure")
            elif "score" in message.lower():
                remediations.append("Review overall content quality and make improvements")
        
        return list(set(remediations))

class SecurityValidationMechanism(QualityAssuranceMechanism):
    """Security validation quality assurance mechanism"""
    
    def __init__(self):
        super().__init__("Security Validation", "Validates content for security vulnerabilities")
        self._initialize_default_rules()
    
    def _initialize_default_rules(self):
        """Initialize default security validation rules"""
        # Injection attack prevention rule
        self.add_rule(QualityAssuranceRule(
            rule_id="security_injection_prevention",
            name="Injection Attack Prevention",
            description="Prevents common injection attack vectors",
            qa_type=QualityAssuranceType.SECURITY,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.BLOCKER,
            conditions=[
                {"forbidden_patterns": [
                    r"<script[^>]*>.*?</script>",
                    r"javascript:",
                    r"vbscript:",
                    r"on\w+\s*=",
                    r"eval\s*\(",
                    r"exec\s*\(",
                    r"SELECT\s+.*\s+FROM",
                    r"INSERT\s+INTO",
                    r"UPDATE\s+.*\s+SET",
                    r"DELETE\s+FROM"
                ]}
            ],
            failure_message="Content contains potential injection attack vectors",
            success_message="Content is free from common injection attack vectors",
            validation_function=self._validate_injection_prevention
        ))
        
        # Credential exposure prevention rule
        self.add_rule(QualityAssuranceRule(
            rule_id="security_credential_exposure",
            name="Credential Exposure Prevention",
            description="Prevents accidental exposure of credentials",
            qa_type=QualityAssuranceType.SECURITY,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.BLOCKER,
            conditions=[
                {"credential_patterns": [
                    r"password\s*[:=]\s*[\'\"][^\'\"]{4,}[\'\"]",
                    r"api[_-]?key\s*[:=]\s*[\'\"][^\'\"]{8,}[\'\"]",
                    r"secret\s*[:=]\s*[\'\"][^\'\"]{8,}[\'\"]",
                    r"token\s*[:=]\s*[\'\"][^\'\"]{10,}[\'\"]"
                ]}
            ],
            failure_message="Content contains potential credential exposure",
            success_message="Content is free from credential exposure",
            validation_function=self._validate_credential_exposure
        ))
        
        # PII detection rule
        self.add_rule(QualityAssuranceRule(
            rule_id="security_pii_detection",
            name="PII Detection",
            description="Detects potential personally identifiable information",
            qa_type=QualityAssuranceType.SECURITY,
            gate_type=QualityGateType.MODEL_OUTPUT,
            severity=QualityIssueSeverity.WARNING,
            conditions=[
                {"pii_patterns": [
                    r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                    r"\b\d{16}\b",             # Credit card
                    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
                    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"  # Phone number
                ]}
            ],
            failure_message="Content contains potential personally identifiable information",
            success_message="Content appears free from PII",
            validation_function=self._validate_pii_detection
        ))
    
    def validate(self, content: str, context: Dict[str, Any]) -> QualityAssuranceResult:
        """Validate content for security issues"""
        start_time = time.time()
        
        issues = []
        metrics = {
            "content_length": len(content),
            "injection_patterns_found": 0,
            "credential_patterns_found": 0,
            "pii_patterns_found": 0,
            "security_score": 100.0
        }
        
        # Apply all security rules
        for rule in self.rules.values():
            if not rule.enabled:
                continue
            
            try:
                rule_issues = self._apply_rule(rule, content, context)
                issues.extend(rule_issues)
                
                # Update metrics
                if rule.rule_id == "security_injection_prevention":
                    metrics["injection_patterns_found"] += len(rule_issues)
                elif rule.rule_id == "security_credential_exposure":
                    metrics["credential_patterns_found"] += len(rule_issues)
                elif rule.rule_id == "security_pii_detection":
                    metrics["pii_patterns_found"] += len(rule_issues)
                    
            except Exception as e:
                issues.append({
                    "rule_id": rule.rule_id,
                    "severity": QualityIssueSeverity.ERROR.value,
                    "message": f"Error applying security rule {rule.rule_id}: {str(e)}",
                    "details": {"error": str(e)}
                })
        
        # Calculate security score
        total_patterns = (
            metrics["injection_patterns_found"] +
            metrics["credential_patterns_found"] +
            metrics["pii_patterns_found"]
        )
        metrics["security_score"] = max(0, 100 - (total_patterns * 10))
        
        # Determine overall status
        status = QualityAssuranceStatus.PASSED
        blocker_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.BLOCKER.value]
        warning_issues = [issue for issue in issues if issue.get("severity") == QualityIssueSeverity.WARNING.value]
        
        if blocker_issues:
            status = QualityAssuranceStatus.BLOCKER
        elif warning_issues:
            status = QualityAssuranceStatus.PASSED  # Warnings don't block, but are noted
        
        execution_time = time.time() - start_time
        
        result = QualityAssuranceResult(
            rule_id="security_validation",
            rule_name="Security Validation",
            status=status,
            issues=issues,
            metrics=metrics,
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            context=context,
            remediation_available=bool(blocker_issues),
            remediation_suggested=self._suggest_remediations(issues)
        )
        
        self._record_result(result)
        return result
    
    def _apply_rule(self, rule: QualityAssuranceRule, content: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply a security validation rule"""
        if rule.validation_function:
            return rule.validation_function(content, rule.conditions, context)
        else:
            # Default validation - check if content is not empty
            if not content.strip():
                return [{
                    "rule_id": rule.rule_id,
                    "severity": rule.severity.value,
                    "message": rule.failure_message or "Content is empty",
                    "details": {"content_length": len(content)}
                }]
            else:
                return []  # No issues
    
    def _validate_injection_prevention(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate for injection attack prevention"""
        issues = []
        
        for condition in conditions:
            forbidden_patterns = condition.get("forbidden_patterns", [])
            
            for pattern in forbidden_patterns:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        issues.append({
                            "rule_id": "security_injection_prevention",
                            "severity": QualityIssueSeverity.BLOCKER.value,
                            "message": f"Potential injection attack vector detected: {pattern}",
                            "details": {
                                "pattern": pattern,
                                "matches": matches[:3],  # Limit to first 3 matches
                                "match_count": len(matches)
                            }
                        })
                except re.error as e:
                    issues.append({
                        "rule_id": "security_injection_prevention",
                        "severity": QualityIssueSeverity.ERROR.value,
                        "message": f"Invalid regex pattern in security rule: {pattern}",
                        "details": {"error": str(e)}
                    })
        
        return issues
    
    def _validate_credential_exposure(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate for credential exposure prevention"""
        issues = []
        
        for condition in conditions:
            credential_patterns = condition.get("credential_patterns", [])
            
            for pattern in credential_patterns:
                try:
                    matches = re.findall(pattern, content, re.IGNORECASE)
                    if matches:
                        issues.append({
                            "rule_id": "security_credential_exposure",
                            "severity": QualityIssueSeverity.BLOCKER.value,
                            "message": f"Potential credential exposure detected: {pattern}",
                            "details": {
                                "pattern": pattern,
                                "matches": matches[:3],
                                "match_count": len(matches)
                            }
                        })
                except re.error as e:
                    issues.append({
                        "rule_id": "security_credential_exposure",
                        "severity": QualityIssueSeverity.ERROR.value,
                        "message": f"Invalid regex pattern in security rule: {pattern}",
                        "details": {"error": str(e)}
                    })
        
        return issues
    
    def _validate_pii_detection(self, content: str, conditions: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate for PII detection"""
        issues = []
        
        for condition in conditions:
            pii_patterns = condition.get("pii_patterns", [])
            
            for pattern in pii_patterns:
                try:
                    matches = re.findall(pattern, content)
                    if matches:
                        # Only flag if matches seem legitimate (not just random numbers)
                        valid_matches = [match for match in matches if self._is_plausible_pii(match)]
                        if valid_matches:
                            issues.append({
                                "rule_id": "security_pii_detection",
                                "severity": QualityIssueSeverity.WARNING.value,
                                "message": f"Potential PII detected: {pattern}",
                                "details": {
                                    "pattern": pattern,
                                    "matches": valid_matches[:3],
                                    "match_count": len(valid_matches)
                                }
                            })
                except re.error as e:
                    issues.append({
                        "rule_id": "security_pii_detection",
                        "severity": QualityIssueSeverity.ERROR.value,
                        "message": f"Invalid regex pattern in security rule: {pattern}",
                        "details": {"error": str(e)}
                    })
        
        return issues
    
    def _is_plausible_pii(self, match: str) -> bool:
        """Determine if a match is plausibly PII"""
        # Simple heuristic to avoid false positives
        # This would be more sophisticated in a real implementation
        
        # For SSN-like patterns
        if re.match(r"\b\d{3}-\d{2}-\d{4}\b", match):
            # Check if it looks like a real SSN (not 000-00-0000)
            parts = match.split('-')
            if parts[0] == "000" or parts[1] == "00" or parts[2] == "0000":
                return False
            return True
        
        # For email-like patterns
        if "@" in match and "." in match:
            return True
        
        # For phone-like patterns
        if re.match(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", match):
            # Check if it has reasonable digit distribution
            digits_only = re.sub(r"[^\d]", "", match)
            if len(digits_only) == 10:
                return True
        
        # For credit card-like patterns
        if re.match(r"\b\d{16}\b", match):
            # Simple Luhn algorithm check
            return self._luhn_check(match)
        
        return False
    
    def _luhn_check(self, card_number: str) -> bool:
        """Simple Luhn algorithm check for credit card numbers"""
        def digits_of(n):
            return [int(d) for d in str(n)]
        
        digits = digits_of(card_number)
        odd_digits = digits[-1::-2]
        even_digits = digits[-2::-2]
        checksum = sum(odd_digits)
        for d in even_digits:
            checksum += sum(digits_of(d*2))
        return checksum % 10 == 0
    
    def _suggest_remediations(self, issues: List[Dict[str, Any]]) -> List[str]:
        """Suggest remediations for security issues"""
        remediations = []
        
        for issue in issues:
            severity = issue.get("severity")
            rule_id = issue.get("rule_id", "")
            
            if severity == QualityIssueSeverity.BLOCKER.value:
                remediations.append("Content must be sanitized to remove security vulnerabilities before proceeding")
                
                if "injection" in rule_id.lower():
                    remediations.append("Remove or sanitize all script tags, JavaScript, and database query patterns")
                elif "credential" in rule_id.lower():
                    remediations.append("Remove all hardcoded credentials and use secure configuration management")
                elif "pii" in rule_id.lower():
                    remediations.append("Remove or anonymize all personally identifiable information")
            
            elif severity == QualityIssueSeverity.WARNING.value:
                if "pii" in rule_id.lower():
                    remediations.append("Review content for PII and consider anonymization if necessary")
        
        return list(set(remediations))

class QualityGate:
    """Quality gate that aggregates multiple QA mechanisms"""
    
    def __init__(self, gate_type: QualityGateType, name: str, description: str):
        self.gate_type = gate_type
        self.name = name
        self.description = description
        self.mechanisms: List[QualityAssuranceMechanism] = []
        self.required_mechanisms: List[str] = []
        self.blocking_severities: List[QualityIssueSeverity] = [
            QualityIssueSeverity.BLOCKER,
            QualityIssueSeverity.CRITICAL
        ]
        self.results_history: List[QualityGateResult] = []
        self.enabled = True
    
    def add_mechanism(self, mechanism: QualityAssuranceMechanism, required: bool = False):
        """Add a QA mechanism to this gate"""
        self.mechanisms.append(mechanism)
        if required:
            self.required_mechanisms.append(mechanism.name)
        logger.info(f"Added QA mechanism {mechanism.name} to gate {self.name}")
    
    def remove_mechanism(self, mechanism_name: str) -> bool:
        """Remove a QA mechanism from this gate"""
        for i, mechanism in enumerate(self.mechanisms):
            if mechanism.name == mechanism_name:
                del self.mechanisms[i]
                if mechanism_name in self.required_mechanisms:
                    self.required_mechanisms.remove(mechanism_name)
                logger.info(f"Removed QA mechanism {mechanism_name} from gate {self.name}")
                return True
        return False
    
    def validate(self, content: str, context: Dict[str, Any]) -> QualityGateResult:
        """Validate content through all mechanisms in this gate"""
        if not self.enabled:
            return QualityGateResult(
                gate_type=self.gate_type,
                overall_status=QualityAssuranceStatus.SKIPPED,
                results=[],
                passed_count=0,
                failed_count=0,
                total_checks=0,
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
                context=context,
                blocking_issues=[],
                recommendations=[]
            )
        
        start_time = time.time()
        
        results = []
        blocking_issues = []
        all_issues = []
        
        # Apply all mechanisms
        for mechanism in self.mechanisms:
            try:
                result = mechanism.validate(content, context)
                results.append(result)
                all_issues.extend(result.issues)
                
                # Check for blocking issues
                for issue in result.issues:
                    severity = QualityIssueSeverity(issue.get("severity", "info"))
                    if severity in self.blocking_severities:
                        blocking_issues.append(issue)
                        
            except Exception as e:
                # Create error result
                error_result = QualityAssuranceResult(
                    rule_id=f"{mechanism.name}_error",
                    rule_name=f"{mechanism.name} Error",
                    status=QualityAssuranceStatus.ERROR,
                    issues=[{
                        "rule_id": f"{mechanism.name}_error",
                        "severity": QualityIssueSeverity.ERROR.value,
                        "message": f"Error in {mechanism.name}: {str(e)}",
                        "details": {"error": str(e)}
                    }],
                    metrics={},
                    execution_time=0.0,
                    timestamp=datetime.now().isoformat(),
                    context=context
                )
                results.append(error_result)
                logger.error(f"Error in QA mechanism {mechanism.name}: {e}")
        
        # Determine overall status
        overall_status = QualityAssuranceStatus.PASSED
        passed_count = 0
        failed_count = 0
        
        for result in results:
            if result.status == QualityAssuranceStatus.PASSED:
                passed_count += 1
            elif result.status in [QualityAssuranceStatus.FAILED, QualityAssuranceStatus.BLOCKER, QualityAssuranceStatus.ERROR]:
                failed_count += 1
                if result.status in [QualityAssuranceStatus.BLOCKER, QualityAssuranceStatus.ERROR]:
                    overall_status = QualityAssuranceStatus.BLOCKER if result.status == QualityAssuranceStatus.BLOCKER else QualityAssuranceStatus.ERROR
                elif overall_status == QualityAssuranceStatus.PASSED and result.status == QualityAssuranceStatus.FAILED:
                    overall_status = QualityAssuranceStatus.FAILED
        
        # If there are blocking issues, override status
        if blocking_issues:
            overall_status = QualityAssuranceStatus.BLOCKER
        
        execution_time = time.time() - start_time
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results, blocking_issues)
        
        result = QualityGateResult(
            gate_type=self.gate_type,
            overall_status=overall_status,
            results=results,
            passed_count=passed_count,
            failed_count=failed_count,
            total_checks=len(results),
            execution_time=execution_time,
            timestamp=datetime.now().isoformat(),
            context=context,
            blocking_issues=blocking_issues,
            recommendations=recommendations
        )
        
        self.results_history.append(result)
        
        # Keep history within reasonable limits
        if len(self.results_history) > 100:
            self.results_history = self.results_history[-50:]
        
        return result
    
    def _generate_recommendations(self, results: List[QualityAssuranceResult], 
                                 blocking_issues: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Add recommendations from individual results
        for result in results:
            if result.remediation_suggested:
                recommendations.extend(result.remediation_suggested)
        
        # Add specific recommendations for blocking issues
        for issue in blocking_issues:
            severity = issue.get("severity")
            message = issue.get("message", "")
            
            if severity == QualityIssueSeverity.BLOCKER.value:
                recommendations.append("Content must be corrected to remove blocking issues before proceeding")
            elif "injection" in message.lower():
                recommendations.append("Remove or sanitize all script tags and JavaScript code")
            elif "credential" in message.lower():
                recommendations.append("Remove all hardcoded credentials")
            elif "pii" in message.lower():
                recommendations.append("Review and anonymize personally identifiable information")
        
        # Remove duplicates and return
        return list(set(recommendations))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get gate statistics"""
        if not self.results_history:
            return {"total_executions": 0}
        
        total_executions = len(self.results_history)
        passed_count = sum(1 for result in self.results_history if result.overall_status == QualityAssuranceStatus.PASSED)
        failed_count = sum(1 for result in self.results_history if result.overall_status == QualityAssuranceStatus.FAILED)
        blocked_count = sum(1 for result in self.results_history if result.overall_status == QualityAssuranceStatus.BLOCKER)
        
        avg_execution_time = statistics.mean([result.execution_time for result in self.results_history])
        
        return {
            "total_executions": total_executions,
            "passed_count": passed_count,
            "failed_count": failed_count,
            "blocked_count": blocked_count,
            "pass_rate": passed_count / max(1, total_executions),
            "average_execution_time": avg_execution_time
        }

class QualityAssuranceOrchestrator:
    """Orchestrates quality assurance across multiple gates and mechanisms"""
    
    def __init__(self):
        self.gates: Dict[QualityGateType, QualityGate] = {}
        self.mechanisms: Dict[str, QualityAssuranceMechanism] = {}
        self.validation_chain: List[QualityGateType] = []
        self.enabled = True
        self._initialize_default_gates()
        self._initialize_default_mechanisms()
    
    def _initialize_default_gates(self):
        """Initialize default quality gates"""
        # Input validation gate
        input_gate = QualityGate(
            gate_type=QualityGateType.MODEL_INPUT,
            name="Model Input Validation Gate",
            description="Validates input content before model processing"
        )
        self.gates[QualityGateType.MODEL_INPUT] = input_gate
        
        # Output validation gate
        output_gate = QualityGate(
            gate_type=QualityGateType.MODEL_OUTPUT,
            name="Model Output Validation Gate",
            description="Validates model output for quality and safety"
        )
        self.gates[QualityGateType.MODEL_OUTPUT] = output_gate
        
        # Final approval gate
        final_gate = QualityGate(
            gate_type=QualityGateType.FINAL_APPROVAL,
            name="Final Approval Gate",
            description="Final quality check before content release"
        )
        self.gates[QualityGateType.FINAL_APPROVAL] = final_gate
    
    def _initialize_default_mechanisms(self):
        """Initialize default QA mechanisms"""
        # Input validation mechanism
        input_validator = InputValidationMechanism()
        self.mechanisms[input_validator.name] = input_validator
        
        # Output validation mechanism
        output_validator = OutputValidationMechanism()
        self.mechanisms[output_validator.name] = output_validator
        
        # Security validation mechanism
        security_validator = SecurityValidationMechanism()
        self.mechanisms[security_validator.name] = security_validator
        
        # Add mechanisms to appropriate gates
        self.gates[QualityGateType.MODEL_INPUT].add_mechanism(input_validator, required=True)
        self.gates[QualityGateType.MODEL_OUTPUT].add_mechanism(output_validator, required=True)
        self.gates[QualityGateType.MODEL_OUTPUT].add_mechanism(security_validator, required=True)
        self.gates[QualityGateType.FINAL_APPROVAL].add_mechanism(security_validator, required=True)
    
    def add_gate(self, gate: QualityGate):
        """Add a quality gate"""
        self.gates[gate.gate_type] = gate
        logger.info(f"Added quality gate: {gate.name}")
    
    def remove_gate(self, gate_type: QualityGateType) -> bool:
        """Remove a quality gate"""
        if gate_type in self.gates:
            del self.gates[gate_type]
            logger.info(f"Removed quality gate: {gate_type.value}")
            return True
        return False
    
    def get_gate(self, gate_type: QualityGateType) -> Optional[QualityGate]:
        """Get a quality gate by type"""
        return self.gates.get(gate_type)
    
    def add_mechanism(self, mechanism: QualityAssuranceMechanism):
        """Add a QA mechanism"""
        self.mechanisms[mechanism.name] = mechanism
        logger.info(f"Added QA mechanism: {mechanism.name}")
    
    def remove_mechanism(self, mechanism_name: str) -> bool:
        """Remove a QA mechanism"""
        if mechanism_name in self.mechanisms:
            del self.mechanisms[mechanism_name]
            logger.info(f"Removed QA mechanism: {mechanism_name}")
            return True
        return False
    
    def get_mechanism(self, mechanism_name: str) -> Optional[QualityAssuranceMechanism]:
        """Get a QA mechanism by name"""
        return self.mechanisms.get(mechanism_name)
    
    def validate_through_gate(self, gate_type: QualityGateType, content: str, 
                             context: Dict[str, Any]) -> QualityGateResult:
        """Validate content through a specific quality gate"""
        if not self.enabled:
            return QualityGateResult(
                gate_type=gate_type,
                overall_status=QualityAssuranceStatus.SKIPPED,
                results=[],
                passed_count=0,
                failed_count=0,
                total_checks=0,
                execution_time=0.0,
                timestamp=datetime.now().isoformat(),
                context=context,
                blocking_issues=[],
                recommendations=[]
            )
        
        gate = self.gates.get(gate_type)
        if not gate:
            raise ValueError(f"Quality gate not found: {gate_type.value}")
        
        return gate.validate(content, context)
    
    def validate_through_chain(self, content: str, context: Dict[str, Any],
                              gate_chain: Optional[List[QualityGateType]] = None) -> List[QualityGateResult]:
        """Validate content through a chain of quality gates"""
        if not self.enabled:
            return []
        
        chain = gate_chain or self.validation_chain or list(self.gates.keys())
        results = []
        
        for gate_type in chain:
            gate_result = self.validate_through_gate(gate_type, content, context)
            results.append(gate_result)
            
            # Stop chain if gate fails
            if gate_result.overall_status in [QualityAssuranceStatus.BLOCKER, QualityAssuranceStatus.ERROR]:
                logger.warning(f"Validation chain stopped at gate {gate_type.value} due to blocking issues")
                break
        
        return results
    
    def set_validation_chain(self, chain: List[QualityGateType]):
        """Set the validation chain order"""
        self.validation_chain = chain
        logger.info(f"Set validation chain order: {[gt.value for gt in chain]}")
    
    def get_gate_statistics(self, gate_type: QualityGateType) -> Dict[str, Any]:
        """Get statistics for a specific gate"""
        gate = self.gates.get(gate_type)
        if gate:
            return gate.get_statistics()
        return {"error": f"Gate not found: {gate_type.value}"}
    
    def get_overall_statistics(self) -> Dict[str, Any]:
        """Get overall QA statistics"""
        stats = {}
        for gate_type, gate in self.gates.items():
            stats[gate_type.value] = gate.get_statistics()
        return stats

# Example usage and testing
def test_quality_assurance_system():
    """Test function for the Quality Assurance System"""
    print("Quality Assurance Mechanisms Test:")
    
    # Create QA orchestrator
    qa_orchestrator = QualityAssuranceOrchestrator()
    
    print(f"Initialized with {len(qa_orchestrator.gates)} gates and {len(qa_orchestrator.mechanisms)} mechanisms")
    
    # Test input validation
    print("\nTesting input validation:")
    sample_input = "<script>alert('XSS')</script> Please review this document."
    
    input_result = qa_orchestrator.validate_through_gate(
        QualityGateType.MODEL_INPUT,
        sample_input,
        {"content_type": "document", "source": "user_input"}
    )
    
    print(f"Input validation result: {input_result.overall_status.value}")
    print(f"Blocking issues found: {len(input_result.blocking_issues)}")
    if input_result.blocking_issues:
        for issue in input_result.blocking_issues[:3]:  # Show first 3
            print(f"  - {issue.get('message', 'No message')}")
    
    # Test output validation
    print("\nTesting output validation:")
    sample_output = """
# Sample Document

This is a sample document for testing the quality assurance system. 
It contains multiple sections and should demonstrate the system's capabilities.

## Introduction
The introduction provides context for the document.

## Main Content
The main content contains detailed information about the topic.

## Conclusion
The conclusion summarizes the key points.
"""
    
    output_result = qa_orchestrator.validate_through_gate(
        QualityGateType.MODEL_OUTPUT,
        sample_output,
        {"content_type": "document", "source": "model_output"}
    )
    
    print(f"Output validation result: {output_result.overall_status.value}")
    print(f"Issues found: {len(output_result.results[0].issues) if output_result.results else 0}")
    print(f"Quality metrics: {output_result.results[0].metrics if output_result.results else {}}")
    
    # Test security validation
    print("\nTesting security validation:")
    insecure_content = """
def authenticate_user(username, password="secret123"):
    # This is a vulnerable authentication function
    if username == "admin" and password == "password123":
        return True
    return False

api_key = "sk-1234567890abcdef1234567890abcdef"
"""
    
    security_result = qa_orchestrator.validate_through_gate(
        QualityGateType.MODEL_OUTPUT,
        insecure_content,
        {"content_type": "code", "source": "model_output"}
    )
    
    print(f"Security validation result: {security_result.overall_status.value}")
    print(f"Blocking issues found: {len(security_result.blocking_issues)}")
    if security_result.blocking_issues:
        for issue in security_result.blocking_issues[:3]:
            print(f"  - {issue.get('message', 'No message')}")
    
    # Test validation chain
    print("\nTesting validation chain:")
    chain_result = qa_orchestrator.validate_through_chain(
        sample_output,
        {"content_type": "document", "source": "model_output", "stage": "final_review"}
    )
    
    print(f"Chain validation completed with {len(chain_result)} gate results")
    for i, result in enumerate(chain_result):
        print(f"  Gate {i+1} ({result.gate_type.value}): {result.overall_status.value}")
    
    # Show statistics
    print("\nOverall statistics:")
    stats = qa_orchestrator.get_overall_statistics()
    for gate_name, gate_stats in stats.items():
        if "total_executions" in gate_stats:
            print(f"  {gate_name}: {gate_stats['total_executions']} executions, "
                  f"{gate_stats.get('passed_count', 0)} passed")
    
    # Test remediation suggestions
    print("\nRemediation suggestions:")
    if chain_result:
        for suggestion in chain_result[0].recommendations[:5]:  # Show first 5
            print(f"  - {suggestion}")
    
    return qa_orchestrator

if __name__ == "__main__":
    test_quality_assurance_system()