"""
Red Team (Assailants) Functionality for OpenEvolve
Implements the Red Team functionality as described in the Sovereign-Grade Decomposition Workflow.
The Red Team is responsible for criticism and flaw detection, acting as adversarial agents that 
actively seek vulnerabilities, inconsistencies, and weaknesses in generated content during the 
critique phase of the workflow.
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
import logging

# Configure logging first
logger = logging.getLogger(__name__)

from llm_utils import _request_openai_compatible_chat, _compose_messages
from content_analyzer import ContentAnalyzer

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig
    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    logger.warning("OpenEvolve backend not available - using fallback implementation")

# Import DTS integration for enhanced adversarial dialogue
try:
    from dts_integration import DTSIntegration, DTSIntegrationConfig
    DTS_AVAILABLE = True
    logger.info("DTS integration available for enhanced adversarial dialogue")
except (ImportError, Exception):
    DTS_AVAILABLE = False
    logger.warning("DTS integration not available - using standard adversarial methods")

from prompt_engineering import PromptEngineeringSystem
from model_orchestration import ModelOrchestrator, OrchestrationRequest, ModelTeam
from quality_assessment import QualityAssessmentEngine, SeverityLevel
from workflow_structures import Team, GauntletDefinition, CritiqueReport, SolutionAttempt, ModelConfig
from workflow_structures import Team, GauntletDefinition, CritiqueReport

class IssueCategory(Enum):
    """Categories of issues the red team can identify"""
    LOGICAL_ERROR = "logical_error"
    SECURITY_VULNERABILITY = "security_vulnerability"
    PERFORMANCE_PROBLEM = "performance_problem"
    COMPLIANCE_ISSUE = "compliance_issue"
    STRUCTURAL_FLAW = "structural_flaw"
    CLARITY_ISSUE = "clarity_issue"
    MAINTAINABILITY_PROBLEM = "maintainability_problem"
    SCALABILITY_ISSUE = "scalability_issue"
    USABILITY_PROBLEM = "usability_problem"
    TECHNICAL_DEBT = "technical_debt"
    DOCUMENTATION_GAP = "documentation_gap"
    EDGE_CASE = "edge_case"

class RedTeamStrategy(Enum):
    """Strategies for red team analysis"""
    SYSTEMATIC = "systematic"
    RANDOM_SAMPLING = "random_sampling"
    FOCUSED_ATTACK = "focused_attack"
    DEEP_DIVE = "deep_dive"
    POKA_YOKE = "poka_yoke"  # Error-proofing approach
    ADVERSARIAL = "adversarial"

@dataclass
class IssueFinding:
    """Represents an issue found by the red team"""
    title: str
    description: str
    severity: SeverityLevel
    category: IssueCategory
    location: Optional[str] = None  # e.g. "line 10", "section 2.3", etc.
    confidence: float = 1.0  # 0-1 confidence in finding
    suggested_fix: Optional[str] = None
    exploit_example: Optional[str] = None  # Example of how issue could be exploited

@dataclass
class RedTeamAssessment:
    """Complete assessment from the red team"""
    findings: List[IssueFinding]
    assessment_summary: str
    confidence_score: float  # Overall confidence in the assessment
    time_taken: float
    assessment_metadata: Dict[str, Any]
    issues_by_severity: Dict[SeverityLevel, int]
    issues_by_category: Dict[IssueCategory, int]
    openevolve_metrics: Optional[Dict[str, Any]] = None  # OpenEvolve metrics if used

class RedTeamMember:
    """Individual red team member with specific expertise"""
    
    def __init__(self, name: str, specializations: List[IssueCategory], 
                 expertise_level: int = 7, attack_method: RedTeamStrategy = RedTeamStrategy.SYSTEMATIC):
        self.name = name
        self.specializations = specializations
        self.expertise_level = expertise_level  # 1-10 scale
        self.attack_method = attack_method
        self.performance_history: List[Dict[str, Any]] = []
        self.reliability_score = 0.9  # Base reliability
        
    def assess_content(self, content: str, content_type: str = "general", 
                       attack_modes: Optional[List[str]] = None) -> List[IssueFinding]:
        """
        Assess content and return a list of findings
        
        Args:
            content: Content to assess
            content_type: Type of content being assessed
            attack_modes: Specific attack modes to focus on (e.g., 'Security Scan', 'Edge Case Exploration')
            
        Returns:
            List of issue findings
        """
        start_time = time.time()
        
        findings = []
        
        # Apply different assessment techniques based on strategy
        if self.attack_method == RedTeamStrategy.SYSTEMATIC:
            findings.extend(self._systematic_assessment(content, content_type))
        elif self.attack_method == RedTeamStrategy.RANDOM_SAMPLING:
            findings.extend(self._random_sampling_assessment(content))
        elif self.attack_method == RedTeamStrategy.FOCUSED_ATTACK:
            findings.extend(self._focused_attack_assessment(content, content_type))
        elif self.attack_method == RedTeamStrategy.DEEP_DIVE:
            findings.extend(self._deep_dive_assessment(content, content_type))
        elif self.attack_method == RedTeamStrategy.POKA_YOKE:
            findings.extend(self._poka_yoke_assessment(content))
        elif self.attack_method == RedTeamStrategy.ADVERSARIAL:
            findings.extend(self._adversarial_assessment(content, content_type))
        
        # Apply attack modes if specified (from gauntlet_definition)
        if attack_modes:
            findings.extend(self._apply_attack_modes(content, attack_modes, content_type))
        
        # Apply expertise multiplier to findings
        adjusted_findings = []
        for finding in findings:
            # Adjust confidence based on expertise level
            adjusted_confidence = finding.confidence * (self.expertise_level / 10.0)
            # Specialization bonus
            if finding.category in self.specializations:
                adjusted_confidence = min(1.0, adjusted_confidence * 1.2)
            
            new_finding = IssueFinding(
                title=finding.title,
                description=finding.description,
                severity=finding.severity,
                category=finding.category,
                location=finding.location,
                confidence=adjusted_confidence,
                suggested_fix=finding.suggested_fix,
                exploit_example=finding.exploit_example
            )
            adjusted_findings.append(new_finding)
        
        # Record performance
        assessment_time = time.time() - start_time
        self.performance_history.append({
            'timestamp': datetime.now(),
            'content_type': content_type,
            'findings_count': len(adjusted_findings),
            'time_taken': assessment_time,
            'attack_modes_used': attack_modes
        })
        
        # Keep only last 20 assessments
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return adjusted_findings

    def _apply_attack_modes(self, content: str, attack_modes: List[str], content_type: str) -> List[IssueFinding]:
        """
        Apply specific attack modes to the content as defined in the gauntlet
        
        Args:
            content: Content to assess
            attack_modes: List of attack modes to apply
            content_type: Type of content being assessed
            
        Returns:
            List of issue findings from attack modes
        """
        findings = []
        
        for mode in attack_modes:
            mode = mode.lower().strip()
            
            if mode == "security scan":
                findings.extend(self._security_scan_assessment(content))
            elif mode == "edge case exploration":
                findings.extend(self._edge_case_assessment(content, content_type))
            elif mode == "assumption challenge":
                findings.extend(self._assumption_challenge_assessment(content))
            elif mode == "compliance check":
                findings.extend(self._compliance_assessment(content))
            elif mode == "logic verification":
                findings.extend(self._logic_verification_assessment(content))
            # Add other attack modes as needed
            
        return findings
    
    def _security_scan_assessment(self, content: str) -> List[IssueFinding]:
        """
        Perform security-focused assessment
        """
        return self._code_systematic_check(content)  # Reuse existing security checks
    
    def _edge_case_assessment(self, content: str, content_type: str) -> List[IssueFinding]:
        """
        Perform edge case exploration assessment
        """
        findings = []
        
        # Look for potential edge cases based on content type
        if content_type == "code":
            # Look for input validation issues
            if re.search(r'\.get\([^)]+\)', content) or re.search(r'\[\s*[^]]+\s*\]', content):
                findings.append(IssueFinding(
                    title="Potential boundary condition issue",
                    description="Code may not properly handle boundary conditions or edge cases",
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.EDGE_CASE,
                    confidence=0.7
                ))
        
        return findings
    
    def _assumption_challenge_assessment(self, content: str) -> List[IssueFinding]:
        """
        Challenge assumptions in the content
        """
        findings = []
        
        # Look for strong assumptions
        assumption_patterns = [
            (r'always|never|all|none|every|each|will always|will never', 'Strong assumption detected'),
            (r'assume|assuming', 'Explicit assumption identified'),
        ]
        
        for pattern, title in assumption_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Assumption found: '{match.group(0)}'",
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.LOGICAL_ERROR,
                    location=f"around: {match.group(0)[:50]}...",
                    confidence=0.6
                ))
        
        return findings
    
    def _compliance_assessment(self, content: str) -> List[IssueFinding]:
        """
        Check compliance with standards
        """
        return self._identify_compliance_gaps(content)  # Reuse existing compliance checks
    
    def _logic_verification_assessment(self, content: str) -> List[IssueFinding]:
        """
        Verify logical consistency
        """
        return self._identify_structural_issues(content)  # Reuse existing logic checks
    
    def _systematic_assessment(self, content: str, content_type: str) -> List[IssueFinding]:
        """Systematically go through content looking for issues"""
        findings = []
        
        # Check common patterns based on content type
        if content_type == "code":
            findings.extend(self._code_systematic_check(content))
        elif content_type == "document":
            findings.extend(self._document_systematic_check(content))
        elif content_type == "protocol":
            findings.extend(self._protocol_systematic_check(content))
        elif content_type == "legal":
            findings.extend(self._legal_systematic_check(content))
        elif content_type == "medical":
            findings.extend(self._medical_systematic_check(content))
        elif content_type == "technical":
            findings.extend(self._technical_systematic_check(content))
        else:  # general
            findings.extend(self._general_systematic_check(content))
        
        return findings
    
    def _code_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for code content"""
        findings = []
        
        # Look for common security issues
        security_patterns = [
            (r'eval\s*\(', 'Use of eval() function', IssueCategory.SECURITY_VULNERABILITY),
            (r'exec\s*\(', 'Use of exec() function', IssueCategory.SECURITY_VULNERABILITY),
            (r'password\s*[:=]\s*[\'"][^\'"]{3,}[\'"]', 'Hardcoded password', IssueCategory.SECURITY_VULNERABILITY),
            (r'API_key\s*[:=]\s*[\'"][^\'"]{8,}[\'"]', 'Hardcoded API key', IssueCategory.SECURITY_VULNERABILITY),
            (r'select\s+\*\s+from', 'SQL injection risk', IssueCategory.SECURITY_VULNERABILITY),
        ]
        
        for pattern, title, category in security_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Found potential security vulnerability: {match.group(0)}",
                    severity=SeverityLevel.CRITICAL,
                    category=category,
                    location=f"line containing: {content.split(chr(10))[min(len(content.split(chr(10)))-1, content.count(chr(10), 0, match.start()))]}",
                    confidence=0.9
                ))

        # Look for performance issues
        perf_patterns = [
            (r'for.*in.*range\(\d{6,}\)', 'Inefficient loop over large range', IssueCategory.PERFORMANCE_PROBLEM),
            (r'\.append\(\)\s+in\s+loop', 'Inefficient list building in loop', IssueCategory.PERFORMANCE_PROBLEM),
        ]

        for pattern, title, category in perf_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Found performance issue: {match.group(0)}",
                    severity=SeverityLevel.HIGH,
                    category=category,
                    location=f"line containing: {content.split(chr(10))[min(len(content.split(chr(10)))-1, content.count(chr(10), 0, match.start()))]}",
                    confidence=0.8,
                    suggested_fix="Consider using list comprehension or pre-allocating memory"
                ))
        
        return findings
    
    def _document_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for document content"""
        findings = []
        
        # Check for missing information
        sentences = re.split(r'[.!?]+', content)
        long_sentences = [s for s in sentences if len(s.split()) > 30]
        
        if long_sentences:
            findings.append(IssueFinding(
                title="Overly complex sentences",
                description=f"Found {len(long_sentences)} sentences with more than 30 words",
                severity=SeverityLevel.MEDIUM,
                category=IssueCategory.CLARITY_ISSUE,
                confidence=0.7,
                suggested_fix="Break down long sentences into shorter, clearer ones"
            ))
        
        return findings
    
    def _protocol_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for protocol content"""
        findings = []
        
        # Check for missing edge cases
        edge_case_keywords = [
            r'\b(exception|error|failure|timeout|retry|fallback)\b',
            r'\b(resilience|recovery|rollback|compensation)\b'
        ]
        
        missing_edge_case = True
        for pattern in edge_case_keywords:
            if re.search(pattern, content, re.IGNORECASE):
                missing_edge_case = False
                break
        
        if missing_edge_case:
            findings.append(IssueFinding(
                title="Missing edge case handling",
                description="Protocol does not appear to address exception or error conditions",
                severity=SeverityLevel.HIGH,
                category=IssueCategory.EDGE_CASE,
                confidence=0.8,
                suggested_fix="Add sections on error handling, fallback procedures, and recovery strategies"
            ))
        
        return findings
    
    def _legal_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for legal content"""
        findings = []
        
        # Check for common missing legal terms
        required_terms = [
            (r'\b(indemnity|hold harmless)\b', 'Missing indemnification clause'),
            (r'\bgoverning law|jurisdiction\b', 'Missing governing law/jurisdiction'),
            (r'\b(disclaimer|warranty disclaimer)\b', 'Missing limitation of liability'),
        ]
        
        for pattern, description in required_terms:
            if not re.search(pattern, content, re.IGNORECASE):
                findings.append(IssueFinding(
                    title=description,
                    description=description,
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.COMPLIANCE_ISSUE,
                    confidence=0.7
                ))
        
        return findings
    
    def _medical_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for medical content"""
        findings = []
        
        # Check for PHI exposure
        phi_patterns = [
            r'\b(patient.*?name|medical.*?record|DOB|date of birth)\b',
        ]
        
        for pattern in phi_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title="Potential PHI exposure",
                    description=f"Found potential protected health information: {match.group(0)}",
                    severity=SeverityLevel.CRITICAL,
                    category=IssueCategory.COMPLIANCE_ISSUE,
                    location=f"around: {match.group(0)[:50]}...",
                    confidence=0.9
                ))
        
        return findings
    
    def _technical_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for technical content"""
        findings = []
        
        # Check for missing security considerations
        if not re.search(r'security|authentication|authorization|encryption|privacy', content, re.IGNORECASE):
            findings.append(IssueFinding(
                title="Missing security considerations",
                description="Technical specification does not appear to address security aspects",
                severity=SeverityLevel.HIGH,
                category=IssueCategory.SECURITY_VULNERABILITY,
                confidence=0.8,
                suggested_fix="Add security requirements and implementation guidelines"
            ))
        
        return findings
    
    def _general_systematic_check(self, content: str) -> List[IssueFinding]:
        """Systematic check for general content"""
        findings = []
        
        # Check for clarity issues
        sentences = re.split(r'[.!?]+', content)
        long_sentences = [s for s in sentences if len(s.split()) > 25]
        
        if len(long_sentences) > 3:
            findings.append(IssueFinding(
                title="Clarity issues",
                description=f"Found {len(long_sentences)} complex sentences that may hurt readability",
                severity=SeverityLevel.MEDIUM,
                category=IssueCategory.CLARITY_ISSUE,
                confidence=0.7,
                suggested_fix="Simplify complex sentences and break them down"
            ))
        
        return findings
    
    def _random_sampling_assessment(self, content: str) -> List[IssueFinding]:
        """Assess content by randomly sampling sections"""
        findings = []
        
        # Split content into paragraphs
        paragraphs = content.split('\n\n')
        if len(paragraphs) < 5:
            # If too short, check the whole content
            return self._systematic_assessment(content, "general")
        
        # Randomly select 30% of paragraphs to check
        sample_size = max(1, len(paragraphs) // 3)
        sampled_paragraphs = random.sample(paragraphs, min(sample_size, len(paragraphs)))
        
        for i, para in enumerate(sampled_paragraphs):
            # Look for issues in the sampled paragraphs
            # Check for potential issues
            if len(para) > 500:  # Very long paragraph
                findings.append(IssueFinding(
                    title="Overly long paragraph",
                    description="Paragraph exceeds recommended length",
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CLARITY_ISSUE,
                    location=f"paragraph {i+1}",
                    confidence=0.6
                ))
        
        return findings
    
    def _focused_attack_assessment(self, content: str, content_type: str) -> List[IssueFinding]:
        """Focus attack on specific vulnerable areas based on content type"""
        findings = []
        
        if content_type == "code":
            # Focus on known vulnerable code patterns
            findings.extend(self._identify_code_vulnerabilities(content))
        elif content_type == "legal":
            # Focus on compliance gaps
            findings.extend(self._identify_compliance_gaps(content))
        elif content_type == "technical":
            # Focus on security and performance
            findings.extend(self._identify_security_performance_gaps(content))
        else:
            # For other types, focus on structural issues
            findings.extend(self._identify_structural_issues(content))
        
        return findings
    
    def _deep_dive_assessment(self, content: str, content_type: str) -> List[IssueFinding]:
        """Perform a deep, thorough assessment"""
        findings = []
        
        # Apply all checks but with more thorough analysis
        findings.extend(self._systematic_assessment(content, content_type))
        
        # Add advanced analysis
        if content_type == "code":
            findings.extend(self._advanced_code_analysis(content))
        elif content_type == "document":
            findings.extend(self._advanced_document_analysis(content))
        
        return findings
    
    def _poka_yoke_assessment(self, content: str) -> List[IssueFinding]:
        """Assess content for error-proofing opportunities"""
        findings = []
        
        # Look for areas where mistakes are likely
        findings.append(IssueFinding(
            title="Lack of verification points",
            description="Content does not include verification or validation steps",
            severity=SeverityLevel.MEDIUM,
            category=IssueCategory.TECHNICAL_DEBT,
            confidence=0.7,
            suggested_fix="Add verification checkpoints to prevent errors"
        ))
        
        return findings
    
    def _adversarial_assessment(self, content: str, content_type: str) -> List[IssueFinding]:
        """Adversarial assessment from an attacker's perspective"""
        findings = []
        
        if content_type == "code":
            # Think like an attacker
            findings.extend(self._adversarial_code_analysis(content))
        elif content_type == "document":
            # Look for misdirection or deception
            findings.extend(self._adversarial_document_analysis(content))
        
        return findings
    
    def _identify_code_vulnerabilities(self, content: str) -> List[IssueFinding]:
        """Identify specific code vulnerabilities"""
        findings = []
        
        # Look for common vulnerability patterns
        vulnerable_patterns = [
            (r'os\.system\(', 'OS command injection', SeverityLevel.CRITICAL),
            (r'subprocess\.call\(', 'Potential command injection', SeverityLevel.HIGH),
            (r'format\([^)]*{[^}]*}\)', 'Potential format string vulnerability', SeverityLevel.MEDIUM),
            (r'\.replace\([^)]*input\(', 'Tainted data replacement', SeverityLevel.HIGH),
        ]
        
        for pattern, title, severity in vulnerable_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Found vulnerable pattern: {match.group(0)}",
                    severity=severity,
                    category=IssueCategory.SECURITY_VULNERABILITY,
                    location=f"line with: {content.split(chr(10))[min(len(content.split(chr(10)))-1, content.count(chr(10), 0, match.start()))]}",
                    confidence=0.85
                ))
        
        return findings
    
    def _identify_compliance_gaps(self, content: str) -> List[IssueFinding]:
        """Identify compliance gaps in legal content"""
        findings = []
        
        # Check for specific compliance markers
        compliance_areas = [
            ('GDPR', r'\b(GDPR|general data protection regulation)\b', 'Missing GDPR compliance'),
            ('HIPAA', r'\b(HIPAA|health insurance portability|protected health information)\b', 'Missing HIPAA compliance'),
            ('SOX', r'\b(SOX|sarbanes-oxley)\b', 'Missing SOX compliance'),
        ]
        
        for area, pattern, description in compliance_areas:
            if not re.search(pattern, content, re.IGNORECASE):
                findings.append(IssueFinding(
                    title=description,
                    description=description,
                    severity=SeverityLevel.HIGH,
                    category=IssueCategory.COMPLIANCE_ISSUE,
                    confidence=0.7
                ))
        
        return findings
    
    def _identify_security_performance_gaps(self, content: str) -> List[IssueFinding]:
        """Identify security and performance gaps in technical content"""
        findings = []
        
        # Check for performance without security
        if re.search(r'performance|speed|efficiency', content, re.IGNORECASE) and \
           not re.search(r'security|authentication|encryption', content, re.IGNORECASE):
            findings.append(IssueFinding(
                title="Performance-focused without security",
                description="Content emphasizes performance but lacks security considerations",
                severity=SeverityLevel.HIGH,
                category=IssueCategory.SECURITY_VULNERABILITY,
                confidence=0.8,
                suggested_fix="Balance performance with security requirements"
            ))
        
        return findings
    
    def _identify_structural_issues(self, content: str) -> List[IssueFinding]:
        """Identify structural issues in content"""
        findings = []
        
        # Check for proper ordering
        content_lower = content.lower()
        if 'summary' in content_lower and 'introduction' in content_lower:
            # Check if summary comes before introduction
            summary_pos = content_lower.find('summary')
            intro_pos = content_lower.find('introduction')
            if summary_pos < intro_pos:
                findings.append(IssueFinding(
                    title="Incorrect document structure",
                    description="Summary appears before introduction",
                    severity=SeverityLevel.MEDIUM,
                    category=IssueCategory.STRUCTURAL_FLAW,
                    confidence=0.8
                ))
        
        return findings
    
    def _advanced_code_analysis(self, content: str) -> List[IssueFinding]:
        """Advanced code analysis for deeper issues"""
        findings = []
        
        # Analyze code complexity
        lines = content.split('\n')
        complex_functions = []
        
        current_function = None
        max_nesting = 0
        current_nesting = 0
        
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('def ') or stripped.startswith('function ') or stripped.startswith('class '):
                current_function = stripped
                current_nesting = 0
            elif stripped:
                # Count nesting level based on indentation
                leading_spaces = len(line) - len(line.lstrip())
                nesting_level = leading_spaces // 4  # Assuming 4-space indentation
                
                if nesting_level > current_nesting:
                    current_nesting = nesting_level
                    if current_nesting > 5:  # Too deeply nested
                        complex_functions.append(current_function)
                        max_nesting = max(max_nesting, current_nesting)
        
        if max_nesting > 5:
            findings.append(IssueFinding(
                title="Excessive code nesting",
                description=f"Found functions with nesting level of {max_nesting}, which hurts readability",
                severity=SeverityLevel.MEDIUM,
                category=IssueCategory.MAINTAINABILITY_PROBLEM,
                confidence=0.8,
                suggested_fix="Refactor deeply nested code into smaller, more manageable functions"
            ))
        
        return findings
    
    def _advanced_document_analysis(self, content: str) -> List[IssueFinding]:
        """Advanced document analysis"""
        findings = []
        
        # Check for consistency in terminology
        content_lower = content.lower()
        words = re.findall(r'\b\w+\b', content_lower)
        unique_words = set(words)
        
        # Look for similar terms that might indicate inconsistency
        # Check for multiple synonymous terms that should be standardized
        if 'user' in unique_words and 'client' in unique_words and 'customer' in unique_words:
            findings.append(IssueFinding(
                title="Inconsistent terminology",
                description="Multiple terms used for similar concepts (user, client, customer)",
                severity=SeverityLevel.LOW,
                category=IssueCategory.CLARITY_ISSUE,
                confidence=0.6,
                suggested_fix="Standardize terminology throughout the document"
            ))
        
        return findings
    
    def _adversarial_code_analysis(self, content: str) -> List[IssueFinding]:
        """Adversarial analysis of code from attacker perspective"""
        findings = []
        
        # Look for authentication bypass patterns
        auth_bypass_patterns = [
            (r'return True', 'Hardcoded authentication success', SeverityLevel.CRITICAL),
            (r'#.*auth.*disabled|debug', 'Authentication disabled in debug mode', SeverityLevel.CRITICAL),
        ]
        
        for pattern, title, severity in auth_bypass_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Potential authentication bypass: {match.group(0)}",
                    severity=severity,
                    category=IssueCategory.SECURITY_VULNERABILITY,
                    location=f"line with: {content.split(chr(10))[min(len(content.split(chr(10)))-1, content.count(chr(10), 0, match.start()))]}",
                    confidence=0.9
                ))
        
        return findings
    
    def _adversarial_document_analysis(self, content: str) -> List[IssueFinding]:
        """Adversarial analysis of document content"""
        findings = []
        
        # Look for misleading or deceptive language
        deceptive_patterns = [
            (r'guarantee|100% sure|always works', 'Overly confident language', SeverityLevel.MEDIUM),
            (r'can be done easily|simple|trivial', 'Downplaying complexity', SeverityLevel.LOW),
        ]
        
        for pattern, title, severity in deceptive_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                findings.append(IssueFinding(
                    title=title,
                    description=f"Potentially misleading language: {match.group(0)}",
                    severity=severity,
                    category=IssueCategory.CLARITY_ISSUE,
                    location=f"around: {match.group(0)[:50]}...",
                    confidence=0.7
                ))
        
        return findings

class RedTeam:
    """Main Red Team orchestrator that manages multiple red team members"""
    
    def __init__(self, orchestrator: ModelOrchestrator = None, 
                 prompt_engineering: PromptEngineeringSystem = None,
                 content_analyzer: ContentAnalyzer = None,
                 quality_assessment: QualityAssessmentEngine = None):
        self.orchestrator = orchestrator
        self.prompt_engineering = prompt_engineering
        self.content_analyzer = content_analyzer
        self.quality_assessment = quality_assessment
        self.team_members: List[RedTeamMember] = []
        self.assessment_history: List[RedTeamAssessment] = []
        
        # Initialize default team members
        self._initialize_default_team()
    
    def _initialize_default_team(self):
        """Initialize a default red team with different specializations"""
        self.add_team_member(RedTeamMember(
            name="SecuritySpecialist",
            specializations=[IssueCategory.SECURITY_VULNERABILITY, IssueCategory.COMPLIANCE_ISSUE],
            expertise_level=9,
            attack_method=RedTeamStrategy.FOCUSED_ATTACK
        ))
        
        self.add_team_member(RedTeamMember(
            name="CodeQualityExpert",
            specializations=[IssueCategory.TECHNICAL_DEBT, IssueCategory.MAINTAINABILITY_PROBLEM],
            expertise_level=8,
            attack_method=RedTeamStrategy.DEEP_DIVE
        ))
        
        self.add_team_member(RedTeamMember(
            name="PerformanceAnalyst",
            specializations=[IssueCategory.PERFORMANCE_PROBLEM, IssueCategory.SCALABILITY_ISSUE],
            expertise_level=7,
            attack_method=RedTeamStrategy.SYSTEMATIC
        ))
        
        self.add_team_member(RedTeamMember(
            name="LogicValidator",
            specializations=[IssueCategory.LOGICAL_ERROR, IssueCategory.EDGE_CASE],
            expertise_level=8,
            attack_method=RedTeamStrategy.POKA_YOKE
        ))
        
        self.add_team_member(RedTeamMember(
            name="ClarityReviewer",
            specializations=[IssueCategory.CLARITY_ISSUE, IssueCategory.DOCUMENTATION_GAP],
            expertise_level=7,
            attack_method=RedTeamStrategy.SYSTEMATIC
        ))
    
    def add_team_member(self, member: RedTeamMember):
        """Add a new red team member"""
        self.team_members.append(member)
    
    def remove_team_member(self, name: str) -> bool:
        """Remove a red team member by name"""
        for i, member in enumerate(self.team_members):
            if member.name == name:
                del self.team_members[i]
                return True
        return False
    
    def assess_content(self, content: str, content_type: str = "general", 
                      custom_requirements: Optional[Dict[str, Any]] = None,
                      strategy: RedTeamStrategy = RedTeamStrategy.SYSTEMATIC,
                      num_members: Optional[int] = None,
                      api_key: Optional[str] = None,
                      model_name: str = "gpt-4o",
                      attack_modes: Optional[List[str]] = None) -> RedTeamAssessment:
        """
        Assess content with the red team, using OpenEvolve when available
        
        Args:
            content: Content to assess
            content_type: Type of content
            custom_requirements: Custom requirements to check
            strategy: Strategy to use for assessment
            num_members: Number of team members to use (None for all)
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
            attack_modes: Specific attack modes to apply (e.g., from gauntlet definition)
        
        Returns:
            RedTeamAssessment with findings
        """
        start_time = time.time()
        
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            assessment = self._assess_with_openevolve_backend(
                content, content_type, custom_requirements, api_key, model_name
            )
            assessment.time_taken = time.time() - start_time
            return assessment
        
        # Fallback to custom implementation
        return self._assess_with_custom_implementation(
            content, content_type, custom_requirements, strategy, num_members, start_time, attack_modes
        )
    
    def _assess_with_openevolve_backend(self, content: str, content_type: str,
                                      custom_requirements: Optional[Dict[str, Any]],
                                      api_key: str, model_name: str) -> RedTeamAssessment:
        """
        Assess content using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.5,  # Lower temperature for more consistent critiques
                max_tokens=4096,
            )
            
            config.llm.models = [llm_config]
            config.max_iterations = 1  # Just one iteration for critique
            config.database.population_size = 1  # Single assessment
            
            # Create a simple evaluator for red team assessment
            def red_team_evaluator(program_path: str, api_key: str, model_name: str) -> Dict[str, Any]:
                            """
                            Evaluator that performs red team assessment on the content using an LLM.
                            """
                            try:
                                with open(program_path, "r", encoding='utf-8') as f:
                                    content = f.read()
                                
                                # Use LLM to assess content for vulnerabilities and generate a score.
                                # This replaces the previous hardcoded score with a dynamic, LLM-driven evaluation.
                                system_prompt = "You are a Red Team AI. Your goal is to find flaws, vulnerabilities, and weaknesses in the provided content. If you find a flaw, explain it clearly. If not, state that the content appears robust. Provide your response as a JSON object with 'score' (0.0-1.0 for robustness), 'justification' (string), and 'targeted_feedback' (string, if applicable, mentioning specific sub-problem IDs like 'sub_1.2' that are faulty)."
                                user_prompt = f"""Critique the following content for flaws and vulnerabilities.
                                Content:
                                ---
                                {content}
                                ---
                                Provide your critique as a JSON object with 'score', 'justification', and 'targeted_feedback'.
                                """
            
                                # Make LLM call to perform red team evaluation
                                try:
                                    llm_response_content = _request_openai_compatible_chat(
                                        api_key=api_key,
                                        base_url="https://api.openai.com/v1",  # Default base URL
                                        model=model_name,
                                        messages=_compose_messages(system_prompt, user_prompt),
                                        temperature=0.5,
                                        max_tokens=1024,
                                        timeout=10,
                                        response_json_format=True  # Request JSON format for structured output
                                    )
                                    if llm_response_content:
                                        llm_parsed_response = json.loads(llm_response_content)
                                        llm_score = llm_parsed_response.get("score", 0.5)
                                    else:
                                        logger.warning("LLM call failed for red team evaluator. Falling back to default score.")
                                        llm_score = 0.5  # Fallback if LLM call fails
                                    
                                    return {
                                        "score": llm_score, 
                                        "timestamp": datetime.now().timestamp(),
                                        "content_length": len(content),
                                        "assessment_completed": True
                                    }
                                except Exception as e:
                                    logger.error(f"Error in LLM call for red team evaluator: {e}", exc_info=True)
                                    return {
                                        "score": 0.5,
                                        "timestamp": datetime.now().timestamp(),
                                        "error": str(e)
                                    }
                            except Exception as e:
                                logger.error(f"Error in red team evaluator: {e}", exc_info=True)
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
                # Run assessment using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=red_team_evaluator,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Process results and create findings
                # to extract specific issues found during the process
                findings = self._extract_findings_from_openevolve_result(result, content_type)
                
                # Count by severity and category
                issues_by_severity = {}
                issues_by_category = {}
                
                for finding in findings:
                    # Count by severity
                    severity = finding.severity
                    issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
                    
                    # Count by category
                    category = finding.category
                    issues_by_category[category] = issues_by_category.get(category, 0) + 1
                
                # Create assessment summary
                summary = self._create_assessment_summary(findings, content_type)
                
                # Calculate confidence score
                confidence_score = self._calculate_confidence_score(findings)
                
                # Create assessment object
                assessment = RedTeamAssessment(
                    findings=findings,
                    assessment_summary=summary,
                    confidence_score=confidence_score,
                    time_taken=0,  # Will be set by caller
                    assessment_metadata={
                        'content_type': content_type,
                        'openevolve_used': True,
                        'custom_requirements_applied': bool(custom_requirements),
                        'assessment_timestamp': datetime.now().isoformat()
                    },
                    issues_by_severity=issues_by_severity,
                    issues_by_category=issues_by_category
                )
                
                # Store in history
                self.assessment_history.append(assessment)
                
                # Keep only last 50 assessments
                if len(self.assessment_history) > 50:
                    self.assessment_history = self.assessment_history[-50:]
                
                return assessment
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            logger.error(f"Error using OpenEvolve backend for red team assessment: {e}", exc_info=True)
            # Fallback to custom implementation
            return self._assess_with_custom_implementation(content, content_type, custom_requirements)
    
    def _extract_findings_from_openevolve_result(self, result, content_type: str) -> List[IssueFinding]:
        """
        Extract issue findings from OpenEvolve evolution result
        """
        findings = []
        
        # Parse the OpenEvolve evolution result to extract specific issues
        # found during the adversarial evolutionary process
        
        # Assuming 'result' is a dictionary containing the output of the evolution run
        # and that the evaluator (red_team_evaluator) stores its findings in a structured way.
        
        # Hypothetical structure: result['best_individual']['evaluation_results']['feedback']
        # where 'feedback' is a list of dictionaries, each representing an issue.
        
        # Let's assume the red_team_evaluator (defined in _assess_with_openevolve_backend)
        # returns a dictionary with 'score', 'justification', and 'targeted_feedback'.
        # We need to extract findings from this 'justification' or 'targeted_feedback'.
        
        # For a more concrete example, let's assume the 'result' object from openevolve_run_evolution
        # contains a 'history' or 'individuals' field, and each individual has an 'evaluation_result'
        # which includes the detailed LLM output from the red_team_evaluator.
        
        # Let's assume the 'result' object has a 'best_individual' key, and its 'evaluation_results'
        # contains the raw LLM response from the red_team_evaluator.
        
        llm_raw_output = None
        if result and 'best_individual' in result and 'evaluation_results' in result['best_individual']:
            # Assuming the red_team_evaluator's full output is stored here
            llm_raw_output = result['best_individual']['evaluation_results'].get('llm_output')
        
        if llm_raw_output:
            try:
                # The red_team_evaluator is designed to return JSON with 'score', 'justification', 'targeted_feedback'
                llm_parsed_output = json.loads(llm_raw_output)
                justification = llm_parsed_output.get('justification', '')
                targeted_feedback = llm_parsed_output.get('targeted_feedback', '')
                
                # Simple parsing of justification into findings
                if justification:
                    findings.append(IssueFinding(
                        title="LLM Justification",
                        description=justification,
                        severity=SeverityLevel.MEDIUM,
                        category=IssueCategory.LOGICAL_ERROR, # Default category
                        confidence=0.7
                    ))
                
                # More detailed parsing of targeted_feedback if it contains structured issues
                # This part would be highly dependent on the exact format of targeted_feedback
                # For now, let's assume targeted_feedback is a string that might contain issue descriptions
                if targeted_feedback:
                    # Split by common separators or look for specific patterns
                    feedback_lines = targeted_feedback.split('\\n')
                    for line in feedback_lines:
                        line = line.strip()
                        if line and not line.startswith('sub_'): # Avoid raw sub-problem IDs for now
                            findings.append(IssueFinding(
                                title="Targeted Feedback",
                                description=line,
                                severity=SeverityLevel.LOW, # Default severity
                                category=IssueCategory.CLARITY_ISSUE, # Default category
                                confidence=0.6
                            ))
            except json.JSONDecodeError:
                # If LLM output is not valid JSON, treat it as a single finding
                findings.append(IssueFinding(
                    title="LLM Raw Output",
                    description=llm_raw_output,
                    severity=SeverityLevel.LOW,
                    category=IssueCategory.CLARITY_ISSUE,
                    confidence=0.5
                ))
        
        # Fallback if no specific findings extracted
        if not findings:
            findings.append(IssueFinding(
                title=f"General OpenEvolve Assessment for {content_type}",
                description=f"OpenEvolve run completed for {content_type} content. No specific structured findings extracted from the result.",
                severity=SeverityLevel.LOW,
                category=IssueCategory.STRUCTURAL_FLAW,
                confidence=0.5
            ))
        
        return findings

    def _assess_with_custom_implementation(self, content: str, content_type: str = "general", 
                                         custom_requirements: Optional[Dict[str, Any]] = None,
                                         strategy: RedTeamStrategy = RedTeamStrategy.SYSTEMATIC,
                                         num_members: Optional[int] = None,
                                         start_time: float = None,
                                         attack_modes: Optional[List[str]] = None) -> RedTeamAssessment:
        """
        Fallback assessment using custom implementation
        """
        if start_time is None:
            start_time = time.time()
        
        # Select team members to use
        selected_members = self.team_members
        if num_members and num_members < len(self.team_members):
            selected_members = random.sample(self.team_members, num_members)
        
        # Perform assessment with all selected members
        all_findings = []
        for member in selected_members:
            # Override member strategy if a specific one is provided
            original_strategy = member.attack_method
            if strategy != RedTeamStrategy.SYSTEMATIC:  # Don't override if systematic (default)
                member.attack_method = strategy
            
            member_findings = member.assess_content(content, content_type, attack_modes)
            all_findings.extend(member_findings)
            
            # Restore original strategy
            member.attack_method = original_strategy
        
        # Consolidate findings
        consolidated_findings = self._consolidate_findings(all_findings)
        
        # Count by severity and category
        issues_by_severity = {}
        issues_by_category = {}
        
        for finding in consolidated_findings:
            # Count by severity
            severity = finding.severity
            issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
            
            # Count by category
            category = finding.category
            issues_by_category[category] = issues_by_category.get(category, 0) + 1
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(consolidated_findings)
        
        # Create assessment summary
        summary = self._create_assessment_summary(consolidated_findings, content_type)
        
        # Create assessment object
        assessment = RedTeamAssessment(
            findings=consolidated_findings,
            assessment_summary=summary,
            confidence_score=confidence_score,
            time_taken=time.time() - start_time,
            assessment_metadata={
                'content_type': content_type,
                'num_team_members': len(selected_members),
                'strategy_used': strategy.value,
                'custom_requirements_applied': bool(custom_requirements),
                'attack_modes_used': attack_modes,
                'assessment_timestamp': datetime.now().isoformat(),
                'openevolve_used': False  # Mark as custom implementation
            },
            issues_by_severity=issues_by_severity,
            issues_by_category=issues_by_category
        )
        
        # Store in history
        self.assessment_history.append(assessment)
        
        # Keep only last 50 assessments
        if len(self.assessment_history) > 50:
            self.assessment_history = self.assessment_history[-50:]
        
        return assessment
    
    def _consolidate_findings(self, findings: List[IssueFinding]) -> List[IssueFinding]:
        """Consolidate similar findings"""
        if not findings:
            return []
        
        # Group findings by title and description similarity
        consolidated = []
        seen_finding_keys = set()
        
        for finding in findings:
            # Create a key based on title and description (normalized)
            key = (finding.title.lower().strip(), finding.description.lower().strip())
            
            if key not in seen_finding_keys:
                # Calculate average confidence for similar findings
                similar_findings = [
                    f for f in findings 
                    if (f.title.lower().strip(), f.description.lower().strip()) == key
                ]
                
                avg_confidence = statistics.mean([f.confidence for f in similar_findings])
                
                # Use the highest severity among similar findings
                max_severity = max([f.severity for f in similar_findings], 
                                  key=lambda s: ['low', 'medium', 'high', 'critical'].index(s.value))
                
                consolidated.append(IssueFinding(
                    title=finding.title,
                    description=finding.description,
                    severity=max_severity,
                    category=finding.category,
                    location=finding.location,
                    confidence=avg_confidence,
                    suggested_fix=finding.suggested_fix,
                    exploit_example=finding.exploit_example
                ))
                
                seen_finding_keys.add(key)
        
        return consolidated
    
    def _calculate_confidence_score(self, findings: List[IssueFinding]) -> float:
        """Calculate overall confidence score based on findings"""
        if not findings:
            return 0.0
        
        # Weight confidence by severity
        severity_weights = {SeverityLevel.LOW: 1, SeverityLevel.MEDIUM: 2, 
                           SeverityLevel.HIGH: 4, SeverityLevel.CRITICAL: 8}
        
        weighted_confidence_sum = 0
        total_weight = 0
        
        for finding in findings:
            weight = severity_weights[finding.severity]
            weighted_confidence_sum += finding.confidence * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        return min(100.0, (weighted_confidence_sum / total_weight) * 100)
    
    def assess_content_with_quality_diversity(self, content: str, content_type: str = "general",
                                            api_key: Optional[str] = None, model_name: str = "gpt-4",
                                            **kwargs) -> RedTeamAssessment:
        """
        Assess content using quality diversity approach (MAP-Elites style).
        This method explores diverse critique strategies to find a wide range of issues.
        
        Args:
            content: Content to assess
            content_type: Type of content being assessed
            api_key: API key for LLM calls
            model_name: Model to use for assessment
            **kwargs: Additional parameters
            
        Returns:
            RedTeamAssessment with diverse findings
        """
        start_time = time.time()
        
        # Use multiple strategies to get diverse findings
        strategies = [
            RedTeamStrategy.SYSTEMATIC,
            RedTeamStrategy.FOCUSED_ATTACK,
            RedTeamStrategy.DEEP_DIVE,
            RedTeamStrategy.ADVERSARIAL
        ]
        
        all_findings = []
        strategy_results = {}
        
        for strategy in strategies:
            try:
                # Create a temporary member with this strategy
                temp_member = RedTeamMember(
                    name=f"QD_{strategy.value}",
                    specializations=list(IssueCategory),
                    attack_method=strategy
                )
                
                # Get findings using this strategy
                strategy_findings = temp_member.assess_content(content, content_type)
                all_findings.extend(strategy_findings)
                strategy_results[strategy.value] = len(strategy_findings)
                
            except Exception as e:
                logger.error(f"Error in strategy {strategy.value} during quality diversity critique: {e}", exc_info=True)
                continue
        
        # Remove duplicate findings (same title and location)
        unique_findings = []
        seen_issues = set()
        
        for finding in all_findings:
            issue_key = (finding.title, finding.location)
            if issue_key not in seen_issues:
                unique_findings.append(finding)
                seen_issues.add(issue_key)
        
        # Calculate diversity metrics
        categories_found = set(f.category for f in unique_findings)
        severities_found = set(f.severity for f in unique_findings)
        
        # Create assessment summary
        summary = f"Quality Diversity Assessment found {len(unique_findings)} unique issues across {len(categories_found)} categories using {len(strategies)} strategies."
        
        # Calculate confidence based on diversity
        diversity_score = (len(categories_found) / len(IssueCategory)) * 0.5 + (len(severities_found) / len(SeverityLevel)) * 0.5
        confidence = min(0.95, 0.7 + diversity_score * 0.25)
        
        # Create metadata
        metadata = {
            "strategies_used": [s.value for s in strategies],
            "strategy_results": strategy_results,
            "diversity_metrics": {
                "categories_covered": len(categories_found),
                "severities_covered": len(severities_found),
                "diversity_score": diversity_score
            },
            "quality_diversity_approach": True
        }
        
        # Count issues by severity and category
        issues_by_severity = {}
        issues_by_category = {}
        
        for severity in SeverityLevel:
            issues_by_severity[severity] = sum(1 for f in unique_findings if f.severity == severity)
            
        for category in IssueCategory:
            issues_by_category[category] = sum(1 for f in unique_findings if f.category == category)
        
        return RedTeamAssessment(
            findings=unique_findings,
            assessment_summary=summary,
            confidence_score=confidence,
            time_taken=time.time() - start_time,
            assessment_metadata=metadata,
            issues_by_severity=issues_by_severity,
            issues_by_category=issues_by_category
        )
    
    def _create_assessment_summary(self, findings: List[IssueFinding], content_type: str) -> str:
        """Create a summary of the assessment"""
        if not findings:
            return "No issues were identified during the red team assessment."
        
        # Count issues by severity
        severity_counts = {}
        for finding in findings:
            severity = finding.severity
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        summary_parts = [
            f"Red Team Assessment Summary for {content_type} content",
            f"Total Issues Found: {len(findings)}",
            "Issues by Severity:"
        ]
        
        for severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH, SeverityLevel.MEDIUM, SeverityLevel.LOW]:
            count = severity_counts.get(severity, 0)
            summary_parts.append(f"  {severity.value.title()}: {count}")
        
        return "\n".join(summary_parts)
    
    def run_red_team_gauntlet(self, solution_attempt: 'SolutionAttempt', gauntlet_def: GauntletDefinition, 
                             team: Team, api_key: str, model_name: str = "gpt-4o") -> CritiqueReport:
        """
        Run a Red Team gauntlet on a solution attempt, following the workflow specification.
        
        Args:
            solution_attempt: The solution to critique
            gauntlet_def: The gauntlet definition with rules and rounds
            team: The Red Team members to use for the critique
            api_key: API key for LLM calls
            model_name: Model name for LLM calls
        
        Returns:
            CritiqueReport with the results of the gauntlet
        """
        start_time = time.time()
        
        # Initialize report components
        reports_by_judge = []
        overall_score = 0.0
        flaw_severity_scores = {}
        identified_flaws = []
        suggested_improvements = []
        is_approved = True  # Start assuming approval, will be set to False if any round fails
        
        # Process each round of the gauntlet
        for round_rule in gauntlet_def.rounds:
            round_findings = []
            
            # For each judge in the team, run the critique
            for i, model_config in enumerate(team.members[:round_rule.quorum_from_panel_size]):
                finding = self._run_single_judge_critique(
                    solution_attempt, 
                    gauntlet_def, 
                    model_config, 
                    api_key, 
                    model_name
                )
                
                if finding:
                    round_findings.append({
                        'judge_model_id': model_config.model_id,
                        'finding': finding,
                        'score': finding.confidence,
                        'justification': finding.description,
                        'timestamp': time.time()
                    })
                    
                    # Add to identified flaws if present
                    if finding.description:
                        identified_flaws.append({
                            'model_id': model_config.model_id,
                            'title': finding.title,
                            'description': finding.description,
                            'severity': finding.severity.value,
                            'category': finding.category.value,
                            'location': finding.location,
                            'confidence': finding.confidence
                        })
        
            # Calculate round metrics
            round_scores = [f['score'] for f in round_findings]
            if round_scores:
                avg_round_score = sum(round_scores) / len(round_scores)
                
                # Check if this round passes based on quorum requirements
                successful_judges = [f for f in round_findings if f['score'] >= 0.5]  # Threshold for success
                
                # Check quorum requirement
                if len(successful_judges) < round_rule.quorum_required_approvals:
                    is_approved = False
                
                # Check minimum overall confidence
                if avg_round_score < round_rule.min_overall_confidence:
                    is_approved = False
                
                # Check score variance if specified
                if round_rule.max_score_variance is not None and len(round_scores) > 1:
                    variance = statistics.variance(round_scores)
                    if variance > round_rule.max_score_variance:
                        is_approved = False
            
            reports_by_judge.extend(round_findings)
        
        # Calculate overall metrics
        if reports_by_judge:
            all_scores = [r['score'] for r in reports_by_judge]
            overall_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
        else:
            overall_score = 0.0
        
        # Create critique report
        critique_report = CritiqueReport(
            solution_attempt_id=solution_attempt.sub_problem_id,
            gauntlet_name=gauntlet_def.name,
            is_approved=is_approved,
            reports_by_judge=reports_by_judge,
            summary=f"Red Team Gauntlet '{gauntlet_def.name}' completed. Approved: {is_approved}",
            overall_score=overall_score,
            flaw_severity_scores=flaw_severity_scores,
            identified_flaws=identified_flaws,
            suggested_improvements=suggested_improvements
        )
        
        return critique_report
    
    def _run_single_judge_critique(self, solution_attempt: 'SolutionAttempt', 
                                  gauntlet_def: GauntletDefinition,
                                  model_config: 'ModelConfig',
                                  api_key: str, 
                                  model_name: str) -> Optional[IssueFinding]:
        """
        Run a single judge critique using LLM
        """
        try:
            # Construct the critique prompt based on the gauntlet's attack modes
            attack_modes_str = ", ".join(gauntlet_def.attack_modes) if gauntlet_def.attack_modes else "general vulnerabilities"
            
            system_prompt = f"""You are a Red Team AI. Your goal is to find flaws, vulnerabilities, and weaknesses in the provided solution. 
            Focus on the following attack modes: {attack_modes_str}.
            Analyze the solution critically and identify all potential issues."""
            
            user_prompt = f"""Critique the following solution for flaws and vulnerabilities:
            ---
            {solution_attempt.content}
            ---
            
            Return your critique as a JSON object with:
            - 'title': Brief title of the issue found
            - 'description': Detailed description of the flaw
            - 'severity': 'critical', 'high', 'medium', or 'low'
            - 'category': One of the following: 'logical_error', 'security_vulnerability', 'performance_problem', 'compliance_issue', 'structural_flaw', 'clarity_issue', 'maintainability_problem', 'scalability_issue', 'usability_problem', 'technical_debt', 'documentation_gap', 'edge_case'
            - 'confidence': Number between 0.0 and 1.0 indicating your confidence in this finding
            - 'location': Where the issue is located (if applicable)
            - 'suggested_fix': How to address this issue (if applicable)
            
            If no significant issues are found, return appropriate values indicating this."""
            
            # Make LLM call to perform critique
            llm_response_content = _request_openai_compatible_chat(
                api_key=api_key,
                base_url=model_config.api_base,
                model=model_config.model_id or model_name,
                messages=_compose_messages(system_prompt, user_prompt),
                temperature=model_config.temperature,
                max_tokens=model_config.max_tokens,
                timeout=30,
                response_json_format=True  # Request JSON format for structured output
            )
            
            if llm_response_content:
                try:
                    llm_parsed_response = json.loads(llm_response_content)
                    
                    # Map severity to our enum
                    severity_map = {
                        'low': SeverityLevel.LOW,
                        'medium': SeverityLevel.MEDIUM,
                        'high': SeverityLevel.HIGH,
                        'critical': SeverityLevel.CRITICAL
                    }
                    
                    # Map category to our enum
                    category_map = {
                        'logical_error': IssueCategory.LOGICAL_ERROR,
                        'security_vulnerability': IssueCategory.SECURITY_VULNERABILITY,
                        'performance_problem': IssueCategory.PERFORMANCE_PROBLEM,
                        'compliance_issue': IssueCategory.COMPLIANCE_ISSUE,
                        'structural_flaw': IssueCategory.STRUCTURAL_FLAW,
                        'clarity_issue': IssueCategory.CLARITY_ISSUE,
                        'maintainability_problem': IssueCategory.MAINTAINABILITY_PROBLEM,
                        'scalability_issue': IssueCategory.SCALABILITY_ISSUE,
                        'usability_problem': IssueCategory.USABILITY_PROBLEM,
                        'technical_debt': IssueCategory.TECHNICAL_DEBT,
                        'documentation_gap': IssueCategory.DOCUMENTATION_GAP,
                        'edge_case': IssueCategory.EDGE_CASE
                    }
                    
                    severity = severity_map.get(llm_parsed_response.get('severity', 'medium'), SeverityLevel.MEDIUM)
                    category = category_map.get(llm_parsed_response.get('category', 'logical_error'), IssueCategory.LOGICAL_ERROR)
                    
                    return IssueFinding(
                        title=llm_parsed_response.get('title', 'Issue Found'),
                        description=llm_parsed_response.get('description', 'No description provided'),
                        severity=severity,
                        category=category,
                        location=llm_parsed_response.get('location', None),
                        confidence=llm_parsed_response.get('confidence', 0.7),
                        suggested_fix=llm_parsed_response.get('suggested_fix', None),
                        exploit_example=None
                    )
                except json.JSONDecodeError:
                    # If JSON parsing fails, create a basic finding from the text
                    return IssueFinding(
                        title="LLM Response Processing Error",
                        description=f"Could not parse LLM response: {llm_response_content[:200]}...",
                        severity=SeverityLevel.MEDIUM,
                        category=IssueCategory.LOGICAL_ERROR,
                        confidence=0.5
                    )
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in single judge critique: {e}", exc_info=True)
            return None
    
    def generate_critique_report(self, assessment: RedTeamAssessment) -> Dict[str, Any]:
        """Generate a detailed critique report"""
        report = {
            "assessment_summary": assessment.assessment_summary,
            "total_findings": len(assessment.findings),
            "confidence_score": assessment.confidence_score,
            "time_taken_seconds": assessment.time_taken,
            "findings_by_severity": {
                severity.value: count 
                for severity, count in assessment.issues_by_severity.items()
            },
            "findings_by_category": {
                category.value: count 
                for category, count in assessment.issues_by_category.items()
            },
            "detailed_findings": [
                {
                    "title": finding.title,
                    "description": finding.description,
                    "severity": finding.severity.value,
                    "category": finding.category.value,
                    "location": finding.location,
                    "confidence": finding.confidence,
                    "suggested_fix": finding.suggested_fix,
                    "exploit_example": finding.exploit_example
                }
                for finding in assessment.findings
            ],
            "recommendations": [
                finding.suggested_fix 
                for finding in assessment.findings 
                if finding.suggested_fix
            ],
            "assessment_metadata": assessment.assessment_metadata
        }
        
        return report
    
    def integrate_with_orchestration(self, content: str, content_type: str = "general") -> Dict[str, Any]:
        """
        Integrate red team assessment with model orchestration
        """
        if not self.orchestrator or not self.prompt_engineering:
            # Fallback to direct assessment if orchestration not available
            assessment = self.assess_content(content, content_type)
            return self.generate_critique_report(assessment)
        
        # Use orchestration for more sophisticated analysis
        try:
            # First, do our internal assessment
            internal_assessment = self.assess_content(content, content_type)
            
            # Then use orchestrator for additional analysis
            critique_prompt = self.prompt_engineering.prompt_manager.instantiate_prompt(
                'red_team_critique',
                variables={
                    'content': content,
                    'content_type': content_type,
                    'compliance_requirements': ''
                }
            )
            
            orchestration_request = OrchestrationRequest(
                content=content,
                prompt=critique_prompt.rendered_prompt,
                team=ModelTeam.RED
            )
            
            request_id = self.orchestrator.submit_request(orchestration_request)
            
            # Wait for orchestration result (robust polling)
            max_wait = 60  # seconds
            poll_interval = 1 # seconds
            start_time = time.time()
            orchestration_result = None
            
            while time.time() - start_time < max_wait:
                status = self.orchestrator.get_request_status(request_id)
                if status and status['status'] == 'completed':
                    orchestration_result = status.get('response')
                    break
                elif status and status['status'] == 'failed':
                    self.logger.error(f"Orchestration request {request_id} failed: {status.get('error', 'Unknown error')}")
                    break
                time.sleep(poll_interval)
            else:
                self.logger.warning(f"Orchestration request {request_id} timed out after {max_wait} seconds.")
            
            # Combine results
            if orchestration_result and orchestration_result.success:
                # Parse the orchestration result to extract structured findings
                try:
                    # Attempt to parse JSON from orchestration result
                    json_result = json.loads(orchestration_result.response)
                    if 'issues' in json_result:
                        # Convert orchestration issues to our format
                        orchestration_findings = self._convert_orchestration_issues(
                            json_result['issues']
                        )
                        
                        # Combine with internal findings
                        combined_findings = internal_assessment.findings + orchestration_findings
                        
                        # Create new assessment with combined findings
                        combined_assessment = RedTeamAssessment(
                            findings=combined_findings,
                            assessment_summary=f"Combined assessment with orchestration. Original: {internal_assessment.assessment_summary}",
                            confidence_score=max(internal_assessment.confidence_score, 
                                                orchestration_result.response_time),
                            time_taken=internal_assessment.time_taken + orchestration_result.response_time,
                            assessment_metadata={**internal_assessment.assessment_metadata, 
                                              **{'orchestration_used': True}},
                            issues_by_severity={},  # Recalculate
                            issues_by_category={}   # Recalculate
                        )
                        
                        # Recalculate severity and category counts
                        combined_assessment.issues_by_severity = {}
                        combined_assessment.issues_by_category = {}
                        for finding in combined_findings:
                            # Count by severity
                            severity = finding.severity
                            combined_assessment.issues_by_severity[severity] = combined_assessment.issues_by_severity.get(severity, 0) + 1
                            
                            # Count by category
                            category = finding.category
                            combined_assessment.issues_by_category[category] = combined_assessment.issues_by_category.get(category, 0) + 1
                        
                        return self.generate_critique_report(combined_assessment)
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.debug("Failed to parse orchestration assessment; falling back to internal assessment: %s", e)
            
            # Return internal assessment if orchestration fails
            return self.generate_critique_report(internal_assessment)
            
        except Exception as e:
            # Fallback to internal assessment if orchestration integration fails
            self.logger.error(f"Orchestration integration failed: {e}", exc_info=True)
            assessment = self.assess_content(content, content_type)
            return self.generate_critique_report(assessment)
    
    def _convert_orchestration_issues(self, orchestration_issues: List[Dict]) -> List[IssueFinding]:
        """Convert orchestration model issues to our IssueFinding format"""
        findings = []
        
        for issue in orchestration_issues:
            # Map orchestration issue to our format
            title = issue.get('title', 'Issue detected')
            description = issue.get('description', 'No description provided')
            severity_str = issue.get('severity', 'medium').lower()
            
            # Map severity
            severity_map = {
                'low': SeverityLevel.LOW,
                'medium': SeverityLevel.MEDIUM,
                'high': SeverityLevel.HIGH,
                'critical': SeverityLevel.CRITICAL
            }
            severity = severity_map.get(severity_str, SeverityLevel.MEDIUM)
            
            # Map category
            category_str = issue.get('category', 'general').lower()
            category_map = {
                'security': IssueCategory.SECURITY_VULNERABILITY,
                'performance': IssueCategory.PERFORMANCE_PROBLEM,
                'compliance': IssueCategory.COMPLIANCE_ISSUE,
                'logical': IssueCategory.LOGICAL_ERROR,
                'structural': IssueCategory.STRUCTURAL_FLAW,
                'clarity': IssueCategory.CLARITY_ISSUE,
                'maintainability': IssueCategory.MAINTAINABILITY_PROBLEM,
                'scalability': IssueCategory.SCALABILITY_ISSUE,
                'usability': IssueCategory.USABILITY_PROBLEM,
                'technical debt': IssueCategory.TECHNICAL_DEBT,
                'documentation': IssueCategory.DOCUMENTATION_GAP,
                'edge case': IssueCategory.EDGE_CASE,
            }
            category = category_map.get(category_str, IssueCategory.LOGICAL_ERROR)
            
            findings.append(IssueFinding(
                title=title,
                description=description,
                severity=severity,
                category=category,
                confidence=issue.get('confidence', 0.8),
                suggested_fix=issue.get('suggestion', None),
                location=issue.get('location', None)
            ))
        
        return findings
    
    def critique_with_quality_diversity(
        self,
        content: str,
        content_type: str = "general",
        api_key: Optional[str] = None,
        model_name: str = "gpt-4o",
        feature_dimensions: Optional[List[str]] = None,
        max_iterations: int = 10
    ) -> RedTeamAssessment:
        """
        Perform critique using quality diversity (MAP-Elites) to find diverse issues
        
        Args:
            content: Content to critique
            content_type: Type of content
            api_key: API key for OpenEvolve
            model_name: Model to use
            feature_dimensions: Behavior dimensions for diversity
            max_iterations: Number of evolution iterations
            
        Returns:
            RedTeamAssessment with diverse findings
        """
        if not OPENEVOLVE_AVAILABLE or not api_key:
            return self.assess_content(content, content_type)
        
        try:
            from openevolve_client import OpenEvolveClient
            
            client = OpenEvolveClient(api_key=api_key)
            
            # Create behavior dimensions if not provided
            if not feature_dimensions:
                feature_dimensions = self._create_behavior_dimensions(content_type)
            
            # Run quality diversity evolution
            result = client.evolve(
                content=content,
                evolution_mode="quality_diversity",
                max_iterations=max_iterations,
                population_size=20,
                temperature=0.8,
                model_name=model_name,
                content_type=content_type,
                feature_dimensions=feature_dimensions,
                archive_size=100
            )
            
            # Extract diverse findings from archive
            findings = self._extract_diverse_findings(result, content_type)
            
            # Count by severity and category
            issues_by_severity = {}
            issues_by_category = {}
            
            for finding in findings:
                severity = finding.severity
                issues_by_severity[severity] = issues_by_severity.get(severity, 0) + 1
                
                category = finding.category
                issues_by_category[category] = issues_by_category.get(category, 0) + 1
            
            # Calculate confidence
            confidence_score = self._calculate_confidence_score(findings)
            
            # Create assessment
            assessment = RedTeamAssessment(
                findings=findings,
                assessment_summary=f"Quality diversity critique found {len(findings)} diverse issues across {len(feature_dimensions)} behavior dimensions",
                confidence_score=confidence_score,
                time_taken=result.get('metrics', {}).get('total_time', 0.0),
                assessment_metadata={
                    'content_type': content_type,
                    'openevolve_used': True,
                    'evolution_mode': 'quality_diversity',
                    'feature_dimensions': feature_dimensions,
                    'archive_size': result.get('metrics', {}).get('archive_size', 0),
                    'openevolve_metrics': result.get('metrics', {})
                },
                issues_by_severity=issues_by_severity,
                issues_by_category=issues_by_category,
                openevolve_metrics=result.get('metrics', {})
            )
            
            self.assessment_history.append(assessment)
            return assessment
            
        except Exception as e:
            logger.error(f"Error using quality diversity for critique: {e}", exc_info=True)
            return self.assess_content(content, content_type)
    
    def _create_behavior_dimensions(self, content_type: str) -> List[str]:
        """
        Create behavior dimensions for quality diversity
        
        Args:
            content_type: Type of content being assessed
            
        Returns:
            List of behavior dimension names
        """
        if content_type == "code":
            return [
                "security_focus",
                "performance_focus",
                "maintainability_focus",
                "logic_correctness",
                "edge_case_coverage"
            ]
        elif content_type == "document":
            return [
                "clarity_level",
                "completeness_level",
                "technical_depth",
                "structure_quality"
            ]
        elif content_type == "protocol":
            return [
                "safety_level",
                "completeness_level",
                "error_handling_coverage",
                "edge_case_coverage"
            ]
        else:
            return [
                "quality_level",
                "completeness_level",
                "clarity_level"
            ]
    
    def _extract_diverse_findings(
        self,
        evolution_result: Dict[str, Any],
        content_type: str
    ) -> List[IssueFinding]:
        """
        Extract diverse findings from quality diversity archive
        
        Args:
            evolution_result: Result from OpenEvolve quality diversity
            content_type: Type of content
            
        Returns:
            List of diverse issue findings
        """
        findings = []
        
        # Get archive from result
        archive = evolution_result.get('archive', [])
        
        # Extract findings from each archive entry
        for entry in archive:
            critique_text = entry.get('code', '')
            behavior = entry.get('behavior', {})
            fitness = entry.get('fitness', 0.0)
            
            # Parse critique to extract issues
            # This is a simplified extraction - in practice, would use LLM to parse
            if critique_text:
                # Create a finding for this archive entry
                # Determine severity based on fitness
                if fitness > 0.8:
                    severity = SeverityLevel.CRITICAL
                elif fitness > 0.6:
                    severity = SeverityLevel.HIGH
                elif fitness > 0.4:
                    severity = SeverityLevel.MEDIUM
                else:
                    severity = SeverityLevel.LOW
                
                # Determine category based on behavior dimensions
                category = self._infer_category_from_behavior(behavior)
                
                finding = IssueFinding(
                    title=f"Issue in {list(behavior.keys())[0] if behavior else 'general'} dimension",
                    description=critique_text[:200],  # Truncate for summary
                    severity=severity,
                    category=category,
                    confidence=fitness,
                    location=None,
                    suggested_fix=None
                )
                
                findings.append(finding)
        
        return findings
    
    def _infer_category_from_behavior(self, behavior: Dict[str, float]) -> IssueCategory:
        """
        Infer issue category from behavior dimensions
        
        Args:
            behavior: Behavior descriptor from archive entry
            
        Returns:
            Inferred issue category
        """
        if not behavior:
            return IssueCategory.LOGICAL_ERROR
        
        # Get dimension with highest value
        max_dimension = max(behavior.items(), key=lambda x: x[1])[0]
        
        # Map dimension to category
        dimension_map = {
            "security_focus": IssueCategory.SECURITY_VULNERABILITY,
            "performance_focus": IssueCategory.PERFORMANCE_PROBLEM,
            "maintainability_focus": IssueCategory.MAINTAINABILITY_PROBLEM,
            "logic_correctness": IssueCategory.LOGICAL_ERROR,
            "edge_case_coverage": IssueCategory.EDGE_CASE,
            "clarity_level": IssueCategory.CLARITY_ISSUE,
            "completeness_level": IssueCategory.DOCUMENTATION_GAP,
            "technical_depth": IssueCategory.TECHNICAL_DEBT,
            "structure_quality": IssueCategory.STRUCTURAL_FLAW,
            "safety_level": IssueCategory.SECURITY_VULNERABILITY,
            "error_handling_coverage": IssueCategory.EDGE_CASE
        }
        
        return dimension_map.get(max_dimension, IssueCategory.LOGICAL_ERROR)
    
    def extract_targeted_feedback(self, critique_report: CritiqueReport, solution_attempt: SolutionAttempt) -> str:
        """
        Extract targeted feedback from critique report that identifies which specific 
        part of the solution (and therefore which original sub-problem) is the likely 
        cause of the failure, as required by the self-healing loop in Stage 5.
        
        Args:
            critique_report: The critique report to extract feedback from
            solution_attempt: The solution that was critiqued
            
        Returns:
            Targeted feedback string identifying problematic sub-problems
        """
        targeted_feedback_parts = []
        
        # Extract feedback from each judge report
        for report in critique_report.reports_by_judge:
            # Look for sub-problem ID mentions in the critique
            if 'justification' in report:
                justification = report['justification']
                # Look for sub-problem ID patterns like "sub_1.1", "sub_2.3", etc.
                subproblem_matches = re.findall(r'sub_\d+\.\d+', justification, re.IGNORECASE)
                if subproblem_matches:
                    targeted_feedback_parts.extend(subproblem_matches)
            
            # Also check the description field in case it's structured differently
            if 'finding' in report and hasattr(report['finding'], 'description'):
                description = report['finding'].description
                subproblem_matches = re.findall(r'sub_\d+\.\d+', description, re.IGNORECASE)
                if subproblem_matches:
                    targeted_feedback_parts.extend(subproblem_matches)
        
        # Get unique sub-problem IDs
        unique_subproblems = list(set(targeted_feedback_parts))
        
        if unique_subproblems:
            return f"Potential issues identified in sub-problems: {', '.join(unique_subproblems)}. These components require rework."
        else:
            # If no specific sub-problems mentioned, analyze the content for structural issues
            content = solution_attempt.content
            # Look for common structural issues that might map to different sub-problems
            structural_issues = []
            
            # Check for integration issues if it's a final assembly
            if "final_solution" in solution_attempt.sub_problem_id:
                if "missing connection" in content.lower() or "inconsistent interface" in content.lower():
                    structural_issues.append("integration_issues_between_components")
            
            if structural_issues:
                return f"Structural issues identified: {', '.join(structural_issues)}. Consider which sub-problems these issues might relate to."
            else:
                return "General feedback: Solution requires review for potential issues. No specific sub-problem IDs were identified in the critique."
    
    def run_adversarial_dialogue_with_dts(self, content: str, content_type: str = "general",
                                         attacker_persona: str = "security_expert",
                                         defender_persona: str = "system_designer",
                                         rounds: int = 3,
                                         use_multi_judge: bool = True) -> Dict[str, Any]:
        """
        Run adversarial dialogue using Dialogue Tree Search (DTS) for enhanced critique.
        
        This method uses DTS to simulate a multi-turn conversation between an attacker
        (red team) and defender (blue team) to deeply explore vulnerabilities and defenses.
        
        Args:
            content: The content to analyze
            content_type: Type of content (code, document, protocol, etc.)
            attacker_persona: Persona for the attacker (red team)
            defender_persona: Persona for the defender (blue team)
            rounds: Number of dialogue rounds to simulate
            use_multi_judge: Whether to use multi-judge scoring for evaluation
            
        Returns:
            Dictionary with results including:
                - dialogue_history: List of conversation turns
                - vulnerabilities_found: List of identified vulnerabilities
                - defense_strategies: List of proposed defense strategies
                - final_score: Overall security/robustness score
                - dts_available: Whether DTS was actually used
        """
        if not DTS_AVAILABLE:
            logger.warning("DTS not available, falling back to standard adversarial assessment")
            # Fall back to standard adversarial assessment
            assessment = self.assess_content(content, content_type, strategy=RedTeamStrategy.ADVERSARIAL)
            return {
                "dialogue_history": [],
                "vulnerabilities_found": [f.title for f in assessment.findings],
                "defense_strategies": [],
                "final_score": assessment.confidence_score,
                "dts_available": False,
                "fallback_used": True,
                "findings_count": len(assessment.findings)
            }
        
        try:
            # Initialize DTS integration
            dts_config = DTSIntegrationConfig(
                max_rounds=rounds,
                use_multi_judge=use_multi_judge,
                attacker_persona=attacker_persona,
                defender_persona=defender_persona
            )
            dts_integration = DTSIntegration(dts_config)
            
            # Run adversarial dialogue
            result = dts_integration.adversarial_dialogue(
                content=content,
                content_type=content_type,
                attacker_persona=attacker_persona,
                defender_persona=defender_persona,
                rounds=rounds
            )
            
            # Convert DTS result to findings
            vulnerabilities_found = []
            if "vulnerabilities" in result:
                vulnerabilities_found = result["vulnerabilities"]
            elif "issues" in result:
                vulnerabilities_found = result["issues"]
            
            defense_strategies = []
            if "defenses" in result:
                defense_strategies = result["defenses"]
            elif "suggestions" in result:
                defense_strategies = result["suggestions"]
            
            # Calculate a score based on the dialogue outcome
            final_score = result.get("score", 0.5)
            if "judge_scores" in result and result["judge_scores"]:
                final_score = sum(result["judge_scores"]) / len(result["judge_scores"])
            
            return {
                "dialogue_history": result.get("dialogue", []),
                "vulnerabilities_found": vulnerabilities_found,
                "defense_strategies": defense_strategies,
                "final_score": final_score,
                "dts_available": True,
                "fallback_used": False,
                "findings_count": len(vulnerabilities_found),
                "dts_result": result
            }
            
        except Exception as e:
            logger.error(f"Error running DTS adversarial dialogue: {e}", exc_info=True)
            # Fall back to standard assessment
            assessment = self.assess_content(content, content_type, strategy=RedTeamStrategy.ADVERSARIAL)
            return {
                "dialogue_history": [],
                "vulnerabilities_found": [f.title for f in assessment.findings],
                "defense_strategies": [],
                "final_score": assessment.confidence_score,
                "dts_available": True,  # DTS was available but failed
                "fallback_used": True,
                "error": str(e),
                "findings_count": len(assessment.findings)
            }
    
    def run_final_red_team_gauntlet(self, final_solution: SolutionAttempt,
                                   gauntlet_def: GauntletDefinition, 
                                   team: Team, api_key: str, 
                                   model_name: str = "gpt-4o") -> CritiqueReport:
        """
        Run the final Red Team Gauntlet on the assembled solution, as specified in Stage 5.
        This checks for integration errors, inconsistencies, or new vulnerabilities that 
        may have arisen from the assembly process.
        
        Args:
            final_solution: The final assembled solution to critique
            gauntlet_def: The gauntlet definition to use
            team: The Red Team to use for the critique
            api_key: API key for LLM calls
            model_name: Model name for LLM calls
            
        Returns:
            CritiqueReport with results of the final Red Team assessment
        """
        return self.run_red_team_gauntlet(final_solution, gauntlet_def, team, api_key, model_name)

# Example usage and testing
def test_red_team():
    """Test function for the Red Team functionality"""
    
    # Create a red team instance
    red_team = RedTeam()
    
    print("Red Team (Critics) Functionality Test:")
    print(f"Team members: {len(red_team.team_members)}")
    
    # Test with sample code content
    sample_code = """
def authenticate_user(username, password):
    # This is a vulnerable authentication function
    if username == "admin" and password == "password123":
        return True
    return False

def process_data(data):
    # Process data without proper validation
    # SECURITY FIX: Use ast.literal_eval for safe parsing of Python literals
    import ast
    try:
        result = ast.literal_eval(data)
    except (ValueError, SyntaxError):
        raise ValueError("Invalid data format: data must be a valid Python literal")
    return result

def main():
    user_input = input("Enter command: ")
    process_data(user_input)
"""
    
    # Assess the content
    assessment = red_team.assess_content(sample_code, "code")
    
    print(f"Assessment completed in {assessment.time_taken:.2f} seconds")
    print(f"Total findings: {len(assessment.findings)}")
    print(f"Confidence score: {assessment.confidence_score:.2f}")
    
    print("\nTop 5 Findings:")
    for i, finding in enumerate(assessment.findings[:5]):
        print(f"  {i+1}. {finding.severity.value.upper()}: {finding.title}")
        print(f"     Category: {finding.category.value}, Confidence: {finding.confidence:.2f}")
    
    # Generate detailed report
    report = red_team.generate_critique_report(assessment)
    print(f"\nDetailed report has {report['total_findings']} findings")
    print(f"Findings by severity: {report['findings_by_severity']}")
    
    # Test different assessment strategies
    print("\nTesting different strategies:")
    strategies = [RedTeamStrategy.SYSTEMATIC, RedTeamStrategy.ADVERSARIAL, RedTeamStrategy.DEEP_DIVE]
    for strategy in strategies:
        assessment = red_team.assess_content(sample_code, "code", strategy=strategy)
        print(f"  {strategy.value}: {len(assessment.findings)} findings")
    
    return red_team

if __name__ == "__main__":
    test_red_team()
