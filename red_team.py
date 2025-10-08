"""
Red Team (Critics) Functionality for OpenEvolve
Implements the Red Team functionality described in the ultimate explanation document.
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
        
    def assess_content(self, content: str, content_type: str = "general") -> List[IssueFinding]:
        """
        Assess content and return a list of findings
        
        Args:
            content: Content to assess
            content_type: Type of content being assessed
            
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
            'time_taken': assessment_time
        })
        
        # Keep only last 20 assessments
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return adjusted_findings
    
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
        # This is a simplified example
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
                      model_name: str = "gpt-4o") -> RedTeamAssessment:
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
            content, content_type, custom_requirements, strategy, num_members, start_time
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
            def red_team_evaluator(program_path: str) -> Dict[str, Any]:
                """
                Evaluator that performs red team assessment on the content
                """
                try:
                    with open(program_path, "r", encoding='utf-8') as f:
                        content = f.read()
                    
                    # Here we would perform the actual red team analysis
                    # For now, we'll return a basic assessment
                    return {
                        "score": 0.5,  # Placeholder score
                        "timestamp": datetime.now().timestamp(),
                        "content_length": len(content),
                        "assessment_completed": True
                    }
                except Exception as e:
                    print(f"Error in red team evaluator: {e}")
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
                # For now, we'll return a basic assessment with placeholder findings
                # In a real implementation, we would parse the evolution result
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
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._assess_with_custom_implementation(content, content_type, custom_requirements)
    
    def _extract_findings_from_openevolve_result(self, result, content_type: str) -> List[IssueFinding]:
        """
        Extract issue findings from OpenEvolve evolution result
        """
        # This is a placeholder implementation
        # In a real implementation, we would parse the evolution result
        # to extract specific issues found during the evolutionary process
        findings = []
        
        # Example: Create a general finding about content type
        findings.append(IssueFinding(
            title=f"Content Type Analysis for {content_type}",
            description=f"Performed analysis on {content_type} content type",
            severity=SeverityLevel.MEDIUM,
            category=IssueCategory.STRUCTURAL_FLAW,
            confidence=0.8
        ))
        
        return findings

    def _assess_with_custom_implementation(self, content: str, content_type: str = "general", 
                                         custom_requirements: Optional[Dict[str, Any]] = None,
                                         strategy: RedTeamStrategy = RedTeamStrategy.SYSTEMATIC,
                                         num_members: Optional[int] = None,
                                         start_time: float = None) -> RedTeamAssessment:
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
            
            member_findings = member.assess_content(content, content_type)
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
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall back to internal assessment
            
            # Return internal assessment if orchestration fails
            return self.generate_critique_report(internal_assessment)
            
        except Exception as e:
            # Fallback to internal assessment if orchestration integration fails
            print(f"Orchestration integration failed: {e}")
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
    result = eval(data)  # Dangerous!
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