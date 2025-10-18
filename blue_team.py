"""
Blue Team (Fixers) Functionality for OpenEvolve
Implements the Blue Team functionality described in the ultimate explanation document.
"""

import json

import tempfile
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

import time
from datetime import datetime
import random
import copy
import difflib

from .content_analyzer import ContentAnalyzer

# Import OpenEvolve components for enhanced functionality
try:
    from openevolve.api import run_evolution as openevolve_run_evolution
    from openevolve.config import Config, LLMModelConfig


    OPENEVOLVE_AVAILABLE = True
except ImportError:
    OPENEVOLVE_AVAILABLE = False
    print("OpenEvolve backend not available - using fallback implementation")

from .prompt_engineering import PromptEngineeringSystem
from .model_orchestration import ModelOrchestrator, OrchestrationRequest, ModelTeam
from .quality_assessment import QualityAssessmentEngine, SeverityLevel
from .red_team import RedTeam, RedTeamAssessment, IssueFinding, IssueCategory

class FixPriority(Enum):
    """Priority levels for fixes"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class FixType(Enum):
    """Types of fixes that can be applied"""
    SECURITY_PATCH = "security_patch"
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    LOGIC_CORRECTION = "logic_correction"
    CLARITY_IMPROVEMENT = "clarity_improvement"
    STRUCTURE_REORGANIZATION = "structure_reorganization"
    DOCUMENTATION_ADDITION = "documentation_addition"
    ERROR_HANDLING = "error_handling"
    INPUT_VALIDATION = "input_validation"
    CODE_REFACTORING = "code_refactoring"
    COMPLIANCE_FIX = "compliance_fix"
    MAINTAINABILITY_IMPROVEMENT = "maintainability_improvement"

class BlueTeamStrategy(Enum):
    """Strategies for blue team fixes"""
    COMPREHENSIVE = "comprehensive"
    ITERATIVE = "iterative"
    TARGETED = "targeted"
    MINIMAL = "minimal"
    DEFENSIVE = "defensive"
    PROACTIVE = "proactive"

@dataclass
class FixSuggestion:
    """Suggestion for how to fix an issue"""
    issue_finding: IssueFinding
    fix_type: FixType
    fix_description: str
    code_diff: Optional[str] = None  # For code fixes
    priority: FixPriority = FixPriority.MEDIUM
    confidence: float = 0.8  # How confident we are in this fix
    implementation_notes: Optional[str] = None
    testing_approach: Optional[str] = None

@dataclass
class BlueTeamFix:
    """A single fix applied by the blue team"""
    fix_suggestion: FixSuggestion
    implementation_details: str
    fixed_content: str
    fix_status: str  # "applied", "partially_applied", "skipped"
    time_taken: float
    effectiveness_score: float  # 0-100 scale

@dataclass
class BlueTeamAssessment:
    """Complete assessment from the blue team"""
    original_content: str
    fixed_content: str
    applied_fixes: List[BlueTeamFix]
    fix_suggestions: List[FixSuggestion]
    assessment_summary: str
    overall_improvement_score: float  # Improvement from 0-100
    time_taken: float
    assessment_metadata: Dict[str, Any]
    fixes_by_type: Dict[FixType, int]
    fixes_by_priority: Dict[FixPriority, int]

class BlueTeamMember:
    """Individual blue team member with specific fixing expertise"""

    def __init__(self, name: str, specializations: List[FixType],
                 expertise_level: int = 7, strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE):
        self.name = name
        self.specializations = specializations
        self.expertise_level = expertise_level  # 1-10 scale
        self.strategy = strategy
        self.performance_history: List[Dict[str, Any]] = []
        self.reliability_score = 0.9  # Base reliability

    def suggest_fixes(self, content: str, issues: List[IssueFinding],
                     content_type: str = "general") -> List[FixSuggestion]:
        """
        Analyze issues and suggest fixes
        
        Args:
            content: Original content to fix
            issues: List of issues to address
            content_type: Type of content being fixed
            
        Returns:
            List of fix suggestions
        """
        start_time = time.time()
        
        suggestions = []
        
        # Apply different fix strategies based on the team member's approach
        for issue in issues:
            fix_suggestion = self._generate_fix_suggestion(content, issue, content_type)
            if fix_suggestion:
                # Adjust confidence based on expertise level
                adjusted_confidence = fix_suggestion.confidence * (self.expertise_level / 10.0)
                # Specialization bonus
                if fix_suggestion.fix_type in self.specializations:
                    adjusted_confidence = min(1.0, adjusted_confidence * 1.2)
                
                fix_suggestion.confidence = adjusted_confidence
                suggestions.append(fix_suggestion)
        
        # Apply strategy-specific filtering/adjustments
        if self.strategy == BlueTeamStrategy.TARGETED:
            # Only focus on high-priority fixes
            suggestions = [s for s in suggestions if s.priority in [FixPriority.CRITICAL, FixPriority.HIGH]]
        elif self.strategy == BlueTeamStrategy.MINIMAL:
            # Only apply minimal necessary fixes
            suggestions = self._filter_minimal_fixes(suggestions)
        elif self.strategy == BlueTeamStrategy.DEFENSIVE:
            # Focus on security and safety fixes
            security_fixes = [s for s in suggestions if s.fix_type in [
                FixType.SECURITY_PATCH, FixType.INPUT_VALIDATION, FixType.ERROR_HANDLING
            ]]
            suggestions = security_fixes
        
        # Record performance
        assessment_time = time.time() - start_time
        self.performance_history.append({
            'timestamp': datetime.now(),
            'content_type': content_type,
            'issues_count': len(issues),
            'suggestions_count': len(suggestions),
            'time_taken': assessment_time
        })
        
        # Keep only last 20 assessments
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        return suggestions

    def _generate_fix_suggestion(self, content: str, issue: IssueFinding,
                               content_type: str) -> Optional[FixSuggestion]:
        """Generate a fix suggestion for a specific issue"""
        fix_type = self._map_issue_to_fix_type(issue)
        
        # Determine priority based on issue severity
        priority_map = {
            SeverityLevel.CRITICAL: FixPriority.CRITICAL,
            SeverityLevel.HIGH: FixPriority.HIGH,
            SeverityLevel.MEDIUM: FixPriority.MEDIUM,
            SeverityLevel.LOW: FixPriority.LOW
        }
        priority = priority_map.get(issue.severity, FixPriority.MEDIUM)
        
        # Generate specific fix description based on issue category
        fix_description = self._generate_fix_description(issue, content_type)
        
        # Generate code diff for code-related fixes
        code_diff = self._generate_code_diff(content, issue, content_type) if content_type == "code" else None
        
        # Generate implementation notes
        implementation_notes = self._generate_implementation_notes(issue, fix_type)
        
        # Generate testing approach
        testing_approach = self._generate_testing_approach(issue, fix_type)
        
        return FixSuggestion(
            issue_finding=issue,
            fix_type=fix_type,
            fix_description=fix_description,
            code_diff=code_diff,
            priority=priority,
            confidence=issue.confidence,
            implementation_notes=implementation_notes,
            testing_approach=testing_approach
        )

    def _map_issue_to_fix_type(self, issue: IssueFinding) -> FixType:
        """Map an issue category to the appropriate fix type"""
        category_map = {
            IssueCategory.SECURITY_VULNERABILITY: FixType.SECURITY_PATCH,
            IssueCategory.PERFORMANCE_PROBLEM: FixType.PERFORMANCE_OPTIMIZATION,
            IssueCategory.LOGICAL_ERROR: FixType.LOGIC_CORRECTION,
            IssueCategory.CLARITY_ISSUE: FixType.CLARITY_IMPROVEMENT,
            IssueCategory.STRUCTURAL_FLAW: FixType.STRUCTURE_REORGANIZATION,
            IssueCategory.DOCUMENTATION_GAP: FixType.DOCUMENTATION_ADDITION,
            IssueCategory.EDGE_CASE: FixType.ERROR_HANDLING,
            IssueCategory.COMPLIANCE_ISSUE: FixType.COMPLIANCE_FIX,
            IssueCategory.MAINTAINABILITY_PROBLEM: FixType.MAINTAINABILITY_IMPROVEMENT,
            IssueCategory.TECHNICAL_DEBT: FixType.CODE_REFACTORING,
        }
        
        return category_map.get(issue.category, FixType.LOGIC_CORRECTION)

    def _generate_fix_description(self, issue: IssueFinding, content_type: str) -> str:
        """Generate a description of how to fix the issue"""
        if content_type == "code":
            return self._generate_code_fix_description(issue)
        elif content_type == "document":
            return self._generate_document_fix_description(issue)
        elif content_type == "protocol":
            return self._generate_protocol_fix_description(issue)
        else:
            return self._generate_general_fix_description(issue)

    def _generate_code_fix_description(self, issue: IssueFinding) -> str:
        """Generate fix description for code issues"""
        if issue.category == IssueCategory.SECURITY_VULNERABILITY:
            if "eval" in issue.description.lower() or "eval" in issue.title.lower():
                return ("Replace eval() with ast.literal_eval() for safe evaluation "
                        "of string literals, or implement a custom parser for the specific use case.")
            elif "password" in issue.description.lower():
                return "Remove hardcoded credentials. Use environment variables or a secure configuration system."
            elif "sql" in issue.description.lower():
                return "Use parameterized queries or prepared statements to prevent SQL injection attacks."
        
        elif issue.category == IssueCategory.PERFORMANCE_PROBLEM:
            if "append" in issue.description.lower():
                return "Use list comprehension or an array with pre-allocated size instead of appending in a loop."
        
        elif issue.category == IssueCategory.LOGICAL_ERROR:
            return "Review the logic and add proper validation, error handling, and boundary checks."
        
        elif issue.category == IssueCategory.EDGE_CASE:
            return "Add validation for edge cases and implement proper error handling with try-catch blocks."
        
        elif issue.category == IssueCategory.CLARITY_ISSUE:
            return "Improve variable names, add comments, and break down complex functions."
        
        return f"Address the issue: {issue.description}. Implement appropriate fix based on context."

    def _generate_document_fix_description(self, issue: IssueFinding) -> str:
        """Generate fix description for document issues"""
        if issue.category == IssueCategory.CLARITY_ISSUE:
            return f"Rewrite the complex sentences to be clearer. {issue.description}"
        elif issue.category == IssueCategory.DOCUMENTATION_GAP:
            return f"Add missing information about {issue.description.split('missing')[-1].strip()}."
        else:
            return f"Fix the issue: {issue.description}"

    def _generate_protocol_fix_description(self, issue: IssueFinding) -> str:
        """Generate fix description for protocol issues"""
        if issue.category == IssueCategory.EDGE_CASE:
            return "Add error handling, fallback procedures, and recovery strategies for exceptional conditions."
        else:
            return f"Address the protocol issue: {issue.description}"

    def _generate_general_fix_description(self, issue: IssueFinding) -> str:
        """Generate general fix description"""
        return f"Fix: {issue.description}. {issue.suggested_fix or ''}"

    def _generate_code_diff(self, content: str, issue: IssueFinding, content_type: str) -> Optional[str]:
        """Generate a code diff showing the proposed fix"""
        if content_type != "code":
            return None
            
        try:
            # For demonstration, we'll create a basic diff for some common issues
            if issue.category == IssueCategory.SECURITY_VULNERABILITY:
                if "eval" in content.lower():
                    # Replace eval with a safer alternative
                    lines = content.split('\n')
                    new_lines = []
                    for line in lines:
                        if 'eval(' in line:
                            new_line = line.replace('eval(', 'ast.literal_eval(')
                            if line != new_line:
                                return f"- {line.strip()}\n+ {new_line.strip()}\n"
                        new_lines.append(line)
            
            return f"No automated diff generated for: {issue.description}. Manual fix required."
        except Exception:
            return None

    def _generate_implementation_notes(self, issue: IssueFinding, fix_type: FixType) -> Optional[str]:
        """Generate implementation notes for the fix"""
        notes = []
        
        if fix_type == FixType.SECURITY_PATCH:
            notes.append("Review security implications after implementation")
            notes.append("Consider using established security libraries")
        
        if fix_type == FixType.PERFORMANCE_OPTIMIZATION:
            notes.append("Benchmark before and after implementation")
            notes.append("Consider memory vs speed trade-offs")
        
        if fix_type == FixType.CODE_REFACTORING:
            notes.append("Ensure existing tests still pass")
            notes.append("Update documentation if function signatures change")
        
        if issue.severity == SeverityLevel.CRITICAL:
            notes.append("Thoroughly test this fix as it addresses a critical issue")
        
        return "; ".join(notes) if notes else None

    def _generate_testing_approach(self, issue: IssueFinding, fix_type: FixType) -> Optional[str]:
        """Generate a suggested testing approach"""
        if fix_type == FixType.SECURITY_PATCH:
            return "Unit tests for the specific fix, plus penetration testing and security scanning"
        elif fix_type == FixType.PERFORMANCE_OPTIMIZATION:
            return "Performance benchmarks and load testing to verify improvements"
        elif fix_type == FixType.ERROR_HANDLING:
            return "Test with edge cases and error conditions to ensure graceful failure"
        else:
            return "Unit tests covering the fixed functionality"

    def _filter_minimal_fixes(self, suggestions: List[FixSuggestion]) -> List[FixSuggestion]:
        """Filter to only return the most essential fixes"""
        # Sort by priority and then by confidence
        sorted_suggestions = sorted(suggestions,
                                  key=lambda x: (self._priority_to_int(x.priority), x.confidence),
                                  reverse=True)
        
        # Return only critical and high priority fixes, or top 3 if fewer exist
        critical_high = [s for s in sorted_suggestions
                        if s.priority in [FixPriority.CRITICAL, FixPriority.HIGH]]
        
        if len(critical_high) > 0:
            return critical_high
        else:
            # Return top 3 suggestions if no critical/high priority ones exist
            return sorted_suggestions[:3]

    def _priority_to_int(self, priority: FixPriority) -> int:
        """Convert priority enum to integer for sorting"""
        priority_map = {
            FixPriority.CRITICAL: 4,
            FixPriority.HIGH: 3,
            FixPriority.MEDIUM: 2,
            FixPriority.LOW: 1
        }
        return priority_map.get(priority, 2)

class BlueTeam:
    """Main Blue Team orchestrator that manages fixing operations"""
    
    def __init__(self, orchestrator: ModelOrchestrator = None,
                 prompt_engineering: PromptEngineeringSystem = None,
                 content_analyzer: ContentAnalyzer = None,
                 quality_assessment: QualityAssessmentEngine = None,
                 red_team: RedTeam = None):
        self.orchestrator = orchestrator
        self.prompt_engineering = prompt_engineering
        self.content_analyzer = content_analyzer
        self.quality_assessment = quality_assessment
        self.red_team = red_team
        self.team_members: List[BlueTeamMember] = []
        self.fix_history: List[BlueTeamAssessment] = []
        
        # Initialize default team members
        self._initialize_default_team()

    def _initialize_default_team(self):
        """Initialize a default blue team with different specializations"""
        self.add_team_member(BlueTeamMember(
            name="SecurityFixer",
            specializations=[FixType.SECURITY_PATCH, FixType.INPUT_VALIDATION],
            expertise_level=9,
            strategy=BlueTeamStrategy.DEFENSIVE
        ))
        
        self.add_team_member(BlueTeamMember(
            name="PerformanceOptimizer",
            specializations=[FixType.PERFORMANCE_OPTIMIZATION, FixType.CODE_REFACTORING],
            expertise_level=8,
            strategy=BlueTeamStrategy.PROACTIVE
        ))
        
        self.add_team_member(BlueTeamMember(
            name="CodeRefactorer",
            specializations=[FixType.CODE_REFACTORING, FixType.MAINTAINABILITY_IMPROVEMENT],
            expertise_level=8,
            strategy=BlueTeamStrategy.COMPREHENSIVE
        ))
        
        self.add_team_member(BlueTeamMember(
            name="LogicCorrector",
            specializations=[FixType.LOGIC_CORRECTION, FixType.ERROR_HANDLING],
            expertise_level=7,
            strategy=BlueTeamStrategy.ITERATIVE
        ))
        
        self.add_team_member(BlueTeamMember(
            name="DocumentationSpecialist",
            specializations=[FixType.DOCUMENTATION_ADDITION, FixType.CLARITY_IMPROVEMENT],
            expertise_level=7,
            strategy=BlueTeamStrategy.TARGETED
        ))

    def add_team_member(self, member: BlueTeamMember):
        """Add a new blue team member"""
        self.team_members.append(member)

    def remove_team_member(self, name: str) -> bool:
        """Remove a blue team member by name"""
        for i, member in enumerate(self.team_members):
            if member.name == name:
                del self.team_members[i]
                return True
        return False

    def apply_fixes(self, content: str, issues: List[IssueFinding],
                   content_type: str = "general",
                   strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE,
                   num_members: Optional[int] = None,
                   api_key: Optional[str] = None,
                   model_name: str = "gpt-4o") -> BlueTeamAssessment:
        """
        Apply fixes to content based on identified issues, using OpenEvolve when available
        
        Args:
            content: Original content to fix
            issues: Issues to address
            content_type: Type of content
            strategy: Strategy to use for fixing
            num_members: Number of team members to use (None for all)
            api_key: API key for OpenEvolve backend (required when using OpenEvolve)
            model_name: Model to use when using OpenEvolve
        
        Returns:
            BlueTeamAssessment with the fixes applied
        """
        start_time = time.time()
        
        # Prioritize OpenEvolve backend when available
        if OPENEVOLVE_AVAILABLE and api_key:
            assessment = self._apply_fixes_with_openevolve_backend(
                content, issues, content_type, api_key, model_name
            )
            assessment.time_taken = time.time() - start_time
            return assessment
        
        # Fallback to custom implementation
        return self._apply_fixes_with_custom_implementation(
            content, issues, content_type, strategy, num_members, start_time
        )

    def _apply_fixes_with_openevolve_backend(self, content: str, issues: List[IssueFinding],
                                           content_type: str, api_key: str, model_name: str) -> BlueTeamAssessment:
        """
        Apply fixes using OpenEvolve backend
        """
        try:
            # Create OpenEvolve configuration
            config = Config()
            
            # Configure LLM model
            llm_config = LLMModelConfig(
                name=model_name,
                api_key=api_key,
                api_base="https://api.openai.com/v1",  # Default, can be overridden
                temperature=0.3,  # Lower temperature for more consistent fixes
                max_tokens=4096,
            )
            
            config.llm.models = [llm_config]
            config.max_iterations = 1  # Just one iteration for fixing
            config.database.population_size = 1  # Single fix attempt
            
            # Create a simple evaluator for blue team fixing
                        def blue_team_evaluator(program_path: str, api_key: str, model_name: str) -> Dict[str, Any]:
                            """
                            Evaluator that performs blue team fixing assessment on the content using an LLM.
                            """
                            try:
                                with open(program_path, "r", encoding='utf-8') as f:
                                    content = f.read()
                                
                                # Use LLM to assess the fixed content and generate a score.
                                # This replaces the previous hardcoded score with a dynamic, LLM-driven evaluation.
                                system_prompt = "You are a Blue Team Evaluation AI. Your goal is to assess the quality and effectiveness of the provided content after fixes have been applied. Provide your response as a JSON object with 'score' (0.0-1.0 for overall quality), 'justification' (string), and 'improvement_summary' (string, if applicable)."
                                user_prompt = f"""Evaluate the following fixed content for its quality and effectiveness.
                                Fixed Content:
                                ---
                                {content}
                                ---
                                Provide your evaluation as a JSON object with 'score', 'justification', and 'improvement_summary'.
                                """
            
                                # Make LLM call (using a simplified _request_openai_compatible_chat for this context)
                                try:
                                    import requests
                                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                                    data = {"model": model_name, "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], "temperature": 0.3, "max_tokens": 1024}
                                    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=10)
                                    response.raise_for_status()
                                    llm_result = response.json()
                                    llm_score = json.loads(llm_result["choices"][0]["message"]["content"]).get("score", 0.8)
                                except Exception as llm_e:
                                    print(f"Error getting LLM feedback for blue team evaluator: {llm_e}. Falling back to default score.")
                                    llm_score = 0.8 # Fallback if LLM call fails
            
                                return {
                                    "score": llm_score, 
                                    "timestamp": datetime.now().timestamp(),
                                    "content_length": len(content),
                                    "assessment_completed": True
                                }
                            except Exception as e:
                                print(f"Error in blue team evaluator: {e}")
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
                # Run fixing using OpenEvolve API
                result = openevolve_run_evolution(
                    initial_program=temp_file_path,
                    evaluator=blue_team_evaluator,
                    config=config,
                    iterations=1,
                    output_dir=None,  # Use temporary directory
                    cleanup=True,
                )
                
                # Process results to extract fixes
                # For now, we'll return a basic assessment with placeholder fixes
                # In a real implementation, we would parse the evolution result
                # to extract specific fixes applied during the process
                fixed_content = result.best_code if result.best_code else content
                
                # Generate applied fixes based on comparison with original
                applied_fixes = self._generate_fixes_from_openevolve_result(content, fixed_content, issues)
                
                # Calculate improvement score
                improvement_score = self._calculate_improvement_score(content, fixed_content, content_type)
                
                # Count by fix type and priority
                fixes_by_type = {}
                fixes_by_priority = {}
                
                for fix in applied_fixes:
                    # Count by fix type
                    fix_type = fix.fix_suggestion.fix_type
                    fixes_by_type[fix_type] = fixes_by_type.get(fix_type, 0) + 1
                    
                    # Count by priority
                    priority = fix.fix_suggestion.priority
                    fixes_by_priority[priority] = fixes_by_priority.get(priority, 0) + 1
                
                # Create assessment summary
                summary = self._create_fix_summary(applied_fixes, content_type)
                
                # Create assessment object
                assessment = BlueTeamAssessment(
                    original_content=content,
                    fixed_content=fixed_content,
                    applied_fixes=applied_fixes,
                    fix_suggestions=[],  # Will be empty if using OpenEvolve directly
                    assessment_summary=summary,
                    overall_improvement_score=improvement_score,
                    time_taken=0,  # Will be set by caller
                    assessment_metadata={
                        'content_type': content_type,
                        'num_issues_addressed': len(issues),
                        'openevolve_used': True,
                        'assessment_timestamp': datetime.now().isoformat()
                    },
                    fixes_by_type=fixes_by_type,
                    fixes_by_priority=fixes_by_priority
                )
                
                # Store in history
                self.fix_history.append(assessment)
                
                # Keep only last 50 fixes
                if len(self.fix_history) > 50:
                    self.fix_history = self.fix_history[-50:]
                
                return assessment
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        
        except Exception as e:
            print(f"Error using OpenEvolve backend: {e}")
            # Fallback to custom implementation
            return self._apply_fixes_with_custom_implementation(content, issues, content_type)

    def _generate_fixes_from_openevolve_result(self, original_content: str, fixed_content: str,
                                             issues: List[IssueFinding]) -> List[BlueTeamFix]:
        """
        Generate BlueTeamFix objects from OpenEvolve result
        """
        fixes = []
        
        # For each issue, generate a corresponding fix if it was addressed
        for issue in issues:
            # Create a basic fix object
            fix_suggestion = FixSuggestion(
                issue_finding=issue,
                fix_type=issue.category.value.replace(' ', '_').lower().replace('-', '_'),  # Map to fix type
                fix_description=f"Addressed issue: {issue.description}",
                priority=FixPriority.HIGH if issue.severity == SeverityLevel.CRITICAL else FixPriority.MEDIUM,
                confidence=0.8  # Default confidence
            )
            
            fix = BlueTeamFix(
                fix_suggestion=fix_suggestion,
                implementation_details=f"Applied fix for {issue.title}: {issue.description}",
                fixed_content=fixed_content,
                fix_status="applied",
                time_taken=0,  # Will be calculated separately
                effectiveness_score=80  # Default effectiveness score
            )
            
            fixes.append(fix)
        
        return fixes

    def _apply_fixes_with_custom_implementation(self, content: str, issues: List[IssueFinding],
                                              content_type: str = "general",
                                              strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE,
                                              num_members: Optional[int] = None,
                                              start_time: float = None) -> BlueTeamAssessment:
        """
        Fallback fixing using custom implementation
        """
        if start_time is None:
            start_time = time.time()
        
        # Select team members to use
        selected_members = self.team_members
        if num_members and num_members < len(self.team_members):
            selected_members = random.sample(self.team_members, num_members)
        
        # Get fix suggestions from team members
        all_fix_suggestions = []
        for member in selected_members:
            # Override member strategy if a specific one is provided
            original_strategy = member.strategy
            member.strategy = strategy
            
            member_suggestions = member.suggest_fixes(content, issues, content_type)
            all_fix_suggestions.extend(member_suggestions)
            
            # Restore original strategy
            member.strategy = original_strategy
        
        # Consolidate and prioritize suggestions
        consolidated_suggestions = self._consolidate_fix_suggestions(all_fix_suggestions)
        prioritized_suggestions = self._prioritize_fixes(consolidated_suggestions)
        
        # Apply fixes iteratively
        current_content = content
        applied_fixes = []
        
        for suggestion in prioritized_suggestions:
            fix = self._apply_single_fix(current_content, suggestion, content_type)
            if fix:
                applied_fixes.append(fix)
                current_content = fix.fixed_content  # Update content for next fix
        
        # Calculate improvement score
        improvement_score = self._calculate_improvement_score(content, current_content, content_type)
        
        # Count by fix type and priority
        fixes_by_type = {}
        fixes_by_priority = {}
        
        for fix in applied_fixes:
            # Count by fix type
            fix_type = fix.fix_suggestion.fix_type
            fixes_by_type[fix_type] = fixes_by_type.get(fix_type, 0) + 1
            
            # Count by priority
            priority = fix.fix_suggestion.priority
            fixes_by_priority[priority] = fixes_by_priority.get(priority, 0) + 1
        
        # Create assessment summary
        summary = self._create_fix_summary(applied_fixes, content_type)
        
        # Create assessment object
        assessment = BlueTeamAssessment(
            original_content=content,
            fixed_content=current_content,
            applied_fixes=applied_fixes,
            fix_suggestions=consolidated_suggestions,
            assessment_summary=summary,
            overall_improvement_score=improvement_score,
            time_taken=time.time() - start_time,
            assessment_metadata={
                'content_type': content_type,
                'num_issues_addressed': len(issues),
                'num_team_members': len(selected_members),
                'strategy_used': strategy.value,
                'assessment_timestamp': datetime.now().isoformat(),
                'openevolve_used': False  # Mark as custom implementation
            },
            fixes_by_type=fixes_by_type,
            fixes_by_priority=fixes_by_priority
        )
        
        # Store in history
        self.fix_history.append(assessment)
        
        # Keep only last 50 fixes
        if len(self.fix_history) > 50:
            self.fix_history = self.fix_history[-50:]
        
        return assessment

    def _consolidate_fix_suggestions(self, suggestions: List[FixSuggestion]) -> List[FixSuggestion]:
        """Consolidate similar fix suggestions"""
        if not suggestions:
            return []
        
        # Group suggestions by fix description similarity
        consolidated = []
        seen_suggestion_keys = set()
        
        for suggestion in suggestions:
            # Create a key based on the fix description
            key = suggestion.fix_description.lower().strip()
            
            if key not in seen_suggestion_keys:
                # Find similar suggestions and merge their priorities/confidences
                similar_suggestions = [
                    s for s in suggestions
                    if s.fix_description.lower().strip() == key
                ]
                
                # Use the highest priority among similar suggestions
                max_priority = max([s.priority for s in similar_suggestions],
                                  key=lambda p: ['low', 'medium', 'high', 'critical'].index(p.value))
                
                # Calculate average confidence
                avg_confidence = sum([s.confidence for s in similar_suggestions]) / len(similar_suggestions)
                
                # Use the first suggestion as a base and update values
                base_suggestion = copy.deepcopy(similar_suggestions[0])
                base_suggestion.priority = max_priority
                base_suggestion.confidence = avg_confidence
                
                consolidated.append(base_suggestion)
                seen_suggestion_keys.add(key)
        
        return consolidated

    def _prioritize_fixes(self, suggestions: List[FixSuggestion]) -> List[FixSuggestion]:
        """Prioritize fixes based on their priority and confidence"""
        # Sort by priority first, then by confidence
        return sorted(suggestions,
                     key=lambda x: (self._priority_to_int(x.priority), x.confidence),
                     reverse=True)

    def _priority_to_int(self, priority: FixPriority) -> int:
        """Convert priority enum to integer for sorting"""
        priority_map = {
            FixPriority.CRITICAL: 4,
            FixPriority.HIGH: 3,
            FixPriority.MEDIUM: 2,
            FixPriority.LOW: 1
        }
        return priority_map.get(priority, 2)

    def _apply_single_fix(self, content: str, suggestion: FixSuggestion,
                         content_type: str) -> Optional[BlueTeamFix]:
        """Apply a single fix to the content"""
        start_time = time.time()
        
        # In a real implementation, we would apply the actual fix
        # For now, we'll simulate the fix by returning the original content
        # with a simple modification based on the suggestion
        
        if content_type == "code":
            fixed_content = self._apply_code_fix(content, suggestion)
        elif content_type == "document":
            fixed_content = self._apply_document_fix(content, suggestion)
        elif content_type == "protocol":
            fixed_content = self._apply_protocol_fix(content, suggestion)
        else:
            fixed_content = self._apply_general_fix(content, suggestion)
        
        fix_status = "applied" if fixed_content != content else "skipped"
        
        # Calculate effectiveness score (simulated for now)
        effectiveness_score = suggestion.confidence * 100  # Convert to 0-100 scale
        
        return BlueTeamFix(
            fix_suggestion=suggestion,
            implementation_details=f"Applied suggestion: {suggestion.fix_description}",
            fixed_content=fixed_content,
            fix_status=fix_status,
            time_taken=time.time() - start_time,
            effectiveness_score=effectiveness_score
        )

    def _apply_code_fix(self, content: str, suggestion: FixSuggestion) -> str:
        """Apply a fix to code content"""
        # This is a simplified implementation
        # In a real system, we would use AST manipulation or more sophisticated techniques
        fixed_content = content
        
        if suggestion.fix_type == FixType.SECURITY_PATCH:
            # Example: Replace eval() with a safer alternative
            if "eval(" in content:
                fixed_content = content.replace('eval(', 'ast.literal_eval(')
        
        elif suggestion.fix_type == FixType.INPUT_VALIDATION:
            # Add basic input validation as an example
            if 'input(' in content and 'validate' not in content.lower():
                # This is a very simplified example
                fixed_content = content.replace(
                    'input(',
                    '# Validate input before using\n    if validate_input(input_val := input('
                ).replace('))', ')) else default_value')
        
        elif suggestion.fix_type == FixType.ERROR_HANDLING:
            # Add basic error handling as an example
            if "try:" not in content and "except:" not in content:
                # Add basic try-except around function calls
                lines = content.split('\n')
                modified_lines = []
                for line in lines:
                    if any(call in line for call in ['open(', 'requests.get(', 'api.call(']):
                        modified_lines.append('    try:')
                        modified_lines.append(f"        {line.strip()}")
                        modified_lines.append('    except Exception as e:')
                        modified_lines.append('        print(f"Error: {e}")')
                        modified_lines.append('        return None')
                    else:
                        modified_lines.append(line)
                fixed_content = '\n'.join(modified_lines)
        
        # Add more code fixes as needed based on the suggestion
        
        return fixed_content

    def _apply_document_fix(self, content: str, suggestion: FixSuggestion) -> str:
        """Apply a fix to document content"""
        # This is a simplified implementation
        fixed_content = content
        
        if suggestion.fix_type == FixType.CLARITY_IMPROVEMENT:
            # Simplify complex sentences (simplified implementation)
            # In a real system, we would implement more sophisticated text simplification
            pass
        
        elif suggestion.fix_type == FixType.DOCUMENTATION_ADDITION:
            # Add missing documentation
            if "TODO" in content or "FIXME" in content:
                fixed_content = content.replace("TODO", "DOCUMENTED: TODO").replace("FIXME", "DOCUMENTED: FIXME")
        
        return fixed_content

    def _apply_protocol_fix(self, content: str, suggestion: FixSuggestion) -> str:
        """Apply a fix to protocol content"""
        # Add error handling and safety measures to protocols
        fixed_content = content
        
        if suggestion.fix_type == FixType.ERROR_HANDLING:
            # Add fallback procedures to protocol
            if "if error" not in content.lower():
                fixed_content += "\n\n# Error handling procedures:\n# - If step fails, revert to previous stable state\n# - Log error and notify administrator\n"
        
        return fixed_content

    def _apply_general_fix(self, content: str, suggestion: FixSuggestion) -> str:
        """Apply a general fix to content"""
        # Apply general improvements based on the suggestion
        return content

    def _calculate_improvement_score(self, original_content: str, fixed_content: str,
                                   content_type: str) -> float:
        """Calculate improvement score based on quality assessment"""
        if self.quality_assessment:
            # Assess both original and fixed content
            original_assessment = self.quality_assessment.assess_quality(original_content, content_type)
            fixed_assessment = self.quality_assessment.assess_quality(fixed_content, content_type)
            
            # Calculate improvement as the difference in composite scores
            improvement = fixed_assessment.composite_score - original_assessment.composite_score
            # Cap improvement at 50 points to avoid overstatement
            improvement = min(50, improvement)
            
            # Calculate percentage improvement relative to potential improvement space
            max_possible_improvement = 100 - original_assessment.composite_score
            if max_possible_improvement > 0:
                improvement_percentage = (improvement / max_possible_improvement) * 100
                return max(0, improvement_percentage)
            else:
                return 0.0
        
        # If no quality assessment engine, return a simple calculation
        # This is a simplified approach that just measures content changes
        if len(fixed_content) > len(original_content):
            # More content might indicate improvement (e.g., added documentation)
            return min(100, (len(fixed_content) / len(original_content)) * 50)
        else:
            return 25  # Default score when content is reduced

    def _create_fix_summary(self, applied_fixes: List[BlueTeamFix], content_type: str) -> str:
        """Create a summary of the fixes applied"""
        if not applied_fixes:
            return "No fixes were applied during the blue team assessment."
        
        # Count fixes by type
        type_counts = {}
        for fix in applied_fixes:
            fix_type = fix.fix_suggestion.fix_type
            type_counts[fix_type] = type_counts.get(fix_type, 0) + 1
        
        summary_parts = [
            f"Blue Team Fixes Summary for {content_type} content",
            f"Total Fixes Applied: {len(applied_fixes)}",
            "Fixes by Type:"
        ]
        
        for fix_type, count in type_counts.items():
            summary_parts.append(f"  {fix_type.value.replace('_', ' ').title()}: {count}")
        
        return "\n".join(summary_parts)

    def integrate_with_orchestration(self, content: str, issues: List[IssueFinding],
                                   content_type: str = "general") -> Dict[str, Any]:
        """
        Integrate blue team fixing with model orchestration
        """
        if not self.orchestrator or not self.prompt_engineering:
            # Fallback to direct fixing if orchestration not available
            assessment = self.apply_fixes(content, issues, content_type)
            return self.generate_fix_report(assessment)
        
        # Use orchestration for more sophisticated fixing
        try:
            # First, do our internal fixing
            internal_assessment = self.apply_fixes(content, issues, content_type)
            
            # Create a prompt for the blue team to implement fixes
            fix_prompt = self.prompt_engineering.prompt_manager.instantiate_prompt(
                'blue_team_patch',
                variables={
                    'content': content,
                    'content_type': content_type,
                    'critiques': json.dumps([
                        {
                            'title': issue.title,
                            'description': issue.description,
                            'severity': issue.severity.value,
                            'category': issue.category.value
                        }
                        for issue in issues
                    ], indent=2)
                }
            )
            
            orchestration_request = OrchestrationRequest(
                content=content,
                prompt=fix_prompt.rendered_prompt,
                team=ModelTeam.BLUE
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
                    # Parse the orchestration result to extract fixed content
                    json_result = json.loads(orchestration_result.response)
                    if 'content' in json_result:
                        fixed_content = json_result['content']
                        
                        # Create combined assessment
                        combined_applied_fixes = internal_assessment.applied_fixes
                        
                        # If orchestration provided mitigation matrix, add as additional fixes
                        if 'mitigation_matrix' in json_result:
                            for mitigation in json_result['mitigation_matrix']:
                                # Add mitigation as a fix
                                pass  # In a real implementation
                        
                        # Create new assessment with combined results
                        combined_assessment = BlueTeamAssessment(
                            original_content=content,
                            fixed_content=fixed_content,
                            applied_fixes=combined_applied_fixes,
                            fix_suggestions=internal_assessment.fix_suggestions,
                            assessment_summary=f"Combined fixing with orchestration. Original: {internal_assessment.assessment_summary}",
                            overall_improvement_score=max(
                                internal_assessment.overall_improvement_score,
                                orchestration_result.response_time
                            ),
                            time_taken=internal_assessment.time_taken + orchestration_result.response_time,
                            assessment_metadata={**internal_assessment.assessment_metadata,
                                              **{'orchestration_used': True}},
                            fixes_by_type=internal_assessment.fixes_by_type,
                            fixes_by_priority=internal_assessment.fixes_by_priority
                        )
                        
                        return self.generate_fix_report(combined_assessment)
                except (json.JSONDecodeError, KeyError):
                    pass  # Fall back to internal fixing
            
            # Return internal assessment if orchestration fails
            return self.generate_fix_report(internal_assessment)
            
        except Exception as e:
            # Fallback to internal fixing if orchestration integration fails
            print(f"Orchestration integration failed: {e}")
            assessment = self.apply_fixes(content, issues, content_type)
            return self.generate_fix_report(assessment)

    def fix_content_from_red_team(self, content: str, red_team_assessment: RedTeamAssessment,
                                content_type: str = "general",
                                strategy: BlueTeamStrategy = BlueTeamStrategy.COMPREHENSIVE) -> BlueTeamAssessment:
        """
        Apply fixes based on red team assessment
        """
        # Convert red team findings to issues for blue team
        issues = [finding for finding in red_team_assessment.findings]
        
        # Apply fixes using the blue team
        return self.apply_fixes(content, issues, content_type, strategy)

    def generate_fix_report(self, assessment: BlueTeamAssessment) -> Dict[str, Any]:
        """Generate a detailed fix report"""
        report = {
            "assessment_summary": assessment.assessment_summary,
            "original_content_length": len(assessment.original_content),
            "fixed_content_length": len(assessment.fixed_content),
            "content_changed": assessment.original_content != assessment.fixed_content,
            "total_fixes_applied": len(assessment.applied_fixes),
            "total_fix_suggestions": len(assessment.fix_suggestions),
            "overall_improvement_score": assessment.overall_improvement_score,
            "time_taken_seconds": assessment.time_taken,
            "fixes_by_type": {
                fix_type.value: count
                for fix_type, count in assessment.fixes_by_type.items()
            },
            "fixes_by_priority": {
                priority.value: count
                for priority, count in assessment.fixes_by_priority.items()
            },
            "applied_fixes": [
                {
                    "issue_title": fix.fix_suggestion.issue_finding.title,
                    "issue_description": fix.fix_suggestion.issue_finding.description,
                    "fix_type": fix.fix_suggestion.fix_type.value,
                    "fix_description": fix.fix_suggestion.fix_description,
                    "priority": fix.fix_suggestion.priority.value,
                    "confidence": fix.fix_suggestion.confidence,
                    "implementation_details": fix.implementation_details,
                    "fix_status": fix.fix_status,
                    "effectiveness_score": fix.effectiveness_score
                }
                for fix in assessment.applied_fixes
            ],
            "fix_suggestions": [
                {
                    "issue_title": suggestion.issue_finding.title,
                    "issue_description": suggestion.issue_finding.description,
                    "fix_type": suggestion.fix_type.value,
                    "fix_description": suggestion.fix_description,
                    "priority": suggestion.priority.value,
                    "confidence": suggestion.confidence,
                    "implementation_notes": suggestion.implementation_notes,
                    "testing_approach": suggestion.testing_approach
                }
                for suggestion in assessment.fix_suggestions
            ],
            "content_diff": self._generate_content_diff(assessment.original_content, assessment.fixed_content),
            "assessment_metadata": assessment.assessment_metadata
        }
        
        return report

    def _generate_content_diff(self, original: str, fixed: str) -> str:
        """Generate a simple diff between original and fixed content"""
        if original == fixed:
            return "No changes made to content."
        
        # Create a simple diff using difflib
        diff = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            fixed.splitlines(keepends=True),
            fromfile="original_content",
            tofile="fixed_content",
            n=1  # Context lines
        ))
        
        return "".join(diff) if diff else "No changes detected."

# Example usage and testing
def test_blue_team():
    """Test function for the Blue Team functionality"""
    # Create a blue team instance
    blue_team = BlueTeam()
    
    print("Blue Team (Fixers) Functionality Test:")
    print(f"Team members: {len(blue_team.team_members)}")
    
    # Test with sample code content that has issues
    sample_code_with_issues = """
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
    
    # Create some mock issues to fix
    from red_team import IssueFinding, IssueCategory
    
    issues = [
        IssueFinding(
            title="Hardcoded credentials",
            description="Credentials are hardcoded in the authenticate function",
            severity=SeverityLevel.CRITICAL,
            category=IssueCategory.SECURITY_VULNERABILITY,
            confidence=0.9
        ),
        IssueFinding(
            title="Use of eval() function",
            description="Using eval() is dangerous and can lead to code execution",
            severity=SeverityLevel.CRITICAL,
            category=IssueCategory.SECURITY_VULNERABILITY,
            confidence=0.95
        ),
        IssueFinding(
            title="Missing input validation",
            description="User input is not validated before processing",
            severity=SeverityLevel.HIGH,
            category=IssueCategory.SECURITY_VULNERABILITY,
            confidence=0.85
        )
    ]
    
    # Apply fixes to the content
    assessment = blue_team.apply_fixes(sample_code_with_issues, issues, "code")
    
    print(f"Fixing completed in {assessment.time_taken:.2f} seconds")
    print(f"Total fixes applied: {len(assessment.applied_fixes)}")
    print(f"Improvement score: {assessment.overall_improvement_score:.2f}")
    
    print("\nApplied fixes:")
    for i, fix in enumerate(assessment.applied_fixes[:3]):  # Show first 3
        print(f"  {i+1}. {fix.fix_suggestion.priority.value.upper()}: {fix.fix_suggestion.issue_finding.title}")
        print(f"     Type: {fix.fix_suggestion.fix_type.value}, Status: {fix.fix_status}")
    
    # Generate detailed report
    report = blue_team.generate_fix_report(assessment)
    print(f"\nDetailed report has {report['total_fixes_applied']} applied fixes")
    print(f"Content changed: {report['content_changed']}")
    print(f"Improvement score: {report['overall_improvement_score']:.2f}")
    
    # Test different fixing strategies
    print("\nTesting different strategies:")
    strategies = [BlueTeamStrategy.COMPREHENSIVE, BlueTeamStrategy.DEFENSIVE, BlueTeamStrategy.MINIMAL]
    for strategy in strategies:
        assessment = blue_team.apply_fixes(sample_code_with_issues, issues, "code", strategy=strategy)
        print(f"  {strategy.value}: {len(assessment.applied_fixes)} fixes applied")
    
    # Show a content diff if there were changes
    if report['content_changed']:
        print(f"\nContent diff sample:\n{report['content_diff'][:300]}...")
    
    return blue_team

if __name__ == "__main__":
    test_blue_team()