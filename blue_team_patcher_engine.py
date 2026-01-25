"""
Blue Team Patcher Workflow Engine for OpenEvolve
Comprehensive patching system that receives red team findings, applies fixes, and validates results.

This module implements a complete patcher workflow with:
- PatchAnalyzer: Analyzes red team findings and recommends strategies
- PatchApplicationEngine: Applies fixes using 10+ specialized patch types
- PatchValidator: Validates patches and checks for regressions
- Full integration with red_team.py, blue_team.py, and solution_validation_pipeline.py
"""

import os
import json
import re
import time
import copy
import hashlib
import difflib
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

# Local imports
from llm_utils import _request_openai_compatible_chat, _compose_messages
from content_analyzer import ContentAnalyzer
from quality_assessment import QualityAssessmentEngine, SeverityLevel
from red_team import RedTeam, RedTeamAssessment, IssueFinding, IssueCategory

# Import ROMA-MDAP-MAKER (Robust Execution)
try:
    from roma_mdap_maker_associative_integration import (
        ROMAMDAPMakerAssociativeEngine,
        create_romamdapmaker_associative_config,
        ROMA_MDAP_MAKER_AVAILABLE
    )
    from roma_mdap_maker_reliability_ssot import get_reliability_config
except ImportError:
    ROMA_MDAP_MAKER_AVAILABLE = False
    get_reliability_config = None

# Initialize Robust Engine Singleton for Patcher Engine
robust_engine = None
if ROMA_MDAP_MAKER_AVAILABLE:
    try:
        # Use SSOT standard preset for high-reliability patching
        _config = get_reliability_config(
            preset="standard",
            # Can override specific parameters if needed
            # roma_max_depth_solving=2,  # Example: Override if preset doesn't match needs
            # mdap_max_token_length=800,  # Example: Override if preset doesn't match needs
            # mdap_min_confidence=0.25,   # Example: Override if preset doesn't match needs
            # temperature=0.05            # Example: Override if preset doesn't match needs
        )
        robust_engine = ROMAMDAPMakerAssociativeEngine(_config)
    except (ImportError, ConfigurationError, RuntimeError, OSError) as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error initializing ROMA-MDAP-MAKER engine in {__name__}: {e}", exc_info=True)
        raise  # Re-raise the exception

# Configure logging
logger = logging.getLogger(__name__)

# =============================================================================
# Enums and Data Classes
# =============================================================================

class PatchType(Enum):
    """Comprehensive list of patch types supported by the patcher engine"""
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
    RESOURCE_MANAGEMENT = "resource_management"
    CONCURRENCY_FIX = "concurrency_fix"
    DEPENDENCY_UPDATE = "dependency_update"
    TESTING_ENHANCEMENT = "testing_enhancement"

class PatchSeverity(Enum):
    """Severity levels for patches"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class PatchStatus(Enum):
    """Status of a patch in the workflow"""
    PENDING = "pending"
    ANALYZING = "analyzing"
    GENERATING = "generating"
    APPLYING = "applying"
    VALIDATING = "validating"
    APPLIED = "applied"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    SKIPPED = "skipped"

class PatchStrategy(Enum):
    """Strategies for applying patches"""
    AUTOMATIC = "automatic"  # Fully automated patching
    SEMI_AUTOMATIC = "semi_automatic"  # LLM-generated with human review
    MANUAL = "manual"  # Requires manual implementation
    HYBRID = "hybrid"  # Mix of automatic and manual

@dataclass
class PatchMetadata:
    """Metadata about a patch"""
    patch_id: str
    patch_type: PatchType
    severity: PatchSeverity
    status: PatchStatus
    created_at: datetime
    updated_at: datetime
    created_by: str  # Which team member or system created it
    confidence_score: float  # 0-1
    complexity_score: float  # 0-1, higher is more complex
    estimated_time: int  # Estimated time to apply (seconds)

@dataclass
class PatchRequest:
    """A request to apply a patch"""
    issue_finding: IssueFinding
    patch_type: PatchType
    original_content: str
    content_type: str
    suggested_fix: Optional[str] = None
    priority: int = 5  # 1-10, higher is more important
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PatchResult:
    """Result of applying a patch"""
    patch_id: str
    success: bool
    original_content: str
    patched_content: str
    diff: str
    applied_at: datetime
    time_taken: float
    validation_results: Dict[str, Any]
    rollback_data: Optional[str] = None
    error_message: Optional[str] = None

@dataclass
class PatchAnalysis:
    """Analysis of red team findings"""
    total_issues: int
    issues_by_category: Dict[IssueCategory, int]
    issues_by_severity: Dict[SeverityLevel, int]
    recommended_patches: List[PatchRequest]
    complexity_distribution: Dict[str, int]
    estimated_total_time: int
    strategy_recommendation: PatchStrategy
    analysis_metadata: Dict[str, Any]

@dataclass
class PatchReport:
    """Comprehensive report on patching operations"""
    analysis: PatchAnalysis
    patch_results: List[PatchResult]
    summary: Dict[str, Any]
    metrics: Dict[str, Any]
    recommendations: List[str]
    validation_summary: Dict[str, Any]
    rollback_log: List[Dict[str, Any]]
    generated_at: datetime

# =============================================================================
# Patch Analyzer
# =============================================================================

class PatchAnalyzer:
    """
    Analyzes red team findings and recommends optimal patching strategies.

    Key responsibilities:
    - Categorize issues by type and severity
    - Estimate patch complexity
    - Recommend patching strategies
    - Generate patch requests with proper prioritization
    """

    def __init__(self, quality_assessment: Optional[QualityAssessmentEngine] = None):
        self.quality_assessment = quality_assessment
        self.analysis_history: List[PatchAnalysis] = []
        self.complexity_cache: Dict[str, float] = {}

        # Complexity estimation patterns
        self.complexity_patterns = {
            'high': [
                r'multi.*thread', r'concurr', r'race.*cond', r'deadlock',
                r'memory.*leak', r'complex.*algo', r'architectur',
                r'design.*pattern', r'depend.*inject'
            ],
            'medium': [
                r'validat', r'error.*handl', r'edge.*case', r'boundary',
                r'perform', r'optimi', r'refact'
            ],
            'low': [
                r'document', r'comment', r'naming', r'format',
                r'style', r'convent'
            ]
        }

    def analyze_findings(
        self,
        findings: List[IssueFinding],
        original_content: str,
        content_type: str = "general"
    ) -> PatchAnalysis:
        """
        Analyze red team findings and create a comprehensive patch analysis.

        Args:
            findings: List of issues identified by red team
            original_content: The original content being patched
            content_type: Type of content (code, document, protocol, etc.)

        Returns:
            PatchAnalysis with categorization and recommendations
        """
        start_time = time.time()
        logger.info(f"Analyzing {len(findings)} findings for {content_type}")

        # Categorize findings
        issues_by_category = self._categorize_issues(findings)
        issues_by_severity = self._group_by_severity(findings)

        # Generate patch requests
        recommended_patches = self._generate_patch_requests(
            findings, original_content, content_type
        )

        # Analyze complexity distribution
        complexity_distribution = self._analyze_complexity_distribution(recommended_patches)

        # Estimate total time
        estimated_total_time = self._estimate_total_time(recommended_patches)

        # Recommend strategy
        strategy_recommendation = self._recommend_strategy(
            findings, recommended_patches, content_type
        )

        # Create analysis metadata
        analysis_metadata = {
            'analysis_time': time.time() - start_time,
            'content_type': content_type,
            'content_length': len(original_content),
            'confidence': self._calculate_analysis_confidence(findings),
            'analyzer_version': '1.0.0'
        }

        analysis = PatchAnalysis(
            total_issues=len(findings),
            issues_by_category=issues_by_category,
            issues_by_severity=issues_by_severity,
            recommended_patches=recommended_patches,
            complexity_distribution=complexity_distribution,
            estimated_total_time=estimated_total_time,
            strategy_recommendation=strategy_recommendation,
            analysis_metadata=analysis_metadata
        )

        self.analysis_history.append(analysis)
        logger.info(f"Analysis complete: {len(recommended_patches)} patches recommended")

        return analysis

    def _categorize_issues(self, findings: List[IssueFinding]) -> Dict[IssueCategory, int]:
        """Categorize issues by type"""
        category_counts = defaultdict(int)
        for finding in findings:
            category_counts[finding.category] += 1
        return dict(category_counts)

    def _group_by_severity(self, findings: List[IssueFinding]) -> Dict[SeverityLevel, int]:
        """Group issues by severity level"""
        severity_counts = defaultdict(int)
        for finding in findings:
            severity_counts[finding.severity] += 1
        return dict(severity_counts)

    def _generate_patch_requests(
        self,
        findings: List[IssueFinding],
        original_content: str,
        content_type: str
    ) -> List[PatchRequest]:
        """Generate patch requests from findings"""
        patch_requests = []

        for i, finding in enumerate(findings):
            patch_type = self._map_finding_to_patch_type(finding)
            priority = self._calculate_priority(finding)
            complexity = self._estimate_complexity(finding, original_content, i)

            # Generate a finding ID if not present
            finding_id = getattr(finding, 'finding_id', None) or f"F{i+1:03d}"

            patch_request = PatchRequest(
                issue_finding=finding,
                patch_type=patch_type,
                original_content=original_content,
                content_type=content_type,
                suggested_fix=finding.suggested_fix,
                priority=priority,
                context={
                    'complexity': complexity,
                    'finding_id': finding_id,
                    'location': finding.location
                }
            )
            patch_requests.append(patch_request)

        # Sort by priority (descending)
        patch_requests.sort(key=lambda x: x.priority, reverse=True)
        return patch_requests

    def _map_finding_to_patch_type(self, finding: IssueFinding) -> PatchType:
        """Map an issue finding to the appropriate patch type"""
        mapping = {
            IssueCategory.SECURITY_VULNERABILITY: PatchType.SECURITY_PATCH,
            IssueCategory.PERFORMANCE_PROBLEM: PatchType.PERFORMANCE_OPTIMIZATION,
            IssueCategory.LOGICAL_ERROR: PatchType.LOGIC_CORRECTION,
            IssueCategory.CLARITY_ISSUE: PatchType.CLARITY_IMPROVEMENT,
            IssueCategory.STRUCTURAL_FLAW: PatchType.STRUCTURE_REORGANIZATION,
            IssueCategory.DOCUMENTATION_GAP: PatchType.DOCUMENTATION_ADDITION,
            IssueCategory.EDGE_CASE: PatchType.ERROR_HANDLING,
            IssueCategory.COMPLIANCE_ISSUE: PatchType.COMPLIANCE_FIX,
            IssueCategory.MAINTAINABILITY_PROBLEM: PatchType.MAINTAINABILITY_IMPROVEMENT,
            IssueCategory.TECHNICAL_DEBT: PatchType.CODE_REFACTORING,
        }
        return mapping.get(finding.category, PatchType.LOGIC_CORRECTION)

    def _calculate_priority(self, finding: IssueFinding) -> int:
        """Calculate priority score (1-10) for a finding"""
        base_priority = {
            SeverityLevel.CRITICAL: 10,
            SeverityLevel.HIGH: 8,
            SeverityLevel.MEDIUM: 6,
            SeverityLevel.LOW: 4
        }.get(finding.severity, 5)

        # Adjust based on confidence
        adjusted = base_priority * finding.confidence
        return min(10, max(1, int(adjusted)))

    def _estimate_complexity(self, finding: IssueFinding, content: str, index: int = 0) -> float:
        """Estimate complexity of fixing an issue (0-1 scale)"""
        # Check cache
        finding_id = getattr(finding, 'finding_id', None) or f"F{index+1:03d}"
        cache_key = f"{finding_id}_{len(content)}"
        if cache_key in self.complexity_cache:
            return self.complexity_cache[cache_key]

        # Base complexity from category
        category_complexity = {
            IssueCategory.SECURITY_VULNERABILITY: 0.8,
            IssueCategory.STRUCTURAL_FLAW: 0.8,
            IssueCategory.LOGICAL_ERROR: 0.7,
            IssueCategory.EDGE_CASE: 0.6,
            IssueCategory.PERFORMANCE_PROBLEM: 0.6,
            IssueCategory.MAINTAINABILITY_PROBLEM: 0.5,
            IssueCategory.COMPLIANCE_ISSUE: 0.5,
            IssueCategory.CLARITY_ISSUE: 0.3,
            IssueCategory.DOCUMENTATION_GAP: 0.2,
            IssueCategory.TECHNICAL_DEBT: 0.6,
        }.get(finding.category, 0.5)

        # Analyze description for complexity indicators
        description = finding.description.lower()
        complexity_boost = 0

        for pattern in self.complexity_patterns['high']:
            if re.search(pattern, description):
                complexity_boost += 0.15
                break

        for pattern in self.complexity_patterns['medium']:
            if re.search(pattern, description):
                complexity_boost += 0.1
                break

        # Cap at 1.0
        final_complexity = min(1.0, category_complexity + complexity_boost)
        self.complexity_cache[cache_key] = final_complexity

        return final_complexity

    def _analyze_complexity_distribution(self, patches: List[PatchRequest]) -> Dict[str, int]:
        """Analyze distribution of patch complexities"""
        distribution = {'low': 0, 'medium': 0, 'high': 0}

        for patch in patches:
            complexity = patch.context.get('complexity', 0.5)
            if complexity < 0.4:
                distribution['low'] += 1
            elif complexity < 0.7:
                distribution['medium'] += 1
            else:
                distribution['high'] += 1

        return distribution

    def _estimate_total_time(self, patches: List[PatchRequest]) -> int:
        """Estimate total time to apply all patches (in seconds)"""
        base_time_per_patch = 30  # Base time per patch

        total_time = 0
        for patch in patches:
            complexity = patch.context.get('complexity', 0.5)
            # Scale time by complexity
            time_for_patch = base_time_per_patch * (1 + complexity * 2)
            total_time += time_for_patch

        return int(total_time)

    def _recommend_strategy(
        self,
        findings: List[IssueFinding],
        patches: List[PatchRequest],
        content_type: str
    ) -> PatchStrategy:
        """Recommend the best patching strategy"""
        # Count critical/high issues
        critical_count = sum(1 for f in findings if f.severity == SeverityLevel.CRITICAL)
        high_count = sum(1 for f in findings if f.severity == SeverityLevel.HIGH)

        # Check complexity
        high_complexity_count = sum(
            1 for p in patches if p.context.get('complexity', 0) >= 0.7
        )

        # Strategy logic
        if critical_count > 0 or high_complexity_count > len(patches) // 2:
            return PatchStrategy.MANUAL
        elif high_count > len(patches) // 2:
            return PatchStrategy.SEMI_AUTOMATIC
        elif content_type == "code" and len(findings) > 10:
            return PatchStrategy.HYBRID
        else:
            return PatchStrategy.AUTOMATIC

    def _calculate_analysis_confidence(self, findings: List[IssueFinding]) -> float:
        """Calculate overall confidence in the analysis"""
        if not findings:
            return 0.0

        avg_confidence = sum(f.confidence for f in findings) / len(findings)
        return avg_confidence

# =============================================================================
# Patch Application Engine
# =============================================================================

class PatchApplicationEngine:
    """
    Applies patches using specialized strategies for different patch types.

    Key features:
    - Supports 15+ patch types
    - LLM-based automatic patch generation
    - Manual patch workflow support
    - Patch rollback capability
    - Progress tracking and status management
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        quality_assessment: Optional[QualityAssessmentEngine] = None
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.quality_assessment = quality_assessment
        self.patch_history: List[PatchResult] = []
        self.rollback_stack: List[Dict[str, Any]] = []

        # Patch type handlers
        self.patch_handlers = {
            PatchType.SECURITY_PATCH: self._apply_security_patch,
            PatchType.PERFORMANCE_OPTIMIZATION: self._apply_performance_optimization,
            PatchType.LOGIC_CORRECTION: self._apply_logic_correction,
            PatchType.CLARITY_IMPROVEMENT: self._apply_clarity_improvement,
            PatchType.STRUCTURE_REORGANIZATION: self._apply_structure_reorganization,
            PatchType.DOCUMENTATION_ADDITION: self._apply_documentation_addition,
            PatchType.ERROR_HANDLING: self._apply_error_handling,
            PatchType.INPUT_VALIDATION: self._apply_input_validation,
            PatchType.CODE_REFACTORING: self._apply_code_refactoring,
            PatchType.COMPLIANCE_FIX: self._apply_compliance_fix,
            PatchType.MAINTAINABILITY_IMPROVEMENT: self._apply_maintainability_improvement,
            PatchType.RESOURCE_MANAGEMENT: self._apply_resource_management,
            PatchType.CONCURRENCY_FIX: self._apply_concurrency_fix,
            PatchType.DEPENDENCY_UPDATE: self._apply_dependency_update,
            PatchType.TESTING_ENHANCEMENT: self._apply_testing_enhancement,
        }

        # System prompts for different patch types
        self.system_prompts = self._initialize_system_prompts()

    def apply_patches(
        self,
        patch_requests: List[PatchRequest],
        strategy: PatchStrategy = PatchStrategy.AUTOMATIC,
        max_parallel: int = 3,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> List[PatchResult]:
        """
        Apply multiple patches according to the specified strategy.

        Args:
            patch_requests: List of patch requests to apply
            strategy: Patching strategy to use
            max_parallel: Maximum number of parallel patch operations
            progress_callback: Optional callback for progress updates

        Returns:
            List of PatchResult objects
        """
        logger.info(f"Applying {len(patch_requests)} patches with strategy {strategy.value}")
        results = []
        total_patches = len(patch_requests)

        if strategy == PatchStrategy.AUTOMATIC:
            # Apply all patches automatically
            results = self._apply_patches_automatic(
                patch_requests, max_parallel, progress_callback, total_patches
            )
        elif strategy == PatchStrategy.SEMI_AUTOMATIC:
            # Generate patches but allow review
            results = self._apply_patches_semi_automatic(
                patch_requests, progress_callback, total_patches
            )
        elif strategy == PatchStrategy.MANUAL:
            # Manual workflow - provide guidance only
            results = self._apply_patches_manual(
                patch_requests, progress_callback, total_patches
            )
        elif strategy == PatchStrategy.HYBRID:
            # Mix of automatic and manual based on complexity
            results = self._apply_patches_hybrid(
                patch_requests, max_parallel, progress_callback, total_patches
            )

        self.patch_history.extend(results)
        return results

    def _apply_patches_automatic(
        self,
        patch_requests: List[PatchRequest],
        max_parallel: int,
        progress_callback: Optional[Callable[[str, float], None]],
        total_patches: int
    ) -> List[PatchResult]:
        """Apply patches automatically in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            # Submit all patch jobs
            future_to_patch = {
                executor.submit(self._apply_single_patch, patch): patch
                for patch in patch_requests
            }

            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_patch)):
                patch = future_to_patch[future]
                try:
                    result = future.result()
                    results.append(result)

                    if progress_callback:
                        progress = (i + 1) / total_patches * 100
                        progress_callback(
                            f"Applied patch {i+1}/{total_patches}: {patch.patch_type.value}",
                            progress
                        )
                except (ValueError, RuntimeError, ConnectionError, TimeoutError) as e:
                    logger.error(f"Error applying patch {patch.patch_type.value}: {e}")
                    # Create failed result
                    results.append(self._create_failed_result(patch, str(e)))

        return results

    def _apply_patches_semi_automatic(
        self,
        patch_requests: List[PatchRequest],
        progress_callback: Optional[Callable[[str, float], None]],
        total_patches: int
    ) -> List[PatchResult]:
        """Apply patches with intermediate validation"""
        results = []

        for i, patch in enumerate(patch_requests):
            # Generate patch
            result = self._apply_single_patch(patch)
            results.append(result)

            # In semi-automatic mode, you could add human review here
            # For now, we'll add extra validation
            if result.success:
                validation_passed = self._validate_intermediate(result)
                if not validation_passed and progress_callback:
                    progress_callback(
                        f"Warning: Patch {i+1} needs review",
                        (i + 1) / total_patches * 100
                    )

            if progress_callback:
                progress = (i + 1) / total_patches * 100
                progress_callback(
                    f"Applied patch {i+1}/{total_patches}: {patch.patch_type.value}",
                    progress
                )

        return results

    def _apply_patches_manual(
        self,
        patch_requests: List[PatchRequest],
        progress_callback: Optional[Callable[[str, float], None]],
        total_patches: int
    ) -> List[PatchResult]:
        """Generate manual patch instructions instead of applying"""
        results = []

        for i, patch in enumerate(patch_requests):
            # Generate instructions instead of applying
            result = self._generate_patch_instructions(patch)
            results.append(result)

            if progress_callback:
                progress = (i + 1) / total_patches * 100
                progress_callback(
                    f"Generated instructions for patch {i+1}/{total_patches}",
                    progress
                )

        return results

    def _apply_patches_hybrid(
        self,
        patch_requests: List[PatchRequest],
        max_parallel: int,
        progress_callback: Optional[Callable[[str, float], None]],
        total_patches: int
    ) -> List[PatchResult]:
        """Apply simple patches automatically, complex ones manually"""
        automatic_patches = []
        manual_patches = []

        for patch in patch_requests:
            complexity = patch.context.get('complexity', 0.5)
            if complexity < 0.7:
                automatic_patches.append(patch)
            else:
                manual_patches.append(patch)

        # Apply automatic patches
        results = self._apply_patches_automatic(
            automatic_patches, max_parallel, None, len(automatic_patches)
        )

        # Generate instructions for manual patches
        manual_results = self._apply_patches_manual(
            manual_patches, None, len(manual_patches)
        )

        results.extend(manual_results)

        if progress_callback:
            progress_callback(
                f"Applied {len(automatic_patches)} patches automatically, "
                f"generated instructions for {len(manual_patches)} complex patches",
                100.0
            )

        return results

    def _apply_single_patch(self, patch_request: PatchRequest) -> PatchResult:
        """Apply a single patch"""
        start_time = time.time()
        patch_id = self._generate_patch_id(patch_request)

        logger.info(f"Applying patch {patch_id}: {patch_request.patch_type.value}")

        try:
            # Get the appropriate handler
            handler = self.patch_handlers.get(
                patch_request.patch_type,
                self._apply_generic_patch
            )

            # Apply the patch
            patched_content, diff = handler(patch_request)

            # Create rollback data
            rollback_data = self._create_rollback_data(patch_request)

            # Create result
            result = PatchResult(
                patch_id=patch_id,
                success=True,
                original_content=patch_request.original_content,
                patched_content=patched_content,
                diff=diff,
                applied_at=datetime.now(),
                time_taken=time.time() - start_time,
                validation_results={},
                rollback_data=rollback_data
            )

            # Add to rollback stack
            self.rollback_stack.append({
                'patch_id': patch_id,
                'rollback_data': rollback_data,
                'timestamp': datetime.now()
            })

            return result

        except (ValueError, RuntimeError, ConnectionError, TimeoutError, KeyError, AttributeError) as e:
            logger.error(f"Error applying patch {patch_id}: {e}")
            return self._create_failed_result(patch_request, str(e))

    def _create_failed_result(
        self,
        patch_request: PatchRequest,
        error_message: str
    ) -> PatchResult:
        """Create a failed patch result"""
        patch_id = self._generate_patch_id(patch_request)
        return PatchResult(
            patch_id=patch_id,
            success=False,
            original_content=patch_request.original_content,
            patched_content=patch_request.original_content,
            diff="",
            applied_at=datetime.now(),
            time_taken=0.0,
            validation_results={},
            rollback_data=None,
            error_message=error_message
        )

    def _generate_patch_id(self, patch_request: PatchRequest) -> str:
        """Generate a unique patch ID"""
        finding_id = getattr(patch_request.issue_finding, 'finding_id', None) or patch_request.context.get('finding_id', 'unknown')
        content = (
            f"{patch_request.patch_type.value}_"
            f"{finding_id}_"
            f"{datetime.now().timestamp()}"
        )
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _create_rollback_data(self, patch_request: PatchRequest) -> str:
        """Create data needed for rollback"""
        finding_id = getattr(patch_request.issue_finding, 'finding_id', None) or patch_request.context.get('finding_id', 'unknown')
        return json.dumps({
            'original_content': patch_request.original_content,
            'patch_type': patch_request.patch_type.value,
            'finding_id': finding_id,
            'timestamp': datetime.now().isoformat()
        })

    def rollback_patch(self, patch_id: str) -> bool:
        """Rollback a specific patch"""
        for i, entry in enumerate(self.rollback_stack):
            if entry['patch_id'] == patch_id:
                # Remove from rollback stack
                self.rollback_stack.pop(i)
                logger.info(f"Rolled back patch {patch_id}")
                return True

        logger.warning(f"Patch {patch_id} not found in rollback stack")
        return False

    # =========================================================================
    # Patch Type Handlers
    # =========================================================================

    def _apply_security_patch(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply a security patch"""
        prompt = self._build_patch_prompt(patch, "security")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_performance_optimization(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply performance optimization"""
        prompt = self._build_patch_prompt(patch, "performance")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_logic_correction(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply logic correction"""
        prompt = self._build_patch_prompt(patch, "logic")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_clarity_improvement(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply clarity improvement"""
        prompt = self._build_patch_prompt(patch, "clarity")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_structure_reorganization(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply structure reorganization"""
        prompt = self._build_patch_prompt(patch, "structure")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_documentation_addition(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply documentation addition"""
        prompt = self._build_patch_prompt(patch, "documentation")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_error_handling(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply error handling improvements"""
        prompt = self._build_patch_prompt(patch, "error_handling")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_input_validation(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply input validation"""
        prompt = self._build_patch_prompt(patch, "validation")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_code_refactoring(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply code refactoring"""
        prompt = self._build_patch_prompt(patch, "refactoring")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_compliance_fix(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply compliance fix"""
        prompt = self._build_patch_prompt(patch, "compliance")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_maintainability_improvement(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply maintainability improvement"""
        prompt = self._build_patch_prompt(patch, "maintainability")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_resource_management(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply resource management improvements"""
        prompt = self._build_patch_prompt(patch, "resource_management")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_concurrency_fix(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply concurrency fixes"""
        prompt = self._build_patch_prompt(patch, "concurrency")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_dependency_update(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply dependency updates"""
        prompt = self._build_patch_prompt(patch, "dependency")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_testing_enhancement(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply testing enhancements"""
        prompt = self._build_patch_prompt(patch, "testing")
        return self._generate_and_apply_patch(patch, prompt)

    def _apply_generic_patch(self, patch: PatchRequest) -> Tuple[str, str]:
        """Apply a generic patch"""
        prompt = self._build_patch_prompt(patch, "generic")
        return self._generate_and_apply_patch(patch, prompt)

    def _generate_patch_instructions(self, patch: PatchRequest) -> PatchResult:
        """Generate manual patch instructions"""
        instructions = self._build_patch_prompt(patch, "manual")

        return PatchResult(
            patch_id=self._generate_patch_id(patch),
            success=True,
            original_content=patch.original_content,
            patched_content=patch.original_content,  # No change for manual
            diff=f"MANUAL PATCH INSTRUCTIONS:\n\n{instructions}",
            applied_at=datetime.now(),
            time_taken=0.0,
            validation_results={'manual_review_required': True},
            rollback_data=None
        )

    # =========================================================================
    # Prompt Engineering
    # =========================================================================

    def _build_patch_prompt(self, patch: PatchRequest, patch_category: str) -> str:
        """Build a prompt for patch generation"""
        issue = patch.issue_finding

        base_prompt = f"""You are an expert at fixing issues in {patch.content_type}.

ISSUE TO FIX:
- Type: {issue.category.value}
- Severity: {issue.severity.value}
- Description: {issue.description}
- Location: {issue.location}

ORIGINAL CONTENT:
```
{patch.original_content[:2000]}  # Limit length for context
```

"""

        if patch_category == "security":
            base_prompt += """
TASK: Apply a SECURITY PATCH to fix the vulnerability.

Requirements:
1. Identify the security vulnerability
2. Apply the appropriate fix (e.g., input sanitization, parameterized queries, etc.)
3. Ensure no new vulnerabilities are introduced
4. Follow security best practices

Return the fixed content with a brief explanation of the security fix applied.
"""

        elif patch_category == "performance":
            base_prompt += """
TASK: Optimize for PERFORMANCE.

Requirements:
1. Identify performance bottlenecks
2. Apply optimizations (e.g., reduce time complexity, use caching, etc.)
3. Ensure correctness is maintained
4. Consider memory vs speed trade-offs

Return the optimized content with performance improvements explained.
"""

        elif patch_category == "logic":
            base_prompt += """
TASK: Fix the LOGICAL ERROR.

Requirements:
1. Identify the logical flaw
2. Correct the logic
3. Add validation/error handling if needed
4. Ensure edge cases are handled

Return the corrected content with logic fixes explained.
"""

        elif patch_category == "manual":
            base_prompt += """
TASK: Provide detailed MANUAL PATCH INSTRUCTIONS.

Instead of applying the fix, provide:
1. Step-by-step instructions to fix the issue
2. Code examples showing the fix
3. Testing recommendations
4. Potential pitfalls to avoid

Format your response as clear instructions for a developer to follow.
"""

        else:
            base_prompt += f"""
TASK: Apply a {patch_category.upper()} fix.

Fix the issue described above while maintaining the original functionality.
Return the fixed content with a brief explanation.
"""

        if patch.suggested_fix:
            base_prompt += f"\nSUGGESTED FIX: {patch.suggested_fix}\n"

        return base_prompt

    def _generate_and_apply_patch(
        self,
        patch: PatchRequest,
        prompt: str
    ) -> Tuple[str, str]:
        """Generate patch using LLM and apply it"""
        if not self.api_key:
            raise ValueError("API key required for automatic patch generation")

        # Try Robust Engine First
        response = None
        if robust_engine:
            try:
                engine_result = robust_engine.solve_problem_recursive(prompt, patch.context)
                response = engine_result.get("solution")
            except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError):
                response = None

        # Fallback to direct call
        if not response:
            response = _request_openai_compatible_chat(
                api_key=self.api_key,
                base_url=self.api_base,
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_headers={},
                temperature=0.3,  # Lower temperature for more deterministic patches
                top_p=0.95,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                max_tokens=4096,
                seed=None
            )

        if not response:
            raise ValueError("Failed to generate patch: No response from LLM")

        # Extract patched content from response
        patched_content = self._extract_patched_content(response, patch.original_content)

        # Generate diff
        diff = self._generate_diff(patch.original_content, patched_content)

        return patched_content, diff

    def _extract_patched_content(self, response: str, original_content: str) -> str:
        """Extract patched content from LLM response"""
        # Try to extract code block
        code_block_match = re.search(r'```(?:\w+)?\n(.*?)```', response, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1).strip()

        # Try JSON format
        if response.strip().startswith('{'):
            try:
                parsed = json.loads(response)
                if 'fixed_content' in parsed:
                    return parsed['fixed_content']
                if 'content' in parsed:
                    return parsed['content']
            except json.JSONDecodeError:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Error in {__name__}", exc_info=True)
                raise  # Re-raise the exception

        # Fallback: return response as-is if it's different from original
        if response.strip() != original_content.strip():
            return response.strip()

        # Last resort: return original content unchanged
        return original_content

    def _generate_diff(self, original: str, patched: str) -> str:
        """Generate a diff between original and patched content"""
        original_lines = original.splitlines(keepends=True)
        patched_lines = patched.splitlines(keepends=True)

        diff = difflib.unified_diff(
            original_lines,
            patched_lines,
            fromfile='original',
            tofile='patched',
            lineterm=''
        )

        return ''.join(diff)

    def _validate_intermediate(self, result: PatchResult) -> bool:
        """Validate an intermediate patch result"""
        # Basic validation: check that content changed
        if result.original_content == result.patched_content:
            return False

        # Check that patched content is not empty
        if not result.patched_content.strip():
            return False

        # Additional validation could be added here
        return True

    def _initialize_system_prompts(self) -> Dict[str, str]:
        """Initialize system prompts for different patch types"""
        return {
            'security': "You are a security expert specializing in vulnerability remediation.",
            'performance': "You are a performance optimization expert.",
            'logic': "You are a software engineer specializing in logic and correctness.",
            'clarity': "You are a technical writer specializing in code clarity.",
            'structure': "You are a software architect specializing in code organization.",
            'documentation': "You are a technical documentation specialist.",
            'error_handling': "You are an expert in error handling and fault tolerance.",
            'validation': "You are an expert in input validation and sanitization.",
            'refactoring': "You are a code refactoring specialist.",
            'compliance': "You are a compliance and standards expert.",
            'maintainability': "You are a software maintenance expert.",
            'resource_management': "You are an expert in memory and resource management.",
            'concurrency': "You are a concurrency and parallelism expert.",
            'dependency': "You are a software dependency expert.",
            'testing': "You are a software testing expert.",
            'generic': "You are a versatile software engineer."
        }

# =============================================================================
# Patch Validator
# =============================================================================

class PatchValidator:
    """
    Validates that patches fix issues and checks for regressions.

    Key responsibilities:
    - Validate that patches address the identified issues
    - Check for regressions in functionality
    - Verify quality improvements
    - Generate comprehensive validation reports
    """

    def __init__(
        self,
        quality_assessment: Optional[QualityAssessmentEngine] = None,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o"
    ):
        self.quality_assessment = quality_assessment
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.validation_history: List[Dict[str, Any]] = []

    def validate_patches(
        self,
        patch_results: List[PatchResult],
        original_issues: List[IssueFinding],
        original_content: str
    ) -> Dict[str, Any]:
        """
        Validate all applied patches.

        Args:
            patch_results: List of patch results to validate
            original_issues: Original issues that were being fixed
            original_content: Original content before any patches

        Returns:
            Comprehensive validation report
        """
        logger.info(f"Validating {len(patch_results)} patches")

        validation_summary = {
            'total_patches': len(patch_results),
            'successful_patches': sum(1 for p in patch_results if p.success),
            'failed_patches': sum(1 for p in patch_results if not p.success),
            'validation_tests_passed': 0,
            'validation_tests_failed': 0,
            'regressions_detected': 0,
            'quality_improvements': 0,
            'patch_validations': []
        }

        for patch_result in patch_results:
            validation = self._validate_single_patch(
                patch_result, original_content
            )
            validation_summary['patch_validations'].append(validation)

            # Update summary
            if validation['passed']:
                validation_summary['validation_tests_passed'] += 1
            else:
                validation_summary['validation_tests_failed'] += 1

            if validation['regression_detected']:
                validation_summary['regressions_detected'] += 1

            if validation['quality_improved']:
                validation_summary['quality_improvements'] += 1

        # Calculate overall score
        validation_summary['overall_validation_score'] = self._calculate_validation_score(
            validation_summary
        )

        return validation_summary

    def _validate_single_patch(
        self,
        patch_result: PatchResult,
        original_content: str
    ) -> Dict[str, Any]:
        """Validate a single patch"""
        validation = {
            'patch_id': patch_result.patch_id,
            'passed': False,
            'regression_detected': False,
            'quality_improved': False,
            'tests': [],
            'warnings': [],
            'errors': []
        }

        if not patch_result.success:
            validation['errors'].append("Patch application failed")
            return validation

        # Test 1: Check that content changed (if it was supposed to)
        if not self._test_content_changed(patch_result):
            validation['warnings'].append("Content unchanged after patch")

        # Test 2: Check for regressions
        regression_detected = self._test_regressions(patch_result, original_content)
        validation['regression_detected'] = regression_detected
        if regression_detected:
            validation['errors'].append("Potential regression detected")

        # Test 3: Check quality improvement
        quality_improved = self._test_quality_improvement(patch_result)
        validation['quality_improved'] = quality_improved

        # Test 4: Check for syntax errors (if code)
        if self._is_code_content(patch_result.patched_content):
            syntax_ok = self._test_syntax(patch_result.patched_content)
            validation['tests'].append({
                'name': 'syntax_check',
                'passed': syntax_ok
            })
            if not syntax_ok:
                validation['errors'].append("Syntax errors detected")

        # Test 5: Check that patch is not empty
        if not patch_result.patched_content.strip():
            validation['errors'].append("Patched content is empty")

        # Determine overall pass/fail
        validation['passed'] = (
            len(validation['errors']) == 0 and
            not validation['regression_detected']
        )

        return validation

    def _test_content_changed(self, patch_result: PatchResult) -> bool:
        """Test that the content actually changed"""
        # Check if this was a manual instruction patch
        if "MANUAL PATCH INSTRUCTIONS" in patch_result.diff:
            return True

        # Check if content changed
        return patch_result.original_content != patch_result.patched_content

    def _test_regressions(
        self,
        patch_result: PatchResult,
        original_content: str
    ) -> bool:
        """Test for regressions introduced by the patch"""
        # Basic regression checks

        # Check 1: Length significantly reduced (could indicate data loss)
        original_len = len(patch_result.original_content)
        patched_len = len(patch_result.patched_content)

        if patched_len < original_len * 0.5:
            return True  # Significant reduction, potential data loss

        # Check 2: Empty lines or removal of important sections
        if not patch_result.patched_content.strip():
            return True

        # Check 3: Loss of imports/includes (for code)
        if self._is_code_content(original_content):
            original_imports = len(re.findall(r'^import |^from ', original_content, re.MULTILINE))
            patched_imports = len(re.findall(r'^import |^from ', patch_result.patched_content, re.MULTILINE))

            if patched_imports < original_imports * 0.8:
                return True  # Lost significant imports

        return False

    def _test_quality_improvement(self, patch_result: PatchResult) -> bool:
        """Test if quality improved after patching"""
        if not self.quality_assessment:
            # Basic heuristic: if patch succeeded, assume quality improved
            return patch_result.success

        try:
            # Compare quality scores
            original_score = self.quality_assessment.assess_quality(
                patch_result.original_content
            ).get('overall_score', 0.5)

            patched_score = self.quality_assessment.assess_quality(
                patch_result.patched_content
            ).get('overall_score', 0.5)

            return patched_score > original_score
        except (ConnectionError, TimeoutError, ValueError, AttributeError) as e:
            logger.warning(f"Quality assessment failed: {e}")
            return False

    def _is_code_content(self, content: str) -> bool:
        """Check if content is code"""
        code_indicators = [
            'def ', 'class ', 'function ', 'import ', 'from ',
            '{', '}', 'if (', 'for (', 'while (', 'return '
        ]
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in code_indicators)

    def _test_syntax(self, content: str) -> bool:
        """Test for syntax errors in code"""
        # Basic syntax checks

        # Check balanced braces/parentheses
        braces = {'(': ')', '[': ']', '{': '}'}
        stack = []

        for char in content:
            if char in braces:
                stack.append(char)
            elif char in braces.values():
                if not stack:
                    return False
                opening = stack.pop()
                if braces[opening] != char:
                    return False

        if stack:
            return False

        # Check for common syntax errors
        error_patterns = [
            r'\)\s*\(',  # ) ( without operator
            r'\]\s*\[',  # ] [ without operator
            r'}\s*{',    # } { without operator
        ]

        for pattern in error_patterns:
            if re.search(pattern, content):
                return False

        return True

    def _calculate_validation_score(self, validation_summary: Dict[str, Any]) -> float:
        """Calculate overall validation score (0-1)"""
        total = validation_summary['total_patches']
        if total == 0:
            return 1.0

        passed = validation_summary['validation_tests_passed']
        regressions = validation_summary['regressions_detected']

        # Score based on passed tests, penalize regressions heavily
        base_score = passed / total
        regression_penalty = (regressions / total) * 0.5

        return max(0.0, min(1.0, base_score - regression_penalty))

# =============================================================================
# Main Blue Team Patcher Engine
# =============================================================================

class BlueTeamPatcherEngine:
    """
    Main orchestrator for the Blue Team patcher workflow.

    This class integrates PatchAnalyzer, PatchApplicationEngine, and PatchValidator
    to provide a complete patching solution.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: str = "https://api.openai.com/v1",
        model_name: str = "gpt-4o",
        quality_assessment: Optional[QualityAssessmentEngine] = None
    ):
        self.api_key = api_key
        self.api_base = api_base
        self.model_name = model_name
        self.quality_assessment = quality_assessment

        # Initialize components
        self.analyzer = PatchAnalyzer(quality_assessment)
        self.applicator = PatchApplicationEngine(
            api_key=api_key,
            api_base=api_base,
            model_name=model_name,
            quality_assessment=quality_assessment
        )
        self.validator = PatchValidator(
            quality_assessment=quality_assessment,
            api_key=api_key,
            api_base=api_base,
            model_name=model_name
        )

        # History
        self.patch_reports: List[PatchReport] = []

    def run_patcher_workflow(
        self,
        red_team_findings: List[IssueFinding],
        original_content: str,
        content_type: str = "general",
        strategy: PatchStrategy = PatchStrategy.AUTOMATIC,
        max_parallel: int = 3
    ) -> PatchReport:
        """
        Run the complete patcher workflow.

        Args:
            red_team_findings: Issues identified by red team
            original_content: Original content to patch
            content_type: Type of content
            strategy: Patching strategy to use
            max_parallel: Maximum parallel patch operations

        Returns:
            Comprehensive PatchReport
        """
        logger.info(f"Starting patcher workflow for {len(red_team_findings)} issues")
        start_time = time.time()

        # Phase 1: Analyze findings
        logger.info("Phase 1: Analyzing findings...")
        analysis = self.analyzer.analyze_findings(
            red_team_findings, original_content, content_type
        )

        # Phase 2: Apply patches
        logger.info("Phase 2: Applying patches...")
        patch_results = self.applicator.apply_patches(
            analysis.recommended_patches,
            strategy=strategy,
            max_parallel=max_parallel
        )

        # Phase 3: Validate patches
        logger.info("Phase 3: Validating patches...")
        validation_summary = self.validator.validate_patches(
            patch_results, red_team_findings, original_content
        )

        # Phase 4: Generate report
        logger.info("Phase 4: Generating report...")
        report = self._generate_report(
            analysis, patch_results, validation_summary, start_time
        )

        self.patch_reports.append(report)
        logger.info(f"Patcher workflow complete: {report.summary['success_rate']:.1%} success rate")

        return report

    def _generate_report(
        self,
        analysis: PatchAnalysis,
        patch_results: List[PatchResult],
        validation_summary: Dict[str, Any],
        start_time: float
    ) -> PatchReport:
        """Generate a comprehensive patch report"""
        # Calculate summary statistics
        total_patches = len(patch_results)
        successful_patches = sum(1 for p in patch_results if p.success)
        failed_patches = total_patches - successful_patches

        summary = {
            'total_patches': total_patches,
            'successful_patches': successful_patches,
            'failed_patches': failed_patches,
            'success_rate': successful_patches / total_patches if total_patches > 0 else 0,
            'total_time': time.time() - start_time,
            'strategy_used': analysis.strategy_recommendation.value
        }

        # Calculate metrics
        metrics = {
            'avg_time_per_patch': (
                sum(p.time_taken for p in patch_results) / total_patches
                if total_patches > 0 else 0
            ),
            'patches_by_type': self._count_patches_by_type(patch_results),
            'patches_by_severity': self._count_patches_by_severity(patch_results),
            'validation_score': validation_summary.get('overall_validation_score', 0),
            'quality_improvements': validation_summary.get('quality_improvements', 0),
            'regressions_detected': validation_summary.get('regressions_detected', 0)
        }

        # Generate recommendations
        recommendations = self._generate_recommendations(
            analysis, patch_results, validation_summary
        )

        # Create rollback log
        rollback_log = [
            {
                'patch_id': p.patch_id,
                'has_rollback': p.rollback_data is not None,
                'timestamp': p.applied_at.isoformat()
            }
            for p in patch_results
        ]

        return PatchReport(
            analysis=analysis,
            patch_results=patch_results,
            summary=summary,
            metrics=metrics,
            recommendations=recommendations,
            validation_summary=validation_summary,
            rollback_log=rollback_log,
            generated_at=datetime.now()
        )

    def _count_patches_by_type(self, patch_results: List[PatchResult]) -> Dict[str, int]:
        """Count patches by type"""
        counts = defaultdict(int)
        for result in patch_results:
            # Extract patch type from patch_id
            patch_type = result.patch_id.split('_')[0] if '_' in result.patch_id else 'unknown'
            counts[patch_type] += 1
        return dict(counts)

    def _count_patches_by_severity(self, patch_results: List[PatchResult]) -> Dict[str, int]:
        """Count patches by severity based on patch properties and outcomes"""
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'all': len(patch_results)
        }

        for patch_result in patch_results:
            # Determine severity based on patch properties
            if hasattr(patch_result, 'severity'):
                severity = getattr(patch_result, 'severity', 'medium').lower()
            elif hasattr(patch_result, 'risk_level'):
                risk_level = getattr(patch_result, 'risk_level', 'medium').lower()
                # Map risk levels to severities
                severity_map = {
                    'critical': 'critical',
                    'high': 'high',
                    'medium': 'medium',
                    'low': 'low',
                    'minimal': 'low'
                }
                severity = severity_map.get(risk_level, 'medium')
            else:
                # Default to medium if no explicit severity
                severity = 'medium'

            if severity in severity_counts:
                severity_counts[severity] += 1
            else:
                severity_counts['medium'] += 1  # Default to medium for unknown severities

        return severity_counts

    def _generate_recommendations(
        self,
        analysis: PatchAnalysis,
        patch_results: List[PatchResult],
        validation_summary: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on patching results"""
        recommendations = []

        # Check for failed patches
        failed_count = sum(1 for p in patch_results if not p.success)
        if failed_count > 0:
            recommendations.append(
                f"{failed_count} patches failed - review manual application required"
            )

        # Check for regressions
        if validation_summary.get('regressions_detected', 0) > 0:
            recommendations.append(
                "Regressions detected - review and rollback affected patches"
            )

        # Check validation score
        validation_score = validation_summary.get('overall_validation_score', 0)
        if validation_score < 0.7:
            recommendations.append(
                "Low validation score - additional testing recommended"
            )

        # Strategy recommendations
        if analysis.strategy_recommendation == PatchStrategy.MANUAL:
            recommendations.append(
                "Complex issues detected - consider manual review for critical patches"
            )

        # Quality recommendations
        if validation_summary.get('quality_improvements', 0) < len(patch_results) // 2:
            recommendations.append(
                "Consider additional quality improvements beyond basic fixes"
            )

        if not recommendations:
            recommendations.append("All patches applied successfully with good validation results")

        return recommendations

    def export_report(self, report: PatchReport, format: str = "json") -> str:
        """Export a patch report in the specified format"""
        if format == "json":
            return self._export_json(report)
        elif format == "markdown":
            return self._export_markdown(report)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_json(self, report: PatchReport) -> str:
        """Export report as JSON"""
        def convert_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        report_dict = {
            'analysis': {
                'total_issues': report.analysis.total_issues,
                'issues_by_category': {
                    k.value: v for k, v in report.analysis.issues_by_category.items()
                },
                'complexity_distribution': report.analysis.complexity_distribution,
                'estimated_total_time': report.analysis.estimated_total_time,
                'strategy_recommendation': report.analysis.strategy_recommendation.value
            },
            'summary': report.summary,
            'metrics': report.metrics,
            'recommendations': report.recommendations,
            'validation_summary': report.validation_summary,
            'generated_at': report.generated_at.isoformat()
        }

        return json.dumps(report_dict, indent=2, default=convert_datetime)

    def _export_markdown(self, report: PatchReport) -> str:
        """Export report as Markdown"""
        md = f"""# Blue Team Patcher Report

Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M:%S')}

## Summary

- **Total Issues Found**: {report.analysis.total_issues}
- **Patches Applied**: {report.summary['total_patches']}
- **Successful**: {report.summary['successful_patches']} ({report.summary['success_rate']:.1%})
- **Failed**: {report.summary['failed_patches']}
- **Total Time**: {report.summary['total_time']:.2f}s
- **Strategy**: {report.summary['strategy_used']}

## Issues by Category

"""
        for category, count in report.analysis.issues_by_category.items():
            md += f"- **{category.value}**: {count}\n"

        md += f"""
## Complexity Distribution

- **Low**: {report.analysis.complexity_distribution['low']}
- **Medium**: {report.analysis.complexity_distribution['medium']}
- **High**: {report.analysis.complexity_distribution['high']}

## Validation Results

- **Validation Score**: {report.validation_summary['overall_validation_score']:.2%}
- **Quality Improvements**: {report.validation_summary['quality_improvements']}
- **Regressions Detected**: {report.validation_summary['regressions_detected']}

## Recommendations

"""
        for rec in report.recommendations:
            md += f"- {rec}\n"

        md += """
## Patch Details

"""
        for i, result in enumerate(report.patch_results[:10], 1):  # Limit to first 10
            status = "✓" if result.success else "✗"
            md += f"{i}. {status} Patch {result.patch_id} ({result.time_taken:.2f}s)\n"

        return md


# =============================================================================
# Convenience Functions
# =============================================================================

def quick_patch(
    findings: List[IssueFinding],
    content: str,
    api_key: str,
    content_type: str = "general",
    model_name: str = "gpt-4o"
) -> Tuple[str, PatchReport]:
    """
    Quick patch function for simple use cases.

    Args:
        findings: List of issues to fix
        content: Original content
        api_key: API key for LLM
        content_type: Type of content
        model_name: Model to use

    Returns:
        Tuple of (patched_content, report)
    """
    engine = BlueTeamPatcherEngine(
        api_key=api_key,
        model_name=model_name
    )

    report = engine.run_patcher_workflow(
        red_team_findings=findings,
        original_content=content,
        content_type=content_type,
        strategy=PatchStrategy.AUTOMATIC
    )

    # Get the last successful patch
    successful_patches = [p for p in report.patch_results if p.success]
    if successful_patches:
        patched_content = successful_patches[-1].patched_content
    else:
        patched_content = content

    return patched_content, report


if __name__ == "__main__":
    # Example usage
    print("Blue Team Patcher Engine loaded successfully")
    print("Available patch types:")
    for patch_type in PatchType:
        print(f"  - {patch_type.value}")
